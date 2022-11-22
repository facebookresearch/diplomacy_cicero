#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import atexit
import logging
import os
import subprocess
import random
from collections import Counter, defaultdict
from functools import reduce
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.optim
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import wandb
from fairdiplomacy.models.base_strategy_model.util import (
    explain_base_strategy_model_decoder_inputs,
)

import heyhi
import conf.conf_cfgs
from fairdiplomacy.models.consts import POWERS, SEASONS, LOCS
from fairdiplomacy.models.base_strategy_model.base_strategy_model import NO_ORDER_ID, Scoring
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import Dataset, shuffle_locations, maybe_augment_targets_inplace
from fairdiplomacy.models.base_strategy_model.load_model import new_model
from fairdiplomacy.models.state_space import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_by_unit,
    EOS_IDX,
)
from fairdiplomacy.selfplay.metrics import Logger
from fairdiplomacy.selfplay.wandb import initialize_wandb_if_enabled
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global

MAIN_VALIDATION_SET_SUFFIX = ""
ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_IDXS_BY_UNIT = get_order_vocabulary_idxs_by_unit()
LOC_TO_IDX = {loc: idx for idx, loc in enumerate(LOCS)}
ORDER_TO_LOC_IDX = torch.LongTensor([LOC_TO_IDX[order.split()[1]] for order in ORDER_VOCABULARY])
ORDER_TYPES = ["H", "-", "S", "C", "R", "B", "D"]
ORDER_TO_TYPE_IDX = torch.LongTensor(
    [
        ORDER_TYPES.index(order.split()[2] if ";" not in order else "B")
        for order in ORDER_VOCABULARY
    ]
)


def process_batch(
    net,
    batch,
    policy_loss_fn,
    value_loss_use_cross_entropy: bool,
    num_scoring_systems: int,
    temperature=1.0,
    p_teacher_force=1.0,
    shuffle_locs=False,
):
    """Calculate a forward pass on a batch

    Returns:
    - policy_losses: [B, S] FloatTensor
    - policy_loss_weights: [B, S] FloatTensor
    - value_losses: [B] FloatTensor
    - value_loss_weights: [B] FloatTensor
    - local_order_idxs: [B, S] LongTensor of sampled order idxs (< 469)
    - final_sos: [B, 7] estimated final sum-of-squares share of each power
    """
    assert p_teacher_force == 1
    device = next(net.parameters()).device

    if shuffle_locs:
        batch = shuffle_locations(batch)

    if "y_final_scores" not in batch:
        assert num_scoring_systems > 0 and num_scoring_systems <= len(Scoring)
        if num_scoring_systems == 1:
            assert Scoring.SOS.value == 0
            batch["y_final_scores"] = batch["sos_scores"]
        elif num_scoring_systems == 2:
            # We randomize between all affine combinations of SOS and DSS scoring,
            assert Scoring.SOS.value == 0
            assert Scoring.DSS.value == 1
            assert len(batch["sos_scores"].shape) == 2  # [batch, powers]
            assert batch["sos_scores"].shape[1] == len(POWERS)
            sos_weight = torch.rand(
                (batch["sos_scores"].shape[0], 1),
                dtype=batch["sos_scores"].dtype,
                device=batch["sos_scores"].device,
            )
            batch["y_final_scores"] = batch["sos_scores"] * sos_weight + batch["dss_scores"] * (
                1.0 - sos_weight
            )
            # Explicitly set x_scoring_system to be our random affine combination
            # instead of whatever the dataset game used.
            batch["x_scoring_system"] = torch.cat([sos_weight, 1.0 - sos_weight], dim=1)
        else:
            assert False, "not implemented"

    # prepare teacher-forcing actions
    teacher_force_orders = (
        cand_idxs_to_order_idxs(batch["y_actions"], batch["x_possible_actions"], pad_out=0)
        if torch.rand(1) < p_teacher_force
        else None
    )
    if teacher_force_orders is not None and batch.get("all_powers"):
        # If using all-powers, don't teacher force actions from invalid powers.
        # We *do* teacher force on weak powers, so we use *_any_strength
        orders_valid_any_strength_mask = (
            batch["valid_power_idxs_any_strength"].gather(1, batch["x_power"].clamp(0)).bool()
        )
        teacher_force_orders[~orders_valid_any_strength_mask] = NO_ORDER_ID

    net_module = net.module if isinstance(net, DistributedDataParallel) else net

    # forward pass
    global_order_idxs, local_order_idxs, logits, final_sos = net(
        **{k: v for k, v in batch.items() if k.startswith("x_")},
        temperature=temperature,
        teacher_force_orders=teacher_force_orders,
        need_policy=net_module.has_policy,
        need_value=net_module.has_value,
    )

    batch = batch.to(device)

    # fill in dummy stuff
    if not net_module.has_policy:
        local_order_idxs = batch["y_actions"]
        logits = torch.zeros_like(batch["x_possible_actions"], dtype=torch.float32)
    if not net_module.has_value:
        final_sos = batch["y_final_scores"]

    # reshape
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    y_actions = batch["y_actions"][:, :seq_len]  # [B, S]
    assert logits.shape[0] == y_actions.shape[0]
    assert len(y_actions.shape) == 2  # [B, S]
    assert len(logits.shape) == 3  # [B, S, possible order idx]

    # compute mask for <EOS> tokens from sequences
    valid_action_mask = y_actions != EOS_IDX
    if batch.get("all_powers"):
        valid_action_mask &= (
            batch["valid_power_idxs"].gather(1, batch["x_power"].clamp(0)).bool()[:, :seq_len]
        )
    valid_action_mask = valid_action_mask.to(torch.float32)

    # just for error checking
    observed_logits_masked = valid_action_mask * logits.gather(
        2, y_actions.clamp(0).unsqueeze(-1)
    ).squeeze(-1)
    if observed_logits_masked.min() < -1e7:
        min_score, min_idx = observed_logits_masked.min(0)
        logging.warning(
            f"!!! Got masked order for {get_order_vocabulary()[y_actions[min_idx]]} !!!"
        )

    # calculate policy loss
    policy_loss = policy_loss_fn(
        logits.reshape(batch_size * seq_len, -1), y_actions.clamp(0).reshape(batch_size * seq_len),
    ).reshape(batch_size, seq_len)

    # calculate sum-of-squares value loss
    assert len(batch["y_final_scores"].shape) == 2, batch["y_final_scores"].shape
    assert batch["y_final_scores"].shape[1] == len(POWERS)
    y_final_scores = batch["y_final_scores"].float().squeeze(1)

    if value_loss_use_cross_entropy:
        # not the most numerically stable, but since final_sos is already softmaxed
        # or whatever, this is easiest, and it should still work fine in practice.
        value_loss = -torch.log(final_sos + 1e-30) * y_final_scores
    else:
        value_loss = torch.square(final_sos - y_final_scores)

    value_loss = torch.mean(value_loss, dim=1)  # Mean over power dimension, [B,7] -> [B]

    # a given state appears multiple times in the dataset for different powers,
    # but we always compute the value loss for each power. So we need to reweight
    # the value loss by 1/num_valid_powers
    value_loss_weights = torch.ones_like(value_loss)
    n_valid_powers = batch["valid_power_idxs"].sum(-1)
    value_loss_weights /= n_valid_powers

    # if all_powers is set, the same is true for non-A phase actions
    policy_loss_weights = valid_action_mask
    if batch.get("all_powers"):
        inv_weights = batch["x_in_adj_phase"] + ((1 - batch["x_in_adj_phase"]) * n_valid_powers)
        assert len(inv_weights.shape) == 1
        assert inv_weights.shape[0] == policy_loss_weights.shape[0]
        inv_weights = inv_weights.unsqueeze(1)  # [B, 1] -> [B, 1]
        policy_loss_weights /= inv_weights  # [B, S] /= [B, 1]  broadcasting

    # local order idxs is always padded to 17 whereas logits and therefore policy loss is not,
    # This matters when the longest decode sequence is shorter than 17. We don't need the extra
    # padding, and we need shapes to match between weights and order idxs later, so go ahead and
    # chop it down now.
    local_order_idxs = local_order_idxs[:, :seq_len]

    return (
        policy_loss,
        policy_loss_weights,
        value_loss,
        value_loss_weights,
        local_order_idxs,
        final_sos,
    )


def cand_idxs_to_order_idxs(idxs, candidates, pad_out=EOS_IDX):
    """Convert from idxs in candidates to idxs in ORDER_VOCABULARY

    Arguments:
    - idxs: [B, S] candidate idxs, each 0 - 469, padding=EOS_IDX
    - candidates: [B, S, 469] order idxs of each candidate, 0 - 13k

    Return [B, S] of order idxs, 0 - 13k, padding=pad_out
    """
    mask = idxs.view(-1) != EOS_IDX
    flat_candidates = candidates.view(-1, candidates.shape[2])
    r = torch.empty_like(idxs).fill_(pad_out).view(-1)
    r[mask] = flat_candidates[mask].gather(1, idxs.view(-1)[mask].unsqueeze(1)).view(-1)
    return r.view(*idxs.shape)


def _normalize_each_row_sum_to_one(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.sum(x, dim=1, keepdim=True) + 1e-30)


def calculate_value_accuracy_weighted_count(
    final_sos: torch.Tensor, y_final_scores: torch.Tensor, value_loss_weights: torch.Tensor
) -> float:
    """Return top-1 accuracy"""
    y_final_scores = y_final_scores.squeeze(1)
    actual_winner = y_final_scores == y_final_scores.max(dim=1, keepdim=True).values
    # We could do this, if we wanted accuracy where guessing any of the top-sos powers in a draw
    # gets you a score of 1/N instead of a score of 1
    # actual_winner = _normalize_each_row_sum_to_one(actual_winner.float())
    guessed_winner = _normalize_each_row_sum_to_one(
        (final_sos == final_sos.max(dim=1, keepdim=True).values).float()
    )
    correct_count = torch.sum(actual_winner * guessed_winner, dim=1)
    assert len(correct_count.size()) == 1  # Should be only batch dimension now
    assert len(value_loss_weights.size()) == 1  # Should be only batch dimension now
    assert value_loss_weights.size()[0] == correct_count.size()[0]
    return float((value_loss_weights * correct_count).sum().item())


def calculate_split_accuracy_weighted_counts(
    local_order_idxs: torch.Tensor, batch: DataFields, policy_loss_weights: torch.Tensor
) -> Dict[str, float]:
    device = local_order_idxs.device

    counts: Dict[str, float] = defaultdict(float)

    y_truth: torch.Tensor = batch["y_actions"][
        : (local_order_idxs.shape[0]), : (local_order_idxs.shape[1])
    ].to(device)
    y_truth_global: torch.Tensor = local_order_idxs_to_global(y_truth, batch["x_possible_actions"])

    assert y_truth.shape == policy_loss_weights.shape

    # first compute valid/correct/incorrect masks which we will slice and dice
    # to calculate split accuracies
    is_valid_weighted: torch.Tensor = (y_truth != EOS_IDX).to(torch.float32) * policy_loss_weights
    is_correct = y_truth == local_order_idxs
    is_valid_correct = is_valid_weighted * is_correct
    is_valid_incorrect = is_valid_weighted * ~is_correct

    # total accuracy
    counts["total.y"] = float(is_valid_correct.sum().item())
    counts["total.n"] = float(is_valid_incorrect.sum().item())

    # stats by sequence step
    correct_by_step = is_valid_correct.sum(0).tolist()
    incorrect_by_step = is_valid_incorrect.sum(0).tolist()
    for i in range(len(correct_by_step)):
        counts[f"step.{i}.y"] = correct_by_step[i]
        counts[f"step.{i}.n"] = incorrect_by_step[i]

    # stats by truth loc
    y_truth_loc = ORDER_TO_LOC_IDX.to(device)[y_truth_global]
    y_truth_loc_1h = y_truth_loc.unsqueeze(-1) == torch.arange(len(LOCS)).to(device).view(1, 1, -1)
    correct_by_loc = (is_valid_correct.unsqueeze(-1) * y_truth_loc_1h).sum(0).sum(0).tolist()
    incorrect_by_loc = (is_valid_incorrect.unsqueeze(-1) * y_truth_loc_1h).sum(0).sum(0).tolist()
    for i, loc in enumerate(LOCS):
        counts[f"loc.{loc}.y"] = correct_by_loc[i]
        counts[f"loc.{loc}.n"] = incorrect_by_loc[i]

    # stats by order type
    y_truth_type = ORDER_TO_TYPE_IDX.to(device)[y_truth_global]
    y_truth_type_1h = y_truth_type.unsqueeze(-1) == torch.arange(len(ORDER_TYPES)).to(device).view(
        1, 1, -1
    )
    correct_by_type = (is_valid_correct.unsqueeze(-1) * y_truth_type_1h).sum(0).sum(0).tolist()
    incorrect_by_type = (is_valid_incorrect.unsqueeze(-1) * y_truth_type_1h).sum(0).sum(0).tolist()
    for i, otype in enumerate(ORDER_TYPES):
        counts[f"type.{otype}.y"] = correct_by_type[i]
        counts[f"type.{otype}.n"] = incorrect_by_type[i]

    # stats by season
    correct_by_season = (
        (is_valid_correct.unsqueeze(2) * batch["x_season"].bool().unsqueeze(1))
        .sum(0)
        .sum(0)
        .tolist()
    )
    incorrect_by_season = (
        (is_valid_incorrect.unsqueeze(2) * batch["x_season"].bool().unsqueeze(1))
        .sum(0)
        .sum(0)
        .tolist()
    )
    for i, season in enumerate(SEASONS):
        season = season[0]
        counts[f"season.{season}.y"] = correct_by_season[i]
        counts[f"season.{season}.n"] = incorrect_by_season[i]

    # stats by year
    for year in range(1901, 1921):
        year_encoding = min(max(0.1 * (year - 1901), 0.0), 5.0)
        is_correct_year = torch.abs(batch["x_year_encoded"] - year_encoding) < 0.01
        assert len(is_correct_year.shape) == 2
        assert is_correct_year.shape[0] == is_valid_correct.shape[0]
        assert is_correct_year.shape[1] == 1
        correct = (is_valid_correct * is_correct_year).sum(0).sum(0).tolist()
        incorrect = (is_valid_incorrect * is_correct_year).sum(0).sum(0).tolist()
        counts[f"year.{year}.y"] = correct  # type:ignore
        counts[f"year.{year}.n"] = incorrect  # type:ignore

    return counts


def calculate_split_value_loss_weighted_sums(
    value_loss: torch.Tensor, batch: DataFields, value_loss_weights: torch.Tensor
) -> Dict[str, float]:
    assert len(value_loss.shape) == 1
    assert len(value_loss_weights.shape) == 1

    stats = {}
    # stats by year
    for year in range(1901, 1921):
        year_encoding = min(max(0.1 * (year - 1901), 0.0), 5.0)
        is_correct_year = torch.abs(batch["x_year_encoded"] - year_encoding) < 0.01
        assert len(is_correct_year.shape) == 2
        assert is_correct_year.shape[0] == value_loss.shape[0]
        assert is_correct_year.shape[1] == 1
        weighted_loss = (
            (value_loss * value_loss_weights * is_correct_year.squeeze(1)).sum(0).tolist()
        )
        weight = (value_loss_weights * is_correct_year.squeeze(1)).sum(0).tolist()

        stats[f"year.{year}.wxsum"] = weighted_loss
        stats[f"year.{year}.wsum"] = weight

    return stats


def validate(
    net,
    val_set,
    policy_loss_fn,
    batch_size,
    value_loss_weight: float,
    *,
    value_loss_use_cross_entropy: bool,
    num_scoring_systems: int,
    convert_inputs_to_half=False,
):
    net_device = next(net.parameters()).device

    with torch.no_grad():
        net.eval()

        batch_loss_sums_and_weights: List[Tuple[float, float, float, float]] = []
        batch_acc_split_weighted_counts: List[Dict[str, float]] = []
        batch_value_accuracy_weighted_counts: List[float] = []
        batch_split_value_loss_weighted_sums: List[Dict[str, float]] = []

        for batch_idxs in torch.arange(len(val_set)).split(batch_size):
            batch = val_set[batch_idxs].to(net_device)
            if convert_inputs_to_half:
                batch = batch.to_half_precision()

            y_actions = batch["y_actions"]
            if y_actions.shape[0] == 0:
                logging.warning(
                    "Got an empty validation batch! y_actions.shape={}".format(y_actions.shape)
                )
                continue
            (
                policy_losses,
                policy_loss_weights,
                value_losses,
                value_loss_weights,
                local_order_idxs,
                final_sos,
            ) = process_batch(
                net,
                batch,
                policy_loss_fn,
                value_loss_use_cross_entropy=value_loss_use_cross_entropy,
                num_scoring_systems=num_scoring_systems,
                temperature=0.001,
                p_teacher_force=1.0,
            )

            batch_loss_sums_and_weights.append(
                (
                    float(torch.sum(policy_losses * policy_loss_weights).item()),
                    float(torch.sum(policy_loss_weights).item()),
                    float(torch.sum(value_losses * value_loss_weights).item()),
                    float(torch.sum(value_loss_weights).item()),
                )
            )
            batch_value_accuracy_weighted_counts.append(
                calculate_value_accuracy_weighted_count(
                    final_sos, batch["y_final_scores"], value_loss_weights
                )
            )
            batch_acc_split_weighted_counts.append(
                calculate_split_accuracy_weighted_counts(
                    local_order_idxs, batch, policy_loss_weights
                )
            )
            batch_split_value_loss_weighted_sums.append(
                calculate_split_value_loss_weighted_sums(value_losses, batch, value_loss_weights)
            )

        net.train()

    # validation loss
    # We explicitly track the weight separately, and in the below reductions, we sum the
    # weight*loss and the weight separately. We also incorporate the weight into the
    # accuracy metrics as well. This gives us the following semantic meanings:
    #
    # ploss: The average neg-log-likelihood assigned to the correct order by the model,
    #   weighted uniformly across all orders, (*not* all actions) teacher forcing on
    #   the prior orders.
    # paccuracy: The proportion of the time the model's top order was the real order,
    #   weighted uniformly across all orders, (*not* all actions) teacher forcing on
    #   the prior orders.
    # vloss: The average squared difference between each entry of value vector of the model
    #   and the game outcome, weighted uniformly across all phases.
    # vaccuracy: The proportion of the time the model's top-valued power was the winning power,
    #   weighted uniformly across all phases. In case of equal-sos draw, you get full credit
    #   for guessing any of those powers. This means that full accuracy would be 1, but random
    #   guessing will be better than 1/7.
    ploss_weighted_sum, ploss_total_weight, vloss_weighted_sum, vloss_total_weight = np.sum(
        np.array(batch_loss_sums_and_weights), axis=0
    ).tolist()
    p_loss = ploss_weighted_sum / ploss_total_weight
    v_loss = vloss_weighted_sum / vloss_total_weight
    valid_loss = (1 - value_loss_weight) * p_loss + value_loss_weight * v_loss

    # validation accuracy
    valid_v_accuracy = sum(batch_value_accuracy_weighted_counts) / vloss_total_weight

    # combine accuracy splits
    split_counts = reduce(
        lambda x, y: Counter({k: x[k] + y[k] for k in set(x.keys()) | set(y.keys())}),
        batch_acc_split_weighted_counts,
        Counter(),
    )
    split_pcts = {
        k: split_counts[k + ".y"] / (split_counts[k + ".y"] + split_counts[k + ".n"] + 1e-6)
        for k in [k.rsplit(".", 1)[0] for k in split_counts.keys()]
    }

    value_split_totals = reduce(
        lambda x, y: Counter({k: x[k] + y[k] for k in set(x.keys()) | set(y.keys())}),
        batch_split_value_loss_weighted_sums,
        Counter(),
    )
    value_splits = {
        k: value_split_totals[k + ".wxsum"] / (value_split_totals[k + ".wsum"] + 1e-6)
        for k in [k.rsplit(".", 1)[0] for k in value_split_totals.keys()]
    }

    # total policy accuracy is computed in the splits
    valid_p_accuracy = split_pcts["total"]

    return valid_loss, p_loss, v_loss, valid_p_accuracy, valid_v_accuracy, split_pcts, value_splits


def main_subproc(*args, **kwargs):
    try:
        _main_subproc(*args, **kwargs)
    finally:
        wandb.finish()  # type: ignore


def _main_subproc(
    rank: int,
    world_size: int,
    args: conf.conf_cfgs.TrainTask,
    train_set: Dataset,
    val_sets: Dict[str, Dataset],
):
    heyhi.setup_logging(label=f"train_{rank}")
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        # distributed training setup
        mp_setup(rank, world_size, args.seed)
        atexit.register(mp_cleanup)
        torch.cuda.set_device(rank)
        logging.info("CUDA device: " + torch.cuda.get_device_name())
    else:
        assert rank == 0 and world_size == 1

    # load checkpoint if specified
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logging.info("Loading checkpoint at {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location="cuda:{}".format(rank))
    else:
        checkpoint = None

    is_master = (rank == 0) and heyhi.is_master()
    metric_logging = Logger(
        is_master=is_master, log_wandb=args.wandb.enabled and not heyhi.is_adhoc()
    )
    global_step = checkpoint.get("global_step", 0) if checkpoint else 0

    def log_scalars(**scalars):
        return metric_logging.log_metrics(scalars, step=global_step, sanitize=True)

    logging.info("Init model...")
    net = new_model(args)

    if is_master:
        if initialize_wandb_if_enabled(args, "train_sl"):
            wandb.watch(net)  # type:ignore

    logging.debug("Model parameters:")
    trainable_parameter_count = 0
    total_parameter_count = 0
    for parameter in net.parameters():
        size = parameter.size()
        trainable_parameter_count += parameter.numel() if parameter.requires_grad else 0
        total_parameter_count += parameter.numel()
        logging.debug(
            "Found parameter tensor with shape: {} (requires_grad {})".format(
                str(size), parameter.requires_grad
            )
        )
    logging.info("TRAINABLE parameter count in model: {}".format(trainable_parameter_count))
    logging.info("TOTAL parameter count in model: {}".format(total_parameter_count))

    if net.get_training_permute_powers() != args.training_permute_powers:
        logging.warning(
            f"WARNING: Overriding net.training_permute_powers to {args.training_permute_powers}"
        )
    net.set_training_permute_powers(args.training_permute_powers)

    # send model to GPU
    if has_gpu:
        logging.debug("net.cuda({})".format(rank))
        net.cuda(rank)
        logging.debug("net {} DistributedDataParallel".format(rank))
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
        logging.debug("net {} DistributedDataParallel done".format(rank))

    # load from checkpoint if specified
    if checkpoint:
        logging.debug("net.load_state_dict")
        net.load_state_dict(checkpoint["model"], strict=True)

    assert args.value_loss_weight is not None
    assert args.num_epochs is not None
    assert args.clip_grad_norm is not None
    assert args.value_decoder_clip_grad_norm is not None
    assert args.lr is not None
    assert args.lr_decay is not None
    lr_decay = args.lr_decay

    # create optimizer, from checkpoint if specified
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optim = torch.optim.Adam(
        net.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
    )
    warmup_epochs = 0 if not args.warmup_epochs else args.warmup_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        (
            lambda epoch: lr_decay ** epoch
            * (1.0 if epoch >= warmup_epochs else (epoch + 1) / warmup_epochs)
        ),
    )

    if checkpoint:
        optim.load_state_dict(checkpoint["optim"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    scaler = None
    if args.auto_mixed_precision:
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # load best losses to not immediately overwrite best checkpoints
    best_loss = checkpoint.get("best_loss") if checkpoint else None
    best_p_loss = checkpoint.get("best_p_loss") if checkpoint else None
    best_v_loss = checkpoint.get("best_v_loss") if checkpoint else None

    if has_gpu:
        train_set_sampler = DistributedSampler(train_set)
    else:
        train_set_sampler = RandomSampler(train_set)

    for epoch in range(checkpoint["epoch"] + 1 if checkpoint else 0, args.num_epochs):
        if has_gpu:
            train_set_sampler.set_epoch(epoch)  # type: ignore
        batches = torch.tensor(list(iter(train_set_sampler)), dtype=torch.long).split(
            args.batch_size
        )

        ploss_weighted_sum_since_last_log = 0.0
        ploss_weight_since_last_log = 0.0
        vloss_weighted_sum_since_last_log = 0.0
        vloss_weight_since_last_log = 0.0

        for batch_i, batch_idxs in enumerate(batches):
            batch = train_set[batch_idxs]
            # import nest

            # print(nest.map(lambda x: x.dtype if hasattr(x, "dtype") else x, batch))
            logging.debug(f"Zero grad {batch_i} ...")

            # check batch is not empty
            if (batch["y_actions"] == EOS_IDX).all():
                logging.warning("Skipping empty epoch {} batch {}".format(epoch, batch_i))
                continue

            # learn
            logging.debug("Starting epoch {} batch {}".format(epoch, batch_i))
            optim.zero_grad()

            torch_context = (
                torch.cuda.amp.autocast_mode.autocast()
                if args.auto_mixed_precision
                else nullcontext()
            )
            with torch_context:
                if args.all_powers:
                    maybe_augment_targets_inplace(
                        batch,
                        single_chances=args.all_powers_add_single_chances,
                        double_chances=args.all_powers_add_double_chances,
                        power_conditioning=args.power_conditioning,
                    )
                (
                    policy_losses,
                    policy_loss_weights,
                    value_losses,
                    value_loss_weights,
                    _,
                    _,
                ) = process_batch(
                    net,
                    batch,
                    policy_loss_fn,
                    value_loss_use_cross_entropy=args.value_loss_use_cross_entropy,
                    num_scoring_systems=args.num_scoring_systems,
                    p_teacher_force=args.teacher_force,
                    shuffle_locs=args.shuffle_locs,
                )
                # Normalizing loss by a slightly weird thing, so as to preserve backwards compatibility
                # with old choices for formulating the loss for the purposes of optimization
                # Note that unlike the way we explicitly avoided overweighting the last batch
                # for value, currently for policy it still overweights it.
                p_loss_opt = torch.sum(policy_losses * policy_loss_weights) / torch.sum(
                    policy_loss_weights > 0.0
                )
                # sum + Explicit division by batch size instead of mean ensures that we don't massively
                # overweight data that happens to fall into the last batch when the last batch has fewer
                # than the full batch size amount of data.
                v_loss_opt = torch.sum(value_losses * value_loss_weights) / args.batch_size
                loss_opt = (
                    1 - args.value_loss_weight
                ) * p_loss_opt + args.value_loss_weight * v_loss_opt
            # backward
            if scaler:
                scaler.scale(loss_opt).backward()
                scaler.unscale_(optim)
            else:
                loss_opt.backward()

            # clip gradients, step
            value_decoder_grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                getattr(net, "module", net).value_decoder.parameters(),  # type: ignore
                args.value_decoder_clip_grad_norm,
            )
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                net.parameters(), args.clip_grad_norm
            )
            if scaler:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            # We only log ploss and vloss every so often, but we accumulate the loss every batch,
            # so that when we log, we can display something slightly less noisy than logging just
            # one batch's values. And ever so slightly less biased by variable weighting, since
            # normalizing by weight every batch is not quite the same as normalizing by weight
            # across the whole dataset.
            #
            # The reported loss here is intended to be entirely consistent and comparble with
            # the loss computed in validation, with the tiny difference of the fact that it is a
            # batch weighted-average rather than a whole dataset weighted-average.
            ploss_weighted_sum_since_last_log += float(
                torch.sum(policy_losses * policy_loss_weights).item()
            )
            ploss_weight_since_last_log += float(torch.sum(policy_loss_weights).item())
            vloss_weighted_sum_since_last_log += float(
                torch.sum(value_losses * value_loss_weights).item()
            )
            vloss_weight_since_last_log += float(torch.sum(value_loss_weights).item())

            # log diagnostics
            LOG_EVERY_BATCHES = 10
            if is_master and batch_i % LOG_EVERY_BATCHES == 0:
                scalars = {
                    "epoch": epoch,
                    "batch": batch_i,
                    "optim/lr": optim.state_dict()["param_groups"][0]["lr"],
                    "optim/grad_norm": grad_norm,
                    "optim/value_decoder_grad_norm": value_decoder_grad_norm,
                    "train/p_loss": ploss_weighted_sum_since_last_log
                    / ploss_weight_since_last_log,
                    "train/v_loss": vloss_weighted_sum_since_last_log
                    / vloss_weight_since_last_log,
                }
                ploss_weighted_sum_since_last_log = 0.0
                ploss_weight_since_last_log = 0.0
                vloss_weighted_sum_since_last_log = 0.0
                vloss_weight_since_last_log = 0.0

                log_scalars(**scalars)
                logging.info(
                    "epoch {} batch {} / {}, ".format(epoch, batch_i, len(batches))
                    + " ".join(f"{k}= {v}" for k, v in scalars.items())
                )
            global_step += 1
            if args.epoch_max_batches and batch_i + 1 >= args.epoch_max_batches:
                logging.info("Exiting early due to epoch_max_batches")
                break

        lr_scheduler.step()

        # calculate validation loss/accuracy
        if not args.skip_validation and is_master:
            logging.info("Calculating val loss...")
            assert MAIN_VALIDATION_SET_SUFFIX in val_sets, list(val_sets)
            for suffix, val_set in val_sets.items():
                (
                    valid_loss,
                    valid_p_loss,
                    valid_v_loss,
                    valid_p_accuracy,
                    valid_v_accuracy,
                    split_pcts,
                    value_splits,
                ) = validate(
                    net,
                    val_set,
                    policy_loss_fn,
                    args.batch_size,
                    value_loss_weight=args.value_loss_weight,
                    value_loss_use_cross_entropy=args.value_loss_use_cross_entropy,
                    num_scoring_systems=args.num_scoring_systems,
                )
                scalars = {
                    "epoch": epoch,
                    f"valid{suffix}/loss": valid_loss,
                    f"valid{suffix}/p_loss": valid_p_loss,
                    f"valid{suffix}/v_loss": valid_v_loss,
                    f"valid{suffix}/p_accuracy": valid_p_accuracy,
                    f"valid{suffix}/v_accuracy": valid_v_accuracy,
                }

                log_scalars(**scalars)
                logging.info("Validation " + " ".join([f"{k}= {v}" for k, v in scalars.items()]))
                for k, v in sorted(split_pcts.items()):
                    logging.info(f"val split epoch= {epoch}: pacc {k} = {v}")
                for k, v in sorted(value_splits.items()):
                    logging.info(f"val split epoch= {epoch}: vloss {k} = {v}")

                # save model
                if args.checkpoint and is_master and suffix == MAIN_VALIDATION_SET_SUFFIX:
                    obj = {
                        "model": net.state_dict(),
                        "optim": optim.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "valid_p_accuracy": valid_p_accuracy,
                        "args": heyhi.conf_to_dict(args),
                        "best_loss": best_loss,
                        "best_p_loss": best_p_loss,
                        "best_v_loss": best_v_loss,
                    }
                    logging.info("Saving checkpoint to {}".format(args.checkpoint))
                    torch.save(obj, args.checkpoint)

                    if epoch % 10 == 0:
                        torch.save(obj, args.checkpoint + ".epoch_" + str(epoch))
                    if best_loss is None or valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(obj, args.checkpoint + ".best")
                    if best_p_loss is None or valid_p_loss < best_p_loss:
                        best_p_loss = valid_p_loss
                        torch.save(obj, args.checkpoint + ".bestp")
                    if best_v_loss is None or valid_v_loss < best_v_loss:
                        best_v_loss = valid_v_loss
                        torch.save(obj, args.checkpoint + ".bestv")


def mp_setup(local_rank, world_size, seed):
    if "SLURM_JOB_NODELIST" in os.environ:
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        master_addr = hostnames.split()[0].decode("utf-8")
        # We are just assuming if we use > 1 machine, then we use 8 gpus per machine.
        rank = heyhi.get_job_env().global_rank * 8 + local_rank
    else:
        master_addr = "localhost"
        rank = local_rank

    logging.info("MASTER_ADDR=%s local_rank=%s global_rank=%s", master_addr, local_rank, rank)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(seed)
    random.seed(seed)


def mp_cleanup():
    torch.distributed.destroy_process_group()


def get_datasets_from_cfg(args: conf.conf_cfgs.TrainTask) -> Tuple[Dataset, Dict[str, Dataset]]:
    """Returns a 2-tuple (train_set, dict of val_sets).

    The main validation dataset has MAIN_VALIDATION_SET_SUFFIX key.
    """
    train_dataset = Dataset(
        args.dataset_params,
        use_validation=False,
        all_powers=args.all_powers,
        input_version=args.input_version,
    )
    val_datasets = {}
    val_datasets[MAIN_VALIDATION_SET_SUFFIX] = Dataset(
        args.dataset_params,
        use_validation=True,
        all_powers=False,
        input_version=args.input_version,
    )
    if args.all_powers:
        val_datasets["_all"] = Dataset(
            args.dataset_params,
            use_validation=True,
            input_version=args.input_version,
            all_powers=True,
        )

    logging.info(f"Train dataset: {train_dataset.stats_str()}")
    for suffix, dataset in val_datasets.items():
        logging.info(f"Val dataset(suffix={suffix}): {dataset.stats_str()}")

    return train_dataset, val_datasets


def run_with_cfg(args: conf.conf_cfgs.TrainTask):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)  # type: ignore

    logging.warning("Args: {}, file={}".format(args, os.path.abspath(__file__)))

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus * heyhi.get_job_env().num_nodes
    logging.info(
        "Using {} GPUs".format(n_gpus) + (", debug_no_mp=True" if args.debug_no_mp else "")
    )

    if args.all_powers_add_single_chances is not None:
        assert args.all_powers
    if args.all_powers_add_double_chances is not None:
        assert args.all_powers

    train_dataset, val_datasets = get_datasets_from_cfg(args)

    # required when using multithreaded DataLoader
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if args.debug_no_mp:
        main_subproc(0, 1, args, train_dataset, val_datasets)
    else:
        torch.multiprocessing.spawn(  # type:ignore
            main_subproc, nprocs=n_gpus, args=(world_size, args, train_dataset, val_datasets)
        )
