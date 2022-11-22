#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Union

import torch

from fairdiplomacy.models.consts import POWERS, LOCS
from fairdiplomacy.models.state_space import (
    get_order_vocabulary,
    EOS_IDX,
)
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global


def top_p_filtering(
    logits: torch.Tensor, top_p: Union[float, torch.Tensor], min_tokens_to_keep=1
) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) filtering.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
        logits: tensor of shape [batch_size, vocab]. Logits distribution shape
        top_p: float or tensor of shape [batch_size, 1]. Keep the top tokens
            with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al.
            (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep: int, make sure we keep at least
            min_tokens_to_keep per batch example in the output

    Returns:
        top_p_mask: boolean tensor of shape [batch_size, vocab] with elements to remove.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return indices_to_remove


def he_init(shape):
    fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
    init_range = (2.0 / fan_in) ** 0.5
    return torch.randn(shape) * init_range


def explain_base_strategy_model_decoder_inputs(
    *, loc_idxs, all_cand_idxs, power, teacher_force_orders, teacher_forces_global: bool = True
) -> None:
    """This is a debugging function that takes input to base_strategy_model and tries to unpack it and print."""
    if teacher_force_orders is None:
        return
    print("loc_idxs.shape = ", loc_idxs.shape)
    print("all_cand_idxs.shape = ", all_cand_idxs.shape)
    print("power.shape = ", power.shape)
    print("teacher_force_orders.shape = ", teacher_force_orders.shape)
    batch_size, num_locs = loc_idxs.shape
    _, max_seq_len, four_hundred_sixty_nine = all_cand_idxs.shape
    assert num_locs == len(LOCS)
    assert four_hundred_sixty_nine == 469
    assert power.shape == (batch_size, max_seq_len)
    assert teacher_force_orders.shape == (batch_size, max_seq_len)

    if not teacher_forces_global:
        print("Will try to convert teacher_force_orders from local to global")
        teacher_force_orders = local_order_idxs_to_global(
            teacher_force_orders, all_cand_idxs, clamp_and_mask=True
        )
    limit = 8
    for batch_index in range(batch_size):
        if batch_index >= limit:
            break
        print("#" * 80)
        print("Batch element", batch_index)
        _explain_base_strategy_model_decoder_inputs_single_element(
            loc_idxs[batch_index],
            all_cand_idxs[batch_index],
            power[batch_index],
            teacher_force_orders[batch_index],
        )


def _explain_base_strategy_model_decoder_inputs_single_element(
    loc_idxs, all_cand_idxs, power, teacher_force_orders
):

    vocab = get_order_vocabulary()
    vocab_dict = dict(enumerate(vocab))

    power_strs = [POWERS[x] for x in power if x != EOS_IDX]
    print(
        "Powers (%d):" % len(power_strs),
        power_strs,
        "and",
        int((power == EOS_IDX).sum()),
        "EOSes",
    )
    orders = [vocab[x] for x in teacher_force_orders if x != EOS_IDX]
    print(
        "Teacher force orders (%d):" % len(orders),
        orders,
        "and",
        int((teacher_force_orders == EOS_IDX).sum()),
        "EOSes",
    )
    if (loc_idxs == -1).all():
        print("Locations are EMPTY! (no orders for the phase?)")
    else:
        valid_loc_ids = sorted(
            [(LOCS[i], int(x)) for i, x in enumerate(loc_idxs) if x != -1], key=lambda x: x[1]
        )
        if (loc_idxs == -2).any():
            assert (loc_idxs < 0).all()
            print("Locations (adjustment phase, no order):", valid_loc_ids)
        else:
            print("Locations (move/retreat phase, in order):", valid_loc_ids)

    print("=== per position")
    for i in range(len(all_cand_idxs)):
        limit = 5
        if (all_cand_idxs[i] == EOS_IDX).all():
            break
        print(
            "%2d: %s y=%40r cands=%s + %d more"
            % (
                i,
                dict(enumerate(POWERS)).get(int(power[i]), "UNK")[:3],
                vocab_dict.get(int(teacher_force_orders[i]), "IndexError"),
                [vocab[idx] for idx in all_cand_idxs[i, :limit] if idx != EOS_IDX],
                max(0, (all_cand_idxs[i] != EOS_IDX).sum() - limit),
            )
        )
        if not (all_cand_idxs[i] == teacher_force_orders[i]).any():
            print("ERROR teacher force orders is not in the candidates")
