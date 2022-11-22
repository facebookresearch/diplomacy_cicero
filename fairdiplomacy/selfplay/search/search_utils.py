#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Callable, Optional, Tuple, Union
import math
import time
import typing
import collections
import torch
from conf import conf_cfgs

from fairdiplomacy.agents.base_strategy_model_wrapper import compute_action_logprobs
from fairdiplomacy.agents.plausible_order_sampling import renormalize_policy
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import (
    encode_all_powers_action,
    encode_power_actions,
    maybe_augment_targets_inplace,
)
from fairdiplomacy.models.consts import N_SCS, POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.base_strategy_model.load_model import SomeBaseStrategyModel
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.typedefs import JointPolicy, PowerPolicies
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global

import nest

from fairdiplomacy.utils.zipn import unzip2


def unparse_device(device: str) -> int:
    if device == "cpu":
        return -1
    assert device.startswith("cuda:"), device
    return int(device.split(":")[-1])


@torch.no_grad()
def create_research_targets_single_rollout(
    is_explore_tensor: torch.Tensor,
    episode_reward: torch.Tensor,
    predicted_values: torch.Tensor,
    alive_powers: torch.Tensor,
    discounting: float = 1.0,
) -> torch.Tensor:
    """Creates a target for value function.

    Args:
        is_explore_tensor: bool tensor [T, power].
        episode_reward: float tensor [power].
        alive_powers: bool tensor [T, power], alive power at the beginning of the end of a phase.
        predicted_values: float tensor [T, power].
        discounting: simplex discounting factor.

    Returns:
        tatgets: float tensor [T, power].
    """
    assert is_explore_tensor.shape[1:] == episode_reward.shape, (
        is_explore_tensor.shape,
        episode_reward.shape,
    )
    assert is_explore_tensor.shape == predicted_values.shape, (
        is_explore_tensor.shape,
        predicted_values.shape,
    )
    # True in production but not in tests
    # assert is_explore_tensor.shape[1] == len(POWERS)
    explore_powers_len = is_explore_tensor.shape[1]

    # Make it so that when any power explores, every power bootstraps at that point
    is_explore_tensor = torch.any(is_explore_tensor, dim=1, keepdim=True)
    is_explore_tensor = torch.repeat_interleave(is_explore_tensor, explore_powers_len, dim=1)

    alive_powers = alive_powers.float()
    # Assuming being alive after 1 phase <-> being alive at the start of the game.
    alive_powers = torch.cat([alive_powers[:1], alive_powers[:-1]], dim=0)
    current_value = episode_reward
    targets = []
    for i in range(len(is_explore_tensor) - 1, -1, -1):
        current_value = torch.where(is_explore_tensor[i], predicted_values[i], current_value)
        targets.append(current_value)
        if discounting < 1.0:
            simplex_center = alive_powers[i] / alive_powers[i].sum()
            simplex_direction = simplex_center - current_value
            current_value = current_value + simplex_direction * (1 - discounting)
    targets = torch.stack(list(reversed(targets))).detach()
    return targets


def pack_adjustment_phase_orders(orders: torch.Tensor) -> torch.Tensor:
    """Pack adjustment orders.

    Adjustment orders are per power by default, but we pack them into single
    tensor.


     (max_actions, 7, N_SCS) -> (max_actions, N_SCS).
    """

    assert len(orders.shape) == 3, orders.shape
    assert orders.shape[-1] == N_SCS, orders.shape

    ret = orders.new_full((orders.shape[0], N_SCS), EOS_IDX)
    lengths = (orders != EOS_IDX).long().sum(-1)
    ret[:, : len(POWERS)] = lengths
    for row_id, per_power_orders in enumerate(orders):
        offset = len(POWERS)
        for power_id, orders in enumerate(per_power_orders):
            length = lengths[row_id, power_id]
            assert offset + length < N_SCS
            ret[row_id, offset : offset + length] = orders[:length]
            offset += length
    return ret


def batch_unpack_adjustment_phase_orders(orders: torch.Tensor) -> torch.Tensor:
    """Unpack adjustment orders.


     (batch, N_SCS) -> (batch, 7, N_SCS).
    """
    ret = orders.new_full((orders.shape[0], len(POWERS), orders.shape[1]), EOS_IDX)
    for row_id, per_power_orders_packed in enumerate(orders):
        offset = len(POWERS)
        for power_id in range(len(POWERS)):
            length = orders[row_id, power_id]
            ret[row_id, power_id, :length] = per_power_orders_packed[offset : offset + length]
            offset += length
    return ret


def power_prob_distributions_to_tensors(
    all_power_prob_distributions: Union[PowerPolicies, JointPolicy],
    max_actions: int,
    x_possible_actions: torch.Tensor,
    x_in_adj_phase: bool,
    x_power: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converting the policies to 2 tensors orders and probs.

    For single power models (PowerPolicies)
        orders (7 X max_actions x MAX_SEQ_LEN)
        probs (7 x max_actions)

    For all power models (JointPolicy)
        orders (1 x max_actions x MAX_SEQ_LEN)
        probs (1 x max_actions)
    """
    assert len(x_possible_actions.shape) == 3 and x_possible_actions.shape[0] == len(
        POWERS
    ), x_possible_actions.shape
    if isinstance(all_power_prob_distributions, dict):
        all_power_prob_distributions = typing.cast(PowerPolicies, all_power_prob_distributions)
        return power_prob_distributions_to_tensors_independent(
            all_power_prob_distributions, max_actions, x_possible_actions, x_in_adj_phase
        )
    else:
        assert x_power is not None, " Required for joint policies"
        assert isinstance(all_power_prob_distributions, list)
        return power_prob_distributions_to_tensors_joint(
            all_power_prob_distributions, max_actions, x_possible_actions, x_in_adj_phase, x_power
        )


def compute_marginal_policy(joint_policy: JointPolicy) -> PowerPolicies:
    power_policies_builder = collections.defaultdict(lambda: collections.defaultdict(float))
    for joint_action, prob in joint_policy:
        for power, action in joint_action.items():
            power_policies_builder[power][action] += prob
    power_policies = {p: {k: v for k, v in d.items()} for p, d in power_policies_builder.items()}
    renormalize_policy(power_policies)
    return power_policies


def power_prob_distributions_to_tensors_joint(
    all_power_prob_distributions: JointPolicy,
    max_actions: int,
    x_possible_actions: torch.Tensor,
    x_in_adj_phase: bool,
    x_power: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converting the policies to 2 tensors  orders and probs.

        orders (1, max_actions x MAX_SEQ_LEN)
        probs (1, max_actions)
    """
    orders_tensors = torch.full((max_actions, N_SCS), EOS_IDX, dtype=torch.long)
    probs_tensor = torch.zeros((max_actions))
    if x_in_adj_phase:
        tmp = orders_tensors.new_full((max_actions, len(POWERS), N_SCS), EOS_IDX)
        for i, (joint_action, prob) in enumerate(all_power_prob_distributions):
            for power_idx, power in enumerate(POWERS):
                if not joint_action.get(power):
                    # Skipping non-acting powers.
                    continue
                action = joint_action[power]
                action_tensor, good = encode_power_actions(
                    action, x_possible_actions[power_idx], x_in_adj_phase, max_seq_len=N_SCS
                )
                assert good, (power, action, x_possible_actions[power_idx])
                tmp[i, power_idx] = action_tensor
            probs_tensor[i] = prob
        orders_tensors = pack_adjustment_phase_orders(tmp)
    else:
        for i, (joint_action, prob) in enumerate(all_power_prob_distributions):
            orders_tensors[i], valid_mask = encode_all_powers_action(
                joint_action, x_possible_actions, x_power, x_in_adj_phase
            )
            assert valid_mask.all(), all_power_prob_distributions
            probs_tensor[i] = prob
    return orders_tensors.unsqueeze(0), probs_tensor.unsqueeze(0)


def power_prob_distributions_to_tensors_independent(
    all_power_prob_distributions: PowerPolicies,
    max_actions: int,
    x_possible_actions: torch.Tensor,
    x_in_adj_phase: bool,
    max_seq_len=MAX_SEQ_LEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converting the policies to 2 tensors  orders and probs.

        orders (7 X max_actions x MAX_SEQ_LEN)
        probs (7 x max_actions)
    """
    orders_tensors = torch.full((len(POWERS), max_actions, max_seq_len), EOS_IDX, dtype=torch.long)
    probs_tensor = torch.zeros((len(POWERS), max_actions))
    for power_idx, power in enumerate(POWERS):
        if not all_power_prob_distributions.get(power):
            # Skipping non-acting powers.
            continue
        actions, probs = unzip2(all_power_prob_distributions[power].items())
        for action_idx, action in enumerate(actions):
            action_tensor, good = encode_power_actions(
                action, x_possible_actions[power_idx], x_in_adj_phase
            )
            assert good, (power, action, x_possible_actions[power_idx])
            orders_tensors[power_idx, action_idx, :MAX_SEQ_LEN] = action_tensor
        probs_tensor[power_idx, : len(probs)] = torch.as_tensor(probs)
    return orders_tensors, probs_tensor


def compute_search_policy_cross_entropy_sampled(
    model: SomeBaseStrategyModel,  # Could be a DDP model instead.
    obs: dict,
    search_policy_orders: torch.Tensor,
    search_policy_probs: torch.Tensor,
    blueprint_probs=None,
    mask=None,
    is_move_phase: Optional[torch.Tensor] = None,
    is_adj_phase: Optional[torch.Tensor] = None,
    using_ddp=False,
    max_prob_cap=1.0,
    is_all_powers: bool = False,
    power_conditioning: Optional[conf_cfgs.PowerConditioning] = None,
    single_power_chances: Optional[float] = None,
    six_power_chances: Optional[float] = None,
):
    """Compute cross entropy loss between model's predictions and SearchBot policy.

    Uses a single sample from the model's policy

    Args:
        model: base_strategy_model model
        obs: dict of obs.
        search_policy_orders: Long tensor [T, B, seven, max_actions, MAX_SEQ_LEN]
        search_policy_probs: Float tensor [T, B, seven, max_actions]
        blueprint_probs: None or Float tensor [T, B, 7, max_actions]
        mask: Bool tensor [T, B, seven]. True, if the policy actually was
            computed at the position.
        is_move_phase: if set, should be [T, B] bool tensor.
        is_adj_phase: it set, should be [T, B] bool tensor. Required for all-power models.
        using_ddp: for distributed data parallel, set True to optimize tensor padding
        is_all_powers: whether the input and the mode is all-power or single-power.
        power_conditioning: optional config to train with conditioning in all_powers mode.
        single_power_chances: if set, will train allpower model on single power inputs.
        six_power_chances: if set, will train allpower model on six power inputs.

    seven is 7 for single-power model and 1 for all-power model

    Returns: a tuple (loss, metrics)
        loss, scalar
        metrics: dict
    """
    # print("XX", nest.map(lambda x: x.shape if hasattr(x, "shape") else x, obs))
    # print("YY", search_policy_orders.shape)
    if mask is not None:
        # An optimization: if there are phases completely masked, we can filter
        # them out in the inputs.
        # We'll gather all valid phases and reshape them into something with
        # time dimension equals 1. This only works for non-recurrent networks.

        # Mask for phases where at least one power had valid entry
        phase_mask = mask.any(dim=2)
        obs = nest.map(lambda x: _apply_2d_mask(x, phase_mask), obs)
        search_policy_orders = _apply_2d_mask(search_policy_orders, phase_mask)
        search_policy_probs = _apply_2d_mask(search_policy_probs, phase_mask)
        if blueprint_probs is not None:
            blueprint_probs = _apply_2d_mask(blueprint_probs, phase_mask)
        if is_move_phase is not None:
            is_move_phase = _apply_2d_mask(is_move_phase, phase_mask)

        mask = _apply_2d_mask(mask, phase_mask)

    # Shape: [T, B, (7 or 1), max_actions, (MAX_SEQ_LEN or N_SCS)]
    time_sz, batch_sz, seven, _, max_num_orders = search_policy_orders.shape

    if is_all_powers:
        assert seven == 1, search_policy_orders.shape
        assert max_num_orders == N_SCS, max_num_orders
        if mask is not None:
            assert tuple(mask.shape) == (time_sz, batch_sz, seven), (
                search_policy_orders.shape,
                mask.shape,
            )
        blueprint_probs = None  # Can't handle per-power probs.
    else:
        assert seven == 7, search_policy_orders.shape
        assert max_num_orders == MAX_SEQ_LEN, max_num_orders
        assert power_conditioning is None, "Cannot use in single power mode"

    # Prepare to sample an action
    # new shape: [T * B * seven, max_actions]
    flat_search_policy_probs = search_policy_probs.flatten(end_dim=2)
    # new shape: [T * B * seven, max_actions, max_num_orders]
    flat_search_policy_local_orders = search_policy_orders.flatten(end_dim=2)

    # Sample an action
    # shape: [T * B * seven, 1]
    sampled_action_indcies = torch.multinomial(
        flat_search_policy_probs + (flat_search_policy_probs.sum(-1) < 1e-3).float().unsqueeze(-1),
        num_samples=1,
    )

    # Select the action.
    # new shape: [T * B * seven, 1]
    flat_search_policy_probs = torch.gather(flat_search_policy_probs, 1, sampled_action_indcies)
    # new shape: [T * B * seven, 1, max_num_orders]
    flat_search_policy_local_orders = torch.gather(
        flat_search_policy_local_orders,
        1,
        sampled_action_indcies.unsqueeze(-1).expand(time_sz * batch_sz * seven, 1, max_num_orders),
    )

    # Reshape power dimension back. The real (seven = 7).
    # flat_search_policy_local_orders:
    #    [T * B * seven, 1, max_num_orders] -> [T * B, 7, max_num_orders]
    if is_all_powers:
        # The fun things are going on here.
        # For non-adjustment phases we have the w
        assert is_adj_phase is not None
        flat_is_adj_phase = is_adj_phase.view(-1)
        flat_search_policy_local_orders_extended = flat_search_policy_local_orders.new_full(
            (time_sz * batch_sz, 7, max_num_orders), EOS_IDX
        )
        flat_search_policy_local_orders_extended[
            ~flat_is_adj_phase, 0
        ] = flat_search_policy_local_orders.squeeze(1)[~flat_is_adj_phase]
        flat_search_policy_local_orders_extended[
            flat_is_adj_phase
        ] = batch_unpack_adjustment_phase_orders(
            flat_search_policy_local_orders.squeeze(1)[flat_is_adj_phase]
        )
        flat_search_policy_local_orders = flat_search_policy_local_orders_extended
    else:
        flat_search_policy_local_orders = flat_search_policy_local_orders.view(
            time_sz * batch_sz, seven, max_num_orders
        )
    flat_obs = nest.map(lambda x: x.flatten(end_dim=1), obs)
    if power_conditioning is not None:
        batch = DataFields(**flat_obs, y_actions=flat_search_policy_local_orders)
        maybe_augment_targets_inplace(
            batch,
            single_chances=single_power_chances,
            double_chances=None,
            six_chances=six_power_chances,
            power_conditioning=power_conditioning,
        )
        flat_search_policy_local_orders = batch.pop("y_actions")
        flat_obs = dict(**batch)

    flat_search_policy_global_orders = local_order_idxs_to_global(
        flat_search_policy_local_orders, flat_obs["x_possible_actions"]
    ).long()

    # Logits shape: [T * B, 7, max_num_orders, 469]
    _, _, logits, _ = model(
        **flat_obs,
        temperature=1.0,
        teacher_force_orders=flat_search_policy_global_orders.clamp(min=0),  # EOS_IDX = -1 -> 0
        pad_to_max=(not using_ddp),
        need_value=False,
    )
    # Shape [T * B, 7]
    logprobs = compute_action_logprobs(flat_search_policy_local_orders, logits)

    # Shape: [T * B, 7].
    maybe_flat_mask = mask.flatten(end_dim=1) if mask is not None else True
    valid_actions = (
        (flat_search_policy_local_orders != EOS_IDX).any(-1) & maybe_flat_mask
    ).float()

    metrics = {}
    metrics["loss_inner/valid_share"] = valid_actions.mean()
    if is_move_phase is not None:
        with torch.no_grad():
            valid_move_mask = valid_actions * is_move_phase.view(-1, 1).float()
            metrics["loss_inner/loss_moves"] = -(logprobs * valid_move_mask).sum() / (
                valid_move_mask.sum() + 1e-10
            )
    if max_prob_cap < 1.0:
        valid_actions_capped = (
            valid_actions * (logprobs.detach() <= math.log(max_prob_cap + 1e-10)).float()
        )
        metrics["loss_inner/valid_below_probability_cap"] = valid_actions_capped.mean()
    else:
        valid_actions_capped = valid_actions

    loss = -(logprobs * valid_actions_capped).sum() / (valid_actions_capped.sum() + 1e-10)

    return loss, metrics


def _apply_2d_mask(x, mask):
    """Applies a mask over first 2 timenstions of a tensor.

    Args:
        x: tensor of shape [T, B, ...any].
        mask: bool tensor of shape [T, B].

    Returns:
        A subset of `x` with shape [1, selected_subset(T * B), ...any].
    """
    return x.flatten(end_dim=1)[mask.view(-1)].unsqueeze(0)


def compute_search_policy_entropy(search_policy_orders, search_policy_probs, mask=None):
    """Compute entropy of the search policy.

    Args:
        search_policy_orders: Long tensor [T, B, 7, max_actions, MAX_SEQ_LEN]
        search_policy_probs: Float tensor [T, B, 7, max_actions]
        mask: None or bool tensor [T, B, 7]

    Returns:
        entropy, scalar
    """
    if mask is not None:
        # Mask for phases where at least one power had valid entry
        phase_mask = mask.any(dim=2)
        search_policy_orders = _apply_2d_mask(search_policy_orders, phase_mask)
        search_policy_probs = _apply_2d_mask(search_policy_probs, phase_mask)
        mask = _apply_2d_mask(mask, phase_mask)

    search_policy_orders = search_policy_orders.flatten(end_dim=2)
    search_policy_probs = search_policy_probs.flatten(end_dim=2)
    has_actions = (search_policy_orders != EOS_IDX).any(-1).any(-1)
    if mask is not None:
        has_actions = has_actions * mask.flatten(end_dim=2)
    probs = search_policy_probs[has_actions]
    return -(probs * torch.log(probs + 1e-8)).sum(-1).mean()


def evs_to_policy(search_policy_evs, *, temperature=1.0, use_softmax=True):
    """Compute policy targets from EVs.

    Args:
        search_policy_evs: Float tensor [T, B, 7, max_actions]. Invalid values are marked with -1.
        temperature: temperature for softmax. Ignored if softmax is not used.
        use_softmax: whether to apply exp() before normalizing.

    Returns:
       search_policy_probs: Float tensor [T, B, 7, max_actions].
    """
    search_policy_probs = search_policy_evs.clone().float()
    invalid_mask = search_policy_evs < -0.5
    if use_softmax:
        search_policy_probs /= temperature
        # Using -1e8 instead of -inf so that softmax is defined even if all
        # orders are masked out.
        search_policy_probs[invalid_mask] = -1e8
        search_policy_probs = search_policy_probs.softmax(-1)
    else:
        search_policy_probs.masked_fill_(invalid_mask, 0.0)
        search_policy_probs /= (invalid_mask + 1e-20).sum(-1, keepdim=True)
    return search_policy_probs.to(search_policy_evs.dtype)


def perform_retry_loop(f: Callable, max_tries: int, sleep_seconds: int):
    """Retries f repeatedly if it raises RuntimeError or ValueError.

    Stops after success, or max_tries, sleeping sleep_seconds each failure."""
    tries = 0
    success = False
    while not success:
        tries += 1
        try:
            f()
            success = True
        except (RuntimeError, ValueError) as e:
            if tries >= max_tries:
                raise
            time.sleep(sleep_seconds)
