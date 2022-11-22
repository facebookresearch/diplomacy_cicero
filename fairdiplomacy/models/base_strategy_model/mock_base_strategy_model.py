#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Infrence only mock class to emulate forward on base_strategy_model model.

MockBaseStrategyModel class defines __call__ methods that have inputs and onputs
identical to real BaseStrategyModel. The outputs are computed using simple heuristics.

For value function returns SoS.

For policy we sample orders for each location independently.  Order at
position `t` will be samples with probability 2 times higher than order at
position `t + 1`. This is to encourage more often sampling of "simple" move order.

Temperature parameter is also respected is applied on top of the distribution
described above.


There are 2 ways to use MockBaseStrategyModel:
  * directly initialize the class,
  * use `base_strategy_model.load_model` with "MOCK" or "MOCKV1" or "MOCKV2" as the model path -
    this allows to use the mock inside agents.
"""

import logging
import math

from typing import FrozenSet, List, Optional, Sequence, Tuple
from fairdiplomacy.models.consts import POWERS, LOCS, LOGIT_MASK_VAL, MAX_SEQ_LEN
from fairdiplomacy.models.base_strategy_model.base_strategy_model import (
    compute_srcloc_idx_of_global_order_idx_plus_one,
)
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.utils.order_idxs import LOC_IDX_OF_ORDER_IDX, local_order_idxs_to_global
from fairdiplomacy.utils.thread_pool_encoding import get_board_state_size
from fairdiplomacy import pydipcc
import torch
import torch.nn.functional


POLICY_SAMPLING_EXPONENT: float = 2.0


# Mapping from power id to a feature id related to owning an SC.
def get_power_id_to_sc_ownership_idx(input_version: int) -> Sequence[int]:
    return pydipcc.encoding_sc_ownership_idxs(input_version)


class MockBaseStrategyModel:
    def __init__(
        self, input_version=1, all_powers=False, fixed_value_output: Optional[List[float]] = None
    ):
        self.input_version = input_version
        self.all_powers = all_powers
        self.use_player_ratings = True
        self.fixed_value_output = fixed_value_output
        self.half = False
        self.use_agent_power = False

        # Register a buffer that maps global order index to source location
        # of that order
        srcloc_idx_of_global_order_idx_plus_one = compute_srcloc_idx_of_global_order_idx_plus_one()
        self.srcloc_idx_of_global_order_idx_plus_one = srcloc_idx_of_global_order_idx_plus_one

    def eval(self) -> None:
        pass

    def cuda(self) -> None:
        pass

    def get_training_permute_powers(self) -> bool:
        return False

    def set_training_permute_powers(self, b: bool) -> None:
        del b

    def parameters(self):
        # Defining this so that device-detection code works.
        logging.warning(
            "MockBaseStrategyModel.parameters() is called. Will return a single CPU tensor"
        )
        return [torch.zeros(1)]

    def get_input_version(self) -> int:
        return self.input_version

    def is_all_powers(self) -> bool:
        return self.all_powers

    def supports_single_power_decoding(self) -> bool:
        return not self.all_powers

    def supports_double_power_decoding(self) -> bool:
        return False

    def get_srcloc_idx_of_global_order_idx_plus_one(self) -> torch.Tensor:
        """Return a tensor mapping global order idx -> location idx of src of order"""
        return self.srcloc_idx_of_global_order_idx_plus_one

    def __call__(
        self,
        *,
        x_board_state,
        x_prev_state,
        x_prev_orders,
        x_season,
        x_year_encoded,
        x_in_adj_phase,
        x_build_numbers,
        x_loc_idxs,
        x_possible_actions,
        temperature,
        top_p=1.0,
        batch_repeat_interleave=None,
        teacher_force_orders=None,
        x_power=None,
        x_has_press=None,
        x_player_ratings=None,
        x_scoring_system=None,
        x_agent_power=None,
        x_current_orders=None,
        need_policy=True,
        need_value=True,
        pad_to_max=False,
        encoded=None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        assert x_power is None, "Using x_power is not supported for the mock agent"

        del x_prev_state
        del x_prev_orders
        del x_season
        del x_year_encoded
        del x_in_adj_phase
        del x_build_numbers
        del x_loc_idxs
        del top_p
        del x_has_press
        del x_player_ratings
        del x_scoring_system
        del x_agent_power
        del x_current_orders
        del teacher_force_orders
        del encoded

        batch_size = len(x_board_state)
        board_state_size = get_board_state_size(self.input_version)
        assert list(x_board_state.shape) == [batch_size, len(LOCS), board_state_size], (
            [batch_size, len(LOCS), board_state_size],
            x_board_state.shape,
        )

        batch_repeat_interleave = batch_repeat_interleave or 1
        device = x_board_state.device

        if need_policy:
            x_possible_actions = torch.repeat_interleave(
                x_possible_actions, batch_repeat_interleave, dim=0
            )
            logits, local_order_idxs = sample_action_batched(x_possible_actions, temperature,)
            logits = logits.to(device)
            local_order_idxs = local_order_idxs.to(device)
            global_order_idxs = local_order_idxs_to_global(
                local_order_idxs, x_possible_actions
            ).long()
            if pad_to_max:
                max_seq_len = MAX_SEQ_LEN
                global_order_idxs = _pad_last_dims(global_order_idxs, [max_seq_len], EOS_IDX)
                local_order_idxs = _pad_last_dims(local_order_idxs, [max_seq_len], EOS_IDX)
                logits = _pad_last_dims(logits, [max_seq_len, 469], LOGIT_MASK_VAL)
        else:
            logits = local_order_idxs = global_order_idxs = None
        if need_value:
            final_sos = _compute_power_values_batched(
                x_board_state, self.input_version, self.fixed_value_output
            ).to(device)
            final_sos = torch.repeat_interleave(final_sos, batch_repeat_interleave, dim=0)
        else:
            final_sos = None

        return global_order_idxs, local_order_idxs, logits, final_sos


def _compute_power_values_single(
    board_state: torch.Tensor, input_version, fixed_value_output: Optional[List[float]]
) -> torch.Tensor:
    """Computes SoS values from board state: maps [81, N_FEATS] to [N_POWERS]."""
    if fixed_value_output is not None:
        return torch.tensor(fixed_value_output)

    power_id_to_feat = get_power_id_to_sc_ownership_idx(input_version)

    scores = [0.0] * len(POWERS)
    for loc, feat_tensor in zip(LOCS, board_state):
        if "/" in loc:
            # We double count coasts.
            continue
        # print(loc, parse_board_features(feat_tensor))
        for power_id, feat_id in enumerate(power_id_to_feat):

            if feat_tensor[int(feat_id)]:
                scores[power_id] += 1
                # print(loc, POWERS[power_id])
                break
    scores = torch.tensor(scores)
    # print(*zip(POWERS, scores))
    if (scores > 17).any():
        scores = (scores > 17).float()
    scores = scores ** 2
    scores /= scores.sum()
    # print(*zip(POWERS, scores))
    return scores


def _compute_power_values_batched(
    board_state: torch.Tensor, input_version, fixed_value_output: Optional[List[float]]
) -> torch.Tensor:
    """Computes SoS values from board state: maps [..., 81, N_FEATS] to [..., N_POWERS]."""
    *batch_shape, _locs, _feats = board_state.shape
    return torch.stack(
        [
            _compute_power_values_single(x, input_version, fixed_value_output)
            for x in board_state.flatten(end_dim=-3)
        ],
        0,
    ).view(*batch_shape, len(POWERS))


def sample_action_batched(
    possible_actions: torch.Tensor, temperature
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a possible actions using exponenitial distribution poer power.

    Input shape is [..., MAX_SEQ_LEN, 469].

    Input will be truncated by the output length, i.e., it will only contain
    logits/orders for positions where orders are possible.

    Returns a tuple (local_order_indices [], logprobs).
    """
    # Probability to sample order `t` is 2x more than `t + 1`.
    exp_step = math.log(2)
    # Shape: [469].
    full_logits = -torch.arange(possible_actions.shape[-1]) * exp_step

    max_real_seq_len = (possible_actions != EOS_IDX).any(-1).flatten(end_dim=-2).any(0).sum()
    possible_actions = possible_actions.transpose(0, -2)[:max_real_seq_len].transpose(0, -2)

    *batch_dims, softmax_size = possible_actions.shape

    # Shape: [mega_batch, 469].
    flat_possible_actions = possible_actions.flatten(end_dim=-2)

    flat_logits = full_logits[None].repeat((len(flat_possible_actions), 1))
    flat_logits[flat_possible_actions == EOS_IDX] = LOGIT_MASK_VAL
    flat_orders = torch.full([len(flat_possible_actions)], -1, dtype=torch.long)
    has_orders = (flat_possible_actions != EOS_IDX).any(-1)
    flat_orders[has_orders] = torch.multinomial(
        torch.nn.functional.softmax(flat_logits[has_orders] / temperature, -1), 1
    ).squeeze(-1)

    logits = flat_logits.view(*batch_dims, softmax_size)
    orders = flat_orders.view(*batch_dims)

    return logits, orders


def _pad_last_dims(tensor, partial_new_shape, pad_value):
    assert len(tensor.shape) >= len(partial_new_shape), (tensor.shape, partial_new_shape)
    new_shape = list(tensor.shape)[: len(tensor.shape) - len(partial_new_shape)] + list(
        partial_new_shape
    )
    new_tensor = tensor.new_full(new_shape, pad_value)
    new_tensor[[slice(None, D) for D in tensor.shape]].copy_(tensor)
    return new_tensor
