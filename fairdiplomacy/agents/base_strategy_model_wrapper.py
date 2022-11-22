#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import datetime
from functools import reduce
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import nest
import torch
import torch.cuda

from fairdiplomacy import pydipcc
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import select_powers_inplace
from fairdiplomacy.models.consts import LOGIT_MASK_VAL, MAX_SEQ_LEN, N_SCS, POWERS
from fairdiplomacy.models.base_strategy_model.base_strategy_model import NO_ORDER_ID
from fairdiplomacy.models.base_strategy_model.load_model import (
    load_base_strategy_model_model,
    load_base_strategy_model_model_cached,
    SomeBaseStrategyModel,
)
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.typedefs import Action, PlayerRating, Power
from fairdiplomacy.utils.batching import batched_forward
from fairdiplomacy.utils.order_idxs import (
    action_strs_to_global_idxs,
    global_order_idxs_to_local,
    loc_idx_of_order_idx,
)
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.timing_ctx import DummyCtx, TimingCtx


class BaseStrategyModelWrapper:
    """Provides an easier interface for BaseStrategyModel inference"""

    def __init__(
        self,
        model_path,
        device: str = "cuda",
        value_model_path=None,
        max_batch_size=int(1e10),
        *,
        half_precision=False,
        skip_base_strategy_model_cache=False,
        force_disable_all_power=False,
    ):
        if not torch.cuda.is_available():
            logging.warning("Using cpu because cuda not available")
            device = "cpu"

        def load_model(path):
            path = str(path)
            if skip_base_strategy_model_cache:
                return load_base_strategy_model_model(
                    checkpoint_path=path, map_location=device, eval=True
                )
            else:
                return load_base_strategy_model_model_cached(
                    checkpoint_path=path, map_location=device
                )

        self.model_path = model_path
        self.model = load_model(model_path)
        self.value_model = load_model(value_model_path) if value_model_path else self.model
        self.device = device
        self.feature_encoder = FeatureEncoder()
        self.max_batch_size = max_batch_size
        self.half_precision = half_precision
        self.force_disable_all_power = force_disable_all_power
        if half_precision:
            self.model.half()
            if self.value_model is not self.model:
                self.value_model.half()
        self.debug_always_resample_multi_disbands = False

    def get_policy_input_version(self) -> int:
        return self.model.get_input_version()

    def get_value_input_version(self) -> int:
        return self.value_model.get_input_version()

    def forward_policy(
        self,
        games: Sequence[pydipcc.Game],
        *,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
        feature_encoder: Optional[FeatureEncoder] = None,
        temperature: float = -1,
        top_p: float = -1,
        timings: Optional[TimingCtx] = None,
        batch_repeat_interleave: Optional[int] = None,
        conditional_orders: Optional[Sequence[Action]] = None,
        force_conditional_orders: bool = True,
    ) -> Tuple[List[List[Action]], torch.Tensor]:
        """Query the policy on a batch of game positions.

        Arguments:
        games: The game positions to make into a batch.
        has_press: Set to True to condition the BaseStrategyModel on there being press, False means non-press.
        agent_power: Set to a power to condition on agent_power, for base_strategy_models that care about it
        game_rating_dict: Dict mapping game_id to player rating
        feature_encoder: The feature encoder to use. Pass this in if have a preinitialized
          thread-pooling encoder. If not provided, will just use a default single-thread encoder.
        temperature (required): softmax temperature
        top_p (required): top p order sampling proportion
        timings: Pass in to obtain timings of the various operations.
        batch_repeat_interleave: Repeat_interleave the provided games to make a larger batch. See documentation in base_strategy_model.py.
        conditional_orders: If set, the generation will be conditioned on these orders.
        force_conditional_orders: Only matters if conditional_orders is set. If
            set, the conditional order will forced in the decoder as well. It
            does not mean than the sampled orders will always contains the
            conditional orders. But it does mean the other orders were sampled
            as if we sampled the conditional orders. Nobody promised it'll be
            easy...

        Returns: A, logprobs, where A[b][p] is the action chosen for power p in batch item b.
        and logprobs is a tensor of shape [len(batch)] of the log probabilities of the actions chosen.
        """
        timings_ = DummyCtx() if timings is None else timings
        del timings

        with timings_("encoding"):
            feature_encoder = feature_encoder or self.feature_encoder
            if self.is_all_powers():
                batch = feature_encoder.encode_inputs_all_powers(
                    games, self.get_policy_input_version()
                )
            else:
                batch = feature_encoder.encode_inputs(games, self.get_policy_input_version())
            if conditional_orders is not None:
                assert (
                    self.is_all_powers()
                ), "Cannot use conditional_orders with not all-power model"
                if len(games) == 1:
                    batch.repeat_batch_(len(conditional_orders))
                    games = list(games) * len(conditional_orders)

                assert len(conditional_orders) == len(games), (
                    len(conditional_orders),
                    len(games),
                )
                batch["x_current_orders"] = torch.stack(
                    [
                        feature_encoder.encode_orders_single_tolerant(
                            game, action, self.get_policy_input_version()
                        )
                        for (game, action) in zip(games, conditional_orders)
                    ],
                    0,
                )
                if force_conditional_orders:
                    batch["teacher_force_orders"] = create_conditional_teacher_force_orders(batch)
                    if batch_repeat_interleave is not None:
                        batch["teacher_force_orders"] = batch[
                            "teacher_force_orders"
                        ].repeat_interleave(batch_repeat_interleave, dim=0)

            batch = self.add_stuff_to_datafields(
                batch,
                has_press=has_press,
                agent_power=agent_power,
                game_rating_dict=game_rating_dict,
            )

        return self.forward_policy_from_datafields(
            batch,
            temperature=temperature,
            top_p=top_p,
            timings=timings_,
            batch_repeat_interleave=batch_repeat_interleave,
        )

    def create_datafield_for_values(
        self,
        games: Sequence[pydipcc.Game],
        *,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
        feature_encoder: Optional[FeatureEncoder] = None,
        timings: Optional[TimingCtx] = None,
    ) -> DataFields:
        """Featurize inputs to be passed to forward_values_from_datafields.

        See forward_values for docs.
        """
        timings_ = DummyCtx() if timings is None else timings
        del timings

        with timings_("encoding"):
            feature_encoder = feature_encoder or self.feature_encoder
            batch = feature_encoder.encode_inputs_state_only(games, self.get_value_input_version())
            batch = batch.to(self.device)
            batch = self.add_stuff_to_datafields(
                batch,
                has_press=has_press,
                agent_power=agent_power,
                game_rating_dict=game_rating_dict,
            )
        return batch

    @classmethod
    def add_stuff_to_datafields(
        cls,
        x: DataFields,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
    ) -> DataFields:
        batch = x
        if has_press:
            _set_press_flags(batch)
        if agent_power is not None:
            _set_agent_power_flags(batch, agent_power)
        if game_rating_dict is not None:
            _set_game_ratings(batch, game_rating_dict)
        return batch

    def forward_values(
        self,
        games: Sequence[pydipcc.Game],
        *,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
        feature_encoder: Optional[FeatureEncoder] = None,
        timings: Optional[TimingCtx] = None,
    ) -> torch.Tensor:
        """Query the value on a batch of game positions.

        Arguments:
        games: The game positions to make into a batch.
        has_press: Set to True to condition the BaseStrategyModel on there being press, False means non-press.
        agent_power: Set to a power to condition on agent_power, for base_strategy_models that care about it
        game_rating_dict: Dict mapping game_id to player rating
        feature_encoder: The feature encoder to use. Pass this in if have a preinitialized
          thread-pooling encoder. If not provided, will just use a default single-thread encoder.
        timings: Pass in to obtain timings of the various operations.

        Returns: values, where values[b][p] is the predicted value for power p in batch item b.
        """
        return self.forward_values_from_datafields(
            self.create_datafield_for_values(
                games,
                has_press=has_press,
                agent_power=agent_power,
                game_rating_dict=game_rating_dict,
                feature_encoder=feature_encoder,
                timings=timings,
            ),
            timings=timings,
        )

    def forward_policy_from_datafields(
        self,
        x: DataFields,
        temperature: float = -1,
        top_p: float = -1,
        batch_repeat_interleave: Optional[int] = None,
        timings: Optional[Union[DummyCtx, TimingCtx]] = None,
    ) -> Tuple[List[List[Action]], torch.Tensor]:
        """Same as forward_policy, but with precomputed DataFields.

        Generally callers should prefer forward_policy instead.
        """
        timings_ = DummyCtx() if timings is None else timings
        del timings

        with timings_("to_half_precision"):
            if self.half_precision:
                x = x.to_half_precision()

        assert temperature > 0
        assert top_p > 0
        with timings_("model"):
            order_idxs, order_logprobs = batched_forward(
                lambda x: self._forward_policy(
                    x,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                ),
                x,
                # Cannot mix batch_repeat_interleave and auto batching.
                batch_size=self.max_batch_size if batch_repeat_interleave is None else int(1e10),
                device=self.device,
            )

        with timings_("model.decode"):
            if not self.is_all_powers():
                decoded = self.feature_encoder.decode_order_idxs(order_idxs)
            else:
                # As documented in base_strategy_model.py, batch_repeat_interleave=N means we compute outputs as
                # if each input element was repeated N times, without explicitly repeating them.
                # So the output "decoded" has an N times larger batch size, but x_in_adj_phase_batched
                # and x_power_batched do not, since they were inputs. So when indexing them, we divide
                # to find out the proper index.
                x_in_adj_phase_batched = x["x_in_adj_phase"].cpu()
                x_power_batched = x["x_power"].cpu()
                div = 1 if batch_repeat_interleave is None else batch_repeat_interleave
                assert isinstance(order_idxs, torch.Tensor)
                decoded = self.feature_encoder.decode_order_idxs_all_powers(
                    order_idxs, x_in_adj_phase_batched, x_power_batched, div
                )

        with timings_("model.decode.tuple"):
            # NOTE: use of tuples here is a bit inconsistent with what searchbot
            # agent does, as well as BaseAgent interface, which expect lists
            # instead.
            decoded = [[tuple(orders) for orders in powers_orders] for powers_orders in decoded]

        return (decoded, cast(torch.Tensor, order_logprobs))

    def forward_values_from_datafields(
        self, x: DataFields, timings: Optional[TimingCtx] = None
    ) -> torch.Tensor:
        """Same as forward_values, but with precomputed DataFields.

        Generally callers should prefer forward_values instead.
        """
        timings_ = DummyCtx() if timings is None else timings
        del timings

        with timings_("to_half_precision"):
            if self.half_precision:
                x = x.to_half_precision()

        with timings_("model"):
            return cast(
                torch.Tensor,
                batched_forward(
                    self._forward_values, x, batch_size=self.max_batch_size, device=self.device
                ),
            )

    def _forward_values(self, x: DataFields):
        x = x.copy()
        x.update(x_loc_idxs=None, x_possible_actions=None, temperature=None, top_p=None)
        _, _, _, values = self.value_model(**x, need_policy=False)
        return values

    def _forward_policy(
        self,
        x: DataFields,
        temperature: float,
        top_p: float,
        batch_repeat_interleave: Optional[int],
    ):
        order_idxs, order_logprobs, _ = forward_model_with_output_transform(
            self.model,
            x,
            batch_repeat_interleave=batch_repeat_interleave,
            temperature=temperature,
            top_p=top_p,
            debug_always_resample_multi_disbands=self.debug_always_resample_multi_disbands,
            need_value=False,
            pad_to_max=True,
        )
        return order_idxs, order_logprobs

    def get_values(
        self, game: pydipcc.Game, *, has_press: bool, agent_power: Optional[Power]
    ) -> torch.Tensor:
        return self.forward_values([game], has_press=has_press, agent_power=agent_power)[0]

    def is_all_powers(self) -> bool:
        return not self.force_disable_all_power and self.model.is_all_powers()


def _set_press_flags(batch: DataFields):
    batch["x_has_press"] = batch["x_board_state"].new_ones(batch["x_board_state"].shape[0], 1)


def _set_agent_power_flags(batch: DataFields, agent_power: Power):
    batch["x_agent_power"] = batch["x_board_state"].new_zeros(
        batch["x_board_state"].shape[0], len(POWERS)
    )
    batch["x_agent_power"][:, POWERS.index(agent_power)] = 1.0


def _set_game_ratings(batch: DataFields, game_rating_dict):
    """
    Populates inputs["x_player_ratings"] with the game's player rating
    Arguments:
    ----------
    - batch: Datafield
    - game_rating_dict: Game id -> player Rating dict
    """
    assert len(game_rating_dict) == batch["x_board_state"].size(0)
    batch["x_player_ratings"] = (
        torch.tensor(list(game_rating_dict.values())).unsqueeze(1).repeat(1, len(POWERS))
    )


def forward_model_with_output_transform(
    model,
    x: DataFields,
    *,
    batch_repeat_interleave=None,
    debug_always_resample_multi_disbands=False,
    **kwargs,
):
    y = model(**x, batch_repeat_interleave=batch_repeat_interleave, **kwargs)
    global_order_idxs, local_order_idxs, logits, final_sos = y

    # print("global_order_idxs", global_order_idxs)
    # print("logprobs", compute_action_logprobs(local_order_idxs, logits))
    resample_duplicate_disbands_inplace(
        global_order_idxs,
        local_order_idxs,
        logits,
        inputs=x,
        model=model,
        batch_repeat_interleave=batch_repeat_interleave,
        debug_always_resample_multi_disbands=debug_always_resample_multi_disbands,
        **kwargs,
    )
    # print("resampled global_order_idxs", global_order_idxs)
    # print("resampled logprobs", compute_action_logprobs(local_order_idxs, logits))

    return global_order_idxs, compute_action_logprobs(local_order_idxs, logits), final_sos


def compute_action_logprobs(local_order_idxs, logits, temperature=None):
    local_order_idxs = local_order_idxs[:, :, : logits.shape[-2]]  # trim off excess seq dim
    invalid_mask = local_order_idxs < 0
    local_order_idxs = local_order_idxs.clamp(
        min=0
    )  # otherwise gather(-1) will blow up. We'll mask these out later
    if temperature is not None:
        logits = logits / temperature
    logprobs = logits.log_softmax(-1)

    sampled_logprobs = logprobs.gather(-1, local_order_idxs.unsqueeze(-1)).squeeze(-1)
    sampled_logprobs[invalid_mask] = 0
    total_logprobs = sampled_logprobs.sum(-1)
    return total_logprobs


def resample_duplicate_disbands_inplace(
    global_order_idxs,
    local_order_idxs,
    logits,
    *,
    inputs: DataFields,
    model: SomeBaseStrategyModel,
    batch_repeat_interleave=None,
    debug_always_resample_multi_disbands=False,
    **kwargs,
):
    """Modify global_order_idxs and local_order_idxs in-place, resampling where there are
    duplicate disband orders, OR out-of-order disbands since that will give
    rise to inconsistent log probabilities for the same action once we sort the action
    in later code.

    Will use model to re-compute what the model thinks the probabilities
    of the resampled disbands should be.
    """
    # Resample all multiple disbands. Since builds are a 1-step decode, any 2+
    # step adj-phase is a disband.
    if local_order_idxs.shape[2] < 2:
        return
    x_possible_actions = inputs["x_possible_actions"]
    x_in_adj_phase = inputs["x_in_adj_phase"]

    if batch_repeat_interleave is not None:
        assert x_in_adj_phase.size()[0] * batch_repeat_interleave == local_order_idxs.size()[0]
        x_in_adj_phase = x_in_adj_phase.repeat_interleave(batch_repeat_interleave, dim=0)
    multi_disband_powers_mask = (
        local_order_idxs[:, :, 1] != EOS_IDX
    ) & x_in_adj_phase.bool().unsqueeze(1)
    if not multi_disband_powers_mask.any():
        return

    # More rigorous check that restricts to cases where it looks like we actually have
    # a duplicated disband. If there are 4 or more disbands, we don't bother and we
    # go ahead and just always resample.
    if not debug_always_resample_multi_disbands:
        # Maps global_order_idx+1 -> src location idx of that order
        # Maps EOS_IDX + 1 -> an idx larger than any location
        mapping = model.get_srcloc_idx_of_global_order_idx_plus_one()

        # Apply the mapping to the first 3 orders in global_order_idxs to get an array saying
        # what location every one of the first 3 disbands is disbanding.
        first_3_disbands = global_order_idxs[:, :, :3]
        srcloc_idxs = torch.gather(mapping, 0, torch.flatten(first_3_disbands + 1)).view(
            *first_3_disbands.shape
        )

        # No need to handle EOS_IDX check since we already know local_order_idxs[:, :, 1] != EOS_IDX
        # We compare to see if the 0th disband is out of order to the 1th disband.
        when_to_resample = srcloc_idxs[:, :, 0] >= srcloc_idxs[:, :, 1]
        if srcloc_idxs.shape[2] >= 3:
            # We compare to see if the 1th disband is out of order to the 2th disband.
            when_to_resample |= (torch.sum(local_order_idxs != EOS_IDX, dim=2) >= 4) | (
                srcloc_idxs[:, :, 1] >= srcloc_idxs[:, :, 2]
            )
        multi_disband_powers_mask &= when_to_resample
        if not multi_disband_powers_mask.any():
            return

    # N.B. we may sample more orders than we need here: we are sampling
    # according to the longest sequence in the batch, not the longest
    # multi-disband sequence. Still, even if there is a 3-unit disband and a
    # 2-unit disband, we will sample the same # for both and mask out the extra
    # orders (see the eos_mask below)
    #
    # Note 1: The longest sequence in the batch may be longer than
    # the # of disband order candidates, so we take the min with
    # logits.shape[3] (#candidates)
    #
    # Note 2: 1e-10 is Laplace smoothing coefficient to make sampling without
    # replacing work for spiky logits which result in 1-hot probs
    #
    # Note 3: Ensure that masked (large negative) logits don't get mixed up
    # with low-prob (but valid) actions due to the smoothing coefficient and
    # the limits of float math.
    try:
        saved_logits_mask = logits[multi_disband_powers_mask][:, 0] == LOGIT_MASK_VAL
        probs = logits[multi_disband_powers_mask][:, 0].softmax(-1) + 1e-10  # See Note 2
        probs[saved_logits_mask] = 1e-34  # See Note 3
        new_local_order_idxs = torch.multinomial(
            probs, min(logits.shape[2], logits.shape[3]), replacement=False  # See Note 1
        )
    except RuntimeError:
        torch.save(
            {
                "global_order_idxs": global_order_idxs,
                "local_order_idxs": local_order_idxs,
                "logits": logits,
                "x_possible_actions": x_possible_actions,
                "x_in_adj_phase": x_in_adj_phase,
                "batch_repeat_interleave": batch_repeat_interleave,
            },
            "resample_duplicate_disbands_inplace.debug.pt",
        )
        raise

    filler = torch.empty(
        new_local_order_idxs.shape[0],
        local_order_idxs.shape[2] - new_local_order_idxs.shape[1],
        dtype=new_local_order_idxs.dtype,
        device=new_local_order_idxs.device,
    ).fill_(-1)
    eos_mask = local_order_idxs == EOS_IDX

    if batch_repeat_interleave is not None:
        assert x_possible_actions.size()[0] * batch_repeat_interleave == local_order_idxs.size()[0]
        x_possible_actions = x_possible_actions.repeat_interleave(batch_repeat_interleave, dim=0)

    new_global_order_idxs = (
        x_possible_actions[multi_disband_powers_mask][:, 0].long().gather(1, new_local_order_idxs)
    )
    num_disbands = torch.sum(local_order_idxs[multi_disband_powers_mask] != EOS_IDX, dim=1)
    assert len(num_disbands.shape) == 1

    # Sort our sampled disbands so that they are sorted by coast-qualified LOC ordering
    # since that's the universal order that everwhere else in our codebase relies on
    # (e.g. that's the order that a model will expect when re-scoring an action)
    local_idxs_to_sort = new_local_order_idxs.cpu().tolist()
    global_idxs_to_sort = new_global_order_idxs.cpu().tolist()
    num_disbands_to_sort = num_disbands.cpu().tolist()
    assert len(local_idxs_to_sort) == len(global_idxs_to_sort)
    assert len(num_disbands_to_sort) == len(global_idxs_to_sort)
    for i in range(len(global_idxs_to_sort)):
        ndisbands = num_disbands_to_sort[i]
        permutation = list(range(ndisbands))
        permutation.sort(key=(lambda j: loc_idx_of_order_idx(global_idxs_to_sort[i][j])))
        local_idxs_to_sort_tmp = local_idxs_to_sort[i][:ndisbands]  # copy so that we don't mutate
        global_idxs_to_sort_tmp = global_idxs_to_sort[i][
            :ndisbands
        ]  # copy so that we don't mutate

        # torch.set_printoptions(sci_mode=False)
        # print("I", i)
        # print("LOCAL", local_idxs_to_sort[i])
        # print("GLOBAL", global_idxs_to_sort[i])
        # print("NUM", num_disbands_to_sort[i])
        # print("PERM",permutation)
        for j in range(ndisbands):
            local_idxs_to_sort[i][j] = local_idxs_to_sort_tmp[permutation[j]]
            global_idxs_to_sort[i][j] = global_idxs_to_sort_tmp[permutation[j]]

    new_local_order_idxs[:] = new_local_order_idxs.new_tensor(local_idxs_to_sort)
    new_global_order_idxs[:] = new_global_order_idxs.new_tensor(global_idxs_to_sort)

    local_order_idxs[multi_disband_powers_mask] = torch.cat([new_local_order_idxs, filler], dim=1)
    global_order_idxs[multi_disband_powers_mask] = torch.cat(
        [new_global_order_idxs, filler,], dim=1,
    )
    local_order_idxs[eos_mask] = EOS_IDX
    global_order_idxs[eos_mask] = EOS_IDX

    if model is None:
        logits[multi_disband_powers_mask] = logits[multi_disband_powers_mask][:, 0].unsqueeze(1)
    else:
        # Since we resampled some duplicate disbands, to get probabilities for
        # stuff like plausible order sampling, piKL, etc., requery model for
        # the real values.
        _, _, new_logits, _ = model(
            **inputs,
            teacher_force_orders=global_order_idxs.detach().clone().clamp(min=0),
            batch_repeat_interleave=batch_repeat_interleave,
            **kwargs,
        )
        logits[multi_disband_powers_mask] = new_logits[multi_disband_powers_mask]

        # torch.set_printoptions(sci_mode=False,threshold=10000,edgeitems=20)
        # print("NEWGLOBALS", global_order_idxs[multi_disband_powers_mask][:,:6])
        # print("NEWLOCALS", local_order_idxs[multi_disband_powers_mask][:,:6])
        # print("NEWLOGITS", logits[multi_disband_powers_mask][:,:3,:6])
        # print("local_order_idxs",local_order_idxs[0:10,1,0:3])
        # print("logits",logits[0:10,1,0:3])
        # print("probs", probs[:].cpu().numpy())


def compute_action_logprobs_from_state(
    base_strategy_model: SomeBaseStrategyModel,
    game: pydipcc.Game,
    power_action_dicts: List[Dict[Power, Action]],
    *,
    has_press: bool,
    agent_power: Optional[Power],
    game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
    batch_size: int,
    debug_only_no_skip_select_inplace: bool = False,
    temperature: Optional[float] = None,
) -> List[float]:
    """Computes log probabilities of actions from a current game state.
    power_action_dicts should be a list of all the actions
    that we want to compute logprobs for. Each dict should be either length 1
    (in the case of normal single-power actions), length 2 (in the case
    of all powers bilateral actions decoding) or length 7 (in the case of
    all-powers joint actions decoding).

    Orders do NOT need to be the same order as base_strategy_model decodes, they will be
    sorted internally before processing. This means that arbitrarly human-typed
    sequences of orders or orders from an outside-sourced game record should work
    so long as they exactly match our standard order syntax.

    For unparseable orders or orders not in our vocab (e.g. long convoys)
    this function will raise an exception.

    """
    half_precision = next(iter(base_strategy_model.parameters())).dtype is torch.float16

    batched_inputs = []
    power_action_dicts_idx_for_input = []

    is_all_powers = base_strategy_model.is_all_powers()
    if is_all_powers:
        observations = FeatureEncoder().encode_inputs_all_powers(
            [game], input_version=base_strategy_model.get_input_version()
        )
    else:
        observations = FeatureEncoder().encode_inputs(
            [game], input_version=base_strategy_model.get_input_version()
        )
    possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]

    supports_single_power_decoding = base_strategy_model.supports_single_power_decoding()
    supports_double_power_decoding = base_strategy_model.supports_double_power_decoding()
    is_adjustment_phase = game.get_current_phase().endswith("A")

    # All powers expects sequence decoding even longer than MAX_SEQ_LEN
    # since MAX_SEQ_LEN only caps the sequence length for a *single* power.
    max_sequence_length = N_SCS if is_all_powers else MAX_SEQ_LEN

    for power_action_dicts_idx, power_action_dict in enumerate(power_action_dicts):
        # On adjustment phases, encode everything separately even for all powers model.
        # We don't model the joint distribution for build and disband.
        if not is_all_powers or is_adjustment_phase:
            for power, action in power_action_dict.items():
                power_id = POWERS.index(power)
                inputs = observations.copy()

                inputs["teacher_force_orders"] = torch.full(
                    (1, max_sequence_length), EOS_IDX, dtype=torch.long
                )
                # Have to convert first and the use its size to handle joined build orders.
                t = torch.as_tensor(
                    action_strs_to_global_idxs(
                        action, match_to_possible_orders=possible_orders, sort_by_loc=True
                    )
                )
                inputs["teacher_force_orders"][0, : len(t)] = t
                inputs["x_power"] = torch.full((1, max_sequence_length), power_id)
                inputs["x_loc_idxs"] = inputs["x_loc_idxs"][:, power_id]
                inputs["x_possible_actions"] = inputs["x_possible_actions"][:, power_id]

                assert len(t) <= max_sequence_length
                assert (
                    len(t) == max_sequence_length
                    or inputs["x_possible_actions"][0, len(t), 0] == EOS_IDX
                ), f"compute_action_logprobs_from_state received partial action {power} {action}"

                power_action_dicts_idx_for_input.append(power_action_dicts_idx)
                batched_inputs.append(inputs)
        else:
            npowers_in_action = len(power_action_dict)
            if npowers_in_action == 1:
                assert supports_single_power_decoding
            elif npowers_in_action == 2:
                assert supports_double_power_decoding
            elif npowers_in_action == 6:
                pass
            elif npowers_in_action == len(POWERS):
                pass
            else:
                assert (
                    False
                ), f"compute_action_logprobs_from_state got action with {npowers_in_action} powers but model does not support this"

            inputs = observations.copy()

            # This is allpower features, so everything except values for the "first power" is EOS_IDX.
            inputs["x_power"] = inputs["x_power"][:, 0]
            inputs["x_possible_actions"] = inputs["x_possible_actions"][:, 0]
            inputs["x_loc_idxs"] = inputs["x_loc_idxs"][:, 0]
            # print("BEFORE x_power", inputs["x_power"])
            # print("BEFORE x_possible_actions", inputs["x_possible_actions"])
            # print("BEFORE x_loc_idxs", inputs["x_loc_idxs"])
            # We need to modify these to include only selected powers.
            if debug_only_no_skip_select_inplace or npowers_in_action < len(POWERS):
                for name in ["x_power", "x_possible_actions", "x_loc_idxs"]:
                    inputs[name] = inputs[name].clone()
                select_powers_inplace(
                    torch.LongTensor([POWERS.index(p) for p in power_action_dict]),
                    inputs["x_power"][0],
                    inputs["x_possible_actions"][0],
                    inputs["x_loc_idxs"][0],
                )
            # print("AFTER x_power", inputs["x_power"])
            # print("AFTER x_possible_actions", inputs["x_possible_actions"])
            # print("AFTER x_loc_idxs", inputs["x_loc_idxs"])

            # Concat all orders together, convert and sort
            concatted_action = reduce(
                (lambda actiona, actionb: actiona + actionb), power_action_dict.values()
            )
            t = torch.as_tensor(
                action_strs_to_global_idxs(
                    concatted_action, match_to_possible_orders=possible_orders, sort_by_loc=True
                )
            )
            inputs["teacher_force_orders"] = torch.full(
                (1, max_sequence_length), EOS_IDX, dtype=torch.long
            )
            inputs["teacher_force_orders"][0, : len(t)] = t

            assert len(t) <= max_sequence_length
            assert (
                len(t) == max_sequence_length
                or inputs["x_possible_actions"][0, len(t), 0] == EOS_IDX
            ), "compute_action_logprobs_from_state received partial action"

            power_action_dicts_idx_for_input.append(power_action_dicts_idx)
            batched_inputs.append(inputs)

    def f(batch):
        base_strategy_model_batch = DataFields(**batch)
        base_strategy_model_batch["teacher_force_orders"] = batch["teacher_force_orders"].clamp(
            min=0
        )
        base_strategy_model_batch = BaseStrategyModelWrapper.add_stuff_to_datafields(
            base_strategy_model_batch,
            has_press=has_press,
            agent_power=agent_power,
            game_rating_dict=game_rating_dict,
        )

        if half_precision:
            base_strategy_model_batch = base_strategy_model_batch.to_half_precision()
        _, _, logits, _ = base_strategy_model(
            **base_strategy_model_batch, need_value=False, temperature=1.0
        )
        # print("RESCOREACTIONS", batch["teacher_force_orders"][:,:6])
        # print("RESCORELOGITS", logits[:,:3,:6])

        try:
            local_indices = global_order_idxs_to_local(
                batch["teacher_force_orders"], batch["x_possible_actions"]
            )
        except Exception:
            local_indices = global_order_idxs_to_local(
                batch["teacher_force_orders"], batch["x_possible_actions"], ignore_missing=True
            )
            shuttle = dict(game=game.to_json(), power_action_dicts=power_action_dicts, batch=batch)
            filename = "debug_global_order.%s.pt" % datetime.datetime.now()
            torch.save(shuttle, filename)
            assert (
                False
            ), f"Failed to convert order idxes for compute_action_logprobs_from_state, see {filename}"

        # Hack to make compute_action_logprobs work. Adding POWER dimension.
        logprobs = compute_action_logprobs(
            local_indices.unsqueeze(1), logits.unsqueeze(1), temperature=temperature
        )
        # Remove fake power.
        logprobs = logprobs.squeeze(1)
        return logprobs

    joined_input = nest.map_many(lambda x: torch.cat(x, 0), *batched_inputs)
    device = next(iter(base_strategy_model.parameters())).device
    logprobs = batched_forward(f, joined_input, batch_size=batch_size, device=device)
    logprobs = cast(torch.Tensor, logprobs)
    assert len(logprobs.shape) == 1
    logprobs = logprobs.cpu().tolist()

    # Some logprobs got fragmented into multiple queries
    # Sum them up into the original queries the user wanted.
    # possibly_joint_actions_idx_for_input indicates the original index of the query.
    assert len(logprobs) == len(power_action_dicts_idx_for_input)
    final_logprobs = [0.0 for _ in range(len(power_action_dicts))]
    for i, logprob in enumerate(logprobs):
        idx = power_action_dicts_idx_for_input[i]
        final_logprobs[idx] += logprob

    return final_logprobs


def create_conditional_teacher_force_orders(batch: DataFields) -> torch.Tensor:
    teacher_force_orders = batch["x_prev_orders"].new_full(
        batch["x_possible_actions"].shape[:-1], NO_ORDER_ID
    )
    for batch_idx, encoded_orders in enumerate(batch["x_current_orders"]):
        assert (
            batch["x_loc_idxs"][batch_idx] >= 0
        ).any(), "Cannot use order conditioning in non-move phases"
        for order_id, loc_id in encoded_orders.transpose(0, 1).tolist():
            if order_id != 0:
                power_idx = 0  # It's all power.
                local_loc_id = batch["x_loc_idxs"][
                    batch_idx, power_idx, loc_id
                ]  # Loc id (global) -> local loc idx.
                teacher_force_orders[batch_idx, power_idx, local_loc_id] = order_id
    return teacher_force_orders
