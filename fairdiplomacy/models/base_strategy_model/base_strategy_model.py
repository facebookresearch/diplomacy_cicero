#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from enum import Enum
from typing import TYPE_CHECKING, Tuple, Optional, Union
import inspect
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

import conf.conf_cfgs
from fairdiplomacy.models.consts import POWERS, LOCS, LOGIT_MASK_VAL, MAX_SEQ_LEN, N_SCS
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.padded_embedding import PaddedEmbedding
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.order_idxs import LOC_IDX_OF_ORDER_IDX, local_order_idxs_to_global
from fairdiplomacy.utils.thread_pool_encoding import get_board_state_size
from fairdiplomacy.models.state_space import (
    get_order_vocabulary,
    EOS_IDX,
)
from fairdiplomacy.models.base_strategy_model.decoders import TransfDecoder
from fairdiplomacy.models.base_strategy_model.util import he_init, top_p_filtering
from fairdiplomacy import pydipcc

EOS_TOKEN = get_order_vocabulary()[EOS_IDX]
# If teacher forcing orders have this id, then a sampled order will be used for
# this position.
NO_ORDER_ID = -2

# Indices for encoding scoring systems for x_scoring_system
class Scoring(Enum):
    SOS = 0  # sum of squares
    DSS = 1  # draw size scoring


class BaseStrategyModelV2(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,  # 120
        board_map_size,  # number of diplomacy map locations, i.e. 81
        order_emb_size,  # 80
        prev_order_emb_size,  # 20
        orders_vocab_size,  # 13k
        lstm_size,  # 200
        lstm_dropout=0,
        lstm_layers=1,
        value_dropout,
        value_decoder_init_scale=1.0,
        value_decoder_activation="relu",
        value_decoder_use_weighted_pool: bool,
        value_decoder_extract_from_encoder: bool,
        featurize_output=False,
        relfeat_output=False,
        featurize_prev_orders=False,
        value_softmax=False,
        encoder_cfg: conf.conf_cfgs.Encoder,
        pad_spatial_size_to_multiple=1,
        all_powers: bool,
        has_single_chances: bool,
        has_double_chances: bool,
        has_policy=True,
        has_value=True,
        use_player_ratings=False,
        use_year=False,
        use_agent_power=False,
        num_scoring_systems=1,  # Uses the first N of the scoring systems
        input_version=1,
        training_permute_powers=False,
        with_order_conditioning=False,
        transformer_decoder=Optional[conf.conf_cfgs.TrainTask.TransformerDecoder],
    ):
        super().__init__()

        self.input_version = input_version
        self.board_state_size = get_board_state_size(input_version)

        self.orders_vocab_size = orders_vocab_size

        # Make the type checker understand what self.order_feats is
        if TYPE_CHECKING:
            self.order_feats = torch.tensor([])
        self.featurize_prev_orders = featurize_prev_orders
        self.prev_order_enc_size = prev_order_emb_size
        if featurize_prev_orders:
            order_feats, _srcs, _dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            self.prev_order_enc_size += self.order_feats.shape[-1]

        # Register a buffer that maps global order index to source location
        # of that order. We register this as a buffer to make sure it's always
        # on the same device as the base_strategy_model itself.
        srcloc_idx_of_global_order_idx_plus_one = compute_srcloc_idx_of_global_order_idx_plus_one()
        self.register_buffer(
            "srcloc_idx_of_global_order_idx_plus_one",
            srcloc_idx_of_global_order_idx_plus_one,
            persistent=False,
        )
        # Make the type checker understand what self.srcloc_idx_of_global_order_idx is
        if TYPE_CHECKING:
            self.srcloc_idx_of_global_order_idx_plus_one = torch.tensor([])

        self.has_policy = has_policy
        self.has_value = has_value
        self.num_scoring_systems = num_scoring_systems

        # Use os.urandom so as to explicitly be different on different distributed data
        # parallel processes and not share seeds.
        self.training_permute_powers = training_permute_powers
        self.permute_powers_rand = np.random.default_rng(seed=list(os.urandom(16)))  # type:ignore

        self.board_map_size = board_map_size
        self.transformer_sequence_len = board_map_size + len(POWERS) + 1

        encoder_kind = encoder_cfg.WhichOneof("encoder")
        assert encoder_kind == "transformer"

        # Note:Due to historical accident the actual size we use everywhere is inter_emb_size*2
        self.inter_emb_size = inter_emb_size

        # These are linear maps we use to embed every input into the tensor we feed to the transformer.
        # Location-keyed inputs
        self.board_emb_linear = nn.Linear(self.board_state_size, inter_emb_size * 2)
        self.prev_board_emb_linear = nn.Linear(self.board_state_size, inter_emb_size * 2)
        self.prev_order_emb_linear = nn.Linear(self.prev_order_enc_size, inter_emb_size * 2)

        if with_order_conditioning:
            self.this_order_emb_linear = nn.Linear(self.prev_order_enc_size, inter_emb_size * 2)
        else:
            self.this_order_emb_linear = None

        # Power-keyed inputs
        self.build_numbers_emb_linear = nn.Linear(1, inter_emb_size * 2)
        self.player_ratings_emb_linear = None
        if use_player_ratings:
            self.player_ratings_emb_linear = nn.Linear(1, inter_emb_size * 2)
        self.agent_power_emb_linear = None
        if use_agent_power:
            self.agent_power_emb_linear = nn.Linear(1, inter_emb_size * 2)

        # Global inputs
        self.season_emb_linear = nn.Linear(3, inter_emb_size * 2)
        self.in_adj_phase_emb_linear = nn.Linear(1, inter_emb_size * 2)
        self.has_press_emb_linear = nn.Linear(1, inter_emb_size * 2)
        self.scoring_system_emb_linear = None
        if self.num_scoring_systems > 1:
            self.scoring_system_emb_linear = nn.Linear(num_scoring_systems, inter_emb_size * 2)
        self.year_emb_linear = None
        if use_year:
            self.year_emb_linear = nn.Linear(1, inter_emb_size * 2)

        if pad_spatial_size_to_multiple > 1:
            self.transformer_sequence_len = (
                (self.transformer_sequence_len + pad_spatial_size_to_multiple - 1)
                // pad_spatial_size_to_multiple
                * pad_spatial_size_to_multiple
            )
        trans_encoder_cfg = getattr(encoder_cfg, encoder_kind)
        assert isinstance(trans_encoder_cfg, conf.conf_cfgs.Encoder.Transformer)
        self.encoder = TransformerEncoder(
            total_input_size=inter_emb_size * 2,
            spatial_size=self.transformer_sequence_len,
            inter_emb_size=inter_emb_size,
            encoder_cfg=trans_encoder_cfg,
        )

        if has_policy:
            if transformer_decoder is not None:
                self.policy_decoder = TransfDecoder(
                    inter_emb_size=inter_emb_size, cfg=transformer_decoder,
                )
            else:
                self.policy_decoder = LSTMBaseStrategyModelDecoder(
                    inter_emb_size=inter_emb_size,
                    spatial_size=self.transformer_sequence_len,
                    orders_vocab_size=orders_vocab_size,
                    lstm_size=lstm_size,
                    order_emb_size=order_emb_size,
                    lstm_dropout=lstm_dropout,
                    lstm_layers=lstm_layers,
                    master_alignments=None,
                    use_simple_alignments=True,
                    power_emb_size=0,
                    featurize_output=featurize_output,
                    relfeat_output=relfeat_output,
                )
        if has_value:
            self.value_decoder = ValueDecoder(
                inter_emb_size=inter_emb_size,
                spatial_size=self.transformer_sequence_len,
                init_scale=value_decoder_init_scale,
                dropout=value_dropout,
                softmax=value_softmax,
                activation=value_decoder_activation,
                use_weighted_pool=value_decoder_use_weighted_pool,
                extract_from_encoder=value_decoder_extract_from_encoder,
            )

        self.prev_order_embedding = nn.Embedding(
            orders_vocab_size, prev_order_emb_size, padding_idx=0
        )

        self.all_powers = all_powers
        self.has_single_chances = has_single_chances
        self.has_double_chances = has_double_chances

    def get_input_version(self) -> int:
        return self.input_version

    def get_training_permute_powers(self) -> bool:
        return self.training_permute_powers

    def set_training_permute_powers(self, b: bool):
        self.training_permute_powers = b

    def is_all_powers(self) -> bool:
        return self.all_powers

    def supports_single_power_decoding(self) -> bool:
        return not self.all_powers or self.has_single_chances

    def supports_double_power_decoding(self) -> bool:
        return self.all_powers and self.has_double_chances

    def get_srcloc_idx_of_global_order_idx_plus_one(self) -> torch.Tensor:
        """Return a tensor mapping (global order idx+1) -> location idx of src of order.
        EOS_IDX+1 is mapped to a value larger than any location idx.
        """
        return self.srcloc_idx_of_global_order_idx_plus_one

    def _embed_orders(self, orders: torch.Tensor, x_board_state: torch.Tensor):
        B, NUM_LOCS, _ = x_board_state.shape
        order_emb = self.prev_order_embedding(orders[:, 0])
        if self.featurize_prev_orders:
            order_emb = torch.cat((order_emb, self.order_feats[orders[:, 0]]), dim=-1)

        # insert the prev orders into the correct board location
        order_exp = x_board_state.new_zeros(B, NUM_LOCS, self.prev_order_enc_size)
        prev_order_loc_idxs = torch.arange(B, device=x_board_state.device).repeat_interleave(
            orders.shape[-1]
        ) * NUM_LOCS + orders[:, 1].reshape(-1)
        order_exp.view(-1, self.prev_order_enc_size).index_add_(
            0, prev_order_loc_idxs, order_emb.view(-1, self.prev_order_enc_size)
        )
        return order_exp

    def encode_state(
        self,
        *,
        x_board_state,
        x_prev_state,
        x_prev_orders,
        x_season,
        x_year_encoded,
        x_in_adj_phase,
        x_build_numbers,
        x_has_press=None,
        x_player_ratings=None,
        x_scoring_system=None,
        x_agent_power=None,
        x_current_orders: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs encoder."""

        # following https://arxiv.org/pdf/2006.04635.pdf , Appendix C
        B, NUM_LOCS, _ = x_board_state.shape
        assert NUM_LOCS == self.board_map_size

        # Preemptively make sure that dtypes of things match, to try to limit the chance of bugs
        # if the inputs were built in an ad-hoc way when are trying to run in fp16.
        assert x_board_state.dtype == x_prev_state.dtype
        assert x_board_state.dtype == x_build_numbers.dtype
        assert x_board_state.dtype == x_season.dtype
        if x_has_press is not None:
            assert x_board_state.dtype == x_has_press.dtype

        # B. insert the prev orders into the correct board location (which is in the second column of x_po)
        x_prev_order_exp = self._embed_orders(x_prev_orders, x_board_state)

        # The final tensor we feed to the encoder for all board-location-keyed data.
        # [B, 81, inter_emb_size*2]
        assert len(x_board_state.shape) == 3
        encoder_input_by_location = self.board_emb_linear(x_board_state)
        assert len(x_prev_state.shape) == 3
        encoder_input_by_location += self.prev_board_emb_linear(x_prev_state)
        assert len(x_prev_order_exp.shape) == 3
        encoder_input_by_location += self.prev_order_emb_linear(x_prev_order_exp)

        if self.this_order_emb_linear is not None:
            if x_current_orders is None:
                # Feed zeroes. This is how we encode conditioning on empty order in
                # FeatureEncoder.encode_orders_single.
                x_current_orders = x_prev_orders.new_zeros(x_prev_orders.shape)
            assert x_current_orders is not None
            this_order_exp = self._embed_orders(x_current_orders, x_board_state)
            encoder_input_by_location += self.this_order_emb_linear(this_order_exp)
        else:
            assert (
                x_current_orders is None
            ), "Got x_current_orders parameter for a model that does not support conditional sampling"

        # The final tensor we feed to the encoder for all power-keyed data.
        # [B, 7, inter_emb_size*2]
        assert x_build_numbers is not None and len(x_build_numbers.shape) == 2
        encoder_input_by_power = self.build_numbers_emb_linear(x_build_numbers.unsqueeze(-1))
        if self.player_ratings_emb_linear:
            if x_player_ratings is not None:
                assert len(x_player_ratings.shape) == 2
                encoder_input_by_power += self.player_ratings_emb_linear(
                    x_player_ratings.unsqueeze(-1)
                )
            else:
                encoder_input_by_power += self.player_ratings_emb_linear(
                    x_board_state.new_ones(B, len(POWERS), 1)
                )
        if self.agent_power_emb_linear:
            if x_agent_power is not None:
                assert len(x_agent_power.shape) == 2
                encoder_input_by_power += self.agent_power_emb_linear(x_agent_power.unsqueeze(-1))
            else:
                encoder_input_by_power += self.agent_power_emb_linear(
                    x_board_state.new_zeros(B, len(POWERS), 1)
                )

        # The final tensor we feed to the encoder for all global singleton data.
        # [B, 1, inter_emb_size*2]
        assert len(x_season.shape) == 2
        encoder_input_global = self.season_emb_linear(x_season.unsqueeze(1))
        assert len(x_in_adj_phase.shape) == 1
        encoder_input_global += self.in_adj_phase_emb_linear(
            x_in_adj_phase.unsqueeze(-1).unsqueeze(-1)
        )
        if x_has_press is not None:
            assert len(x_has_press.shape) == 2
            encoder_input_global += self.has_press_emb_linear(x_has_press.unsqueeze(1))
        else:
            # If not provided, default to treating it as no press
            encoder_input_global += self.has_press_emb_linear(x_board_state.new_zeros(B, 1, 1))

        if self.scoring_system_emb_linear is not None:
            if x_scoring_system is not None:
                assert len(x_scoring_system.shape) == 2
                encoder_input_global += self.scoring_system_emb_linear(
                    x_scoring_system.unsqueeze(1)
                )
            else:
                # If we're training a model that supports scoring systems but is not provided
                # at inference time, then assume it's sos, at training fail
                assert (
                    not self.training
                ), "Training a model with scoring systems but not providing it as input"
                encoder_input_global += self.scoring_system_emb_linear(
                    F.one_hot(
                        x_board_state.new_full(
                            (B, 1), fill_value=Scoring.SOS.value, dtype=torch.long
                        ),
                        num_classes=self.num_scoring_systems,
                    ).to(x_board_state.dtype)
                )

        if self.year_emb_linear is not None:
            encoder_input_global += self.year_emb_linear(x_year_encoded.unsqueeze(1))

        # Concat everything.
        # [B, 81+7+1, inter_emb_size*2]
        encoder_input = torch.cat(
            [encoder_input_by_location, encoder_input_by_power, encoder_input_global], dim=1
        )

        if self.transformer_sequence_len != encoder_input.shape[1]:
            # pad -> (batch, transformer_sequence_len, channels)
            assert self.transformer_sequence_len > encoder_input.shape[1]
            assert len(encoder_input.shape) == 3
            encoder_input = F.pad(
                encoder_input, (0, 0, 0, self.transformer_sequence_len - encoder_input.shape[1])
            )

        encoded = self.encoder(encoder_input)
        return encoded

    def forward(
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
        need_policy=True,
        need_value=True,
        pad_to_max=False,
        x_current_orders: Optional[torch.Tensor] = None,
        encoded: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        B indexes independent elements of a batch.
        S indexes orderable locations.

        Arguments:
        - x_board_state: [B, 81, board_state_size]
        - x_prev_state: [B, 2, 100], long
        - x_prev_orders: [B, 81, 40]
        - x_season: [B, 3]
        - x_year_encoded: [B, 1]
        - x_in_adj_phase: [B], bool
        - x_build_numbers: [B, 7]
        - x_loc_idxs: int8, [B, 81] or [B, 7, 81]
             The sequence of location idxs to decode, or the sequence of location idxs to decode
             for each of the 7 powers, by numbering the locations in order by 0, 1, 2, ...
             and marking locations that don't have a unit to be ordered with EOS_IDX.
             On builds and disbands and retreats, the build and disband and retreat locations must be marked with "-2".
             Models are generally only trained on location idxs decoding in sorted order.
        - x_possible_actions: long, [B, S, 469] or [B, 7, S, 469]
             For each sequence idx (or for each power for each seq idx), the list of (up to 469)
             possible global_order_idxs that can be legally issued for that unit,
             On builds, the 0th sequence idx just lists the possible combined builds.
             On adjustment-disbands, the successive sequence idxs are each one disband, and we have a hack
             to resample duplicate disbands.
             On retreats, the successive sequence idxs are the location-sorted places needing to retreat.
             The position at which an order is in this list oof 469 is known as "local" order idx.

        - temperature: softmax temp, lower = more deterministic; must be either
          a float or a tensor of [B, 1]
        - top_p: probability mass to samples from, lower = more spiky; must
          be either a float or a tensor of [B, 1]
        - batch_repeat_interleave: if set to a value k, will behave as if [B] dimension was
            was actually [B*k] in size, with each element repeated k times
            (e.g. [1,2,3] k=2 -> [1,1,2,2,3,3]), on all tensors EXCEPT teacher_force_orders
        - teacher_force_orders: [B, S] or [B, 7, S] long or None,
            global ORDER idxs, NOT local idxs. This is 0-padded.
            If batch_repeat_interleave is provided, then the shape must be
            [B*batch_repeat_interleave, S] or [B*batch_repeat_interleave, 7, S].
        - x_power: [B, S] long, [B, 7, S] long, or None.
            Labels which power idx is being decoded for each item in the sequence, or for each item
            in the sequence for each of the 7 sequences for the different powers.
            On movement phases, generally this tensor will just be constant, or constant per
            each power.
            On all powers, the [B,7,S] form or None must be used, S is expected to be equal to 34.
            Only [:,0,:] is used on movement phases and retreat phases, and the sequence simply
            walks through all the orderable locations in order, labeling which power.
            Build and adjustment-disband are still encoded non-jointly and use all 7 sequences separately
            per power.

        - x_has_press: [B, 1] or None
        - x_player_ratings: [B, 7] player rating percentile (0 - 1) for each player, or None
        - x_agent_power: [B, 7] one-hot indicator of agent power or None
        - x_scoring_system: [B, num_scoring_systems] or None - for each scoring system, the weight
            on that scoring system, weights should add up to 1 for each batch element.
        - need_policy: if not set, global_order_idxs, local_order_idxs, and logits will be None.
        - need_value: if not set, final_sos in Result will be None
        - pad_to_max, if set, will pad all output tensors to [..., MAX_SEQ_LEN, 469]. Use that
            to make torch.nn.DataPatallel to work.

        - x_current_orders: [B, 81, 40]: orders for this phase to condition on. Only
            possible if with_order_conditioning is True

        if x_power is None or [B, 7, 34] Long, the model will decode for all 7 powers.
            - loc_idxs, all_cand_idxs (i.e. x_possible_actions), and teacher_force_orders must have an
              extra axis at dim=1 with size 7
            - global_order_idxs and order_scores will be returned with an extra axis
              at dim=1 with size 7
            - if x_power is [B, 7, 34] Long, non-A phases are expected to be encoded in [:,0,:]
        else x_power must be [B, S] Long and only one power's sequence will be decoded

        Returns:
          - global_order_idxs [B, S] or [B, 7, S]: idx in ORDER_VOCABULARY of sampled
            orders for each power
          - local_order_idxs [B, S] or [B, 7, S]: idx in all_cand_idxs of sampled
            orders for each power
          - logits [B, S, C] or [B, 7, S, C]: masked pre-softmax logits of each
            candidate order, 0 < S <= 17, 0 < C <= 469
          - final_sos [B, 7]: estimated sum of squares share for each power
        """

        assert not (need_policy and not self.has_policy)
        assert not (need_value and not self.has_value)

        assert need_policy or need_value

        power_permutation_matrix = None
        if self.training and self.training_permute_powers:
            (
                x_board_state,
                x_prev_state,
                (x_build_numbers, x_player_ratings, x_agent_power),
                power_permutation_matrix,
            ) = _apply_permute_powers(
                input_version=self.input_version,
                permute_powers_rand=self.permute_powers_rand,
                x_board_state=x_board_state,
                x_prev_state=x_prev_state,
                per_power_tensors=(x_build_numbers, x_player_ratings, x_agent_power),
            )

        if encoded is None:
            encoded = self.encode_state(
                x_board_state=x_board_state,
                x_prev_state=x_prev_state,
                x_prev_orders=x_prev_orders,
                x_season=x_season,
                x_year_encoded=x_year_encoded,
                x_in_adj_phase=x_in_adj_phase,
                x_build_numbers=x_build_numbers,
                x_has_press=x_has_press,
                x_player_ratings=x_player_ratings,
                x_scoring_system=x_scoring_system,
                x_agent_power=x_agent_power,
                x_current_orders=x_current_orders,
            )

        if not need_value:
            final_sos = None
        else:
            final_sos = self.value_decoder(encoded)
            if batch_repeat_interleave is not None:
                final_sos = torch.repeat_interleave(final_sos, batch_repeat_interleave, dim=0)

            if power_permutation_matrix is not None and final_sos is not None:
                final_sos = torch.matmul(
                    power_permutation_matrix.to(final_sos.device), final_sos.unsqueeze(2),
                ).squeeze(2)

        if not need_policy:
            global_order_idxs = local_order_idxs = logits = None
        else:
            # NOTE - "all_powers" here indicates whether we are decoding as an
            # model trained to predict the joint action distribution instead of
            # decoding powers only individually.
            # This is NOT the same thing as _forward_all_powers, because
            # _forward_all_powers simply means whether we are decoding all 7
            # powers (whether jointly or individually).
            # So, for example, it is not a bug that we may call
            # _forward_all_powers even when all_powers is False.
            all_powers = x_power is not None and len(x_power.shape) == 3
            if all_powers:
                assert (
                    self.all_powers
                ), "BaseStrategyModel got all_powers query but model is not all_powers"
            if x_power is None or all_powers:
                global_order_idxs, local_order_idxs, logits = _forward_all_powers(
                    policy_decoder=self.policy_decoder,
                    enc=encoded,
                    loc_idxs=x_loc_idxs,
                    cand_idxs=x_possible_actions,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                    teacher_force_orders=teacher_force_orders,
                    power=x_power,
                )
            else:
                global_order_idxs, local_order_idxs, logits = _forward_one_power(
                    policy_decoder=self.policy_decoder,
                    enc=encoded,
                    loc_idxs=x_loc_idxs,
                    cand_idxs=x_possible_actions,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                    teacher_force_orders=teacher_force_orders,
                    power=x_power,
                )
            if pad_to_max:
                global_order_idxs, local_order_idxs, logits = _pad_to_max(
                    global_order_idxs, local_order_idxs, logits, all_powers
                )

        return global_order_idxs, local_order_idxs, logits, final_sos


class BaseStrategyModel(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,  # fairdiplomacy.utils.thread_pool_encoding.get_board_state_size
        # prev_orders_size,  # 40
        inter_emb_size,  # 120
        power_emb_size,  # 60
        season_emb_size,  # 20,
        num_blocks,  # 16
        A,  # 81x81
        master_alignments,
        orders_vocab_size,  # 13k
        lstm_size,  # 200
        order_emb_size,  # 80
        prev_order_emb_size,  # 20
        lstm_dropout=0,
        lstm_layers=1,
        encoder_dropout=0,
        value_dropout,
        use_simple_alignments=False,
        value_decoder_init_scale=1.0,
        featurize_output=False,
        relfeat_output=False,
        featurize_prev_orders=False,
        residual_linear=False,
        merged_gnn=False,
        encoder_layerdrop=0,
        value_softmax=False,
        encoder_cfg=None,
        pad_spatial_size_to_multiple=1,
        all_powers,
        has_policy=True,
        has_value=True,
        use_player_ratings=False,
        input_version=1,
        training_permute_powers=False,
    ):
        super().__init__()

        assert board_state_size == get_board_state_size(
            input_version
        ), f"Board state size {board_state_size} does not match expected for version {input_version}"
        self.input_version = input_version
        self.board_state_size = board_state_size

        self.orders_vocab_size = orders_vocab_size

        # Make the type checker understand what self.order_feats is
        if TYPE_CHECKING:
            self.order_feats = torch.tensor([])
        self.featurize_prev_orders = featurize_prev_orders
        self.prev_order_enc_size = prev_order_emb_size
        if has_policy and featurize_prev_orders:
            order_feats, _srcs, _dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            self.prev_order_enc_size += self.order_feats.shape[-1]

        # Register a buffer that maps global order index to source location
        # of that order
        srcloc_idx_of_global_order_idx_plus_one = compute_srcloc_idx_of_global_order_idx_plus_one()
        self.register_buffer(
            "srcloc_idx_of_global_order_idx_plus_one",
            srcloc_idx_of_global_order_idx_plus_one,
            persistent=False,
        )
        # Make the type checker understand what self.srcloc_idx_of_global_order_idx_plus_one is
        if TYPE_CHECKING:
            self.srcloc_idx_of_global_order_idx_plus_one = torch.tensor([])

        self.has_policy = has_policy
        self.has_value = has_value
        self.use_player_ratings = use_player_ratings
        self.use_agent_power = False
        # Use os.urandom so as to explicitly be different on different distributed data
        # parallel processes and not share seeds.
        self.training_permute_powers = training_permute_powers
        self.permute_powers_rand = np.random.default_rng(seed=list(os.urandom(16)))  # type:ignore

        self.spatial_size = A.size()[0]

        encoder_kind = encoder_cfg.WhichOneof("encoder")
        extra_input_size = len(POWERS) + season_emb_size + 1
        if self.use_player_ratings:
            extra_input_size += len(POWERS) + 1
        board_state_input_dim = board_state_size + extra_input_size
        prev_orders_input_dim = board_state_size + self.prev_order_enc_size + extra_input_size
        if encoder_kind == "transformer":
            if pad_spatial_size_to_multiple > 1:
                self.spatial_size = (
                    (self.spatial_size + pad_spatial_size_to_multiple - 1)
                    // pad_spatial_size_to_multiple
                    * pad_spatial_size_to_multiple
                )
            encoder_cfg = getattr(encoder_cfg, encoder_kind)
            self.encoder = TransformerEncoder(
                total_input_size=board_state_input_dim + prev_orders_input_dim,
                spatial_size=self.spatial_size,
                inter_emb_size=inter_emb_size,
                encoder_cfg=encoder_cfg,
            )
        elif encoder_kind is None:  # None == graph encoder
            if pad_spatial_size_to_multiple > 1:
                raise ValueError(
                    "pad_spatial_size_to_multiple > 1 not supported for graph conv encoder"
                )
            self.encoder = BaseStrategyModelEncoder(
                board_state_size=board_state_input_dim,
                prev_orders_size=prev_orders_input_dim,
                inter_emb_size=inter_emb_size,
                num_blocks=num_blocks,
                A=A,
                dropout=encoder_dropout,
                residual_linear=residual_linear,
                merged_gnn=merged_gnn,
                layerdrop=encoder_layerdrop,
            )
        else:
            assert False

        if has_policy:
            self.policy_decoder = LSTMBaseStrategyModelDecoder(
                inter_emb_size=inter_emb_size,
                spatial_size=self.spatial_size,
                orders_vocab_size=orders_vocab_size,
                lstm_size=lstm_size,
                order_emb_size=order_emb_size,
                lstm_dropout=lstm_dropout,
                lstm_layers=lstm_layers,
                master_alignments=master_alignments,
                use_simple_alignments=use_simple_alignments,
                power_emb_size=power_emb_size,
                featurize_output=featurize_output,
                relfeat_output=relfeat_output,
            )

        if has_value:
            self.value_decoder = ValueDecoder(
                inter_emb_size=inter_emb_size,
                spatial_size=self.spatial_size,
                init_scale=value_decoder_init_scale,
                dropout=value_dropout,
                softmax=value_softmax,
                use_weighted_pool=False,
                extract_from_encoder=False,
            )

        self.season_lin = nn.Linear(3, season_emb_size)
        self.prev_order_embedding = nn.Embedding(
            orders_vocab_size, prev_order_emb_size, padding_idx=0
        )

        self.all_powers = all_powers

    def get_input_version(self) -> int:
        return self.input_version

    def get_training_permute_powers(self) -> bool:
        return self.training_permute_powers

    def set_training_permute_powers(self, b: bool):
        self.training_permute_powers = b

    def is_all_powers(self) -> bool:
        return self.all_powers

    def supports_single_power_decoding(self) -> bool:
        return not self.all_powers

    def supports_double_power_decoding(self) -> bool:
        return False

    def get_srcloc_idx_of_global_order_idx_plus_one(self) -> torch.Tensor:
        """Return a tensor mapping (global order idx+1) -> location idx of src of order.
        EOS_IDX+1 is mapped to a value larger than any location idx.
        """
        return self.srcloc_idx_of_global_order_idx_plus_one

    def forward(
        self,
        *,
        x_board_state,
        x_prev_state,
        x_prev_orders,
        x_season,
        x_year_encoded,
        x_in_adj_phase,  # Unused
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
        need_policy=True,
        need_value=True,
        pad_to_max=False,
        x_current_orders=None,
        encoded=None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        See docs for base_strategy_modelv2
        """
        del encoded  # Not supported.

        # following https://arxiv.org/pdf/2006.04635.pdf , Appendix C
        B, NUM_LOCS, _ = x_board_state.shape

        # Preemptively make sure that dtypes of things match, to try to limit the chance of bugs
        # if the inputs were built in an ad-hoc way when are trying to run in fp16.
        assert x_board_state.dtype == x_prev_state.dtype
        assert x_board_state.dtype == x_build_numbers.dtype
        assert x_board_state.dtype == x_season.dtype
        if x_has_press is not None:
            assert x_board_state.dtype == x_has_press.dtype

        assert not (need_policy and not self.has_policy)
        assert not (need_value and not self.has_value)

        assert need_policy or need_value

        assert (
            x_current_orders is None
        ), "Old base_strategy_model does not support x_current_orders"

        power_permutation_matrix = None
        if self.training and self.training_permute_powers:
            (
                x_board_state,
                x_prev_state,
                (x_build_numbers, x_player_ratings, x_agent_power),
                power_permutation_matrix,
            ) = _apply_permute_powers(
                input_version=self.input_version,
                permute_powers_rand=self.permute_powers_rand,
                x_board_state=x_board_state,
                x_prev_state=x_prev_state,
                per_power_tensors=(x_build_numbers, x_player_ratings, x_agent_power),
            )
            assert x_build_numbers is not None

        # A. get season and prev order embs
        x_season_emb = self.season_lin(x_season)

        x_prev_order_emb = self.prev_order_embedding(x_prev_orders[:, 0])
        if self.featurize_prev_orders:
            x_prev_order_emb = torch.cat(
                (x_prev_order_emb, self.order_feats[x_prev_orders[:, 0]]), dim=-1
            )

        # B. insert the prev orders into the correct board location (which is in the second column of x_po)
        x_prev_order_exp = x_board_state.new_zeros(B, NUM_LOCS, self.prev_order_enc_size)
        prev_order_loc_idxs = torch.arange(B, device=x_board_state.device).repeat_interleave(
            x_prev_orders.shape[-1]
        ) * NUM_LOCS + x_prev_orders[:, 1].reshape(-1)
        x_prev_order_exp.view(-1, self.prev_order_enc_size).index_add_(
            0, prev_order_loc_idxs, x_prev_order_emb.view(-1, self.prev_order_enc_size)
        )

        # concatenate the subcomponents into board state and prev state, following the paper
        x_build_numbers_exp = x_build_numbers[:, None].expand(-1, NUM_LOCS, -1)
        x_season_emb_exp = x_season_emb[:, None].expand(-1, NUM_LOCS, -1)

        if x_has_press is not None:
            x_has_press_exp = x_has_press[:, None].expand(-1, NUM_LOCS, 1)
        else:
            x_has_press_exp = x_board_state.new_zeros(B, NUM_LOCS, 1)

        if self.use_player_ratings:
            if x_player_ratings is not None:
                x_player_ratings_exp = x_player_ratings[:, None].expand(-1, NUM_LOCS, len(POWERS))
            else:
                # assume player have top rating if not supplied
                x_player_ratings_exp = x_board_state.new_ones((B, NUM_LOCS, len(POWERS)))

            # assert that the encoding of the ownership of units of powers is contiguous
            encoding_unit_ownership_idxs = pydipcc.encoding_unit_ownership_idxs(self.input_version)
            assert tuple(
                x - encoding_unit_ownership_idxs[0] for x in encoding_unit_ownership_idxs
            ) == tuple(range(len(POWERS)))

            # add in the rating for the controlling power at each loc
            loc_power_idx = encoding_unit_ownership_idxs[0]
            loc_power = x_board_state[:, :, loc_power_idx : loc_power_idx + len(POWERS)]
            # assert each location controlled by at most one power
            # assert loc_power.sum(-1).max() <= 1  # this assert is too slow
            unit_player_ratings = (x_player_ratings_exp * loc_power).sum(-1, keepdim=True)
            x_player_ratings_exp = torch.cat((x_player_ratings_exp, unit_player_ratings), dim=-1)
        else:
            # append an empty tensor for ratings (noop)
            x_player_ratings_exp = x_board_state.new_zeros((B, NUM_LOCS, 0))

        assert x_player_ratings_exp.dtype == x_board_state.dtype

        x_bo_hat = torch.cat(
            (
                x_board_state,
                x_build_numbers_exp,
                x_season_emb_exp,
                x_has_press_exp,
                x_player_ratings_exp,
            ),
            dim=-1,
        )
        x_po_hat = torch.cat(
            (
                x_prev_state,
                x_prev_order_exp,
                x_build_numbers_exp,
                x_season_emb_exp,
                x_has_press_exp,
                x_player_ratings_exp,
            ),
            dim=-1,
        )

        assert x_bo_hat.size()[1] == x_po_hat.size()[1]
        if self.spatial_size != x_bo_hat.size()[1]:
            # pad (batch, 81, channels) -> (batch, spatial_size, channels)
            assert self.spatial_size > x_bo_hat.size()[1]
            assert len(x_bo_hat.size()) == 3
            x_bo_hat = F.pad(x_bo_hat, (0, 0, 0, self.spatial_size - x_bo_hat.size()[1]))
        if self.spatial_size != x_po_hat.size()[1]:
            # pad (batch, 81, channels) -> (batch, spatial_size, channels)
            assert self.spatial_size > x_po_hat.size()[1]
            assert len(x_po_hat.size()) == 3
            x_po_hat = F.pad(x_po_hat, (0, 0, 0, self.spatial_size - x_po_hat.size()[1]))

        if isinstance(self.encoder, TransformerEncoder):
            encoded = self.encoder(torch.cat([x_bo_hat, x_po_hat], -1))
        else:
            encoded = self.encoder(x_bo_hat, x_po_hat)

        if need_value:
            final_sos = self.value_decoder(encoded)
            if batch_repeat_interleave is not None:
                final_sos = torch.repeat_interleave(final_sos, batch_repeat_interleave, dim=0)

            if power_permutation_matrix is not None and final_sos is not None:
                final_sos = torch.matmul(
                    power_permutation_matrix.to(final_sos.device), final_sos.unsqueeze(2),
                ).squeeze(2)
        else:
            final_sos = None

        all_powers = x_power is not None and len(x_power.shape) == 3

        if not need_policy:
            global_order_idxs = local_order_idxs = logits = None
        else:
            if x_power is None or all_powers:
                global_order_idxs, local_order_idxs, logits = _forward_all_powers(
                    policy_decoder=self.policy_decoder,
                    enc=encoded,
                    loc_idxs=x_loc_idxs,
                    cand_idxs=x_possible_actions,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                    teacher_force_orders=teacher_force_orders,
                    power=x_power,
                )
            else:
                global_order_idxs, local_order_idxs, logits = _forward_one_power(
                    policy_decoder=self.policy_decoder,
                    enc=encoded,
                    loc_idxs=x_loc_idxs,
                    cand_idxs=x_possible_actions,
                    temperature=temperature,
                    top_p=top_p,
                    batch_repeat_interleave=batch_repeat_interleave,
                    teacher_force_orders=teacher_force_orders,
                    power=x_power,
                )
            if pad_to_max:
                global_order_idxs, local_order_idxs, logits = _pad_to_max(
                    global_order_idxs, local_order_idxs, logits, all_powers
                )

        return global_order_idxs, local_order_idxs, logits, final_sos


def check_permute_powers():
    # This is a safeguard so that if base_strategy_model is modified,
    # we don't forget to update apply_permute_powers
    # If you are updating this function, please consider whether the new base_strategy_model input you
    # are adding needs to also have permutations of the powers applied to it.
    # Please update apply_permute_powers if it does.
    expected_keys = {
        "x_possible_actions",
        "batch_repeat_interleave",
        "teacher_force_orders",
        "x_board_state",
        "x_prev_state",
        "need_value",
        "x_season",
        "x_year_encoded",
        "need_policy",
        "x_build_numbers",
        "x_prev_orders",
        "x_current_orders",
        "x_in_adj_phase",
        "x_power",
        "x_has_press",
        "x_player_ratings",
        "x_scoring_system",
        "x_agent_power",
        "top_p",
        "self",
        "temperature",
        "pad_to_max",
        "x_loc_idxs",
        "encoded",
    }
    actual_keys = set(inspect.signature(BaseStrategyModel.forward).parameters.keys())
    assert (
        expected_keys == actual_keys
    ), "New inputs added to base_strategy_model, please consider effects on symmetry augmentation"
    actual_keys = set(inspect.signature(BaseStrategyModelV2.forward).parameters.keys())
    assert (
        expected_keys == actual_keys
    ), "New inputs added to base_strategy_modelv2, please consider effects on symmetry augmentation"


def _pad_to_max(global_order_idxs, local_order_idxs, logits, all_powers: bool):
    max_seq_len = N_SCS if all_powers else MAX_SEQ_LEN
    global_order_idxs = _pad_last_dims(global_order_idxs, [max_seq_len], EOS_IDX)
    local_order_idxs = _pad_last_dims(local_order_idxs, [max_seq_len], EOS_IDX)
    logits = _pad_last_dims(logits, [max_seq_len, 469], LOGIT_MASK_VAL)
    return (global_order_idxs, local_order_idxs, logits)


def _forward_one_power(
    *,
    policy_decoder,
    enc,
    loc_idxs,
    cand_idxs,
    power,
    temperature,
    top_p,
    batch_repeat_interleave,
    teacher_force_orders,
):
    assert len(loc_idxs.shape) == 2, loc_idxs.shape
    assert len(cand_idxs.shape) == 3, cand_idxs.shape

    if batch_repeat_interleave is not None:
        if teacher_force_orders is not None:
            assert (
                teacher_force_orders.shape[0] == batch_repeat_interleave * enc.shape[0]
            ), teacher_force_orders.shape
        (enc, loc_idxs, cand_idxs, power, temperature, top_p,) = apply_batch_repeat_interleave(
            (enc, loc_idxs, cand_idxs, power, temperature, top_p,), batch_repeat_interleave,
        )

    global_order_idxs, local_order_idxs, logits = policy_decoder(
        enc,
        loc_idxs,
        cand_idxs,
        temperature=temperature,
        top_p=top_p,
        teacher_force_orders=teacher_force_orders,
        power=power,
    )

    return global_order_idxs, local_order_idxs, logits


def _forward_all_powers(
    *,
    policy_decoder,
    enc,
    loc_idxs,
    cand_idxs,
    temperature,
    teacher_force_orders,
    top_p,
    batch_repeat_interleave,
    log_timings=False,
    power=None,
):
    timings = TimingCtx()

    assert len(loc_idxs.shape) == 3
    assert len(cand_idxs.shape) == 4

    with timings("policy_decoder_prep"):
        if batch_repeat_interleave is not None:
            if teacher_force_orders is not None:
                assert teacher_force_orders.shape[0] == batch_repeat_interleave * enc.shape[0], (
                    teacher_force_orders.shape,
                    batch_repeat_interleave,
                    enc.shape,
                )
            (enc, loc_idxs, cand_idxs, power, temperature, top_p,) = apply_batch_repeat_interleave(
                (enc, loc_idxs, cand_idxs, power, temperature, top_p,), batch_repeat_interleave,
            )

        NPOWERS = len(POWERS)
        enc_repeat = enc.repeat_interleave(NPOWERS, dim=0)
        loc_idxs = loc_idxs.view(-1, loc_idxs.shape[2])
        cand_idxs = cand_idxs.view(-1, *cand_idxs.shape[2:])
        temperature = repeat_interleave_if_tensor(temperature, NPOWERS, dim=0)
        top_p = repeat_interleave_if_tensor(top_p, NPOWERS, dim=0)
        teacher_force_orders = (
            teacher_force_orders.view(-1, *teacher_force_orders.shape[2:])
            if teacher_force_orders is not None
            else None
        )

        if power is None:
            # N.B. use repeat, not repeat_interleave, for power only. Each
            # batch is contiguous, and we want a sequence of power idxs for each batch
            power = (
                torch.arange(NPOWERS, device=enc.device)
                .view(-1, 1)
                .repeat(enc.shape[0], cand_idxs.shape[1])
            )
        else:
            # This is all-powers encoding: validate shape and use power idxs from input
            assert len(power.shape) == 3, power.shape
            assert power.shape[1] == NPOWERS, power.shape
            assert power.shape[2] == N_SCS, power.shape
            power = power.view(-1, N_SCS)

    with timings("policy_decoder"):
        # [B, 17, 469] -> [B, 17].
        valid_mask = (cand_idxs != EOS_IDX).any(dim=-1)
        # [B, 17] -> [B].
        phase_has_orders = valid_mask.any(-1)

        def pack(maybe_tensor):
            if isinstance(maybe_tensor, torch.Tensor):
                return maybe_tensor[phase_has_orders]
            return maybe_tensor

        def unpack(tensor, fill_value):
            B = len(phase_has_orders)
            new_tensor = tensor.new_full((B,) + tensor.shape[1:], fill_value)
            new_tensor[phase_has_orders] = tensor
            return new_tensor

        enc_repeat = pack(enc_repeat)
        loc_idxs = pack(loc_idxs)
        cand_idxs = pack(cand_idxs)
        power = pack(power)
        temperature = pack(temperature)
        top_p = pack(top_p)
        teacher_force_orders = pack(teacher_force_orders)

        global_order_idxs, local_order_idxs, logits = policy_decoder(
            enc_repeat,
            loc_idxs,
            cand_idxs,
            temperature=temperature,
            top_p=top_p,
            teacher_force_orders=teacher_force_orders,
            power=power,
        )
        global_order_idxs = unpack(global_order_idxs, EOS_IDX)
        local_order_idxs = unpack(local_order_idxs, EOS_IDX)
        logits = unpack(logits, LOGIT_MASK_VAL)

    with timings("finish"):
        # reshape
        valid_mask = valid_mask.view(-1, NPOWERS, *valid_mask.shape[1:])
        global_order_idxs = global_order_idxs.view(-1, NPOWERS, *global_order_idxs.shape[1:])
        local_order_idxs = local_order_idxs.view(-1, NPOWERS, *local_order_idxs.shape[1:])
        logits = logits.view(-1, NPOWERS, *logits.shape[1:])

        # mask out garbage outputs
        eos_fill = torch.empty_like(global_order_idxs, requires_grad=False).fill_(EOS_IDX)
        global_order_idxs = torch.where(valid_mask, global_order_idxs, eos_fill)
        local_order_idxs = torch.where(valid_mask, local_order_idxs, eos_fill)

    if log_timings:
        logging.debug(f"Timings[model, B={enc.shape[0]}]: {timings}")

    return global_order_idxs, local_order_idxs, logits


def _apply_permute_powers(
    input_version: int,
    permute_powers_rand: np.random.Generator,
    x_board_state: torch.Tensor,
    x_prev_state: torch.Tensor,
    per_power_tensors: Tuple[Optional[torch.Tensor], ...],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...], torch.Tensor]:
    """Applies a random permutation of the 7 powers to each element of the batch.

    This function helps with data augmentation.
    This function is slightly fragile, in that it relies on specific knowledge of which x_* inputs
    to base_strategy_model need to have power-based permutations applied. If in the future we add more inputs,
    this function will also need to be updated. check_permute_powers() is used as a safeguard
    for this.

    Notably, we do NOT need to apply any permutation to the order-related parts of base_strategy_model's
    inputs, such as the order encoding on the input end, or the order idxs, teacher force orders,
    and other related values on the output end.

    All of these do not use the power itself as part of the input or output encoding.

    On the output end, the only thing we do need to inverse-permute is the values, which is handled
    by returning this matrix and having the caller apply it as appropriate to the values.

    Arguments:
    input_version: the board state input feature encoding version.
    permute_powers_rand: per-model random number generator for permutations
    x_board_state: x_board_state
    x_prev_state: x_prev_state
    per_power_tensors: Any number of tensors of shape [B,7,*] that encode per-power information.

    Returns:
    The same x_* tensors passed in, but permuted.
    Also returns the permutatation matrices applied to each batch element, with shape:
    [b, src, dst]
    indicating that for batch element b, power src was permuted to power dst.
    Batch multiplying on the left by this matrix
    (torch.matmul with this matrix as first argument) will undo the permutation.
    """

    device = x_board_state.device
    permutation = np.arange(len(POWERS))
    batch_size = x_board_state.size()[0]
    board_state_size = get_board_state_size(input_version)
    assert input_version >= 2  # Only version 2 features support power permutations
    assert len(x_board_state.size()) == 3
    assert x_board_state.size()[2] == board_state_size
    assert len(x_prev_state.size()) == 3
    assert x_prev_state.size()[2] == board_state_size

    for per_power_tensor in per_power_tensors:
        if per_power_tensor is not None:
            assert len(per_power_tensor.size()) >= 2
            assert per_power_tensor.size()[1] == len(POWERS)

    power_permutation = np.zeros((batch_size, len(POWERS)), dtype=np.int32)
    power_permutation_matrix = np.zeros((batch_size, len(POWERS), len(POWERS)), dtype=np.float32)
    for i in range(batch_size):
        permute_powers_rand.shuffle(permutation)
        power_permutation[i] = permutation
        for p in range(len(POWERS)):
            power_permutation_matrix[i, p, permutation[p]] = 1.0

    power_permutation_matrix = torch.tensor(power_permutation_matrix, device=device)

    # Shape (batch_size, board_state_size, board_state_size)
    batch_state_permutation_matrix = pydipcc.encode_board_state_pperm_matrices(
        power_permutation, input_version
    )
    batch_state_permutation_matrix = torch.tensor(batch_state_permutation_matrix, device=device)

    # (batch_size, NUM_LOCS, board_state_size)
    # = (batch_size, NUM_LOCS, board_state_size) * (batch_size, board_state_size, board_state_size)
    x_board_state = torch.matmul(x_board_state, batch_state_permutation_matrix)
    x_prev_state = torch.matmul(x_prev_state, batch_state_permutation_matrix)

    per_power_tensors_results = []
    for per_power_tensor in per_power_tensors:
        if per_power_tensor is not None:
            if len(per_power_tensor.shape) == 2:
                # (batch_size, 1, NUM_POWERS)
                # = (batch_size, 1, NUM_POWERS) * (batch_size, NUM_POWERS, NUM_POWERS)
                per_power_tensor = torch.matmul(
                    per_power_tensor.unsqueeze(1), power_permutation_matrix
                ).squeeze(1)
            elif len(per_power_tensor.shape) == 3:
                # (batch_size, NUM_POWERS, C)
                # = (batch_size, NUM_POWERS, NUM_POWERS)^-1 * (batch_size, NUM_POWERS, C)
                # And the inverse of a permutation matrix is its transpose.
                per_power_tensor = torch.matmul(
                    power_permutation_matrix.transpose(1, 2), per_power_tensor
                )
            else:
                assert "Per power tensors of dim >= 4 not implemented, add an implementation if needed"
        per_power_tensors_results.append(per_power_tensor)
    return (
        x_board_state,
        x_prev_state,
        tuple(per_power_tensors_results),
        power_permutation_matrix,
    )


def compute_alignments(loc_idxs, step, A):
    # -2 is used on adjustment phases to flag locations for builds, or locations
    # with armies to be disbanded
    alignments = torch.matmul(((loc_idxs == step) | (loc_idxs == -2)).to(A.dtype), A)
    alignments /= torch.sum(alignments, dim=1, keepdim=True) + 1e-5
    # alignments = torch.where(
    #     torch.isnan(alignments), torch.zeros_like(alignments), alignments
    # )

    return alignments


def repeat_interleave_if_tensor(x, reps, dim):
    if hasattr(x, "repeat_interleave"):
        return x.repeat_interleave(reps, dim=dim)
    return x


def apply_batch_repeat_interleave(tensors, batch_repeat_interleave):
    return tuple(
        repeat_interleave_if_tensor(tensor, batch_repeat_interleave, dim=0) for tensor in tensors
    )


class LSTMBaseStrategyModelDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        spatial_size,
        orders_vocab_size,
        lstm_size,
        order_emb_size,
        lstm_dropout,
        lstm_layers,
        master_alignments,
        use_simple_alignments=False,
        power_emb_size,
        featurize_output=False,
        relfeat_output=False,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.spatial_size = spatial_size
        self.order_emb_size = order_emb_size
        self.lstm_dropout = lstm_dropout
        self.power_emb_size = power_emb_size

        self.order_embedding = nn.Embedding(orders_vocab_size, order_emb_size)
        self.cand_embedding = PaddedEmbedding(orders_vocab_size, lstm_size, padding_idx=EOS_IDX)
        if power_emb_size > 0:
            self.power_lin = nn.Linear(len(POWERS), power_emb_size)
        else:
            self.power_lin = None

        self.lstm = nn.LSTM(
            2 * inter_emb_size + order_emb_size + power_emb_size,
            lstm_size,
            batch_first=True,
            num_layers=self.lstm_layers,
        )

        self.use_simple_alignments = use_simple_alignments

        if master_alignments is not None:
            self.master_alignments = nn.Parameter(master_alignments).requires_grad_(False)

        # Make the type checker understand what self.order_feats is
        if TYPE_CHECKING:
            self.order_feats = torch.tensor([])
        self.featurize_output = featurize_output
        if featurize_output:
            order_feats, srcs, dsts = compute_order_features()
            self.register_buffer("order_feats", order_feats)
            order_decoder_input_sz = self.order_feats.shape[1]
            self.order_feat_lin = nn.Linear(order_decoder_input_sz, order_emb_size)

            # this one has to stay as separate w, b
            # for backwards compatibility
            self.order_decoder_w = nn.Linear(order_decoder_input_sz, lstm_size)
            self.order_decoder_b = nn.Linear(order_decoder_input_sz, 1)

        # Make the type checker understand what self.order_srcs,order_dsts is
        if TYPE_CHECKING:
            self.order_srcs = torch.tensor([])
            self.order_dsts = torch.tensor([])
        self.relfeat_output = relfeat_output
        if relfeat_output:
            assert featurize_output, "Can't have relfeat_output without featurize_output (yet)"
            order_feats, srcs, dsts = compute_order_features()
            self.register_buffer("order_srcs", srcs)
            self.register_buffer("order_dsts", dsts)
            order_relfeat_input_sz = 2 * inter_emb_size

            self.order_relfeat_src_decoder_w = nn.Linear(order_relfeat_input_sz, lstm_size + 1)
            self.order_relfeat_dst_decoder_w = nn.Linear(order_relfeat_input_sz, lstm_size + 1)

            self.order_emb_relfeat_src_decoder_w = nn.Linear(order_emb_size, lstm_size + 1)
            self.order_emb_relfeat_dst_decoder_w = nn.Linear(order_emb_size, lstm_size + 1)

    def get_order_loc_feats(self, cand_order_locs, enc_w, out_w, enc_lin=None):
        B, L, D = enc_w.shape
        flat_order_locs = cand_order_locs.view(-1)
        valid = (flat_order_locs > 0).nonzero(as_tuple=False).squeeze(-1)
        # offsets of the order into the flattened enc_w tensor
        order_offsets = (
            cand_order_locs + torch.arange(B, device=cand_order_locs.device).view(B, 1) * L
        )
        valid_order_offsets = order_offsets.view(-1)[valid]
        valid_order_w = enc_w.view(-1, D)[valid_order_offsets]
        if enc_lin:
            valid_order_w = enc_lin(valid_order_w)
        out_w.view(-1, out_w.shape[-1]).index_add_(0, valid, valid_order_w)

    def forward(
        self,
        enc,
        loc_idxs,
        all_cand_idxs,
        power,
        temperature=1.0,
        top_p=1.0,
        teacher_force_orders=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timings = TimingCtx()
        with timings("dec.prep"):
            device = enc.device

            if (loc_idxs == -1).all():
                return (
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    torch.empty(*all_cand_idxs.shape[:2], dtype=torch.long, device=device).fill_(
                        EOS_IDX
                    ),
                    enc.new_zeros(*all_cand_idxs.shape),
                )

            # embedding for the last decoded order
            order_emb = enc.new_zeros(enc.shape[0], self.order_emb_size)

            # power embeddings for each lstm step
            assert tuple(power.shape) == tuple(
                all_cand_idxs.shape[:2]
            ), f"{power.shape} != {all_cand_idxs.shape[:2]}"

            all_power_embs = None
            if self.power_lin:
                # clamp power to avoid -1 padding crashing one_hot
                power_1h = torch.nn.functional.one_hot(power.long().clamp(0), len(POWERS)).to(
                    enc.dtype
                )
                all_power_embs = self.power_lin(power_1h)

            # return values: chosen order idxs, candidate idxs, and logits
            all_global_order_idxs = []
            all_local_order_idxs = []
            all_logits = []

            order_enc = enc.new_zeros(enc.shape[0], self.spatial_size, self.order_emb_size)

            self.lstm.flatten_parameters()
            hidden = (
                enc.new_zeros(self.lstm_layers, enc.shape[0], self.lstm_size),
                enc.new_zeros(self.lstm_layers, enc.shape[0], self.lstm_size),
            )

            # reuse same dropout weights for all steps
            dropout_in = (
                enc.new_zeros(
                    enc.shape[0], 1, enc.shape[2] + self.order_emb_size + self.power_emb_size,
                )
                .bernoulli_(1 - self.lstm_dropout)
                .div_(1 - self.lstm_dropout)
                .requires_grad_(False)
            )
            dropout_out = (
                enc.new_zeros(enc.shape[0], 1, self.lstm_size)
                .bernoulli_(1 - self.lstm_dropout)
                .div_(1 - self.lstm_dropout)
                .requires_grad_(False)
            )

            # find max # of valid cand idxs per step
            max_cand_per_step = (all_cand_idxs != EOS_IDX).sum(dim=2).max(dim=0).values  # [S]

            if self.relfeat_output:
                src_relfeat_w = self.order_relfeat_src_decoder_w(enc)
                dst_relfeat_w = self.order_relfeat_dst_decoder_w(enc)

        for step in range(all_cand_idxs.shape[1]):
            with timings("dec.power_emb"):
                power_emb = all_power_embs[:, step] if all_power_embs is not None else None

            with timings("dec.loc_enc"):
                num_cands = max_cand_per_step[step]
                cand_idxs = all_cand_idxs[:, step, :num_cands].long().contiguous()

                if self.use_simple_alignments:
                    # -2 is used on adjustment phases to flag locations for builds, or locations
                    # with armies to be disbanded
                    alignments = ((loc_idxs == step) | (loc_idxs == -2)).to(enc.dtype)
                else:
                    # do static attention
                    alignments = compute_alignments(loc_idxs, step, self.master_alignments)

                if self.spatial_size != alignments.size()[1]:
                    # pad (batch, 81) -> (batch, spatial_size)
                    assert self.spatial_size > alignments.size()[1]
                    assert len(alignments.size()) == 2
                    alignments = F.pad(alignments, (0, self.spatial_size - alignments.size()[1]))

                # print('alignments', alignments.mean(), alignments.std())
                loc_enc = torch.matmul(alignments.unsqueeze(1), enc).squeeze(1)

            with timings("dec.lstm"):
                input_list = [loc_enc, order_emb]
                if power_emb is not None:
                    input_list.append(power_emb)

                lstm_input = torch.cat(input_list, dim=1).unsqueeze(1)
                if self.training and self.lstm_dropout > 0.0:
                    lstm_input = lstm_input * dropout_in

                out, hidden = self.lstm(lstm_input, hidden)
                if self.training and self.lstm_dropout > 0.0:
                    out = out * dropout_out

                out = out.squeeze(1).unsqueeze(2)

            with timings("dec.cand_emb"):
                cand_emb = self.cand_embedding(cand_idxs)

            with timings("dec.logits"):
                logits = torch.matmul(cand_emb, out).squeeze(2)  # [B, <=469]

                if self.featurize_output:
                    # a) featurize based on one-hot features
                    cand_order_feats = self.order_feats[cand_idxs]
                    order_w = torch.cat(
                        (
                            self.order_decoder_w(cand_order_feats),
                            self.order_decoder_b(cand_order_feats),
                        ),
                        dim=-1,
                    )

                    if self.relfeat_output:
                        cand_srcs = self.order_srcs[cand_idxs]
                        cand_dsts = self.order_dsts[cand_idxs]

                        # b) featurize based on the src and dst encoder features
                        self.get_order_loc_feats(cand_srcs, src_relfeat_w, order_w)
                        self.get_order_loc_feats(cand_dsts, dst_relfeat_w, order_w)

                        # c) featurize based on the src and dst order embeddings
                        self.get_order_loc_feats(
                            cand_srcs,
                            order_enc,
                            order_w,
                            enc_lin=self.order_emb_relfeat_src_decoder_w,
                        )
                        self.get_order_loc_feats(
                            cand_dsts,
                            order_enc,
                            order_w,
                            enc_lin=self.order_emb_relfeat_dst_decoder_w,
                        )

                    # add some ones to out so that the last element of order_w is a bias
                    out_with_ones = torch.cat((out, out.new_ones(out.shape[0], 1, 1)), dim=1)
                    order_scores_featurized = torch.bmm(order_w, out_with_ones)
                    logits += order_scores_featurized.squeeze(-1)

            with timings("dec.invalid_mask"):
                # unmask where there are no actions or the sampling will crash. The
                # losses at these points will be masked out later, so this is safe.
                invalid_mask = ~(cand_idxs != EOS_IDX).any(dim=1)
                if invalid_mask.all():
                    # early exit
                    # logging.debug(f"Breaking at step {step} because no more orders to give")
                    for _step in range(step, all_cand_idxs.shape[1]):  # fill in garbage
                        all_global_order_idxs.append(
                            torch.empty(
                                all_cand_idxs.shape[0],
                                dtype=torch.long,
                                device=all_cand_idxs.device,
                            ).fill_(EOS_IDX)
                        )
                        all_local_order_idxs.append(
                            torch.empty(
                                all_cand_idxs.shape[0],
                                dtype=torch.long,
                                device=all_cand_idxs.device,
                            ).fill_(EOS_IDX)
                        )
                    break

                cand_mask = cand_idxs != EOS_IDX
                cand_mask[invalid_mask] = 1

            with timings("dec.logits_mask"):
                # make logits for invalid actions a large negative
                # We also deliberately call float() here, not to(enc.dtype), because even in fp16
                # once we have logits we want to cast up to fp32 for doing the masking, temperature,
                # and softmax.
                logits = torch.min(logits, cand_mask.float() * 1e9 + LOGIT_MASK_VAL)
                all_logits.append(logits)

            with timings("dec.logits_temp_top_p"):
                with torch.no_grad():
                    filtered_logits = logits.detach().clone()
                    top_p_min = top_p.min().item() if isinstance(top_p, torch.Tensor) else top_p
                    if top_p_min < 0.999:
                        filtered_logits.masked_fill_(
                            top_p_filtering(filtered_logits, top_p=top_p), -1e9
                        )
                    filtered_logits /= temperature

            with timings("dec.sample"):
                local_order_idxs = Categorical(logits=filtered_logits).sample()
                all_local_order_idxs.append(local_order_idxs)

            with timings("dec.order_idxs"):
                # skip clamp_and_mask since it is handled elsewhere and is slow
                global_order_idxs = local_order_idxs_to_global(
                    local_order_idxs, cand_idxs, clamp_and_mask=False
                )
                all_global_order_idxs.append(global_order_idxs)

            with timings("dec.order_emb"):
                sampled_order_input = global_order_idxs.masked_fill(
                    global_order_idxs == EOS_IDX, 0
                )
                if teacher_force_orders is None:
                    order_input = sampled_order_input
                else:
                    order_input = torch.where(
                        teacher_force_orders[:, step] == NO_ORDER_ID,
                        sampled_order_input,
                        teacher_force_orders[:, step],
                    )

                order_emb = self.order_embedding(order_input)
                if self.featurize_output:
                    order_emb += self.order_feat_lin(self.order_feats[order_input])

                if self.relfeat_output:
                    order_enc = order_enc + order_emb[:, None] * alignments[:, :, None]

        with timings("dec.fin"):
            stacked_global_order_idxs = torch.stack(all_global_order_idxs, dim=1)
            stacked_local_order_idxs = torch.stack(all_local_order_idxs, dim=1)
            stacked_logits = cat_pad_sequences(
                [x.unsqueeze(1) for x in all_logits],
                seq_dim=2,
                cat_dim=1,
                pad_value=LOGIT_MASK_VAL,
            )
            r = stacked_global_order_idxs, stacked_local_order_idxs, stacked_logits

        # logging.debug(f"Timings[dec, {enc.shape[0]}x{step}] {timings}")

        return r


def _pad_last_dims(tensor, partial_new_shape, pad_value):
    assert len(tensor.shape) >= len(partial_new_shape), (tensor.shape, partial_new_shape)
    new_shape = list(tensor.shape)[: len(tensor.shape) - len(partial_new_shape)] + list(
        partial_new_shape
    )
    new_tensor = tensor.new_full(new_shape, pad_value)
    new_tensor[[slice(None, D) for D in tensor.shape]].copy_(tensor)
    return new_tensor


class BaseStrategyModelEncoder(nn.Module):
    def __init__(
        self,
        *,
        board_state_size,
        prev_orders_size,
        inter_emb_size,  # 120
        num_blocks,  # 16
        A,  # 81x81
        dropout,
        residual_linear=False,
        merged_gnn=False,
        layerdrop=0,
    ):
        super().__init__()

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            BaseStrategyModelBlock(
                in_size=board_state_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                dropout=dropout,
                residual_linear=residual_linear,
            )
        )
        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                BaseStrategyModelBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    dropout=dropout,
                    residual_linear=residual_linear,
                )
            )

        if layerdrop > 1e-5:
            assert 0 < layerdrop <= 1.0, layerdrop
            self.layerdrop_rng = np.random.RandomState(0)
        else:
            self.layerdrop_rng = None
        self.layerdrop = layerdrop

        # prev orders blocks
        self.prev_orders_blocks = nn.ModuleList()
        self.prev_orders_blocks.append(
            BaseStrategyModelBlock(
                in_size=prev_orders_size,
                out_size=inter_emb_size,
                A=A,
                residual=False,
                dropout=dropout,
                residual_linear=residual_linear,
            )
        )
        for _ in range(num_blocks - 1):
            self.prev_orders_blocks.append(
                BaseStrategyModelBlock(
                    in_size=inter_emb_size,
                    out_size=inter_emb_size,
                    A=A,
                    residual=True,
                    dropout=dropout,
                    residual_linear=residual_linear,
                )
            )

        self.merged_gnn = merged_gnn
        if self.merged_gnn:
            self.merged_blocks = nn.ModuleList()
            for _ in range(num_blocks // 2):
                self.merged_blocks.append(
                    BaseStrategyModelBlock(
                        in_size=2 * inter_emb_size,
                        out_size=2 * inter_emb_size,
                        A=A,
                        residual=True,
                        dropout=dropout,
                        residual_linear=residual_linear,
                    )
                )

    def forward(self, x_bo, x_po):
        def apply_blocks_with_layerdrop(blocks, tensor):
            for i, block in enumerate(blocks):
                drop = (
                    i > 0
                    and self.training
                    and self.layerdrop_rng is not None
                    and self.layerdrop_rng.uniform() < self.layerdrop
                )
                if drop:
                    # To make distrubited happy we need to have grads for all params.
                    dummy = sum(w.sum() * 0 for w in block.parameters())
                    tensor = dummy + tensor
                else:
                    tensor = block(tensor)
            return tensor

        y_bo = apply_blocks_with_layerdrop(self.board_blocks, x_bo)
        y_po = apply_blocks_with_layerdrop(self.prev_orders_blocks, x_po)
        state_emb = torch.cat([y_bo, y_po], -1)

        if self.merged_gnn:
            state_emb = apply_blocks_with_layerdrop(self.merged_blocks, state_emb)
        return state_emb


class BaseStrategyModelBlock(nn.Module):
    def __init__(
        self, *, in_size, out_size, A, dropout, residual=True, residual_linear=False,
    ):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.dropout = nn.Dropout(dropout or 0.0)
        self.residual = residual
        self.residual_linear = residual_linear
        if residual_linear:
            self.residual_lin = nn.Linear(in_size, out_size)

    def forward(self, x):
        # Shape [batch_idx, location, channel]
        y = self.graph_conv(x)
        if self.residual_linear:
            y += self.residual_lin(x)
        y = self.batch_norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.residual:
            y += x
        return y


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        total_input_size,
        spatial_size,
        inter_emb_size,
        encoder_cfg: conf.conf_cfgs.Encoder.Transformer,
    ):
        super().__init__()
        # Torch's encoder implementation has the restriction that the input size must match
        # the output size and also be equal to the number of heads times the channels per head
        # in the attention layer. That means that the input size must be evenly divisible by
        # the number of heads.

        # Also due to historical accident, inter_emb_size is actually only half of the actual internal
        # number of channels, this is the reason for all the "* 2" everywhere.
        num_heads = encoder_cfg.num_heads
        channels_per_head = inter_emb_size * 2 // num_heads
        assert inter_emb_size * 2 == channels_per_head * num_heads

        self.initial_linear = nn.Linear(total_input_size, inter_emb_size * 2, bias=False)
        self.initial_positional_bias = nn.Parameter(he_init((spatial_size, inter_emb_size * 2)))
        self.blocks = nn.ModuleList()
        assert encoder_cfg.num_blocks is not None, "num_blocks is required"
        for _ in range(encoder_cfg.num_blocks):
            self.blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=inter_emb_size * 2,
                    nhead=encoder_cfg.num_heads,
                    dim_feedforward=encoder_cfg.ff_channels,
                    dropout=encoder_cfg.dropout,
                    activation=encoder_cfg.activation,
                )
            )

        layerdrop = encoder_cfg.layerdrop
        if layerdrop is not None and layerdrop > 1e-5:
            assert 0 < layerdrop <= 1.0, layerdrop
            self.layerdrop_rng = np.random.RandomState(0)
        else:
            self.layerdrop_rng = None
        self.layerdrop = layerdrop

    def forward(self, x_encoder_input):
        x = self.initial_linear(x_encoder_input)
        x = x + self.initial_positional_bias
        # x: Shape [batch_size, spatial_size, inter_emb_size*2]
        # But torch needs [spatial_size, batch_size, inter_emb_size*2]
        x = x.transpose(0, 1).contiguous()

        def apply_blocks_with_layerdrop(blocks, tensor):
            for i, block in enumerate(blocks):
                drop = (
                    self.training
                    and self.layerdrop_rng is not None
                    and self.layerdrop_rng.uniform() < self.layerdrop
                )
                if drop:
                    # To make distributed happy we need to have grads for all params.
                    dummy = sum(w.sum() * 0 for w in block.parameters())
                    tensor = dummy + tensor
                else:
                    tensor = block(tensor)
            return tensor

        x = apply_blocks_with_layerdrop(self.blocks, x)

        x = x.transpose(0, 1).contiguous()
        return x


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, A):
        super().__init__()
        """
        A -> (81, 81)
        """
        self.A = nn.Parameter(A).requires_grad_(False)
        self.W = nn.Parameter(he_init((len(self.A), in_size, out_size)))
        self.b = nn.Parameter(torch.zeros(1, 1, out_size))

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """

        x = x.transpose(0, 1)  # (b, N, in )               => (N, b, in )
        x = torch.matmul(x, self.W)  # (N, b, in) * (N, in, out) => (N, b, out)
        x = x.transpose(0, 1)  # (N, b, out)               => (b, N, out)
        x = torch.matmul(self.A, x)  # (b, N, N) * (b, N, out)   => (b, N, out)
        x += self.b

        return x


class ValueDecoder(nn.Module):
    def __init__(
        self,
        *,
        inter_emb_size,
        spatial_size,
        dropout,
        init_scale=1.0,
        softmax=False,
        activation="relu",
        use_weighted_pool: bool,
        extract_from_encoder: bool,
    ):
        super().__init__()
        assert (
            not use_weighted_pool or not extract_from_encoder
        ), "Cannot specify more than one of these at once"
        if extract_from_encoder:
            self.lin = nn.Linear(inter_emb_size * 2, 1)
        elif use_weighted_pool:
            self.softmax_weight_lin = nn.Linear(inter_emb_size * 2, 1)
            self.prelin = nn.Linear(inter_emb_size * 2, inter_emb_size)
            self.lin = nn.Linear(inter_emb_size, len(POWERS))
        else:
            emb_flat_size = spatial_size * inter_emb_size * 2
            self.prelin = nn.Linear(emb_flat_size, inter_emb_size)
            self.lin = nn.Linear(inter_emb_size, len(POWERS))

        self.extract_from_encoder = extract_from_encoder
        self.use_weighted_pool = use_weighted_pool

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.0)
        self.softmax = softmax
        self.activation = activation

        # scale down init
        torch.nn.init.xavier_normal_(self.lin.weight, gain=init_scale)
        bound = init_scale / (len(POWERS) ** 0.5)
        torch.nn.init.uniform_(self.lin.bias, -bound, bound)

    def _activation(self, y):
        if self.activation == "relu":
            y = F.relu(y)
        elif self.activation == "gelu":
            y = F.gelu(y)
        else:
            assert False, f"Unsupported value decoder activation: {self.activation}"
        return y

    def forward(self, enc):
        """Returns [B, 7] FloatTensor summing to 1 across dim=1
        Input enc should be shape [B, spatial_size, inter_emb_size*2] """
        B = enc.shape[0]

        if self.extract_from_encoder:
            # We assume the encoder spatial dimension is LOCS, POWERS, GLOBAL.
            assert enc.shape[1] == len(LOCS) + len(POWERS) + 1
            # Extract out the part corresponding to powers
            y = enc[:, len(LOCS) : len(LOCS) + len(POWERS), :]
            # Linear, then directly use this as logits, no dropout or activation
            # or anything else.
            y = self.lin(y).squeeze(2)
        elif self.use_weighted_pool:
            # Compute spatial weight
            weight = F.softmax(self.softmax_weight_lin(enc), dim=1)  # [B,spatial,1]
            # Compute weighted mean of encoder
            y = torch.bmm(weight.view(B, 1, -1), enc).squeeze(1)  # [B, inter_emb_size*2]
            y = self.prelin(y)
            y = self._activation(y)
            y = self.dropout(y)
            y = self.lin(y)
        else:
            y = enc.view(B, -1)
            y = self.prelin(y)
            y = self._activation(y)
            y = self.dropout(y)
            y = self.lin(y)

        if self.softmax:
            y = F.softmax(y, -1)
        else:
            y = y ** 2
            y = y / y.sum(dim=1, keepdim=True)
        return y


def compute_order_features():
    """Returns a [13k x D] tensor where each row contains (one-hot) features for one order in the vocabulary."""

    order_vocabulary = get_order_vocabulary()
    # assert order_vocabulary[0] == EOS_TOKEN
    # order_vocabulary = order_vocabulary[1:]  # we'll fix this up at the end
    order_split = [o.split() for o in order_vocabulary]

    # fixup strange stuff in the dataset
    for s in order_split:
        # fixup "A SIL S A PRU"
        if len(s) == 5 and s[2] == "S":
            s.append("H")
        # fixup "A SMY - ROM VIA"
        if len(s) == 5 and s[-1] == "VIA":
            s.pop()

    loc_idx = {loc: i for i, loc in enumerate(LOCS)}
    unit_idx = {"A": 0, "F": 1}
    order_type_idx = {
        t: i for i, t in enumerate(sorted(list(set([s[2] for s in order_split if len(s) > 2]))))
    }

    num_locs = len(loc_idx)
    feats = []
    srcs, dsts = [], []
    for o in order_split:
        u = o[3:] if len(o) >= 6 else o
        srcT = torch.zeros(num_locs)
        dstT = torch.zeros(num_locs)
        unitT = torch.zeros(len(unit_idx))
        orderT = torch.zeros(len(order_type_idx))
        underlyingT = torch.zeros(len(order_type_idx))

        if not o[2].startswith("B"):  # lets ignore the concatenated builds, they're tricky
            src_loc = loc_idx[u[1]]
            dst_loc = loc_idx[u[3]] if len(u) >= 4 else loc_idx[u[1]]
            srcT[src_loc] = 1
            dstT[dst_loc] = 1
            unitT[unit_idx[o[0]]] = 1
            orderT[order_type_idx[o[2]]] = 1
            underlyingT[order_type_idx[u[2]]] = 1
            srcs.append(src_loc)
            dsts.append(dst_loc)
        else:
            srcs.append(-1)
            dsts.append(-1)

        feats.append(torch.cat((srcT, dstT, unitT, orderT, underlyingT), dim=-1))

    feats = torch.stack(feats, dim=0)

    return feats, torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)


def compute_srcloc_idx_of_global_order_idx_plus_one():
    """Return a tensor mapping (global order idx+1) -> location idx of src of order.
    EOS_IDX+1 is mapped to a value larger than any location idx.
    """
    return torch.tensor([len(LOCS) + 1000] + LOC_IDX_OF_ORDER_IDX, dtype=torch.long)
