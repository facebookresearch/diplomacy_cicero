#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Sequence, Optional

import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.typedefs import Action, Order
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN

MAX_INPUT_VERSION = pydipcc.max_input_version()
# Default value for backwards-compatibility for code that doesn't specify
DEFAULT_INPUT_VERSION = 1

# Number of feature channels for board state for base_strategy_model input
def get_board_state_size(input_version):
    return pydipcc.board_state_enc_width(input_version)


class FeatureEncoder:

    nothread_pool_singleton: Optional[pydipcc.ThreadPool] = None

    def __init__(self, *, num_threads: int = 0):
        """Initialize a FeatureEncoder

        Arguments:
          num_threads (optional int): If specified, uses a thread pool with this many threads.
        """
        if num_threads <= 0:
            self.thread_pool = self._get_nothread_pool()
        else:
            self.thread_pool = pydipcc.ThreadPool(
                num_threads, ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN
            )

    @classmethod
    def _get_nothread_pool(cls) -> pydipcc.ThreadPool:
        if cls.nothread_pool_singleton is None:
            cls.nothread_pool_singleton = pydipcc.ThreadPool(
                0, ORDER_VOCABULARY_TO_IDX, MAX_VALID_LEN
            )
        return cls.nothread_pool_singleton

    def encode_orders_single_strict(
        self, orders: Sequence[Order], input_version: int
    ) -> torch.Tensor:
        """Write a single sequence of orders as a feature tensor. The same format as
        used from x_prev_orders features.
        Any orders that fail to strictly match the exact strings in the order_vocabulary
        may SILENTLY be ignored!
        """
        return self.thread_pool.encode_orders_single_strict(orders, input_version)

    def encode_orders_single_tolerant(
        self, game: pydipcc.Game, orders: Sequence[Order], input_version: int
    ) -> torch.Tensor:
        """Same as encode_orders_single_strict, however will tolerate certain differences in whether
        supportees are coast-qualified or not, using the Game object to disambiguate.
        However, other invalidly formatted orders besides that may still be SILENTLY ignored!
        """
        return self.thread_pool.encode_orders_single_tolerant(game, orders, input_version)

    def encode_inputs(self, games: Sequence[pydipcc.Game], input_version: int) -> DataFields:
        """Encode a batch of inputs sufficient to predict policy and value.

        Arguments:
        games: The batch of game positions to encode.
        input_version (optional int): What version of the input features to a base_strategy_model to use.
            See dipcc/dipcc/cc/encoding.h
        """
        return DataFields(self.thread_pool.encode_inputs_multi(games, input_version))

    def encode_inputs_state_only(
        self, games: Sequence[pydipcc.Game], input_version: int
    ) -> DataFields:
        """Encode a batch of inputs sufficient to predict value only. Does not allow predicting policy.

        Arguments:
        games: The batch of game positions to encode.
        input_version (optional int): What version of the input features to a base_strategy_model to use.
            See dipcc/dipcc/cc/encoding.h
        """
        return DataFields(self.thread_pool.encode_inputs_state_only_multi(games, input_version))

    def encode_inputs_all_powers(
        self, games: Sequence[pydipcc.Game], input_version: int
    ) -> DataFields:
        """Encode a batch of inputs sufficient to predict policy and value, for a base_strategy_model that was
        trained to decode all power policies at once.

        Arguments:
        games: The batch of game positions to encode.
        input_version (optional int): What version of the input features to a base_strategy_model to use.
            See dipcc/dipcc/cc/encoding.h
        """
        return DataFields(self.thread_pool.encode_inputs_all_powers_multi(games, input_version))

    def decode_order_idxs(self, order_idxs):
        return self.thread_pool.decode_order_idxs(order_idxs)

    def decode_order_idxs_all_powers(
        self,
        order_idxs: torch.Tensor,
        x_in_adj_phase: torch.Tensor,
        x_power: torch.Tensor,
        batch_repeat_interleave: int,
    ):
        if x_in_adj_phase.dtype == torch.half:
            x_in_adj_phase = x_in_adj_phase.float()
        return self.thread_pool.decode_order_idxs_all_powers(
            order_idxs, x_in_adj_phase, x_power, batch_repeat_interleave
        )

    def process_multi(self, games: Sequence[pydipcc.Game]) -> None:
        self.thread_pool.process_multi(games)
