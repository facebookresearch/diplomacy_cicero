#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.game import POWERS
import numpy as np
import torch

import heyhi
from fairdiplomacy.models.consts import N_SCS
from fairdiplomacy import pydipcc
from fairdiplomacy.data.dataset import encode_power_actions
from fairdiplomacy.selfplay.search.search_utils import (
    batch_unpack_adjustment_phase_orders,
    pack_adjustment_phase_orders,
)
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder

GAME_PATH = heyhi.PROJ_ROOT / "unit_tests/data/game_no_press_from_selfplay_long.json"


def test_pack_unpack():
    with GAME_PATH.open() as stream:
        game_full = pydipcc.Game.from_json(stream.read())

    encoder = FeatureEncoder()

    all_phases = game_full.get_all_phase_names()
    for phase in all_phases:
        if phase.endswith("A"):
            game = game_full.rolled_back_to_phase_end(phase)
            power_actions = game.get_orders()
            game = game_full.rolled_back_to_phase_start(phase)
            features = encoder.encode_inputs_all_powers([game], pydipcc.max_input_version())

            power_encoded_orders_list = []
            for power_i, power in enumerate(POWERS):
                action = power_actions.get(power, tuple())
                print("Encoding", power, action)
                assert features["x_in_adj_phase"].item()
                power_i = POWERS.index(power)
                encoded_actions, is_valid = encode_power_actions(
                    tuple(action),
                    features["x_possible_actions"][0, power_i],
                    x_in_adj_phase=True,
                    max_seq_len=N_SCS,
                )
                power_encoded_orders_list.append(encoded_actions)
                assert is_valid

            power_encoded_orders = torch.stack(power_encoded_orders_list, 0).unsqueeze(0)
            packed = pack_adjustment_phase_orders(power_encoded_orders)
            unpacked = batch_unpack_adjustment_phase_orders(packed)
            np.testing.assert_array_equal(power_encoded_orders.numpy(), unpacked.numpy())
