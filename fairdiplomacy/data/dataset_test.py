#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

import nest
import numpy as np
import torch

from conf import conf_cfgs
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import (
    maybe_augment_targets_inplace,
    shuffle_locations,
)
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.order_idxs import action_strs_to_global_idxs, global_order_idxs_to_local
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder


class LocationShuffleTest(unittest.TestCase):
    def testAllPowers(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs([Game()] * 6, input_version=1)
        print(nest.map(lambda x: x.shape, features))
        shuffle_locations(features)

    def testSinglePower(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs([Game()] * 6, input_version=1)
        # Removing power dimension and predenting everything is just a batch.
        for name in ["x_build_numbers", "x_loc_idxs", "x_possible_actions"]:
            features[name] = features[name][:, 0]
        print(nest.map(lambda x: x.shape, features))
        shuffle_locations(features)


class AugmentAllPowerTest(unittest.TestCase):
    def testAugmentCondition(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs_all_powers([Game()], input_version=1)
        power_orders = [
            "A VIE - GAL, F TRI - ALB, A BUD - SER".split(", "),
            "F EDI - NWG, F LON - NTH, A LVP - EDI".split(", "),
            "F BRE - MAO, A PAR - BUR, A MAR S A PAR - BUR".split(", "),
            "F KIE - DEN, A MUN - RUH, A BER - KIE".split(", "),
            "F NAP - ION, A ROM - APU, A VEN H".split(", "),
            "F STP/SC - BOT, A MOS - UKR, A WAR - GAL, F SEV - BLA".split(", "),
            "F ANK - BLA, A SMY - CON, A CON - BUL".split(", "),
        ]
        joint_action = sum(power_orders, [])

        action_ids = action_strs_to_global_idxs(joint_action, sort_by_loc=True)
        global_actions = torch.full([1, 7, 34], EOS_IDX, dtype=torch.long)
        global_actions[0, 0, : len(action_ids)] = torch.LongTensor(action_ids)
        y_actions = global_order_idxs_to_local(global_actions, features["x_possible_actions"],)

        batch = DataFields(**features, y_actions=y_actions)
        print(nest.map(lambda x: x.shape, batch))

        power_conditioning_cfg = conf_cfgs.PowerConditioning(
            prob=0.5, min_num_power=1, max_num_power=2
        )

        for i in range(100):
            np.random.seed(i)
            maybe_augment_targets_inplace(
                batch,
                single_chances=None,
                double_chances=None,
                power_conditioning=power_conditioning_cfg,
            )
