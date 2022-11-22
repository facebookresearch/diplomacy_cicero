#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import unittest
import torch

from fairdiplomacy.agents.base_strategy_model_wrapper import resample_duplicate_disbands_inplace
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.models.base_strategy_model.base_strategy_model import BaseStrategyModel
from fairdiplomacy.models.base_strategy_model.mock_base_strategy_model import MockBaseStrategyModel
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestResampleDuplicateDisbands(unittest.TestCase):
    def test_2020_09_01(self):
        X = torch.load(
            UNIT_TEST_DIR + "/data/resample_duplicate_disbands_inplace.debug.2020.09.01.pt",
            map_location="cpu",
        )
        model: BaseStrategyModel = MockBaseStrategyModel(input_version=3)  # type:ignore
        # Make up fake inputs for MockBaseStrategyModel that have the right shapes.
        inputs = FeatureEncoder().encode_inputs([Game()], input_version=3)
        resample_duplicate_disbands_inplace(
            X["order_idxs"], X["sampled_idxs"], X["logits"], inputs=inputs, model=model,
        )

        # for these inputs, there are only five valid disbands for England (1),
        # so check that all sampled idxs are valid (< 5)
        self.assertTrue((X["sampled_idxs"][0, 1] < 5).all())
