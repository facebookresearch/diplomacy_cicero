#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import inspect
import unittest

from fairdiplomacy.models.base_strategy_model.base_strategy_model import BaseStrategyModel
from fairdiplomacy.models.base_strategy_model.mock_base_strategy_model import MockBaseStrategyModel
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder


class MockBaseStrategyModelTest(unittest.TestCase):
    def test_mock_base_strategy_model_mimics_real_base_strategy_model(self):
        self.assertEqual(
            inspect.signature(MockBaseStrategyModel.__call__),
            inspect.signature(BaseStrategyModel.forward),
        )

    def test_mock_base_strategy_model_simple(self):
        game = Game()
        base_strategy_model = MockBaseStrategyModel()
        input_version = base_strategy_model.get_input_version()
        global_order_idxs, local_order_idxs, logits, final_sos = base_strategy_model(
            **FeatureEncoder().encode_inputs([game], input_version=input_version), temperature=1.0
        )
        assert global_order_idxs is not None
        assert local_order_idxs is not None
        assert logits is not None
        assert final_sos is not None

        self.assertEqual(tuple(global_order_idxs.shape), (1, 7, 4))
        self.assertEqual(tuple(local_order_idxs.shape), (1, 7, 4))
        self.assertEqual(tuple(logits.shape), (1, 7, 4, 469))
        self.assertEqual(tuple(final_sos.shape), (1, 7))

        # AUSTRIA has only 3 unsit and so last order must be -1.
        self.assertEqual(global_order_idxs[0, 0, -1], -1)
        self.assertEqual(local_order_idxs[0, 0, -1], -1)

        self.assertEqual(final_sos.flatten().tolist(), game.get_scores())

        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.process()
        _, _, _, final_sos = base_strategy_model(
            **FeatureEncoder().encode_inputs([game], input_version=input_version), temperature=1.0
        )
        assert final_sos is not None

        self.assertEqual(final_sos.flatten().tolist(), game.get_scores())

        game.process()
        global_order_idxs, local_order_idxs, logits, final_sos = base_strategy_model(
            **FeatureEncoder().encode_inputs([game], input_version=input_version), temperature=1.0
        )
        assert global_order_idxs is not None
        assert local_order_idxs is not None
        assert logits is not None
        assert final_sos is not None

        self.assertEqual(final_sos.flatten().tolist(), game.get_scores())

        # Everyone by AUSTRIA (power_id=0) has no orders
        self.assertTrue((global_order_idxs[0, 1:] == -1).all())
        self.assertTrue((local_order_idxs[0, 1:] == -1).all())

    def test_mock_base_strategy_model_temperature_0(self):
        game = Game()
        base_strategy_model = MockBaseStrategyModel()
        input_version = base_strategy_model.get_input_version()
        _, local_order_idxs, _, _ = base_strategy_model(
            **FeatureEncoder().encode_inputs([game], input_version=input_version),
            temperature=0.00001
        )
        self.assertTrue(((local_order_idxs == 0) | (local_order_idxs == -1)).all())

    def test_batch_repeat_interleave(self):
        game = Game()
        base_strategy_model = MockBaseStrategyModel()
        input_version = base_strategy_model.get_input_version()

        global_order_idxs, local_order_idxs, logits, final_sos = base_strategy_model(
            **FeatureEncoder().encode_inputs([game], input_version=input_version),
            temperature=1.0,
            batch_repeat_interleave=3
        )
        assert global_order_idxs is not None
        assert local_order_idxs is not None
        assert logits is not None
        assert final_sos is not None

        self.assertEqual(tuple(global_order_idxs.shape), (3, 7, 4))
        self.assertEqual(tuple(local_order_idxs.shape), (3, 7, 4))
        self.assertEqual(tuple(logits.shape), (3, 7, 4, 469))
        self.assertEqual(tuple(final_sos.shape), (3, 7))
