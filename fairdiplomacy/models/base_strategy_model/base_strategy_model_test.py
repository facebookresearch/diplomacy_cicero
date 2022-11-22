#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

import pytest
import torch.testing

import heyhi
from fairdiplomacy.models.base_strategy_model.load_model import new_model
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder


CFG_TRANSFORMER = heyhi.CONF_ROOT / "c02_sup_train/for_tests/sl_20211102_heavy_transfdec.prototxt"
CFG_LSTM = heyhi.CONF_ROOT / "c02_sup_train/for_tests/sl_202106_heavy.prototxt"
INPUT_VERSION = 2


class SmokeBaseStrategyModelTest(unittest.TestCase):
    def test_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(heyhi.load_config(CFG_TRANSFORMER).train)

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)
        base_strategy_model(**features, temperature=1.0)

    def test_featurizedin_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(
                CFG_TRANSFORMER, overrides=["transformer_decoder.featurize_input=1"]
            ).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)

    def test_featurizedout_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(
                CFG_TRANSFORMER, overrides=["transformer_decoder.featurize_output=1"]
            ).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)

    def test_explicit_location_input_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(
                CFG_TRANSFORMER, overrides=["transformer_decoder.explicit_location_input=1"]
            ).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)

    def test_posenc_input_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(
                CFG_TRANSFORMER, overrides=["transformer_decoder.positional_encoding=1"]
            ).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)

    def test_extra_normalization_transormer_decoder(self):
        game = Game()
        features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(
                CFG_TRANSFORMER,
                overrides=["transformer_decoder.transformer.extra_normalization=1"],
            ).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)

    def test_transormer_decoder_allpower(self):
        game = Game()
        features = FeatureEncoder().encode_inputs_all_powers([game], input_version=INPUT_VERSION)
        # Taking max to catch all bugs related with local-global indices.
        teacher_force_orders = features["x_possible_actions"].max(-1).values

        base_strategy_model = new_model(
            heyhi.load_config(CFG_TRANSFORMER, overrides=["all_powers=1"]).train
        )

        base_strategy_model(**features, temperature=1.0, teacher_force_orders=teacher_force_orders)


@pytest.mark.parametrize("cfg", [CFG_TRANSFORMER, CFG_LSTM])
def test_sampling_consistency(cfg):
    # logits during sampling and teacher forcing should match.
    game = Game()
    features = FeatureEncoder().encode_inputs([game], input_version=INPUT_VERSION)
    base_strategy_model = new_model(heyhi.load_config(cfg).train)
    base_strategy_model.eval()
    global_orders, _local_orders, logits, _ = base_strategy_model(**features, temperature=0.1)
    _, _, forced_logits, _ = base_strategy_model(
        **features, temperature=1.0, teacher_force_orders=global_orders.clamp_min(0)
    )
    print(logits.shape)  # [1, 7, 34, 469]
    # To simplify error message, first check for AUS and first timestamp.
    torch.testing.assert_allclose(logits[0, 0, 0], forced_logits[0, 0, 0])
    torch.testing.assert_allclose(logits, forced_logits)
