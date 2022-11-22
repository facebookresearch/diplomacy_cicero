#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
import heyhi.conf
import numpy as np

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent


class TestBQRE1PLambdas(unittest.TestCase):
    def test(self):
        cfg = heyhi.conf.load_config(
            heyhi.conf.CONF_ROOT / "common/agents/for_tests/bqre1p_20210821_rol0.prototxt",
            overrides=["bqre1p.base_searchbot_cfg.model_path=MOCKV2",],
        )
        agent = build_agent_from_cfg(cfg)
        assert isinstance(agent, BQRE1PAgent)
        qre_type2lambdas = {
            type_id: spec.qre_lambda for type_id, spec in agent.qre_type2spec.items()
        }
        assert qre_type2lambdas == {
            0: (3 ** 1) * np.float32(1e-6),
            1: (3 ** 2) * np.float32(1e-6),
            2: (3 ** 3) * np.float32(1e-6),
            3: (3 ** 4) * np.float32(1e-6),
            4: (3 ** 5) * np.float32(1e-6),
            5: (3 ** 6) * np.float32(1e-6),
            6: (3 ** 7) * np.float32(1e-6),
            7: (3 ** 8) * np.float32(1e-6),
            8: (3 ** 9) * np.float32(1e-6),
            9: (3 ** 10) * np.float32(1e-6),
        }

        # Also test the old deprecated way of specifying lambdas
        cfg = heyhi.conf.load_config(
            heyhi.conf.CONF_ROOT / "common/agents/for_tests/bqre1p_20210821_rol0.prototxt",
            overrides=[
                "bqre1p.base_searchbot_cfg.model_path=MOCKV2",
                "bqre1p.player_types=NULL",
                "bqre1p.lambda_min=1e-6",
                "bqre1p.lambda_multiplier=3",
            ],
        )
        agent = build_agent_from_cfg(cfg)
        assert isinstance(agent, BQRE1PAgent)
        qre_type2lambdas = {
            type_id: spec.qre_lambda for type_id, spec in agent.qre_type2spec.items()
        }
        assert qre_type2lambdas == {
            0: (3 ** 1) * np.float32(1e-6),
            1: (3 ** 2) * np.float32(1e-6),
            2: (3 ** 3) * np.float32(1e-6),
            3: (3 ** 4) * np.float32(1e-6),
            4: (3 ** 5) * np.float32(1e-6),
            5: (3 ** 6) * np.float32(1e-6),
            6: (3 ** 7) * np.float32(1e-6),
            7: (3 ** 8) * np.float32(1e-6),
            8: (3 ** 9) * np.float32(1e-6),
            9: (3 ** 10) * np.float32(1e-6),
        }
