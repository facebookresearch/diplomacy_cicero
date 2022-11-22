#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict, cast
import unittest

from fairdiplomacy.pydipcc import Game

from conf import agents_cfgs
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.player import Player
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Action

DEFAULT_CFG: Dict[str, Any] = dict(
    base_searchbot_cfg=dict(
        model_path="MOCK",
        n_rollouts=10,
        device=-1,
        use_final_iter=0,
        rollouts_cfg=dict(max_rollout_length=0,),
        plausible_orders_cfg=dict(n_plausible_orders=10, batch_size=10, req_size=10,),
        qre=dict(eta=10.0, target_pi="BLUEPRINT"),
    ),
    num_player_types=5,
    lambda_min=1e-30,
    lambda_multiplier=1e5,
    agent_type=1,
)


class BQREAgentTest(unittest.TestCase):
    def test_expected_values_sum_to_one_start(self):
        game = Game()
        agent = BQRE1PAgent(agents_cfgs.BQRE1PAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        brm_result = player.run_search(game)
        bcfr_data = brm_result.brm_data
        assert bcfr_data is not None
        for ptype, cfr_data in bcfr_data.type_cfr_data.items():
            vals = [cfr_data.avg_utility(p) for p in POWERS]
            self.assertAlmostEqual(sum(vals), 1.0, places=4)
        vals = [brm_result.avg_utility(p) for p in POWERS]
        self.assertAlmostEqual(sum(vals), 1.0, places=4)
        vals = [
            brm_result.avg_action_utility(p, list(brm_result.get_bp_policy()[p])[0])
            for p in POWERS
        ]
        self.assertAlmostEqual(sum(vals), 1.0, places=4)

    def test_get_orders_by_smoke(self):
        game = Game()
        agent = BQRE1PAgent(agents_cfgs.BQRE1PAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        orders = player.get_orders(game)
        self.assertTrue(type(orders), Action)

    def test_powers_without_policies(self):
        game = Game()
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.process()
        game.process()
        agent = BQRE1PAgent(agents_cfgs.BQRE1PAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        # It's a build phase and only AUSTRIA has a move.
        plausible_orders_policy = player.get_plausible_orders_policy(game)
        self.assertEqual(plausible_orders_policy["FRANCE"], {tuple(): 1.0})
        self.assertEqual(plausible_orders_policy["AUSTRIA"], {("A BUD B",): 1.0})
        bcfr_result = player.run_search(game)
        bcfr_data = bcfr_result.brm_data
        assert bcfr_data is not None
        for ptype, cfr_data in bcfr_data.type_cfr_data.items():
            vals = [cfr_data.avg_utility(p) for p in POWERS]
            self.assertAlmostEqual(sum(vals), 1.0, places=4)
        self.assertEqual(plausible_orders_policy["FRANCE"], {tuple(): 1.0})
        self.assertEqual(plausible_orders_policy["AUSTRIA"], {("A BUD B",): 1.0})

    def test_powers_without_policies_order_limit(self):
        # The same test as above, but use per-loc-limits.
        game = Game()
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.process()
        game.process()
        # It's a build phase and only AUSTRIA has a move.
        cfg = DEFAULT_CFG.copy()
        cfg["base_searchbot_cfg"]["plausible_orders_cfg"]["max_actions_units_ratio"] = 5
        agent = BQRE1PAgent(agents_cfgs.BQRE1PAgent(**cfg))
        player = Player(agent, power="FRANCE")
        plausible_orders_policy = player.get_plausible_orders_policy(game)
        self.assertEqual(plausible_orders_policy["FRANCE"], {tuple(): 1.0})
        self.assertEqual(plausible_orders_policy["AUSTRIA"], {("A BUD B",): 1.0})
