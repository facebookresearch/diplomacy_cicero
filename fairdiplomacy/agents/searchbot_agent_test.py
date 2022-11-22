#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict
from fairdiplomacy.agents.player import Player
from fairdiplomacy.typedefs import Action, Policy
import unittest

from conf import agents_cfgs
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game

DEFAULT_CFG: Dict[str, Any] = dict(
    model_path="MOCK",
    n_rollouts=10,
    device=-1,
    use_final_iter=0,
    rollouts_cfg=dict(max_rollout_length=0,),
    plausible_orders_cfg=dict(n_plausible_orders=10, batch_size=10, req_size=10,),
)


class CFRAgentTest(unittest.TestCase):
    def test_expected_values_sum_to_one_start(self):
        game = Game()
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        cfr_data = player.run_search(game).cfr_data
        assert cfr_data is not None
        vals = [cfr_data.avg_utility(p) for p in POWERS]
        self.assertAlmostEqual(sum(vals), 1.0, places=4)

    def test_get_orders_by_smoke(self):
        game = Game()
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        orders = player.get_orders(game)
        self.assertTrue(type(orders), Action)

    def test_get_policy_by_smoke(self):
        game = Game()
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        policy = player.run_search(game).get_population_policy()["FRANCE"]
        self.assertTrue(type(policy), Policy)
        self.assertNotEqual(policy, {})

    def test_powers_without_policies(self):
        game = Game()
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.process()
        game.process()
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**DEFAULT_CFG))
        player = Player(agent, power="FRANCE")
        # It's a build phase and only AUSTRIA has a move.
        plausible_orders_policy = player.get_plausible_orders_policy(game)
        self.assertEqual(plausible_orders_policy["FRANCE"], {tuple(): 1.0})
        self.assertEqual(plausible_orders_policy["AUSTRIA"], {("A BUD B",): 1.0})
        cfr_result = player.run_search(game)
        cfr_data = cfr_result.cfr_data
        assert cfr_data is not None
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
        cfg["plausible_orders_cfg"]["max_actions_units_ratio"] = 5
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**cfg))
        player = Player(agent, power="FRANCE")
        plausible_orders_policy = player.get_plausible_orders_policy(game)
        self.assertEqual(plausible_orders_policy["FRANCE"], {tuple(): 1.0})
        self.assertEqual(plausible_orders_policy["AUSTRIA"], {("A BUD B",): 1.0})

    def test_proposal_net_by_smoke(self):
        # Check that a network is re-used by default.
        agent = SearchBotAgent(agents_cfgs.SearchBotAgent(**DEFAULT_CFG))
        self.assertTrue(agent.base_strategy_model == agent.proposal_base_strategy_model)

        # Using MOCKv2 instead of MOCK to avoid reusing of the net.
        agent = SearchBotAgent(
            agents_cfgs.SearchBotAgent(**DEFAULT_CFG, rollout_model_path="MOCKV2")
        )
        player = Player(agent, power="FRANCE")
        self.assertTrue(agent.base_strategy_model != agent.proposal_base_strategy_model)

        game = Game()
        policy = player.run_search(game).get_population_policy()["FRANCE"]
        self.assertTrue(type(policy), Policy)
        self.assertNotEqual(policy, {})
