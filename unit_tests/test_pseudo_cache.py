#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from fairdiplomacy.agents.searchbot_agent import SearchBotAgentState
from fairdiplomacy.pydipcc import Game


class TestPseudoOrderCache(unittest.TestCase):
    def test_searchbot_incremental_pseudo_cache(self):
        agent_power = "AUSTRIA"
        state = SearchBotAgentState(agent_power)
        game = Game()

        self.assertIsNone(state.get_last_search_result(game))
        self.assertIsNone(state.get_last_pseudo_orders(game))

        a = {"AUSTRIA": ("A BUD - SER",)}
        state.update(game, agent_power, None, a)

        self.assertIsNone(state.get_last_search_result(game))
        a1 = state.get_last_pseudo_orders(game)
        self.assertEqual(a1, a)
        game.set_orders(agent_power, a[agent_power])
        game.process()  # -> F1901M
        self.assertIsNone(state.get_last_pseudo_orders(game))

        game.process()  # -> W1901A

        a2 = {"AUSTRIA": ("A BUD B",)}
        game.set_metadata("last_dialogue_phase", game.current_short_phase)
        state.update(game, agent_power, None, a2)

        self.assertEqual(state.get_last_pseudo_orders(game), a2)

        game_future = Game(game)
        game_future.set_orders(agent_power, a2[agent_power])
        game_future.process()  # -> S1902M
        a3 = {"AUSTRIA": ("A SER - ALB",)}
        state.update(game_future, agent_power, None, a3)

        # check that we keep the cache for the different dialogue and rollout phases
        self.assertEqual(state.get_last_pseudo_orders(game), a2)
        self.assertEqual(state.get_last_pseudo_orders(game_future), a3)

        # check that when dialogue phase is new, the cache is empty
        game.set_orders(agent_power, a2[agent_power])
        game.process()  # -> S1902M
        game.set_metadata("last_dialogue_phase", game.current_short_phase)
        self.assertIsNone(state.get_last_pseudo_orders(game))
