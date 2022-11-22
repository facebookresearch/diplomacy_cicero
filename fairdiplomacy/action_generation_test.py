#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy import action_generation
from fairdiplomacy.utils import thread_pool_encoding
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY_TO_IDX, LOCS

TEST_DATA = heyhi.PROJ_ROOT / "unit_tests/data"
GAME_WITH_COASTS = TEST_DATA / "game_fva_order_idx_with_coasts_test.json"


class TestLocalGenerator(unittest.TestCase):
    def testInitialGameSingleLocation(self):
        game = pydipcc.Game()
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR - BUR", "A MAR S A PAR - BUR", "F BRE H"]],
            selected_location="BRE",
        )
        # MAR is not close to BRE so the order shouldn't change.
        for action in actions:
            self.assertIn("A MAR S A PAR - BUR", action)
        # PAR must be coordinated.
        for action in actions:
            self.assertIn("A PAR - BUR", action)
        print(actions)
        # Six posibilities for BRE:
        # F BRE - ENG, F BRE - GAS, F BRE - MAO, F BRE - PIC, F BRE H, F BRE S F LON - ENG,
        self.assertEqual(len(actions), 6)

    def testInitialGameAllLocations(self):
        game = pydipcc.Game()
        actions = action_generation.generate_coordinated_local_modifications(
            game, power="FRANCE", actions=[["A PAR - BUR", "A MAR S A PAR - BUR", "F BRE H"]]
        )
        # Stability check.
        self.assertEqual(len(actions), 13)

    def testInitialGameAllLocationsWithHoles(self):
        game = pydipcc.Game()
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR - BUR", "A MAR S A PAR - BUR", "F BRE H"]],
            with_holes=True,
        )
        # Stability check.
        self.assertEqual(len(actions), 334)

    def testInitialGameAllLocationsWithLimit(self):
        game = pydipcc.Game()
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR - BUR", "A MAR S A PAR - BUR", "F BRE H"]],
            max_actions=10,
        )
        # Limiting to 10, expecting 10.
        self.assertEqual(len(actions), 10)

    def testInitialGameAllLocationsClose(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A MAR - BUR"])
        game.process()
        # Now all 3 units are in the Paris' cluster, and so we expect the same
        # number of modificaions as for non-local generator.
        actions = action_generation.generate_coordinated_local_modifications(
            game, power="FRANCE", actions=[["A PAR - PIC", "A BUR S A PAR - PIC", "F BRE H"]]
        )
        actions_no_limit = action_generation.get_all_possible_orders(game, power="FRANCE")
        self.assertEqual(set(actions), set(actions_no_limit))

    def testModificationOnUncoordinatedOrder(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A MAR - SPA"])
        game.process()
        # Units in BRE, PAR, SPA.
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR S F BRE - PIC", "A SPA S A PAR - GAS", "F BRE H"]],
            selected_location="SPA",
        )
        # No orders produced as the original order is not coordinated.
        self.assertEqual(set(actions), set())
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR S F BRE - PIC", "A SPA S A PAR - GAS", "F BRE H"]],
            selected_location="SPA",
            fix_uncoordinated_base=True,
        )
        # After fixing uncoordinated, some orders to be found
        self.assertNotEqual(set(actions), set())

    def testLocationsAreSorted(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A MAR - SPA"])
        game.process()

        x_possible_actions = (
            thread_pool_encoding.FeatureEncoder()
            .encode_inputs([game], input_version=1)["x_possible_actions"]
            .squeeze(0)
        ).tolist()[POWERS.index("FRANCE")]

        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A PAR S F BRE - PIC", "A SPA S A PAR - GAS", "F BRE H"]],
        )
        for action in actions:
            for order, possible_ids in zip(action, x_possible_actions):
                possible_ids = [x for x in possible_ids if x != -1]
                self.assertIn(ORDER_VOCABULARY_TO_IDX[order], possible_ids)

        # Changing order of actions doesn't change the order of the outputs
        actions = action_generation.generate_coordinated_local_modifications(
            game,
            power="FRANCE",
            actions=[["A SPA S A PAR - GAS", "A PAR S F BRE - PIC", "F BRE H"]],
        )
        for action in actions:
            for order, possible_ids in zip(action, x_possible_actions):
                possible_ids = [x for x in possible_ids if x != -1]
                self.assertIn(ORDER_VOCABULARY_TO_IDX[order], possible_ids)

    def testLocationsAreSortedComplex(self):
        with GAME_WITH_COASTS.open() as stream:
            game = pydipcc.Game.from_json(stream.read())
        power = "FRANCE"

        ACTION = (
            "F DEN - BAL",
            "A HOL - BEL",
            "F NWY S F STP/SC",
            "A BUR S A MUN",
            "A RUH S A MUN",
            "A KIE S A MUN",
            "A PAR - GAS",
            "F WES - TUN",
            "F MAR - SPA/SC",
            "A MUN H",
            "F STP/SC H",
            "F LYO S F TUS - TYS",
            "A PIE - VEN",
            "F ION - AEG",
            "F TUS - TYS",
            "A ROM - NAP",
        )

        x_possible_actions = (
            thread_pool_encoding.FeatureEncoder()
            .encode_inputs([game], input_version=1)["x_possible_actions"]
            .squeeze(0)
        ).tolist()[POWERS.index(power)]

        actions = action_generation.generate_coordinated_local_modifications(
            game, power=power, actions=[ACTION], max_actions=1,
        )
        for action in actions:
            print(action)
            for order, possible_ids in zip(action, x_possible_actions):
                possible_ids = [x for x in possible_ids if x != -1]
                self.assertIn(ORDER_VOCABULARY_TO_IDX[order], possible_ids)


class TestPerLocationGenerator(unittest.TestCase):
    def testLocationsAreSortedComplex(self):
        with GAME_WITH_COASTS.open() as stream:
            game = pydipcc.Game.from_json(stream.read())
        loc_to_orders = action_generation.get_power_per_loc_orders(game, "FRANCE")
        locs = [orders[0].split()[1] for loc, orders in loc_to_orders.items()]
        self.assertEqual(locs, sorted(locs, key=LOCS.index))
        for loc in locs:
            self.assertIn(loc, game.get_all_possible_orders())
