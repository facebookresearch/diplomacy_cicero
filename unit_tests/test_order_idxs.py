#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib
from typing import Union
import unittest

import numpy as np
import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import Action
from fairdiplomacy.models.consts import MAX_SEQ_LEN, LOCS
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.utils.order_idxs import (
    global_order_idxs_to_local,
    is_action_valid,
    local_order_idxs_to_global,
    action_strs_to_global_idxs,
    global_order_idxs_to_str,
    MAX_VALID_LEN,
    OrderIdxConversionException,
    canonicalize_action,
)
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from parlai_diplomacy.utils.game2seq.format_helpers.orders import (
    OrdersFlattener,
    OrdersUnflattener,
)


def load_game(path: Union[pathlib.Path, str]) -> pydipcc.Game:
    """Load a pydipcc game."""
    with open(path) as f:
        game = pydipcc.Game.from_json(f.read())
    return game


class TestLocalToGlobal(unittest.TestCase):
    def test_global_indices_to_local(self):
        game = pydipcc.Game()
        observations = FeatureEncoder().encode_inputs([game], input_version=1)
        x_possible_actions = observations["x_possible_actions"]
        assert list(x_possible_actions.shape) == [1, 7, MAX_SEQ_LEN, 469], x_possible_actions.shape

        # Static random.
        true_local_indices = (torch.arange(7 * MAX_SEQ_LEN) ** 2 % 469).view(1, 7, MAX_SEQ_LEN)

        global_indices = local_order_idxs_to_global(true_local_indices, x_possible_actions)

        true_local_indices[global_indices == -1] = -1

        local_indices = global_order_idxs_to_local(global_indices, x_possible_actions)

        # Test one index for start.
        self.assertEqual(true_local_indices[0, 0, 0], 0)
        self.assertEqual(global_indices[0, 0, 0], x_possible_actions[0, 0, 0, 0])
        self.assertEqual(local_indices[0, 0, 0], 0)

        # Go all in.
        np.testing.assert_array_equal(local_indices.numpy(), true_local_indices.numpy())

    def test_missing_idx(self):
        # only 100 and 101 are possible
        x_possible_actions = torch.full((MAX_SEQ_LEN, MAX_VALID_LEN), EOS_IDX)
        x_possible_actions[0, 0] = 100
        x_possible_actions[0, 1] = 101

        # we attempt to convert 90, which is not possible
        global_indices = torch.full((MAX_SEQ_LEN,), EOS_IDX)
        global_indices[0] = 90

        # with ignore_missing=True, results should be EOS_IDX
        local_indices = global_order_idxs_to_local(
            global_indices, x_possible_actions, ignore_missing=True
        )
        np.testing.assert_array_equal(local_indices.numpy(), EOS_IDX)

        # with ignore_missing=False, should throw
        with self.assertRaises(OrderIdxConversionException):
            local_indices = global_order_idxs_to_local(
                global_indices, x_possible_actions, ignore_missing=False
            )

    def test_negative_idx(self):
        game = pydipcc.Game()
        observations = FeatureEncoder().encode_inputs([game], input_version=1)
        x_possible_actions = observations["x_possible_actions"]
        local_idxs = torch.LongTensor(
            [EOS_IDX if x == EOS_IDX else 0 for x in x_possible_actions[0, 0, :, 0]]
        )
        assert EOS_IDX in local_idxs, "what we're testing for"
        global_idxs = local_order_idxs_to_global(local_idxs, x_possible_actions[0, 0, :, :])


class TestActionStrsToGlobalIdxs(unittest.TestCase):
    def test_action_strs_to_global_idxs_combined(self):
        # test that fn accepts combined or uncombined inputs
        uncombined = ["F STP/NC B", "A MOS B"]
        combined = [";".join(sorted(uncombined))]

        combined_idxs = action_strs_to_global_idxs(combined)
        uncombined_idxs = action_strs_to_global_idxs(uncombined)

        self.assertEqual(combined_idxs, uncombined_idxs)

        # test that global_order_idxs_to_str converts back
        reverted_strs = global_order_idxs_to_str(combined_idxs)
        self.assertEqual(reverted_strs, uncombined)

    def test_sort_by_loc(self):
        orders = ("A VIE - GAL", "F TRI - ALB", "A BUD - RUM")
        order_idxs = action_strs_to_global_idxs(orders, sort_by_loc=True)
        loc_idxs = [LOCS.index(order.split()[1]) for order in global_order_idxs_to_str(order_idxs)]
        self.assertEqual(loc_idxs, sorted(loc_idxs))

    def test_match_to_possible_orders(self):
        game = pydipcc.Game()

        # this should throw because "A MOS S F STP" is not in the vocab
        self.assertRaises(
            OrderIdxConversionException, action_strs_to_global_idxs, ["A MOS S F STP"]
        )

        # this should NOT throw
        order_idxs = action_strs_to_global_idxs(
            ["A MOS S F STP"],
            match_to_possible_orders=[
                d for ds in game.get_all_possible_orders().values() for d in ds
            ],
        )
        self.assertEqual(global_order_idxs_to_str(order_idxs), ["A MOS S F STP/SC"])

    def test_try_options(self):
        toidx = action_strs_to_global_idxs
        tostr = global_order_idxs_to_str

        # try_vias necessary for parsing a convoy-only movement
        # and try_vias has precedent over return_hold_for_invalid
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A SWE - LVN"], try_vias=False)
        self.assertEqual(tostr(toidx(["A SWE - LVN"], try_vias=True)), ["A SWE - LVN VIA"])
        self.assertEqual(
            tostr(toidx(["A SWE - LVN"], try_vias=False, return_hold_for_invalid=True)),
            ["A SWE H"],
        )
        self.assertEqual(
            tostr(toidx(["A SWE - LVN"], try_vias=True, return_hold_for_invalid=True)),
            ["A SWE - LVN VIA"],
        )

        # try_vias doesn't work for the exact same movement for a fleet, which can't via
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F SWE - LVN"], try_vias=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F SWE - LVN"], try_vias=True)
        self.assertEqual(
            tostr(toidx(["F SWE - LVN"], try_vias=False, return_hold_for_invalid=True)),
            ["F SWE H"],
        )
        self.assertEqual(
            tostr(toidx(["F SWE - LVN"], try_vias=True, return_hold_for_invalid=True)), ["F SWE H"]
        )

        # try_vias doesn't interfere with the optional via on optional-via movements
        self.assertEqual(tostr(toidx(["A BRE - PIC"], try_vias=False)), ["A BRE - PIC"])
        self.assertEqual(tostr(toidx(["A BRE - PIC"], try_vias=True)), ["A BRE - PIC"])
        self.assertEqual(tostr(toidx(["A BRE - PIC VIA"], try_vias=False)), ["A BRE - PIC VIA"])
        self.assertEqual(tostr(toidx(["A BRE - PIC VIA"], try_vias=True)), ["A BRE - PIC VIA"])

        # return_hold_for_invalid works when specified and not when not
        self.assertEqual(tostr(toidx(["F NTH - BLA"], return_hold_for_invalid=True)), ["F NTH H"])
        self.assertEqual(
            tostr(toidx(["F NTH is the best!"], return_hold_for_invalid=True)), ["F NTH H"]
        )
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F NTH - BLA"], return_hold_for_invalid=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F NTH is the best!"], return_hold_for_invalid=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F NTH"], return_hold_for_invalid=True)
        # return_hold_for_invalid doesn't help for armies in seas, since that's
        # not even a valid hold
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A NTH - BLA"], return_hold_for_invalid=True)

        # We don't have coastal-specific supports in pydipcc or our order idx
        # Since we don't, try_strip_coasts is needed to parse them and convert
        # them into whole-country supports.
        # And try_strip_coasts has precedence over return_hold_for_invalid
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F MAO S F WES - SPA/SC"])
        self.assertEqual(
            tostr(toidx(["F MAO S F WES - SPA/SC"], try_strip_coasts=True)),
            ["F MAO S F WES - SPA"],
        )
        self.assertEqual(
            tostr(
                toidx(
                    ["F MAO S F WES - SPA/SC"], try_strip_coasts=True, return_hold_for_invalid=True
                )
            ),
            ["F MAO S F WES - SPA"],
        )
        self.assertEqual(
            tostr(toidx(["F MAO S F WES - SPA/SC"], return_hold_for_invalid=True)), ["F MAO H"]
        )

        # Try strip coasts works for armies that try to move to a coast and has precedence
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A MAR - SPA/NC"])
        self.assertEqual(tostr(toidx(["A MAR - SPA/NC"], try_strip_coasts=True)), ["A MAR - SPA"])
        self.assertEqual(
            tostr(toidx(["A MAR - SPA/NC"], try_strip_coasts=True, return_hold_for_invalid=True)),
            ["A MAR - SPA"],
        )
        # Try strip coasts works for movements to a coast that doesn't exist and has precedence
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A MAR - SPA/EC"])
        self.assertEqual(tostr(toidx(["A MAR - SPA/EC"], try_strip_coasts=True)), ["A MAR - SPA"])
        self.assertEqual(
            tostr(toidx(["A MAR - SPA/EC"], try_strip_coasts=True, return_hold_for_invalid=True)),
            ["A MAR - SPA"],
        )
        # Doesn't help for a fleet since a fleet needs to specify the coast
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F MAR - SPA/EC"])
        with self.assertRaises(OrderIdxConversionException):
            toidx(["F MAR - SPA/EC"], try_strip_coasts=True)
        self.assertEqual(
            tostr(toidx(["F MAR - SPA/EC"], try_strip_coasts=True, return_hold_for_invalid=True)),
            ["F MAR H"],
        )

        # Try strip coasts works in conjunction with try_vias
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/NC"], try_vias=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/NC"], try_vias=False, try_strip_coasts=True)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/NC"], try_vias=True)
        self.assertEqual(
            tostr(toidx(["A PIE - SPA/NC"], try_vias=True, try_strip_coasts=True)),
            ["A PIE - SPA VIA"],
        )

        # Try strip coasts works in conjunction with try_vias even for a nonexisting coast
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/WC"], try_vias=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/WC"], try_vias=False, try_strip_coasts=True)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA/WC"], try_vias=True)
        self.assertEqual(
            tostr(toidx(["A PIE - SPA/WC"], try_vias=True, try_strip_coasts=True)),
            ["A PIE - SPA VIA"],
        )

        # Try strip coasts doesn't interfere with try_vias when no coast is specified to begin with
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA"], try_vias=False)
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A PIE - SPA"], try_vias=False, try_strip_coasts=True)
        self.assertEqual(tostr(toidx(["A PIE - SPA"], try_vias=True)), ["A PIE - SPA VIA"])
        self.assertEqual(
            tostr(toidx(["A PIE - SPA"], try_vias=True, try_strip_coasts=True)),
            ["A PIE - SPA VIA"],
        )

        # We don't have long convoys in our order idx even though they would be legal orders
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A NWY - GRE"])
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A NWY - GRE VIA"])
        with self.assertRaises(OrderIdxConversionException):
            toidx(["A NWY - GRE"], try_vias=True)
        self.assertEqual(tostr(toidx(["A NWY - GRE"], return_hold_for_invalid=True)), ["A NWY H"])
        self.assertEqual(
            tostr(toidx(["A NWY - GRE"], try_vias=True, return_hold_for_invalid=True)), ["A NWY H"]
        )


class TestCanonicalizeAction(unittest.TestCase):
    def test_canonicalize_action(self):
        # test that it puts things in loc-order

        a1: Action = ("F PAR - BUR", "A LON H", "A BUL S A RUM - SER")
        self.assertEqual(
            canonicalize_action(a1), ("A LON H", "F PAR - BUR", "A BUL S A RUM - SER")
        )

        # test that it doesn't blow up with a busted action, and still returns a deterministic order
        a2: Action = ("F PAR - BUR", "Z Z Z", "CRAZY!!", "#*(&@#")
        self.assertEqual(canonicalize_action(a2[::-1]), canonicalize_action(a2))

        # test that conversion back and forthbetween the Action and parlai representations is the identity

        ca = canonicalize_action(a1)  # canonical Action
        cp = OrdersFlattener(1).flatten_action(a1)  # canonical Parlai

        self.assertEqual(
            OrdersUnflattener(1).unflatten_action(OrdersFlattener(1).flatten_action(ca)), ca
        )
        self.assertEqual(
            OrdersFlattener(1).flatten_action(OrdersUnflattener(1).unflatten_action(cp)), cp
        )


class TestIsJointActionValid(unittest.TestCase):
    def test_movement_phase(self):
        game_path = "unit_tests/data/test_game_order_idxs.json"
        game_full = load_game(game_path)
        phase = "S1901M"
        game = game_full.rolled_back_to_phase_start(phase)
        self.assertTrue(
            is_action_valid(game, "ENGLAND", ("F EDI - NTH", "F LON - ENG", "A LVP - WAL"))
        )

        # missing order for a location
        self.assertFalse(is_action_valid(game, "ENGLAND", ("F EDI - NTH", "F LON - ENG")))
        # more orders than needed
        self.assertFalse(
            is_action_valid(
                game, "ENGLAND", ("F EDI - NTH", "F LON - ENG", "A LVP - WAL", "A LVP - NAO")
            )
        )
        # repeated order
        self.assertFalse(
            is_action_valid(game, "ENGLAND", ("F EDI - NTH", "A LVP - WAL", "A LVP - WAL"))
        )
        # out of vocab order
        self.assertFalse(
            is_action_valid(game, "ENGLAND", ("F EDI - NTH", "F LON - ENG", "A LVP - NWG"))
        )

    def test_retreat_phase(self):
        game_path = "unit_tests/data/test_game_order_idxs.json"
        game_full = load_game(game_path)
        phase = "S1902R"
        game = game_full.rolled_back_to_phase_start(phase)
        self.assertTrue(is_action_valid(game, "ENGLAND", ("F ENG R LON",)))

        # missing order for a location
        self.assertFalse(is_action_valid(game, "ENGLAND", ()))
        # more orders than needed
        self.assertFalse(is_action_valid(game, "ENGLAND", ("F ENG R IRI", "F ENG D")))
        # repeated order
        self.assertFalse(is_action_valid(game, "ENGLAND", ("F ENG D", "F ENG D")))
        # impossible order
        self.assertFalse(is_action_valid(game, "ENGLAND", ("F NTH D",)))

    def test_adjustment_phase(self):
        game_path = "unit_tests/data/test_game_order_idxs.json"
        game_full = load_game(game_path)
        phase = "W1905A"
        game = game_full.rolled_back_to_phase_start(phase)
        self.assertTrue(is_action_valid(game, "ITALY", ("F NAP B", "F ROM B")))

        # it is fine to build less
        self.assertTrue(is_action_valid(game, "ITALY", ("F NAP B",)))
        # cannot build more
        self.assertFalse(is_action_valid(game, "GERMANY", ("A KIE B", "A MUN B")))
        # wrong place to build
        self.assertFalse(is_action_valid(game, "GERMANY", ("A HOL B",)))
        # wrong number of destroy
        self.assertFalse(is_action_valid(game, "FRANCE", ("F IRI D", "F MAO D")))
