#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from fairdiplomacy import pydipcc
import fairdiplomacy.utils.orders


class TestUtilsOrders(unittest.TestCase):
    def test_utils_orders(self):
        game = pydipcc.Game()
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS H"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_hold(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS - PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_move(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS S A PIC - PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS S A PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS S A PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_hold(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS S A PIC - PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_support_move(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("F GAS C A BRE - MUN"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_convoy(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS R PAR"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_retreat(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS D"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("A GAS B"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_disband(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS H"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS S A PIC - PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS S A PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS D"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS R PAR"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("A GAS B"), True)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("F GAS C A BRE - MUN"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build("ABCD EFG"), False)
        self.assertEqual(fairdiplomacy.utils.orders.is_build(""), False)

        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS H"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS - PAR"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS - PAR VIA"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS R PAR"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS S F ENG - BRE"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A GAS S A BUR"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A BRE D"), "A")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("A BRE B"), "A")

        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS H"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS - PAR"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS - PAR VIA"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS R PAR"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS S F ENG - BRE"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F GAS S A BUR"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F BRE D"), "F")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_type("F BRE B"), "F")

        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A GAS H"), "GAS")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A GAS - PAR"), "GAS")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A GAS - PAR VIA"), "GAS")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A GAS R PAR"), "GAS")
        self.assertEqual(
            fairdiplomacy.utils.orders.get_unit_location("A GAS S F ENG - BRE"), "GAS"
        )
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A GAS S A BUR"), "GAS")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A BRE D"), "BRE")
        self.assertEqual(fairdiplomacy.utils.orders.get_unit_location("A BRE B"), "BRE")

        self.assertEqual(
            fairdiplomacy.utils.orders.get_move_or_retreat_destination("A GAS - PAR"), "PAR"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_move_or_retreat_destination("A GAS - PAR VIA"), "PAR"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_move_or_retreat_destination("A GAS R PAR"), "PAR"
        )

        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_type("F GAS S A MAR - SPA"), "A"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_type("F GAS S A MAR - SPA"), "A"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_type("F GAS S F SPA/SC"), "F"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_location("F GAS S F SPA/SC"), "SPA/SC"
        )

        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_location("F GAS S F SPA/SC"), "SPA/SC"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_location("F MAO S A POR"), "POR"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_location("F MAO S A POR - SPA"), "POR"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_supported_unit_destination("F MAO S A POR - SPA"), "SPA"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_convoyed_unit_location("F MAO C A POR - SPA"), "POR"
        )
        self.assertEqual(
            fairdiplomacy.utils.orders.get_convoyed_unit_destination("F MAO C A POR - SPA"), "SPA"
        )
