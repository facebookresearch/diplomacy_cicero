#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.agents.parlai_order_handler import filter_orders
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.timestamp import Timestamp
import os
import json
import pickle as pkl
import unittest
from typing import List

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Phase, Power


class TestParlaiOrderFiltering(unittest.TestCase):
    def test_parlai_order_filtering(self):
        game = Game()
        old_allowed_orders = game.get_all_possible_orders()

        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.set_orders("TURKEY", ["F ANK - CON", "A CON - SMY", "A SMY - ARM"])
        game.process()
        game.set_orders("TURKEY", ["F CON - BUL/EC"])
        game.process()
        game.process()

        allowed_orders = game.get_all_possible_orders()

        all_orders_to_try = []
        all_legal_orders = []
        for orders in old_allowed_orders.values():
            all_orders_to_try.extend(orders)
        for orders in allowed_orders.values():
            all_orders_to_try.extend(orders)
            all_legal_orders.extend(orders)
        all_orders_to_try = sorted(list(set(all_orders_to_try)))

        result = ""
        for i in range((len(all_orders_to_try) + 4) // 5):
            orders = all_orders_to_try[i * 5 : i * 5 + 4]
            good, bad = filter_orders(orders, all_legal_orders)

            # Check whether filter_orders mutated any orders
            good_idx = 0
            has_difference = False
            for order in orders:
                if order in bad:
                    continue
                if good[good_idx] != order:
                    has_difference = True
                good_idx += 1

            if has_difference:
                result += "***DIFFERENCE***\n"
            result += f"{orders} -> {good}\n"
            if len(bad) > 0:
                result += f"BAD = {bad}\n"

        print(result)
        self.assertEqual(
            result,
            """['A ARM - ANK', 'A ARM - SEV', 'A ARM - SMY', 'A ARM - SYR'] -> ['A ARM - ANK', 'A ARM - SEV', 'A ARM - SMY', 'A ARM - SYR']
['A ARM S A MOS - SEV', 'A ARM S A SMY', 'A ARM S A SMY - ANK', 'A ARM S A SMY - SYR'] -> ['A ARM S A MOS - SEV', 'A ARM S A SMY', 'A ARM S A SMY - ANK', 'A ARM S A SMY - SYR']
['A BER - KIE', 'A BER - MUN', 'A BER - PRU', 'A BER - SIL'] -> ['A BER - KIE', 'A BER - MUN', 'A BER - PRU', 'A BER - SIL']
['A BER S A MUN', 'A BER S A MUN - KIE', 'A BER S A MUN - SIL', 'A BER S A WAR - PRU'] -> ['A BER S A MUN', 'A BER S A MUN - KIE', 'A BER S A MUN - SIL', 'A BER S A WAR - PRU']
['A BER S F KIE', 'A BUD - GAL', 'A BUD - RUM', 'A BUD - SER'] -> ['A BER S F KIE']
BAD = ['A BUD - GAL', 'A BUD - RUM', 'A BUD - SER']
['A BUD - VIE', 'A BUD H', 'A BUD S A VEN - TRI', 'A BUD S A VIE'] -> []
BAD = ['A BUD - VIE', 'A BUD H', 'A BUD S A VEN - TRI', 'A BUD S A VIE']
['A BUD S A VIE - TRI', 'A BUD S A WAR - GAL', 'A BUD S F SEV - RUM', 'A BUD S F TRI'] -> []
BAD = ['A BUD S A VIE - TRI', 'A BUD S A WAR - GAL', 'A BUD S F SEV - RUM', 'A BUD S F TRI']
['A CON - BUL', 'A CON - SMY', 'A CON H', 'A CON S A SMY'] -> []
BAD = ['A CON - BUL', 'A CON - SMY', 'A CON H', 'A CON S A SMY']
['A CON S F ANK', 'A LVP - CLY', 'A LVP - EDI', 'A LVP - WAL'] -> ['A LVP - CLY', 'A LVP - EDI', 'A LVP - WAL']
BAD = ['A CON S F ANK']
['A LVP H', 'A LVP S F EDI', 'A LVP S F EDI - CLY', 'A LVP S F EDI - YOR'] -> ['A LVP H', 'A LVP S F EDI', 'A LVP S F EDI - CLY', 'A LVP S F EDI - YOR']
['A LVP S F LON - YOR', 'A MAR - BUR', 'A MAR - GAS', 'A MAR - PIE'] -> ['A LVP S F LON - YOR', 'A MAR - BUR', 'A MAR - GAS', 'A MAR - PIE']
['A MAR H', 'A MAR S A MUN - BUR', 'A MAR S A PAR - BUR', 'A MAR S A PAR - GAS'] -> ['A MAR H', 'A MAR S A MUN - BUR', 'A MAR S A PAR - BUR', 'A MAR S A PAR - GAS']
***DIFFERENCE***
['A MAR S F BRE - GAS', 'A MAR S F MAO - GAS', 'A MAR S F MAO - SPA', 'A MAR S F MAO - SPA/NC'] -> ['A MAR S F MAO - GAS', 'A MAR S F MAO - SPA', 'A MAR S F MAO - SPA']
BAD = ['A MAR S F BRE - GAS']
['A MOS - LVN', 'A MOS - SEV', 'A MOS - STP', 'A MOS - UKR'] -> ['A MOS - LVN', 'A MOS - SEV', 'A MOS - STP', 'A MOS - UKR']
['A MOS H', 'A MOS S A ARM - SEV', 'A MOS S A WAR', 'A MOS S A WAR - LVN'] -> ['A MOS H', 'A MOS S A ARM - SEV', 'A MOS S A WAR', 'A MOS S A WAR - LVN']
***DIFFERENCE***
['A MOS S F SEV', 'A MOS S F STP', 'A MOS S F STP/SC', 'A MOS S F STP/SC - LVN'] -> ['A MOS S F SEV', 'A MOS S F STP/SC', 'A MOS S F STP/SC', 'A MOS S F STP/SC - LVN']
['A MUN - BOH', 'A MUN - BUR', 'A MUN - KIE', 'A MUN - RUH'] -> ['A MUN - BOH', 'A MUN - BUR', 'A MUN - KIE', 'A MUN - RUH']
['A MUN - TYR', 'A MUN H', 'A MUN S A BER', 'A MUN S A BER - KIE'] -> ['A MUN - TYR', 'A MUN H', 'A MUN S A BER', 'A MUN S A BER - KIE']
['A MUN S A MAR - BUR', 'A MUN S A PAR - BUR', 'A MUN S A VEN - TYR', 'A MUN S A VIE - BOH'] -> ['A MUN S A MAR - BUR', 'A MUN S A PAR - BUR', 'A MUN S A VEN - TYR', 'A MUN S A VIE - BOH']
['A MUN S A WAR - SIL', 'A MUN S F KIE', 'A MUN S F KIE - BER', 'A PAR - BRE'] -> ['A MUN S A WAR - SIL', 'A MUN S F KIE', 'A MUN S F KIE - BER', 'A PAR - BRE']
['A PAR - GAS', 'A PAR - PIC', 'A PAR H', 'A PAR S A MAR - BUR'] -> ['A PAR - GAS', 'A PAR - PIC', 'A PAR H', 'A PAR S A MAR - BUR']
['A PAR S A MUN - BUR', 'A PAR S F BRE', 'A PAR S F BRE - GAS', 'A PAR S F BRE - PIC'] -> ['A PAR S A MUN - BUR']
BAD = ['A PAR S F BRE', 'A PAR S F BRE - GAS', 'A PAR S F BRE - PIC']
['A PAR S F MAO - GAS', 'A ROM - APU', 'A ROM - NAP', 'A ROM - TUS'] -> ['A PAR S F MAO - GAS', 'A ROM - APU', 'A ROM - NAP', 'A ROM - TUS']
['A ROM H', 'A ROM S A VEN', 'A ROM S A VEN - APU', 'A ROM S A VEN - TUS'] -> ['A ROM H', 'A ROM S A VEN', 'A ROM S A VEN - APU', 'A ROM S A VEN - TUS']
['A ROM S F NAP - APU', 'A ROM S F TRI - VEN', 'A SER - ALB', 'A SER - BUD'] -> ['A ROM S F NAP - APU', 'A ROM S F TRI - VEN', 'A SER - ALB', 'A SER - BUD']
['A SER - GRE', 'A SER - RUM', 'A SER - TRI', 'A SER H'] -> ['A SER - GRE', 'A SER - RUM', 'A SER - TRI', 'A SER H']
***DIFFERENCE***
['A SER S A VIE - BUD', 'A SER S A VIE - TRI', 'A SER S F BUL', 'A SER S F BUL/EC'] -> ['A SER S A VIE - BUD', 'A SER S A VIE - TRI', 'A SER S F BUL/EC', 'A SER S F BUL/EC']
['A SER S F SEV - RUM', 'A SER S F TRI', 'A SER S F TRI - ALB', 'A SMY - ANK'] -> ['A SER S F SEV - RUM', 'A SER S F TRI', 'A SER S F TRI - ALB', 'A SMY - ANK']
['A SMY - CON', 'A SMY - SYR', 'A SMY H', 'A SMY S A ARM'] -> ['A SMY - CON', 'A SMY - SYR', 'A SMY H', 'A SMY S A ARM']
['A SMY S A ARM - SYR', 'A SMY S A CON', 'A SMY S A CON - ANK', 'A SMY S F ANK'] -> ['A SMY S A ARM - SYR']
BAD = ['A SMY S A CON', 'A SMY S A CON - ANK', 'A SMY S F ANK']
['A SMY S F ANK - CON', 'A SMY S F BUL/EC - CON', 'A SMY S F SEV - ARM', 'A VEN - APU'] -> ['A SMY S F BUL/EC - CON', 'A SMY S F SEV - ARM', 'A VEN - APU']
BAD = ['A SMY S F ANK - CON']
['A VEN - ROM', 'A VEN - TRI', 'A VEN - TUS', 'A VEN - TYR'] -> ['A VEN - ROM', 'A VEN - TRI', 'A VEN - TUS', 'A VEN - TYR']
['A VEN S A BUD - TRI', 'A VEN S A MAR - PIE', 'A VEN S A MUN - TYR', 'A VEN S A ROM'] -> ['A VEN S A MAR - PIE', 'A VEN S A MUN - TYR', 'A VEN S A ROM']
BAD = ['A VEN S A BUD - TRI']
['A VEN S A ROM - TUS', 'A VEN S A SER - TRI', 'A VEN S A VIE - TRI', 'A VEN S A VIE - TYR'] -> ['A VEN S A ROM - TUS', 'A VEN S A SER - TRI', 'A VEN S A VIE - TRI', 'A VEN S A VIE - TYR']
['A VEN S F NAP - ROM', 'A VEN S F TRI', 'A VIE - BOH', 'A VIE - BUD'] -> ['A VEN S F NAP - ROM', 'A VEN S F TRI', 'A VIE - BOH', 'A VIE - BUD']
['A VIE - TRI', 'A VIE - TYR', 'A VIE H', 'A VIE S A BUD'] -> ['A VIE - TRI', 'A VIE - TYR', 'A VIE H']
BAD = ['A VIE S A BUD']
['A VIE S A BUD - TRI', 'A VIE S A MUN - BOH', 'A VIE S A MUN - TYR', 'A VIE S A SER - BUD'] -> ['A VIE S A MUN - BOH', 'A VIE S A MUN - TYR', 'A VIE S A SER - BUD']
BAD = ['A VIE S A BUD - TRI']
['A VIE S A VEN - TRI', 'A VIE S A VEN - TYR', 'A VIE S A WAR - GAL', 'A VIE S F TRI'] -> ['A VIE S A VEN - TRI', 'A VIE S A VEN - TYR', 'A VIE S A WAR - GAL', 'A VIE S F TRI']
['A WAR - LVN', 'A WAR - MOS', 'A WAR - PRU', 'A WAR - SIL'] -> ['A WAR - LVN', 'A WAR - MOS', 'A WAR - PRU', 'A WAR - SIL']
['A WAR H', 'A WAR S A BER - PRU', 'A WAR S A BER - SIL', 'A WAR S A BUD - GAL'] -> ['A WAR H', 'A WAR S A BER - PRU', 'A WAR S A BER - SIL']
BAD = ['A WAR S A BUD - GAL']
['A WAR S A MOS - LVN', 'A WAR S A MOS - UKR', 'A WAR S A MUN - SIL', 'A WAR S A VIE - GAL'] -> ['A WAR S A MOS - LVN', 'A WAR S A MOS - UKR', 'A WAR S A MUN - SIL', 'A WAR S A VIE - GAL']
['F ANK - ARM', 'F ANK - BLA', 'F ANK - CON', 'F ANK H'] -> []
BAD = ['F ANK - ARM', 'F ANK - BLA', 'F ANK - CON', 'F ANK H']
['F ANK S A SMY - ARM', 'F ANK S A SMY - CON', 'F ANK S F SEV - ARM', 'F ANK S F SEV - BLA'] -> []
BAD = ['F ANK S A SMY - ARM', 'F ANK S A SMY - CON', 'F ANK S F SEV - ARM', 'F ANK S F SEV - BLA']
['F BRE - GAS', 'F BRE - MAO', 'F BRE - PIC', 'F BRE H'] -> []
BAD = ['F BRE - GAS', 'F BRE - MAO', 'F BRE - PIC', 'F BRE H']
['F BRE S A PAR - GAS', 'F BRE S A PAR - PIC', 'F BRE S F LON - ENG', 'F BUL/EC - BLA'] -> ['F BUL/EC - BLA']
BAD = ['F BRE S A PAR - GAS', 'F BRE S A PAR - PIC', 'F BRE S F LON - ENG']
['F BUL/EC - RUM', 'F BUL/EC H', 'F BUL/EC S A SER - RUM', 'F BUL/EC S A SMY - CON'] -> ['F BUL/EC - RUM', 'F BUL/EC H', 'F BUL/EC S A SER - RUM', 'F BUL/EC S A SMY - CON']
['F BUL/EC S F SEV - RUM', 'F EDI - CLY', 'F EDI - NTH', 'F EDI - NWG'] -> ['F BUL/EC S F SEV - RUM', 'F EDI - CLY', 'F EDI - NTH', 'F EDI - NWG']
['F EDI H', 'F EDI S A LVP - CLY', 'F EDI S A LVP - YOR', 'F EDI S F LON - NTH'] -> ['F EDI H', 'F EDI S A LVP - CLY', 'F EDI S A LVP - YOR', 'F EDI S F LON - NTH']
['F KIE - BAL', 'F KIE - BER', 'F KIE - DEN', 'F KIE - HEL'] -> ['F KIE - BAL', 'F KIE - BER', 'F KIE - DEN', 'F KIE - HEL']
['F KIE H', 'F KIE S A BER', 'F KIE S A MUN - BER', 'F LON - ENG'] -> ['F KIE H', 'F KIE S A BER', 'F KIE S A MUN - BER', 'F LON - ENG']
['F LON - WAL', 'F LON - YOR', 'F LON H', 'F LON S A LVP - WAL'] -> ['F LON - WAL', 'F LON - YOR', 'F LON H', 'F LON S A LVP - WAL']
['F LON S F BRE - ENG', 'F LON S F EDI - NTH', 'F LON S F EDI - YOR', 'F LON S F MAO - ENG'] -> ['F LON S F EDI - NTH', 'F LON S F EDI - YOR', 'F LON S F MAO - ENG']
BAD = ['F LON S F BRE - ENG']
['F MAO - ENG', 'F MAO - GAS', 'F MAO - IRI', 'F MAO - NAF'] -> ['F MAO - ENG', 'F MAO - GAS', 'F MAO - IRI', 'F MAO - NAF']
['F MAO - POR', 'F MAO - SPA/NC', 'F MAO - SPA/SC', 'F MAO - WES'] -> ['F MAO - POR', 'F MAO - SPA/NC', 'F MAO - SPA/SC', 'F MAO - WES']
['F MAO S A MAR - GAS', 'F MAO S A MAR - SPA', 'F MAO S A PAR - BRE', 'F MAO S A PAR - GAS'] -> ['F MAO S A MAR - GAS', 'F MAO S A MAR - SPA', 'F MAO S A PAR - BRE', 'F MAO S A PAR - GAS']
['F NAP - APU', 'F NAP - ION', 'F NAP - ROM', 'F NAP - TYS'] -> ['F NAP - APU', 'F NAP - ION', 'F NAP - ROM', 'F NAP - TYS']
['F NAP S A ROM', 'F NAP S A ROM - APU', 'F NAP S A VEN - APU', 'F NAP S A VEN - ROM'] -> ['F NAP S A ROM', 'F NAP S A ROM - APU', 'F NAP S A VEN - APU', 'F NAP S A VEN - ROM']
['F SEV - BLA', 'F SEV - RUM', 'F SEV H', 'F SEV S A ARM'] -> ['F SEV - BLA', 'F SEV - RUM', 'F SEV H', 'F SEV S A ARM']
['F SEV S A SER - RUM', 'F SEV S A SMY - ARM', 'F SEV S F ANK - ARM', 'F SEV S F ANK - BLA'] -> ['F SEV S A SER - RUM', 'F SEV S A SMY - ARM']
BAD = ['F SEV S F ANK - ARM', 'F SEV S F ANK - BLA']
['F SEV S F BUL/EC - RUM', 'F STP/SC - BOT', 'F STP/SC - FIN', 'F STP/SC - LVN'] -> ['F SEV S F BUL/EC - RUM', 'F STP/SC - BOT', 'F STP/SC - FIN', 'F STP/SC - LVN']
['F STP/SC S A MOS - LVN', 'F STP/SC S A WAR - LVN', 'F TRI - ADR', 'F TRI - ALB'] -> ['F STP/SC S A MOS - LVN', 'F STP/SC S A WAR - LVN', 'F TRI - ADR', 'F TRI - ALB']
['F TRI H', 'F TRI S A ROM - VEN', 'F TRI S A SER - ALB', 'F TRI S A VEN'] -> ['F TRI H', 'F TRI S A ROM - VEN', 'F TRI S A SER - ALB', 'F TRI S A VEN']
""",
        )
