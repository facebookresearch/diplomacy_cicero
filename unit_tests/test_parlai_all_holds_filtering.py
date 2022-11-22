#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import unittest
from typing import List

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import JointAction, Power

from parlai_diplomacy.utils.game2seq.factory import get_input_format, sequence_formatter_factory
from parlai_diplomacy.utils.game2seq.typing import Metadata


UNIT_TEST_DIR = os.path.dirname(__file__)
# for purposes of this test, we always look at England and phase W1901A
ENGLAND = "ENGLAND"
RUSSIA = "RUSSIA"
PHASE = "S1901M"
UNITS = {
    "AUSTRIA": ["A VIE", "F TRI", "A BUD"],
    "ENGLAND": ["F EDI", "F LON", "A LVP"],
    "FRANCE": ["F BRE", "A PAR", "A MAR"],
    "GERMANY": ["F KIE", "A MUN", "A BER"],
    "ITALY": ["F NAP", "A ROM", "A VEN"],
    "RUSSIA": ["F STP/SC", "A MOS", "A WAR", "F SEV"],
    "TURKEY": ["F ANK", "A SMY", "A CON"],
}
GROUND_TRUTH_ORDERS = {
    "AUSTRIA": ["A VIE - GAL", "F TRI H", "A BUD - RUM"],
    "ENGLAND": ["F EDI - NTH", "A LVP - YOR", "F LON - ENG"],
    "FRANCE": ["F BRE - MAO", "A PAR - BUR", "A MAR S A PAR - BUR"],
    "GERMANY": ["F KIE - DEN", "A BER - KIE", "A MUN - BUR"],
    "ITALY": ["F NAP - ION", "A ROM - APU", "A VEN H"],
    "RUSSIA": ["F SEV - RUM", "A WAR - UKR", "A MOS - SEV", "F STP/SC - BOT"],
    "TURKEY": ["A CON - BUL", "A SMY - CON", "F ANK - BLA"],
}


def load_game(last_phase_orders: JointAction):
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")
    with open(fle, "r") as f:
        game_object = Game.from_json(f.read())

    # rollback to movement phase
    game_object = game_object.rolled_back_to_phase_start(PHASE)

    # set new orders
    for pwr, orders in last_phase_orders.items():
        game_object.set_orders(pwr, orders)

    game_object.process()

    return game_object


class TestAllHoldsFiltering(unittest.TestCase):
    def _get_sequences(self, orders: JointAction, task_name: str, metadata: Metadata):
        game = load_game(orders)
        formatter = sequence_formatter_factory(task_name, 1)
        input_format = get_input_format(task_name)
        return formatter.change_format(game, input_format, metadata)

    def _get_all_holds_orders(self, powers: List[Power]) -> JointAction:
        orders = {}
        for power in POWERS:
            if power in powers:
                orders[power] = [f"{unit} H" for unit in UNITS[power]]
            else:
                orders[power] = GROUND_TRUTH_ORDERS[power]

        return orders

    def _get_metadata(self):
        return {
            "power_metadata": {},
            "filter_all_holds": True,
            "opt": {},
        }

    def test_single_order(self):
        metadata = self._get_metadata()

        orders = self._get_all_holds_orders([ENGLAND])
        seq = self._get_sequences(orders, "state_order_chunk", metadata)[PHASE][ENGLAND]
        assert "output" not in seq  # all holds are filtered

        orders = self._get_all_holds_orders([RUSSIA])
        seq = self._get_sequences(orders, "state_order_chunk", metadata)[PHASE][ENGLAND]
        assert seq["output"] == "A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]"

    def test_all_orders(self):
        metadata = self._get_metadata()

        orders = self._get_all_holds_orders([ENGLAND])
        seq = self._get_sequences(orders, "state_allorder_chunk", metadata)[PHASE][ENGLAND]
        assert "output" not in seq  # all holds are filtered

        orders = self._get_all_holds_orders([RUSSIA])
        seq = self._get_sequences(orders, "state_allorder_chunk", metadata)[PHASE][ENGLAND]
        assert (
            seq["output"]
            == "France: A MAR S A PAR - BUR; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - APU; A VEN H; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - BUR; F KIE - DEN [EO_O]\nAustria: A BUD - RUM; A VIE - GAL; F TRI H [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS H; A WAR H; F SEV H; F STP/SC H [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]"
        )

    def test_all_orders_mark(self):
        metadata = self._get_metadata()
        metadata["opt"]["allorders_mark_all_holds"] = True

        orders = self._get_all_holds_orders([ENGLAND])
        seq = self._get_sequences(orders, "state_allorder_chunk", metadata)[PHASE][ENGLAND]
        assert "output" not in seq  # all holds are filtered

        orders = self._get_all_holds_orders([RUSSIA])
        seq = self._get_sequences(orders, "state_allorder_chunk", metadata)[PHASE][ENGLAND]
        # BAD marked in output
        assert (
            seq["output"]
            == "France: A MAR S A PAR - BUR; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - APU; A VEN H; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - BUR; F KIE - DEN [EO_O]\nAustria: A BUD - RUM; A VIE - GAL; F TRI H [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: BAD A MOS H; A WAR H; F SEV H; F STP/SC H [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]"
        )

    def test_all_orders_independent(self):
        metadata = self._get_metadata()

        orders = self._get_all_holds_orders([ENGLAND])
        seq = self._get_sequences(orders, "state_allorderindependent_chunk", metadata)[PHASE][
            ENGLAND
        ]
        assert "output" not in seq[1]  # No orders for England

        # does have orders for Italy
        assert seq[0]["output"] == "A BUD - RUM; A VIE - GAL; F TRI H [EO_O]"
