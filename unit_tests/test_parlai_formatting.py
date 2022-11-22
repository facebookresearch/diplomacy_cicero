#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import unittest

from parlai_diplomacy.utils.game2seq.factory import get_input_format, sequence_formatter_factory
from parlai_diplomacy.utils.game2seq.typing import Metadata
from parlai_diplomacy.utils.game2seq.format_helpers.state import StateFlattener
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageObjectPart,
    MessageHistoryUnflattener,
    get_last_speaker,
)
from parlai_diplomacy.utils.game2seq.format_helpers.orders import OrdersFlattener
from parlai_diplomacy.utils.game2seq.format_helpers.misc import POT_TYPE_CONVERSION
from parlai_diplomacy.utils.game2seq.dialogue_prediction import (
    DialoguePredictionFormatter,
    TrainingDialoguePredictionFormatter,
)

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Phase, Power
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.timestamp import Timestamp

from .test_pydipcc import TestMatchingWebdip


UNIT_TEST_DIR = os.path.dirname(__file__)
# for purposes of this test, we always look at England and phase W1901A
ENGLAND = "ENGLAND"
RUSSIA = "RUSSIA"
PHASE = "W1901A"
RETREAT_PHASE = "F1902R"
MOVEMENT_PHASE = "S1901M"
MOVEMENT_PHASE_2 = "F1901M"
MOVEMENT_PHASE_3 = "S1902M"


def load_game():
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")
    with open(fle, "r") as f:
        game_json = json.load(f)
    with open(fle, "r") as f:
        game_object = Game.from_json(f.read())
    # roll forward to the next phase
    last_phase_orders = game_json["phases"][-1]["orders"]
    for pwr, orders in last_phase_orders.items():
        game_object.set_orders(pwr, orders)
    game_object.process()

    return game_object


class TestOrderFormatting(unittest.TestCase):
    def _get_sequences(self, task_name: str, metadata: Metadata = None, training: bool = False):
        game = load_game()
        if metadata is None:
            metadata = self._get_metadata(set_power_features=False)
        self.formatter = sequence_formatter_factory(
            task_name, metadata.get("task_version", 0), training=training
        )  # Uses version 1 by default
        input_format = get_input_format(task_name)
        return self.formatter.change_format(game, input_format, metadata)

    def _get_metadata(self, set_power_features=True) -> Metadata:
        metadata = {}
        metadata["opt"] = {
            "include_player_ratings": set_power_features,
            "include_player_chattiness": set_power_features,
        }
        metadata["power_metadata"] = {}
        if set_power_features:
            for power in POWERS:
                metadata["power_metadata"][power] = {"rating": 1, "chattiness": 1}

        return metadata

    def test_state_order_formatting(self):
        seq = self._get_sequences("state_order_chunk")[PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A England:",
        )
        self.assertEqual(seq["output"], "F LON B; F LVP B [EO_O]")
        output_order = set(self.formatter.orders_unflattener.unflatten_action(seq["output"]))
        self.assertEqual(output_order, {"F LON B", "F LVP B"})

        seq = self._get_sequences("state_order_chunk")[PHASE][RUSSIA]
        self.assertEqual(
            seq["input"],
            "units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A Russia:",
        )
        self.assertEqual(seq["output"], " [EO_O]")
        output_order = set(self.formatter.orders_unflattener.unflatten_action(seq["output"]))
        self.assertEqual(output_order, set())

    def test_game_info(self):
        metadata = self._get_metadata(set_power_features=False)
        metadata["opt"]["include_game_info"] = True
        metadata["anon"] = "NON-ANON"
        metadata["phase_minutes"] = "5"
        metadata["pot_type"] = POT_TYPE_CONVERSION["Points-per-supply-center"]
        metadata["all_unknowns"] = True
        seq = self._get_sequences("state_order_chunk", metadata=metadata)[PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A England NON-ANON 5min PPSC ALL-UNK:",
        )

    def test_shortstate_order_versioning(self):
        # test version changing during retreat
        metadata = self._get_metadata(set_power_features=False)
        # test version 0
        metadata["task_version"] = 0
        seq = self._get_sequences("shortstate_order_chunk", metadata=metadata)[RETREAT_PHASE][
            ENGLAND
        ]
        self.assertEqual(
            seq["input"],
            "units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA [EO_STATE] F1902R England:",
        )

        # test version 1
        metadata["task_version"] = 1
        seq = self._get_sequences("shortstate_order_chunk", metadata=metadata)[RETREAT_PHASE][
            ENGLAND
        ]
        self.assertEqual(
            seq["input"],
            "units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {'A MUN': ['RUH', 'KIE', 'BOH']}; Italy: {}; Russia: {}; Turkey: {} [EO_STATE] F1902R England:",
        )

    def test_message_history_shortstate_order_formatting(self):
        seq = self._get_sequences("message_history_shortstate_order_chunk")[PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A England:",
        )
        self.assertEqual(seq["output"], "F LON B; F LVP B [EO_O]")

        seq = self._get_sequences("message_history_shortstate_order_chunk")[PHASE][RUSSIA]
        self.assertEqual(
            seq["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A Russia:",
        )
        self.assertEqual(seq["output"], " [EO_O]")

    def test_messages_history_shortstate_allorder_formatting(self):
        seq = self._get_sequences("message_history_shortstate_allorder_chunk")[PHASE][ENGLAND]
        self.assertEqual(
            seq["output"],
            "France: A PAR B [EO_O]\nItaly: F NAP B [EO_O]\nGermany: A KIE B; F BER B [EO_O]\nAustria: A BUD B [EO_O]\nTurkey: A ANK B; F CON B [EO_O]\nRussia:  [EO_O]\nEngland: F LON B; F LVP B [EO_O]",
        )
        all_orders = self.formatter.orders_unflattener.unflatten_joint_action(seq["output"])
        self.assertEqual(all_orders[ENGLAND], ("F LON B", "F LVP B"))

    def test_message_history_orderhistorysincelastmovementphase_shortstate_allorder_formatting(
        self,
    ):
        seq = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_shortstate_allorder_chunk",
        )[PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A England:",
        )
        seq = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_shortstate_allorder_chunk",
        )[RETREAT_PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M]\nS1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]\nEngland -> Germany: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nFrance -> England: Porta lorem mollis aliquam ut. [EO_M]\nEngland -> France: Morbi non arcu risus quis varius quam. [EO_M]\nF1902M\nEngland -> Germany: In mollis nunc sed id semper risus in hendrerit gravida. [EO_M]\nFrance -> England: Viverra justo nec ultrices dui. [EO_M]\nGermany -> England: Tempus imperdiet nulla malesuada pellentesque elit eget. [EO_M]\nEngland -> Germany: Porta lorem mollis aliquam ut. [EO_M]\nEngland -> Germany: Risus viverra adipiscing at in tellus integer feugiat scelerisque. [EO_M]\nItaly -> England: Placerat orci nulla pellentesque dignissim enim sit amet venenatis urna. [EO_M]\nEngland -> Germany: Fringilla phasellus faucibus scelerisque eleifend donec pretium. [EO_M]\nEngland -> Italy: Congue mauris rhoncus aenean vel elit scelerisque mauris pellentesque pulvinar. [EO_M]\nItaly -> England: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nItaly -> England: Ut faucibus pulvinar elementum integer. [EO_M] F1902M\nEngland: A BEL S A MUN - BUR; F BRE S F IRI - MAO; F ENG S F BRE; F IRI - MAO; F NTH - NWY [EO_O]\nFrance: A BUR S A SIL - MUN; A GAS - BRE; A PAR S A GAS - BRE; F POR - MAO [EO_O]\nItaly: A APU - VEN; A VEN - TYR; F ION - ADR; F TUN - ION [EO_O]\nGermany: A HOL H; A KIE - BER; A MUN H; F BAL - SWE; F DEN S F BAL - SWE [EO_O]\nAustria: A BUD - GAL; A RUM S A BUD - GAL; A SIL - MUN; F TRI H [EO_O]\nTurkey: A ANK - ARM; A GRE H; A SER S A GRE; F AEG - ION; F BLA S A RUM [EO_O]\nRussia: A UKR - RUM; A WAR - GAL; F BOT - BAL; F SEV - RUM [EO_O] units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA [EO_STATE] F1902R England:",
        )

    def test_orderhistorysincelastmovementphase_shortstate_allorder_formatting(self,):
        seq = self._get_sequences("orderhistorysincelastmovementphase_shortstate_allorder_chunk")[
            PHASE
        ][ENGLAND]
        self.assertEqual(
            seq["input"],
            "F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A England:",
        )

    def test_message_history_orderhistorysincelastmovementphase_allorder_formatting(self,):
        seq = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_allorder_chunk",
        )[PHASE][ENGLAND]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] W1901A England:",
        )

        seq = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_allorder_chunk"
        )[PHASE][RUSSIA]
        self.assertEqual(
            seq["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] W1901A Russia:",
        )

    def test_messagehistory_state_allorderindependent_formatting(self):
        seqs = self._get_sequences("message_history_state_allorderindependent_chunk",)[PHASE][
            RUSSIA
        ]
        self.assertEqual(len(seqs), 7)  # 7 examples here
        # check Russian prediction for Germany
        self.assertEqual(
            seqs[3]["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A Russia for Germany:",
        )
        self.assertEqual(seqs[3]["output"], "A KIE B; F BER B [EO_O]")
        # Russia prediction for itself
        self.assertEqual(seqs[5]["output"], " [EO_O]")
        # Check format of input prompt
        self.assertEqual(seqs[2]["input"][-18:], "Russia for France:")

    def test_message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_formatting(
        self,
    ):
        seqs = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk",
        )[PHASE][RUSSIA]
        self.assertEqual(len(seqs), 7)  # 7 examples here
        # check Russian prediction for Germany
        self.assertEqual(
            seqs[3]["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A Russia for Germany:",
        )
        # output contains A- and M-phase
        self.assertEqual(
            self.formatter.orders_unflattener.unflatten_rollout_action(
                seqs[3]["output"], current_phase="W1901A"
            )["W1901A"],
            ("A KIE B", "F BER B"),
        )
        self.assertEqual(
            self.formatter.orders_unflattener.unflatten_rollout_action(
                seqs[3]["output"], current_phase="W1901A"
            )["S1902M"],
            ("F DEN - SWE", "A HOL H", "A KIE S A HOL", "A MUN H", "F BER - BAL"),
        )
        # Check format of input prompt
        self.assertEqual(seqs[2]["input"][-18:], "Russia for France:")

    def test_messagehistory_state_allorderrollout_formatting(self):
        seq = self._get_sequences("message_history_state_allorderrollout_chunk",)[PHASE][RUSSIA]
        self.assertEqual(
            seq["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A Russia:",
        )
        # check that the output rollsout until the next movement phase (this is an adjustment phase)
        self.assertEqual(
            seq["output"],
            "W1901A\nEngland: F LON B; F LVP B [EO_O]\nFrance: A PAR B [EO_O]\nItaly: F NAP B [EO_O]\nGermany: A KIE B; F BER B [EO_O]\nAustria: A BUD B [EO_O]\nTurkey: A ANK B; F CON B [EO_O]\nRussia:  [EO_O]\nS1902M\nEngland: A BEL S A MUN - BUR; F BRE - MAO; F LON - ENG; F LVP - IRI; F NTH S A BEL [EO_O]\nFrance: A BUR S A HOL - BEL; A PAR S A BUR; A SPA - GAS; F POR - MAO [EO_O]\nItaly: A APU H; A VEN - TRI; F NAP - ION; F TUN H [EO_O]\nGermany: A HOL H; A KIE S A HOL; A MUN H; F BER - BAL; F DEN - SWE [EO_O]\nAustria: A BUD - GAL; A GAL - SIL; A RUM S A BUD - GAL; F TRI H [EO_O]\nTurkey: A ANK H; A BUL - GRE; A SER S A BUL - GRE; F BLA S A RUM; F CON - AEG [EO_O]\nRussia: A UKR - RUM; A WAR - GAL; F BOT - SWE; F SEV - RUM [EO_O]",
        )

    def test_messagehistory_shortstate_orderrollout_formatting(self):
        seq = self._get_sequences("message_history_shortstate_orderrollout_chunk",)[PHASE][RUSSIA]
        self.assertEqual(
            seq["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M]\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]\nItaly -> Russia: Massa massa ultricies mi quis hendrerit dolor magna eget est. [EO_M]\nGermany -> Russia: Erat nam at lectus urna duis convallis convallis. [EO_M]\nRussia -> Italy: Amet massa vitae tortor condimentum lacinia. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nItaly -> Russia: Ac turpis egestas integer eget aliquet nibh praesent. [EO_M]\nTurkey -> Russia: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nRussia -> Germany: Vestibulum lectus mauris ultrices eros. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nRussia -> Turkey: Senectus et netus et malesuada fames ac. [EO_M]\nF1901M\nRussia -> Austria: Massa tincidunt nunc pulvinar sapien et ligula ullamcorper. [EO_M]\nRussia -> Italy: Lacinia at quis risus sed vulputate odio ut enim blandit. [EO_M]\nAustria -> Russia: Vestibulum morbi blandit cursus risus. [EO_M]\nRussia -> Turkey: Cursus vitae congue mauris rhoncus aenean vel elit scelerisque mauris. [EO_M]\nRussia -> Germany: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nGermany -> Russia: Tortor at auctor urna nunc. [EO_M]\nRussia -> Turkey: Gravida neque convallis a cras semper auctor neque. [EO_M]\nRussia -> Germany: Semper quis lectus nulla at volutpat diam ut venenatis. [EO_M]\nRussia -> Italy: Odio ut sem nulla pharetra diam sit amet nisl suscipit. [EO_M]\nRussia -> Germany: Dapibus ultrices in iaculis nunc. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nItaly -> Russia: Diam maecenas ultricies mi eget mauris pharetra et. [EO_M]\nRussia -> Turkey: Urna porttitor rhoncus dolor purus. [EO_M]\nGermany -> Russia: Ac feugiat sed lectus vestibulum. [EO_M]\nRussia -> Italy: Nullam ac tortor vitae purus faucibus ornare suspendisse. [EO_M]\nRussia -> Italy: Non arcu risus quis varius quam quisque id diam. [EO_M]\nItaly -> Russia: Lorem ipsum dolor sit amet consectetur adipiscing elit. [EO_M]\nTurkey -> Russia: Sed risus pretium quam vulputate dignissim. [EO_M]\nRussia -> Germany: Ut faucibus pulvinar elementum integer. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA [EO_STATE] W1901A Russia:",
        )
        # check that the output rollsout until the next movement phase (this is an adjustment phase)
        self.assertEqual(
            seq["output"],
            "W1901A\n [EO_O]\nS1902M\nA UKR - RUM; A WAR - GAL; F BOT - SWE; F SEV - RUM [EO_O]",
        )
        # now check for a retreat phase
        seq = self._get_sequences("message_history_shortstate_orderrollout_chunk",)[RETREAT_PHASE][
            ENGLAND
        ]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M]\nS1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]\nEngland -> Germany: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nFrance -> England: Porta lorem mollis aliquam ut. [EO_M]\nEngland -> France: Morbi non arcu risus quis varius quam. [EO_M]\nF1902M\nEngland -> Germany: In mollis nunc sed id semper risus in hendrerit gravida. [EO_M]\nFrance -> England: Viverra justo nec ultrices dui. [EO_M]\nGermany -> England: Tempus imperdiet nulla malesuada pellentesque elit eget. [EO_M]\nEngland -> Germany: Porta lorem mollis aliquam ut. [EO_M]\nEngland -> Germany: Risus viverra adipiscing at in tellus integer feugiat scelerisque. [EO_M]\nItaly -> England: Placerat orci nulla pellentesque dignissim enim sit amet venenatis urna. [EO_M]\nEngland -> Germany: Fringilla phasellus faucibus scelerisque eleifend donec pretium. [EO_M]\nEngland -> Italy: Congue mauris rhoncus aenean vel elit scelerisque mauris pellentesque pulvinar. [EO_M]\nItaly -> England: Montes nascetur ridiculus mus mauris vitae ultricies leo. [EO_M]\nItaly -> England: Ut faucibus pulvinar elementum integer. [EO_M] units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA [EO_STATE] F1902R England:",
        )
        self.assertEqual(
            seq["output"], "F1902R\n [EO_O]\nW1902A\nF LON B [EO_O]\nS1903M\n [EO_O]",
        )

    def test_task_token(self):
        metadata = self._get_metadata()
        metadata["opt"]["task_token"] = "order"
        seq = self._get_sequences("shortstate_order_chunk", metadata=metadata)[RETREAT_PHASE][
            ENGLAND
        ]
        self.assertEquals(
            seq["input"],
            "units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA [EO_STATE] F1902R England 1 1 order:",
        )

    def test_historical_order_coast_canonicalization(self):
        # =========================================================
        # SUPPORT MOVE FROM COAST, QUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP/SC - LVN"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA/NC - GAS"])
        metadata = self._get_metadata(set_power_features=False)
        task_name = "orderhistorysincelastmovementphase_shortstate_allorder_chunk"
        self.formatter = sequence_formatter_factory(
            task_name, metadata.get("task_version", 0), training=False
        )  # Uses version 1 by default
        input_format = get_input_format(task_name)
        seq = self.formatter.change_format(game, input_format, metadata)["S1902M"]["FRANCE"]
        print(seq["input"])
        print(seq["output"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP/SC - LVN [EO_O]
W1901A
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O] units: Austria: A BUD, A VIE, F TRI; England: A LVP, F EDI, F LON; France: A MAR, A PAR, F SPA/NC; Germany: A BER, A MUN, F KIE; Italy: A ROM, A VEN, F NAP; Russia: A MOS, A WAR, F SEV, F STP/SC; Turkey: A CON, A SMY, F ANK [EO_STATE] S1902M France:""",
        )
        self.assertEqual(
            seq["output"],
            """England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA/NC - GAS [EO_O]""",
        )

        # =========================================================
        # SUPPORT MOVE FROM COAST, UNQUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP - LVN"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA - GAS"])
        metadata = self._get_metadata(set_power_features=False)
        task_name = "orderhistorysincelastmovementphase_shortstate_allorder_chunk"
        self.formatter = sequence_formatter_factory(
            task_name, metadata.get("task_version", 0), training=False
        )  # Uses version 1 by default
        input_format = get_input_format(task_name)
        seq = self.formatter.change_format(game, input_format, metadata)["S1902M"]["FRANCE"]
        print(seq["input"])
        print(seq["output"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP - LVN [EO_O]
W1901A
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O] units: Austria: A BUD, A VIE, F TRI; England: A LVP, F EDI, F LON; France: A MAR, A PAR, F SPA/NC; Germany: A BER, A MUN, F KIE; Italy: A ROM, A VEN, F NAP; Russia: A MOS, A WAR, F SEV, F STP/SC; Turkey: A CON, A SMY, F ANK [EO_STATE] S1902M France:""",
        )
        self.assertEqual(
            seq["output"],
            """England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA - GAS [EO_O]""",
        )

        # =========================================================
        # SUPPORT HOLD AT COAST, QUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP/SC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA/NC"])
        metadata = self._get_metadata(set_power_features=False)
        task_name = "orderhistorysincelastmovementphase_shortstate_allorder_chunk"
        self.formatter = sequence_formatter_factory(
            task_name, metadata.get("task_version", 0), training=False
        )  # Uses version 1 by default
        input_format = get_input_format(task_name)
        seq = self.formatter.change_format(game, input_format, metadata)["S1902M"]["FRANCE"]
        print(seq["input"])
        print(seq["output"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP [EO_O]
W1901A
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O] units: Austria: A BUD, A VIE, F TRI; England: A LVP, F EDI, F LON; France: A MAR, A PAR, F SPA/NC; Germany: A BER, A MUN, F KIE; Italy: A ROM, A VEN, F NAP; Russia: A MOS, A WAR, F SEV, F STP/SC; Turkey: A CON, A SMY, F ANK [EO_STATE] S1902M France:""",
        )
        self.assertEqual(
            seq["output"],
            """England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA [EO_O]""",
        )

        # =========================================================
        # SUPPORT HOLD AT COAST, UNQUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA"])
        metadata = self._get_metadata(set_power_features=False)
        task_name = "orderhistorysincelastmovementphase_shortstate_allorder_chunk"
        self.formatter = sequence_formatter_factory(
            task_name, metadata.get("task_version", 0), training=False
        )  # Uses version 1 by default
        input_format = get_input_format(task_name)
        seq = self.formatter.change_format(game, input_format, metadata)["S1902M"]["FRANCE"]
        print(seq["input"])
        print(seq["output"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP [EO_O]
W1901A
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O] units: Austria: A BUD, A VIE, F TRI; England: A LVP, F EDI, F LON; France: A MAR, A PAR, F SPA/NC; Germany: A BER, A MUN, F KIE; Italy: A ROM, A VEN, F NAP; Russia: A MOS, A WAR, F SEV, F STP/SC; Turkey: A CON, A SMY, F ANK [EO_STATE] S1902M France:""",
        )
        self.assertEqual(
            seq["output"],
            """England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA [EO_O]""",
        )


class TestDialogueFormatting(unittest.TestCase):
    def _get_sequences(
        self,
        task_name: str,
        metadata: Metadata = None,
        pseudo_orders=None,
        recipient: Power = None,
    ):
        game = load_game().rolled_back_to_phase_start(PHASE)
        default_opt = {"extend_state_history_since_last_n_movement_phase": 0, "task": task_name}
        if metadata is None:
            metadata = {"opt": default_opt, "power_metadata": {}}
        else:
            metadata = {**metadata, "opt": {**metadata["opt"], **default_opt}}

        if pseudo_orders is not None:
            metadata["pseudo_orders"] = {game.current_short_phase: {ENGLAND: pseudo_orders}}
        input_format = get_input_format(task_name)
        timestamp = Timestamp.from_centis(
            160582615148572200
        )  # 300 centis more than last timestamp from previous phase
        seqs = DialoguePredictionFormatter(version=1).change_format(
            game, input_format, metadata, ENGLAND, recipient=recipient, timestamp=timestamp,
        )
        return seqs

    def _get_pseudo_orders(self, phase: Phase = PHASE) -> PseudoOrders:
        # return a set of "pseudo orders"
        return PseudoOrders(
            {  # type: ignore
                phase: {
                    "FRANCE": ("A PAR B",),
                    "ITALY": ("F NAP B",),
                    "GERMANY": ("A KIE B", "F BER B"),
                    "AUSTRIA": ("A BUD B",),
                    "TURKEY": ("A ANK B", "F CON B"),
                    "RUSSIA": tuple(),
                    "ENGLAND": ("F LON B", "F LVP B"),
                }
            }
        )

    def test_message_history_state_dialogue_formatting(self):
        seq = self._get_sequences("message_history_state_dialogue_chunk")[PHASE]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] W1901A England:",
        )

    def test_message_history_dialogue_with_sleep_times_formatting(self):
        metadata = {"opt": {"add_sleep_times": True}, "power_metadata": {}}
        seq = self._get_sequences("message_history_state_dialogue_chunk", metadata)[PHASE]
        self.assertEqual(
            seq["input"],
            "S1901M\n0 England -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\n3 England -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\n10 France -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\n4 England -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\n12 England -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\n2 Germany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\n20 England -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\n10 England -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\n5 France -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\n3 Germany -> England: Eu facilisis sed odio morbi.. [EO_M]\n6 Russia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\n3 England -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\n0 England -> France: Ac feugiat sed lectus vestibulum. [EO_M]\n1 Germany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\n11 Germany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\n8 France -> England: Non consectetur a erat nam at lectus. [EO_M]\n3 England -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\n7 Germany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\n5 England -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\n3 France -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\n7 Russia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\n4 France -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\n6 England -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\n25 England -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] units: Austria: A GAL, A RUM, F TRI; England: A BEL, F BRE, F NTH; France: A BUR, A SPA, F POR; Germany: A HOL, A MUN, F DEN; Italy: A APU, A VEN, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A BUL, A SER, F BLA\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 1, 'homes': ['BUD', 'VIE']}; England: {'count': 2, 'homes': ['EDI', 'LON', 'LVP']}; France: {'count': 1, 'homes': ['MAR', 'PAR']}; Germany: {'count': 2, 'homes': ['BER', 'KIE']}; Italy: {'count': 1, 'homes': ['NAP', 'ROM']}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 2, 'homes': ['ANK', 'CON', 'SMY']} [EO_STATE] 3 W1901A England:",
        )

    def test_message_history_lastmovementorder_pseudoorder_dialogue_format(self):
        seq = self._get_sequences(
            "message_history_lastmovementorder_pseudoorder_dialogue_chunk",
            pseudo_orders=self._get_pseudo_orders(),
        )[PHASE]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] France: A PAR B [EO_O]\nItaly: F NAP B [EO_O]\nGermany: A KIE B; F BER B [EO_O]\nAustria: A BUD B [EO_O]\nTurkey: A ANK B; F CON B [EO_O]\nRussia:  [EO_O]\nEngland: F LON B; F LVP B [EO_O] W1901A England:",
        )

    def test_get_last_speaker(self):
        seq = self._get_sequences(
            "message_history_lastmovementorder_pseudoorder_dialogue_chunk",
            pseudo_orders=self._get_pseudo_orders(),
        )[PHASE]
        speaker, phase = get_last_speaker(seq["input"])
        self.assertEqual(speaker, "ENGLAND")
        self.assertEqual(phase, "F1901M")

    def test_message_history_dialogue_with_recipient_format(self):
        metadata = {"opt": {}, "power_metadata": {}}
        metadata["opt"]["add_recipient_to_prompt"] = True
        seq = self._get_sequences(
            "message_history_dialogue_chunk", metadata=metadata, recipient="GERMANY"
        )[PHASE]
        # Check that the recipient is added to the end of the prompt
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] W1901A England -> Germany:",
        )

        # Now set it to False and check that there is no recipient
        metadata["opt"]["add_recipient_to_prompt"] = False
        seq = self._get_sequences(
            "message_history_dialogue_chunk", metadata=metadata, recipient="GERMANY"
        )[PHASE]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] W1901A England:",
        )


class TestTrainingDialogueFormatting(unittest.TestCase):
    def _get_metadata(
        self,
        include_player_ratings: bool = True,
        include_player_chattiness: bool = True,
        task: str = "",
    ):
        game_id = 2
        metadata = {
            "game_id": game_id,
            "pseudo_orders": {},
            "power_metadata": {},
            "opt": {
                "include_player_ratings": include_player_ratings,
                "include_player_chattiness": include_player_chattiness,
                "task": task,
                "extend_state_history_since_last_n_movement_phase": 0,
            },
        }
        pso = "England: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O]\nFrance: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]"
        for ph in (MOVEMENT_PHASE, MOVEMENT_PHASE_2, PHASE, MOVEMENT_PHASE_3):
            for pow in range(1, 8):
                for i in range(1, 10):
                    k = f"{game_id}-{ph}-{pow}-{i}"
                    metadata["pseudo_orders"][k] = pso
        for power in POWERS:
            metadata["power_metadata"][power] = {"rating": 1, "chattiness": 1}

        return metadata

    def _get_sequences(self, task_name: str, metadata: Metadata = None, **metadata_kwargs):
        game = load_game()
        input_format = get_input_format(task_name)
        if metadata is None:
            metadata = self._get_metadata(**metadata_kwargs, task=task_name)
        seqs = TrainingDialoguePredictionFormatter(version=1).change_format(
            game, input_format, metadata
        )
        return seqs

    def test_message_history_state_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences("message_history_state_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 0, 'homes': []}; England: {'count': 0, 'homes': []}; France: {'count': 0, 'homes': []}; Germany: {'count': 0, 'homes': []}; Italy: {'count': 0, 'homes': []}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 0, 'homes': []} [EO_STATE] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_state_message_history_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences("state_message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 0, 'homes': []}; England: {'count': 0, 'homes': []}; France: {'count': 0, 'homes': []}; Germany: {'count': 0, 'homes': []}; Italy: {'count': 0, 'homes': []}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 0, 'homes': []} [EO_STATE] S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_dialogue_with_recipient_format(self):
        metadata = self._get_metadata()
        metadata["opt"]["add_recipient_to_prompt"] = True
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] S1902M England -> Germany 1 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_pseudoorder_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences("message_history_pseudoorder_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_2person_pseudoorder_dialogue_format(self):
        # test 2 person pseudo orders
        metadata = self._get_metadata(include_player_ratings=False)
        metadata["opt"]["all_power_pseudo_orders"] = False
        seq = self._get_sequences("message_history_pseudoorder_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE
        ][ENGLAND][-1]
        self.assertEqual(
            seq["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1901M England 1:",
        )
        self.assertEqual(
            seq["output"],
            "S1901M\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]",
        )

    def test_message_history_lastorder_pseudoorder_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences(
            "message_history_lastorder_pseudoorder_dialogue_chunk", metadata=metadata
        )[MOVEMENT_PHASE_3][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] W1901A\nEngland: F LON B; F LVP B [EO_O]\nFrance: A PAR B [EO_O]\nItaly: F NAP B [EO_O]\nGermany: A KIE B; F BER B [EO_O]\nAustria: A BUD B [EO_O]\nTurkey: A ANK B; F CON B [EO_O]\nRussia:  [EO_O] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_lastmovementorder_pseudoorder_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences(
            "message_history_lastmovementorder_pseudoorder_dialogue_chunk", metadata=metadata
        )[MOVEMENT_PHASE_3][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_orderhistorysincelastmovementphase_pseudoorder_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences(
            "message_history_orderhistorysincelastmovementphase_pseudoorder_dialogue_chunk",
            metadata=metadata,
        )[MOVEMENT_PHASE_3][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] F1901M\nEngland: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]\nFrance: A BUR H; A MAR - SPA; F MAO - POR [EO_O]\nItaly: A APU H; A VEN H; F ION - TUN [EO_O]\nGermany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]\nAustria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]\nTurkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]\nRussia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O]\nW1901A\nEngland: F LON B; F LVP B [EO_O]\nFrance: A PAR B [EO_O]\nItaly: F NAP B [EO_O]\nGermany: A KIE B; F BER B [EO_O]\nAustria: A BUD B [EO_O]\nTurkey: A ANK B; F CON B [EO_O]\nRussia:  [EO_O] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_pseudoorder_generation_message_history_state_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        metadata["pseudo_order_gen"] = True
        seq = self._get_sequences("message_history_state_dialogue_chunk", metadata=metadata,)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M]\nS1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M] units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 0, 'homes': []}; England: {'count': 0, 'homes': []}; France: {'count': 0, 'homes': []}; Germany: {'count': 0, 'homes': []}; Italy: {'count': 0, 'homes': []}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 0, 'homes': []} [EO_STATE] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )
        self.assertEqual(seq[0]["example_id"], "2-S1902M-1-1")

    def test_message_history_state_pseudoorder_dialogue_format(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences(
            "message_history_state_pseudoorder_dialogue_chunk", metadata=metadata
        )[MOVEMENT_PHASE_3][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON\nretreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}\ncenters: Austria: BUD, RUM, TRI, VIE; England: BEL, BRE, EDI, LON, LVP; France: MAR, PAR, POR, SPA; Germany: BER, DEN, HOL, KIE, MUN; Italy: NAP, ROM, TUN, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, BUL, CON, SER, SMY\nhomes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY\nbuilds: Austria: {'count': 0, 'homes': []}; England: {'count': 0, 'homes': []}; France: {'count': 0, 'homes': []}; Germany: {'count': 0, 'homes': []}; Italy: {'count': 0, 'homes': []}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 0, 'homes': []} [EO_STATE] France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]\nItaly: A ROM - VEN; A VEN - TYR; F NAP - ION [EO_O]\nGermany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]\nAustria: A BUD - SER; A VIE - BUD; F TRI - ALB [EO_O]\nTurkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]\nRussia: A MOS - UKR; A WAR H; F SEV - BLA; F STP/SC - BOT [EO_O]\nEngland: A LVP - YOR; F EDI - NTH; F LON - ENG [EO_O] S1902M England 1:",
        )
        self.assertEqual(
            seq[0]["output"], "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]",
        )

    def test_message_history_dialogue_without_silence_formatting(self):
        metadata = self._get_metadata()
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[PHASE][
            ENGLAND
        ]
        self.assertEqual(len(seq), 0)  # no examples without explicit silence tokens

    def test_message_history_dialogue_with_sleep_formatting(self):
        metadata = self._get_metadata()
        metadata["opt"]["include_sleep_messages"] = True

        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE
        ][
            ENGLAND
        ]  # More interesting in this phase

        self.assertEqual(
            seq[0]["output"], "S1901M\nEngland -> Sleep: OUT 10 FRANCE [EO_M]",
        )

    def test_message_history_dialogue_no_rating_formatting(self):
        metadata = self._get_metadata(include_player_ratings=False)
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_3
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M]\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]\nGermany -> England: Pellentesque elit ullamcorper dignissim cras. [EO_M]\nFrance -> England: Non consectetur a erat nam at lectus. [EO_M]\nEngland -> Germany: Nulla pharetra diam sit amet nisl suscipit. [EO_M]\nGermany -> England: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nEngland -> France: Lobortis elementum nibh tellus molestie nunc non blandit massa. [EO_M]\nFrance -> England: Aliquam faucibus purus in massa tempor nec feugiat nisl. [EO_M]\nRussia -> England: Suspendisse faucibus interdum posuere lorem ipsum. [EO_M]\nFrance -> England: Et pharetra pharetra massa massa ultricies mi quis. [EO_M]\nEngland -> Germany: Lacus laoreet non curabitur gravida arcu ac. [EO_M]\nEngland -> France: Vel risus commodo viverra maecenas accumsan. [EO_M] S1902M England 1:",
        )

    def test_dialogue_output_format(self):
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=self._get_metadata())[
            MOVEMENT_PHASE
        ][ENGLAND]
        self.assertEqual(
            seq[0]["output"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]",
        )
        msg_lst = MessageHistoryUnflattener(1).unflatten_messages(seq[0]["output"], MOVEMENT_PHASE)
        self.assertEqual(len(msg_lst), 1)
        self.assertEqual(
            msg_lst[0]["message"],
            "Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum.",
        )
        self.assertEqual(msg_lst[0][MessageObjectPart.RECIPIENT], "FRANCE")

    def test_single_turn_dialogue_formatting(self):
        metadata = self._get_metadata()
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE
        ][RUSSIA]
        self.assertEqual(
            seq[4]["input"],
            "S1901M\nRussia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\nAustria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\nAustria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\nRussia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\nItaly -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\nTurkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\nRussia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\nRussia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M] S1901M Russia 1 1:",
        )
        self.assertEqual(
            seq[4]["output"],
            "S1901M\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]",
        )

    def test_single_turn_dialogue_formatting_with_sleep_times(self):
        metadata = self._get_metadata()
        metadata["opt"]["add_sleep_times"] = True
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE
        ][RUSSIA]
        self.assertEqual(
            seq[4]["input"],
            "S1901M\n0 Russia -> Austria: Sit amet massa vitae tortor condimentum lacinia quis vel eros. [EO_M]\n9 Austria -> Russia: Viverra maecenas accumsan lacus vel. [EO_M]\n4 Austria -> Russia: Nam libero justo laoreet sit amet cursus sit amet dictum. [EO_M]\n2 Russia -> Turkey: Nunc eget lorem dolor sed. [EO_M]\n2 Italy -> Russia: Vitae nunc sed velit dignissim sodales ut eu. [EO_M]\n2 Turkey -> Russia: Nunc eget lorem dolor sed viverra. [EO_M]\n16 Russia -> Austria: Pellentesque habitant morbi tristique senectus et netus et malesuada. [EO_M]\n7 Russia -> Italy: Maecenas sed enim ut sem viverra aliquet eget. [EO_M] 7 S1901M Russia 1 1:",
        )
        self.assertEqual(
            seq[4]["output"],
            "S1901M\nRussia -> Turkey: A iaculis at erat pellentesque adipiscing.. [EO_M]",
        )

    def test_single_turn_dialogue_formatting_in_a_row(self):
        """
        Test single turn dialogue formatting when one power sends multiple
        messages in a row to the same power.
        """
        metadata = self._get_metadata()

        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_2
        ][RUSSIA]

        self.assertEqual(len(seq), 13)

    def test_single_turn_dialogue_formatting_response_view_without_pseudo_order(self):
        """
        Test single turn dialogue formatting when the teacher is response view without pseudo-order
        """
        metadata = self._get_metadata()
        metadata["opt"]["response_view_dialogue_model"] = True
        metadata["opt"]["pseudo_order_generation"] = False
        seq = self._get_sequences("message_history_dialogue_chunk", metadata=metadata)[
            MOVEMENT_PHASE_2
        ][ENGLAND]
        self.assertEqual(
            seq[0]["input"],
            "S1901M\nEngland -> France: Mi tempus imperdiet nulla malesuada pellentesque elit eget gravida cum. [EO_M]\nEngland -> Germany: Ut faucibus pulvinar elementum integer. [EO_M]\nFrance -> England: Congue nisi vitae suscipit tellus mauris a diam maecenas sed. [EO_M]\nEngland -> Italy: Et tortor at risus viverra adipiscing at in tellus.. [EO_M]\nEngland -> France: Nulla pellentesque dignissim enim sit amet. [EO_M]\nGermany -> England: Morbi tincidunt augue interdum velit euismod in pellentesque. [EO_M]\nEngland -> Germany: Congue quisque egestas diam in arcu cursus euismod. [EO_M]\nEngland -> Russia: Eget aliquet nibh praesent tristique magna sit amet purus. [EO_M]\nFrance -> England: Pellentesque id nibh tortor id aliquet lectus proin nibh nisl. [EO_M]\nGermany -> England: Eu facilisis sed odio morbi.. [EO_M]\nRussia -> England: Sed felis eget velit aliquet sagittis id. [EO_M]\nEngland -> France: Rhoncus dolor purus non enim praesent elementum. [EO_M]\nF1901M\nEngland -> France: Ac feugiat sed lectus vestibulum. [EO_M] F1901M England 1 1:",
        )
        self.assertEqual(
            seq[0]["output"],
            "F1901M\nGermany -> England: Porttitor eget dolor morbi non arcu risus quis. [EO_M]",
        )

    def test_message_unflattening(self):
        """
        Test message unflattening
        """
        unflattener = MessageHistoryUnflattener(1)
        multiple_msgs_str = "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]\nEngland -> Germany: Rhoncus dolor purus.\nNon enim praesent elementum. [EO_M]"
        multiple_msgs_parsed = unflattener.unflatten_messages(multiple_msgs_str, "S1902M")
        self.assertEqual(len(multiple_msgs_parsed), 2)  # Should contain 2 messages
        for msg in multiple_msgs_parsed:
            self.assertEqual(msg[MessageObjectPart.SENDER], "ENGLAND")
            self.assertEqual(msg[MessageObjectPart.RECIPIENT], "GERMANY")
            self.assertEqual(msg[MessageObjectPart.PHASE], "S1902M")

        self.assertEqual(
            multiple_msgs_parsed[0][MessageObjectPart.MESSAGE], "Viverra accumsan in nisl nisi."
        )
        self.assertEqual(
            multiple_msgs_parsed[1][MessageObjectPart.MESSAGE],
            "Rhoncus dolor purus.\nNon enim praesent elementum.",
        )

        single_msg_str = "S1902M\nEngland -> Germany: Viverra accumsan in nisl nisi. [EO_M]"
        single_msg_parsed = unflattener.unflatten_messages(single_msg_str, "S1902M")
        self.assertEqual(len(single_msg_parsed), 1)  # Should contain 1 message
        self.assertEqual(
            single_msg_parsed[0][MessageObjectPart.MESSAGE], "Viverra accumsan in nisl nisi."
        )

    def test_trivial_r_phase_retreat_state_v2(self):
        game = TestMatchingWebdip._get_trivial_r_phase_game()
        seq = StateFlattener(2).flatten_state(
            game.get_state(), game.current_short_phase, short_version=True
        )
        retreats = seq.split("retreats:")[1]
        self.assertIn("RUM", retreats)

    def test_flatten_rollout_joint_action_bilateral_phasemajor(self):
        flattener = OrdersFlattener(2)
        rollout_joint_action = {
            "F1901R": {"RUSSIA": ("F RUM R BLA",), "TURKEY": ()},
            "W1901A": {"RUSSIA": (), "TURKEY": ("A CON B",)},
            "S1902M": {
                "RUSSIA": ("F STP/SC - BOT", "A MOS - UKR", "A WAR - GAL", "F BLA - RUM"),
                "TURKEY": ("A RUM S A BUD", "F ANK - BLA", "A SMY - ARM", "A CON - BUL"),
            },
        }
        flattened = flattener.flatten_rollout_joint_action_bilateral_phasemajor(
            rollout_joint_action, "RUSSIA", "TURKEY", speaker_first=False
        )
        print(flattened)
        expected = (
            "F1901R"
            "\nTURKEY: "
            "\nRUSSIA: F RUM R BLA"
            "\nW1901A"
            "\nTURKEY: A CON B"
            "\nRUSSIA: "
            "\nS1902M"
            "\nTURKEY: A CON BUL; A RUM S A BUD; A SMY ARM; F ANK BLA"
            "\nRUSSIA: A MOS UKR; A WAR GAL; F BLA RUM; F STP/SC BOT"
        )

        self.assertEqual(flattened, expected)

    def test_historical_dialogue_coast_canonicalization(self):
        # =========================================================
        # SUPPORT MOVE FROM COAST, QUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP/SC - LVN"])
        game.process()
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F SPA/NC - GAS",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP/SC - LVN [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA/NC - GAS [EO_O] S1902M France 1 1:""",
        )

        # =========================================================
        # SUPPORT MOVE TO COAST, QUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.set_orders("RUSSIA", ["F STP/SC - LVN"])
        game.process()
        game.set_orders("RUSSIA", ["A MOS S F LVN - STP/SC"])
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F MAO - SPA/SC",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F LVN - STP/SC [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F MAO - SPA/SC [EO_O] S1902M France 1 1:""",
        )

        # =========================================================
        # SUPPORT MOVE TO COAST, UNQUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.set_orders("RUSSIA", ["F STP/SC - LVN"])
        game.process()
        game.set_orders("RUSSIA", ["A MOS S F LVN - STP"])
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F MAO - SPA",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F LVN - STP [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F MAO - SPA [EO_O] S1902M France 1 1:""",
        )

        # =========================================================
        # SUPPORT MOVE FROM COAST, UNQUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP - LVN"])
        game.process()
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F SPA - GAS",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP - LVN [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA - GAS [EO_O] S1902M France 1 1:""",
        )

        # =========================================================
        # SUPPORT HOLD AT COAST, QUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP/SC"])
        game.process()
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F SPA/NC",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA [EO_O] S1902M France 1 1:""",
        )

        # =========================================================
        # SUPPORT HOLD AT COAST, UNQUALIFIED
        game = Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.set_orders("RUSSIA", ["A MOS S F STP"])
        game.process()
        game.process()
        metadata = self._get_metadata().copy()
        metadata["pseudo_orders"] = {
            "S1902M": {
                "FRANCE": PseudoOrders(
                    {
                        "S1902M": {
                            "FRANCE": ("A MAR S F SPA",),
                            "ITALY": tuple(),
                            "GERMANY": tuple(),
                            "AUSTRIA": tuple(),
                            "TURKEY": tuple(),
                            "RUSSIA": tuple(),
                            "ENGLAND": tuple(),
                        }
                    }
                )
            }
        }

        timestamp = Timestamp.from_centis(160582615148572200)
        task_name = "message_history_lastmovementorder_pseudoorder_dialogue_chunk"
        recipient = "ENGLAND"
        input_format = get_input_format(task_name)
        seq = DialoguePredictionFormatter(version=1).change_format(
            game,
            input_format,
            metadata,
            speaker="FRANCE",
            recipient=recipient,
            timestamp=timestamp,
        )["S1902M"]

        print(seq["input"])
        self.assertEqual(
            seq["input"],
            """F1901M
England:  [EO_O]
France: F MAO - SPA/NC [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia: A MOS S F STP [EO_O] England:  [EO_O]
Italy:  [EO_O]
Germany:  [EO_O]
Austria:  [EO_O]
Turkey:  [EO_O]
Russia:  [EO_O]
France: A MAR S F SPA [EO_O] S1902M France 1 1:""",
        )
