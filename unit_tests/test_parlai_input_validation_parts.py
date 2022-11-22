#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import unittest

from parlai_diplomacy.utils.game2seq import input_validation as R
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
from fairdiplomacy.models.consts import LOCS


class InputValidationPartsTests(unittest.TestCase):
    def test_loc(self):
        for loc in LOCS:
            R.validate(R.InputValidator([], "", {}, 1).LOC, loc)
        self.assertRaises(
            ValueError, R.validate, R.InputValidator([], "", {}, 1).LOC, "F LON"
        )  # unit is not valid loc

    def test_unit(self):
        for loc in LOCS:
            R.validate(R.InputValidator([], "", {}, 1).UNIT, random.choice("AF") + " " + loc)

    def test_order(self):
        # Version 1
        for valid_order in [
            "F LON - BEL VIA",
            "F BLA S F RUM",
            "F NTH S A LON - BEL",
            "F STP/NC B",
        ]:
            R.validate(R.InputValidator([], "", {}, 1).ORDER, valid_order)

        # Version 2
        for valid_order in [
            "F LON BEL VIA",
            "F BLA S F RUM",
            "F NTH S A LON BEL",
            "F STP/NC B",
        ]:
            R.validate(R.InputValidator([], "", {}, 2).ORDER, valid_order)

        # Version 1
        for invalid_order in [
            "F LON - A BEL VIA",
            "F BLAH S F RUM",
            "F NTH C A LON - BEL VIA",
            "F STP/NC B STP/NC",
        ]:
            self.assertRaises(
                ValueError, R.validate, R.InputValidator([], "", {}, 1).ORDER, invalid_order
            )

        # Version 2
        for invalid_order in [
            "F LON A BEL VIA",
            "F BLAH - S F RUM",
            "F NTH C A LON BEL VIA",
            "F STP/NC B STP/NC",
        ]:
            self.assertRaises(
                ValueError, R.validate, R.InputValidator([], "", {}, 2).ORDER, invalid_order
            )

    def test_message(self):
        # Version 1
        R.validate(R.InputValidator([], "", {}, 1).MESSAGE, "England -> Turkey: msg [EO_M]")

        # Version 2
        R.validate(R.InputValidator([], "", {}, 2).MESSAGE, "ENGLAND -> TURKEY: msg")

        # Version 1
        self.assertRaises(
            ValueError,
            R.validate,
            R.InputValidator([], "", {}, 1).MESSAGE,
            "England -> Turkey: msg\n[EO_M]",
        )  # no newlines

        # Version 2
        self.assertRaises(
            ValueError,
            R.validate,
            R.InputValidator([], "", {}, 1).MESSAGE,
            "England -> TURKEY: msg",
        )  # no capitalization

    def test_phase(self):
        for valid_phase in ["S1901M", "W1901A", "F1999R"]:
            R.validate(R.InputValidator([], "", {}, 1).PHASE, valid_phase)
        for invalid_phase in ["S1901", "1901", "[EO_M]"]:
            self.assertRaises(
                ValueError, R.validate, R.InputValidator([], "", {}, 1).PHASE, invalid_phase
            )

    def test_power_action(self):
        # Version 1
        x = "England: A LVP - YOR; F EDI - NWG; F LON - NTH [EO_O]"
        R.validate(R.InputValidator([], "", {}, 1).POWER_ACTION, x)

        # Version 2
        x = "ENGLAND: A LVP YOR; F EDI NWG; F LON NTH"
        R.validate(R.InputValidator([], "", {}, 2).POWER_ACTION, x)

    def test_joint_action(self):
        # Version 1
        x = """
England: A LVP - YOR; F EDI - NWG; F LON - NTH [EO_O]
Italy: A ROM - APU; A VEN H; F NAP - ION [EO_O]
Germany: A BER - KIE; A MUN - RUH; F KIE - DEN [EO_O]
Austria: A BUD - SER; A VIE - GAL; F TRI - ALB [EO_O]
Turkey: A CON - BUL; A SMY - CON; F ANK - BLA [EO_O]
Russia: A MOS - UKR; A WAR - GAL; F SEV - BLA; F STP/SC - BOT [EO_O]
France: A MAR - SPA; A PAR - BUR; F BRE - MAO [EO_O]""".strip()
        R.validate(R.InputValidator([], "", {}, 1).JOINT_ACTION, x)

        # Version 2
        x = """
ENGLAND: A LVP YOR; F EDI NWG; F LON NTH
ITALY: A ROM APU; A VEN H; F NAP ION
GERMANY: A BER KIE; A MUN RUH; F KIE DEN
AUSTRIA: A BUD SER; A VIE GAL; F TRI ALB
TURKEY: A CON BUL; A SMY CON; F ANK BLA
RUSSIA: A MOS UKR; A WAR GAL; F SEV BLA; F STP/SC BOT
FRANCE: A MAR SPA; A PAR BUR; F BRE MAO""".strip()
        R.validate(R.InputValidator([], "", {}, 2).JOINT_ACTION, x)

    def test_order_history(self):
        # Version 1
        x = """
F1901M
England: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]
France: A BUR H; A MAR - SPA; F MAO - POR [EO_O]
Italy: A APU H; A VEN H; F ION - TUN [EO_O]
Germany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]
Austria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]
Turkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]
Russia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O]
W1901A
England: F LON B; F LVP B [EO_O]
France: A PAR B [EO_O]
Italy: F NAP B [EO_O]
Germany: A KIE B; F BER B [EO_O]
Austria: A BUD B [EO_O]
Turkey: A ANK B; F CON B [EO_O]
Russia:  [EO_O]""".strip()
        R.validate(R.InputValidator([], "", {}, 1).ORDER_HISTORY_SINCE_LAST_MOVEMENT, x)

        # Version 2
        x = """
F1902M
ENGLAND: A YOR H; F EDI NTH; F NTH HEL; F NWY S F EDI NTH
FRANCE: A BEL H; A PIC LON VIA; A SPA TUN VIA; F ENG C A PIC LON; F LYO TYS; F WES C A SPA TUN
ITALY: A ROM H; A VEN H; F NAP H
GERMANY: A BOH TYR; A VIE BOH; F HOL HEL; F KIE H
AUSTRIA: A SER S A TRI; A TRI S F ADR VEN; F ADR VEN
TURKEY: A CON H; A GRE ALB; A SMY ANK; F AEG GRE
RUSSIA: A BUD TRI; A GAL VIE; A LVN PRU; A WAR GAL; F BUL/EC S F RUM; F RUM S F BUL; F SWE DEN
W1902A
ENGLAND: F EDI D
FRANCE: A BRE B; A PAR B
AUSTRIA: F ADR D
RUSSIA: A SEV B; A WAR B; F STP/SC B""".strip()
        R.validate(R.InputValidator([], "", {}, 2).ORDER_HISTORY_SINCE_LAST_MOVEMENT, x)

        # Version 2, empty retreat phase; this may happen when there are no valid retreats (only disbands)
        x = """F1902M
ENGLAND: A BEL H; F LON ENG; F NTH NWY; F SWE S F NTH NWY
FRANCE: A GAS MAR; A PAR BUR; A PIC S A PAR BUR; F BRE ENG; F MAO S F BRE ENG
ITALY: A TRI S A RUM BUD; A TUN H; A VEN S A TRI; F ADR S A TRI; F NAP ION
GERMANY: A BUR GAS; A HOL S A BEL; A RUH BUR; F BAL S F SWE; F DEN S F SWE
AUSTRIA: A ALB S F GRE; A BUD S A VIE; A VIE S A BUD; F GRE S A ALB
TURKEY: A BUL S A SER; A SER S A RUM BUD; F AEG GRE; F ANK BLA
RUSSIA: A RUM BUD; A STP S F BAR NWY; A UKR GAL; F BAR NWY; F SEV BLA; F SKA S F BAR NWY
F1902R
"""
        R.validate(R.InputValidator([], "", {}, 2).ORDER_HISTORY_SINCE_LAST_MOVEMENT, x)

    def test_power_units(self):
        # Version 1
        x = "Austria: A BUD, A GAL, A RUM, F TRI"
        R.validate(R.InputValidator([], "", {}, 1).POWER_UNITS, x)

        # Version 2
        x = "AUSTRIA: A BUD, A GAL, A RUM, F TRI"
        R.validate(R.InputValidator([], "", {}, 2).POWER_UNITS, x)

    def test_shortstate(self):
        # Version 1
        x = """units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON [EO_STATE]""".strip()
        R.validate(R.InputValidator([], "", {}, 1).SHORTSTATE, x)

        # Version 2
        x = """units: AUSTRIA: *A GAL, A BUL, A TYR, A VIE, A WAR, F AEG; ENGLAND: A BEL, A KIE, A NWY, F HEL, F NTH, F SKA; FRANCE: A BER, A MAR, A MUN, A TUS, F LYO, F WES; GERMANY: *A KIE, A BUR, A DEN; ITALY: A RUM, A SMY, F ADR, F EAS, F TYS; RUSSIA: A BOH, A GAL, F SWE; TURKEY: A ANK, A SEV, F BLA, F CON
retreats: AUSTRIA: A GAL - BUD / SIL; GERMANY: A KIE - RUH
        """.strip()
        R.validate(R.InputValidator([], "", {}, 2).SHORTSTATE, x)

        # Version 2, no valid retreats
        x = "units: AUSTRIA: A BUD, A SER, A TRI, A VIE, F BUL/SC; ENGLAND: A LVP, F LON, F YOR; FRANCE: A BEL, A GAS, A PIC, F ENG, F IRI, F MAO; GERMANY: A BOH, A DEN, A HOL, A MUN, A TYR, F NWY; ITALY: A SMY, A VEN, F AEG, F EAS; RUSSIA: A GAL, A RUM, A STP, A UKR, F SWE; TURKEY: A ANK, A SEV, F BLA, F CON\nretreats: RUSSIA: F SEV - "
        R.validate(R.InputValidator([], "", {}, 2).SHORTSTATE, x)

    def test_message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk(
        self,
    ):
        # Version 1
        x = """
S1901M
England -> France: Lorem Ipsum [EO_M]
England -> Germany: Lorem Ipsum [EO_M]
France -> England: Lorem Ipsum [EO_M]
England -> Italy: Lorem Ipsum [EO_M]
England -> France: Lorem Ipsum [EO_M]
Germany -> England: Lorem Ipsum [EO_M]
England -> Germany: Lorem Ipsum [EO_M]
England -> Russia: Lorem Ipsum [EO_M]
France -> England: Lorem Ipsum [EO_M]
Germany -> England: Lorem Ipsum [EO_M]
Russia -> England: Lorem Ipsum [EO_M]
England -> France: Lorem Ipsum [EO_M]
F1901M
England -> France: Lorem Ipsum [EO_M]
Germany -> England: Lorem Ipsum [EO_M]
Germany -> England: Lorem Ipsum [EO_M]
France -> England: Lorem Ipsum [EO_M]
England -> Germany: Lorem Ipsum [EO_M]
Germany -> England: Lorem Ipsum [EO_M]
England -> France: Lorem Ipsum [EO_M]
W1901A
France -> England: Lorem Ipsum [EO_M]
Russia -> England: Lorem Ipsum [EO_M]
France -> England: Lorem Ipsum [EO_M]
England -> Germany: Lorem Ipsum [EO_M]
England -> France: Lorem Ipsum [EO_M]
S1902M
England -> Germany: Lorem Ipsum [EO_M]
England -> Germany: Lorem Ipsum [EO_M]
France -> England: Lorem Ipsum [EO_M]
England -> France: Lorem Ipsum [EO_M] F1901M
England: A YOR - BEL VIA; F ENG - BRE; F NTH C A YOR - BEL [EO_O]
France: A BUR H; A MAR - SPA; F MAO - POR [EO_O]
Italy: A APU H; A VEN H; F ION - TUN [EO_O]
Germany: A KIE - HOL; A MUN - BUR; F DEN - SWE [EO_O]
Austria: A BUD - RUM; A GAL S A BUD - RUM; F TRI - VEN [EO_O]
Turkey: A BUL - SER; A CON - BUL; F BLA S A BUD - RUM [EO_O]
Russia: A MOS - WAR; A UKR - RUM; F BOT - SWE; F SEV S A UKR - RUM [EO_O]
W1901A
England: F LON B; F LVP B [EO_O]
France: A PAR B [EO_O]
Italy: F NAP B [EO_O]
Germany: A KIE B; F BER B [EO_O]
Austria: A BUD B [EO_O]
Turkey: A ANK B; F CON B [EO_O]
Russia:  [EO_O] units: Austria: A BUD, A GAL, A RUM, F TRI; England: A BEL, F BRE, F LON, F LVP, F NTH; France: A BUR, A PAR, A SPA, F POR; Germany: A HOL, A KIE, A MUN, F BER, F DEN; Italy: A APU, A VEN, F NAP, F TUN; Russia: A UKR, A WAR, F BOT, F SEV; Turkey: A ANK, A BUL, A SER, F BLA, F CON [EO_STATE] S1902M England for Austria:""".strip()

        fmt = "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk"
        formatter = sequence_formatter_factory(fmt, 1)
        regex = formatter.get_input_validation_regex(fmt, {})
        R.validate(regex, x)

        # Version 2
        x = """
S1901M
ENGLAND -> FRANCE: Lorem Ipsum
FRANCE -> ENGLAND: Lorem Ipsum
F1901M
ENGLAND -> FRANCE: Lorem Ipsum
GERMANY -> FRANCE: Lorem Ipsum
W1901A
FRANCE -> ENGLAND: Lorem Ipsum
S1902M
TURKEY -> FRANCE: Lorem Ipsum
FRANCE -> ITALY: Lorem Ipsum
ITALY -> FRANCE: Lorem Ipsum
F1901M
ENGLAND: A YOR BEL VIA; F ENG BRE; F NTH C A YOR BEL
FRANCE: A BUR H; A MAR SPA; F MAO POR
ITALY: A APU H; A VEN H; F ION TUN
GERMANY: A KIE HOL; A MUN BUR; F DEN SWE
AUSTRIA: A BUD RUM; A GAL S A BUD RUM; F TRI VEN
TURKEY: A BUL SER; A CON BUL; F BLA S A BUD RUM
RUSSIA: A MOS WAR; A UKR RUM; F BOT SWE; F SEV S A UKR RUM
W1901A
ENGLAND: F LON B; F LVP B
FRANCE: A PAR B
ITALY: F NAP B
GERMANY: A KIE B; F BER B
AUSTRIA: A BUD B
TURKEY: A ANK B; F CON B
units: AUSTRIA: A BUD, A GAL, A RUM, F TRI; ENGLAND: A BEL, F BRE, F LON, F LVP, F NTH; FRANCE: A BUR, A PAR, A SPA, F POR; GERMANY: A HOL, A KIE, A MUN, F BER, F DEN; ITALY: A APU, A VEN, F NAP, F TUN; RUSSIA: A UKR, A WAR, F BOT, F SEV; TURKEY: A ANK, A BUL, A SER, F BLA, F CON
S1902M FRANCE for FRANCE:""".strip()

        fmt = "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk"
        formatter = sequence_formatter_factory(fmt, 2)
        regex = formatter.get_input_validation_regex(fmt, {})
        R.validate(regex, x)

    def test_message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk__no_messages(
        self,
    ):
        # Version 1
        x = """
F1901M
England: A YOR - NWY VIA; F NTH C A YOR - NWY; F NWG - BAR [EO_O]
France: A GAS - SPA; A SPA - POR; F MAO - ENG [EO_O]
Italy: A APU - TUN VIA; A ROM - VEN; F ION C A APU - TUN [EO_O]
Germany: A BUR - MUN; A KIE - BER; F DEN - SWE [EO_O]
Austria: A RUM H; A TYR - VEN; F ALB - GRE [EO_O]
Turkey: A ARM - SEV; A BUL - GRE; F ANK - BLA [EO_O]
Russia: A SIL - BOH; A UKR - GAL; F BOT - SWE; F SEV H [EO_O]
W1901A
England: F LON B [EO_O]
France: F BRE B; F MAR B [EO_O]
Italy: A VEN B [EO_O]
Germany: F KIE B [EO_O]
Austria: A TRI B [EO_O]
Turkey: F SMY B [EO_O]
Russia:  [EO_O] units: Austria: A RUM, A TRI, A TYR, F ALB; England: A NWY, F BAR, F LON, F NTH; France: A POR, A SPA, F BRE, F ENG, F MAR; Germany: A BER, A MUN, F DEN, F KIE; Italy: A ROM, A TUN, A VEN, F ION; Russia: A BOH, A GAL, F BOT, F SEV; Turkey: A ARM, A BUL, F BLA, F SMY [EO_STATE] S1902M France for Austria:""".strip()
        fmt = "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk"
        formatter = sequence_formatter_factory(fmt, 1)
        regex = formatter.get_input_validation_regex(fmt, {})
        R.validate(regex, x)

        # Version 2
        x = """
F1901M
ENGLAND: A YOR BEL VIA; F ENG BRE; F NTH C A YOR BEL
FRANCE: A BUR H; A MAR SPA; F MAO POR
ITALY: A APU H; A VEN H; F ION TUN
GERMANY: A KIE HOL; A MUN BUR; F DEN SWE
AUSTRIA: A BUD RUM; A GAL S A BUD RUM; F TRI VEN
TURKEY: A BUL SER; A CON BUL; F BLA S A BUD RUM
RUSSIA: A MOS WAR; A UKR RUM; F BOT SWE; F SEV S A UKR RUM
W1901A
ENGLAND: F LON B; F LVP B
FRANCE: A PAR B
ITALY: F NAP B
GERMANY: A KIE B; F BER B
AUSTRIA: A BUD B
TURKEY: A ANK B; F CON B
units: AUSTRIA: A BUD, A GAL, A RUM, F TRI; ENGLAND: A BEL, F BRE, F LON, F LVP, F NTH; FRANCE: A BUR, A PAR, A SPA, F POR; GERMANY: A HOL, A KIE, A MUN, F BER, F DEN; ITALY: A APU, A VEN, F NAP, F TUN; RUSSIA: A UKR, A WAR, F BOT, F SEV; TURKEY: A ANK, A BUL, A SER, F BLA, F CON
S1902M FRANCE for FRANCE:""".strip()
        fmt = "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk"
        formatter = sequence_formatter_factory(fmt, 2)
        regex = formatter.get_input_validation_regex(fmt, {})
        R.validate(regex, x)

    def test_retreats(self):
        # Version 1
        x = "retreats: Austria: {}; England: {}; France: {}; Germany: {'A MUN': ['RUH', 'KIE', 'BOH']}; Italy: {}; Russia: {}; Turkey: {}"
        R.validate(R.InputValidator([], "", {}, 1).RETREATS, x)

        # Version 2
        x = "retreats: ENGLAND: F BEL - PIC; RUSSIA: A RUM - GAL / UKR, F SWE - BOT / SKA"
        R.validate(R.InputValidator([], "", {}, 2).RETREATS, x)

    def test_retreats_power(self):
        # Version 1
        x = "Germany: {'A MUN': ['RUH', 'KIE', 'BOH']}"
        R.validate(R.InputValidator([], "", {}, 1).POWER_RETREATS, x)

        # Version 2
        x = "GERMANY: A MUN - RUH / KIE / BOH"
        R.validate(R.InputValidator([], "", {}, 2).POWER_RETREATS, x)

    def test_retreats_power_empty(self):
        # Version 1 (In V2, we don't have empty power retreats)
        x = "Germany: {}"
        R.validate(R.InputValidator([], "", {}, 1).POWER_RETREATS, x)

    def test_shortstate_with_retreats(self):
        # Version 1
        x = """units: Austria: A BUD, A MUN, A RUM, F TRI; England: A BEL, F BRE, F ENG, F IRI, F NWY; France: A BUR, A GAS, A PAR, F POR; Germany: *A MUN, A BER, A HOL, F DEN, F SWE; Italy: A TYR, A VEN, F ADR, F TUN; Russia: A UKR, A WAR, F BAL, F SEV; Turkey: A ARM, A GRE, A SER, F AEG, F BLA
retreats: Austria: {}; England: {}; France: {}; Germany: {'A MUN': ['RUH', 'KIE', 'BOH']}; Italy: {}; Russia: {}; Turkey: {} [EO_STATE]"""
        R.validate(R.InputValidator([], "", {}, 1).SHORTSTATE, x)

        # Version 2
        x = """units: AUSTRIA: A TRI; ENGLAND: A NWY, A STP, F BAR, F BOT, F NWG, F SWE; FRANCE: *A MUN, A BEL, A BUR, A ROM, A VEN, F ENG, F TUN, F TYS; GERMANY: *A BEL, A KIE, A MUN, A RUH, F BER; ITALY: A TUS, A TYR; RUSSIA: A LVN, A MOS, A VIE, F FIN; TURKEY: A BUD, A SER, A SEV, A UKR, F ADR, F AEG, F BLA, F ION
retreats: FRANCE: A MUN - BOH / SIL; GERMANY: A BEL - HOL"""
        R.validate(R.InputValidator([], "", {}, 2).SHORTSTATE, x)

    def test_longstate(self):
        x = """units: Austria: A BUD, A VIE, F TRI; England: A LVP, F EDI, F LON; France: A MAR, A PAR, F BRE; Germany: A BER, A MUN, F KIE; Italy: A ROM, A VEN, F NAP; Russia: A MOS, A WAR, F SEV, F STP/SC; Turkey: A CON, A SMY, F ANK
retreats: Austria: {}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {}
centers: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: BRE, MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY
homes: Austria: BUD, TRI, VIE; England: EDI, LON, LVP; France: BRE, MAR, PAR; Germany: BER, KIE, MUN; Italy: NAP, ROM, VEN; Russia: MOS, SEV, STP, WAR; Turkey: ANK, CON, SMY
builds: Austria: {'count': 0, 'homes': []}; England: {'count': 0, 'homes': []}; France: {'count': 0, 'homes': []}; Germany: {'count': 0, 'homes': []}; Italy: {'count': 0, 'homes': []}; Russia: {'count': 0, 'homes': []}; Turkey: {'count': 0, 'homes': []} [EO_STATE]"""
        R.validate(R.InputValidator([], "", {}, 1).LONGSTATE, x)

    def test_power_rollout_action_a(self):
        # Version 1
        x = """Germany: A KIE B; F BER B [EO_O]
S1902M
A HOL H; A KIE - RUH; A MUN S A BEL - BUR; F BER - BAL; F DEN - SWE [EO_O]"""
        R.validate(R.InputValidator([], "", {}, 1).POWER_ROLLOUT_ACTION, x)

        # Version 2
        x = """GERMANY: A KIE B; F BER B
S1902M
A HOL H; A KIE RUH; A MUN S A BEL BUR; F BER BAL; F DEN SWE"""
        R.validate(R.InputValidator([], "", {}, 2).POWER_ROLLOUT_ACTION, x)

    def test_power_rollout_action_r(self):
        # Version 1
        x = """Germany:  [EO_O]
W1901A
A BER B; F KIE B [EO_O]
S1902M
A BER - MUN; A DEN H; A MUN - BOH; F HOL S F KIE - HEL; F KIE - HEL [EO_O]"""
        R.validate(R.InputValidator([], "", {}, 1).POWER_ROLLOUT_ACTION, x)

        # Version 2
        x = (
            """GERMANY: """
            + """
W1901A
A BER B; F KIE B
S1902M
A BER MUN; A DEN H; A MUN BOH; F HOL S F KIE HEL; F KIE HEL"""
        )
        R.validate(R.InputValidator([], "", {}, 2).POWER_ROLLOUT_ACTION, x)

    def test_rollout_bilateral_action(self):
        x = """Germany: A KIE B; F BER B [EO_O]
S1902M
A HOL H; A KIE - RUH; A MUN S A BEL - BUR; F BER - BAL; F DEN - SWE [EO_O]
England: A LVP B; F LON B [EO_O]
S1902M
A BEL - BUR; A LVP - WAL; F BRE - MAO; F LON - ENG; F NTH - NWY [EO_O]"""
        R.validate(R.InputValidator([], "", {}, 1).ROLLOUT_BILATERAL_ACTION, x)

        # Version 2
        x = """GERMANY: A KIE B; F BER B
S1902M
A HOL H; A KIE RUH; A MUN S A BEL BUR; F BER BAL; F DEN SWE
ENGLAND: A LVP B; F LON B
S1902M
A BEL BUR; A LVP WAL; F BRE MAO; F LON ENG; F NTH NWY"""
        R.validate(R.InputValidator([], "", {}, 2).ROLLOUT_BILATERAL_ACTION, x)

    def test_power_action_empty(self):
        # Version 1
        x = """Germany:  [EO_O]"""
        R.validate(R.InputValidator([], "", {}, 1).POWER_ACTION, x)

        # Version 2
        x = """GERMANY: """
        R.validate(R.InputValidator([], "", {}, 2).POWER_ACTION, x)

    def test_rollout_bilateral_action_r(self):
        # Version 1
        x = """Germany:  [EO_O]
W1901A
A BER B; F KIE B [EO_O]
S1902M
A BER - MUN; A DEN H; A MUN - BOH; F HOL S F KIE - HEL; F KIE - HEL [EO_O]
England:  [EO_O]
W1901A
F LON B [EO_O]
S1902M
A WAL - PIC VIA; F ENG C A WAL - PIC; F LON S F ENG; F NWY - NTH [EO_O]"""
        R.validate(R.InputValidator([], "", {}, 1).ROLLOUT_BILATERAL_ACTION, x)

        # # Version 2
        x = (
            """GERMANY: """
            + """
W1901A
A BER B; F KIE B
S1902M
A BER MUN; A DEN H; A MUN BOH; F HOL S F KIE HEL; F KIE HEL
ENGLAND: """
            + """
W1901A
F LON B
S1902M
A WAL PIC VIA; F ENG C A WAL PIC; F LON S F ENG; F NWY NTH"""
        )
        R.validate(R.InputValidator([], "", {}, 2).ROLLOUT_BILATERAL_ACTION, x)

    def test_optional_sep(self):
        R.validate(rf"([ ]|^)hello", "hello")

    def test_draw_state_0(self):
        x = "\nDRAWS: "
        R.validate(R.InputValidator([], "", {}, 2).DRAW_STATE, x)

    def test_draw_state_1(self):
        x = "\nDRAWS: AUSTRIA"
        R.validate(R.InputValidator([], "", {}, 2).DRAW_STATE, x)

    def test_draw_state_2(self):
        x = "\nDRAWS: AUSTRIA ENGLAND"
        R.validate(R.InputValidator([], "", {}, 2).DRAW_STATE, x)

    def test_empty_retreats_list(self):
        x = """retreats: Austria: {'A RUM': []}; England: {}; France: {}; Germany: {}; Italy: {}; Russia: {}; Turkey: {'F ION': ['ADR', 'AEG', 'EAS']}"""
        R.validate(R.InputValidator([], "", {}, 1).RETREATS, x)

    def test_shortstate_with_centers(self):
        x = """units: AUSTRIA: A GAL, A RUM, A VIE, A WAR; ENGLAND: A RUH, F ENG, F IRI, F NTH, F STP/NC, F WES; FRANCE: A BEL, A BRE, A PAR, F POR; GERMANY: A HOL, A KIE, A MUN, F BER, F SWE; ITALY: A TRI, A VEN, F ADR, F ION; RUSSIA: A PRU, A UKR; TURKEY: A ARM, A GRE, A SER, A SEV, F AEG, F BLA
centers: AUSTRIA: BUD, RUM, VIE, WAR; ENGLAND: EDI, LON, LVP, NWY, STP; FRANCE: BEL, BRE, MAR, PAR, POR, SPA; GERMANY: BER, DEN, HOL, KIE, MUN, SWE; ITALY: NAP, ROM, TRI, TUN, VEN; RUSSIA: MOS; TURKEY: ANK, BUL, CON, GRE, SER, SEV, SMY"""
        R.validate(R.InputValidator([], "", {"include_centers_state": True}, 2).SHORTSTATE, x)

    def test_shortstate_with_centers_and_builds(self):
        x = """units: AUSTRIA: A GAL, A RUM, A VIE, A WAR; ENGLAND: A RUH, F ENG, F IRI, F NTH, F STP/NC, F WES; FRANCE: A BEL, A BRE, A PAR, F POR; GERMANY: A HOL, A KIE, A MUN, F BER, F SWE; ITALY: A TRI, A VEN, F ADR, F ION; RUSSIA: A PRU, A UKR; TURKEY: A ARM, A GRE, A SER, A SEV, F AEG, F BLA
centers: AUSTRIA: BUD, RUM, VIE, WAR; ENGLAND: EDI, LON, LVP, NWY, STP; FRANCE: BEL, BRE, MAR, PAR, POR, SPA; GERMANY: BER, DEN, HOL, KIE, MUN, SWE; ITALY: NAP, ROM, TRI, TUN, VEN; RUSSIA: MOS; TURKEY: ANK, BUL, CON, GRE, SER, SEV, SMY
builds: ENGLAND -1 FRANCE 1 ITALY 1 RUSSIA -1 TURKEY 1"""
        R.validate(
            R.InputValidator(
                [], "", {"include_centers_state": True, "include_builds_state": True}, 2
            ).SHORTSTATE,
            x,
        )

    # Disbling this test because we now allow single M phase in extended rollouts,
    # since this is sometimes seen at the end of games where the game completes.
    # def test_rollout_all_action_single_m_phase(self):
    #     # should fail -- does not rollout S1901M
    #     x = "F EDI NWG; F LON NTH; A LVP YOR"
    #     self.assertRaises(
    #         R.InputValidationException,
    #         R.validate,
    #         R.InputValidator([], "", {}, 3).ROLLOUT_ALL_ACTION,
    #         x,
    #     )

    def test_rollout_all_action_double_m_phase(self):
        # should not fail -- rolls out S1901M to F1901M
        x = "F EDI NWG; F LON NTH; A LVP YOR\nF1901M\nA YOR NWY VIA; F NTH C A YOR NWY; F NWG BAR"
        R.validate(R.InputValidator([], "", {}, 3).ROLLOUT_ALL_ACTION, x)

    def test_message_with_unix_timestamp(self):
        good = "86400 RUSSIA -> TURKEY: hello"
        bad = "1652330920 RUSSIA -> TURKEY: hello"
        regex = R.InputValidator([], "", {"add_sleep_times": True}, 3).MESSAGE
        R.validate(regex, good)
        self.assertRaises(R.InputValidationException, R.validate, regex, bad)
