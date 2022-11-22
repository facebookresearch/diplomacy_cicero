#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
from parlai_diplomacy.utils import special_tokens as st

"""
Test special token loading utilities.

This will ensure that changes cannot be made to these lists without breaking main.
"""


class TestSpecialTokens(unittest.TestCase):
    def test_special_tokens(self):
        v1_toks = st.load_special_tokens()
        # Ensure this list is immutable
        self.assertEquals(v1_toks, st.SPECIAL_TOKENS_V1)
        # Ensure this list is sorted correctly
        self.assertEquals(v1_toks, sorted(v1_toks, key=len, reverse=True))

        v2_toks = st.load_special_tokens_v2()
        # Ensure this list is immutable
        self.assertEquals(
            v2_toks,
            [
                "[REDACTED]",
                "NON-ANON",
                "HASDRAWS",
                "Austria",
                "England",
                "Germany",
                "AUSTRIA",
                "ENGLAND",
                "GERMANY",
                "ALL-UNK",
                "PRIVATE",
                "NODRAWS",
                "France",
                "Russia",
                "Turkey",
                "FRANCE",
                "RUSSIA",
                "TURKEY",
                "SPA/NC",
                "STP/SC",
                "BUL/SC",
                "STP/NC",
                "BUL/EC",
                "SPA/SC",
                "PUBLIC",
                "Italy",
                "ITALY",
                "ANON",
                "PPSC",
                "VEN",
                "ALB",
                "KIE",
                "BAR",
                "NWG",
                "TUS",
                "EDI",
                "GRE",
                "PRU",
                "BUD",
                "HEL",
                "IRI",
                "SKA",
                "GAL",
                "TYS",
                "RUM",
                "NAP",
                "SMY",
                "LON",
                "ADR",
                "BOH",
                "EAS",
                "BEL",
                "ANK",
                "MAR",
                "APU",
                "TUN",
                "PIE",
                "SPA",
                "HOL",
                "SIL",
                "MUN",
                "YOR",
                "LYO",
                "ION",
                "TYR",
                "CON",
                "WES",
                "ENG",
                "NAF",
                "UKR",
                "AEG",
                "SER",
                "ROM",
                "WAR",
                "BUR",
                "VIA",
                "VIE",
                "LVP",
                "GAS",
                "BAL",
                "BUL",
                "BLA",
                "TRI",
                "ARM",
                "SWE",
                "RUH",
                "NTH",
                "NWY",
                "BOT",
                "DEN",
                "NAO",
                "WAL",
                "BER",
                "PIC",
                "MOS",
                "STP",
                "BRE",
                "PAR",
                "SEV",
                "MAO",
                "SYR",
                "FIN",
                "LVN",
                "CLY",
                "POR",
                "BAD",
                "SOS",
                "WTA",
                "->",
            ],
        )
        # Ensure this list is sorted correctly
        self.assertEquals(v2_toks, sorted(v2_toks, key=len, reverse=True))
