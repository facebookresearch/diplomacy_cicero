#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Special token utilities
"""
import os
from parlai.utils import logging
from typing import List


SPECIAL_TOKENS_V1 = [
    "NON_SILENCE",
    "[EO_STATE]",
    "[REDACTED]",
    "Austria",
    "England",
    "Germany",
    "AUSTRIA",
    "ENGLAND",
    "GERMANY",
    "SILENCE",
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
    "[EO_O]",
    "[EO_M]",
    "Italy",
    "ITALY",
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
]


SPECIAL_TOKENS_V2_ADDITIONS = [
    "BAD",  # Bad messages
    "ALL-UNK",  # Unknown phases
    "ANON",  # Anonymous game
    "NON-ANON",  # Non-anonymous game
    "PUBLIC",  # Public draw votes
    "PRIVATE",  # Private draw votes
    "NODRAWS",  # No draw data
    "HASDRAWS",  # Has draw data available
    "PPSC",  # Points per supply center
    "SOS",  # Sum of squares
    "WTA",  # Winner take all
    "->",
]

# tokens which are no longer needed (unused) in v2
SPECIAL_TOKENS_V2_DELETIONS = ["SILENCE", "NON_SILENCE", "[EO_STATE]", "[EO_O]", "[EO_M]"]


def _load_special_tokens(special_tokens_lst: List[str]) -> List[str]:
    """
    Sort special tokens in the proper order
    """
    special_tokens = sorted(special_tokens_lst, key=len, reverse=True)
    return special_tokens


def load_special_tokens():
    """
    Load v1 special tokens
    """
    logging.success("Loading v1 special tokens")
    return _load_special_tokens(SPECIAL_TOKENS_V1)


def load_special_tokens_v2():
    """
    Load with the additional v2 special tokens
    """
    logging.success("Loading v2 special tokens")
    new_lst = [
        x for x in SPECIAL_TOKENS_V1 if x not in SPECIAL_TOKENS_V2_DELETIONS
    ] + SPECIAL_TOKENS_V2_ADDITIONS

    return _load_special_tokens(new_lst)
