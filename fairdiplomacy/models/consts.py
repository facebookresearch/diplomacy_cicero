#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List
import numpy as np
from fairdiplomacy.models.state_space import get_board_alignments, get_adjacency_matrix
from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import Power
from .preprocess_adjacency import preprocess_adjacency

LOCS = pydipcc.Game.LOC_STRS[:]
LOC_NAMES = set(
    [
        "ADRIATIC SEA",
        "AEGEAN SEA",
        "ALBANIA",
        "ANKARA",
        "APULIA",
        "ARMENIA",
        "BALTIC SEA",
        "BARENTS SEA",
        "BELGIUM",
        "BERLIN",
        "BLACK SEA",
        "BOHEMIA",
        "BREST",
        "BUDAPEST",
        "BULGARIA",
        "BULGARIA (EAST COAST)",
        "BULGARIA (SOUTH COAST)",
        "BURGUNDY",
        "CLYDE",
        "CONSTANTINOPLE",
        "DENMARK",
        "EASTERN MEDITERRANEAN",
        "EDINBURGH",
        "ENGLISH CHANNEL",
        "FINLAND",
        "GALICIA",
        "GASCONY",
        "GREECE",
        "GULF OF BOTHNIA",
        "GULF OF LYON",
        "HELGOLAND BIGHT",
        "HOLLAND",
        "IONIAN SEA",
        "IRISH SEA",
        "KIEL",
        "LIVERPOOL",
        "LIVONIA",
        "LONDON",
        "MARSEILLES",
        "MID-ATLANTIC OCEAN",
        "MOSCOW",
        "MUNICH",
        "NAPLES",
        "NORTH AFRICA",
        "NORTH ATLANTIC OCEAN",
        "NORTH SEA",
        "NORWAY",
        "NORWEGIAN SEA",
        "PARIS",
        "PICARDY",
        "PIEDMONT",
        "PORTUGAL",
        "PRUSSIA",
        "ROME",
        "RUHR",
        "RUMANIA",
        "SERBIA",
        "SEVASTOPOL",
        "SILESIA",
        "SKAGERRAK",
        "SMYRNA",
        "SPAIN",
        "SPAIN (NORTH COAST)",
        "SPAIN (SOUTH COAST)",
        "ST PETERSBURG",
        "ST PETERSBURG (NORTH COAST)",
        "ST PETERSBURG (SOUTH COAST)",
        "SWEDEN",
        "SWITZERLAND",
        "SYRIA",
        "TRIESTE",
        "TUNIS",
        "TUSCANY",
        "TYROLIA",
        "TYRRHENIAN SEA",
        "UKRAINE",
        "VENICE",
        "VIENNA",
        "WALES",
        "WARSAW",
        "WESTERN MEDITERRANEAN",
        "YORKSHIRE",
    ]
)
POWERS: List[Power] = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
POWER2IDX = {v: k for k, v in enumerate(POWERS)}
SEASONS = ["SPRING", "FALL", "WINTER"]
MAX_SEQ_LEN = 17  # can't have 18 orders in one phase or you've already won
N_SCS = 34  # number of supply centers
ADJACENCY_MATRIX = preprocess_adjacency(get_adjacency_matrix())
MASTER_ALIGNMENTS = np.stack(get_board_alignments(LOCS, False, 1, 81))
COASTAL_HOME_SCS = [
    "TRI",
    "EDI",
    "LVP",
    "LON",
    "BRE",
    "MAR",
    "BER",
    "KIE",
    "NAP",
    "ROM",
    "VEN",
    "SEV",
    "STP",
    "STP/NC",
    "STP/SC",
    "ANK",
    "CON",
    "SMY",
]
LOGIT_MASK_VAL = -1e8
