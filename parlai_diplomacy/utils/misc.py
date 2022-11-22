#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.typedefs import GameJson
from glob import glob
import json
import os
import sys
from typing import Dict, List, Tuple, TypeVar, Iterator, Union

from fairdiplomacy.pydipcc import Game
from parlai_diplomacy.utils import datapath_constants as constants

"""
File for miscellaneous utility functions.

These utilities are *not* specific to the game of Diplomacy.
"""


#####################################################
#  Utilities for iterating through the game directory
#####################################################
def game_iter(
    obj: bool = True, full_press: bool = False
) -> Iterator[Tuple[Union[Game, GameJson], str]]:
    """
    Iterator for all games. Returns tuple of game and game_path.

    - obj: return game object if True, game JSON is false
    - full_press: only return full press games
    - test: only return test games
    """
    latest_data_dir = constants.LATEST_DATA_DIR

    folder = "full_press_games" if full_press else "all_games"
    for game_path in glob(os.path.join(latest_data_dir, folder, "game_*.json")):
        with open(game_path, "r") as f:
            if obj:
                # Return game object
                game = Game.from_json(f.read())
            else:
                # Return game JSON
                game = json.loads(f.read())
        yield game, game_path


def test_game_iter(obj: bool = True) -> Iterator[Tuple[Union[Game, GameJson], str]]:
    """
    Iterator for test games

    - obj: return game object if True, game JSON is false
    """
    with open(constants.TEST_ID_PATH, "r") as f:
        test_ids = [int(x) for x in f.read().splitlines()]

    latest_data_dir = constants.LATEST_DATA_DIR
    for game_id in test_ids:
        game_path = os.path.join(latest_data_dir, "all_games", f"game_{game_id}.json")
        if not os.path.exists(game_path):
            continue
        if obj:
            # Return game object
            game = Game.from_json(f.read())
        else:
            # Return game JSON
            game = json.loads(f.read())
        yield game, game_path


#####################################################
#  Utilities for getting dictionary order
#####################################################


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def version_info():
    assert (
        sys.version_info.major == 3 and sys.version_info.minor >= 7
    ), "Dicts don't guarantee order"


def last_dict_item(D: Dict[_T1, _T2]) -> Tuple[_T1, _T2]:
    version_info()
    return list(D.items())[-1]


def last_dict_value(D: Dict[_T1, _T2]) -> _T2:
    return last_dict_item(D)[1]


def last_dict_key(D: Dict[_T1, _T2]) -> _T1:
    return last_dict_item(D)[0]


def get_ordered_dict_keys(D: Dict[_T1, _T2]) -> List[_T1]:
    version_info()
    return list(D.keys())


#####################################################
#  Utilities for text coloring
#####################################################


class color:
    """
    Colors for highlighting text on terminal

    More here: https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
    """

    BLACK = "\u001b[30m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    BG_WHITE = "\u001b[47m"
    BG_RED = "\u001b[41;1m"
    BG_GREEN = "\u001b[42m"
    RAINBOW = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE]

    def _color(self, text: str, color: str) -> str:
        return f"{color}{text}{self.END}"

    def bold(self, text: str) -> str:
        return self._color(text, self.BOLD)

    def blue(self, text: str) -> str:
        return self._color(text, self.BLUE)

    def yellow(self, text: str) -> str:
        return self._color(text, self.YELLOW)

    def red(self, text: str) -> str:
        return self._color(text, self.RED)

    def purple(self, text: str) -> str:
        return self._color(text, self.PURPLE)

    def green(self, text: str) -> str:
        return self._color(text, self.GREEN)

    def cyan(self, text: str) -> str:
        return self._color(text, self.CYAN)


LOCATIONS = [
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
    "SPA/NC",
    "SPA",
    "HOL",
    "STP/SC",
    "SIL",
    "MUN",
    "BUL/SC",
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
    "STP/SC",
    "VIE",
    "BUL/EC",
    "LVP",
    "GAS",
    "BAL",
    "SPA/SC",
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
