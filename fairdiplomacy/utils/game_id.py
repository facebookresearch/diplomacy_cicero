#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import re

from fairdiplomacy.typedefs import GameID


def extract_game_id_str(path) -> str:
    # "/path/to/game_1234.json" -> "game_1234.json"
    return path.rsplit("/", 1)[-1]


def extract_game_id_int(path) -> GameID:
    # "/path/to/game_1234.json" -> 1234
    return int(re.findall(r"([0-9]+)", path)[-1])
