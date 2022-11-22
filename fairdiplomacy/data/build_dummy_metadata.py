#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict, namedtuple
import argparse
import logging
import os
import sqlite3
import json
import glob
import re

import torch
import torch.nn as nn
import torch.optim as optim

from fairdiplomacy.models.consts import POWERS

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s")


def make_ratings_table(game_folder):
    game_jsons = glob.glob(os.path.join(game_folder, "**/game_*.json"), recursive=True)
    pattern = re.compile("game_(\\d+).json")
    game_ids = []
    for f in game_jsons:
        m = pattern.search(f)
        if m:
            game_ids.append(int(m.group(1)))

    game_stats = {
        game_id: {
            "id": game_id,
            "press_type": "NoPress",
            **{
                pwr: {"id": -1, "points": -1, "status": "NoPress", "logit_rating": 0,}
                for pwr in POWERS
            },
        }
        for game_id in game_ids
    }
    return game_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game-path", required=True, help="Path to folder with game_XXX.json files."
    )
    parser.add_argument("--out", required=True, help="Dump output pickle to this file")

    args = parser.parse_args()

    game_stats = make_ratings_table(args.game_path)

    with open(args.out, "w") as f:
        json.dump(game_stats, f)
