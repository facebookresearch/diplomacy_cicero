#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import logging
import os
import re
import sqlite3
import json
from collections import defaultdict, namedtuple
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter  # type: ignore
from typing import Optional, Dict, Set, Any, List

from fairdiplomacy.typedefs import Power
from fairdiplomacy.data.build_dataset import (
    find_all_games,
    check_is_good_game,
    load_messages,
    GoodGameCheckException,
)


"""
Before building the metadata, please ensure that the dataset has been built via `build_dataset.py`.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

TABLE_GAMES = "redacted_games"
TABLE_MEMBERS = "redacted_members"

STATUS_TO_RANK = {"Won": 0, "Drawn": 1, "Survived": 2, "Defeated": 3, "Resigned": 3}

COUNTRY_ID_TO_POWER: Dict[int, Power] = {
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}


MemberRow = namedtuple("MemberRow", "user game country status")


def string_to_none_or_int(string: str):
    if string == "None":
        return None
    return int(string)


def compute_logit_ratings(game_stats, wd, userids: List[int]):
    userid_to_reluserid = {userid: i for i, userid in enumerate(userids)}

    # 1. construct dataset

    # 1a. find all u1 > u2 pairs from the game stats
    dataset = []
    POWERS = COUNTRY_ID_TO_POWER.values()
    for game in game_stats.values():
        for pwr0 in POWERS:
            if game[pwr0] is None:
                continue
            p0 = game[pwr0]["points"]
            id0 = userid_to_reluserid[game[pwr0]["id"]]
            for pwr1 in POWERS:
                if game[pwr1] is None:
                    continue
                p1 = game[pwr1]["points"]
                id1 = userid_to_reluserid[game[pwr1]["id"]]
                if pwr0 == pwr1:
                    continue
                if p0 > p1:
                    dataset.append((id0, id1))
                if p0 < p1:
                    dataset.append((id1, id0))

    # 1b. shuffle
    dataset = torch.tensor(dataset, dtype=torch.long)
    dataset = dataset[torch.randperm(len(dataset))]

    # 1c. split into train and val
    N_val = int(len(dataset) * 0.05)
    val_dataset = dataset[:N_val]
    train_dataset = dataset[N_val:]

    user_scores = Parameter(torch.zeros(len(userids)))
    optimizer = optim.Adagrad([user_scores], lr=1e0)

    # cross entropy loss where P(win) = softmax(score0, score1)
    def L(dataset):
        return -user_scores[dataset].log_softmax(-1)[:, 0].mean()

    # run gradient descent to optimize the loss
    for epoch in range(100):
        optimizer.zero_grad()
        train_loss = L(train_dataset) + wd * (user_scores ** 2).mean()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = L(val_dataset)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.5f}  Val Loss: {val_loss:.5f} Mean: {user_scores.abs().mean():.5f} ( {user_scores.min():.5f} - {user_scores.max():.5f} ) "
            )

    return user_scores.tolist()


def make_game_stats_and_user_stats(
    db,
    sqlid_to_jsonid_map: Dict[Optional[int], Optional[int]],
    game_jsonids: Set[int],
    ratings_wd: float,
):
    member_rows = [
        MemberRow(*row)
        for row in db.execute(
            f"SELECT hashed_userID, hashed_gameID, countryID, status FROM {TABLE_MEMBERS}"
        ).fetchall()
    ]
    db_game_metadata = {
        game_id: dict(
            press_type=press_type,
            pot_type=pot_type,
            phase_minutes=phase_minutes,
            anon=anon,
            missing_player_policy=missing_player_policy,
            draw_type=draw_type,
        )
        for game_id, press_type, pot_type, phase_minutes, anon, missing_player_policy, draw_type in db.execute(
            f"SELECT hashed_id, pressType, potType, phaseMinutes, anon, missingPlayerPolicy, drawType from {TABLE_GAMES}"
        ).fetchall()
    }

    game_jsonid_to_sqlid: Dict[int, int] = {
        jsonid: sqlid
        for sqlid, jsonid in sqlid_to_jsonid_map.items()
        if jsonid in game_jsonids and sqlid is not None
    }
    assert len(game_jsonid_to_sqlid) == len(game_jsonids)
    game_sqlids: Set[int] = {v for v in game_jsonid_to_sqlid.values() if v is not None}

    member_rows = [row for row in member_rows if row.game in game_sqlids]

    user_ids = sorted(set(int(r.user) for r in member_rows))
    user_stats: Dict[int, Dict] = {user_id: defaultdict(float) for user_id in user_ids}

    member_dict = {(r.game, int(r.country)): r for r in member_rows}

    print(f"Found {len(game_jsonids)} games, {len(user_ids)} users")

    WIN_STATI = ("Won", "Drawn")
    game_stats = {}
    for game_jsonid, game_sqlid in tqdm(game_jsonid_to_sqlid.items()):
        winners = []
        for country_id in range(1, 7 + 1):
            k = (game_sqlid, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                user_id = int(member_row.user)
                this_user_stats = user_stats[user_id]
                this_user_stats["total"] += 1
                if member_row.status not in STATUS_TO_RANK:
                    continue
                this_user_stats[member_row.status] += 1
                if member_row.status in WIN_STATI:
                    winners.append(this_user_stats)

        # allot points to winners
        for winner in winners:
            winner["total_points"] += 1.0 / len(winners)

        # check if messages exist for game
        messages = load_messages(db, game_sqlid)

        # collect stats
        this_game_stats: Dict[str, Any] = {"id": game_sqlid}
        this_game_db_metadata = db_game_metadata[game_sqlid]
        for key in this_game_db_metadata:
            this_game_stats[key] = this_game_db_metadata[key]
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_sqlid, country_id)
            if k in member_dict:
                member_row = member_dict[k]
                u = int(member_row.user)
                this_game_stats[pwr] = {
                    "id": u,
                    "points": 1.0 / len(winners) if member_row.status in WIN_STATI else 0,
                    "status": member_row.status,
                }
                if messages is not None:
                    this_game_stats[pwr]["messages_sent"] = len(
                        [x for x in messages if x["fromCountryID"] == country_id]
                    )
                else:
                    this_game_stats[pwr]["messages_sent"] = 0
            else:
                this_game_stats[pwr] = None

        # collect stats that we use to check reliablity of phase assignment of messages
        message_phase_stats = {}
        if messages is not None:
            for msg in messages:
                phase = msg["phase"]
                if phase not in message_phase_stats:
                    message_phase_stats[phase] = 0
                message_phase_stats[phase] += 1
        this_game_stats["message_phase_stats"] = message_phase_stats

        # count stats about redaction tokens and messages and characters in messages
        num_redactions = 0
        num_messages = 0
        num_words = 0
        num_characters = 0
        if messages is not None:
            for msg in messages:
                num_redactions += len(re.findall(r"\[\d+\]", msg["message"]))
                num_messages += 1
                num_words += len(msg["message"].split(" "))
                num_characters += len(msg["message"])
        this_game_stats["message_stats"] = dict(
            num_redactions=num_redactions,
            num_messages=num_messages,
            num_words=num_words,
            num_characters=num_characters,
        )

        game_stats[game_sqlid] = this_game_stats

    print("Computing logit scores")
    ratings = compute_logit_ratings(game_stats, ratings_wd, user_ids)
    assert len(ratings) == len(user_stats) == len(user_ids)
    for i in range(len(user_ids)):
        user_id = user_ids[i]
        user_stats[user_id]["logit_rating"] = ratings[i]

    print("Adding final stats")
    for game_id in game_stats.keys():
        this_game_stats = game_stats[game_id]
        if this_game_stats is None:
            continue
        for country_id in range(1, 7 + 1):
            pwr = COUNTRY_ID_TO_POWER[country_id]
            k = (game_id, country_id)
            if k in member_dict:
                this_game_stats[pwr].update(**user_stats[int(member_dict[k].user)])

    assert all(user_stats[u]["total"] > 0 for u in user_ids)
    for u in user_ids:
        user_stats[u]["id"] = u

    # map back to the json ID
    game_stats_keyed_on_json_id = {sqlid_to_jsonid_map[k]: v for k, v in game_stats.items()}

    return game_stats_keyed_on_json_id, user_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Dump output json to this file")
    parser.add_argument(
        "--db-path", help="Path to SQLITE db file", required=True,
    )
    parser.add_argument(
        "--game-folder",
        help="Directory containing Game JSONS. Generated by build_dataset.py",
        required=True,
    )
    parser.add_argument(
        "--ratings-wd",
        type=float,
        default=0.05,
        help="Weight decay for ratings logit computation.",
    )
    args = parser.parse_args()

    db = sqlite3.connect(args.db_path)

    hash_fle = os.path.join(args.game_folder, "hash_to_sqlid_to_jsonid.txt")
    if not os.path.isfile(hash_fle):
        raise RuntimeError(
            f"Hash file not found in {args.game_folder}. Please ensure build_dataset.py ran successfully before building metdata"
        )

    sql2json = {}
    with open(hash_fle, "r") as f:
        lines = [x.split(" ") for x in f.read().splitlines()]
        for _, sqlid, game_json_id in lines:
            sql2json[string_to_none_or_int(sqlid)] = string_to_none_or_int(game_json_id)

    game_jsonids: Set[int] = {
        int(g.split("game_")[1].split(".json")[0])
        for g in glob(args.game_folder + "/all_games/game_*.json")
    }

    game_stats, user_stats = make_game_stats_and_user_stats(
        db, sql2json, game_jsonids, ratings_wd=args.ratings_wd
    )

    with open(args.out, "w") as f:
        json.dump(game_stats, f)

    with open(os.path.dirname(args.out) + "/user_stats.json", "w") as f:
        json.dump(user_stats, f)
