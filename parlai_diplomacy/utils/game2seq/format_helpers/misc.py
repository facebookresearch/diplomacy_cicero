#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Utils for converting game JSONs into string sequences for model input
"""
import time
import json
from typing import Dict, Any

import parlai.utils.logging as logging

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import GameJson, Phase, Power, Timestamp
from parlai_diplomacy.utils.game2seq.typing import Metadata


COUNTRY_ID_TO_POWER = {
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}
POWER_TO_COUNTRY_ID = {v: k for k, v in COUNTRY_ID_TO_POWER.items()}

POT_TYPE_CONVERSION = {
    "Points-per-supply-center": "PPSC",
    "Sum-of-squares": "SOS",
    "Unranked": "U",
    "Winner-takes-all": "WTA",
}

CORRUPTED_NEWLINE = "~N~"

INF_SLEEP_TIME = Timestamp.from_seconds(1000 * 365 * 86400)


class ParlaiDecodingError(RuntimeError):
    pass


def organize_game_by_phase(game_json: GameJson) -> Dict[str, Any]:
    """
    Organize a single game JSON by phase.
    """
    new_data = {}
    for phase in game_json["phases"]:
        new_data[phase["name"]] = phase

    return new_data


def convert_all_timestamps(game_json):
    """
    Convert all game timestamps to Timestamp objects, assuming they are currently
    stored as integer centiseconds (which is how they're stored on disk).

    """
    for phase in game_json.keys():
        for msg in game_json[phase]["messages"]:
            msg["original_time_sent"] = msg["time_sent"]
            msg["time_sent"] = Timestamp.from_centis(msg["time_sent"])

    return game_json


def load_json(path: str) -> Dict[Any, Any]:
    """
    Simple utility for loading a JSON from a path
    """
    start_time = time.time()
    with open(path, "r") as f:
        data = json.load(f)
    tot_time = time.time() - start_time
    logging.log(f"Time to load json: {round(tot_time, 2)} seconds", level=logging.SPAM)

    return data


def get_example_key(game_id: int, speaker: Power, phase: Phase, ind: int):
    """
    Key to retreiving the train pseudo order
    """
    # N.B. player_id is NOT the index in POWERS !!!
    player_id = POWER_TO_COUNTRY_ID[speaker]
    return f"{game_id}-{phase}-{player_id}-{ind}"


def get_current_time():
    """Return current unix time in microseconds"""
    return int(time.time() * 1e6)


def verify_phase_format(phase: str):
    """
    Asserts that a phase string is valid.
    """
    if phase != "COMPLETED":
        try:
            int(phase[1:5])
        except ValueError:
            raise ValueError(f"{phase[1:5]} not a valid year")

        season = phase[0]
        phase_type = phase[5]
        assert season in {"S", "F", "W"}, f"{season} not a valid season"
        assert phase_type in {"M", "R", "A"}, f"{phase_type} not a valid phase type"


def format_player_prompt_token(phase: Phase, player: Power, metadata: Metadata):
    """
    Format the player prompt token.

    Args:
        phase: current shorthand phase (e.g. S1901M)
        player: player that we are prompting (e.g. ENGLAND or England)
        metadata: game metadata
    """
    # verify input format
    assert player.upper() in POWERS, f"{player.upper()} not a valid power"
    verify_phase_format(phase)

    # speaker
    prompt = f"{phase} " + (
        player.capitalize() if metadata.get("task_version", 1) <= 1 else player
    )
    if not metadata:
        return f"{prompt}:"

    # rating
    rating = None
    if metadata["opt"].get("include_player_ratings"):
        rating = metadata["power_metadata"][player.upper()]["rating"]
        prompt = f"{prompt} {rating}"

    # chattiness
    chattiness = None
    if metadata["opt"].get("include_player_chattiness"):
        chattiness = metadata["power_metadata"][player.upper()]["chattiness"]
        prompt = f"{prompt} {chattiness}"

    # game info
    if metadata["opt"].get("include_game_info", False):
        # anon or non-anon
        anon = metadata["anon"]
        prompt = f"{prompt} {anon}"
        # length
        phase_min = metadata["phase_minutes"]
        prompt = f"{prompt} {phase_min}min"
        # pot_type
        pot_type = metadata["pot_type"]
        prompt = f"{prompt} {pot_type}"
        # all unknowns
        if metadata["all_unknowns"]:
            prompt = f"{prompt} ALL-UNK"

    # draw info
    if metadata["opt"].get("include_draw_info", False):
        # draw type
        draw_type = metadata["draw_type"]
        prompt = f"{prompt} {draw_type}"
        # data has draws
        has_draws = metadata["has_draw_votes"]
        draw_tok = "NODRAWS" if not has_draws else "HASDRAWS"
        prompt = f"{prompt} {draw_tok}"

    # dialogue filter
    if metadata.get("two_powers_dialogue"):
        prompt = "{} two powers".format(prompt)

    # task token
    if "task_token" in metadata["opt"]:
        task_token = metadata["opt"]["task_token"]
        prompt = f"{prompt} {task_token}"

    return f"{prompt}:"


def add_recipient_to_prompt_token(
    curr_prompt_token: str, sender: Power, recipient: Power, metadata: Metadata, version: int,
) -> str:
    """
    Add the recipient to the prompt token.

    For example, "S190M England:" becomes "S1901M England -> France:"
    """
    if version <= 1:
        sender = sender.capitalize()
        recipient = recipient.capitalize()

    assert sender in curr_prompt_token

    new_prompt_token = curr_prompt_token.replace(sender, f"{sender} -> {recipient}")

    return new_prompt_token


def modify_input_prompt_for_power(seq_input: str, power: Power, version: int) -> str:
    assert seq_input[-1] == ":", seq_input
    if version <= 1:
        power = power.capitalize()
    return seq_input[:-1] + f" for {power}:"


def add_end_token(text: str, end_token: str) -> str:
    """
    Add end token
    Note: there is an inverse function called `remove_end_token`,
    if `add_end_token` is updated, `remove_end_token` also need to be updated!!!
    """
    return f"{text} {end_token}"


def remove_end_token(text: str, end_token: str) -> str:
    """
    Remove end token, the inverse of add_end_token
    Note: there is an inverse function called `add_end_token`,
    if `remove_end_token` is updated, `add_end_token` also need to be updated!!!
    """
    return text.replace(f" {end_token}", "")


def get_output_type(task_name: str) -> str:
    task_parts = task_name.split("_")
    assert task_parts[-1] in ("chunk", "evaldump"), f"Task name {task_name} not properly formatted"
    output_type = task_parts[-2]

    return output_type


def get_input_format(task_name: str) -> str:
    task_parts = task_name.split("_")[:-2]
    return "_".join(task_parts)


def uncorrupt_newlines(msg_txt: str) -> str:
    """
    Replace corrupted newlines (~N~) with newline characters
    """
    corrupted_newline_cnt = msg_txt.count(CORRUPTED_NEWLINE)
    if corrupted_newline_cnt > 0:
        for i in reversed(range(1, corrupted_newline_cnt + 1)):
            # replace `i` corrupted newlines in a row
            corrupted_newlines = " " + " ".join([CORRUPTED_NEWLINE for _ in range(i)]) + " "
            fixed_newlines = "\n" * i
            msg_txt = msg_txt.replace(corrupted_newlines, fixed_newlines)

    return msg_txt


def corrupt_newlines(msg_txt: str) -> str:
    """
    Replace new lines characters with corrupted newline characters (~N~)
    """
    newline_cnt = msg_txt.count("\n")
    if newline_cnt > 0:
        for i in reversed(range(1, newline_cnt + 1)):
            # replace `i` newlines in a row
            corrupted_newlines = " " + " ".join([CORRUPTED_NEWLINE for _ in range(i)]) + " "
            newlines = "\n" * i
            msg_txt = msg_txt.replace(newlines, corrupted_newlines)

    return msg_txt


# Top 60 ampersand tokens appearing in the training data with obvious replacements
AMPERSAND_REPLACEMENTS = [
    ("don&t", "don't"),
    ("can&t", "can't"),
    ("it&s", "it's"),
    ("I&m", "I'm"),
    ("didn&t", "didn't"),
    ("won&t", "won't"),
    ("&sigh&", "*sigh*"),
    ("I&ll", "I'll"),
    ("&not&", "*not*"),
    ("&and&", "*and*"),
    ("doesn&t", "doesn't"),
    ("let&s", "let's"),
    ("isn&t", "isn't"),
    ("&edit&", "*edit*"),
    ("It&s", "It's"),
    ("i&ll", "i'll"),
    ("that&s", "that's"),
    ("&you&", "*you*"),
    ("Let&s", "Let's"),
    ("i&m", "i'm"),
    ("Don&t", "Don't"),
    ("&really&", "*really*"),
    ("&I&", "*I*"),
    ("&think&", "*think*"),
    ("&shrug&", "*shrug*"),
    ("That&s", "That's"),
    ("&will&", "*will*"),
    ("&is&", "*is*"),
    ("&could&", "*could*"),
    ("&might&", "*might*"),
    ("haven&t", "haven't"),
    ("&cough&", "*cough*"),
    ("I&d", "I'd"),
    ("&very&", "*very*"),
    ("&me&", "*me*"),
    ("wasn&t", "wasn't"),
    ("what&s", "what's"),
    ("&that&", "*that*"),
    ("&should&", "*should*"),
    ("wouldn&t", "wouldn't"),
    ("you&re", "you're"),
    ("&do&", "*do*"),
    ("&can&", "*can*"),
    ("you&ll", "you'll"),
    ("I&ve", "I've"),
    ("&need&", "*need*"),
    ("&shrugs&", "*shrugs*"),
    ("aren&t", "aren't"),
    ("&both&", "*both*"),
    ("we&ll", "we'll"),
    ("he&s", "he's"),
    ("&if&", "*if*"),
    ("&facepalm&", "*facepalm*"),
    ("&much&", "*much*"),
    ("&grins&", "*grins*"),
    ("&all&", "*all*"),
    ("&too&", "*too*"),
    ("&only&", "*only*"),
    ("&never&", "*never*"),
    ("&any&", "*any*"),
]


def uncorrupt_ampersands(msg_txt: str) -> str:
    """
    Uncorrupt certain tokens with ampersands

    We replace the top 60 most frequently appearing tokens with ampersand and an obvious replacement
    """
    if "&" not in msg_txt:
        # fail fast here
        return msg_txt

    replace_dct = {k: v for k, v in AMPERSAND_REPLACEMENTS}

    split_msg = msg_txt.split(" ")
    for i in range(len(split_msg)):
        if split_msg[i] in replace_dct:
            split_msg[i] = replace_dct[split_msg[i]]

    return " ".join(split_msg)


def remove_trailing_carriage_return(msg_txt: str) -> str:
    """
    Remove trailing carraige return.

    In the latest version of the dataset (as of April 2022) there is a "\r" at the
    end of messages. Remove it.
    """
    if msg_txt.endswith("\r"):
        return msg_txt[:-1]
    return msg_txt
