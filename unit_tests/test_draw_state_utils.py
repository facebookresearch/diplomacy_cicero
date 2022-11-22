#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from copy import deepcopy
import os
import json
import unittest

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import GameJson
from fairdiplomacy.utils.typedefs import build_message_dict
from fairdiplomacy.timestamp import Timestamp

from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    convert_all_timestamps,
    organize_game_by_phase,
)
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    get_gamejson_draw_state,
    get_last_timestamp_gamejson,
)

"""
Test fetching draw state from Game Json utils
"""


UNIT_TEST_DIR = os.path.dirname(__file__)
LAST_PHASE = "W1902A"
LAST_TIME = 160747351487475100


def load_game() -> GameJson:
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")
    with open(fle, "r") as f:
        game_json = json.load(f)

    game_json_by_phase = convert_all_timestamps(organize_game_by_phase(game_json))

    # Add some fake messages
    i = 0
    for power in POWERS:
        i += 1
        game_json_by_phase[LAST_PHASE]["messages"].append(
            build_message_dict(
                power, "ALL", "<DRAW>", LAST_PHASE, Timestamp.from_centis(LAST_TIME + i * 1000)
            )
        )
        i += 1
        game_json_by_phase[LAST_PHASE]["messages"].append(
            build_message_dict(
                power,
                "ALL",
                "Blah blah blah",
                LAST_PHASE,
                Timestamp.from_centis(LAST_TIME + i * 1000),
            )
        )

    return game_json_by_phase


class TestDrawStateUtils(unittest.TestCase):
    def test_get_last_timestamp(self):
        game_json = load_game()

        game_without_last_phase = deepcopy(game_json)
        del game_without_last_phase[LAST_PHASE]

        timestamp = get_last_timestamp_gamejson(game_without_last_phase)
        assert timestamp == Timestamp.from_centis(LAST_TIME)

        timestamp = get_last_timestamp_gamejson(game_json)
        assert timestamp == Timestamp.from_centis(LAST_TIME) + Timestamp.from_centis(
            14 * 1000
        )  # 7 powers, 2 messages each

    def test_get_draw_state(self):
        game_json = load_game()
        metadata = {"opt": {"include_draw_state": True}}

        draw_state = get_gamejson_draw_state(
            game_json, until_time=Timestamp.from_centis(LAST_TIME), metadata=metadata
        )
        powers_have_drawn = {k for k, v in draw_state.items() if v}

        assert not powers_have_drawn  # No draw yet
        for i in range(14):
            timestamp = Timestamp.from_centis(LAST_TIME + (i + 1) * 1000)
            draw_state = get_gamejson_draw_state(
                game_json, until_time=timestamp, metadata=metadata
            )
            powers_have_drawn = {k for k, v in draw_state.items() if v}
            idx = (i // 2) % 7
            pows = POWERS[: idx + 1]
            assert set(pows) == powers_have_drawn
