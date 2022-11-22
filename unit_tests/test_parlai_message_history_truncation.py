#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import unittest

from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power, GameJson

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageHistoryBuilder
from parlai_diplomacy.utils.game2seq.format_helpers.misc import organize_game_by_phase


UNIT_TEST_DIR = os.path.dirname(__file__)
ENGLAND = "ENGLAND"
RUSSIA = "RUSSIA"
PHASE = "W1902A"


def load_game():
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")

    with open(fle, "r") as f:
        game_object = Game.from_json(f.read())

    game_json = json.loads(game_object.to_json())
    game_json_by_phase = organize_game_by_phase(game_json)

    return game_json_by_phase


class TestMessageHistoryTruncation(unittest.TestCase):
    def _get_message_history(self, game_json: GameJson, speaker: Power, truncation: int):
        return MessageHistoryBuilder(1).extract_message_history_from_game_json(
            game_json, PHASE, speaker, truncation=truncation
        )

    def test_truncation(self):
        game_json = load_game()

        # England
        history = self._get_message_history(game_json, ENGLAND, 10)
        assert len(history) == 3
        assert history[0][0]["phase"] == "F1902M"

        history = self._get_message_history(game_json, ENGLAND, 100)
        assert len(history) == 6
        assert history[0][0]["phase"] == "F1901M"

        history = self._get_message_history(game_json, ENGLAND, 1000)
        assert len(history) == 7
        assert history[0][0]["phase"] == "S1901M"

        # Russia
        history = self._get_message_history(game_json, RUSSIA, 10)
        assert len(history) == 3
        assert history[0][0]["phase"] == "F1902M"

        history = self._get_message_history(game_json, RUSSIA, 100)
        assert len(history) == 4
        assert history[0][0]["phase"] == "S1902M"

        history = self._get_message_history(game_json, RUSSIA, 1000)
        assert len(history) == 7
        assert history[0][0]["phase"] == "S1901M"
