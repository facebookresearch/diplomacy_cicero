#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import unittest
import random
from typing import Tuple

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryBuilder,
    is_draw_msg,
    is_unvote_draw_msg,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import organize_game_by_phase

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import GameJson
from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN
from fairdiplomacy.timestamp import Timestamp


UNIT_TEST_DIR = os.path.dirname(__file__)

"""
Test building all possible message histories
"""


def load_game() -> Game:
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")
    with open(fle, "r") as f:
        game_object = Game.from_json(f.read())
        game_object = game_object.rolled_back_to_phase_end("F1901M")

    # add some draw/undraw vote messages to the history
    for i, power in enumerate(POWERS):
        msg = (
            DRAW_VOTE_TOKEN if random.choice([0, 1]) else UNDRAW_VOTE_TOKEN
        )  # sometimes add draw, sometimes add undraw
        game_object.add_message(
            power, "ALL", msg, Timestamp.from_centis(160582615148573000 + i * 1000)
        )

    return game_object


class TestMessageHistoryBuilding(unittest.TestCase):
    def _setup_test(self, version: int) -> Tuple[Game, GameJson, MessageHistoryBuilder]:
        game = load_game()
        game_json_by_phase = organize_game_by_phase(json.loads(game.to_json()))
        builder = MessageHistoryBuilder(version)

        return game, game_json_by_phase, builder

    def test_build_v1(self):
        game, game_json, builder = self._setup_test(1)
        for power in POWERS:
            message_histories = builder.build_all_possible_message_histories(
                game.current_short_phase, power, game_json
            )
            if power == POWERS[-1]:
                # Add some checks on the message history length here
                self.assertEqual(len(message_histories), 5)
                self.assertEqual(len(message_histories[-1][0][-1]), 14)

            for history, output in message_histories:
                for msg in history[-1]:
                    # There should be no draw/undraw messages in the input
                    self.assertFalse(is_draw_msg(msg))
                    self.assertFalse(is_unvote_draw_msg(msg))

                for msg in output:
                    # There should be no draw/undraw messages in the output
                    self.assertFalse(is_draw_msg(msg))
                    self.assertFalse(is_unvote_draw_msg(msg))

    def test_build_v2(self):
        game, game_json, builder = self._setup_test(2)
        for power in POWERS:
            message_histories = builder.build_all_possible_message_histories(
                game.current_short_phase, power, game_json
            )
            if power == POWERS[-1]:
                self.assertEqual(len(message_histories), 5)  # Same # of training examples
                self.assertEqual(
                    len(message_histories[-1][0][-1]), 14
                )  # History should be the same

            for _, output in message_histories:
                for msg in output:
                    # There should be no draw/undraw messages in the output
                    self.assertFalse(is_draw_msg(msg))
                    self.assertFalse(is_unvote_draw_msg(msg))

    def test_build_v2_output_draws(self):
        game, game_json, builder = self._setup_test(2)
        for power in POWERS:
            message_histories = builder.build_all_possible_message_histories(
                game.current_short_phase, power, game_json, output_draw_messages=True,
            )
            if power == POWERS[-1]:
                self.assertEqual(
                    len(message_histories), 6
                )  # Should be an extra training example here, because we include draw messages
                self.assertEqual(
                    len(message_histories[-1][0][-1]), 21
                )  # The history is also longer here, because we include additional messages
                self.assertEqual(
                    len(message_histories[-2][0][-1]), 14
                )  # But for the previous training example, there should be the same amount of message history as before

            # Last output message should be a draw/undraw message for each power
            self.assertTrue(
                is_draw_msg(message_histories[-1][1][0])
                or is_unvote_draw_msg(message_histories[-1][1][0])
            )
