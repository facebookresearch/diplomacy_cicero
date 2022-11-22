#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
import unittest
import json

import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN
from fairdiplomacy_external.webdip_api import (
    turn_to_phase,
    phase_to_turn,
    phase_to_phasetype,
    webdip_state_to_game,
    inplace_handle_messages_with_none_phases,
    UnexpectedWebdipBehaviorException,
)

TEST_DATA = heyhi.PROJ_ROOT / "unit_tests/data"


class TestTurnToPhase(unittest.TestCase):
    def test_turn_to_from_phase(self):
        game = pydipcc.Game.from_json(open(TEST_DATA / "game_100012_anonymized.json").read())
        for phase in game.get_all_phase_names():
            print(phase, phase_to_turn(phase), phase_to_phasetype(phase))
            self.assertEqual(turn_to_phase(phase_to_turn(phase), phase_to_phasetype(phase)), phase)

    def test_inplace_handle_messages_with_none_phases_prefix(self):
        messages = [
            {"phase": None},
            {"phase": None},
            {"phase": "S1901M"},
            {"phase": "F1901M"},
            {"phase": "W1901A"},
        ]
        inplace_handle_messages_with_none_phases(messages, turn=1, status_json_phase="Builds")
        self.assertEqual(
            [m["phase"] for m in messages], ["S1901M", "S1901M", "S1901M", "F1901M", "W1901A"]
        )

    def test_inplace_handle_messages_with_none_phases_throws(self):
        messages = [
            {"phase": None},
            {"phase": "S1901M"},
            {"phase": None},
            {"phase": "F1901M"},
            {"phase": "W1901A"},
        ]
        self.assertRaises(
            UnexpectedWebdipBehaviorException,
            inplace_handle_messages_with_none_phases,
            messages,
            turn=1,
            status_json_phase="Builds",
        )


class TestStatusJsonToGame(unittest.TestCase):
    @staticmethod
    def status_json_with_vote_str(vote_str: str = "Voted for Draw"):
        # status json after england has sent a <vote_str> vote
        return json.loads(
            """{"gameID": 416, "countryID": 1, "variantID": 1, "potType": "Unranked", "phaseLengthInMinutes": 5, "turn": 0, "phase": "Diplomacy", "gameOver": "No", "pressType": "Regular", "phases": [{"units": [{"unitType": "Fleet", "retreating": "No", "terrID": 2, "countryID": 1}, {"unitType": "Army", "retreating": "No", "terrID": 3, "countryID": 1}, {"unitType": "Fleet", "retreating": "No", "terrID": 6, "countryID": 1}, {"unitType": "Fleet", "retreating": "No", "terrID": 11, "countryID": 3}, {"unitType": "Army", "retreating": "No", "terrID": 12, "countryID": 3}, {"unitType": "Army", "retreating": "No", "terrID": 15, "countryID": 3}, {"unitType": "Army", "retreating": "No", "terrID": 22, "countryID": 6}, {"unitType": "Army", "retreating": "No", "terrID": 23, "countryID": 6}, {"unitType": "Fleet", "retreating": "No", "terrID": 24, "countryID": 6}, {"unitType": "Fleet", "retreating": "No", "terrID": 27, "countryID": 7}, {"unitType": "Army", "retreating": "No", "terrID": 29, "countryID": 7}, {"unitType": "Army", "retreating": "No", "terrID": 31, "countryID": 7}, {"unitType": "Fleet", "retreating": "No", "terrID": 79, "countryID": 7}, {"unitType": "Fleet", "retreating": "No", "terrID": 37, "countryID": 4}, {"unitType": "Army", "retreating": "No", "terrID": 38, "countryID": 4}, {"unitType": "Army", "retreating": "No", "terrID": 41, "countryID": 4}, {"unitType": "Fleet", "retreating": "No", "terrID": 46, "countryID": 2}, {"unitType": "Army", "retreating": "No", "terrID": 47, "countryID": 2}, {"unitType": "Army", "retreating": "No", "terrID": 49, "countryID": 2}, {"unitType": "Army", "retreating": "No", "terrID": 72, "countryID": 5}, {"unitType": "Fleet", "retreating": "No", "terrID": 73, "countryID": 5}, {"unitType": "Army", "retreating": "No", "terrID": 74, "countryID": 5}], "messages": [{"message": "__VOTE_STR__", "fromCountryID": 1, "toCountryID": 1, "timeSent": 1649971335, "phaseMarker": "Diplomacy"}], "publicVotesHistory": [{"vote": "__VOTE_STR__", "countryID": 1, "timeSent": 1649971335, "phaseMarker": "Diplomacy"}], "centers": [{"terrID": 2, "countryID": 1}, {"terrID": 3, "countryID": 1}, {"terrID": 6, "countryID": 1}, {"terrID": 7, "countryID": 0}, {"terrID": 8, "countryID": 0}, {"terrID": 10, "countryID": 0}, {"terrID": 11, "countryID": 3}, {"terrID": 12, "countryID": 3}, {"terrID": 15, "countryID": 3}, {"terrID": 17, "countryID": 0}, {"terrID": 19, "countryID": 0}, {"terrID": 20, "countryID": 0}, {"terrID": 21, "countryID": 0}, {"terrID": 22, "countryID": 6}, {"terrID": 23, "countryID": 6}, {"terrID": 24, "countryID": 6}, {"terrID": 27, "countryID": 7}, {"terrID": 29, "countryID": 7}, {"terrID": 31, "countryID": 7}, {"terrID": 32, "countryID": 7}, {"terrID": 34, "countryID": 0}, {"terrID": 35, "countryID": 0}, {"terrID": 36, "countryID": 0}, {"terrID": 37, "countryID": 4}, {"terrID": 38, "countryID": 4}, {"terrID": 41, "countryID": 4}, {"terrID": 43, "countryID": 0}, {"terrID": 44, "countryID": 0}, {"terrID": 46, "countryID": 2}, {"terrID": 47, "countryID": 2}, {"terrID": 49, "countryID": 2}, {"terrID": 72, "countryID": 5}, {"terrID": 73, "countryID": 5}, {"terrID": 74, "countryID": 5}], "turn": 0, "phase": "Diplomacy", "orders": []}], "standoffs": [], "occupiedFrom": [], "votes": "Draw", "orderStatus": "", "status": "Playing", "drawType": "draw-votes-public", "processTime": "1650230174", "orderStatuses": {"1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": ""}, "publicVotes": {"1": "Draw", "2": null, "3": null, "4": null, "5": null, "6": null, "7": null}}""".replace(
                "__VOTE_STR__", vote_str
            )
        )

    def test_draw_votes_converted(self):
        game = webdip_state_to_game(self.status_json_with_vote_str("Voted for Draw"))
        self.assertEqual(len(game.messages), 1)  # ignore draw self-message
        self.assertEqual(
            list(game.messages.values())[0]["message"], DRAW_VOTE_TOKEN
        )  # draw vote converted to Game format

    def test_draw_unvotes_converted(self):
        for vote_str in ["Un-Voted for Draw", "Un-voted for Draw"]:
            game = webdip_state_to_game(self.status_json_with_vote_str(vote_str))
            self.assertEqual(len(game.messages), 1)  # ignore self-message
            self.assertEqual(
                list(game.messages.values())[0]["message"], UNDRAW_VOTE_TOKEN
            )  # draw vote converted to Game format

    def test_other_votes_ignored(self):
        for vote_str in [
            "Voted for Pause",
            "Voted for Cancel",
            "Un-Voted for Pause",
            "Un-voted for Cancel",
        ]:
            game = webdip_state_to_game(self.status_json_with_vote_str(vote_str))
            self.assertEqual(len(game.messages), 0)

    def test_status_json_with_post_game_messages(self):
        # Load a webdip status json (with real messages redacted) which
        # contains post-game messages
        status_json = json.load(open(f"{TEST_DATA}/status_json_with_post_game_messages.json"))
        game = webdip_state_to_game(status_json)
        self.assertEqual(game.current_short_phase, "COMPLETED")
