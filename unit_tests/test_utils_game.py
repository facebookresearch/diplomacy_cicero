#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import unittest

from fairdiplomacy import pydipcc
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.typedefs import Timestamp

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestUtilsGame(unittest.TestCase):
    def test_game_from_view_of(self):
        game = pydipcc.Game()
        game.add_message("ITALY", "AUSTRIA", "hi there", Timestamp.from_centis(12345))
        self.assertEqual(len(game.messages), 1)
        self.assertEqual(len(game_from_view_of(game, "ITALY").messages), 1)
        self.assertEqual(len(game_from_view_of(game, "AUSTRIA").messages), 1)
        self.assertEqual(len(game_from_view_of(game, "ENGLAND").messages), 0)
