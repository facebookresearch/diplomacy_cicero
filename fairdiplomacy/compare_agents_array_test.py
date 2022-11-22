#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import tempfile
import pathlib
import unittest

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy import pydipcc

from fairdiplomacy.compare_agents_array import (
    get_power_scores_from_folder,
    get_eval_game_file_name,
)


class TestGameDiscovery(unittest.TestCase):
    def test_test_discovery(self):
        game = pydipcc.Game()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            for p in POWERS:
                with (temp_dir / get_eval_game_file_name(p, 0)).open("w") as stream:
                    stream.write(game.to_json())
            power_scores = get_power_scores_from_folder(temp_dir)
            self.assertEqual(len(power_scores), len(POWERS))
            powers, _ = zip(*power_scores)
            self.assertEqual(set(powers), set(POWERS))
