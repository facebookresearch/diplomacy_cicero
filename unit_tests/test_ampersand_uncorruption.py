#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from parlai_diplomacy.utils.game2seq.format_helpers.misc import uncorrupt_ampersands

"""
Test uncorruption of ampersands for messages
"""


class TestAmpersandUncorruption(unittest.TestCase):
    def test_ampersand_uncorruption(self):
        self.assertEqual(
            uncorrupt_ampersands("I don&t know about that&"), "I don't know about that&",
        )

        self.assertEqual(
            uncorrupt_ampersands("&shrug&"), "*shrug*",
        )

        self.assertEqual(
            uncorrupt_ampersands("&could"), "&could",
        )

        self.assertEqual(
            uncorrupt_ampersands("!@## !@% ^*(($%@$ @#$ #@"), "!@## !@% ^*(($%@$ @#$ #@",
        )

        self.assertEqual(
            uncorrupt_ampersands("That&s true but it doesn&t matter"),
            "That's true but it doesn't matter",
        )

        self.assertEqual(
            uncorrupt_ampersands("&cough& &cough &cough&"), "*cough* &cough *cough*",
        )

        self.assertEqual(
            uncorrupt_ampersands("Sorry, &you&"), "Sorry, *you*",
        )

        self.assertEqual(
            uncorrupt_ampersands("i wasn&t suggesting that"), "i wasn't suggesting that",
        )
