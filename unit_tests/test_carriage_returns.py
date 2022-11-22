#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from parlai_diplomacy.utils.game2seq.format_helpers.misc import remove_trailing_carriage_return


class TestCarriageReturns(unittest.TestCase):
    def test_carriage_returns_editing(self):
        # Test removal of trailing carriage returns that occurs in our webdip data.
        does_not_get_edited = [
            "",
            "Hey England! I am planning to move to channel.",
            "Hey England! I am planning to move to channel\r.",
        ]
        does_get_edited = [
            "\r",
            "\r\r",
            "Hey England! I am planning to move to channel.\r",
            "Hey England! I am planning to move to channel\r.\r",
        ]
        edit_results = [
            "",
            "\r",
            "Hey England! I am planning to move to channel.",
            "Hey England! I am planning to move to channel\r.",
        ]

        for message in does_not_get_edited:
            filtered = remove_trailing_carriage_return(message)
            assert filtered is not None and filtered == message, message
        for i, message in enumerate(does_get_edited):
            filtered = remove_trailing_carriage_return(message)
            expected = edit_results[i]
            assert filtered is not None and filtered == expected, (filtered, expected)
