#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from fairdiplomacy.timestamp import Timestamp


class TestTimestamp(unittest.TestCase):
    def test_creation_methods(self):
        self.assertEqual(Timestamp.from_centis(42).to_centis(), 42)
        self.assertEqual(Timestamp.from_seconds(42).to_seconds_int(), 42)
        self.assertEqual(Timestamp.from_seconds(42).to_seconds_int(), 42)
        self.assertEqual(Timestamp.from_seconds(42).to_centis(), 4200)
