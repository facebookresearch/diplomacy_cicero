#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.models.base_strategy_model.base_strategy_model import check_permute_powers
import os
import unittest

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestCheckPermutePowers(unittest.TestCase):
    def test_check_permute_powers(self):
        check_permute_powers()
