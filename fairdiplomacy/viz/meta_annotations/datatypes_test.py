#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

from parameterized import parameterized

from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.viz.meta_annotations.datatypes import NonsenseDataType, PseudoOrdersDataType


class PseudoOrdersDataTypeTest(unittest.TestCase):
    @parameterized.expand([[1]])
    def test_cycle(self, version):
        data = PseudoOrders(
            {
                "S1901M": {
                    "ITALY": ("F NAP - ION", "A ROM - APU", "A VEN H"),
                    "AUSTRIA": ("A VIE - GAL", "F TRI - ALB", "A BUD - SER"),
                }
            }
        )
        encoded = PseudoOrdersDataType.dump(data, version)
        decoded = PseudoOrdersDataType.load(encoded, version)
        self.assertEqual(data.val, decoded.val)


class NonsenseDataTypeTest(unittest.TestCase):
    @parameterized.expand([[1]])
    def test_NonsenseDataTypeTest_cycle(self, version):
        data = ("Rejected by Pseudo Order Classifier", str({"extra": "data"}))
        encoded = NonsenseDataType.dump(data, version)
        decoded = NonsenseDataType.load(encoded, version)
        self.assertEqual(data, decoded)
