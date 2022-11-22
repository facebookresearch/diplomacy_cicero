#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
import unittest
import numpy as np
import torch

from fairdiplomacy.agents.base_search_agent import sample_orders_from_policy


class TestSampleOrdersFromPolicy(unittest.TestCase):
    def test(self):
        power_actions = {
            "RUSSIA": [],
            "GERMANY": ["a", "b", "c"],
            "ENGLAND": ["d", "e", "f"],
            "TURKEY": ["g"],
            "AUSTRIA": ["h", "i", "j", "k"],
            "FRANCE": ["l"],
            "ITALY": [],
        }
        power_action_probs = {
            "RUSSIA": [],
            "GERMANY": [
                1.0,
                2.0,
                3.0,
            ],  # sample_orders_from_policy actually is robust to unnormed probs
            "ENGLAND": [
                torch.tensor(4.0),
                torch.tensor(0.5),
                torch.tensor(1.5),
            ],  # and is robust to float vs tensor of float
            "TURKEY": [0.999999],
            "AUSTRIA": [0.1, 0.0, 0.9, 0.0],
            "FRANCE": [0.0],
            "ITALY": [],
        }

        np.random.seed(654321)
        action_counts = defaultdict(int)
        for _ in range(1000):
            (sampled_idxs, power_sampled_orders) = sample_orders_from_policy(
                power_actions, power_action_probs
            )
            for power in power_sampled_orders:
                action = power_sampled_orders[power]
                if action:
                    idx = sampled_idxs[power]
                    assert power_actions[power][idx] == action
                    action_counts[action] += 1

        action_counts = dict(action_counts)
        expected = {
            "b": 325,
            "e": 90,
            "g": 1000,
            "j": 902,
            "l": 1000,
            "c": 515,
            "d": 678,
            "h": 98,
            "f": 232,
            "a": 160,
        }
        self.assertEqual(action_counts, expected)
