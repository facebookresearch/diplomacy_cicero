#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import unittest
import random
from fairdiplomacy.typedefs import OutboundMessageDict, RolloutJointAction, JointAction

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryFlattener,
    MessageHistoryUnflattener,
    MessageObjectPart,
)
from parlai_diplomacy.utils.game2seq.format_helpers.orders import (
    OrdersFlattener,
    OrdersUnflattener,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import organize_game_by_phase
from parlai_diplomacy.utils.game2seq.order_prediction import (
    get_rollout_joint_action_from_game_json,
)

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.pydipcc import Game


UNIT_TEST_DIR = os.path.dirname(__file__)

"""
Tests for checking the format helper flatteners and unflatteners.

For each object we check that: obj = unflatten(flatten(obj))
"""


def load_game():
    fle = os.path.join(UNIT_TEST_DIR, "data/game_1_anonymized_truncated.json")
    with open(fle, "r") as f:
        game_json = json.load(f)
    with open(fle, "r") as f:
        game_object = Game.from_json(f.read())
    # roll forward to the next phase
    last_phase_orders = game_json["phases"][-1]["orders"]
    for pwr, orders in last_phase_orders.items():
        game_object.set_orders(pwr, orders)
    game_object.process()

    return game_object


class TestOrderFlattener2UnflattenerV1(unittest.TestCase):
    """
    Testing flattening/unflattening functions in format_helpers/orders.py
    """

    VERSION = 1
    PHASE_IDS = {0, 1, 2, 3, 4}

    def _setup_test(self, version: int = 1):
        self.orders_flattener = OrdersFlattener(version)
        self.orders_unflattener = OrdersUnflattener(version)
        self.game = load_game()
        self.game_json = organize_game_by_phase(json.loads(self.game.to_json()))
        self.phases = self.game.get_all_phases()

    def test_action(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase = self.phases[phase_id]
            for orders in phase.orders.values():
                flat_orders = self.orders_flattener.flatten_action(orders)
                unflattened_orders = self.orders_unflattener.unflatten_action(flat_orders)
                # Check that the action is the same
                self.assertEqual(set(orders), set(unflattened_orders))

    def test_joint_action(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase = self.phases[phase_id]
            joint_action = phase.orders
            for speaker in POWERS:
                flat_orders = self.orders_flattener.flatten_joint_action(joint_action, speaker)
                unflattened_orders = self.orders_unflattener.unflatten_joint_action(flat_orders)
                # The unflattened orders should contain a key for every single power in v1
                # The original joint action may not contain orders for every single power, due to
                # webdip irregularities (e.g. if there is an empty order)
                assert set(joint_action.keys()).issubset(set(unflattened_orders.keys()))
                # Check that the orders are equal for every power
                for key in unflattened_orders.keys():
                    self.assertEqual(
                        set(joint_action.get(key, set())), set(unflattened_orders[key])
                    )

    def test_rollout_action(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase_name = self.phases[phase_id].name
            for power in POWERS:
                rollout_joint_action, _ = get_rollout_joint_action_from_game_json(
                    self.game_json, phase_name, power, {}
                )
                rollout_action = {
                    phase: joint_action[power]
                    for phase, joint_action in rollout_joint_action.items()
                }
                flattened_rollout_action = self.orders_flattener.flatten_rollout_action(
                    rollout_action
                )
                unflattened_rollout_action = self.orders_unflattener.unflatten_rollout_action(
                    flattened_rollout_action, current_phase=phase_name,
                )
                # Check they have the same phases
                self.assertEqual(
                    set(rollout_action.keys()), set(unflattened_rollout_action.keys())
                )
                for phase, action in unflattened_rollout_action.items():
                    # Check that the orders are equal
                    self.assertEqual(
                        set(rollout_action[phase]), set(action),
                    )

    def test_rollout_joint_action(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase_name = self.phases[phase_id].name
            for power in POWERS:
                rollout_joint_action, _ = get_rollout_joint_action_from_game_json(
                    self.game_json, phase_name, power, {}
                )
                assert rollout_joint_action is not None
                flattened_rollout_joint_action = self.orders_flattener.flatten_rollout_joint_action(
                    rollout_joint_action, power
                )
                unflattened_rollout_joint_action = self.orders_unflattener.unflatten_rollout_joint_action(
                    flattened_rollout_joint_action
                )
                # Check they have the same phases
                self.assertEqual(
                    set(rollout_joint_action.keys()), set(unflattened_rollout_joint_action.keys())
                )
                for phase, joint_action in rollout_joint_action.items():
                    # Check that the orders are equal for every power for each phase
                    for power in unflattened_rollout_joint_action[phase].keys():
                        self.assertEqual(
                            set(joint_action.get(power, set())),
                            set(unflattened_rollout_joint_action[phase][power]),
                        )

    def _assert_equal_rollout_joint_action(self, a: RolloutJointAction, b: RolloutJointAction):
        self.assertEqual(set(a.keys()), set(b.keys()))
        for phase in a.keys():
            self._assert_equal_joint_action(a[phase], b[phase])

    def _assert_equal_joint_action(self, a: JointAction, b: JointAction):
        self.assertEqual(set(a.keys()), set(b.keys()))
        for power in a.keys():
            self.assertEqual(set(a[power]), set(b[power]))

    def test_rollout_bilateral_joint_action_phasemajor(self):
        if self.VERSION == 1:
            # Not Implemented for V1
            return
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase_name = self.phases[phase_id].name
            for power in POWERS:
                rollout_joint_action, _ = get_rollout_joint_action_from_game_json(
                    self.game_json, phase_name, power, {}
                )
                recipient = random.choice([x for x in POWERS if x != power])
                assert rollout_joint_action is not None
                flattened_phasemajor = self.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                    rollout_joint_action, power, recipient, speaker_first=False
                )
                unflattened = self.orders_unflattener.unflatten_rollout_joint_action_bilateral_phasemajor(
                    flattened_phasemajor
                )
                self._assert_equal_rollout_joint_action(
                    unflattened,
                    PseudoOrders.from_rollout_joint_action(rollout_joint_action)
                    .as_bilateral(power, recipient)
                    .as_rollout_joint_action(),
                )

    def test_rollout_bilateral_joint_action_powermajor(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase_name = self.phases[phase_id].name
            for power in POWERS:
                rollout_joint_action, _ = get_rollout_joint_action_from_game_json(
                    self.game_json, phase_name, power, {}
                )
                recipient = random.choice([x for x in POWERS if x != power])
                assert rollout_joint_action is not None
                flattened_rollout_joint_action_bilateral = self.orders_flattener.flatten_rollout_joint_action_bilateral_powermajor(
                    rollout_joint_action, power, recipient
                )
                unflattened_rollout_joint_action_bilateral = self.orders_unflattener.unflatten_rollout_joint_action_bilateral_powermajor(
                    flattened_rollout_joint_action_bilateral, phase_name
                )
                self._assert_equal_rollout_joint_action(
                    unflattened_rollout_joint_action_bilateral,
                    PseudoOrders.from_rollout_joint_action(rollout_joint_action)
                    .as_bilateral(power, recipient)
                    .as_rollout_joint_action(),
                )


class TestOrderFlattener2UnflattenerV2(TestOrderFlattener2UnflattenerV1):
    VERSION = 2


class TestMessageHistoryFormatter2UnformatterV1(unittest.TestCase):
    VERSION = 1
    PHASE_IDS = {0, 1, 2, 3, 4}

    def _setup_test(self, version: int = 1):
        self.messagehistory_flattener = MessageHistoryFlattener(version)
        self.messagehistory_unflattener = MessageHistoryUnflattener(version)
        self.game = load_game()
        self.game_json = organize_game_by_phase(json.loads(self.game.to_json()))
        self.phases = self.game.get_all_phases()

    def _check_msgs_equal(self, msg1: OutboundMessageDict, msg2: OutboundMessageDict):
        for key in [
            MessageObjectPart.SENDER,
            MessageObjectPart.RECIPIENT,
            MessageObjectPart.MESSAGE,
            MessageObjectPart.PHASE,
        ]:
            self.assertEqual(msg1[key], msg2[key])

    def test_message(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase = self.phases[phase_id]
            phase_name = phase.name
            messages = phase.messages
            for message in messages.values():
                flat_message = self.messagehistory_flattener.flatten_message(message)
                unflattened_message = self.messagehistory_unflattener.unflatten_single_message(
                    flat_message, phase_name
                )
                self._check_msgs_equal(message, unflattened_message)

    def test_phase_messages(self):
        self._setup_test(self.VERSION)
        for phase_id in self.PHASE_IDS:
            phase = self.phases[phase_id]
            phase_name = phase.name
            messages = list(phase.messages.values())
            flat_messages = self.messagehistory_flattener.flatten_phase_messages(messages)
            unflattened_messages = self.messagehistory_unflattener.unflatten_messages(
                flat_messages, phase_name
            )
            self.assertEqual(len(messages), len(unflattened_messages))
            for message, unflattened_message in zip(messages, unflattened_messages):
                self._check_msgs_equal(message, unflattened_message)


class TestMessageHistoryFormatter2UnformatterV2(unittest.TestCase):
    VERSION = 2
