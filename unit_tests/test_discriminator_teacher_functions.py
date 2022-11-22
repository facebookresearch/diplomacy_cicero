#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import unittest

from parlai_diplomacy.tasks.discriminator.agents import (
    change_conversation_participants,
    change_entities,
    pick_message_from_context,
    message_from_incorrect_phase,
    BaseCorruptedMessagesValidTeacher,
)
from parlai_diplomacy.tasks.discriminator import change_entity_utils
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart


UNIT_TEST_DIR = os.path.dirname(__file__)

SENDER = MessageObjectPart.SENDER
RECEIVER = MessageObjectPart.RECIPIENT
MESSAGE_TEXT = MessageObjectPart.MESSAGE


EXAMPLE_MESSAGE_CONTEXT = (
    "S1901M\n"
    "France -> England: Hello, what would you think about working together? [EO_M] \n"
    "England -> Germany: How about an alliance against France? [EO_M] \n"
    "F1901M\n"
    "France -> England: That is cool [EO_M] \n"
    "S1902M\n"
    "England -> France: Would it not be better to take advantage of the Mediterranean? [EO_M] \n"
    "Germany -> England: yeah, i missed a turn though so ill just take these Sc. [EO_M] \n"
    "F1902M\n"
    "England -> Turkey: Heh, yeah, Germany I never feared as much as France. [EO_M] \n"
    "S1903M\n"
    "Turkey -> England: You're welcome. [EO_M] \n"
    "England -> Turkey: Done. [EO_M] F1903M England 3: "
)


EXAMPLE_MESSAGE_CONTEXT_NO_PREVIOUS_PHASE_DATA = (
    "S1901M\n" "England -> Turkey: Done. [EO_M] S1901M England 3: "
)

OTHER_PHASES_MESSAGES = (
    "Hello, what would you think about working together? [EO_M] \n"
    "How about an alliance against France? [EO_M] \n"
    "That is cool [EO_M] \n"
    "Would it not be better to take advantage of the Mediterranean? [EO_M] \n"
    "yeah, i missed a turn though so ill just take these Sc. [EO_M] \n"
    "Heh, yeah, Germany I never feared as much as France. [EO_M] \n"
    "You're welcome. [EO_M] \n"
)


class MockTokenizer:
    def tokenize(self, s):
        return s.replace("?", " ? ").replace("->", " -> ").split(" ")


class TestSenderReceiverChange(unittest.TestCase):
    def test_sender_receiver_change(self):
        parsed_msg = {SENDER: "France", RECEIVER: "England"}

        change_conversation_participants(parsed_msg, sender=True)
        self.assertIsNotNone(parsed_msg)
        self.assertNotEqual(parsed_msg[SENDER], "France")
        self.assertEqual(parsed_msg[RECEIVER], "England")

        parsed_msg = {SENDER: "Austria", RECEIVER: "Turkey"}
        change_conversation_participants(parsed_msg)
        self.assertIsNotNone(parsed_msg)
        self.assertEqual(parsed_msg[SENDER], "Austria")
        self.assertNotEqual(parsed_msg[RECEIVER], "Turkey")


class TestEntityChange(unittest.TestCase):
    def setUp(self):
        self._tokenizer = MockTokenizer()
        self._ratio = 1.0
        self._filter = {"locations", "symbols", "powers", "power_adjs", "noisy_locations"}
        change_entity_utils.POWERS_LOWER = {"england", "france"}
        change_entity_utils.SYMBOLS_LOWER = {"fin", "swe"}
        change_entity_utils.LOCS_LOWER = {"black sea", "london"}
        change_entity_utils.NOISY_LOCS_LOWER = {
            "the black sea",
            "the blk sea",
            "blk sea",
            "baltic",
        }  # not testing noisy locations atm
        self._trie = change_entity_utils.build_entity_trie()

    def test_no_entity_to_change(self):
        msg_txt = "there is no entity to swap"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, self._filter)
        self.assertEqual(msg_txt, parsed_msg[MESSAGE_TEXT])

    def test_entities_to_swap(self):
        msg_txt = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, self._filter)
        self.assertEqual(parsed_msg[MESSAGE_TEXT], "I am France moving from SWE to Fin")

    def test_filter_entities_to_swap(self):
        msg_txt = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, {"locations"})
        self.assertEqual(parsed_msg[MESSAGE_TEXT], "I am England moving from FIN to Swe")

        msg_txt = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, {"powers"})
        self.assertEqual(parsed_msg[MESSAGE_TEXT], "I am France moving from FIN to Swe")

        msg_txt = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, {"symbols"})
        self.assertEqual(parsed_msg[MESSAGE_TEXT], "I am England moving from SWE to Fin")

    def test_consistant_entities_to_swap(self):
        msg_txt = "Should we help France or should we attack france?"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, self._filter)
        self.assertEqual(
            parsed_msg[MESSAGE_TEXT], "Should we help England or should we attack england?"
        )

        msg_txt = "Should we help France or should we attack france?"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, 0.001, self._filter)
        self.assertEqual(
            parsed_msg[MESSAGE_TEXT], "Should we help England or should we attack england?"
        )

    def test_movement_entities_to_swap(self):
        msg_txt = "fin->SWE will finish me."
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, self._filter)
        self.assertEqual(parsed_msg[MESSAGE_TEXT], "swe->FIN will finish me.")

    def test_multi_token_entity_swap(self):
        msg_txt = "DMZ the blk sea? what do you think?"
        parsed_msg = {MESSAGE_TEXT: msg_txt}
        change_entities(parsed_msg, self._trie, self._ratio, self._filter)
        options = [
            "DMZ baltic? what do you think?",
            "DMZ the black sea? what do you think?",
            "DMZ blk sea? what do you think?",
        ]
        self.assertIn(parsed_msg[MESSAGE_TEXT], options)

    def test_find_matches(self):
        msg_txt_1 = "Should we help France or should we attack france?"
        expects_1 = {("france", "powers"): [(15, 23, " France "), (42, 50, " france?")]}

        msg_txt_2 = "fin->swe will finish me."
        expects_2 = {("fin", "symbols"): [(0, 5, " fin-")], ("swe", "symbols"): [(5, 10, ">swe ")]}

        msg_txt_3 = "DMZ Black sea?"
        expects_3 = {("black sea", "locations"): [(4, 15, " Black sea?")]}

        msg_txt_4 = "swe-fin will finish me."
        expects_4 = {("swe", "symbols"): [(0, 5, " swe-")], ("fin", "symbols"): [(4, 9, "-fin ")]}

        msg_txt_5 = "DMZ Black seas?"
        expects_5 = {}

        matches_1 = change_entity_utils.search_entities(" " + msg_txt_1 + " ", self._trie)
        self.assertEqual(matches_1, expects_1)

        matches_2 = change_entity_utils.search_entities(" " + msg_txt_2 + " ", self._trie)
        self.assertEqual(matches_2, expects_2)

        matches_3 = change_entity_utils.search_entities(" " + msg_txt_3 + " ", self._trie)
        self.assertEqual(matches_3, expects_3)

        matches_4 = change_entity_utils.search_entities(" " + msg_txt_4 + " ", self._trie)
        self.assertEqual(matches_4, expects_4)

        matches_5 = change_entity_utils.search_entities(" " + msg_txt_5 + " ", self._trie)
        self.assertEqual(matches_5, expects_5)

    def test_get_consistant_replacements(self):
        replacements = change_entity_utils.get_consistant_replacements(
            "france", "powers", [(15, 23, " France "), (42, 50, " france?")]
        )
        expects_replacements = [
            (15, 23, " France ", " England "),
            (42, 50, " france?", " england?"),
        ]
        self.assertEqual(replacements, expects_replacements)

        replacements = change_entity_utils.get_consistant_replacements(
            "swe", "symbols", [(0, 5, " swe-")]
        )
        expects_replacements = [(0, 5, " swe-", " fin-")]
        self.assertEqual(replacements, expects_replacements)

    def test_splice_replacements(self):
        msg_txt = "Should we help France or should we attack france?"
        msg_txt_replaced = change_entity_utils.splice_replace(
            " " + msg_txt + " ", 15, 23, " England "
        )
        self.assertEqual(
            msg_txt_replaced.strip(), "Should we help England or should we attack france?"
        )

        msg_txt = "Should we help France or should we attack france?"
        msg_txt_replaced = change_entity_utils.splice_replace(
            " " + msg_txt + " ", 0, 23, " England "
        )
        self.assertEqual(msg_txt_replaced.strip(), "England or should we attack france?")


class TestPickMessageFromContext(unittest.TestCase):
    def test_can_pick(self):
        selected = pick_message_from_context(EXAMPLE_MESSAGE_CONTEXT)
        self.assertIsNotNone(selected)
        self.assertTrue(selected in OTHER_PHASES_MESSAGES)

    def test_can_not_pick(self):
        selected = pick_message_from_context(EXAMPLE_MESSAGE_CONTEXT_NO_PREVIOUS_PHASE_DATA)
        self.assertIsNone(selected)


class TestMessageFromOtherPhases(unittest.TestCase):
    def test_previous_phase(self):
        MSG_TXT = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: MSG_TXT}
        r = message_from_incorrect_phase(parsed_msg, EXAMPLE_MESSAGE_CONTEXT)
        self.assertTrue(r)
        self.assertNotEqual(parsed_msg[MESSAGE_TEXT], MSG_TXT)
        self.assertTrue(parsed_msg[MESSAGE_TEXT] in OTHER_PHASES_MESSAGES)

    def test_no_previous_phase(self):
        MSG_TXT = "I am England moving from FIN to Swe"
        parsed_msg = {MESSAGE_TEXT: MSG_TXT}
        r = message_from_incorrect_phase(
            parsed_msg, EXAMPLE_MESSAGE_CONTEXT_NO_PREVIOUS_PHASE_DATA
        )
        self.assertFalse(r)
        self.assertEqual(parsed_msg[MESSAGE_TEXT], MSG_TXT)
        self.assertFalse(parsed_msg[MESSAGE_TEXT] in OTHER_PHASES_MESSAGES)


class TestValidTeachers(unittest.TestCase):
    def test_datafile(self):
        opt = {
            "task": "validcorrupted_message_history_dialoguediscriminator_evaldump",
            "valid_corrupted_dtype": "non-existing",
            "n_corrupted_valid_examples": 100,
            "context_message_format": "message_history",
        }
        with self.assertRaises(AssertionError):
            BaseCorruptedMessagesValidTeacher(opt)

    def test_get_examples(self):
        opt = {
            "task": "validcorrupted_message_history_dialoguediscriminator_evaldump",
            "valid_corrupted_dtype": f"unittests:{UNIT_TEST_DIR}",
            "datatype": "train",
            "context_message_format": "message_history_shortstate",
            "n_corrupted_valid_examples": 100,
        }
        teacher = BaseCorruptedMessagesValidTeacher(opt)
        self.assertIsNotNone(teacher)
        retrieved_examples = set()
        for _ in range(20 * teacher.num_examples()):
            ex = teacher.act()
            self.assertIsNotNone(ex)
            if ex.is_padding():
                continue
            self.assertTrue("text" in ex)
            self.assertTrue("labels" in ex)
            retrieved_examples.add(str(ex))

        self.assertEqual(len(retrieved_examples), 2)
