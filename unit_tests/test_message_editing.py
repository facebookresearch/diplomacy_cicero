#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
import math
from fairdiplomacy import pydipcc
from fairdiplomacy.game import sort_phase_key

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.format_helpers.message_editing import (
    MessageEditing,
    MessageFiltering,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import corrupt_newlines
from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN
from fairdiplomacy.typedefs import MessageDict, Power, Timestamp, List
from fairdiplomacy.utils.typedefs import build_message_dict


TEST_MSG: MessageDict = build_message_dict(
    "ENGLAND", "FRANCE", " ~N~ ~N~ diplomacy is fun ~N~ ", "S1901M", Timestamp.from_seconds(1),
)
TEST_MSG_HISTORY_REPEAT: List[MessageDict] = [
    build_message_dict(
        "ENGLAND", "TURKEY", " ~N~ ~N~ diplomacy is fun ~N~ ", "S1901M", Timestamp.from_seconds(1),
    ),
]
TEST_MSG_HISTORY_NO_REPEAT: List[MessageDict] = [
    build_message_dict(
        "ENGLAND",
        "TURKEY",
        " ~N~ ~N~ this is not a repeat! ~N~ ",
        "S1901M",
        Timestamp.from_seconds(1),
    ),
    build_message_dict(
        "ENGLAND",
        "FRANCE",
        " ~N~ ~N~ this is not a repeat either! ~N~ ",
        "S1901M",
        Timestamp.from_seconds(3),
    ),
]

TEST_OFFENSIVE_MSG: MessageDict = build_message_dict(
    "ENGLAND", "FRANCE", "fuck you", "S1901M", Timestamp.from_seconds(5)
)
DRAW_MSG: MessageDict = build_message_dict(
    "ENGLAND", "FRANCE", DRAW_VOTE_TOKEN, "S1901M", Timestamp.from_seconds(5)
)
DISCUSS_DRAW_MSG: MessageDict = build_message_dict(
    "ENGLAND", "FRANCE", "Should we vote for a Draw?", "S1901M", Timestamp.from_seconds(5)
)
SHORT_MSG: MessageDict = build_message_dict(
    "ENGLAND", "FRANCE", "hi", "S1901M", Timestamp.from_seconds(5)
)


class TestMessageEditing(unittest.TestCase):
    TIME_SENT = Timestamp.from_seconds(5)

    def _create_new_msg(
        self, sender: Power = "ENGLAND", recipient: Power = "TURKEY", message="Hello."
    ) -> MessageDict:
        x = build_message_dict(sender, recipient, message, "S1901M", self.TIME_SENT)
        self.TIME_SENT += Timestamp.from_seconds(1)
        return x

    def _create_redacted_msg(self, redacted_pct: float):
        assert redacted_pct <= 1 and redacted_pct >= 0
        num_redact = math.ceil(redacted_pct * 10)
        num_non_redact = 10 - num_redact
        redacted = "[8675309]"
        nonredacted = "jenny"

        msg_text = " ".join(
            [redacted for _ in range(num_redact)] + [nonredacted for _ in range(num_non_redact)]
        )
        return self._create_new_msg(message=msg_text)

    def test_newline_editing(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        msg_editor = MessageEditing(
            edit_newlines=True, edit_names=False, edit_weird_capitalization=False,
        )
        should_filter = msg_filterer.should_filter_message(TEST_MSG, TEST_MSG_HISTORY_REPEAT)
        assert not should_filter
        filtered = msg_editor.maybe_edit_message(TEST_MSG)
        assert filtered[MessageObjectPart.MESSAGE] == "\n\ndiplomacy is fun\n"

        corrupted = corrupt_newlines(filtered[MessageObjectPart.MESSAGE])
        assert corrupted == TEST_MSG[MessageObjectPart.MESSAGE]

    def test_repeat_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        should_filter = msg_filterer.should_filter_message(TEST_MSG, TEST_MSG_HISTORY_REPEAT)
        assert should_filter

        should_filter = msg_filterer.should_filter_message(TEST_MSG, TEST_MSG_HISTORY_NO_REPEAT)
        assert not should_filter

    def test_offensive_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=True,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=True,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )

        should_filter = msg_filterer.should_filter_message(
            TEST_OFFENSIVE_MSG, TEST_MSG_HISTORY_NO_REPEAT
        )
        assert should_filter

        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )

        should_filter = msg_filterer.should_filter_message(
            TEST_OFFENSIVE_MSG, TEST_MSG_HISTORY_NO_REPEAT
        )
        assert not should_filter

    def test_consecutive_short_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        LONG = TEST_MSG_HISTORY_NO_REPEAT[1]
        SHORT = TEST_OFFENSIVE_MSG
        # 2 short messages
        should_filter = msg_filterer.should_filter_message(SHORT, [SHORT])
        assert should_filter

        # Long then short is ok
        should_filter = msg_filterer.should_filter_message(SHORT, [LONG])
        assert not should_filter

        # Short then lng is ok
        should_filter = msg_filterer.should_filter_message(LONG, [SHORT])
        assert not should_filter

    def test_redacted_message_editing(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=True,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )

        # should filter if >20% or more is redacted
        should_filter = msg_filterer.should_filter_message(
            self._create_redacted_msg(0.3), TEST_MSG_HISTORY_REPEAT
        )
        assert should_filter

        should_filter = msg_filterer.should_filter_message(
            self._create_redacted_msg(0.8), TEST_MSG_HISTORY_REPEAT
        )
        assert should_filter

        # should not filter if <=20% is redacted
        should_filter = msg_filterer.should_filter_message(
            self._create_redacted_msg(0.1), TEST_MSG_HISTORY_REPEAT
        )
        assert not should_filter
        assert (
            self._create_redacted_msg(0.1)[MessageObjectPart.MESSAGE]
            == "[8675309] jenny jenny jenny jenny jenny jenny jenny jenny jenny"
        )

    def test_ampersands_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=True,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )

        # should filter if >20% or more is redacted
        should_filter = msg_filterer.should_filter_message(
            self._create_new_msg(message="&"), TEST_MSG_HISTORY_REPEAT
        )
        assert should_filter

        should_filter = msg_filterer.should_filter_message(
            self._create_new_msg(message="Hello & what's up"), TEST_MSG_HISTORY_REPEAT
        )
        assert should_filter

        # Should not filter if message does not contain ampersands
        should_filter = msg_filterer.should_filter_message(
            self._create_new_msg(message="Hello, what's up"), TEST_MSG_HISTORY_REPEAT
        )
        assert not should_filter

    def test_draw_messages(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=True,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=True,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        # should NOT filter draw message
        should_filter = msg_filterer.should_filter_message(DRAW_MSG, [])
        assert not should_filter

        # SHOULD filter similarly short message, if it's not a draw message
        should_filter = msg_filterer.should_filter_message(SHORT_MSG, [])
        assert should_filter

    def test_filter_messages_about_draw_messages(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=True,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=True,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=True,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        # should not filter message about draws
        should_filter = msg_filterer.should_filter_message(DISCUSS_DRAW_MSG, [])
        assert not should_filter

        # should filter message about draws
        should_filter = msg_filterer.should_filter_message(
            DISCUSS_DRAW_MSG, [], game_is_missing_draw_votes=True
        )
        assert should_filter

    def test_names_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=True,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        # Name filtering is nontrivial, as you can see by some of these examples.
        # We don't try to be perfect.
        does_not_get_filtered = [
            "",
            "Hey England! I am planning to move to channel.",
            "russia really let you off scott-free, huh?",
            "How about we round robin these centers?",
            "Was looking for a hail mary in these last turns.",
            "Joy let me off the hook",
            "Victor stabbed me",
            "If you can grab another SC, YOU WILL BE THE VICTOR!",
            "My dear France, your proposal delights me\n-in her own hand, The Queen of England.",
            "don't get so upset at me, jesus, I didn't even attack you",
        ]
        does_get_filtered = [
            "Jane",
            "Hey John! I am planning to move to channel.",
            "Hey john! I am planning to move to channel.",
            "Hey JOHN! I am planning to move to channel.",
            "bob said hi",
            "Fancy yourself a George Washington, huh?",
            "my dear watson, how could you guess?",
            "My dear France, your proposal delights me\n-in her own hand, Queen Elizabeth.",
            "don't get so upset at me, mike, I didn't even attack you",
            "bob!bomb!adam!bob-bob!hilly,billy,chilly,sally~sandy;mandy;mark;marc.freed.fred",
        ]

        for message in does_not_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert not should_filter, message
        for message in does_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert should_filter, message

    def test_names_editing(self):
        msg_editor = MessageEditing(
            edit_newlines=False, edit_names=True, edit_weird_capitalization=False,
        )
        # Name filtering is nontrivial, as you can see by some of these examples.
        # We don't try to be perfect.
        does_not_get_edited = [
            "",
            "Hey England! I am planning to move to channel.",
            "russia really let you off scott-free, huh?",
            "How about we round robin these centers?",
            "Was looking for a hail mary in these last turns.",
            "Joy let me off the hook",
            "Victor stabbed me",
            "If you can grab another SC, YOU WILL BE THE VICTOR!",
            "My dear France, your proposal delights me\n-in her own hand, The Queen of England.",
            "don't get so upset at me, jesus, I didn't even attack you",
        ]
        does_get_edited = [
            "Jane",
            "Hey John! I am planning to move to channel.",
            "Hey john! I am planning to move to channel.",
            "Hey JOHN! I am planning to move to channel.",
            "bob said hi",
            "Fancy yourself a George Washington, huh?",
            "my dear watson, how could you guess?",
            "My dear France, your proposal delights me\n-in her own hand, Queen Elizabeth.",
            "don't get so upset at me, mike, I didn't even attack you",
            "bob!bomb!adam!bob-bob!hilly,billy,chilly,sally~sandy;mandy;mark;marc.freed.fred",
        ]
        edit_results = [
            "[1]",
            "Hey [1]! I am planning to move to channel.",
            "Hey [1]! I am planning to move to channel.",
            "Hey [1]! I am planning to move to channel.",
            "[1] said hi",
            "Fancy yourself a [1] [1], huh?",
            "my dear [1], how could you guess?",
            "My dear France, your proposal delights me\n-in her own hand, Queen [1].",
            "don't get so upset at me, [1], I didn't even attack you",
            "[1]!bomb![1]!bob-bob!hilly,[1],chilly,[1]~sandy;[1];mark;[1].freed.[1]",
        ]

        for message in does_not_get_edited:
            edited = msg_editor.maybe_edit_message(self._create_new_msg(message=message))
            assert edited is not None and edited["message"] == message, message
        for i, message in enumerate(does_get_edited):
            edited = msg_editor.maybe_edit_message(self._create_new_msg(message=message))
            expected = edit_results[i]
            assert edited is not None and edited["message"] == expected, (edited, expected)

    def test_urls_emails_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=True,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )

        does_not_get_filtered = [
            "",
            "a.b",
            "c d aaaa.b e fg",
            "Hello world!",
            "I think you should move ENG@NTH. Ok?",
            "http",
            "http://",
            "user @foo.domain"
            "user@foo. domain"
            "Try checking out diplomacy_awesome_strategies.com for some tips",
        ]
        does_get_filtered = [
            "user@foo.domain",
            "user-+@FOOBAR123.org",
            "I think you should move ENG@NTH.Ok?",
            "Check this out-http://www.google.com",
            "Here-[12931]://www.google.com",
            "://youtube/a=b",
            "://youtube.com/a=b",
            "Try checking out diplomacy_awesome_strategies.com/ for some tips",
            "Try checking out diplomacy_awesome_strategies.com/?page=12&123 for some tips",
        ]

        for message in does_not_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert not should_filter, message
        for message in does_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert should_filter, message

    def test_weird_capitalization_editing(self):
        msg_editor = MessageEditing(
            edit_newlines=False, edit_names=False, edit_weird_capitalization=True,
        )
        does_not_get_edited = [
            "",
            "!==?-",
            "I am england. You are england? They are england too!",
            "I am England. You are England? They are England too!",
            "I am eNGLAND. You are eNGLAND? They are eNGLAND too!",
            "I am ENGLAND. You are ENGLAND? They are ENGLAND too!",
            "I am pro-england. You are pro-england? They are pro-england too!",
            "I am pro-England. You are pro-England? They are pro-England too!",
            "I am pro-eNGland. You are pro-eNGland? They are pro-eNGland too!",
            "I am Pro-england. You are Pro-england? They are Pro-england too!",
            "I am PRO-england. You are PRO-england? They are PRO-england too!",
            "I am PRO-England. You are PRO-England? They are PRO-England too!",
            "I am pro_eNGLAND. You are pro_eNGLAND? They are pro_eNGLAND too!",
            "I am Pro-eNGLAND. You are Pro-eNGLAND? They are Pro-eNGLAND too!",
            "I am PRO-eNGLAND. You are PRO-eNGLAND? They are PRO-eNGLAND too!",
            "I am Pro-ENGLAND. You are Pro-ENGLAND? They are Pro-ENGLAND too!",
        ]
        # The cases we try to catch are specifically when the first word
        # is all lowercase and where the subsequent word is zero or more lowercase
        # followed by all uppercase.
        does_get_edited = [
            "I am pro-eNGLAND. You are pro-eNGLAND? They are pro-eNGLAND too!",
            "I am pro-ENGLAND. You are pro-ENGLAND? They are pro-ENGLAND too!",
            "anti-mediTERRANEAN anti-ITALY anti-GERMAN anti-gERMANY",
        ]
        edit_results = [
            "I am pro-England. You are pro-England? They are pro-England too!",
            "I am pro-England. You are pro-England? They are pro-England too!",
            "anti-Mediterranean anti-Italy anti-German anti-Germany",
        ]

        for message in does_not_get_edited:
            edited = msg_editor.maybe_edit_message(self._create_new_msg(message=message))
            assert edited is not None and edited["message"] == message, message
        for i, message in enumerate(does_get_edited):
            edited = msg_editor.maybe_edit_message(self._create_new_msg(message=message))
            expected = edit_results[i]
            assert edited is not None and edited["message"] == expected, (edited, expected)

    def test_mutes_filtering(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=True,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        does_not_get_filtered = [
            "",
            "Hey England, how are you doing?",
            "I cannot send message to this France.",
            "Has this country muted you?",
        ]
        does_get_filtered = [
            "this country has muted you",
            "This country has muted you",
            "Cannot send message this country has muted you",
            "Cannot send message  this country has muted you",
            "Cannot send message, this country has muted you",
            "Cannot send message; this country has muted you",
            "Cannot send message; this country has muted you.",
            "cannot send message; this country has muted you",
            "cannot send message; this country has muted you.",
        ]

        for message in does_not_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert not should_filter, message
        for message in does_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert should_filter, message

    def test_filter_grounding(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=True,
            filter_insults=False,
            grounding_last_playable_year=None,
        )
        does_not_get_filtered = [
            "Hey England, how about we team up against France?",
            "Last phase, you promised me we have a DMZ",
            "Do you plan to draw or solo next?",
            "France is a really experienced player",
        ]
        does_get_filtered = [
            "Italy hasn't entered their orders",
            "Good morning!",
            "I'm voting cancel",
        ]

        for message in does_not_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert not should_filter, message
        for message in does_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert should_filter, message

    def test_filter_insults(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=True,
            grounding_last_playable_year=None,
        )
        does_not_get_filtered = [
            "France is a really experienced player",
            "Your move last turn was bad.",
            "Your move last turn was not terrible.",
            "Your move last turn wasn't terrible.",
            "Your move last turn isn't so horrible.",
            "I feel terrible for you and Austria.",
            "Sorry, I'm bad at this game",
            "I'm awfully suspicious of England's intentions here.",
            "I have horrible suspicions about England's intentions here.",
            "Well that was pretty unexpected of Turkey.",
            # - random other phrases
            "Hey England, how about we team up against France?",
            "Last phase, you promised me we have a DMZ",
            "Do you plan to draw or solo next?",
        ]
        does_get_filtered = [
            "France is a really dumb player",
            "Your move last turn was terrible.",
            "Your move last turn is so horrible.",
            "Russia is stupid.",
            "France has been an idiotic ally so far.",
            "France has been a poor player so far.",
            "Sorry, Germany's bad at this game",
            "Sorry, you're bad at this game",
            "Well that was pretty incompetent of Turkey.",
            "You're an awful person",
            "You're horrible.",
            "I'm not a moron, geez.",
            "That's pretty pathetic. You need to talk Germany instead.",
            "What a retard",
            "Do you even know how to play this game?",
            "lol you're so hilariously bad at Diplomacy.",
            "It's ridiculous that you think this will work.",
        ]

        for message in does_not_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert not should_filter, message

        for message in does_get_filtered:
            should_filter = msg_filterer.should_filter_message(
                self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT
            )
            assert should_filter, message

    def test_filter_grounding_end_of_game(self):
        msg_filterer = MessageFiltering(
            filter_offensive_language=False,
            filter_phase_repeats=False,
            filter_consecutive_short=False,
            filter_excess_redacted=False,
            filter_any_redacted=False,
            filter_ampersands=False,
            filter_names=False,
            filter_urls_emails=False,
            filter_draw_discussion_when_missing_votes=False,
            filter_mutes=False,
            filter_grounding=False,
            filter_insults=False,
            grounding_last_playable_year=1904,
        )
        does_not_get_filtered = [
            "Hey England, how about we team up against France?",
            "Last phase, you promised me we have a DMZ",
            "France is a really experienced player",
            "I will move in S1904",
        ]
        does_get_filtered = [
            ("Watch out, France is going to solo if you don't stop him!", "S1902M"),
            ("How about we shoot for a draw with Germany?", "S1902M"),
            ("Longer-term we should try to stop England.", "S1903M"),
            ("In the future we should try to stop England.", "S1903M"),
            ("Eventually we should try to stop England.", "S1903M"),
            ("England should gain a lot in a upcoming several years", "S1903M"),
            ("Can you support me to Belgium next year?", "S1904M"),
            ("Can you support me to Belgium next spring?", "S1904M"),
            ("Do you plan to build a fleet?", "S1904M"),
            ("Can you support me to Belgium in a few turns?", "S1904M"),
            ("Can you support me to Belgium next turn?", "F1904M"),
            ("Can you support me to Belgium next season?", "F1904M"),
            ("By next fall, you might be in trouble", "F1904M"),
            ("By next autumn, you might be in trouble", "F1904M"),
            ("I will move in S1905", "S1901M"),
        ]

        game = pydipcc.Game()

        while sort_phase_key(game.current_short_phase) < sort_phase_key("S1905M"):
            for message in does_not_get_filtered:
                should_filter = msg_filterer.should_filter_message(
                    self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT, game=game
                )
                assert not should_filter, message

            for message, first_filter_season in does_get_filtered:
                should_filter = msg_filterer.should_filter_message(
                    self._create_new_msg(message=message), TEST_MSG_HISTORY_REPEAT, game=game
                )
                assert should_filter == (
                    sort_phase_key(game.current_short_phase) >= sort_phase_key(first_filter_season)
                ), message
            game.process()
