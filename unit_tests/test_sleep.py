#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
from fairdiplomacy.agents.parlai_message_handler import (
    apply_initiate_sleep_heuristic,
    apply_consecutive_outbound_messages_heuristic,
    SleepSixTimesCache,
    pseudoorders_initiate_sleep_heuristics_should_trigger,
)
from fairdiplomacy.game import POWERS
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.timestamp import Timestamp
from fairdiplomacy.utils.game import is_replying_to
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME


class TestIsReplyingTo(unittest.TestCase):
    def test_england_to_france(self):
        game = Game()
        game.add_message("ENGLAND", "FRANCE", "test", Timestamp.from_seconds(1))
        self.assertTrue(is_replying_to(game, "FRANCE", "ENGLAND"))
        self.assertTrue(not is_replying_to(game, "ENGLAND", "FRANCE"))
        self.assertTrue(not is_replying_to(game, "FRANCE", "GERMANY"))

    def test_england_to_france_with_extra_messages(self):
        game = Game()
        game.add_message("ENGLAND", "FRANCE", "test", Timestamp.from_seconds(1))
        game.add_message("GERMANY", "FRANCE", "test", Timestamp.from_seconds(2))
        game.add_message("FRANCE", "GERMANY", "test", Timestamp.from_seconds(3))

        # france is still replying to england despite a convo with germany
        self.assertTrue(is_replying_to(game, "FRANCE", "ENGLAND"))
        self.assertTrue(not is_replying_to(game, "ENGLAND", "FRANCE"))


class TestSleepHeuristics(unittest.TestCase):
    def test_initiate_heuristic_tiebreaking(self):
        power = "ENGLAND"
        orig_sleep_times = {
            "AUSTRIA": (INF_SLEEP_TIME, 0.9),  # should not talk, highest p(inf)
            "FRANCE": (Timestamp.from_seconds(30 * 60), 0.8),  # should talk first
            "GERMANY": (Timestamp.from_seconds(60 * 60), 0.8),  # should talk second
            "ITALY": (INF_SLEEP_TIME, 0.8),  # should not talk, middle p(inf)
            "RUSSIA": (Timestamp.from_seconds(90 * 60), 0.8),  # should talk third
            "TURKEY": (INF_SLEEP_TIME, 0.7),  # should not talk, lowest p(inf)
        }

        # send messages to each recipient
        game = Game()
        recipients = []
        sleep_cache = SleepSixTimesCache()
        for _ in range(6):
            sleep_times = apply_initiate_sleep_heuristic(
                game, power, orig_sleep_times, restrict_to_powers=POWERS
            )
            sleep_cache.set_sleep_times(game, sleep_times)
            recipient, _ = sleep_cache.get_recipient_sleep_time(game)
            recipients.append(recipient)
            game.add_message(power, recipient, "blah", Timestamp(0), increment_on_collision=True)

        # matches order described above
        self.assertEqual(recipients, ["FRANCE", "GERMANY", "RUSSIA", "TURKEY", "ITALY", "AUSTRIA"])

    def test_consecutive_outbound_messages_heuristic(self):
        sender, recipient, other = "ENGLAND", "FRANCE", "TURKEY"
        sleep_times = {
            recipient: (Timestamp.from_seconds(60), 0.8),
            other: (Timestamp.from_seconds(120), 0.6),
        }
        # after 3 messages, allow sending
        game = Game()
        game.add_message(sender, recipient, "message 1", Timestamp(1), increment_on_collision=True)
        game.add_message(sender, recipient, "message 2", Timestamp(1), increment_on_collision=True)
        game.add_message(sender, recipient, "message 3", Timestamp(1), increment_on_collision=True)
        self.assertNotEqual(
            apply_consecutive_outbound_messages_heuristic(game, sender, sleep_times)[recipient][0],
            INF_SLEEP_TIME,
        )

        # after 4 messages, do not allow sending
        game.add_message(sender, recipient, "message 4", Timestamp(1), increment_on_collision=True)
        self.assertEqual(
            apply_consecutive_outbound_messages_heuristic(game, sender, sleep_times)[recipient][0],
            INF_SLEEP_TIME,
        )

        # after 4 messages, allow sending to other power
        self.assertNotEqual(
            apply_consecutive_outbound_messages_heuristic(game, sender, sleep_times)[other][0],
            INF_SLEEP_TIME,
        )

        # after inbound message, allow sending
        game.add_message(
            recipient, sender, "inbound message", Timestamp(1), increment_on_collision=True
        )
        self.assertNotEqual(
            apply_consecutive_outbound_messages_heuristic(game, sender, sleep_times)[recipient][0],
            INF_SLEEP_TIME,
        )


class TestPseudoSleepInitiateHeuristic(unittest.TestCase):
    def test_pseudo_sleep_initiate_heuristic(self):
        # Heuristic should not trigger for a normal set of moves
        game = Game()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "S1901M": {
                        "ITALY": ("A VEN - TRI", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - GAL", "F TRI - ALB"),
                    }
                }
            ),
        )
        # Heuristic should trigger with support hold, regardless of coordination
        game = Game()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "S1901M": {
                        "ITALY": ("A VEN S F TRI", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - GAL", "F TRI - ALB"),
                    }
                }
            ),
        )

        # Heuristic should trigger with support move to new area
        game = Game()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "S1901M": {
                        "ITALY": ("A VEN S A VIE - TYR", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - GAL", "F TRI - ALB"),
                    }
                }
            ),
        )

        # As well as them supporting us
        game = Game()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "S1901M": {
                        "ITALY": ("A VEN - TYR", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE S A VEN - TYR", "F TRI - ALB"),
                    }
                }
            ),
        )

        # But not support of power to itself
        game = Game()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "S1901M": {
                        "ITALY": ("A VEN - APU", "F NAP - ION", "A ROM S A VEN - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE H", "F TRI - ALB"),
                    }
                }
            ),
        )

        # And no supporting power to their own empty SC
        game = Game()
        game.set_orders("AUSTRIA", ("F TRI - ADR",))
        game.process()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A VEN S F ADR - TRI", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE H", "F ADR - TRI"),
                    }
                }
            ),
        )
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A VEN S F ADR", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE H", "F ADR - TRI"),
                    }
                }
            ),
        )

        # Attacking a common enemy
        game = Game()
        game.set_orders("GERMANY", ("A MUN - TYR",))
        game.process()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A VEN - TYR", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - TYR", "F TRI - ADR"),
                    }
                }
            ),
        )
        game = Game()
        game.process()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A VEN - TYR", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - TYR", "F TRI - ADR"),
                    }
                }
            ),
        )

        # Or a common enemy's SC
        game = Game()
        game.set_orders("GERMANY", ("A MUN - BOH",))
        game.set_orders("ITALY", ("A VEN - TYR",))
        game.process()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A TYR - MUN", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - BOH", "F TRI - ADR"),
                    }
                }
            ),
        )
        game = Game()
        game.set_orders("GERMANY", ("A MUN - BOH",))
        game.set_orders("ITALY", ("A VEN - TYR",))
        game.process()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A TYR - VIE", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - BOH", "F TRI - ADR"),
                    }
                }
            ),
        )
        game = Game()
        game.set_orders("GERMANY", ("A MUN - BOH",))
        game.set_orders("ITALY", ("A VEN - TYR",))
        game.process()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A TYR - MUN", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - TYR", "F TRI - ADR"),
                    }
                }
            ),
        )
        game = Game()
        game.set_orders("GERMANY", ("A MUN - SIL",))
        game.set_orders("ITALY", ("A VEN - TYR",))
        game.process()
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game,
            "ITALY",
            "AUSTRIA",
            PseudoOrders(
                {
                    "F1901M": {
                        "ITALY": ("A TYR - MUN", "F NAP - ION", "A ROM - APU"),
                        "AUSTRIA": ("A BUD - SER", "A VIE - BOH", "F TRI - ADR"),
                    }
                }
            ),
        )

    def test_pseudo_sleep_initiate_heuristic_only_initiate(self):
        pseudos = PseudoOrders(
            {
                "S1901M": {
                    "ITALY": ("A VEN S F TRI", "F NAP - ION", "A ROM - APU"),
                    "AUSTRIA": ("A BUD - SER", "A VIE - GAL", "F TRI - ALB"),
                }
            }
        )

        # these pseudos do trigger the heuristic at phase start
        game = Game()
        assert pseudoorders_initiate_sleep_heuristics_should_trigger(
            game, "ITALY", "AUSTRIA", pseudos
        )

        # once we've already sent a message, heuristic should not trigger
        game.add_message("ITALY", "AUSTRIA", "This is a convo initiation message", Timestamp(1))
        assert not pseudoorders_initiate_sleep_heuristics_should_trigger(
            game, "ITALY", "AUSTRIA", pseudos
        )
