#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
import json
import os

import numpy.testing
import torch
from fairdiplomacy.data.build_dataset import get_valid_coastal_variant
from fairdiplomacy.models.base_strategy_model.base_strategy_model import Scoring
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state
import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.thread_pool_encoding import (
    MAX_INPUT_VERSION,
    FeatureEncoder,
    get_board_state_size,
)
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY, action_strs_to_global_idxs
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.models.consts import MAX_SEQ_LEN, N_SCS, POWERS, LOCS
from fairdiplomacy.typedefs import JointAction, Timestamp


TEST_DATA = heyhi.PROJ_ROOT / "unit_tests/data"
GAME_WITH_COASTS = TEST_DATA / "game_fva_order_idx_with_coasts_test.json"


# Some sampled orders indices for the begining of a game
ORDERS_IDXS_ALL_POWER = torch.tensor(
    [
        [
            7190,
            8785,
            2130,
            6619,
            8697,
            2866,
            2287,
            2518,
            444,
            11899,
            2417,
            5179,
            11460,
            10161,
            3349,
            4848,
            5010,
            12066,
            670,
            5801,
            3929,
            1080,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    ]
    + [[-1] * 34] * 6
).unsqueeze(0)


class TestPydipcc(unittest.TestCase):
    def test_import_inplace(self):
        game = pydipcc.Game()
        game.set_orders("RUSSIA", ["F SEV - BLA"])
        game.process()

        game2 = pydipcc.Game()
        game2.from_json_inplace(game.to_json())
        self.assertEqual(game.current_short_phase, game2.current_short_phase)

    def test_rollback_preserve_orders_true(self):
        game = pydipcc.Game()
        game.set_orders("RUSSIA", ["F SEV - BLA"])
        game.process()
        game = game.rolled_back_to_phase_end("S1901M")
        self.assertEqual(game.get_orders()["RUSSIA"], ["F SEV - BLA"])

    def test_rollback_preserve_orders_false(self):
        game = pydipcc.Game()
        game.set_orders("RUSSIA", ["F SEV - BLA"])
        game.process()
        game = game.rolled_back_to_phase_start("S1901M")
        self.assertEqual(len(game.get_orders()), 0)

    def test_rollback_messages_time0(self):
        game = pydipcc.Game.from_json(open(TEST_DATA / "game_100012_anonymized.json").read())
        game = game.rolled_back_to_timestamp_start(Timestamp(0))
        self.assertEqual(len(game.messages), 0)

    def test_add_message(self):
        game = pydipcc.Game()
        game.add_message("FRANCE", "AUSTRIA", "blast from the past", Timestamp.from_centis(12345))
        self.assertEqual(len(game.messages), 1)
        assert 12345 in game.messages.keys(), game.messages.keys()

        with self.assertRaises(Exception):
            game.add_message(
                "FRANCE", "AUSTRIA", "duplicate timestamp!", Timestamp.from_centis(12345)
            )

    def test_delete_message_at_timestamp(self):
        game = pydipcc.Game()
        assert game.get_last_message_timestamp() == 0

        game.add_message("FRANCE", "AUSTRIA", "blast from the past", Timestamp.from_centis(12345))
        self.assertEqual(len(game.messages), 1)
        assert 12345 in game.messages.keys(), game.messages.keys()
        assert game.get_last_message_timestamp() == Timestamp.from_centis(12345)
        game.add_message("AUSTRIA", "FRANCE", "back at ya", Timestamp.from_centis(34567))
        self.assertEqual(len(game.messages), 2)
        assert 12345 in game.messages.keys(), game.messages.keys()
        assert 34567 in game.messages.keys(), game.messages.keys()
        assert game.get_last_message_timestamp() == Timestamp.from_centis(34567)

        game.delete_message_at_timestamp(Timestamp.from_centis(12345))
        self.assertEqual(len(game.messages), 1)
        assert 12345 not in game.messages.keys(), game.messages.keys()
        assert 34567 in game.messages.keys(), game.messages.keys()
        assert game.get_last_message_timestamp() == Timestamp.from_centis(34567)

        game.add_message(
            "FRANCE", "AUSTRIA", "duplicate timestamp added back!", Timestamp.from_centis(12345)
        )
        self.assertEqual(len(game.messages), 2)
        assert 12345 in game.messages.keys(), game.messages.keys()
        assert 34567 in game.messages.keys(), game.messages.keys()
        assert game.get_last_message_timestamp() == Timestamp.from_centis(34567)

        game.process()
        assert game.get_last_message_timestamp() == Timestamp.from_centis(34567)
        self.assertEqual(len(game.get_phase_history()[-1].messages), 2)

        game.delete_message_at_timestamp(Timestamp.from_centis(34567))
        self.assertEqual(len(game.messages), 0)
        self.assertEqual(len(game.get_phase_history()[-1].messages), 1)
        assert 12345 in game.get_phase_history()[-1].messages
        assert game.get_last_message_timestamp() == Timestamp.from_centis(12345)

        game.delete_message_at_timestamp(Timestamp.from_centis(12345))
        self.assertEqual(len(game.messages), 0)
        self.assertEqual(len(game.get_phase_history()[-1].messages), 0)
        assert 12345 not in game.get_phase_history()[-1].messages
        assert 34567 not in game.get_phase_history()[-1].messages
        assert game.get_last_message_timestamp() == 0

    def test_influence(self):
        game = pydipcc.Game()
        self.assertEqual(set(game.get_state()["influence"]["FRANCE"]), {"PAR", "MAR", "BRE"})
        game.set_orders("FRANCE", ["A PAR - BUR"])
        game.process()
        self.assertEqual(
            set(game.get_state()["influence"]["FRANCE"]), {"PAR", "MAR", "BRE", "BUR"}
        )
        game.set_orders("FRANCE", ["A BUR - BEL"])
        game.set_orders("GERMANY", ["A MUN - BUR"])
        game.process()
        self.assertEqual(
            set(game.get_state()["influence"]["FRANCE"]), {"PAR", "MAR", "BRE", "BEL"}
        )

    def test_clear_orders(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A PAR - BUR"])
        game.set_orders("GERMANY", ["A MUN - RUH"])
        self.assertEqual(game.get_orders()["FRANCE"], ["A PAR - BUR"])
        self.assertEqual(game.get_orders()["GERMANY"], ["A MUN - RUH"])
        game.clear_orders()
        for power, orders in game.get_orders():
            self.assertEqual(orders, [])

    def test_get_messages_timestamp_class(self):
        game = pydipcc.Game()
        game.add_message("FRANCE", "AUSTRIA", "blast from the past", Timestamp.from_centis(12345))
        timestamp = list(game.messages.keys())[0]

        # dict key and time_sent value must be Timestamp instance
        self.assertEqual(timestamp.to_centis(), 12345)
        self.assertEqual(list(game.messages.values())[0]["time_sent"].to_centis(), 12345)

        # timestamp must be serializable / deserializable
        # then check the same thing
        game = pydipcc.Game.from_json(game.to_json())
        self.assertEqual(timestamp.to_centis(), 12345)
        self.assertEqual(list(game.messages.values())[0]["time_sent"].to_centis(), 12345)

    def serialize_metadata(self):
        game = pydipcc.Game()
        game.set_metadata("foo", "bar")
        self.assertEqual(game.get_metadata("foo"), "bar")

        j = game.to_json()
        game2 = pydipcc.Game.from_json(j)
        self.assertEqual(game2.get_metadata("foo"), "bar")

    def test_clone_n_times(self):
        game = pydipcc.Game()
        game.game_id = "a_game"
        games = game.clone_n_times(5)
        self.assertEqual(len(games), 5)
        self.assertEqual(games[0].game_id, "a_game_0")
        self.assertEqual(games[1].game_id, "a_game_1")


class TestEncoding(unittest.TestCase):
    def test_russia_four_builds(self):
        """Test for a bug in which coastal builds were not in russia's possible orders"""
        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            game = pydipcc.Game.from_json(f.read())
        encoder = FeatureEncoder()
        fields = encoder.encode_inputs([game], input_version=1)
        assert ORDER_VOCABULARY[13201] == "A MOS B;A SEV B;A WAR B;F STP/NC B"
        assert (
            13201 in fields["x_possible_actions"][0, 5]
        ), "Order not found in Russia's possible orders"

    def test_all_powers_encoding_m_phase(self):
        encoder = FeatureEncoder()
        game = pydipcc.Game()

        x_allp = encoder.encode_inputs_all_powers([game], input_version=1)
        x_orig = encoder.encode_inputs([game], input_version=1)

        # For M-phase, only encode in power-0 spot
        assert (x_allp["x_possible_actions"][0, 1:] == EOS_IDX).all()
        assert (x_allp["x_power"][0, 1:] == EOS_IDX).all()

        # Both encoding should contain the same set of possible actions
        assert (
            x_allp["x_possible_actions"][x_allp["x_possible_actions"] != EOS_IDX].sum()
            == x_orig["x_possible_actions"][x_orig["x_possible_actions"] != EOS_IDX].sum()
        )

        # Check that they're in the right spot
        power_steps = [0] * 7
        for step, power_i in enumerate(x_allp["x_power"][0, 0]):
            if power_i == EOS_IDX:
                break
            assert (
                x_allp["x_possible_actions"][0, 0, step]
                == x_orig["x_possible_actions"][0, power_i, power_steps[power_i]]
            ).all()
            power_steps[power_i] += 1
        assert sum(power_steps) == 22  # num. starting units

        # Validate x_loc_idxs
        assert x_allp["x_loc_idxs"].shape[0] == 1, x_allp["x_loc_idxs"].shape[0]
        assert (x_allp["x_loc_idxs"][0, 1:] == EOS_IDX).all()
        assert x_allp["x_loc_idxs"][0, 0].max() == 21
        assert len(x_allp["x_loc_idxs"][0, 0][x_allp["x_loc_idxs"][0, 0] != EOS_IDX]) == 22

    def test_all_powers_encoding_a_phase(self):
        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            game = pydipcc.Game.from_json(f.read())
        assert game.current_short_phase.endswith("A")

        encoder = FeatureEncoder()

        x_allp = encoder.encode_inputs_all_powers([game], input_version=1)
        x_orig = encoder.encode_inputs([game], input_version=1)

        # All values should be equal except for longer seq len
        for k in x_orig.keys():
            if x_allp[k].ndim >= 3 and x_allp[k].shape[2] == N_SCS:
                # shrink tensor in seq dim so they can be compared
                assert (x_allp[k][:, :, MAX_SEQ_LEN:] == -1).all()
                x_allp[k] = x_allp[k][:, :, :MAX_SEQ_LEN]
            assert (x_allp[k] == x_orig[k]).all()

    def test_json_idempotence1(self):
        # If all the data writing and reading is correct, it should be the case that
        # two cycles of jsoning equals one cycle of jsoning.
        # (one might not equal zero due to ordering of dict fields, dropping extraneous data, etc).

        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            game0 = pydipcc.Game.from_json(f.read())

        json1 = game0.to_json()
        game1 = pydipcc.Game.from_json(json1)
        json2 = game1.to_json()
        assert json1 == json2

    def test_json_idempotence2(self):
        # A slightly stricter test than test_json_idempotence1, which might miss cases where
        # data is lost when parsing json -> game.
        game0 = pydipcc.Game()
        game0.set_orders("RUSSIA", ["F SEV - BLA"])
        game0.add_message("ENGLAND", "FRANCE", "hi", Timestamp.from_centis(12345))
        game0.process()
        game0.set_orders("RUSSIA", ["F BLA - RUM"])
        game0.add_message("FRANCE", "ENGLAND", "yo", Timestamp.from_centis(12345))

        json1 = game0.to_json()
        game1 = pydipcc.Game.from_json(json1)
        json2 = game1.to_json()
        print(json1)
        print(json2)
        assert json1 == json2

    def test_alive_powers(self):
        game = pydipcc.Game()
        self.assertEqual(game.get_alive_powers(), POWERS)
        self.assertEqual(game.get_alive_power_ids(), list(range(len(POWERS))))

    def test_encode_orders(self):
        game = pydipcc.Game()
        power_orders = dict([("RUSSIA", ["F SEV - BLA"]), ("GERMANY", ["F KIE - HOL"])])
        for power, orders in power_orders.items():
            game.set_orders(power, orders)
        game.process()

        encoder = FeatureEncoder()
        x_prev_orders = encoder.encode_inputs([game], 2)["x_prev_orders"][0]

        this_orders_strict = encoder.encode_orders_single_strict(sum(power_orders.values(), []), 2)
        this_orders_tolerant = encoder.encode_orders_single_tolerant(
            game, sum(power_orders.values(), []), 2
        )

        print(x_prev_orders)
        print(this_orders_strict)
        print(this_orders_tolerant)

        self.assertTrue((x_prev_orders == this_orders_strict).all())
        self.assertTrue((x_prev_orders == this_orders_tolerant).all())

    def test_alive_powers_fva(self):
        with GAME_WITH_COASTS.open() as stream:
            game = pydipcc.Game.from_json(stream.read())
        alive = sorted(["AUSTRIA", "FRANCE"], key=POWERS.index)
        self.assertEqual(game.get_alive_powers(), alive)
        alive_ids = [POWERS.index(p) for p in alive]
        self.assertEqual(game.get_alive_power_ids(), alive_ids)

    def test_set_orders_build(self):
        game = pydipcc.Game()
        game.set_orders("GERMANY", ["F KIE - HOL"])
        self.assertEqual(game.get_orders()["GERMANY"], ["F KIE - HOL"])
        assert "F KIE" in game.get_units()["GERMANY"]
        assert "A KIE" not in game.get_units()["GERMANY"]
        game.process()
        self.assertEqual(game.get_orders().get("GERMANY", []), [])
        assert "F KIE" not in game.get_units()["GERMANY"]
        assert "A KIE" not in game.get_units()["GERMANY"]
        game.process()
        game.set_orders("GERMANY", ["A KIE B"])
        self.assertEqual(game.get_orders()["GERMANY"], ["A KIE B"])
        assert "F KIE" not in game.get_units()["GERMANY"]
        assert "A KIE" not in game.get_units()["GERMANY"]
        game.process()
        self.assertEqual(game.get_orders().get("GERMANY", []), [])
        assert "F KIE" not in game.get_units()["GERMANY"]
        assert "A KIE" in game.get_units()["GERMANY"]

    def test_game_phase_of_message_and_rollback(self):
        game = pydipcc.Game()
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(0)) == "S1901M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(10000)) == "S1901M"
        game.add_message(
            "FRANCE", "ENGLAND", "hey england, how's it going", Timestamp.from_centis(1000)
        )
        game.add_message(
            "FRANCE", "ENGLAND", "hey england, how's it going", Timestamp.from_centis(1001)
        )
        game.process()  # finish S1901M
        game.add_message("ENGLAND", "FRANCE", "not bad", Timestamp.from_centis(2000))
        game.process()  # finish F1901M
        game.add_message("FRANCE", "ENGLAND", "ok cool", Timestamp.from_centis(3000))
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(0)) == "S1901M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(1999)) == "S1901M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(2000)) == "F1901M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(2999)) == "F1901M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(3000)) == "S1902M"
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(4000)) == "S1902M"
        game.set_orders("GERMANY", ["F KIE - HOL"])
        game.add_message("TURKEY", "TURKEY", "hello me", Timestamp.from_centis(3100))
        game.process()  # finish S1902M
        game.process()  # finish F1902M
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(4000)) == "S1902M"
        game.add_message("TURKEY", "TURKEY", "hello me again", Timestamp.from_centis(3200))
        assert game.phase_of_last_message_at_or_before(Timestamp.from_centis(4000)) == "W1902A"
        game.process()  # finish W1902A
        assert game.get_current_phase() == "S1903M"

        def len_messages(game_copy, phase):
            return sum(
                len(phase_data.messages)
                for phase_data in game_copy.get_all_phases()
                if phase_data.name == phase
            )

        # start and end include the message on the exact time stamp or not
        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(1000))
        assert game_copy.get_current_phase() == "S1901M"
        assert len_messages(game_copy, "S1901M") == 0
        assert len_messages(game_copy, "F1901M") == 0
        game_copy = game.rolled_back_to_timestamp_end(Timestamp.from_centis(1000))
        assert game_copy.get_current_phase() == "S1901M"
        assert len_messages(game_copy, "S1901M") == 1
        assert len_messages(game_copy, "F1901M") == 0
        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(1001))
        assert game_copy.get_current_phase() == "S1901M"
        assert len_messages(game_copy, "S1901M") == 1
        assert len_messages(game_copy, "F1901M") == 0
        game_copy = game.rolled_back_to_timestamp_end(Timestamp.from_centis(1001))
        assert game_copy.get_current_phase() == "S1901M"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 0
        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(1999))
        assert game_copy.get_current_phase() == "S1901M"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 0

        # rolling back to the phase of the message but don't get the message since start.
        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(2000))
        assert game_copy.get_current_phase() == "F1901M"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 0

        # rolling back to the phase of the message and do get the message since end.
        game_copy = game.rolled_back_to_timestamp_end(Timestamp.from_centis(2000))
        assert game_copy.get_current_phase() == "F1901M"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 1
        assert "F HOL" not in game_copy.get_state()["units"]["GERMANY"]

        # rolling back to just before timestamp still goes all the way to the phase
        # of the previous message, two phases ago
        # staged orders is empty even though that phase had orders
        game_copy = game.rolled_back_to_timestamp_end(Timestamp.from_centis(3199))
        assert game_copy.get_current_phase() == "S1902M"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 1
        assert len_messages(game_copy, "S1902M") == 2
        assert len_messages(game_copy, "F1902M") == 0
        assert len_messages(game_copy, "W1902A") == 0
        assert len(game_copy.get_staged_phase_data().orders) == 0
        assert "F HOL" not in game_copy.get_state()["units"]["GERMANY"]

        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(3200))
        assert game_copy.get_current_phase() == "W1902A"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 1
        assert len_messages(game_copy, "S1902M") == 2
        assert len_messages(game_copy, "F1902M") == 0
        assert len_messages(game_copy, "W1902A") == 0
        assert "F HOL" in game_copy.get_state()["units"]["GERMANY"]

        # there are no messages on the last phase, so rolling back to timestamp
        # always only goes to the last phase that does have a message
        game_copy = game.rolled_back_to_timestamp_start(Timestamp.from_centis(100000))
        assert game_copy.get_current_phase() == "W1902A"
        assert len_messages(game_copy, "S1901M") == 2
        assert len_messages(game_copy, "F1901M") == 1
        assert len_messages(game_copy, "S1902M") == 2
        assert len_messages(game_copy, "F1902M") == 0
        assert len_messages(game_copy, "W1902A") == 1
        assert "F HOL" in game_copy.get_state()["units"]["GERMANY"]

    def test_phase_data_getters(self):
        game = pydipcc.Game()
        assert game.get_current_phase() == "S1901M"
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.add_message("RUSSIA", "TURKEY", "ahoy Turkey, it's F1901M", Timestamp.from_centis(0))
        game.process()
        assert game.get_current_phase() == "F1901M"
        game.set_orders("ENGLAND", ["F LON - ENG"])
        game.add_message(
            "TURKEY", "RUSSIA", "hey there russia, it's S1902M", Timestamp.from_centis(10)
        )
        game.process()
        assert game.get_current_phase() == "W1901A"
        game.set_orders("AUSTRIA", ["A BUD B"])
        game.add_message("AUSTRIA", "AUSTRIA", "i love builds!", Timestamp.from_centis(20))
        game.process()
        assert game.get_current_phase() == "S1902M"
        game.set_orders("GERMANY", ["A MUN - BOH"])
        game.add_message(
            "FRANCE", "ENGLAND", "hey there england, it's S1902M", Timestamp.from_centis(30)
        )
        assert len(game.get_phase_history()) == 3
        assert len(game.get_all_phases()) == 4
        assert game.get_all_phase_names() == ["S1901M", "F1901M", "W1901A", "S1902M"]

        for i in range(3):
            assert game.get_phase_history()[i].to_dict() == game.get_all_phases()[i].to_dict()

        assert game.get_all_phases()[3].to_dict() == game.get_staged_phase_data().to_dict()
        assert "A MUN - BOH" in game.get_staged_phase_data().orders["GERMANY"]
        assert (
            "GERMANY" not in game.get_phase_data().orders
            or "A MUN - BOH" not in game.get_phase_data().orders["GERMANY"]
        )
        assert (
            "hey there england, it's S1902M"
            == game.get_staged_phase_data().messages[Timestamp.from_centis(30)]["message"]
        )
        assert len(game.get_phase_data().messages) == 0

    def test_set_all_orders(self):
        game = pydipcc.Game()
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.set_orders("ENGLAND", ["F LON - ENG"])
        game.set_orders("RUSSIA", ["A WAR - PRU", "F STP/SC - FIN"])
        game.set_orders("ITALY", ["A VEN - PIE", "F NAP - TYS"])

        def has_order(game, power, order):
            return (
                power in game.get_staged_phase_data().orders
                and order in game.get_staged_phase_data().orders[power]
            )

        assert has_order(game, "AUSTRIA", "A BUD - SER")
        assert has_order(game, "ENGLAND", "F LON - ENG")
        assert has_order(game, "RUSSIA", "A WAR - PRU")
        assert has_order(game, "RUSSIA", "F STP/SC - FIN")
        assert has_order(game, "ITALY", "A VEN - PIE")
        assert has_order(game, "ITALY", "F NAP - TYS")

        game.set_all_orders(
            {
                "ENGLAND": ("F EDI - NTH",),
                "RUSSIA": ("F STP/SC - FIN",),
                "ITALY": ("A VEN - TYR", "F NAP - ION", "A ROM - APU"),
                "TURKEY": ("A CON - BUL",),
            }
        )

        assert not has_order(game, "AUSTRIA", "A BUD - SER")
        assert not has_order(game, "ENGLAND", "F LON - ENG")
        assert not has_order(game, "RUSSIA", "A WAR - PRU")
        assert has_order(game, "RUSSIA", "F STP/SC - FIN")
        assert not has_order(game, "ITALY", "A VEN - PIE")
        assert not has_order(game, "ITALY", "F NAP - TYS")

        assert has_order(game, "ENGLAND", "F EDI - NTH")
        assert has_order(game, "ITALY", "A VEN - TYR")
        assert has_order(game, "ITALY", "F NAP - ION")
        assert has_order(game, "ITALY", "A ROM - APU")
        assert has_order(game, "TURKEY", "A CON - BUL")

    def test_set_orders_wrong_power(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A PAR - BUR", "F SEV - BLA"])

        # don't include "F SEV - BLA", owned by TURKEY
        self.assertEqual(game.get_orders()["FRANCE"], ["A PAR - BUR"])
        self.assertEqual(game.get_orders().get("TURKEY", []), [])

    def test_set_orders_no_unit(self):
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["A BUR - PAR"])

        # no orders set
        self.assertEqual(game.get_orders().get("FRANCE", []), [])

    def test_board_history_hash(self):
        game = pydipcc.Game()
        game.set_orders("AUSTRIA", ["A BUD - SER"])
        game.process()
        game.set_orders("ENGLAND", ["F LON - ENG"])
        game.process()

        game_same = pydipcc.Game()
        game_same.set_orders("AUSTRIA", ["A BUD - SER"])
        game_same.process()
        game_same.set_orders("ENGLAND", ["F LON - ENG"])
        game_same.process()

        self.assertEqual(game.compute_board_hash(), game_same.compute_board_hash())
        self.assertEqual(game.compute_order_history_hash(), game_same.compute_order_history_hash())

        # Same orders, but in different sequence.
        game2 = pydipcc.Game()
        game2.set_orders("ENGLAND", ["F LON - ENG"])
        game2.process()
        game2.set_orders("AUSTRIA", ["A BUD - SER"])
        game2.process()

        self.assertEqual(game.compute_board_hash(), game2.compute_board_hash())
        self.assertNotEqual(game.compute_order_history_hash(), game2.compute_order_history_hash())


class TestActionSorting(unittest.TestCase):
    def test_to_json(self):
        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            game = pydipcc.Game.from_json(f.read())

        for phase in json.loads(game.to_json())["phases"]:
            power_actions = phase["orders"]
            for power, action in power_actions.items():
                self.assertEqual(
                    action, sorted(action, key=lambda order: LOCS.index(order.split()[1]))
                )

    def test_phase_history(self):
        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            game = pydipcc.Game.from_json(f.read())

        for phase in game.get_phase_history():
            power_actions = phase.orders
            for power, action in power_actions.items():
                self.assertEqual(
                    action, sorted(action, key=lambda order: LOCS.index(order.split()[1]))
                )

    def test_get_orders(self):
        with open(os.path.dirname(__file__) + "/data/test_game_russia_four_builds.json") as f:
            old_game = pydipcc.Game.from_json(f.read())

        game = pydipcc.Game()
        for power, action in old_game.get_phase_history()[0].orders.items():
            game.set_orders(power, list(reversed(action)))  # put them in not LOCS-ordered
        for power, action in game.get_orders().items():
            self.assertEqual(
                action, sorted(action, key=lambda order: LOCS.index(order.split()[1]))
            )

    def test_decode_order_idxs_builds(self):
        global_order_idx = action_strs_to_global_idxs(["A NAP B;A VEN B;F ROM B"])[0]
        idxs = torch.full((1, 7, 17), EOS_IDX)
        idxs[0, POWERS.index("ITALY"), 0] = global_order_idx
        orders = FeatureEncoder().decode_order_idxs(idxs)[0][POWERS.index("ITALY")]
        self.assertEqual(orders, ["A NAP B", "F ROM B", "A VEN B"])
        # NAP < ROM < VEN in LOCS ordering

    def test_decode_order_idxs_allpower(self):
        batch = FeatureEncoder().encode_inputs_all_powers([pydipcc.Game()], 3)
        div = 1
        print(ORDERS_IDXS_ALL_POWER.shape, batch["x_in_adj_phase"].shape)
        print(ORDERS_IDXS_ALL_POWER)
        ground_truth = decode_order_idxs_allpower_golden(
            ORDERS_IDXS_ALL_POWER, batch["x_in_adj_phase"], batch["x_power"], div
        )
        fast = FeatureEncoder().decode_order_idxs_all_powers(
            ORDERS_IDXS_ALL_POWER, batch["x_in_adj_phase"], batch["x_power"], div
        )
        self.assertEqual(fast, ground_truth)


def decode_order_idxs_allpower_golden(order_idx, x_in_adj_phase_batched, x_power_batched, div):
    decoded = FeatureEncoder().decode_order_idxs(order_idx)
    for (i, powers_orders) in enumerate(decoded):
        x_in_adj_phase = x_in_adj_phase_batched[i // div]
        x_power = x_power_batched[i // div]
        assert len(powers_orders) == 7
        if x_in_adj_phase:
            continue
        assert all(len(orders) == 0 for orders in powers_orders[1:])
        all_orders = powers_orders[0]  # all powers' orders
        powers_orders[0] = []
        for power_idx, order in zip(x_power[0], all_orders):
            if power_idx == -1:
                break
            powers_orders[power_idx].append(order)
    return decoded


class TestScoring(unittest.TestCase):
    def test_scoring(self):
        with open(os.path.dirname(__file__) + "/data/game_no_press_from_selfplay_long.json") as f:
            gamesos = pydipcc.Game.from_json(f.read())
            gamedss = pydipcc.Game(gamesos)
            gamedss.set_scoring_system(pydipcc.Game.SCORING_DSS)
            gamedss2 = pydipcc.Game.from_json(gamedss.to_json())

        assert pydipcc.Game.SCORING_SOS == Scoring.SOS.value
        assert pydipcc.Game.SCORING_DSS == Scoring.DSS.value

        assert gamesos.get_phase_history()[77].name == "S1918M"
        numpy.testing.assert_almost_equal(
            gamesos.get_phase_history()[77].get_scores(pydipcc.Game.SCORING_SOS),
            [64 / 276, 81 / 276, 0.0, 1 / 276, 0.0, 81 / 276, 49 / 276],
        )
        numpy.testing.assert_almost_equal(
            gamesos.get_phase_history()[77].get_scores(pydipcc.Game.SCORING_DSS),
            [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2],
        )

        gamesos = gamesos.rolled_back_to_phase_start("S1918M")
        gamedss = gamedss.rolled_back_to_phase_start("S1918M")
        gamedss2 = gamedss2.rolled_back_to_phase_start("S1918M")

        self.assertEqual(
            gamesos.get_alive_powers(), ["AUSTRIA", "ENGLAND", "GERMANY", "RUSSIA", "TURKEY"]
        )
        self.assertEqual(
            gamedss.get_alive_powers(), ["AUSTRIA", "ENGLAND", "GERMANY", "RUSSIA", "TURKEY"]
        )
        self.assertEqual(
            gamedss2.get_alive_powers(), ["AUSTRIA", "ENGLAND", "GERMANY", "RUSSIA", "TURKEY"]
        )

        # Centers are AUS ENG FRA GER ITA RUS TUR
        # 8 9 0 1 0 9 7
        numpy.testing.assert_almost_equal(
            gamesos.get_scores(), [64 / 276, 81 / 276, 0.0, 1 / 276, 0.0, 81 / 276, 49 / 276]
        )
        numpy.testing.assert_almost_equal(
            gamedss.get_scores(), [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2]
        )
        numpy.testing.assert_almost_equal(
            gamedss2.get_scores(), [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2]
        )

        numpy.testing.assert_almost_equal(
            gamesos.get_scores(pydipcc.Game.SCORING_SOS),
            [64 / 276, 81 / 276, 0.0, 1 / 276, 0.0, 81 / 276, 49 / 276],
        )
        numpy.testing.assert_almost_equal(
            gamedss.get_scores(pydipcc.Game.SCORING_SOS),
            [64 / 276, 81 / 276, 0.0, 1 / 276, 0.0, 81 / 276, 49 / 276],
        )
        numpy.testing.assert_almost_equal(
            gamedss2.get_scores(pydipcc.Game.SCORING_SOS),
            [64 / 276, 81 / 276, 0.0, 1 / 276, 0.0, 81 / 276, 49 / 276],
        )

        numpy.testing.assert_almost_equal(
            gamesos.get_scores(pydipcc.Game.SCORING_DSS), [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2]
        )
        numpy.testing.assert_almost_equal(
            gamedss.get_scores(pydipcc.Game.SCORING_DSS), [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2]
        )
        numpy.testing.assert_almost_equal(
            gamedss2.get_scores(pydipcc.Game.SCORING_DSS), [0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2]
        )

        numpy.testing.assert_almost_equal(
            compute_game_scores_from_state(0, gamesos.get_phase_data().state).square_score,
            64 / 276,
        )
        numpy.testing.assert_almost_equal(
            compute_game_scores_from_state(0, gamesos.get_phase_data().state).draw_score, 0.2
        )

        self.assertEqual(gamesos.get_scoring_system(), pydipcc.Game.SCORING_SOS)
        self.assertEqual(gamedss.get_scoring_system(), pydipcc.Game.SCORING_DSS)
        self.assertEqual(gamedss2.get_scoring_system(), pydipcc.Game.SCORING_DSS)


class TestInputVersionFeatureWidth(unittest.TestCase):
    def test(self):
        assert MAX_INPUT_VERSION >= 3
        assert get_board_state_size(input_version=1) == 35
        assert get_board_state_size(input_version=2) == 38
        assert get_board_state_size(input_version=3) == 38


class TestDuplicateCoastalOrders(unittest.TestCase):
    def test_v1v2_bad_v3_good(self):
        with open(os.path.dirname(__file__) + "/data/buggyordersgame.json") as f:
            game = pydipcc.Game.from_json(f.read())
        x_possible_actions_v1 = FeatureEncoder().encode_inputs([game], input_version=1)[
            "x_possible_actions"
        ]
        # Coastal order 6484 is replicated 3 times
        assert (x_possible_actions_v1[0][-1, 1] == 6484).sum() == 3

        x_possible_actions_v2 = FeatureEncoder().encode_inputs([game], input_version=2)[
            "x_possible_actions"
        ]
        # Coastal order 6484 is replicated 3 times
        assert (x_possible_actions_v2[0][-1, 1] == 6484).sum() == 3

        x_possible_actions_v3 = FeatureEncoder().encode_inputs([game], input_version=3)[
            "x_possible_actions"
        ]
        # Input version 3 should be fixing this duplicated order bug.
        # Coastal order 6484 is replicated 1 times
        assert (x_possible_actions_v3[0][-1, 1] == 6484).sum() == 1

        # The orders are exactly identical other than this duplication.
        v1idxs = x_possible_actions_v1[0][-1, 1]
        v2idxs = x_possible_actions_v2[0][-1, 1]
        v3idxs = x_possible_actions_v3[0][-1, 1]
        v1idxs_filtered = v1idxs[(v1idxs != 6484) & (v2idxs != EOS_IDX)]
        v2idxs_filtered = v2idxs[(v2idxs != 6484) & (v2idxs != EOS_IDX)]
        v3idxs_filtered = v3idxs[(v3idxs != 6484) & (v3idxs != EOS_IDX)]
        numpy.testing.assert_equal(v1idxs_filtered.numpy(), v3idxs_filtered.numpy())
        numpy.testing.assert_equal(v2idxs_filtered.numpy(), v3idxs_filtered.numpy())

        # There are 40 other besides this one.
        assert len(v3idxs_filtered) == 40


class TestDoubleConvoyUnresolveSupport(unittest.TestCase):
    def test(self):
        with open(os.path.dirname(__file__) + "/data/double_convoy_unresolve_support.json") as f:
            game = pydipcc.Game.from_json(f.read())
        # This used to crash
        game.process()
        assert game.current_short_phase == "S1908R"
        assert game.get_units() == {
            "AUSTRIA": [],
            "ENGLAND": ["A YOR", "F LON", "F NTH", "A WAL", "F DEN", "A NWY"],
            "FRANCE": [
                "F ENG",
                "A BEL",
                "A HOL",
                "F SPA/SC",
                "A MAR",
                "A MUN",
                "*A WAL",
                "*F NTH",
            ],
            "GERMANY": ["A KIE"],
            "ITALY": ["F WES", "A PIE", "F ION", "F AEG", "F SMY"],
            "RUSSIA": ["F FIN", "A BER", "A MOS", "A TYR", "A GAL", "A TRI", "A BUD", "A SER"],
            "TURKEY": ["F SEV", "F BLA", "A GRE", "F CON", "F BUL/SC", "*A SER"],
        }


class TestMatchingWebdip(unittest.TestCase):
    @staticmethod
    def _get_trivial_r_phase_game() -> pydipcc.Game:
        game = pydipcc.Game()
        game.set_all_orders(
            {
                "RUSSIA": ("A MOS - UKR", "A WAR - GAL"),
                "TURKEY": ("A CON - BUL", "A SMY - CON"),
                "AUSTRIA": ("A BUD - RUM", "F TRI - ADR", "A VIE - TRI"),
            }
        )
        game.process()
        game.set_all_orders(
            {
                "RUSSIA": ("F SEV S A UKR - RUM", "A UKR - RUM"),
                "TURKEY": ("A BUL - SER", "A CON - BUL"),
                "AUSTRIA": ("A TRI - BUD",),
            }
        )
        game.process()
        # A RUM is force-disbanded, and no other unit is dislodged: this is a trivial R-phase
        return game

    def test_trivial_r_phase(self):
        game = self._get_trivial_r_phase_game()
        self.assertEqual(game.current_short_phase, "F1901R")
        self.assertEqual(game.get_all_possible_orders()["RUM"], ["A RUM D"])

    def test_trivial_r_phase_state(self):
        game = self._get_trivial_r_phase_game()
        self.assertIn("A RUM", game.get_state()["retreats"]["AUSTRIA"], game.get_state())

    def test_trivial_r_phase_to_from_json(self):
        game = self._get_trivial_r_phase_game()
        game = pydipcc.Game.from_json(game.to_json())  # dislodged unit preserved in json dump
        self.assertEqual(game.get_all_possible_orders()["RUM"], ["A RUM D"])

    def test_skip_a_phase(self):
        game = pydipcc.Game()
        game.process()
        game.process()
        # webdip does skip over A-phases where nobody has possible orders
        self.assertEqual(game.current_short_phase, "S1902M")


class TestCrashOnSupportToInvalidCoast(unittest.TestCase):
    def test_it_gogogo(self):
        game = pydipcc.Game()
        game.set_orders("ITALY", ["F NAP - ION"])
        game.set_orders("RUSSIA", ["F SEV - BLA"])
        game.process()
        game.set_orders("ITALY", ["F ION - GRE"])
        game.process()
        game.process()
        game.set_orders("RUSSIA", ["F BLA S F GRE - BUL"])
        game.set_orders("ITALY", ["F GRE - BUL/EC"])
        # This used to crash on older pydipcc
        game.process()
        assert "F GRE" in game.get_units()["ITALY"]


class TestFields(unittest.TestCase):
    def test_to_json_is_press(self):
        game = pydipcc.Game(is_full_press=True)
        self.assertTrue(json.loads(game.to_json())["is_full_press"])

    def test_to_json_no_press(self):
        game = pydipcc.Game(is_full_press=False)
        self.assertFalse(json.loads(game.to_json())["is_full_press"])

    def test_to_from_json(self):
        game = pydipcc.Game(is_full_press=False)
        self.assertFalse(pydipcc.Game.from_json(game.to_json()).is_full_press)

    def test_default_map_name(self):
        self.assertEqual(pydipcc.Game().map_name, "standard")

    def test_map_to_json(self):
        self.assertEqual(json.loads(pydipcc.Game().to_json())["map"], "standard")

    def test_map_from_json(self):
        j = json.loads(pydipcc.Game().to_json())
        j["map"] = "fva"
        game = pydipcc.Game.from_json(json.dumps(j))
        self.assertEqual(game.map_name, "fva")

    def test_id_from_json(self):
        j = json.loads(pydipcc.Game().to_json())
        x = j["id"]
        game = pydipcc.Game.from_json(json.dumps(j))
        self.assertEqual(game.game_id, x)

    def test_version(self):
        j = json.loads(pydipcc.Game().to_json())
        self.assertEqual(j["version"], "1.0")

    def test_read_scoring_system_int(self):
        j = json.loads(pydipcc.Game().to_json())
        assert isinstance(pydipcc.Game.SCORING_DSS, int)
        j["scoring_system"] = pydipcc.Game.SCORING_DSS
        game = pydipcc.Game.from_json(json.dumps(j))
        self.assertEqual(game.get_scoring_system(), pydipcc.Game.SCORING_DSS)

    def test_read_scoring_system_strings(self):
        for string, val in [
            ("sum_of_squares", pydipcc.Game.SCORING_SOS),
            ("draw_size", pydipcc.Game.SCORING_DSS),
        ]:
            j = json.loads(pydipcc.Game().to_json())
            j["scoring_system"] = string
            game = pydipcc.Game.from_json(json.dumps(j))
            self.assertEqual(game.get_scoring_system(), val)

    def test_read_bad_scoring_system_string(self):
        j = json.loads(pydipcc.Game().to_json())
        j["scoring_system"] = "bad_value"
        self.assertRaises(Exception, pydipcc.Game.from_json, json.dumps(j))


class TestStalemate(unittest.TestCase):
    def test_stalemate_1(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(1)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_1_with_scgain(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(1)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"AUSTRIA": ("A BUD - RUM",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1901A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1902A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_2(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(2)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 2)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_2_with_scgain(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(2)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"AUSTRIA": ("A BUD - RUM",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), True)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1901A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1902A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1903M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1903M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1903A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 2)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_3_with_later_scgain(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(3)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"AUSTRIA": ("A BUD - RUM",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), True)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1902A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1903M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1903M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1903A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1904M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1904M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1904A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"AUSTRIA": ("A BUD B",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "S1905M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 2)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1905M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 2)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 3)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_must_actually_gain_center(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(1)
        self.assertEqual(game.get_current_phase(), "S1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {
                "AUSTRIA": ("A BUD - RUM", "A VIE - BOH"),
                "ENGLAND": ("F EDI - NWG", "F LON - NTH"),
                "RUSSIA": ("F STP/SC - FIN", "A MOS - STP"),
            }
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1901M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 0)
        self.assertEqual(game.any_sc_occupied_by_new_power(), True)
        gamecopy = pydipcc.Game(game)
        # Austria moves out and russia and england bounce
        game.set_all_orders(
            {"AUSTRIA": ("A RUM - GAL",), "ENGLAND": ("F NWG - NWY",), "RUSSIA": ("A STP - NWY",),}
        )
        game.process()
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)

    def test_stalemate_with_retake(self):
        game = pydipcc.Game()
        game.set_draw_on_stalemate_years(2)
        game.process()
        game.process()
        self.assertEqual(game.get_current_phase(), "S1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"AUSTRIA": ("F TRI - ADR", "A BUD H"), "ITALY": ("A VEN - TRI",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902M")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), True)
        # Austria retakes trieste
        game.set_all_orders(
            {"AUSTRIA": ("A BUD S A VIE - TRI", "A VIE - TRI"), "ITALY": ("A TRI H",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), True)
        game.process()
        self.assertEqual(game.get_current_phase(), "F1902R")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.set_all_orders(
            {"ITALY": ("A VEN D",),}
        )
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        self.assertEqual(game.get_current_phase(), "W1902A")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 1)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)
        game.process()
        # Retaken centers don't stop a stalemate
        self.assertEqual(game.get_current_phase(), "COMPLETED")
        self.assertEqual(game.get_consecutive_years_without_sc_change(), 2)
        self.assertEqual(game.any_sc_occupied_by_new_power(), False)


class TestSomeGameGetters(unittest.TestCase):
    def test_some_game_getters(self):
        game = pydipcc.Game()
        game.set_orders("TURKEY", ["A CON - BUL"])
        game.process()
        self.assertEqual(game.get_unit_power_at("STP"), "RUSSIA")
        self.assertEqual(game.get_unit_power_at("STP/SC"), "RUSSIA")
        self.assertEqual(game.get_unit_power_at("STP/NC"), None)
        self.assertEqual(game.get_unit_power_at("STP/WC"), None)
        self.assertEqual(game.get_unit_type_at("STP"), "F")
        self.assertEqual(game.get_unit_type_at("STP/SC"), "F")
        self.assertEqual(game.get_unit_type_at("STP/NC"), None)
        self.assertEqual(game.get_unit_type_at("STP/WC"), None)
        self.assertEqual(game.get_unit_power_at("BUL"), "TURKEY")
        self.assertEqual(game.get_unit_power_at("BUL/SC"), None)
        self.assertEqual(game.get_unit_power_at("BUL/EC"), None)
        self.assertEqual(game.get_unit_type_at("BUL"), "A")
        self.assertEqual(game.get_unit_type_at("BUL/SC"), None)
        self.assertEqual(game.get_unit_type_at("BUL/EC"), None)
        self.assertEqual(game.get_unit_power_at("MAO"), None)
        self.assertEqual(game.get_unit_power_at("MAR"), "FRANCE")
        self.assertEqual(game.get_unit_type_at("MAO"), None)
        self.assertEqual(game.get_unit_type_at("MAR"), "A")

        self.assertEqual(game.get_unit_type_at("ABCDEFG"), None)

        self.assertEqual(game.is_supply_center("MAO"), False)
        self.assertEqual(game.is_supply_center("KIE"), True)
        self.assertEqual(game.is_supply_center("STP"), True)
        self.assertEqual(game.is_supply_center("STP/SC"), True)
        self.assertEqual(game.is_supply_center("STP/NC"), True)
        self.assertEqual(game.is_supply_center("STP/WC"), False)
        self.assertEqual(game.is_supply_center("BUL/EC"), True)
        self.assertEqual(game.is_supply_center("BUL/SC"), True)
        self.assertEqual(game.is_supply_center("BUL/NC"), False)

        self.assertEqual(game.get_supply_center_power("GAL"), None)
        self.assertEqual(game.get_supply_center_power("RUM"), None)
        self.assertEqual(game.get_supply_center_power("STP"), "RUSSIA")
        self.assertEqual(game.get_supply_center_power("STP/SC"), "RUSSIA")
        self.assertEqual(game.get_supply_center_power("STP/NC"), "RUSSIA")
        self.assertEqual(game.get_supply_center_power("STP/EC"), None)

        self.assertEqual(game.get_supply_center_power("BUL"), None)
        game.process()
        game.process()
        self.assertEqual(game.get_supply_center_power("BUL"), "TURKEY")


class TestCoastOrderVariants(unittest.TestCase):
    def test_coast_order_variants1(self):
        # ===========================================================================
        # Support hold a special coast via province. (DATASET FORMAT, PSEUDOS FORMAT)
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Strict encoder does NOT handle it correctly
        assert (
            "A MAR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        # Tolerant encoder DOES handle it correctly
        assert (
            "A MAR S F SPA/NC"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F SPA" in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F SPA"], match_to_possible_orders=possible_orders, sort_by_loc=True
                )[0]
            ]
            == "A MAR S F SPA/NC"
        )
        # get_valid_coastal_variant does NOT add the coast!
        assert (
            get_valid_coastal_variant("A MAR S F SPA".split(), possible_orders)
            == "A MAR S F SPA".split()
        )

        # ===========================================================================
        # Support hold a special coast via province, south coast. (DATASET FORMAT, PSEUDOS FORMAT)
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/SC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Strict encoder does NOT handle it correctly
        assert (
            "A MAR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        # Tolerant encoder DOES handle it correctly
        assert (
            "A MAR S F SPA/SC"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F SPA" in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F SPA"], match_to_possible_orders=possible_orders, sort_by_loc=True
                )[0]
            ]
            == "A MAR S F SPA/SC"
        )
        # get_valid_coastal_variant does NOT add the coast!
        assert (
            get_valid_coastal_variant("A MAR S F SPA".split(), possible_orders)
            == "A MAR S F SPA".split()
        )

        # ===========================================================================
        # Support hold a special coast via territory. (OLD DATASET FORMAT, BASESTRATEGYMODEL FORMAT)
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA/NC"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Encoder DOES handle it correctly
        assert (
            "A MAR S F SPA/NC"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        assert (
            "A MAR S F SPA/NC"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F SPA/NC" in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F SPA"], match_to_possible_orders=possible_orders, sort_by_loc=True
                )[0]
            ]
            == "A MAR S F SPA/NC"
        )
        # get_valid_coastal_variant does preserve the coast
        assert (
            get_valid_coastal_variant("A MAR S F SPA/NC".split(), possible_orders)
            == "A MAR S F SPA/NC".split()
        )

        # ===========================================================================
        # Support move FROM a special coast via province.
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA - GAS"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Strict encoder does NOT handle it correctly
        assert (
            "A MAR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        # Tolerant encoder DOES handle it correctly
        assert (
            "A MAR S F SPA/NC - GAS"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders does NOT have it:
        assert "A MAR S F SPA - GAS" not in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F SPA - GAS"],
                    match_to_possible_orders=possible_orders,
                    sort_by_loc=True,
                )[0]
            ]
            == "A MAR S F SPA/NC - GAS"
        )
        # get_valid_coastal_variant DOES add the coast
        assert (
            get_valid_coastal_variant("A MAR S F SPA - GAS".split(), possible_orders)
            == "A MAR S F SPA/NC - GAS".split()
        )

        # ===========================================================================
        # Support move FROM a special coast via province, south coast.
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.set_orders("FRANCE", ["A MAR - BUR"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/SC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A BUR S F SPA - MAR"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Strict encoder does NOT handle it correctly
        assert (
            "A BUR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        # Tolerant encoder DOES handle it correctly
        assert (
            "A BUR S F SPA/SC - MAR"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders does NOT have it:
        assert "A BUR S F SPA - MAR" not in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A BUR S F SPA - MAR"],
                    match_to_possible_orders=possible_orders,
                    sort_by_loc=True,
                )[0]
            ]
            == "A BUR S F SPA/SC - MAR"
        )
        # get_valid_coastal_variant DOES add the coast
        assert (
            get_valid_coastal_variant("A BUR S F SPA - MAR".split(), possible_orders)
            == "A BUR S F SPA/SC - MAR".split()
        )

        # ===========================================================================
        # Support move FROM a special coast via territory. (OLD DATASET FORMAT, BASESTRATEGYMODEL FORMAT, DATASET FORMAT, PSEUDOS FORMAT)
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["F MAO - SPA/NC"])
        game.process()
        game.process()
        game.set_orders("FRANCE", ["A MAR S F SPA/NC - GAS"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Encoder DOES handle it correctly
        assert (
            "A MAR S F SPA/NC - GAS"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        assert (
            "A MAR S F SPA/NC - GAS"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F SPA/NC - GAS" in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F SPA/NC - GAS"],
                    match_to_possible_orders=possible_orders,
                    sort_by_loc=True,
                )[0]
            ]
            == "A MAR S F SPA/NC - GAS"
        )
        # get_valid_coastal_variant does preserve the coast
        assert (
            get_valid_coastal_variant("A MAR S F SPA/NC - GAS".split(), possible_orders)
            == "A MAR S F SPA/NC - GAS".split()
        )

        # ===========================================================================
        # Support move TO a special coast via province. (OLD DATASET FORMAT, BASESTRATEGYMODEL FORMAT, DATASET FORMAT, PSEUDOS FORMAT)
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["A MAR S F MAO - SPA"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Encoder DOES handle it correctly
        assert (
            "A MAR S F MAO - SPA"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        assert (
            "A MAR S F MAO - SPA"
            == ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F MAO - SPA" in possible_orders
        # action_strs_to_global_idxs DOES convert it correctly
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F MAO - SPA"],
                    match_to_possible_orders=possible_orders,
                    sort_by_loc=True,
                )[0]
            ]
            == "A MAR S F MAO - SPA"
        )
        # get_valid_coastal_variant does preserve NOT having the coast
        assert (
            get_valid_coastal_variant("A MAR S F MAO - SPA".split(), possible_orders)
            == "A MAR S F MAO - SPA".split()
        )

        # ===========================================================================
        # Support move TO a special coast via territory, north.
        game = pydipcc.Game()
        game.set_orders("FRANCE", ["F BRE - MAO"])
        game.process()
        game.set_orders("FRANCE", ["A MAR S F MAO - SPA/NC"])
        encoder = FeatureEncoder()
        input_version = 3
        possible_orders = [d for ds in game.get_all_possible_orders().values() for d in ds]
        # Encoder DOES NOT handle it correctly
        assert (
            "A MAR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_strict(game.get_orders()["FRANCE"], input_version)[
                    0, 0
                ]
            ]
        )
        assert (
            "A MAR"
            not in ORDER_VOCABULARY[
                encoder.encode_orders_single_tolerant(
                    game, game.get_orders()["FRANCE"], input_version
                )[0, 0]
            ]
        )
        # Possible orders DOES have it:
        assert "A MAR S F MAO - SPA/NC" in possible_orders
        # action_strs_to_global_idxs DOES convert it, but INCORRECTLY
        assert (
            ORDER_VOCABULARY[
                action_strs_to_global_idxs(
                    ["A MAR S F MAO - SPA/NC"],
                    match_to_possible_orders=possible_orders,
                    sort_by_loc=True,
                )[0]
            ]
            == "A MAR S F MAO - SPA"
        )
        # get_valid_coastal_variant preserves it
        assert (
            get_valid_coastal_variant("A MAR S F MAO - SPA/NC".split(), possible_orders)
            == "A MAR S F MAO - SPA/NC".split()
        )
