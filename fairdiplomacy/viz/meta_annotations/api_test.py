#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib
import tempfile
import unittest

from fairdiplomacy import pydipcc
from fairdiplomacy.timestamp import Timestamp
from fairdiplomacy.viz.meta_annotations import api
from fairdiplomacy.pseudo_orders import PseudoOrders


PSEUDO_ORDER = PseudoOrders(
    {
        "S1901M": {
            "ITALY": ("F NAP - ION", "A ROM - APU", "A VEN H"),
            "AUSTRIA": ("A VIE - GAL", "F TRI - ALB", "A BUD - SER"),
        }
    }
)


def tempfile_decorator(f):
    def wrapped(*args, **kwargs):
        with tempfile.NamedTemporaryFile() as file_obj:
            return f(*args, **kwargs, outpath=pathlib.Path(file_obj.name))

    return wrapped


class ApiTest(unittest.TestCase):
    @tempfile_decorator
    def test_empty(self, outpath):
        game = pydipcc.Game()
        with api.maybe_kickoff_annotations(game, outpath):
            pass
        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])
        self.assertEqual(data.misc, [])

    @tempfile_decorator
    def test_follow_the_game(self, outpath):
        game = pydipcc.Game()
        with api.maybe_kickoff_annotations(game, outpath):
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.add_message("AUSTRIA", "FRANCE", "Sup?", Timestamp.from_seconds(124))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)
        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])
        self.assertEqual(data.misc, [])

    @tempfile_decorator
    def test_pseudo_order(self, outpath):
        game = pydipcc.Game()
        with api.maybe_kickoff_annotations(game, outpath):
            api.add_pseudo_orders_next_msg(PSEUDO_ORDER)
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)
        data = api.read_annotations(outpath)
        self.assertEqual(len(data.pseudoorders), 1)
        self.assertEqual(data.pseudoorders[0][1].val, PSEUDO_ORDER.val)
        self.assertEqual(data.pseudoorders[0][0].phase, "S1901M")
        self.assertEqual(data.pseudoorders[0][0].timestamp_centis, 123 * 100)
        self.assertEqual(data.nonsenses, [])
        self.assertEqual(data.misc, [])

    @tempfile_decorator
    def test_log_misc_next_message_legacy(self, outpath):
        game = pydipcc.Game()
        tag = "policy"
        policy = {"data": "here"}
        with api.maybe_kickoff_annotations(game, outpath):
            api.add_dict_next_msg(policy, tag=tag, version=1)
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)
        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 1)
        self.assertEqual(data.misc[0][1], policy)
        self.assertEqual(data.misc[0][0].phase, "S1901M")
        self.assertEqual(data.misc[0][0].timestamp_centis, 123 * 100)

    @tempfile_decorator
    def test_log_misc_next_message(self, outpath):
        game = pydipcc.Game()
        tag = "policy"
        policy = {"data": "here"}
        with api.maybe_kickoff_annotations(game, outpath):
            api.add_dict_next_msg(policy, tag=tag)
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)
        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 1)
        self.assertEqual(data.misc[0][1], policy)
        self.assertEqual(data.misc[0][0].phase, "S1901M")
        self.assertEqual(data.misc[0][0].timestamp_centis, 123 * 100)

    @tempfile_decorator
    def test_log_misc_next_message_partial_reject(self, outpath):
        game = pydipcc.Game()
        with api.maybe_kickoff_annotations(game, outpath):
            for i in ("good", "bad"):
                api.add_dict_next_msg({f"data_{i}": i}, tag=i)
            api.after_message_generation_failed(bad_tags=["bad"])
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)

        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])
        self.assertEqual(len(data.misc), 1)
        self.assertEqual(data.misc[0][1], {"data_good": "good"})

    @tempfile_decorator
    def test_log_misc_next_message_full_reject(self, outpath):
        game = pydipcc.Game()
        with api.maybe_kickoff_annotations(game, outpath):
            for i in ("good", "bad"):
                api.add_dict_next_msg({f"data_{i}": i}, tag=i)
            api.after_message_generation_failed()
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)

        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])
        self.assertEqual(len(data.misc), 0)

    @tempfile_decorator
    def test_log_misc_this_phase(self, outpath):
        game = pydipcc.Game()
        tag = "policy"
        policy = {"data": "here"}
        with api.maybe_kickoff_annotations(game, outpath):
            game.process()
            api.after_new_phase(game)
            api.add_dict_this_phase(policy, tag=tag)
            game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
            api.after_last_message_add(game)
            game.process()
            api.after_new_phase(game)
        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 1)
        self.assertEqual(data.misc[0][1], policy)
        self.assertEqual(data.misc[0][0].phase, "F1901M")
        self.assertEqual(data.misc[0][0].timestamp_centis, None)

    @tempfile_decorator
    def test_log_misc_next_message_with_two_commits_without_start(self, outpath):
        tag = "some_tag"

        game = pydipcc.Game()
        api.start_global_annotation(game, outpath)
        api.add_dict_next_msg(dict(data="msg1"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
        api.after_last_message_add(game)
        api.commit_annotations(game)

        # No start after commit
        game = pydipcc.Game()
        api.add_dict_next_msg(dict(data="msg2"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(124))
        api.after_last_message_add(game)
        api.stop_global_annotation_and_commit(game)

        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 2)
        self.assertEqual(data.misc[0][1]["data"], "msg1")
        self.assertEqual(data.misc[1][1]["data"], "msg2")

    @tempfile_decorator
    def test_log_misc_next_message_with_requeue(self, outpath):
        tag = "some_tag"

        game = pydipcc.Game()
        api.start_global_annotation(game, outpath)
        api.add_dict_next_msg(dict(data="msg1"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
        api.after_last_message_add(game)

        # No commit and start anew.
        game = pydipcc.Game()
        api.start_global_annotation(game, outpath)
        api.add_dict_next_msg(dict(data="msg2"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(124))
        api.after_last_message_add(game)
        api.stop_global_annotation_and_commit(game)

        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 1)
        self.assertEqual(data.misc[0][1]["data"], "msg2")

    @tempfile_decorator
    def test_log_misc_next_message_with_requeue_with_commit(self, outpath):
        tag = "some_tag"

        game = pydipcc.Game()
        api.start_global_annotation(game, outpath)
        api.add_dict_next_msg(dict(data="msg1"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(123))
        api.after_last_message_add(game)
        api.stop_global_annotation_and_commit(game)

        # No commit and start anew.
        api.start_global_annotation(game, outpath)
        api.add_dict_next_msg(dict(data="msg2"), tag=tag)
        game.add_message("FRANCE", "AUSTRIA", "Hello!", Timestamp.from_seconds(124))
        api.after_last_message_add(game)
        api.stop_global_annotation_and_commit(game)

        data = api.read_annotations(outpath)
        self.assertEqual(data.pseudoorders, [])
        self.assertEqual(data.nonsenses, [])

        self.assertEqual(len(data.misc), 2)
        self.assertEqual(data.misc[0][1]["data"], "msg1")
        self.assertEqual(data.misc[1][1]["data"], "msg2")
