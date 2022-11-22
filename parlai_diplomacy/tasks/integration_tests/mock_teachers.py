#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import json


FILE_DIR = os.path.dirname(__file__)


class MockTrainTeacher:
    def __init__(self) -> None:
        self._idx = 0
        self._acts = self._load_mock_acts()

    def _load_mock_acts(self):
        acts = []
        fle = os.path.join(FILE_DIR, "data/dialog_teacher_data.jsonl")
        with open(fle, "r") as f:
            for js_str in f:
                acts.append(json.loads(js_str))
        return acts

    def act(self):
        n = self.get_num_samples({})[0]
        ret = self._acts[self._idx]
        self._idx = (self._idx + 1) % n
        return ret

    def get_player_metadata(self, game, game_id):
        return {
            "AUSTRIA": {"rating": 5},
            "ENGLAND": {"rating": 4},
            "FRANCE": {"rating": 3},
            "GERMANY": {"rating": 2},
            "ITALY": {"rating": 1},
            "RUSSIA": {"rating": 2},
            "TURKEY": {"rating": 3},
            "game_id": 1,
            "is_training": True,
            "task_version": 1,
        }

    def get_num_samples(self, opt):
        n = len(self._acts)
        return (n, n)
