#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from enum import Enum
from collections import defaultdict
from typing import Dict, List
import os

from parlai.core.loader import register_teacher
from parlai.utils import logging

from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN

from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.utils.datapath_constants import (
    LATEST_DATA_DIR,
    DRAW_VOTE_TRAIN_GAMES_FLE,
    DRAW_TEST_IDS,
)


"""
File for all draw classifier teachers.

These are for training a model to predict whether or not to draw at a given turn.
"""


class DrawVoteStatus(Enum):
    DRAW = "DRAW"
    NODRAW = "NODRAW"
    UNDRAW = "UNDRAW"


class BaseDrawClassifierChunkTeacher(BaseDialogueChunkTeacher):
    """
    Classifier task for predicting for every message whether it is a
    - Draw vote
    - Undraw vote
    - Regular message

    Only trained on game IDs which contain draws
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser = BaseDialogueChunkTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)
        argparser.set_defaults(
            output_draw_messages=True, task_version=3,
        )

        return argparser

    def _set_chunk_idx_to_game_ids(self):
        """
        Override to only load game IDs with draw votes
        """

        game_fle = os.path.join(LATEST_DATA_DIR, "full_press_games", "game_{}.json")
        with open(DRAW_VOTE_TRAIN_GAMES_FLE, "r") as f:
            all_draw_train_ids = [int(x) for x in f.read().splitlines()]

        # Save 50 for validation
        draw_train_ids = all_draw_train_ids[:-50]
        draw_valid_ids = all_draw_train_ids[-50:]

        train_chunks = []
        num_train_games = len(draw_train_ids)
        sz = num_train_games // 20  # 20 chunks
        for i in range(20):
            start = i * sz
            end = (i + 1) * sz if i < (19) else num_train_games
            train_chunks.append([game_fle.format(x) for x in draw_train_ids[start:end]])

        self.chunk_idx_to_game_ids = {
            "train": {i: lst for i, lst in enumerate(train_chunks)},
            "valid": {10000: [game_fle.format(x) for x in draw_valid_ids]},
            "test": {20000: [game_fle.format(x) for x in DRAW_TEST_IDS]},
        }

    def load_from_chunk(self, chunk_idx: int) -> List[Dict]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        examples = super().load_from_chunk(chunk_idx)
        label_dist = defaultdict(int)

        for example in examples:
            label = example["labels"][0].split(": ")[-1]
            if label == DRAW_VOTE_TOKEN:
                class_label = DrawVoteStatus.DRAW.value
            elif label == UNDRAW_VOTE_TOKEN:
                class_label = DrawVoteStatus.UNDRAW.value
            else:
                class_label = DrawVoteStatus.NODRAW.value

            label_dist[class_label] += 1
            example["labels"] = [class_label]

        logging.info("Label distribution:")
        for k, v in label_dist.items():
            logging.info(f"\t{k}: {v}")

        return examples


@register_teacher("message_history_drawclassifier_chunk")
class MessageHistoryDrawClassifierChunkTeacher(BaseDrawClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY information
    - [Message History] -> [Message]
    Label is whether or not to draw
    """

    pass


@register_teacher("message_history_state_drawclassifier_chunk")
class MessageHistoryStateDrawClassifierChunkTeacher(BaseDrawClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is whether or not to draw
    """

    pass


@register_teacher("message_history_shortstate_drawclassifier_chunk")
class MessageHistoryShortStateDrawClassifierChunkTeacher(BaseDrawClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is whether or not to draw
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_drawclassifier_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortstateDrawClassifierChunkTeacher(
    BaseDrawClassifierChunkTeacher
):
    """
    Text field (input) contains MESSAGE HISTORY and ORDER HISTORY SINCE LAST MOVEMENT and STATE information
    - [State, Message History] -> [Message]
    Label is whether or not to draw
    """

    pass
