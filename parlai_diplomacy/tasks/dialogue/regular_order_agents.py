#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai.core.loader import register_teacher
from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher


"""
File for all dialogue teachers that don't load pseudo orders
"""


@register_teacher("message_history_dialogue_chunk")
class MessageHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY information
    - [Message History] -> [Message]
    Label is the next dialogue
    """

    pass


@register_teacher("message_history_state_dialogue_chunk")
class MessageHistoryStateDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is the next dialogue
    """

    pass


@register_teacher("message_history_shortstate_dialogue_chunk")
class MessageHistoryShortStateDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is the next dialogue
    """

    pass


@register_teacher("state_message_history_dialogue_chunk")
class StateMessageHistoryDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [State, Message History] -> [Message]
    Label is the next dialogue
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_shortstate_dialogue_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseShortstateDialogueChunkTeacher(
    BaseDialogueChunkTeacher
):
    """
    Text field (input) contains MESSAGE HISTORY and ORDER HISTORY SINCE LAST MOVEMENT and STATE information
    - [State, Message History] -> [Message]
    Label is the next dialogue
    """

    pass
