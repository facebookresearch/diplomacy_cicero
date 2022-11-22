#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from parlai.core.loader import register_teacher

from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher

"""
File for all dialogue teachers THAT load pseudo orders
"""


class BasePseudoorderDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Streaming data base dialogue teacher for messages/orders.

    Loads predicted pseudo orders

    Label is next message
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = "Base Dialogue Chunk with pseudo orders"

    def requires_pseudo_orders(self):
        # Override: Always true for anything that inherits from this agent
        return True


@register_teacher("message_history_pseudoorder_dialogue_chunk")
class MessageHistoryPseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_shortstate_pseudoorder_dialogue_chunk")
class MessageHistoryShortstatePseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_lastorder_pseudoorder_dialogue_chunk")
class MessageHistoryLastorderPseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then LAST PHASE ORDER then PSEUDO_ORDER information
    Label is the order given by the player
    """

    pass


@register_teacher("message_history_lastmovementorder_pseudoorder_dialogue_chunk")
class MessageHistoryLastmovementorderPseudoorderDialogueChunkTeacher(
    BasePseudoorderDialogueChunkTeacher
):
    """
    Text field (input) contains MESSAGE_HISTORY then LAST MOVEMENT PHASE ORDERS then PSEUDO_ORDER information
    Label is the order given by the player
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_pseudoorder_dialogue_chunk")
class MessageHistoryOrderhistorysincelastmovementphasePseudoorderDialogueChunkTeacher(
    BasePseudoorderDialogueChunkTeacher
):
    """
    Text field (input) contains MESSAGE_HISTORY then LAST MOVEMENT PHASE ORDERS then PSEUDO_ORDER information
    Label is the order given by the player
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_dialogue_chunk"
)
class MessageHistoryOrderhistorysincelastmovementphaseShortstatePseudoorderDialogueChunkTeacher(
    BasePseudoorderDialogueChunkTeacher
):
    """
    Text field (input) contains MESSAGE_HISTORY then LAST MOVEMENT PHASE ORDERS then PSEUDO_ORDER information
    Label is the order given by the player
    """

    pass
