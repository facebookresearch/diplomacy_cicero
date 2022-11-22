#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.tasks.dialogue.pseudo_order_agents import BasePseudoorderDialogueChunkTeacher


"""
File for all recipient classifier teachers.

These are for training a model to predict the recipient of an agent's next
message provided the dialogue
"""


class BaseRecipientClassifierChunkTeacher(BaseDialogueChunkTeacher):
    """
    Override to only return the message recipient
    """

    def _check_incompatible_opt(self, opt) -> None:
        super()._check_incompatible_opt(opt)
        if opt.get("output_draw_messages") and "pseudoorder" in opt["task"]:
            raise RuntimeError(
                "Draw messages are not currently compatible with pseudo orders. Must have `--output-draw-messages False`"
            )

    def create_message(self, queue_output, entry_idx=0) -> "Message":
        """
        Given the tuple output of the queue, return an act.
        """
        # labels should be like `England -> Germany: message` or `GERMANY: message` for v2
        # we take this string and modify it to only return Germany
        label = queue_output["labels"][0].split(":")[0].split("-> ")[-1]

        possible_labels = (
            ["England", "France", "Italy", "Germany", "Austria", "Turkey", "Russia"]
            if self.opt["task_version"] <= 1
            else ["ENGLAND", "FRANCE", "ITALY", "GERMANY", "AUSTRIA", "TURKEY", "RUSSIA"]
        )
        if self.opt["task_version"] >= 2 and self.opt.get("output_draw_messages", False):
            # Must include messages to ALL
            possible_labels.append("ALL")

        assert label in possible_labels

        queue_output["labels"] = [label]

        return Message(queue_output)


class BaseWithPseudoOrderRecipientClassifierChunkTeacher(
    BaseRecipientClassifierChunkTeacher, BasePseudoorderDialogueChunkTeacher
):
    """
    Extends base recipient classifier to also work with pseudo orders as input
    """

    pass


@register_teacher("message_history_recipientclassifier_chunk")
class MessageHistoryRecipientClassifierChunkTeacher(BaseRecipientClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY information
    - [Message History] -> [Message]
    Label is the next recipient
    """

    pass


@register_teacher("message_history_state_recipientclassifier_chunk")
class MessageHistoryStateRecipientClassifierChunkTeacher(BaseRecipientClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is the next recipient
    """

    pass


@register_teacher("message_history_shortstate_recipientclassifier_chunk")
class MessageHistoryShortStateRecipientClassifierChunkTeacher(BaseRecipientClassifierChunkTeacher):
    """
    Text field (input) contains MESSAGE HISTORY and STATE information
    - [Message History, State] -> [Message]
    Label is the next recipient
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_recipientclassifier_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortstateRecipientClassifierChunkTeacher(
    BaseRecipientClassifierChunkTeacher
):
    """
    Text field (input) contains MESSAGE HISTORY and ORDER HISTORY SINCE LAST MOVEMENT and STATE information
    - [State, Message History] -> [Message]
    Label is the next recipient
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_recipientclassifier_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortstatePseudoorderRecipientClassifierChunkTeacher(
    BaseWithPseudoOrderRecipientClassifierChunkTeacher
):
    """
    Text field (input) contains MESSAGE HISTORY and ORDER HISTORY SINCE LAST MOVEMENT and STATE and PSEUDO_ORDER information
    Label is next recipient
    """

    pass
