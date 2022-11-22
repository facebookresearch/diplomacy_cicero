#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai.core.loader import register_teacher

from parlai_diplomacy.metrics.order_predictions import OrderPredMetricMixin
from parlai_diplomacy.tasks.order.base_order_agent import BaseDiplomacyOrderTeacher


"""
File streaming messages and board state data to predict orders.
"""


class _BaseOrderTeacher(OrderPredMetricMixin, BaseDiplomacyOrderTeacher):
    """
    Base teacher: label is the order for the given player
    """

    pass


@register_teacher("state_order_chunk")
class StateOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    pass


@register_teacher("shortstate_order_chunk")
class ShortstateOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    pass


@register_teacher("speaker_token_order_chunk")
class SpeakerTokenOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains player information only

    Label is the order given by the player
    """

    pass


@register_teacher("dummy_token_order_chunk")
class DummyTokenOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains only UNK.

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_state_order_chunk")
class MessageHistoryStateOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains MESSAGE HISTORY then STATE information
    """

    pass


@register_teacher("message_history_shortstate_order_chunk")
class MessageHistoryShortStateOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains MESSAGE then STATE information
    """

    pass


@register_teacher("message_history_order_chunk")
class MessageHistoryOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains MESSAGE information only
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_shortstate_order_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateOrderChunkTeacher(
    _BaseOrderTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_order_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseOrderChunkTeacher(_BaseOrderTeacher):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT information
    """

    pass
