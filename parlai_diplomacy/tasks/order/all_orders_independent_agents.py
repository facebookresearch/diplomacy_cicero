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
Tasks for predict all orders from the perspect of a single power, with many
predictions instead of a single prediction.
"""


class _BaseAllOrderIndependentTeacher(OrderPredMetricMixin, BaseDiplomacyOrderTeacher):
    """
    Base teacher for predicting all orders from the perspective of a single power
    as separate predictions

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher("state_allorderindependent_chunk")
class StateAllOrderIndependentChunkTeacher(_BaseAllOrderIndependentTeacher):
    """
    Text field (input) STATE information

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher("message_history_state_allorderindependent_chunk")
class MessageHistoryStateAllOrderIndependentChunkTeacher(_BaseAllOrderIndependentTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher("message_history_shortstate_allorderindependent_chunk")
class MessageHistoryShortStateAllOrderIndependentChunkTeacher(_BaseAllOrderIndependentTeacher):
    """
    Text field (input) contains MESSAGE then SHORTSTATE information

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependent_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateAllOrderIndependentChunkTeacher(
    _BaseAllOrderIndependentTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_allorderindependent_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseAllOrderIndependentChunkTeacher(
    _BaseAllOrderIndependentTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT information

    Label is order for a power from the perspective of some power.
    """

    pass


@register_teacher("orderhistorysincelastmovementphase_shortstate_allorderindependent_chunk")
class OrderHistorySinceLastMovementPhaseShortStateAllOrderIndependentChunkTeacher(
    _BaseAllOrderIndependentTeacher
):
    """
    Text field (input) contains ORDER HISTORY SINCE LAST MOVEMENT then STATE information

    Label is order for a power from the perspective of some power.
    """

    pass
