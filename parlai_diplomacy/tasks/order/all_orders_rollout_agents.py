#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai.core.loader import register_teacher

from parlai_diplomacy.tasks.order.all_orders_agents import _BaseAllOrderTeacher

"""
File streaming messages and board state data to predict all orders up until the next movement phase.
"""


##############################
# NO PRESS TEACHERS
##############################


@register_teacher("state_allorderrollout_chunk")
class StateAllorderRolloutChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players rolled out up until the next movement phase
    """

    pass


@register_teacher("shortstate_allorderrollout_chunk")
class ShortstateAllorderRolloutChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players rolled out up until the next movement phase
    """

    pass


##############################
# PRESS TEACHERS
##############################


@register_teacher("message_history_state_allorderrollout_chunk")
class MessageHistoryStateAllorderRolloutChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE then STATE information
    """

    pass


@register_teacher("message_history_shortstate_allorderrollout_chunk")
class MessageHistoryShortStateAllorderRolloutChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE then SHORTSTATE information
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_allorderrollout_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateAllorderRolloutChunkTeacher(
    _BaseAllOrderTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_allorderrollout_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseAllorderRolloutChunkTeacher(
    _BaseAllOrderTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT information
    """

    pass


@register_teacher("orderhistorysincelastmovementphase_shortstate_allorderrollout_chunk")
class OrderHistorySinceLastMovementPhaseShortStateAllorderRolloutChunkTeacher(
    _BaseAllOrderTeacher
):
    """
    Text field (input) contains ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_prefix_shortstate_allorderrollout_chunk")
class MessageHistoryPrefixStateOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE prefixes then STATE information
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateAllorderIndependentRolloutChunkTeacher(
    _BaseAllOrderTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass
