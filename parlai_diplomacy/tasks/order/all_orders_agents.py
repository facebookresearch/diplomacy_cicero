#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai.core.loader import register_teacher

from parlai_diplomacy.metrics.order_predictions import AllOrderPredMetricMixin
from parlai_diplomacy.tasks.order.base_order_agent import BaseDiplomacyOrderTeacher

"""
File streaming messages and board state data to predict orders.
"""


class _BaseAllOrderTeacher(AllOrderPredMetricMixin, BaseDiplomacyOrderTeacher):
    """
    Base teacher: label is all orders with the current player last
    """

    def __init__(self, opt, shared=None):
        if opt["allorders_mark_all_holds"] and not opt["filter_all_holds"]:
            raise RuntimeError("To mark all holds, must have `--filter-all-holds True`")

        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser, partial_opt):
        argparser.add_argument(
            "--allorders-mark-all-holds",
            type=bool,
            default=False,  # backwards compatibility
            help="Mark all holds orders for a unit",
        )
        return BaseDiplomacyOrderTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)


##############################
# NO PRESS TEACHERS
##############################


@register_teacher("state_allorder_chunk")
class StateAllorderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players
    """

    pass


@register_teacher("shortstate_allorder_chunk")
class ShortstateAllorderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains STATE information only

    Label is all orders given by ALL players
    """

    pass


@register_teacher("speaker_token_allorder_chunk")
class SpeakerTokenAllorderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains player information only

    Label is all orders given by ALL players
    """

    pass


@register_teacher("dummy_token_allorder_chunk")
class DummyTokenAllorderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains only UNK.

    Label is all orders given by ALL players
    """

    pass


##############################
# PRESS TEACHERS
##############################


@register_teacher("message_history_state_allorder_chunk")
class MessageHistoryStateAllOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE then STATE information
    """

    pass


@register_teacher("message_history_shortstate_allorder_chunk")
class MessageHistoryShortStateAllOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE then SHORTSTATE information
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_shortstate_allorder_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateAllOrderChunkTeacher(
    _BaseAllOrderTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_allorder_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseAllOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT information
    """

    pass


@register_teacher("orderhistorysincelastmovementphase_shortstate_allorder_chunk")
class OrderHistorySinceLastMovementPhaseShortStateAllOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_prefix_shortstate_allorder_chunk")
class MessageHistoryPrefixStateOrderChunkTeacher(_BaseAllOrderTeacher):
    """
    Text field (input) contains MESSAGE prefixes then STATE information
    """

    pass
