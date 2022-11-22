#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai_diplomacy.tasks.dialogue.pseudo_order_agents import BasePseudoorderDialogueChunkTeacher
from parlai.core.loader import register_teacher

from parlai_diplomacy.metrics.order_predictions import AllOrderPredMetricMixin

"""
Predicting plausible pseudo orders
"""


@register_teacher("message_history_shortstate_plausiblepseudoorder_chunk")
@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_plausiblepseudoorder_chunk"
)
@register_teacher("message_history_orderhistorysincelastmovementphase_plausiblepseudoorder_chunk")
@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_actualorders_plausiblepseudoorder_chunk"
)
class _BasePlausiblePseudoOrderTeacher(
    AllOrderPredMetricMixin, BasePseudoorderDialogueChunkTeacher
):
    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser = BasePseudoorderDialogueChunkTeacher.add_cmdline_args(
            argparser, partial_opt=partial_opt
        )

        argparser.add_argument(
            "--speaker-first",
            type=bool,
            default=False,
            help="Predict the speaker pseudo-order first (otherwise speaker is predicted last)",
        )
        return argparser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "Base Plausible Pseudo Orders Chunk with pseudo orders"
