#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from typing import List
from parlai.core.loader import register_teacher
from parlai.utils import logging

from fairdiplomacy.typedefs import Phase
from fairdiplomacy.utils.typedefs import is_phase_name

from parlai_diplomacy.metrics.order_predictions import OrderPredMetricMixin
from parlai_diplomacy.tasks.order.base_order_agent import BaseDiplomacyOrderTeacher


"""
Teachers for predict orders for a player rolled out to the next movement phase.
"""


class _BaseOrderRolloutTeacher(OrderPredMetricMixin, BaseDiplomacyOrderTeacher):
    """
    Base teacher: label is the order for the given player
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt):
        argparser.add_argument(
            "--rollout-except-movement",
            type=bool,
            default=True,  # backwards compatibility
            help="Only rollout on builds/retreats, instead of rolling out for every phase",
        )
        argparser.add_argument(
            "--filter-lies-through-rollout",
            type=bool,
            default=False,
            help="Filter examples with lies through every phase of the rollout, instead of just the first phase",
        )
        return BaseDiplomacyOrderTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)

    def _get_phases_rolled_out(self, label: str) -> List[Phase]:
        """
        Return a list of which phases were rolled out based on the target example.

        E.g.
            S1901M
            A LVP YOR; F EDI NTH; F LON ENG
            F1901M
            A YOR BEL VIA; F ENG BRE; F NTH C A YOR BEL
        returns
            [S1901M, F1901M]
        """
        phases = []
        split_lines = label.split("\n")
        if split_lines:
            assert is_phase_name(split_lines[0]), f"Improperly formatted target sequence: {label}"
        for line in split_lines:
            if is_phase_name(line):
                phases.append(line)

        return phases

    def should_filter_for_lie_scores(self, ex) -> bool:
        """
        Return True iff example should be excluded due to lie score filtering

        Override to filter examples with lies on any phase in the rollout.
        """
        if (
            self.lie_detector_annotations is None
            or self.opt["lie_detector_filter_above_stdev"] is None
        ):
            return False

        if ex["game_id"] not in self.lie_detector_annotations:
            logging.warning(f"Game with no lie scores: {ex['game_id']}")
            return False

        # Check which phases we are rolling out to
        label_key = "labels" if "labels" in ex else "eval_labels"
        label = ex.get(label_key)[0]
        if self.opt.get("filter_lies_through_rollout", False):
            phases_to_filter = self._get_phases_rolled_out(label)
        else:
            phases_to_filter = [ex["phase_id"]]  # Filter lies in the current phase only

        assert (
            self.output_type == "orderrollout"
        ), "Currently only implemented for single phase rollout"
        # for an orders rollout model, we remove any data-point if any of the
        # phases being rolled out contain lies
        for _, phase_to_score in self.lie_detector_annotations[ex["game_id"]].items():
            for phase in phases_to_filter:
                if (
                    phase in phase_to_score
                    and phase_to_score[phase] > self.opt["lie_detector_filter_above_stdev"]
                ):
                    return True
        return False


@register_teacher("state_orderrollout_chunk")
class StateOrderRolloutChunkTeacher(_BaseOrderRolloutTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    pass


@register_teacher("shortstate_orderrollout_chunk")
class ShortstateOrderRolloutChunkTeacher(_BaseOrderRolloutTeacher):
    """
    Text field (input) contains STATE information only

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_state_orderrollout_chunk")
class MessageHistoryStateOrderRolloutChunkTeacher(_BaseOrderRolloutTeacher):
    """
    Text field (input) contains MESSAGE HISTORY then STATE information
    """

    pass


@register_teacher("message_history_shortstate_orderrollout_chunk")
class MessageHistoryShortStateOrderRolloutChunkTeacher(_BaseOrderRolloutTeacher):
    """
    Text field (input) contains MESSAGE then STATE information
    """

    pass


@register_teacher("message_history_orderrollout_chunk")
class MessageHistoryOrderRolloutChunkTeacher(_BaseOrderRolloutTeacher):
    """
    Text field (input) contains MESSAGE information only
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_orderrollout_chunk"
)
class MessageHistoryOrderHistorySinceLastMovementPhaseShortStateOrderRolloutChunkTeacher(
    _BaseOrderRolloutTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT then STATE information
    """

    pass


@register_teacher("message_history_orderhistorysincelastmovementphase_orderrollout_chunk")
class MessageHistoryOrderHistorySinceLastMovementPhaseOrderRolloutChunkTeacher(
    _BaseOrderRolloutTeacher
):
    """
    Text field (input) contains MESSAGE then then ORDER HISTORY SINCE LAST MOVEMENT information
    """

    pass
