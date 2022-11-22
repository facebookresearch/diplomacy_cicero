#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from parlai_diplomacy.tasks.base_diplomacy_agent import BaseDiplomacyTeacher

"""
Base Order Agent
"""


class BaseDiplomacyOrderTeacher(BaseDiplomacyTeacher):
    """
    Base teacher: label is all orders with the current player last
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt):
        argparser.add_argument(
            "--filter-all-holds",
            type=bool,
            default=False,  # backwards compatibility
            help="Filter all holds orders from movement phases when there are at least 3 units",
        )
        argparser.add_argument(
            "--train-on-message-prefixes",
            type=bool,
            default=False,
            help="Take a random prefix history of messages",
        )
        argparser.add_argument(
            "--train-two-powers-view-orders",
            type=bool,
            default=False,
            help="If set, will train on bilateral dialogue views as well, i.e., when only a dialogue for 2 powers is given. This flag will only have affect with AllOrderIndependentPredictionFormatter and AllOrderIndependentRolloutPredictionFormatter",
        )
        return BaseDiplomacyTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)
