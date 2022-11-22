#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Typing for game to sequence formatting
"""
from enum import Enum
from typing import Any, Dict, List
from fairdiplomacy.typedefs import JointAction, Phase, Power, Action, MessageDict


PhaseMsgs = List[MessageDict]
MsgHistoryList = List[PhaseMsgs]
OrderHistoryDict = Dict[Phase, JointAction]
Metadata = Dict[str, Any]

FlatState = str
FlatOrders = str

OrderPredictionOutput = Dict[Phase, Dict[Power, Any]]
AllOrderIndependentPredictionOutput = Dict[Phase, Dict[Power, List[Dict[str, Any]]]]
DialoguePredictionOutput = Dict[Phase, Dict[str, str]]
TrainingDialoguePredictionOutput = Dict[Phase, Dict[Power, List[Dict[str, str]]]]

# Train rollout pseudo orders are of the following format:
#         {
#             "self": <self actions>
#             "partner": <partner actions>
#             "rollout_self": <rollout self>
#             "rollout_partner": <rollout partner>
#         }
TrainRolloutPseudoOrderDict = Dict[str, str]

ParlAIAct = Dict[str, Any]


class DiplomacySequencePart(Enum):
    pass


# Valid items for order format
class OrderSequencePart(DiplomacySequencePart):
    STATE = 1
    MESSAGE_HISTORY = 2
    ORDER_HISTORY_SINCE_LAST_MOVEMENT = 3
    DUMMY_TOKEN = 4


# Valid items for dialogue format.
class DialogueSequencePart(DiplomacySequencePart):
    HISTORY = 2
    STATE = 3
    PSEUDO_ORDERS = 4
    LAST_MOVEMENT_ORDER = 6
    LAST_PHASE_ORDER = 7
    MESSAGE_HISTORY_WITH_CURRENT_OUTPUT = 8
    ORDER_HISTORY_SINCE_LAST_MOVE = 9
    ELAPSED_TIME = 10
    ACTUAL_ORDERS = 11
