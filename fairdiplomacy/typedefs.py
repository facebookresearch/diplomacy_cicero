#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from enum import Enum
from typing import Any, Tuple, Dict, List, Union, Callable, Optional, FrozenSet
from typing_extensions import TypedDict
from fairdiplomacy.timestamp import Timestamp
from collections import namedtuple
from torch import Tensor

Power = str
Location = str
Order = str
Action = Tuple[Order, ...]
Phase = str
PlayerType = int
PlayerRating = float
GameJson = Dict[str, Any]
JointAction = Dict[Power, Action]
JointActionValues = Dict[Power, float]
FrozenJointAction = FrozenSet[Tuple[Power, Action]]
# The convention is that the action tuples are (agent_action, other_power_action)
BilateralConditionalValueTable = Dict[Tuple[Action, Action], Tensor]
ConditionalValueTable = Dict[FrozenJointAction, Tensor]
RolloutJointAction = Dict[Phase, JointAction]
RolloutAction = Dict[Phase, Action]
GameID = int

RolloutResult = Tuple[JointAction, JointActionValues]
RolloutResults = List[RolloutResult]

Policy = Dict[Action, float]
PowerPolicies = Dict[Power, Policy]
JointPolicy = List[Tuple[JointAction, float]]
PlausibleOrders = Dict[Power, List[Action]]
PlayerTypePolicies = Dict[PlayerType, PowerPolicies]

Message = str
MessageRecipients = List[Power]
Messages = List[Message]

CurrentDrawState = Dict[Power, bool]


class OutboundMessageDict(TypedDict):
    sender: Power
    recipient: Power
    message: str
    phase: Phase


class MessageDict(OutboundMessageDict):
    # all fields above, plus:
    time_sent: Timestamp


Json = Any

# See https://docs.python.org/3/library/enum.html#others.
# This enum can be directly compared against strings.
class StrEnum(str, Enum):
    pass


# Defaults added to maintain backward compatibility when unpickling context
# objects saved to disk. Can be removed if seen after 28 April 2022.
Context = namedtuple(
    "Context", ("gameID", "countryID", "api_url", "api_key"), defaults=[None, None]
)

# Power -> (sleep_time, probability)
SleepTimes = Dict[Power, Tuple[Timestamp, float]]


class MessageHeuristicResult(Enum):
    NONE = 0
    FORCE = 1
    BLOCK = 2
