#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import re
from typing import Union, Optional

from fairdiplomacy.pydipcc import Game
from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN
from fairdiplomacy.typedefs import (
    JointAction,
    MessageDict,
    OutboundMessageDict,
    Phase,
    Power,
    RolloutJointAction,
    Timestamp,
)


def is_phase_name(text: str) -> bool:
    if not text:
        return False
    if text == "COMPLETED":
        return True
    return re.match(r"^[A-Z]\d{4}[A-Z]$", text) is not None


def is_rollout_joint_action(pseudo_orders: Union[JointAction, RolloutJointAction]) -> bool:
    """
    Returns True or False corresponding to whether the pseudo orders are in
    rollout format
    """
    keys = list(pseudo_orders.keys())
    if not keys:
        # An empty regular action
        return False
    rollout_pseudo_orders = is_phase_name(keys[0])
    if rollout_pseudo_orders:
        # check that all keys are phase names
        for key in pseudo_orders.keys():
            assert is_phase_name(key)

    return rollout_pseudo_orders


def get_last_message(game: Game) -> Optional[MessageDict]:
    for phase in reversed(game.get_all_phases()):
        if phase.messages:
            return list(phase.messages.values())[-1]
    return None


def increment_last_message_time(game: Game, increment: Timestamp) -> Timestamp:
    """Return a timestamp `increment` later than the last message sent in `game`"""
    last_message_sent = get_last_message(game)
    last_message_time = (
        last_message_sent["time_sent"] if last_message_sent else Timestamp.from_seconds(0)
    )
    return last_message_time + increment


def build_outbound_message_dict(
    sender: Power, recipient: Power, message: str, phase: Phase,
) -> OutboundMessageDict:
    return {
        "sender": sender,
        "recipient": recipient,
        "message": message,
        "phase": phase,
    }


def build_message_dict(
    sender: Power, recipient: Power, message: str, phase: Phase, time_sent: Timestamp,
) -> MessageDict:
    return with_time_sent(
        build_outbound_message_dict(sender, recipient, message, phase), time_sent
    )


def build_draw_vote_message_dict(
    sender: Power, phase: Phase, timestamp: Timestamp,
) -> MessageDict:
    """
    Build a draw vote message
    """
    return with_time_sent(
        {"sender": sender, "recipient": "ALL", "message": DRAW_VOTE_TOKEN, "phase": phase,},
        timestamp,
    )


def build_undraw_vote_message_dict(
    sender: Power, phase: Phase, timestamp: Timestamp,
) -> MessageDict:
    """
    Build a undraw vote message
    """
    return with_time_sent(
        {"sender": sender, "recipient": "ALL", "message": UNDRAW_VOTE_TOKEN, "phase": phase,},
        timestamp,
    )


def with_time_sent(m: OutboundMessageDict, t: Timestamp) -> MessageDict:
    """Add a timestamp to an OutboundMessageDict"""
    return {
        "sender": m["sender"],
        "recipient": m["recipient"],
        "message": m["message"],
        "phase": m["phase"],
        "time_sent": t,
    }
