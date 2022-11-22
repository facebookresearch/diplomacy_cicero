#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
import json
from typing import Dict, Optional

from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import (
    Action,
    CurrentDrawState,
    Location,
    MessageDict,
    Order,
    Phase,
    Power,
)
from fairdiplomacy.models.consts import POWERS

from parlai_diplomacy.utils.game2seq.typing import MessageDict
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageObjectPart,
    is_draw_msg,
    is_unvote_draw_msg,
)


def game_from_view_of(game: pydipcc.Game, power: Power) -> pydipcc.Game:
    """
    Return a game object from the view of a specific power
    """
    assert power in POWERS
    if power is None:
        return pydipcc.Game(game)

    j = json.loads(game.to_json())

    for phase in j["phases"]:
        phase["messages"] = [
            m
            for m in phase["messages"]
            if m[MessageObjectPart.SENDER] == power
            or m[MessageObjectPart.RECIPIENT] == power
            or m[MessageObjectPart.RECIPIENT] == "ALL"
        ]

    new_game = pydipcc.Game.from_json(json.dumps(j))
    return new_game


def game_from_two_party_view(
    game: pydipcc.Game, power_one: Power, power_two: Power, *, add_message_to_all: bool = True
) -> pydipcc.Game:
    """
    Return a game object from the view of 2 parties.

    In other words, can only see messages that these two powers have sent each other.
    """
    assert power_one in POWERS
    assert power_two in POWERS
    assert power_one != power_two

    j = json.loads(game.to_json())

    def _valid_msg(msg: MessageDict) -> bool:
        sender = msg[MessageObjectPart.SENDER]
        recipient = msg[MessageObjectPart.RECIPIENT]
        if sender not in {power_one, power_two}:
            return False

        if add_message_to_all:
            if recipient not in {power_one, power_two, "ALL"}:
                return False
        else:
            if recipient not in {power_one, power_two}:
                return False

        return True

    for phase in j["phases"]:
        phase["messages"] = [m for m in phase["messages"] if _valid_msg(m)]

    new_game = pydipcc.Game.from_json(json.dumps(j))
    return new_game


def assert_game_from_view_of(game: pydipcc.Game, power: Power) -> None:
    """
    Check that there are no messages in the game object that were not
    sent or recieved by the specified power.
    """
    assert power in POWERS, f"{power} is not a standard format Power"

    def assert_sender_or_receiver(m: MessageDict):
        assert (
            m[MessageObjectPart.SENDER] == power
            or m[MessageObjectPart.RECIPIENT] == power
            or m[MessageObjectPart.RECIPIENT] == "ALL"
        ), "Information leak, make sure to call `game_from_view_of` before passing game object to the formatter"

    # check current phase messages
    for m in game.messages.values():
        assert_sender_or_receiver(m)

    # check message history
    for phase_msgs in game.message_history.values():
        for m in phase_msgs.values():
            assert_sender_or_receiver(m)


def year_of_phase(phase: Phase) -> int:
    return int(phase[1:-1])


def next_M_phase(phase: Phase):
    year = year_of_phase(phase)
    season = "F" if phase[0] == "S" else "S"
    next_year = year if phase[0] == "S" else year + 1
    return f"{season}{next_year}M"


def get_game_draw_state(game: pydipcc.Game) -> CurrentDrawState:
    """
    Returns dict with boolean flag corresponding to which players have currently voted for a draw
    """
    drawstate: CurrentDrawState = defaultdict(lambda: False)
    for phase in game.get_all_phases():
        for message in phase.messages.values():
            if is_draw_msg(message):
                drawstate[message[MessageObjectPart.SENDER]] = True
            elif is_unvote_draw_msg(message):
                drawstate[message[MessageObjectPart.SENDER]] = False

    return drawstate


def get_last_message_between(
    game: pydipcc.Game, a: Power, b: Power, current_phase_only: bool = False
) -> Optional[MessageDict]:
    all_messages = (
        list(game.messages.values())
        if current_phase_only
        else [m for phase in game.get_all_phases() for m in phase.messages.values()]
    )
    for m in reversed(all_messages):
        if {m["sender"], m["recipient"]} == {a, b}:
            return m
    return None


def get_last_message_from(
    game: pydipcc.Game, power: Power, current_phase_only: bool = False
) -> Optional[MessageDict]:
    all_messages = (
        list(game.messages.values())
        if current_phase_only
        else [m for phase in game.get_all_phases() for m in phase.messages.values()]
    )
    for m in reversed(all_messages):
        if m["sender"] == power:
            return m
    return None


def is_replying_to(game: pydipcc.Game, a: Power, b: Power) -> bool:
    """Return True if a is replying to b"""
    msg = get_last_message_between(game, a, b, current_phase_only=True)
    return msg is not None and msg["sender"] == b


def is_friendly_xpower_support_or_convoy(
    game: pydipcc.Game, order: Order, other_power: Power
) -> bool:
    """Returns true if this is a cross-power support or convoy from the ordering power to other_power
    except for some supports that heuristically might be hostile.
    """
    pieces = order.split()
    # Support hold
    if len(pieces) == 5 and pieces[2] == "S":
        supportee = pieces[4][:3]  # root location, remove special coasts
        return game.get_unit_power_at(supportee) == other_power
    # Convoy
    if len(pieces) == 7 and pieces[2] == "C" and pieces[5] == "-":
        supportee = pieces[4][:3]  # root location, remove special coasts
        return game.get_unit_power_at(supportee) == other_power
    # Support move
    if len(pieces) == 7 and pieces[2] == "S" and pieces[5] == "-":
        supportee = pieces[4][:3]  # root location, remove special coasts
        if game.get_unit_power_at(supportee) == other_power:
            # Most important case of a hostile support is to support a power
            # into their own empty supply center where they are self-bouncing.
            # It's also possible that we're supporting them into their
            # own empty supply center as protection against someone else,
            # but that's hard to determine by hardcoded heurstic, so we just
            # accept missing this case.
            target = pieces[6][:3]  # root location, remove special coasts
            if not (
                game.get_unit_power_at(target) is None
                and game.get_supply_center_power(target) == other_power
            ):
                return True
    return False


def action_has_any_direct_attack(game: pydipcc.Game, action: Action, other_power: Power) -> bool:
    """Returns true if action has a move to at least one province or SC held by other_power"""
    for order in action:
        pieces = order.split()
        # Movement
        if (len(pieces) == 4 or len(pieces) == 5) and pieces[2] == "-":
            target = pieces[3][:3]  # root location, remove special coasts
            if (
                game.get_unit_power_at(target) == other_power
                or game.get_supply_center_power(target) == other_power
            ):
                return True
    return False


def action_supports_move(action: Action, source: Location, target: Location) -> bool:
    """Check if this action has at least one support move for source - target"""
    for order in action:
        pieces = order.split()
        if (
            len(pieces) == 7
            and pieces[2] == "S"
            and pieces[5] == "-"
            and source[:3] == pieces[4][:3]  # root location, remove special coasts
            and target[:3] == pieces[6][:3]  # root location, remove special coasts
        ):
            return True
    return False


def number_of_moves_to(action: Action, target: Location) -> int:
    """Count how many orders there are in this action moving to target"""
    count = 0
    for order in action:
        pieces = order.split()
        if (len(pieces) == 4 or len(pieces) == 5) and pieces[2] == "-":
            if target[:3] == pieces[3][:3]:  # root location, remove special coasts
                count += 1
    return count


def action_vacates_other_power_sc_and_does_not_attack(
    game: pydipcc.Game, action: Action, other_power: Power, other_power_action: Action
):
    """Returns true if on an SC of another power and vacating it and not taking anything else."""
    if action_has_any_direct_attack(game, action, other_power):
        return False
    for order in action:
        pieces = order.split()
        # Movement
        if (len(pieces) == 4 or len(pieces) == 5) and pieces[2] == "-":
            source = pieces[1][:3]  # root location, remove special coasts
            # Off of other power SC
            if game.get_supply_center_power(source) == other_power:
                target = pieces[3][:3]  # root location, remove special coasts
                # That nominally would succeed.
                if (
                    game.get_unit_power_at(target) is None
                    or action_supports_move(action, source, target)
                    or action_supports_move(other_power_action, source, target)
                ) and number_of_moves_to(action, target) == 1:
                    # Since we are not directly attacking the other_power, we also
                    # know that we aren't simply, e.g. taking another of their SCs instead.
                    # And we also know that we aren't simply scooting pieces forward moving another
                    # piece to this SC.
                    return True
    return False
