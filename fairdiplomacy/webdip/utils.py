#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional
from fairdiplomacy.typedefs import Phase


def turn_to_phase(turn: int, subphase: str) -> Optional[Phase]:
    """Convert webdip notion of turn and phase into dipcc notion of phase.

    Subphase should usually be "Diplomacy" or "Retreats" or "Builds".
    Below are remarks on other special values of subphase observed in webdip
    or on webdip data:

    "Pre-game" - presumably webdip might legitimately send this for pregame
    chat, but also there is a segment of our historical data where a lot of
    messages on other phase are erroneously labeled "Pre-game" so we have to
    be careful if changing the handling of it that the correct things will
    happen in the callers.

    "Unknown" - we *think* this only occurs in historical data. Back from a
    time when webdip didn't record the phase of messages at all within a
    season?

    "Finished" - This occurs for messages that are after a game ends.
    However, webdip's historical data features the oddity that for solo
    games, the messages sent on the last phase by the players BEFORE the game
    actually ended are also marked as "Finished".
    """
    if subphase not in {"Diplomacy", "Retreats", "Builds"}:
        return None

    year = turn // 2 + 1901
    if subphase == "Builds":
        return f"W{year}A"
    else:
        season = "S" if turn % 2 == 0 else "F"
        phasetype = "M" if subphase == "Diplomacy" else "R"
        return f"{season}{year}{phasetype}"


def phase_to_turn(phase: Phase) -> int:
    year = int(phase[1:5])
    season = phase[0]

    seasons = {"S": 0, "F": 1, "W": 1}
    return (year - 1901) * 2 + seasons[season]


def phase_to_phasetype(phase: Phase) -> str:
    phase_types = {"M": "Diplomacy", "R": "Retreats", "A": "Builds"}

    return phase_types[phase[5]]
