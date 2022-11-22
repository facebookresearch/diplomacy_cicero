#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import enum

from fairdiplomacy import pydipcc


def board_state_to_np(phase, input_version=1):
    """Encode the current game state as an 81 x board_state_size np array

    See section 4.1 and Figure 2 of the MILA paper for an explanation.
    """
    if type(phase) == pydipcc.PhaseData:
        return pydipcc.encode_board_state_from_phase(phase, input_version)
    else:
        return pydipcc.encode_board_state_from_json(json.dumps(phase.state), input_version)


def get_power_at_loc(state, loc):
    """Return the power with a unit at loc, or owning the supply at loc, or None"""
    # check for units
    for power, units in state["units"].items():
        if "A " + loc in units or "F " + loc in units:
            return power

    # supply owner, or None
    return get_supply_center_power(state, loc)


def get_supply_center_power(state, loc):
    """Return the owner of the supply center at loc, or None if not a supply"""
    for power, centers in state["centers"].items():
        if loc.split("/")[0] in centers:
            return power
    return None
