#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Utilities for state flattening
"""
from typing import Any, Dict
from fairdiplomacy.typedefs import GameJson, Phase

from parlai_diplomacy.utils.game2seq.format_helpers.base_helper import BaseFormatHelper
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    COUNTRY_ID_TO_POWER,
    add_end_token,
)
from parlai_diplomacy.utils.game2seq.format_helpers.orders import get_last_n_movement_phases
import parlai_diplomacy.utils.misc as misc


def build_state_history_dct(game_json: GameJson, cur_phase: Phase):
    """
    Return order history dict given a game json

    get all speakers' previous states, including current one
    """
    states = {}
    for phase in game_json:
        if phase == "is_partial" or phase == "partial":
            continue

        states[phase] = game_json[phase]["state"]

        if phase == cur_phase:
            break

    return states


class StateFlattener(BaseFormatHelper):
    def flatten_state(self, state, phase, short_version=False, opt: Dict = {}):
        """
        Flatten the state in game*.json
        """

        def flatten_country_status(key, country_status):
            if self.version >= 2 and key == "builds":
                return "builds: " + " ".join(
                    [f"{p} {d['count']}" for p, d in country_status.items() if d["count"] != 0]
                )
            status_list = []
            for country, status in country_status.items():
                if (
                    self.version >= 2 and key == "retreats"
                ):  # retreats are flattened differently in V2
                    if not status:
                        # No need to show countries that can't retreat
                        continue
                    else:
                        # {'A MUN': ['RUH', 'KIE', 'BOH']} --> "A MUN - BOH / KIE / RUH"
                        str_status = ", ".join(
                            [
                                f"{unit} - {' / '.join(sorted(locs))}"
                                for unit, locs in status.items()
                            ]
                        )
                        status_list.append(f"{country}: {str_status}")
                else:
                    if type(status) is not list:
                        status = [str(status)]
                    status = sorted(status)
                    if self.version <= 1:
                        country = country.capitalize()
                    status_list.append(f'{country}: {", ".join(status)}')

            final_status = f'{key}: {"; ".join(status_list)}'
            return final_status

        # which state keys get included into the flattened sequence?
        keys = ["units"]
        if not short_version:
            keys.extend(["retreats", "centers", "homes", "influence", "civil_disorder", "builds"])
        else:
            if self.version >= 1 and phase.endswith("R"):
                keys.append("retreats")
            if opt.get("include_centers_state"):
                keys.append("centers")
            if opt.get("include_builds_state") and phase.endswith("A"):
                keys.append("builds")

        state_list = [flatten_country_status(key, state[key]) for key in keys if key in state]
        final_state_str = "\n".join(state_list)
        # maybe add end_or_state_token: [EO_STATE]
        if self.version <= 1:
            # In V2 we remove the [EO_STATE] token
            final_state_str = add_end_token(final_state_str, "[EO_STATE]")

        return final_state_str

    def flatten_state_since_last_n_movement_phases(
        self,
        num_movement_phases: int,
        state_history_dict: Dict[Phase, Any],
        short_version=False,
        opt: Dict = {},
    ):
        last_n_movement_phase = get_last_n_movement_phases(state_history_dict, num_movement_phases)
        if not last_n_movement_phase:
            return ""
        ordered_phases = misc.get_ordered_dict_keys(state_history_dict)
        last_n_movement_phase_idx = ordered_phases.index(last_n_movement_phase)
        phases_since_last_n_movement_phase = ordered_phases[last_n_movement_phase_idx:]
        state_history_since_last_n_movement_phase = {
            p: self.flatten_state(state_history_dict[p], p, short_version=short_version, opt=opt,)
            for p in phases_since_last_n_movement_phase
        }

        phase_state = []
        for phase_name in phases_since_last_n_movement_phase:
            phase_state.append(phase_name)
            phase_state.append(state_history_since_last_n_movement_phase[phase_name])

        return "\n".join(phase_state)
