#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from enum import Enum
from typing import Dict, Tuple

from fairdiplomacy.typedefs import Phase, JointAction, RolloutJointAction, Power
from fairdiplomacy.game import sort_phase_key
from parlai_diplomacy.utils.game2seq.format_helpers.orders import is_movement_phase


class RolloutType(Enum):
    NONE = 1
    RA_ONLY = 2
    EXTENDED = 3


class PseudoOrders:
    def __init__(self, val: Dict[Phase, JointAction]):
        self.val = val

    @classmethod
    def from_joint_action(cls, x: JointAction, phase: Phase) -> "PseudoOrders":
        return cls({phase: x})

    @classmethod
    def from_rollout_joint_action(cls, x: RolloutJointAction) -> "PseudoOrders":
        return cls(x)

    def check_rollout(self, rollout_type: RolloutType) -> bool:
        """
        Checks if these are valid rollout or non-rollout pseudo-orders of the given type.
        """
        if rollout_type == RolloutType.NONE:
            # Not a rollout; should contain only one phase
            return len(self.val) == 1

        sorted_phases = sorted(list(self.val.keys()), key=sort_phase_key)
        curr_phase = sorted_phases[0]
        all_movement_phases = [x for x in sorted_phases if x.endswith("M")]
        if curr_phase.endswith("M"):
            if rollout_type == RolloutType.EXTENDED:
                # Should have two movement phases
                return len(all_movement_phases) == 2 or "COMPLETED" in sorted_phases
            else:
                # Should only have the current movement phase, and no other phases
                return len(self.val) == 1
        else:
            # Non-movement phase; should have one movement phase
            return len(all_movement_phases) == 1 or "COMPLETED" in sorted_phases

    def is_bilateral(self) -> bool:
        return not any(len(joint_action) > 2 for joint_action in self.val.values())

    def as_non_rollout(self) -> "PseudoOrders":
        """Convert rollout PSO to non-rollout PSO by extracting the first phase"""
        phase = min(self.val.keys(), key=sort_phase_key)
        return PseudoOrders({phase: self.val[phase]})

    def as_rollout_except_movement_action(self) -> "PseudoOrders":
        """Converts rollout type EXTENDED to RA_ONLY"""
        assert self.check_rollout(RolloutType.EXTENDED), self.val

        sorted_phases = sorted(list(self.val.keys()), key=sort_phase_key)

        new_raw_pseudo_orders = {}
        for phase in sorted_phases:
            new_raw_pseudo_orders[phase] = self.val[phase]
            if is_movement_phase(phase):
                break

        new_pseudo_orders = PseudoOrders(new_raw_pseudo_orders)
        assert new_pseudo_orders.check_rollout(RolloutType.RA_ONLY), new_pseudo_orders.val

        return new_pseudo_orders

    def as_bilateral(self, power_one: Power, power_two: Power) -> "PseudoOrders":
        """Convert joint actions to bilateral joint actions"""
        return PseudoOrders(
            {
                phase: {p: a for p, a in joint_action.items() if p == power_one or p == power_two}
                for phase, joint_action in self.val.items()
            }
        )

    def phases(self):
        return self.val.keys()

    def first_phase_and_joint_action(self) -> Tuple[Phase, JointAction]:
        phase = min(self.val.keys(), key=sort_phase_key)
        return phase, self.val[phase]

    def first_joint_action(self) -> JointAction:
        phase = min(self.val.keys(), key=sort_phase_key)
        return self.val[phase]

    def as_rollout_joint_action(self) -> RolloutJointAction:
        return self.val

    def __eq__(self, other: "PseudoOrders"):
        return self.val == other.val

    def __repr__(self):
        return f"PseudoOrders({self.val})"

    def __getitem__(self, phase: Phase) -> JointAction:
        return self.val[phase]
