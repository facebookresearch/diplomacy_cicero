#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Utilities for setting options in formatting
"""
from typing import Dict

from fairdiplomacy.pseudo_orders import RolloutType


def expects_bilateral_pseudo_orders(opt: Dict) -> bool:
    return not opt.get("all_power_pseudo_orders", True) or opt.get(
        "single_view_pseudo_orders", False
    )


def expects_recipient(opt) -> bool:
    """
    Given a model opt, return True or False indicating whether this agent expects a recipient
    """
    return (
        not opt.get("all_power_pseudo_orders", True)
        or opt.get("single_view_pseudo_orders", False)
        or opt.get("add_recipient_to_prompt", False)
    )


def expects_rollout_type(opt) -> RolloutType:
    if opt.get("rollout_pseudo_orders", False):
        if opt.get("rollout_except_movement", True):
            return RolloutType.RA_ONLY
        else:
            return RolloutType.EXTENDED
    else:
        return RolloutType.NONE
