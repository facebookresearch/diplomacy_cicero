#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple

from fairdiplomacy.typedefs import Phase

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def sort_phase_key(phase: Phase) -> Tuple[int, int, int]:
    if phase == "COMPLETED":
        return (10 ** 5, 0, 0)
    else:
        return (
            int(phase[1:5]) - 1900,
            {"S": 0, "F": 1, "W": 2}[phase[0]],
            {"M": 0, "R": 1, "A": 2}[phase[5]],
        )


def sort_phase_key_string(phase: Phase) -> str:
    a, b, c = sort_phase_key(phase)
    assert 0 <= a <= 10 ** 6, a
    assert 0 <= b < 10, b
    assert 0 <= c < 10, c
    return f"{a:07d}_{b}_{c}"
