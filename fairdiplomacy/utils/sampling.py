#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from typing import Dict, Optional, Tuple, TypeVar, Union, List

from fairdiplomacy.utils.zipn import unzip2

X = TypeVar("X")
K = TypeVar("K")
V = TypeVar("V")


def normalize_p_dict(distribution: Dict[X, float]) -> Dict[X, float]:
    """
    Normalize distribution to sum to 1
    """
    sump = sum(distribution.values())
    distribution = {power: (p / sump) for power, p in distribution.items()}

    return distribution


def sample_p_dict(d: Dict[X, float], *, rng: Optional[np.random.RandomState] = None) -> X:
    if rng is None:
        rng = np.random

    xs = list(d.keys())
    ps = [float(p) for p in d.values()]
    # do some exact normalization in float space, rng.choice is super picky about summing to 1
    if abs(sum(ps) - 1) > 0.001:
        raise ValueError(f"Values sum to {sum(ps)}")
    psum = sum(ps)
    ps = [p / psum for p in ps]

    idx = rng.choice(range(len(ps)), p=ps)
    return xs[idx]


def sample_p_list(probs: Union[List[int], List[float], Tuple[float, ...], torch.Tensor]) -> int:
    # Manually iterating is faster than np.random.choice, it turns out - np.random.choice
    # is surprisingly slow (possibly due to conversion to numpy array).
    # It also saves us from having to manually normalize the array if the policy
    # probabilities don't add up exactly to 1.0 due to float wonkiness.
    lenprobs = len(probs)
    sumprobs = sum(probs)
    r = sumprobs * np.random.random()  # type:ignore
    idx = 0
    while idx < lenprobs - 1:
        r -= probs[idx]
        if r < 0:
            return idx
        idx += 1
    return idx


def sample_p_joint_list(policy: List[Tuple[X, float]]) -> X:
    _, probs = unzip2(policy)
    return policy[sample_p_list(probs)][0]


def argmax_p_dict(d: Dict[K, float]) -> K:
    return max(d.items(), key=lambda e: e[1])[0]


def argmin_p_dict(d: Dict[K, float]) -> K:
    return min(d.items(), key=lambda e: e[1])[0]
