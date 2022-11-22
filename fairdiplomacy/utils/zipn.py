#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Non-variadic zip and unzip functions that are easier for the type checker to reason about.
"""

from typing import Iterable, Iterator, Sequence, Tuple, TypeVar, cast


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def zip2(L1: Sequence[T1], L2: Sequence[T2]) -> Iterator[Tuple[T1, T2]]:
    for e1, e2 in zip(L1, L2):
        yield cast(T1, e1), cast(T2, e2)


def unzip2(S: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    L1, L2 = zip(*S)
    return cast(Tuple[T1, ...], L1), cast(Tuple[T2, ...], L2)
