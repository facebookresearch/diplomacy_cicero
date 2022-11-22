#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
from datetime import datetime, timezone
from typing import Union


class Timestamp(int):
    """Used to represent message times.

    Timestamps are always stored internally as centiseconds. Methods are
    provided to convert to/from centiseconds and seconds.

    Timestamp subclasses int, so __repr__ and json serialization comes for
    free.
    """

    __slots__ = ()

    @classmethod
    def from_centis(cls, x: int):
        return Timestamp(x)

    @classmethod
    def from_seconds(cls, x: Union[int, float]):
        return Timestamp(int(x * 100))

    @classmethod
    def now(cls):
        # returns system time since 1970-01-01 UTC, which are the timestamps
        # used by webdip
        return cls.from_seconds(datetime.now(timezone.utc).timestamp())

    def to_centis(self) -> int:
        return int(self)

    def to_seconds_float(self) -> float:
        return float(self / 100)

    def to_seconds_int(self) -> int:
        return self // 100

    def __add__(self, other: "Timestamp") -> "Timestamp":
        return Timestamp(self.to_centis() + other.to_centis())

    def __sub__(self, other: "Timestamp") -> "Timestamp":
        return Timestamp(self.to_centis() - other.to_centis())
