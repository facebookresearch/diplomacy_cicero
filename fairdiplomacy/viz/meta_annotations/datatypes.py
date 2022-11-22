#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Serialization/deserialization for different data types.

The file contains definition of different meta datatypes and how to save them.
Each type inherits from MetaType and defines a pair of functions:

  - load converts a jsonnable object to the target data type. It takes a version
    parameters and should be able to handle old data types.
  - dump converts the data from some data type to the a jsonnable object.

The derived classes may have multiple "dump" functions that correspond to older
versions of saving data. The reason to keep them is to check that dump/load
cycle works for them via unittests.

"""
import abc
from typing import Any, ClassVar, Dict, Generic, NewType, Optional, Tuple, TypeVar, Union
from fairdiplomacy.pseudo_orders import PseudoOrders

T = TypeVar("T")
JsonnableDict = Dict[str, Any]

SerializableState = NewType("SerializableState", Any)


NonsenseAnnotation = Tuple[str, str]


class MetaType(Generic[T], abc.ABC):
    VERSION: ClassVar[int]
    # All messages that start with DEFAULT_TAG will be sent to the corresponding
    # MetaType for decoding.
    DEFAULT_TAG: ClassVar[str]

    @classmethod
    @abc.abstractmethod
    def dump(cls, data: T, version: Optional[int] = None) -> SerializableState:
        """Dumps the latest representation of data"""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, state: SerializableState, version: int) -> T:
        """Converts dumped data for any version before current to the original form."""
        pass


class PseudoOrdersDataType(MetaType[PseudoOrders]):
    VERSION: ClassVar[int] = 1
    DEFAULT_TAG: ClassVar[str] = "pseudoorders"

    @classmethod
    def dump(cls, data: PseudoOrders, version: Optional[int] = None) -> SerializableState:
        assert version in (1, None), version
        return data.val

    @classmethod
    def load(cls, state: SerializableState, version: int) -> PseudoOrders:
        assert version == cls.VERSION
        # Converting List -> Tuple.
        return PseudoOrders(
            {
                phase: {power: tuple(action) for power, action in joint_action.items()}
                for phase, joint_action in state.items()
            }
        )


class NonsenseDataType(MetaType[NonsenseAnnotation]):
    VERSION: ClassVar[int] = 1
    DEFAULT_TAG: ClassVar[str] = "nonsense"

    @classmethod
    def dump(cls, data: NonsenseAnnotation, version: Optional[int] = None) -> SerializableState:
        assert version in (1, None), version
        return data

    @classmethod
    def load(cls, state: SerializableState, version: int) -> NonsenseAnnotation:
        assert version == cls.VERSION
        a, b, = state
        return (a, b)


KNOWN_TAGS = frozenset(x.DEFAULT_TAG for x in MetaType.__subclasses__())
