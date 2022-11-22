#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Callable, Generator, List

# Doesn't help type checking, this type def is just so that the type signature
# is more intuitive/self-documenting. A NestedCollection is any nesting of
# tuples, lists, dicts. We could try specifying this as a recursive union type,
# but even type checkers that handle recursive types will often end up throwing
# type errors in user code due to complaining about user code not handling
# every possible type (dict,list,tuple, etc) because we can't express in the
# type system the invariant that the exact nesting fed in is the same as the
# nesting back out. So we don't try.
NestedCollection = Any

def map(f:Callable,n:NestedCollection) -> NestedCollection: ...
def map_many(f:Callable[[List[Any]],Any],*args: NestedCollection) -> NestedCollection: ...
def map_many2(f:Callable[[Any,Any],Any],n1: NestedCollection,n2: NestedCollection) -> NestedCollection: ...

def for_each(f:Callable,n:NestedCollection): ...
def for_each2(f:Callable,n1:NestedCollection,n2:NestedCollection): ...
def pack_as(n: NestedCollection, Iterable) -> NestedCollection: ...
def flatten(n: NestedCollection) -> Generator: ...
def front(n:NestedCollection) -> Any: ...
