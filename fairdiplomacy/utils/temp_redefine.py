#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib


@contextlib.contextmanager
def temp_redefine(obj, field: str, value):
    old_value = getattr(obj, field)
    setattr(obj, field, value)
    try:
        yield None
    finally:
        setattr(obj, field, old_value)
