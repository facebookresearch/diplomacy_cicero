#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from ._nest import *


def flatten(n):
    if isinstance(n, tuple) or isinstance(n, list):
        for sn in n:
            yield from flatten(sn)
    elif isinstance(n, dict):
        for key in sorted(n.keys()):  # C++ nests are std::maps ordered by key.
            yield from flatten(n[key])
    else:
        yield n
