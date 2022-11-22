#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Base format helper
"""


class BaseFormatHelper:
    def __init__(self, version: int):
        self.version = version
        assert version in {0, 1, 2, 3}, f"Version {self.version} not supported"
