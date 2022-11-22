#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
def device_id_to_str(device_id: int) -> str:
    return f"cuda:{device_id}" if device_id >= 0 else "cpu"
