#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Basic example which allows local human keyboard input to talk to a trained model.

For documentation, see parlai.scripts.interactive.
"""
import random
from parlai.scripts.interactive import Interactive
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()


if __name__ == "__main__":
    random.seed(42)
    Interactive.main()
