#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Basic example which iterates through the tasks specified and evaluates the given model
on them.

For more documentation, see parlai.scripts.eval_model.
"""
from parlai.scripts.eval_model import EvalModel
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

if __name__ == "__main__":
    EvalModel.main()
