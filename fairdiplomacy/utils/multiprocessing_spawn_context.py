#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.multiprocessing as mp

global_ctx = None


def get_multiprocessing_ctx():
    """Get an instance of torch.multiprocessing that uses the "spawn" context method.
  The instance returned is distinct from multiprocessing or torch.multiprocessing. This is to avoid clashing
  with poorly-written libraries or python dependencies that may set the start method of the global
  multiprocessing to something other than "spawn"."""
    global global_ctx
    if global_ctx is None:
        global_ctx = mp.get_context("spawn")
    return global_ctx
