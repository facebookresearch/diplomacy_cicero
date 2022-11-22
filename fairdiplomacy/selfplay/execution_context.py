#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import dataclasses
from typing import Optional

# Exploit trainer is called by 3 sets of processes.
# This class summaries the info identifying which process is which kind.
@dataclasses.dataclass(frozen=True)
class ExecutionContext:
    # During ddp, equal to the GPU that will be used for training, currently.
    # 0 for training master (including if not using_ddp).
    # 1 to (ddp_world_size-1) for training helpers (only when using_ddp).
    # Always None for rollouters.
    training_ddp_rank: Optional[int]

    # Using DistributedDataParallel?
    using_ddp: bool
    # Always 1 if not using_ddp
    ddp_world_size: int

    @property
    def is_training_master(self) -> bool:
        return self.training_ddp_rank is not None and self.training_ddp_rank == 0

    @property
    def is_training_helper(self) -> bool:
        return self.training_ddp_rank is not None and self.training_ddp_rank != 0

    @property
    def is_rollouter(self) -> bool:
        return self.training_ddp_rank is None
