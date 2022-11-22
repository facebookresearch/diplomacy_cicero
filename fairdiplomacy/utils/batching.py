#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Callable, List, Union
import torch

import nest

TensorNest = Union[torch.Tensor, Dict[str, "TensorNest"], List["TensorNest"]]


@torch.no_grad()
def batched_forward(
    callable: Callable,
    data_dict: TensorNest,
    batch_size: int,
    device: Union[str, torch.device] = "cuda",
) -> TensorNest:
    """Apply a function in batches to a nested dict of tensors batched over first dimension.

    data_dict is a nest of tensors, i.e., Union[Tensor,Dict[str, Tensor]]
    """
    size = len(next(nest.flatten(data_dict)))
    results = []
    for start in range(0, size, batch_size):
        batch_data = nest.map(lambda x: x[start : start + batch_size].to(device), data_dict)
        batch_results = callable(batch_data)
        batch_results = nest.map(lambda x: x.cpu(), batch_results)
        results.append(batch_results)
    return nest.map_many(lambda x: torch.cat(x, 0), *results)
