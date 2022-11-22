#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import torch
from fairdiplomacy.utils.tensorlist import TensorList


class DataFields(dict):
    BOOL_STORAGE_FIELDS = ["x_board_state", "x_prev_state"]

    def select(self, idx):
        return DataFields({k: v[idx] for k, v in self.items()})

    @classmethod
    def cat(cls, L: list):
        if len(L) > 0:
            d = {}
            for k in L[0]:
                try:
                    d[k] = _cat([x[k] for x in L])
                except Exception as e:
                    print("Exception caught cat'ing key:", k, file=sys.stderr)
                    raise e
            return cls(d)
        else:
            return cls()

    @classmethod
    def stack(cls, L: list, dim: int = 0):
        if len(L) > 0:
            d = {}
            for k in L[0]:
                try:
                    d[k] = torch.stack([x[k] for x in L], dim)
                except Exception as e:
                    print("Exception caught stacking key:", k, file=sys.stderr)
                    raise e
            return cls(d)
        else:
            return cls()

    def repeat_batch_(self, n_repeat):
        for k, v in self.items():
            dim = v.dim()
            repeat = [n_repeat] + [1 for _ in range(dim - 1)]
            self[k] = v.repeat(repeat)

    def to_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.bool)
        return self

    def from_storage_fmt_(self):
        for f in DataFields.BOOL_STORAGE_FIELDS:
            self[f] = self[f].to(torch.float32)
        return self

    def to_half_precision(self):
        return DataFields(
            {
                k: v.to(torch.float16)
                if hasattr(v, "to") and hasattr(v, "dtype") and v.dtype == torch.float32
                else v
                for k, v in self.items()
            }
        )

    def to(self, *args, **kwargs):
        return DataFields(
            {k: v.to(*args, **kwargs) if hasattr(v, "to") else v for k, v in self.items()}
        )

    def copy(self) -> "DataFields":
        return DataFields(super().copy())


def _cat(x):
    return TensorList.cat(x) if isinstance(x[0], TensorList) else torch.cat(x)
