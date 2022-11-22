#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from functools import reduce


def div_roundup(x, y):
    return (x + y - 1) // y


BITMASK_DTYPE = torch.int32
BITMASK_N = 31


def to_bitmask(T: torch.Tensor):
    T_flat = T.view(-1)
    if T.dtype != torch.bool:
        T = T.clamp(min=0, max=1)
    NT = T.nelement()
    NM = div_roundup(NT, BITMASK_N)
    M = torch.zeros(NM, dtype=BITMASK_DTYPE, device=T.device)
    for i, chunk in enumerate(T_flat.chunk(BITMASK_N)):
        M[: len(chunk)] += chunk * 2 ** i
    return M


def from_bitmask(M: torch.Tensor, shape: torch.Size, dtype=torch.bool):
    assert M.dtype == BITMASK_DTYPE
    NM = M.nelement()
    c = 0
    s = []
    for i in range(BITMASK_N):
        s.append((M // (2 ** i)) % 2)

    res_flat = torch.cat(s).to(dtype)
    Ne = reduce(lambda x, y: x * y, shape)
    assert res_flat.shape[0] >= Ne, f"{NM} < {Ne}"
    assert (res_flat.shape[0] - Ne) < BITMASK_N, f"{M.shape} {res_flat.shape[0]} {Ne} {BITMASK_N}"
    res_flat = res_flat[:Ne]
    return res_flat.view(shape)


if __name__ == "__main__":
    for N in (1, 7, 31, 32, 33, 63, 64, 65, 127, 200, 1000, 5000, 10000):
        T = torch.rand(N).gt(0.5)
        M = to_bitmask(T)
        # print(M)
        assert M.nelement() == div_roundup(N, BITMASK_N)
        U = from_bitmask(M, T.shape)
        assert all(T == U), f"{T.float() - U.float()}"
