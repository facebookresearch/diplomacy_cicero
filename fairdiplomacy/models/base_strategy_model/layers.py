#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, LayerNorm, Dropout, Linear, MultiheadAttention
from torch.nn.modules.transformer import _get_activation_fn


class L2RTransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        extra_normalization: bool = True,
    ):
        super(L2RTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.extra_normalization = extra_normalization
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(L2RTransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self, tgt: Tensor, memory: Tensor, *, partial_tgt: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if partial_tgt is None:
            all_inputs = tgt
        else:
            all_inputs = torch.cat([partial_tgt, tgt], dim=0)
        partial_size = 0 if partial_tgt is None else len(partial_tgt)
        tgt_mask = generate_square_subsequent_mask(len(tgt) + partial_size, tgt.device)[
            partial_size:
        ]

        # Rename to avoid confusion.
        residual = tgt
        del tgt

        if self.extra_normalization:
            tgt2 = self.norm1(residual)
            tgt2 = self.self_attn(tgt2, all_inputs, all_inputs, attn_mask=tgt_mask)[0]
            tgt2 = self.dropout1(tgt2)
            residual = residual + tgt2

            tgt2 = self.norm2(residual)
            tgt2 = self.multihead_attn(tgt2, memory, memory)[0]
            tgt2 = self.dropout2(tgt2)
            residual = residual + tgt2

            tgt2 = self.norm3(residual)
            tgt2 = self.dropout(self.activation(self.linear1(tgt2)))
            tgt2 = self.dropout(self.linear2(tgt2))
            residual = residual + tgt2
        else:
            tgt2 = self.self_attn(residual, all_inputs, all_inputs, attn_mask=tgt_mask)[0]
            residual = residual + self.dropout1(tgt2)
            residual = self.norm1(residual)
            tgt2 = self.multihead_attn(residual, memory, memory)[0]
            residual = residual + self.dropout2(tgt2)
            residual = self.norm2(residual)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(residual))))
            residual = residual + self.dropout3(tgt2)
            residual = self.norm3(residual)

        return residual, all_inputs


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
    # return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
