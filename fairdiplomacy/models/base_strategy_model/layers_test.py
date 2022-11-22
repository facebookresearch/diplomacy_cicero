#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.testing

from fairdiplomacy.models.base_strategy_model.layers import L2RTransformerDecoderLayer


D_MODEL = 5


def _build_layer() -> L2RTransformerDecoderLayer:
    layer = L2RTransformerDecoderLayer(
        d_model=D_MODEL, nhead=1, dim_feedforward=D_MODEL * 2, dropout=0.1,
    )
    layer.eval()
    return layer


def test_l2r_transformer_decoder_casuality_small():
    torch.manual_seed(0)
    layer = _build_layer()
    bsz, time = 1, 2
    inputs = torch.rand((time, bsz, D_MODEL))
    memory = torch.rand((13, bsz, D_MODEL))
    corrupted_inputs = inputs.clone()
    corrupted_inputs[1] = 0.0
    print("inputs", inputs.squeeze(1))
    print("corrupted_inputs", corrupted_inputs.squeeze(1))
    with torch.no_grad():
        outputs, _ = layer(inputs, memory)
        corrupted_outputs, _ = layer(corrupted_inputs, memory)
    print("outputs", outputs.squeeze(1))
    print("corrupted_outputs", corrupted_outputs.squeeze(1))
    torch.testing.assert_allclose(outputs[0], corrupted_outputs[0], atol=1e-5, rtol=1e-5)


def test_l2r_transformer_decoder_casuality():
    torch.manual_seed(0)
    layer = _build_layer()
    bsz, time = 8, 7
    inputs = torch.rand((time, bsz, D_MODEL))
    memory = torch.rand((13, bsz, D_MODEL))
    corrupted_inputs = inputs.clone()
    corrupted_inputs[4:] = 0.0
    with torch.no_grad():
        outputs, _ = layer(inputs, memory)
        corrupted_outputs, _ = layer(corrupted_inputs, memory)
    torch.testing.assert_allclose(outputs[0], corrupted_outputs[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_allclose(outputs[:4], corrupted_outputs[:4], atol=1e-5, rtol=1e-5)


def test_l2r_transformer_decoder_incremental_decoding():
    torch.manual_seed(0)
    layer = _build_layer()
    bsz, time = 8, 7
    inputs = torch.rand((time, bsz, D_MODEL))
    memory = torch.rand((13, bsz, D_MODEL))
    with torch.no_grad():
        outputs, _ = layer(inputs, memory)

        partials = []
        state = None
        for i in range(time):
            partial, state = layer(inputs[i : i + 1], memory, partial_tgt=state)
            partials.append(partial)
    torch.testing.assert_allclose(outputs, torch.cat(partials, 0), atol=1e-5, rtol=1e-5)
