#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import dataclasses
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

import conf.conf_cfgs
from fairdiplomacy.models.consts import LOGIT_MASK_VAL
from fairdiplomacy.utils.padded_embedding import PaddedEmbedding
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.order_idxs import local_order_idxs_to_global
from fairdiplomacy.models.base_strategy_model.layers import L2RTransformerDecoderLayer
from fairdiplomacy.models.base_strategy_model.util import he_init, top_p_filtering
from fairdiplomacy.models.state_space import (
    EOS_IDX,
    get_order_vocabulary,
)

TOKEN_PADDING = "EMPTY"
MAX_POSITIONAL_ENCODING_SIZE = 128


class TransfDecoder(nn.Module):
    def __init__(
        self, *, inter_emb_size, cfg: conf.conf_cfgs.TrainTask.TransformerDecoder,
    ):
        super().__init__()

        self.inner_dim = cfg.inner_dim
        assert self.inner_dim

        self.project_enc = nn.Linear(inter_emb_size * 2, self.inner_dim)
        if cfg.transformer.extra_normalization:
            self.enc_ln = nn.LayerNorm(self.inner_dim)
        else:
            self.enc_ln = None

        self._transformer_layers: List[L2RTransformerDecoderLayer] = []
        assert cfg.transformer.num_blocks is not None, "num_blocks is required"
        for _ in range(cfg.transformer.num_blocks):
            self._transformer_layers.append(
                L2RTransformerDecoderLayer(
                    d_model=self.inner_dim,
                    nhead=cfg.transformer.num_heads,
                    dim_feedforward=cfg.transformer.ff_channels,
                    dropout=cfg.transformer.dropout,
                    activation=cfg.transformer.activation,
                    extra_normalization=cfg.transformer.extra_normalization,
                )
            )
        if cfg.transformer.extra_normalization:
            self.final_ln = nn.LayerNorm(self.inner_dim)
        else:
            self.final_ln = None
        # Wrap into module list to save params, but using raw list for typing.
        self.transformer_layers = nn.ModuleList(self._transformer_layers)

        self.vocab = _build_token_order_vocab(featurize_order_old)

        # Static (context) encoding.
        self.power_emb = PaddedEmbedding(7, self.inner_dim, padding_idx=EOS_IDX)
        self.explicit_location_input = cfg.explicit_location_input

        # Order pre-processing.
        self.featurize_input = cfg.featurize_input
        self.featurize_output = cfg.featurize_output
        self.order_embedding = PaddedEmbedding(
            len(self.vocab.order_vocab), self.inner_dim, padding_idx=EOS_IDX
        )
        # We don't want to have high weight on orders if we don't see them.
        self.order_embedding.module.weight.data *= 0.01
        self.bpe_vocab = nn.EmbeddingBag(len(self.vocab.token_vocab), self.inner_dim)
        self.bpe_vocab.weight.data *= 0.1

        if self.featurize_input and self.featurize_output and not cfg.share_input_output_features:
            self.bpe_vocab_output = nn.EmbeddingBag(len(self.vocab.token_vocab), self.inner_dim)
        else:
            self.bpe_vocab_output = self.bpe_vocab

        # +1 for the bias.
        self.softmax_w = nn.Linear(self.inner_dim, self.inner_dim)
        self.softmax_b = nn.Linear(self.inner_dim, 1)

        if cfg.positional_encoding:
            self.positional_encoding = nn.Parameter(  # type:ignore
                he_init((MAX_POSITIONAL_ENCODING_SIZE, self.inner_dim))
            )
        else:
            self.positional_encoding = None

        self.order2tokens: torch.Tensor
        self.register_buffer("order2tokens", self.vocab.order2tokens)

    def forward(
        self,
        enc: torch.Tensor,
        loc_idxs: torch.Tensor,
        all_cand_idxs: torch.Tensor,
        power: torch.Tensor,
        temperature: Union[float, torch.Tensor] = 1.0,
        top_p: Union[float, torch.Tensor] = 1.0,
        teacher_force_orders: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forwards a model in either teacher-forcing or sampling modes.

        Args:
            enc shape: [B, 81, self.inner_dim]
            loc_idxs shape: [B, MAX_SEQ_LEN]
            all_cand_idxs: [B, MAX_SEQ_LEN, 469]
            power shape: [B, MAX_SEQ_LEN]
            teacher_force_orders shape: [B, MAX_SEQ_LEN] or None

        Sampling args (ignored if teacher_force_orders is not None)
            temperature: sampling temperature.
            top_p: parameter for nucleus sampling.

        Returns:
          - global_orders [B, S]
          - local_orders [B, S]
          - logits [B, S, 469]

        If teacher forcing orders are given,
            global_orders and local_orders are argmax predictions given teacher forced orders
            temperature and top_p are ignored
        If teacher forcing orders are NOT given,
            global_orders and local_orders will be sampled from the model;
            temperature and top_p are applied at per-token level.
        """

        enc = self.project_enc(enc)
        if self.enc_ln is not None:
            enc = self.enc_ln(enc)
        all_cand_idxs = all_cand_idxs.long()
        if teacher_force_orders is None:
            return self.sample(enc, loc_idxs, all_cand_idxs, power, temperature, top_p)
        else:
            return self.compute_loss(
                enc, loc_idxs, all_cand_idxs, power, teacher_force_orders.long()
            )

    def compute_loss(
        self,
        enc: torch.Tensor,
        loc_idxs: torch.Tensor,
        all_cand_idxs: torch.Tensor,
        power: torch.Tensor,
        tf_global_orders: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if (loc_idxs == EOS_IDX).all():
            eos_indices = all_cand_idxs.new_full(all_cand_idxs.shape[:2], fill_value=EOS_IDX)
            return (eos_indices, eos_indices, enc.new_zeros(*all_cand_idxs.shape))

        # Shape: [B, MAX_SEQ_LEN].
        bad_positions = (all_cand_idxs == EOS_IDX).all(-1)

        timings = TimingCtx()
        with timings("dec.prep"):
            # global_order_idxs_to_local expects tf_global_orders to always have a
            # match in all_cand_idxs. That's not the case if tf_global_orders is
            # clipped at zero, while all_cand_idxs has only -1 for this position.
            # We re-introduce -1 to avoid this issue.
            tf_global_orders[bad_positions] = EOS_IDX

        def shift_right(tensor, dim):
            shape = list(tensor.shape)
            shape[dim] = 1
            return torch.cat([tensor.new_zeros(shape), tensor[:, :-1]], dim=dim)

        static_features = self._compute_static_features(enc, all_cand_idxs, power, loc_idxs)

        with timings("dec.body"):
            # Shape: [B, MAX_SEQ_LEN, inner_dim].
            encoded_inputs = (
                shift_right(self._embed_input_orders(tf_global_orders.clamp_min(0)), dim=1,)
                + static_features
            )

            # Pre-transformer reshape: [time, batch, hidden].
            enc = enc.transpose(0, 1)
            outputs = encoded_inputs.transpose(0, 1)

            for layer in self._transformer_layers:
                # print(enc.device, outputs.device, tf_inputs.device)
                outputs, _ = layer(outputs, enc)
            if self.final_ln is not None:
                outputs = self.final_ln(outputs)
            # Post-transformer reshape: [batch, time, hidden].
            outputs = outputs.transpose(0, 1)

            # Shape: [batch, time, 469, hidden], [batch, time, 469]
            softmax_matrices_w, softmax_matrices_b = self._compute_softmax_matrices(all_cand_idxs)

            # Shape: [batch * time, 469]
            logits_flat = torch.bmm(
                softmax_matrices_w.view(-1, *softmax_matrices_w.shape[-2:]),
                outputs.reshape(-1, self.inner_dim, 1),
            ).squeeze(-1) + softmax_matrices_b.view(-1, *softmax_matrices_b.shape[2:])

            # Shape: [batch, time, 469]
            logits = logits_flat.view(softmax_matrices_w.shape[:3])
            logits[all_cand_idxs == EOS_IDX] = LOGIT_MASK_VAL

        # Ok, this is a pretty dumb thing. We compute teacher-forced greedy
        # decode. This code should not be here - it's up to the training code to
        # compute that. But this follows interface of LSTMBaseStrategyModelDecoder.
        with torch.no_grad():
            # Shape: [batch, time]
            _, all_local_order_idxs = logits.max(-1)
            all_local_order_idxs[bad_positions] = EOS_IDX
            # Shape: [batch, time]
            all_global_order_idxs = torch.stack(
                [
                    local_order_idxs_to_global(
                        all_local_order_idxs[:, step], all_cand_idxs[:, step], clamp_and_mask=True
                    )
                    for step in range(all_local_order_idxs.shape[1])
                ],
                dim=1,
            )

        return all_global_order_idxs, all_local_order_idxs, logits

    @torch.no_grad()
    def sample(
        self,
        enc: torch.Tensor,
        loc_idxs: torch.Tensor,
        all_cand_idxs: torch.Tensor,
        power: torch.Tensor,
        temperature: Union[float, torch.Tensor] = 1.0,
        top_p: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shape: [batch, time, hidden].
        static_features = self._compute_static_features(enc, all_cand_idxs, power, loc_idxs)
        # Shape: [batch, time, 469, hidden]
        softmax_matrices_w, softmax_matrices_b = self._compute_softmax_matrices(all_cand_idxs)

        # Pre-transformer reshape: [time, batch, hidden].
        enc = enc.transpose(0, 1)

        all_local_order_idxs = []
        all_global_order_idxs = []
        all_logits = []
        cached_inputs = {i: None for i in range(len(self._transformer_layers))}
        prev_step_orders: Optional[torch.Tensor] = None
        for step in range(all_cand_idxs.shape[1]):
            # Shape: [B, inner_dim].
            inputs = static_features[:, step]
            if step > 0:
                prev_step_orders = all_global_order_idxs[-1]
                assert prev_step_orders is not None
                inputs = inputs + self._embed_input_orders(prev_step_orders)

            # Adding time dimension.
            # Shape: [1, B, inner_dim].
            outputs = inputs.unsqueeze(0)
            for i, layer in enumerate(self._transformer_layers):
                outputs, cached_inputs[i] = layer(outputs, enc, partial_tgt=cached_inputs[i])
            if self.final_ln is not None:
                outputs = self.final_ln(outputs)

            # Shape: [B, 469].
            logits = (
                torch.bmm(softmax_matrices_w[:, step], outputs.squeeze(0).unsqueeze(-1)).squeeze(
                    -1
                )
                + softmax_matrices_b[:, step]
            )
            logits[all_cand_idxs[:, step] == EOS_IDX] = LOGIT_MASK_VAL
            all_logits.append(logits)

            filtered_logits = _alter_logits(logits, top_p=top_p, temperature=temperature)
            filtered_logits[all_cand_idxs[:, step] == EOS_IDX] = LOGIT_MASK_VAL

            local_order_idxs = Categorical(logits=filtered_logits).sample()
            all_local_order_idxs.append(local_order_idxs)

            global_order_idxs = local_order_idxs_to_global(
                local_order_idxs, all_cand_idxs[:, step], clamp_and_mask=False
            )
            all_global_order_idxs.append(global_order_idxs)

        # Shape: [B, MAX_SEQ_LEN].
        bad_positions = (all_cand_idxs == EOS_IDX).all(-1)
        all_global_order_idxs_joined = torch.stack(all_global_order_idxs, 1)
        all_local_order_idxs_joined = torch.stack(all_local_order_idxs, 1)
        all_logits_joined = torch.stack(all_logits, 1)
        all_global_order_idxs_joined[bad_positions] = EOS_IDX
        all_local_order_idxs_joined[bad_positions] = EOS_IDX

        return (
            all_global_order_idxs_joined,
            all_local_order_idxs_joined,
            all_logits_joined,
        )

    def _embed_input_orders(self, orders):
        embeddings = self.order_embedding(orders)
        if self.featurize_input:
            tokens = self.order2tokens[orders]
            embeddings = embeddings + (
                self.bpe_vocab(tokens.view(-1, tokens.shape[-1])).view(
                    *tokens.shape[:-1], self.inner_dim
                )
            )
        return embeddings

    def _compute_static_features(self, enc, all_cand_idxs, power, loc_idxs):
        del all_cand_idxs  # Not used.
        embedded = self.power_emb(power)
        if self.explicit_location_input:
            # Shape loc_idxs: [B, NUM_LOCS]
            # Shape enc: [B, NUM_LOCS + extra, inner_dim]
            # Shape alignments: [B, MAX_SEQ_LEN, NUM_LOCS]
            # alignments[b, i, j] == 1 -> at position i location j is important
            b, num_locs = loc_idxs.shape
            _, seq_len, _ = embedded.shape
            loc_idx_viewed = loc_idxs.view(b, 1, num_locs)
            alignments = (
                (
                    loc_idx_viewed
                    == torch.arange(seq_len, device=loc_idxs.device).view(1, seq_len, 1)
                )
                | (loc_idx_viewed == -2)
            ).to(enc.dtype)
            embedded = embedded + torch.bmm(alignments, enc[:, :num_locs])

        if self.positional_encoding is not None:
            embedded = embedded + self.positional_encoding[None, : embedded.shape[1]]

        return embedded

    def _compute_softmax_matrices(self, all_cand_idxs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: [batch, time, 469, inner_dim + 1]
        softmax_matrices = self.order_embedding(all_cand_idxs)

        if self.featurize_output:
            # Shape: [B, time, 469, num_tokens].
            all_cand_token_idxs = eos_aware_lookup(
                self.order2tokens, all_cand_idxs, self.vocab.empty_token_id
            )
            *batches_shape, num_tokens = all_cand_token_idxs.shape
            # Shape: [B, time, 469, inner_dim].
            featurized_softmaxes = self.bpe_vocab_output(
                all_cand_token_idxs.view(-1, num_tokens)
            ).view(*batches_shape, -1)
            softmax_matrices = softmax_matrices + featurized_softmaxes
        w = self.softmax_w(softmax_matrices)
        b = self.softmax_b(softmax_matrices).squeeze(-1)
        return w, b


def _alter_logits(
    logits, *, temperature: Union[float, torch.Tensor], top_p: Union[float, torch.Tensor],
) -> torch.Tensor:
    filtered_logits = logits.detach().clone()
    top_p_min = float(top_p.min().item()) if isinstance(top_p, torch.Tensor) else top_p
    if top_p_min < 0.999:
        filtered_logits.masked_fill_(top_p_filtering(filtered_logits, top_p=top_p), -1e9)
    filtered_logits /= temperature
    return filtered_logits


def eos_aware_lookup(
    embedding: torch.Tensor, indices: torch.Tensor, fill_value: Union[int, float]
) -> torch.Tensor:
    """Does embedding[indices], but maps EOS_IDX to fill_value."""
    mask_flat = (indices == EOS_IDX).view(-1)
    emb_dim = embedding.shape[-1]
    result_flat = embedding.new_full((indices.numel(), emb_dim), fill_value)
    result_flat[mask_flat] = embedding[indices.view(-1)[mask_flat]]
    return result_flat.view(*indices.shape, emb_dim)


def featurize_order_old(order: str) -> List[str]:
    split = order.split()

    # fixup "A SIL S A PRU"
    if len(split) == 5 and split[2] == "S":
        split.append("H")
    # fixup "A SMY - ROM VIA"
    if len(split) == 5 and split[-1] == "VIA":
        split.pop()

    feats = []
    u = split[3:] if len(split) >= 6 else split

    if not split[2].startswith("B"):  # lets ignore the concatenated builds, they're tricky
        feats.append(f"UNIT_{split[0]}")
        feats.append(f"ACTION_{split[2]}")
        feats.append(f"INF_UNIT_{u[0]}")
        feats.append(f"INF_SRC_{u[1]}")
        feats.append(f"INF_ACTION_{u[2]}")
        feats.append("INF_DST_%s" % (u[3] if len(u) >= 4 else u[1]))
    return feats


@dataclasses.dataclass
class OrderTokenVocab:
    order_vocab: List[str]
    token_vocab: List[str]
    token2id: Dict[str, int]
    order2tokens: torch.Tensor
    max_order_tokens: int
    empty_token_id: int


def _build_token_order_vocab(_tokenize) -> OrderTokenVocab:
    def _pad(x, max_length):
        return list(x) + [TOKEN_PADDING] * (max_length - len(x))

    order_vocab = get_order_vocabulary()
    max_order_tokens = max(len(_tokenize(order)) for order in order_vocab)
    tokenized_orders = [_pad(_tokenize(order), max_order_tokens) for order in order_vocab]
    tokens = [TOKEN_PADDING] + sorted(
        set(
            token
            for order_tokens in tokenized_orders
            for token in order_tokens
            if token != TOKEN_PADDING
        )
    )
    token2id = {token: i for i, token in enumerate(tokens)}
    order2tokens = torch.full(
        size=(len(order_vocab), max_order_tokens), fill_value=EOS_IDX, dtype=torch.long
    )
    for i, order_tokens in enumerate(tokenized_orders):
        for j, token in enumerate(order_tokens):
            order2tokens[i, j] = token2id[token]
    assert token2id[TOKEN_PADDING] == 0
    # Because of the padding
    assert not (order2tokens == EOS_IDX).any()
    return OrderTokenVocab(
        order_vocab=order_vocab,
        token_vocab=tokens,
        token2id=token2id,
        max_order_tokens=max_order_tokens,
        order2tokens=order2tokens,
        empty_token_id=0,
    )


if __name__ == "__main__":
    from collections import defaultdict
    import random

    vocab = _build_token_order_vocab(featurize_order_old)
    print("Num orders:", len(vocab.order_vocab))
    print("Num tokens:", len(vocab.token_vocab))
    print(vocab.token_vocab)
    print("Random examples:")
    size2orders = defaultdict(list)
    for order in vocab.order_vocab:
        size2orders[len(order.split())].append(order)
    for sz in sorted(size2orders):
        for _ in range(2):
            order = random.choice(size2orders[sz])
            print(
                "%40s -> %s"
                % (
                    order,
                    [
                        vocab.token_vocab[x]
                        for x in vocab.order2tokens[vocab.order_vocab.index(order)]
                    ],
                )
            )
