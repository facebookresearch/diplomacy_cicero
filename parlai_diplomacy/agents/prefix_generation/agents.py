#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Override Torch Generator Agent to allow decoding with a prefix
"""

import torch
import torch.nn.functional as F
from torch import LongTensor
from typing import Optional, List

from parlai.agents.bart.bart import BartAgent
from parlai.core.loader import register_agent
from parlai.core.message import Message
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import neginf
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


def add_cmdline_args(parser: ParlaiParser, partial_opt: Optional[Opt] = None) -> ParlaiParser:
    """
    Add CLI args.
    """
    agent = parser.add_argument_group("Prefix generation")
    # Model files and GPUs
    agent.add_argument(
        "--prefix-key",
        type=str,
        default="pseudo_orders_prefix",
        help="Key corresponding to the prefix contained in an act to condition generation",
    )


class _PrefixMixin(TorchGeneratorAgent):
    def get_prefix_tokens(self, batch: Batch) -> Optional[List[LongTensor]]:
        """
        Override this function in Torch Generator Agents, which returns
        prefix tokens to seed decoding.

        Returned is a bsz long list of prefix tensors.
        """
        return batch["prefix_vecs"]

    def vectorize(
        self, *args, **kwargs,
    ):
        obs = super().vectorize(*args, **kwargs)
        prefix_key = self.opt["prefix_key"]
        if prefix_key in obs and obs[prefix_key] is not None:
            obs["prefix_vec"] = torch.LongTensor(self.dict.txt2vec(obs[prefix_key]))
        return obs

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        assert not sort

        batch = super().batchify(obs_batch, sort)
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        _, exs = zip(*valid_obs)
        prefix_vecs = [ex.get("prefix_vec", self.EMPTY) for ex in exs]  # type: ignore
        batch["prefix_vecs"] = prefix_vecs

        return batch

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score, token_metadata) tuples for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # type: ignore
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details) for _ in range(bsz)
            ]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None:
                ##############################################################
                #  NOTE: This is the only code that is different from the code in
                #  parlai/core/torch_generator_agent.py
                #  We make adjustments here to allow for using different prefixes for
                #  every element in the batch.
                ##############################################################
                use_mask = False
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                for i, _ in enumerate(beams):
                    prefix_tokens_i = prefix_tokens[i]
                    if _ts < prefix_tokens_i.size(0):
                        use_mask = True
                        prefix_tok = prefix_tokens_i[_ts]
                        prefix_mask[
                            i, :, prefix_tok
                        ] = False  # everything except prefix toks should be neginf
                    else:
                        # This prefix has finished, nothing should be set to neginf
                        prefix_mask[i, :, :] = False
                if use_mask:  # Only bother masking if one beam hasn't finished
                    score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)]
            )
            incr_state = model.reorder_decoder_incremental_state(incr_state, incr_state_inds)
            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(decoder_input, selection, incr_state_inds)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, "_rerank_beams"):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams


@register_agent("bart_prefix")
class BartCustomInferenceAgent(_PrefixMixin, BartAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add commandline args to this agent
        """
        parser = BartAgent.add_cmdline_args(parser, partial_opt)
        parser = add_cmdline_args(parser, partial_opt)
        return parser
