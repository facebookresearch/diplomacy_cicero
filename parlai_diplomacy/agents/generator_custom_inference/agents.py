#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Override Torch Generator Agents for various inference time purposes:
- Candidate rescoring (change behavior)
- Allow setting of prefix tokens
"""
import torch
from torch import BoolTensor, LongTensor, softmax
import torch.nn.functional as F
import torch.nn
from typing import Optional, List

from parlai.agents.bart.bart import BartAgent
from parlai.core.loader import register_agent
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import (
    TorchGeneratorAgent,
    NucleusSampling as ParlAINucleusSampling,
    _PathSelection,
    _PathSelectionTokenDetails as _ParlAIPathSelectionTokenDetails,
)
from parlai.core.params import ParlaiParser
from parlai.utils import logging
from parlai.utils.torch import neginf
from parlai.agents.bart.modules import BartModel
from parlai_diplomacy.utils.special_tokens import load_special_tokens


class _PathSelectionTokenDetails(_ParlAIPathSelectionTokenDetails, total=False):
    top_ranked_token: str
    cdf: float


class NucleusSampling(ParlAINucleusSampling):
    """
    Nucelus, aka top-p sampling (Holtzman et al., 2019).
    Samples from a truncated distribution which covers a fixed CDF proportion
    of the original distribution.
    Typical values of p are 0.3 and 0.9.
    See https://arxiv.org/abs/1904.09751 for details.
    """

    def __init__(self, p, probability_cutoff, *args, **kwargs):
        super().__init__(p, *args, **kwargs)
        self.probability_cutoff = probability_cutoff

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)

        # The subtraction here is to get the exclusive prefix sum,
        # to guarantee the first element is not masked
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= self.p
        trunc_sprobs = sprobs.detach().clone()
        trunc_sprobs[mask] = 0

        if self.probability_cutoff > 0:
            # Mask probability cutoff
            max_prob = torch.max(sprobs[0]).item()
            mask = sprobs < min(max_prob, self.probability_cutoff)
            trunc_sprobs[mask] = 0

        trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(1))
        choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        # Convert back to logspace.
        scores = trunc_sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprobs = sprobs[hyp_ids, choices].log().view(-1).cpu().numpy()
            tok_ranks = choices.view(-1).cpu().numpy()

            """Start: Custom Diplomacy logic"""
            _, greedy_tok_ids = logprobs.max(1)
            # hyp_ids has length equal to beam_size, so it will never be too large. Therefore,
            # I think it is fine to do a regular for-loop/list comprehension here, and this approach
            # seems to be much easier than doing this in PyTorch. (Unfortunately, PyTorch does
            # not seem to support combining slicing and indexing by tensors.)
            cdf_choices = [
                sprobs[hyp_id, : (choice + 1)].sum(dim=-1).item()
                for hyp_id, choice in zip(hyp_ids, choices)
            ]
            greedy_toks = [
                (
                    "__start__"
                    if tok_id[0].item() == self.bos
                    else "__end__"
                    if tok_id[0].item() == self.eos
                    else self.block_list.dict.vec2txt(tok_id)
                )
                for tok_id in greedy_tok_ids.view(-1, 1)
            ]
            """End: Custom Diplomacy logic"""

            token_details = []
            for tok_logprob, tok_rank, greedy_tok, cdf in zip(
                tok_logprobs, tok_ranks, greedy_toks, cdf_choices
            ):
                token_details.append(
                    {"token_logprob": tok_logprob, "token_rank": int(tok_rank), "top_ranked_token": greedy_tok, "cdf": cdf}  # type: ignore
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


def add_cmdline_args(parser: ParlaiParser, partial_opt: Optional[Opt] = None) -> ParlaiParser:
    """
    Add CLI args.
    """
    agent = parser.add_argument_group("Diplomacy custom inference arguments")
    agent.add_argument(
        "--topp-special",
        type="bool",
        default=False,
        help="During top-p sampling, dynamically adjust p for special tokens",
    )
    agent.add_argument(
        "--topp-special-threshold",
        type=float,
        default=0.0,
        help="If we use special top-p sampling, if the top token has probability above this threshold we do greedy sampling",
    )
    agent.add_argument(
        "--probability-cutoff",
        type=float,
        default=0.0,
        help="If >0, mask tokens that have probability below this threshold",
    )
    return agent


class NucleusSpecialSampling(ParlAINucleusSampling):
    """
    A variant of nucleus sampling which decodes greedily when the most likely token
    is a special token.
    """

    def __init__(
        self,
        special_toks: List[str],
        special_tok_ids: List[int],
        p: float,
        threshold: float,
        *args,
        **kwargs,
    ):
        """
        - special_toks: List of special token IDs
        - p: nucleus p
        """
        super().__init__(p, *args, **kwargs)
        assert len(special_toks) == len(special_tok_ids)
        self.special_tok_ids = special_tok_ids
        self.special_tok_vals = {ind: tok for ind, tok in zip(special_tok_ids, special_toks)}
        self.threshold = threshold

    def select_paths(self, logprobs, prior_scores, current_length):
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        # Most likely index
        assert sinds.size(0) == 1, "Special nucleus sampling currently only works with -bs 1"
        most_likely = sinds[0][0].item()
        most_likely_prob = round(sprobs[0][0].item(), 4)
        if most_likely in self.special_tok_ids and most_likely_prob > self.threshold:
            # Temp p is made very low, so that we select the most likely token with probability 1
            temp_p = 1e-10
            most_likely_val = self.special_tok_vals[most_likely]
            next_most_likely = self.special_tok_vals.get(sinds[0][1].item(), "NOT A SPECIAL TOK")
            next_most_likely_prob = round(sprobs[0][1].item(), 4)
            logging.warning(
                f"Special token {most_likely_val} ({most_likely_prob}) sampled with probability 1; "
                f"Next most likely token: {next_most_likely} ({next_most_likely_prob})"
            )
        else:
            # Not a special token OR probability is not high enough, use the designated p value
            temp_p = self.p

        # The subtraction here is to get the exclusive prefix sum,
        # to guarantee the first element is not masked
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= temp_p
        trunc_sprobs = sprobs.detach().clone()
        trunc_sprobs[mask] = 0
        trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(1))
        choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        # Convert back to logspace.
        scores = trunc_sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores

        token_details: Optional[List[_ParlAIPathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprobs = sprobs[hyp_ids, choices].log().view(-1).cpu().numpy()
            tok_ranks = choices.view(-1).cpu().numpy()
            token_details = []

            for tok_logprob, tok_rank in zip(tok_logprobs, tok_ranks):
                token_details.append({"token_logprob": tok_logprob, "token_rank": int(tok_rank)})

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


class _DiplomacyCustomInferenceMixin(TorchGeneratorAgent):
    def rank_eval_label_candidates(self, batch, batchsize):
        """
        Rank label_candidates during eval_step.

        Must have `--rank-candidates` set to True.
        Rather than roughly computing PPL to rank the candidates (as
        done in ParlAI), we sort by the total loss (sum of per-token losses)
        and return the per-token losses for later manipulation.
        """
        # compute roughly ppl to rank candidates
        cand_choices = []
        cand_choices_toks_scores = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        for i in range(batchsize):
            num_cands = len(batch.candidate_vecs[i])
            enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
            cands, _ = self._pad_tensor(batch.candidate_vecs[i])
            cands = cands.to(batch.text_vec.device)
            scores, _ = self.model.decode_forced(enc, cands)
            score_view = scores.reshape(num_cands * cands.size(1), -1)
            cand_losses = F.cross_entropy(score_view, cands.view(-1), reduction="none").view(
                num_cands, cands.size(1)
            )
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX).float()
            cand_token_losses = cand_losses * mask
            # TorchGeneratorAgent returns and sorts by the mean of `cand_token_losses`
            summed_cand_losses = cand_token_losses.sum(dim=1)
            _, ordering = summed_cand_losses.sort()
            cand_choices.append([batch.candidates[i][o] for o in ordering])
            # return both the token and the corresponding token losses, for later manipulation
            cand_choices_toks_scores.append(
                [
                    tuple(
                        zip(
                            [self.dict[tok] for tok in cands[o].tolist()],  # toks
                            cand_token_losses[o].tolist(),  # scores
                        )
                    )
                    for o in ordering
                ]
            )

        return cand_choices, cand_choices_toks_scores

    def set_prefix_tokens(self, prefix_tokens: Optional[LongTensor]):
        """
        Setter for the prefix tokens variable
        """
        self._prefix_tokens = prefix_tokens

    def get_prefix_tokens(self, batch: Batch) -> Optional[LongTensor]:
        """
        Override this function in Torch Generator Agents, which returns
        prefix tokens to seed decoding.

        Returned tensor should be of dimension bsz x len(prefix)
        """
        if hasattr(self, "_prefix_tokens") and self._prefix_tokens is not None:
            assert batch.batchsize == 1, "Can only do prefix tokens with batch size of 1"
            return self._prefix_tokens.unsqueeze(0)

        # prefix tokens were not set, return None
        return None

    # Copied from parlai/core/torch_generator_agent.py with minor changes to speed up prefix_tokens
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

        do_fast_prefix_decoding = (
            prefix_tokens is not None
            and prefix_tokens.shape[1] > 0
            and getattr(self, "_use_fast_prefix_decoding", False)
            and isinstance(self, BartAgent)
            # and False
        )

        if not do_fast_prefix_decoding:
            return super()._generate(batch, beam_size, max_ts, prefix_tokens)

        logging.debug(f"Doing fast prefix token generation!")
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
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

        ########################################################################
        # We precompute the logprobs for the prefix.
        # This isn't strictly necessary but it's nice if our scores are consistent
        # with the old version for comparison.
        assert prefix_tokens is not None
        prefix_input = prefix_tokens[:, :-1]
        prefix_input = model._get_initial_forced_decoder_input(bsz, prefix_input).to(dev)
        latent, _incr_state = model.decoder(prefix_input, encoder_states)
        prefix_logits = model.output(latent)[:, 1:]  # Trim the EOS token
        #################################################################################

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        Ndict = self.model.embeddings.weight.shape[0]
        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            if _ts >= prefix_tokens.size(1):
                score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
                # only need the final hidden state to make the word prediction
                score = score[:, -1:, :]
                score = model.output(score)
                # score contains softmax scores for bsz * beam_size samples
                score = score.view(bsz, beam_size, -1)
            else:
                score = torch.full(
                    (bsz, beam_size, Ndict),
                    neginf(torch.float32),
                    device=dev,
                    dtype=torch.float32,
                )
                score[:, :, prefix_tokens[:, _ts]] = prefix_logits[:, _ts, prefix_tokens[:, _ts]]

            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore

            for i, b in enumerate(beams):
                if not b.is_done():
                    # print(f"Advancing with {score.shape} {score.topk(3, dim=-1)}")
                    b.advance(score[i])

            incr_state_inds = torch.cat(
                [beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)]
            )
            if incr_state is not None:
                # this is typically done in reorder_decoder_incremental_state, but that assumes
                # that it will happen one token at a time.
                if _ts == prefix_tokens.size(1):
                    for incr_state_l in incr_state.values():
                        incr_state_l["self_attn"]["prev_mask"] = incr_state_l["self_attn"][
                            "prev_mask"
                        ][:, -1:, :]
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
        # print(f"beam_preds_scores= {beam_preds_scores}")
        return beam_preds_scores, beams

    def _treesearch_factory(self, device, verbose=True):
        """
        Override from Torch Generator Agent to add the option of Special Nucleus search
        """
        method = self.opt.get("inference", "greedy")
        topp_special = self.opt.get("topp_special", False)

        if method == "nucleus" and topp_special:
            special_toks = load_special_tokens()
            # remove END OF MESSAGE token
            if "[EO_M]" in special_toks:
                special_toks.remove("[EO_M]")
            special_tok_ids = [self.dict[tok] for tok in special_toks]
            return NucleusSpecialSampling(
                special_toks,
                special_tok_ids,
                self.opt["topp"],
                self.opt["topp_special_threshold"],
                self.opt.get("beam_size", 1),
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get("beam_length_penalty", 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            return super()._treesearch_factory(device, verbose=verbose)


@register_agent("bart_custom_inference")
class BartCustomInferenceAgent(_DiplomacyCustomInferenceMixin, BartAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Override to add init-fairseq-model arg.
        """
        parser = BartAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        parser = add_cmdline_args(parser, partial_opt)
        return parser

    def _treesearch_factory(self, device, verbose=True):
        """
        Override from Torch Generator Agent to add the option of Nucleus search
        """
        method = self.opt.get("inference", "greedy")
        beam_size = self.opt.get("beam_size", 1)

        if method == "nucleus":
            return NucleusSampling(
                self.opt["topp"],
                self.opt.get("probability_cutoff", 0.0),
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get("beam_length_penalty", 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            return super()._treesearch_factory(device, verbose=verbose)


class _DiplomacyCustomNucleusInferenceMixin(TorchGeneratorAgent):
    def rank_eval_label_candidates(self, batch, batchsize):
        """
        Rank label_candidates during eval_step.

        Must have `--rank-candidates` set to True.
        Rather than roughly computing PPL to rank the candidates (as
        done in ParlAI), we sort by the total loss (sum of per-token losses)
        and return the per-token losses for later manipulation.

        This version uses a nucleus-sampling based truncated scoring scheme.
        If a token's probability falls outside the nucleus of the token-distribution,
        it is forced to 0. That is, it will render a candidate's probability as 0 if it can
        not be sampling using nucleus sampling.
        """
        # compute roughly ppl to rank candidates
        cand_choices = []
        cand_choices_toks_scores = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        for i in range(batchsize):
            num_cands = len(batch.candidate_vecs[i])
            enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
            cands, _ = self._pad_tensor(batch.candidate_vecs[i])
            cands = cands.to(batch.text_vec.device)
            scores, _ = self.model.decode_forced(enc, cands)
            p = self.opt["topp"]  # topp defaults to 0.9 if not explicitly set.
            mask_nucleus = self._mask_nucleus(
                scores, p
            )  # returns a mask of scores that fall outside the nucleus
            mask_nucleus_view = mask_nucleus.reshape(num_cands * cands.size(1), -1)
            mask_truncate_view_cand_select = torch.gather(
                mask_nucleus_view, 1, cands.view(-1).unsqueeze(1)
            ).squeeze()

            score_view = scores.reshape(num_cands * cands.size(1), -1)
            cand_losses = F.cross_entropy(score_view, cands.view(-1), reduction="none")

            # truncate losses which correspond to being outsite the nucleus (p) portion of the distribution
            cand_losses[mask_truncate_view_cand_select] = float("inf")
            cand_losses = cand_losses.view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            cand_losses[cands == self.NULL_IDX] = 0.0  # set the loss on null_idx tokens to 0.0
            # setting to 0 is better than using mask as pointwise multiplication because we have infs in our matrix now.

            # everything follows the same as regular scoring from this point on...

            summed_cand_losses = cand_losses.sum(dim=1)
            _, ordering = summed_cand_losses.sort()
            cand_choices.append([batch.candidates[i][o] for o in ordering])
            # return both the token and the corresponding token losses, for later manipulation
            cand_choices_toks_scores.append(
                [
                    tuple(
                        zip(
                            [self.dict[tok] for tok in cands[o].tolist()],  # toks
                            cand_losses[o].tolist(),  # scores
                        )
                    )
                    for o in ordering
                ]
            )

        return cand_choices, cand_choices_toks_scores

    def _mask_nucleus(self, logits, p) -> BoolTensor:
        """
        Takes logit scores of for each vocab item at each token and a cutoff nucleus prob.
        Computes if a token's probability (in the candidate) falls outside the nucleus.
        Returns a mask tensor which indicates which vocab item is within or outside the
        nucleus of the token distribution.
        """
        assert 0.0 < p < 1.0, "p should be 0. < p < 1.0"
        probs = torch.softmax(logits, dim=-1)
        probs_sorted, sort_index = probs.sort(dim=-1, descending=True)
        cumsum_probs_sorted = probs_sorted.cumsum(-1)
        # the subtraction with probs_sorted below is to ensure that at least one value will remain.
        # this is mainly needed when p is small.
        mask = (
            cumsum_probs_sorted - probs_sorted >= p
        )  # the mask will contain true if we are going to truncate
        mask_unsorted = mask.gather(
            -1, sort_index.argsort(-1)
        )  # gather can "undo" a sorting using the sort_indexes
        return mask_unsorted


@register_agent("bart_nucleus_score")
class BartCustomNucleusInferenceAgent(_DiplomacyCustomNucleusInferenceMixin, BartAgent):
    pass
