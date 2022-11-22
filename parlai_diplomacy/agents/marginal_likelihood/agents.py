#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Override Torch Classifier Agents to marginalize over multiple positive labels.
"""
import logging
import torch.nn.functional as F
import torch
from torch import nn
from typing import Dict, Any

from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.loader import register_agent
from parlai.utils.io import PathManager
from parlai_diplomacy.agents.bart_classifier.agents import BartClassifierAgent


class MarginalLikelihoodCriterion(nn.Module):
    def forward(self, scores, labels):
        # Labels is a mask that is 1 for positive classes
        correct_scores = scores.masked_fill(~labels, -65504)  # Min fp16 value
        return torch.logsumexp(scores, dim=1) - torch.logsumexp(correct_scores, dim=1)


@register_agent("bart_marginal_likelihood")
class BartMarginalClassifierAgent(BartClassifierAgent):
    def build_criterion(self):
        return MarginalLikelihoodCriterion()

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        # Build a tensor containing the index of all positive labels
        labels_key = "labels" if "labels" in obs else "eval_labels"
        if labels_key not in obs or obs[labels_key] is None:
            return
        try:
            obs[labels_key + "_vec"] = torch.LongTensor(
                [self.class_dict[label] for label in obs[labels_key]]
            )
            return obs
        except Exception:
            logging.error(obs)
            raise

    def _get_label_tensor(self, batch):
        # Convert label indices to a binary mask
        labels_vec = batch.label_vec
        labels_mask = labels_vec.new_zeros(
            labels_vec.size(0), max(len(self.class_dict), labels_vec.size(1)), dtype=torch.bool
        ).scatter_(1, labels_vec, 1)
        # Zero out ahy padding from labels_vec (which will have index 0)
        labels_mask[:, 0] = 0
        return labels_mask[:, : len(self.class_dict)]

    def _update_confusion_matrix(*args, **kwargs):
        # do nothing -- default implementation does not play nicely with multilabel
        return

    def load(self, path: str) -> Dict[str, Any]:
        """
        Copied from https://github.com/facebookresearch/ParlAI/blob/d4fded078e18ace3b8f000c8a21490150173253d/parlai/core/torch_agent.py#L2065

        but catching ValueError when loading optimizer
        """
        import parlai.utils.pickle

        with PathManager.open(path, "rb") as f:
            states = torch.load(
                f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )
        if "model" in states:
            self.load_state_dict(states["model"])
        if "optimizer" in states and hasattr(self, "optimizer"):
            try:
                # this will fail to load when fine-tuning a bart model trained
                # with a different criterion, so catch the exception
                self.optimizer.load_state_dict(states["optimizer"])
            except ValueError:
                pass
        return states
