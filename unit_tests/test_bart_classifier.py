#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import unittest
import logging

import torch
from parlai.core.torch_agent import DictionaryAgent

from parlai_diplomacy.agents.bart_classifier.agents import BartClassifierModel

UNIT_TEST_DIR = os.path.dirname(__file__)


class TestBartClassifierModel(unittest.TestCase):
    def test_decoder_tokens(self):
        opt = {"embedding_size": 8, "ffn_size": 8, "n_layers": 2, "n_heads": 1}
        model = BartClassifierModel(opt, DictionaryAgent({}), 2)
        inputs = torch.zeros(1, 1)
        tokens = model._get_initial_forced_decoder_input(1, inputs).squeeze()

        # check that <EOS><BOS> is used as decoder seed, i.e. that BartModel's
        # _get_initial_forced_decoder_input is being called
        self.assertEqual(tokens[0], model.END_IDX)
        self.assertEqual(tokens[1], model.START_IDX)
