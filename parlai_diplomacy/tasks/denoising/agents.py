#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Tuple, Dict

from parlai.core.loader import register_teacher
from parlai.core.params import ParlaiParser
from fairdiplomacy.typedefs import Phase, Power
from parlai_diplomacy.metrics.classifiers import ClassifierMetricMixin
from parlai_diplomacy.tasks.base_diplomacy_agent import BaseDiplomacyTeacher
from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.tasks.discriminator.change_entity_utils import (
    CARDINALS,
    DIPLOMACY_NOUNS,
    DIPLOMACY_PLURAL_NOUNS,
    NEGATIONY_WORDS,
    PLURAL_CARDINALS,
)
from parlai_diplomacy.tasks.discriminator.noisy_locations import NOISY_LOCATIONS
from parlai_diplomacy.utils.game2seq.format_helpers.misc import modify_input_prompt_for_power
import math
import random

SUPPORTED_MASKING_TARGETS = {
    "noisy_locations": NOISY_LOCATIONS,
    "cardinals": CARDINALS,
    "plural_cardinals": PLURAL_CARDINALS,
    "negations": NEGATIONY_WORDS,
}


def _mask_untargeted_tokens(
    tokens: List[str],
    noise_ratio: float,
    max_mask: int,
    mask_token: str,
    maybe_mask_multiple_tokens: Optional[bool] = True,
) -> List[str]:
    # Don't edit tokens before this index. Setting to 1 means that the recipient won't be masked.
    # assumes task-version = 3
    prefix_length = 1

    num_to_mask = (
        math.floor(noise_ratio * (len(tokens) - prefix_length))
        if maybe_mask_multiple_tokens
        else 1
    )
    current_length = len([token for token in tokens if token != mask_token])
    goal_length = current_length - num_to_mask

    if num_to_mask > 1:
        while current_length > goal_length:
            # Randomly mask spans until we've masked half the orignal tokens
            position = random.randint(prefix_length, len(tokens))
            mask_len = random.randint(1, min(max_mask, current_length - goal_length))
            current_length -= len(
                [t for t in tokens[position : position + mask_len] if t != mask_token]
            )
            tokens = tokens[:position] + [mask_token] + tokens[position + mask_len :]

        assert current_length == len([t for t in tokens if t != mask_token])
    else:
        # Randomly mask spans until we've masked half the orignal tokens
        position = random.randint(prefix_length, len(tokens) - 1)
        mask_len = random.randint(1, max_mask)
        tokens = tokens[:position] + [mask_token] + tokens[position + mask_len :]

    return tokens


def _maybe_mask_targeted_tokens(
    tokens: List[str],
    mask_target_type: str,
    mask_cardinals_before_dipl_nouns_only: bool,
    mask_token: str,
) -> Optional[List[str]]:
    assert mask_target_type in SUPPORTED_MASKING_TARGETS.keys(), (
        mask_target_type,
        SUPPORTED_MASKING_TARGETS.keys(),
    )

    def _clean_token(token: str) -> str:
        return "".join([c for c in token.lower() if c.isalpha()])

    def _clean_token_keep_apostrophes(token: str) -> str:
        return "".join([c for c in token.lower() if c.isalpha() or c == "'"])

    # Randomly search through tokens to find token to mask
    for i in random.sample(range(len(tokens)), len(tokens)):
        if "cardinals" in mask_target_type:
            if mask_target_type == "cardinals":
                cardinal_set = CARDINALS
                dipl_nouns = DIPLOMACY_NOUNS
            elif mask_target_type == "plural_cardinals":
                cardinal_set = PLURAL_CARDINALS
                dipl_nouns = DIPLOMACY_PLURAL_NOUNS
            else:
                raise NotImplementedError("Need to implement!")

            # Case 1: cardinal at end of string
            if _clean_token(tokens[i]) in cardinal_set and (
                i + 1 == len(tokens) or tokens[i + 1] == "[EO_M]"
            ):
                tokens[i] = mask_token
                return tokens
            # Case 2: cardinal not at end of string
            elif _clean_token(tokens[i]) in cardinal_set and (
                not mask_cardinals_before_dipl_nouns_only
                or _clean_token(tokens[i + 1]) in dipl_nouns
            ):
                num_tokens_to_mask = 2 if mask_cardinals_before_dipl_nouns_only else 1
                # A possible heuristic: set (i + 1) to (i + 2) to mask diplomacy noun (e.g. "builds" or "fleets") after cardinal as well
                tokens = tokens[:i] + [mask_token] + tokens[(i + num_tokens_to_mask) :]
                return tokens
        elif mask_target_type == "noisy_locations":
            spans = [[1, _clean_token(tokens[i])]]
            if i + 1 < len(tokens):
                spans.append([2, " ".join([_clean_token(tokens[i + j]) for j in range(2)])])
            if i + 2 < len(tokens):
                spans.append([3, " ".join([_clean_token(tokens[i + j]) for j in range(3)])])

            for span in random.sample(spans, len(spans)):
                if span[1] in SUPPORTED_MASKING_TARGETS["noisy_locations"]:
                    tokens = tokens[:i] + [mask_token] + tokens[(i + span[0]) :]
                    return tokens
        elif mask_target_type == "negations":
            if (
                _clean_token(tokens[i]) in SUPPORTED_MASKING_TARGETS["negations"]
                or _clean_token_keep_apostrophes(tokens[i])[-3:] == "n't"
            ):
                tokens[i] = mask_token
                return tokens

    return None


@register_teacher("message_history_denoising_chunk")
class DenoisingChunkTeacher(BaseDialogueChunkTeacher):
    """
    Streaming data base Denoising teacher.

    Input: masked message
    Output: original message
    """

    def __init__(self, opt, shared=None):
        random.seed(opt.get("mask_seed", 1))
        super().__init__(opt, shared)
        self.id = "Base Denoising Chunk"

        self.noise = 0.5
        self.max_mask = 3
        self.mask_token = "MASK"

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        parser.add_argument(
            "--mask-seed",
            type=int,
            default=1,
            help="Whether to have simple classifier or a sequence discriminator, which appends the label at the end of message.",
        )
        parser.add_argument(
            "--mask-target-type",
            type=str,
            default=None,
            help="Optionally set specific masking target.",
            choices=SUPPORTED_MASKING_TARGETS.keys(),
        )
        parser.add_argument(
            "--unfocused-masking-on-examples-with-target-type",
            type=str,
            default=None,
            help="Optionally set specific masking target to filter examles by, while still using unfocused masking.",
            choices=SUPPORTED_MASKING_TARGETS.keys(),
        )
        parser.add_argument(
            "--unfocused-masking-on-examples-with-target-type-with-masking-of-target",
            type=str,
            default=None,
            help="Optionally set specific masking target to filter examles by, while still using unfocused masking. Always maskes target tokens",
            choices=SUPPORTED_MASKING_TARGETS.keys(),
        )
        parser.add_argument(
            "--mask-cardinals-before-dipl-nouns-only",
            type=bool,
            default=True,
            help="Optionally set whether to mask all kinds of cardinals or only those before Diplomacy nouns (e.g. 'builds' or 'armies')",
        )
        parser.add_argument(
            "--maybe-mask-multiple-tokens",
            type=bool,
            default=True,
            help="Controls whether we can mask more than one token (num of tokens to mask chosen stochastically)",
        )
        return parser

    @property
    def model_type(self) -> str:
        return "dialogue"

    @property
    def format(self) -> str:
        return "message_history_dialogue_chunk"

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        result = []

        assert self.opt["task_version"] == 3, self.opt["task_version"]

        for ex in examples:
            tokens = ex["labels"][0].split()

            if self.opt["mask_target_type"] is not None:
                assert not self.opt[
                    "maybe_mask_multiple_tokens"
                ], "Just supporting masking single targeted token for now"
                masked_tokens = _maybe_mask_targeted_tokens(
                    tokens,
                    self.opt["mask_target_type"],
                    self.opt["mask_cardinals_before_dipl_nouns_only"],
                    self.mask_token,
                )
            elif self.opt["unfocused_masking_on_examples_with_target_type"] is not None:
                if (
                    _maybe_mask_targeted_tokens(
                        [*tokens],
                        self.opt["unfocused_masking_on_examples_with_target_type"],
                        self.opt["mask_cardinals_before_dipl_nouns_only"],
                        self.mask_token,
                    )
                    is not None
                ):
                    masked_tokens = _mask_untargeted_tokens(
                        tokens,
                        self.noise,
                        self.max_mask,
                        self.mask_token,
                        self.opt["maybe_mask_multiple_tokens"],
                    )
                else:
                    masked_tokens = None
            elif (
                self.opt["unfocused_masking_on_examples_with_target_type_with_masking_of_target"]
                is not None
            ):
                if (
                    _maybe_mask_targeted_tokens(
                        [*tokens],
                        self.opt[
                            "unfocused_masking_on_examples_with_target_type_with_masking_of_target"
                        ],
                        self.opt["mask_cardinals_before_dipl_nouns_only"],
                        self.mask_token,
                    )
                    is not None
                ):
                    masked_tokens = _maybe_mask_targeted_tokens(
                        [*tokens],
                        self.opt[
                            "unfocused_masking_on_examples_with_target_type_with_masking_of_target"
                        ],
                        self.opt["mask_cardinals_before_dipl_nouns_only"],
                        self.mask_token,
                    )
                    assert masked_tokens is not None, tokens
                    masked_tokens = _mask_untargeted_tokens(
                        [*masked_tokens],
                        self.noise,
                        self.max_mask,
                        self.mask_token,
                        self.opt["maybe_mask_multiple_tokens"],
                    )
                else:
                    masked_tokens = None
            else:
                masked_tokens = _mask_untargeted_tokens(
                    tokens,
                    self.noise,
                    self.max_mask,
                    self.mask_token,
                    self.opt["maybe_mask_multiple_tokens"],
                )

            if masked_tokens is not None:
                # Hackily remove all the existing context, and replace it with the noised message
                ex["text"] = " ".join(masked_tokens)

                result.append(ex)
        return result
