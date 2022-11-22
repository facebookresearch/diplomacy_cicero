#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import math
import json
import os
import random
import sys
import traceback
from abc import ABC
from collections import deque
from glob import glob
from typing import Any, List, Dict, Optional

import numpy as np
from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import ChunkTeacher
from parlai.utils import logging
from parlai.utils.misc import warn_once

import parlai_diplomacy.tasks.common_task_utils as utls
import parlai_diplomacy.utils.datapath_constants as constants
from parlai_diplomacy.utils.game2seq import input_validation
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    POT_TYPE_CONVERSION,
    get_input_format,
    get_output_type,
    organize_game_by_phase,
)
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
from parlai_diplomacy.utils.datapath_constants import LATEST_VERSION
from parlai_diplomacy.utils.game2seq.typing import Metadata

from fairdiplomacy.annotate.lie_detection import read_lie_detector_annotations
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Phase, Power


"""
File containing a base teacher for all ParlAI Diplomacy teachers.
"""


@register_teacher("base_diplomacy")
class BaseDiplomacyTeacher(ChunkTeacher, ABC):
    """
    Base teacher for all Diplomacy data teachers.
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser.add_argument(
            "--n-chunks",
            type=int,
            default=-1,
            help="Number of chunks to load, default to -1 (loading all chunks for that data type), "
            "only useful for calculation such as data_stat to save time, normally it should be -1",
        )
        argparser.add_argument(
            "--counting-examples",
            type=bool,
            default=False,
            help="Loading teacher for counting purposes; should be False almost always",
        )
        argparser.add_argument(
            "--include-task-token",
            type=bool,
            default=False,
            help="Include token to indicate which task the model should perform; useful for multitasking",
        )
        argparser.add_argument(
            "--message-history-truncation",
            type=int,
            default=2048,
            help="Message history word truncation",
        )
        argparser.add_argument(
            "--task-version",
            type=int,
            default=LATEST_VERSION,
            help="Versioning for tasks. Useful for upgrading opt.",
        )
        # game metadata flags
        argparser.add_argument(
            "--include-game-info",
            type=bool,
            default=False,
            help="Include info about the game, like whether it is anonymous, how long the phases are, etc.",
        )
        argparser.add_argument(
            "--include-player-ratings",
            type=bool,
            default=False,
            help="Include player ratings in prompts for all models",
        )
        argparser.add_argument(
            "--include-draw-info",
            type=bool,
            default=False,
            help="Include info about whether the game is a private or public draw",
        )
        argparser.add_argument(
            "--include-draw-state",
            type=bool,
            default=False,
            help="Include a summary of the powers who have voted for a draw",
        )
        argparser.add_argument(
            "--hide-empty-draw-state",
            type=bool,
            default=False,
            help="Don't show empty draw state",
        )
        argparser.add_argument(
            "--include-centers-state",
            type=bool,
            default=False,
            help="Include centers in shortstate",
        )
        argparser.add_argument(
            "--include-builds-state",
            type=bool,
            default=False,
            help="Include builds in shortstate",
        )
        argparser.add_argument(
            "--player-rating-max",
            type=int,
            default=5,
            choices={5, 10, 20, 25, 50},
            help="Range of player ratings 1-max; default is 5",
        )
        argparser.add_argument(
            "--player-rating-percentiles",
            type=str,
            default="games_played",
            choices={"games_played", "messages_sent"},
            help="How to divide the player rating percentiles: can be done by the number of games a player has played, "
            "or by # of messages sent, which may be more appropriate for dialogue tasks",
        )
        argparser.add_argument(
            "--set-player-rating",
            type=int,
            default=-1,
            help="Sets player rating, defaults to -1 (Uses player ratings from database for evaluation and training),"
            "For generation, this flag has to be set if include_player_ratings was set during training",
        )
        # dialogue only flags
        argparser.add_argument(
            "--include-player-chattiness",
            type=bool,
            default=False,
            help="Include player chattiness in prompts for all dialogue models",
        )
        argparser.add_argument(
            "--set-player-chattiness",
            type=int,
            choices=[-1] + list(range(1, 21)),
            default=-1,
            help="Sets player chattiness, defaults to -1 (Uses player chattiness "
            "from database for evaluation and training),"
            "For generation, this flag has to be set if include_player_chattiness was set during training",
        )
        argparser.add_argument(
            "--only-phase",
            type=str,
            default=None,
            help="If set, only return examples from this phase. Probably only used for debugging.",
        )
        argparser.add_argument(
            "--only-game-id",
            type=int,
            default=None,
            help="If set, only return examples from this game. Probably only used for debugging.",
        )
        argparser.add_argument(
            "--only-chunk",
            type=int,
            default=-1,
            help="If >0, only load examples from this chunk. Probably only useful for debugging",
        )
        argparser.add_argument(
            "--skip-input-validation",
            type=bool,
            default=False,
            help="Skip regex input validation",
        )
        argparser.add_argument(
            "--input-validation-check-pct",
            type=float,
            default=0.1,
            help="Only check 10 percent of inputs via regex for speed",
        )
        argparser.add_argument(
            "--lie-detector-annotations-dir", help="Path to lie detector annotations dir"
        )
        argparser.add_argument(
            "--lie-detector-filter-above-stdev",
            type=float,
            default=None,
            help="Don't train on data with lie scores above this # of stdev from the mean",
        )
        argparser.add_argument(
            "--counterfactual-game-cache",
            type=int,
            default=0,
            help="If N > 0, builds a cache of N counterfactual games to pass through the metadata object",
        )
        argparser.add_argument(
            "--chunk-size", type=int, default=80, help="Number of games per chunk",
        )
        return argparser

    def __init__(self, opt, shared=None):
        self.id = "Base Diplomacy Teacher"
        self.opt = opt

        datatype = self.opt["datatype"]
        if "train:stream" in datatype:
            raise RuntimeError(
                "Temporary hack: Do *not* run with -dt train:stream! See PR #615 for more info."
            )
        self.data_split = datatype.split(":")[0]

        opt = self._override_opt(opt)
        self._initialize_formatter()

        if shared is None:
            # set map
            self.opt = opt
            if self.opt.get("n_chunks", -1) > 0 and self.opt.get("loading_chunks", 1000) != 1000:
                raise RuntimeError(
                    "To load a specific number of chunks, must specify --loading-chunks 1000"
                )
            self._set_chunk_idx_to_game_ids()
            self._set_game_metadata()
            self.game_cache = deque([], maxlen=opt["counterfactual_game_cache"])
        else:
            self.chunk_idx_to_game_ids = shared["chunk_idx_to_game_ids"]
            self.game_metadata = shared["game_metadata"]
            self.game_cache = shared["game_cache"]

        lie_annotations_dir = opt["lie_detector_annotations_dir"]
        if lie_annotations_dir and shared is None:
            logging.info(f"Reading lie detector annotations from: {lie_annotations_dir}")
            self.lie_detector_annotations = read_lie_detector_annotations(lie_annotations_dir)
        elif lie_annotations_dir and shared is not None:
            self.lie_detector_annotations = shared["lie_detector_annotations"]
        else:
            self.lie_detector_annotations = None

        super().__init__(opt, shared)

    @property
    def format(self) -> str:
        """
        Return formatting option.

        See `parlai_diplomacy.utils.game2seq` for descriptions.
        """
        task_name = self.opt["task"].split(":")[0]
        return task_name

    @property
    def output_type(self) -> str:
        """
        Return one of several options: order, dialogue, recipient, sleep
        """
        return get_output_type(self.format)

    @property
    def input_format(self) -> str:
        return get_input_format(self.format)

    def _initialize_formatter(self):
        self.formatter = sequence_formatter_factory(
            self.format, self.opt["task_version"], training=True
        )

        if not self.opt.get("skip_input_validation"):
            self.opt["input_validation_regex"] = self.formatter.get_input_validation_regex(
                self.format, self.opt
            )
            self.input_validation_failures = 0
            self.input_validation_total = 0

    def _override_opt(self, opt):
        """
        override some arguments here
        """
        if opt.get("include_player_chattiness") and self.output_type not in [
            "dialogue",
        ]:
            logging.warn(
                "Teacher is not a dialogue model, so setting --include_player_chattiness to False"
            )

            opt["include_player_chattiness"] = False
            opt["set_player_chattiness"] = -1

        return opt

    def get_buffersize(self):
        """
        Size of buffer.
        Override this in your child class to change the buffer size.
        """
        return 10_000

    def _get_data_folder(self):
        return constants.GAME_JSONS

    def _get_game_metadata_path(self):
        return constants.GAME_METADATA_PATH

    def _get_chattiness_metadata_path(self):
        return constants.CHATTINESS_METADATA_PATH

    def _get_n_chunks(self):
        datatype = self.opt["datatype"]
        if "valid" not in datatype:
            return self.opt.get("n_chunks", -1)
        return -1

    def _set_chunk_idx_to_game_ids(self):
        chunk_size = self.opt["chunk_size"]
        logging.info(f"[ Mapping chunk idxs to game ID ... chunk_size={chunk_size} ]")
        folder = self._get_data_folder()
        file_lst = glob(folder)
        game_id_to_file = {int(x.split("game_")[-1].split(".json")[0]): x for x in file_lst}
        game_ids = sorted(game_id_to_file.keys())

        train_game_ids = []
        valid_game_ids = []
        test_game_ids = []

        # load test game IDs
        with open(constants.TEST_ID_PATH, "r") as f:
            test_game_ids_from_file = [int(x) for x in f.read().splitlines()]

        for x in game_ids:
            if x in constants.VALID_GAME_IDS:
                valid_game_ids.append(x)
            elif x in test_game_ids_from_file:
                test_game_ids.append(x)
            else:
                train_game_ids.append(x)

        # for debugging
        if self.opt.get("only_game_id"):
            train_game_ids = [self.opt.get("only_game_id")]
            valid_game_ids = [self.opt.get("only_game_id")]
            test_game_ids = [self.opt.get("only_game_id")]

        logging.info(
            f"Found {len(train_game_ids)} train game IDs and {len(valid_game_ids)} valid game IDs and {len(test_game_ids)} test game IDs"
        )

        train_fles = [game_id_to_file[x] for x in train_game_ids]
        valid_fles = [game_id_to_file[x] for x in valid_game_ids]
        test_fles = [game_id_to_file[x] for x in test_game_ids]

        def chunkify(xs: List, chunk_size: int, chunk_id_offset: int = 0) -> Dict[int, List]:
            """Returns chunk id -> List"""
            n_chunks = math.ceil(len(xs) / chunk_size)
            return {
                (chunk_id_offset + chunk_i): [
                    game_id for i, game_id in enumerate(xs) if i % n_chunks == chunk_i
                ]
                for chunk_i in range(n_chunks)
            }

        # a chunk here is actually a list of files
        self.chunk_idx_to_game_ids = {
            "train": chunkify(train_fles, chunk_size),
            "valid": chunkify(valid_fles, chunk_size, chunk_id_offset=10000),
            "test": chunkify(test_fles, chunk_size, chunk_id_offset=20000),
        }

        # Debugging: limit to a particular chunk
        if self.opt.get("only_chunk", -1) > 0:
            assert self.opt["only_chunk"] in self.chunk_idx_to_game_ids["train"]
            train_ids = {
                self.opt["only_chunk"]: self.chunk_idx_to_game_ids["train"][self.opt["only_chunk"]]
            }
            self.chunk_idx_to_game_ids["train"] = train_ids

    def _load_ratings(self, game_metadata_json):
        percentiles_type = self.opt.get("player_rating_percentiles", "games_played")
        rating_max = self.opt.get("player_rating_max", 5)
        logging.info(
            f"Loading rating data with options: player_rating_percentiles={percentiles_type}, rating_max={rating_max}"
        )

        if percentiles_type == "games_played":
            ratings = [
                game[pwr]["logit_rating"]
                for game in game_metadata_json.values()
                for pwr in POWERS
                if (pwr in game and game[pwr] is not None)
            ]
        elif percentiles_type == "messages_sent":
            ratings = []
            for game in game_metadata_json.values():
                for pwr in POWERS:
                    if not (pwr in game and game[pwr] is not None):
                        continue
                    for _ in range(game[pwr].get("messages_sent", 0)):
                        ratings.append(game[pwr]["logit_rating"])
        else:
            raise RuntimeError(f"Ratings percentiles format {percentiles_type} invalid")

        assert 100 % rating_max == 0, "Must evenly divide into 100"
        step = 100 // rating_max
        rating_buckets = list(range(0, 100 + 1, step))
        rating_percentiles = np.percentile(ratings, rating_buckets).tolist()
        # add epsilon to the upper bound
        rating_percentiles[-1] += utls.EPSILON

        # now redo ratings given the buckets and the percentiles
        for _, game in game_metadata_json.items():
            for pwr in POWERS:
                if pwr in game and game[pwr] is not None:
                    game[pwr]["rating"] = np.digitize(
                        game[pwr]["logit_rating"], rating_percentiles
                    )

    def _set_game_metadata(self):
        game_metadata_json_path = self._get_game_metadata_path()

        with open(game_metadata_json_path) as meta_f:
            game_metadata_json = json.load(meta_f)

        game_metadata = {}

        # set up player ratings
        if self.opt.get("include_player_ratings", False):
            self._load_ratings(game_metadata_json)

        # set up player chattiness
        if self.opt.get("include_player_chattiness", False):
            path = self._get_chattiness_metadata_path()
            if path is None:
                raise RuntimeError("Chattiness annotations are not available")
            with open(path) as chat_f:
                game_metadata["chattiness_metadata"] = json.load(chat_f)

            chat_lengths = [
                max_length
                for game_id, pwrs in game_metadata["chattiness_metadata"].items()
                for pwr, max_length in pwrs.items()
                if pwr in POWERS
            ]

            chattiness_percentiles = np.percentile(chat_lengths, utls.CHATTINESS_BUCKETS).tolist()
            chattiness_percentiles[-1] += utls.EPSILON
            game_metadata["chattiness_percentiles"] = chattiness_percentiles

        # save all metadata
        game_metadata["metadata"] = game_metadata_json

        self.game_metadata = game_metadata

    def get_num_samples(self, opt):
        """
        Return the number of samples given the datatype.

        NOTE: we deliberately put an incorrect samples to avoid counting;
        this will not affect training, except for the fact that the number of
        epochs will *always* be incorrect
        """
        return 20_000_000, 20_000_000

    def get_fold_chunks(self, opt) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt["datatype"].split(":")[0]
        n_chunks = self._get_n_chunks()
        all_chunk_idxs = list(self.chunk_idx_to_game_ids[datatype].keys())
        chunk_idxs_to_load = all_chunk_idxs[:n_chunks] if n_chunks > 0 else all_chunk_idxs

        if n_chunks > 0:
            warn_once(
                f"Loading only {n_chunks} chunks out of {len(chunk_idxs_to_load)} chunks in datatype {datatype}!"
            )
            chunk_idxs_to_load = chunk_idxs_to_load[:n_chunks]

        return chunk_idxs_to_load

    def _maybe_validate_examples(self, examples):
        # validate inputs via regex
        if not (self.opt.get("skip_input_validation") or "," in self.opt["task"]):
            for ex in examples:
                if random.random() > self.opt["input_validation_check_pct"]:
                    # by default, only run on 10% of examples
                    continue
                try:
                    self.input_validation_total += 1
                    input_validation.validate(
                        self.opt["input_validation_regex"],
                        ex["text"],
                        throw_on_failure=not self.opt.get("log_input_validation"),
                    )
                except input_validation.InputValidationException:
                    # Log first failure, then periodically (on powers of 2 after 128) afterwards
                    self.input_validation_failures += 1
                    if self.input_validation_failures == 1 or (
                        self.input_validation_failures >= 128
                        and (self.input_validation_failures & (self.input_validation_failures - 1))
                        == 0
                    ):
                        logging.warning(
                            f"Input validation failure ({self.input_validation_failures} / {self.input_validation_total} = {self.input_validation_failures / self.input_validation_total}):"
                        )
                        traceback.print_exc()

    def load_from_chunk(self, chunk_idx: int) -> List[Dict]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        # in this case, a chunk maps to a set of files rather than a single file
        chunk_paths = self.chunk_idx_to_game_ids[self.data_split][chunk_idx]
        directory = os.path.dirname(self.folder)
        fles_to_load = [os.path.join(directory, path) for path in chunk_paths]
        logging.info(f"Loading {len(fles_to_load)} games from chunk ID {chunk_idx}")
        if len(fles_to_load) > 100:
            logging.warning(
                "This chunk is particular large. Expect longer wait times when formatting."
            )

        examples = []
        for fle in fles_to_load:
            with open(fle, "r") as f:
                game = Game.from_json(f.read())
            game_id = int(fle.split("game_")[-1].split(".json")[0])
            try:
                data = list(self._generate_example_tuples(game, game_id))
                self._maybe_validate_examples(data)
            except Exception:
                # Parlai deadlocks when a dataloader thread raises an exception. Here we
                # catch exceptions and exit the process instead of raising an exception
                # into the parlai core code.
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            examples += data

            if len(examples) >= int(os.environ.get("MAX_EXAMPLES", 1e10)):
                break

        logging.info(f"Loaded {len(examples)} examples from chunk {chunk_idx}.")

        return examples

    def _extract_metadata_fields(
        self, metadata: Dict[str, Any], metadata_fields: List[str]
    ) -> Dict[str, Any]:
        return {field: metadata[field] for field in metadata_fields if field in metadata}

    def build_example(
        self, game_id: int, speaker: Power, phase: Phase, ex: Dict[str, str], metadata: Metadata
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ParlAI training example
        """
        example = {
            "task_version": metadata["task_version"],
            "game_id": game_id,
            "phase_id": phase,
            "player": speaker,
            "text": ex["input"],
            "labels": [ex["output"]],
            "episode_done": True,
            "metadata_partial": self._extract_metadata_fields(
                metadata,
                [
                    "power_metadata",
                    "anon",
                    "phase_minutes",
                    "pot_type",
                    "draw_type",
                    "has_draw_votes",
                ],
            ),
        }
        if "example_id" in ex:
            example["example_id"] = ex["example_id"]

        return example

    def _generate_example_tuples(self, game, game_id):
        """
        Given a game JSON as input, creates examples for training

        :param game: Game JSON in DipCC format
        :return: list of examples
        """
        metadata = self.get_player_metadata(
            game, game_id
        )  # this adds additional metadata to the game
        metadata["game_cache"] = self.game_cache

        examples = []
        if (
            self.output_type == "order"
            or self.output_type == "allorder"
            or self.output_type == "allorderindependent"
        ):
            # Filter all holds flag for order prediction task
            if self.opt.get("filter_all_holds", False):
                metadata["filter_all_holds"] = True

        seqs = self.formatter.change_format(game, self.input_format, metadata)

        for phase, phase_data in seqs.items():
            for speaker, data in phase_data.items():
                assert (
                    "message_history_prefix" not in self.format
                ), "Not supported anymore. Use --train-on-message-prefixes"
                if (
                    self.output_type == "dialogue"
                    or self.output_type == "sleepclassifier"
                    or self.output_type == "sleepsix"
                    or self.output_type == "recipientclassifier"
                    or self.output_type == "drawclassifier"
                    or self.output_type == "allorderindependent"
                    or self.output_type == "allorderindependentrollout"
                    or self.output_type == "plausiblepseudoorder"
                    or self.output_type == "liedetector"
                    or self.output_type == "humanvsmodeldiscriminator"
                ):
                    # type checking; for dialogue, we return a list of a possible
                    # examples as we iteratively build up the examples for each phase
                    # for the all order message history and all order many tasks,
                    # we have a list of examples for every order prediction
                    assert type(data) is list
                else:
                    assert type(data) is dict
                    data = [data]
                for ex in data:
                    if "output" not in ex:
                        warn_once("Some examples are missing orders/messages; skipping")
                        continue
                    if self.opt.get("only_phase") is not None and self.opt["only_phase"] != phase:
                        continue

                    # Build a parlai example
                    example = self.build_example(game_id, speaker, phase, ex, metadata)
                    if example is None:
                        continue

                    if self.should_filter_for_lie_scores(example):
                        continue

                    examples.append(example)

        self.add_to_cache(game)

        return examples

    def add_to_cache(self, game):
        # Maybe cache game
        if self.opt.get("counterfactual_game_cache", 0) > 0:
            game_json = json.loads(game.to_json())
            game_json_by_phase = organize_game_by_phase(game_json)
            self.game_cache.append(game_json_by_phase)

    def get_player_metadata(self, game, game_id) -> Dict:
        """
        Adds player metadata (like rating, chattiness, etc.)

        :param game:
        """
        str_game_id = str(game_id)
        metadata = {}

        # POWER SPECIFIC METADATA
        metadata["power_metadata"] = {power: {} for power in POWERS}

        # set rating
        if self.opt.get("include_player_ratings", False):
            if self.opt.get("set_player_rating", -1) > -1:
                # set player rating from the CLI
                for _, dct in metadata["power_metadata"].items():
                    dct["rating"] = self.opt["set_player_rating"]
            elif str_game_id in self.game_metadata["metadata"]:
                curr_game_metadata = self.game_metadata["metadata"][str_game_id]
                # get actual player rating
                for power, dct in metadata["power_metadata"].items():
                    if power in curr_game_metadata and curr_game_metadata[power] is not None:
                        dct["rating"] = curr_game_metadata[power]["rating"]

        # set chattiness
        if self.opt.get("include_player_chattiness", False):
            if self.opt.get("set_player_chattiness", -1) > -1:
                # set player rating from the CLI
                for _, dct in metadata["power_metadata"].items():
                    dct["chattiness"] = self.opt["set_player_chattiness"]
            elif str_game_id in self.game_metadata["chattiness_metadata"]:
                # get actual player rating
                for power, dct in metadata["power_metadata"].items():
                    chattiness = self.game_metadata["chattiness_metadata"][str_game_id][power][
                        "chattiness"
                    ]
                    chattiness = np.digitize(
                        chattiness, self.game_metadata["chattiness_percentiles"]
                    )
                    dct["chattiness"] = chattiness
            else:
                logging.warning("Chattiness metadata is missing")
                for _, dct in metadata.items():
                    dct["chattiness"] = 0

        # GAME METADATA
        game_metadata = self.game_metadata["metadata"][str_game_id]
        # game info: anonymous, phase length, pot type, contains unknown phases
        if self.opt.get("include_game_info", False):
            metadata["anon"] = "ANON" if game_metadata["anon"] == "Yes" else "NON-ANON"
            metadata["phase_minutes"] = game_metadata["phase_minutes"]
            metadata["pot_type"] = POT_TYPE_CONVERSION[game_metadata["pot_type"]]
            if (
                "Unknown" in game_metadata["message_phase_stats"]
                and len(game_metadata["message_phase_stats"]) == 1
            ):
                metadata["all_unknowns"] = True
            else:
                metadata["all_unknowns"] = False
        # draw info: private vs. public
        if self.opt.get("include_draw_info", False):
            metadata["draw_type"] = (
                "PUBLIC" if game_metadata["draw_type"] == "draw-votes-public" else "PRIVATE"
            )

        # Counts of messages ad words and redaction tokens
        metadata["message_stats"] = game_metadata["message_stats"]

        # high level metadata
        metadata["game_id"] = game_id
        metadata["opt"] = self.opt

        # task specific args
        metadata["is_training"] = True  # assert that the model is training
        metadata["task_version"] = self.opt["task_version"]  # track versioning

        # draw vote metadata from Game object
        metadata["has_draw_votes"] = game.get_metadata("has_draw_votes") == "True"

        return metadata

    def create_message(self, queue_output, entry_idx=0) -> "Message":
        """
        Given the tuple output of the queue, return an act.
        """
        return queue_output

    def share(self):
        shared = super().share()
        shared["chunk_idx_to_game_ids"] = self.chunk_idx_to_game_ids
        shared["game_metadata"] = self.game_metadata
        shared["game_cache"] = self.game_cache
        shared["lie_detector_annotations"] = self.lie_detector_annotations
        return shared

    def _get_base_msg(self, queue_output):
        base_msg = {
            "episode_done": True,
            "player_id": queue_output["player_id"],
            "player": queue_output["player"],
            "game_id": queue_output["game_id"],
            "phase_id": queue_output["phase_id"],
            "labels": [queue_output["order"]],
        }

        base_msg.update(queue_output)

        return base_msg

    def should_filter_for_lie_scores(self, ex) -> bool:
        """Return True iff example should be excluded due to lie score filtering"""
        if (
            self.lie_detector_annotations is None
            or self.opt["lie_detector_filter_above_stdev"] is None
        ):
            return False

        if ex["game_id"] not in self.lie_detector_annotations:
            logging.warning(f"Game with no lie scores: {ex['game_id']}")
            return False

        if self.output_type == "allorder" or self.output_type == "order":
            # for an all-orders model, remove a datapoint if any of the powers'
            # orders were marked as a lie
            for pair, phase_to_score in self.lie_detector_annotations[ex["game_id"]].items():
                if (
                    ex["phase_id"] in phase_to_score
                    and phase_to_score[ex["phase_id"]]
                    > self.opt["lie_detector_filter_above_stdev"]
                ):
                    return True
            return False
        else:
            raise NotImplementedError(self.output_type)
