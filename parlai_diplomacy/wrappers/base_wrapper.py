#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import functools
import logging
import json
from typing import Dict, FrozenSet, List, Any, Tuple, Optional
from torch import LongTensor

from parlai.core.agents import create_agent_from_model_file
from parlai_diplomacy.utils.game2seq import input_validation
from parlai_diplomacy.utils.game2seq.typing import Metadata, ParlAIAct
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    POT_TYPE_CONVERSION,
    get_input_format,
    get_output_type,
)
from parlai_diplomacy.utils.datapath_constants import LATEST_VERSION
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
import parlai_diplomacy.utils.loading as load

from fairdiplomacy.timestamp import Timestamp
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy import pydipcc
from fairdiplomacy.pseudo_orders import PseudoOrders, RolloutType
from fairdiplomacy.typedefs import Power, Phase, RolloutAction
from fairdiplomacy.utils.game import assert_game_from_view_of
from fairdiplomacy.utils.slack import GLOBAL_SLACK_EXCEPTION_SWALLOWER
from parlai_diplomacy.utils.game2seq.format_helpers.opt_utils import (
    expects_bilateral_pseudo_orders,
    expects_rollout_type,
)


load.register_all_agents()


"""
Base module for wrapping a ParlAI agent to produce orders, dialogue, or a
classification prediction from a game object.
"""

Overrides = Dict[str, Any]


@functools.lru_cache()
def load_parlai_agent_cached(*, model_path: str, frozen_overrides: FrozenSet[Tuple[str, Any]]):
    """
    Cache loading of the parlai agents.
    """
    override_opts = {k: v for k, v in frozen_overrides}
    return create_agent_from_model_file(model_path, override_opts)


def load_opt(path):
    opt_path = path + ".opt"
    with open(opt_path, "r") as f:
        opt = json.load(f)
    return opt


def freeze_dictionary_args(args: Dict[str, Any]) -> Optional[FrozenSet[Tuple[str, Any]]]:
    """
    Freeze override dictionary args into a tuple for the purpose of caching.

    Returns None if cannot be frozen
    """
    # First check that none of the values are mutable
    for v in args.values():
        if isinstance(v, (dict, set, list)):
            return None

    # Then, return a tuple corresponding to the keys and values of the dictionary,
    # in sorted order of the keys
    return frozenset((k, args[k]) for k in args.keys())


class BaseWrapper(ABC):
    OPT_OVERRIDE = {"interactive_mode": True, "skip_generation": False}

    def __init__(self, model_path: str, additional_args=None):
        override_opts = self._load_overrides(overrides=additional_args.get("overrides"))
        # Try freezing args for cacheed loading
        frozen_args = freeze_dictionary_args(override_opts)
        if frozen_args is not None:
            self.parlai_agent = load_parlai_agent_cached(
                model_path=model_path, frozen_overrides=frozen_args
            )
        else:
            # Else just load regularly
            logging.warning("Override opts must be immutable types to cache model load")
            self.parlai_agent = create_agent_from_model_file(model_path, override_opts)
        self.metadata = self._get_player_metadata(self.opt)
        self._initialize_formatter()

        if "input_validation_regex" not in self.opt:
            self.opt["input_validation_regex"] = self.formatter.get_input_validation_regex(
                self.task_name, self.opt
            )

        self.seen_input_example = False

    @property
    def opt(self):
        return self.parlai_agent.opt

    @property
    def task_name(self):
        return self.opt["task"].split(":")[0]

    @property
    def input_format_str(self):
        return get_input_format(self.task_name)

    @property
    def output_type(self):
        return get_output_type(self.task_name)

    def _initialize_formatter(self):
        self.formatter = sequence_formatter_factory(
            self.task_name, self.metadata["task_version"], training=False
        )

    def _load_overrides(self, overrides: Overrides = None):
        """
        Set agent overrides using those specified in the config, as well as
        default overrides.
        """
        opts = {}
        opts.update(self.OPT_OVERRIDE)
        if overrides:
            opts.update(overrides)

        logging.info(f"[{self.__class__.__name__}] Using overrides: {opts}")
        return opts

    def _get_player_metadata(self, opt: Dict[Any, Any]) -> Metadata:
        """
        Set metadata for a given player based on the opt
        """
        metadata: Metadata = {}

        # POWER METADATA
        metadata["power_metadata"] = {power: {} for power in POWERS}
        # player ratings
        if opt.get("include_player_ratings", False):
            player_rating = opt.get("set_player_rating", None)
            assert (
                player_rating is not None and player_rating != -1
            ), "Must set `overrides.set_player_rating`."
            logging.info(f"Setting player rating to: {player_rating}")
            for _, v in metadata["power_metadata"].items():
                v["rating"] = player_rating

        # player chattiness
        if opt.get("include_player_chattiness", False):
            player_chattiness = opt.get("set_player_chattiness", None)
            assert player_chattiness is not None, "Must set `overrides.set_player_chattiness`."
            logging.info(f"Setting player chattiness to: {player_chattiness}")
            for _, v in metadata["power_metadata"].items():
                v["chattiness"] = player_chattiness

        # HIGH LEVEL METADATA

        # add opt
        metadata["opt"] = opt

        # task versioning
        version = opt.get("task_version", 0)
        metadata["task_version"] = version
        if version < LATEST_VERSION:
            task = opt["task"]
            stars = "\n" + "*" * 80 + "\n"
            logging.warning(
                f"\n{stars}WARNING: using version {version} for task {task}, "
                f"which is lower than the latest version {LATEST_VERSION}!"
                f"{stars}\n"
            )

        # game info
        metadata["anon"] = "ANON"  # play anonymous games

        # phase minutes
        if "phase_minutes" in opt:
            err_msg = "Setting phase minutes via a config is DEPRECATED. Phase length will be inferred from the game object."
            if opt["phase_minutes"] != 1440:
                # 1440 is the default, this means someone intentionally tried to set phase minutes to something else
                raise RuntimeError(err_msg)
            else:
                # Default is set, just add a warning
                logging.warning(err_msg)

        assert opt.get("pot_type", "Sum-of-squares") in POT_TYPE_CONVERSION.keys()
        metadata["pot_type"] = POT_TYPE_CONVERSION[opt.get("pot_type", "Sum-of-squares")]
        logging.info(f"Setting pot type to: {metadata['pot_type']}")
        metadata["all_unknowns"] = False

        # draw info
        metadata["draw_type"] = "PUBLIC"  # play with a public draw type;
        metadata["has_draw_votes"] = True

        return metadata

    def add_literal_to_block_list(self, literal: str) -> None:
        """
        Add a literal phrase to the beam block list for blocking on generation.

        Please invoke this method carefully! BPE dictionaries make this process complicated.
        """
        assert hasattr(
            self.parlai_agent, "beam_block_list"
        ), "This is not a generative model; cannot block tokens"
        self.parlai_agent.beam_block_list._add_literal(literal)
        ngram = self.parlai_agent.dict.txt2vec(literal)
        logging.warning(f"Added phrase {literal} to block list, which is parsed as: {ngram}")

    def get_candidate_logprobs(
        self,
        act: Dict[str, Any],
        skip_prefix: Optional[bool] = False,
        skip_end_token: Optional[bool] = False,
    ) -> List[float]:
        """Get the conditional logprob of each BPE token in each candidate.

        Returns: log p(x) for each token x in each candidate in act['cand_scores'] """
        end_token = self.parlai_agent.dict[self.parlai_agent.END_IDX]
        scores = []
        for cand_score in act["cand_scores"]:
            toks, tok_scores = zip(*cand_score)
            start = (toks.index(":") + 1) if skip_prefix else 0
            end = toks.index(end_token) if skip_end_token else len(toks)
            scores.append(tok_scores[start:end])
        return [-float(sum(scores)) for scores in scores]

    def set_generation_args(
        self,
        inference: str,
        beam_size: Optional[int] = None,
        topk: Optional[int] = None,
        topp: Optional[float] = None,
        beam_delay: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Overrides:
        """
        Set generation args for decoding.

        Returns a dict of the original args.
        """

        og_inference = self.opt.get("inference")
        if inference is None:
            raise RuntimeError(
                "ParlAI agent is not a generative model. Cannot change generation args."
            )

        # change inference
        assert inference in {
            "beam",
            "greedy",
            "topk",
            "nucleus",
            "delayedbeam",
        }, f"Inference {inference} is not valid"
        self.parlai_agent.opt["inference"] = inference

        # change beam size
        og_beam_size = self.opt.get("beam_size", 1)
        if beam_size is not None:
            assert beam_size > 0, "Beam size must be greater than zero"
            self.parlai_agent.opt["beam_size"] = beam_size
            self.parlai_agent.beam_size = beam_size

        # change top k
        og_topk = self.parlai_agent.opt["topk"]
        if topk is not None:
            self.parlai_agent.opt["topk"] = topk

        # change topp
        og_topp = self.parlai_agent.opt["topp"]
        if topp is not None:
            self.parlai_agent.opt["topp"] = topp

        # change beam delay
        og_beam_delay = self.parlai_agent.opt["beam_delay"]
        if beam_delay is not None:
            self.parlai_agent.opt["beam_delay"] = beam_delay

        # change temperature
        og_temperature = self.parlai_agent.temperature
        if temperature is not None:
            self.parlai_agent.temperature = temperature

        # return original args so that we can change them back at will
        og_args = {
            "inference": og_inference,
            "beam_size": og_beam_size,
            "topk": og_topk,
            "topp": og_topp,
            "beam_delay": og_beam_delay,
            "temperature": og_temperature,
        }

        return og_args

    def get_model_pred(
        self,
        input_seq: str,
        candidates: Optional[List[str]] = None,
        prefix_str: Optional[str] = None,
    ) -> ParlAIAct:
        """
        Return the model's prediction for an input sequence.

        Args:
        - input_seq (str): input sequence for which we are getting a prediction
        - candidates (Optional[List[str]]): optional list of candidates for scoring
        - prefix_str (Optional[str]): optional prefix string for decoding
        """

        ex = {"text": input_seq, "episode_done": True}

        # candidate ranking
        old_rank_candidates = self.parlai_agent.rank_candidates
        # some test agents don't have this flag
        old_skip_generation = getattr(self.parlai_agent, "skip_generation", False)
        if candidates is not None:
            assert candidates, "Candidates is empty"
            assert prefix_str is None, f"{prefix_str}"
            self.parlai_agent.rank_candidates = True
            self.parlai_agent.skip_generation = True
            ex["label_candidates"] = candidates

        # decoding a prefix
        if prefix_str:
            prefix_toks = LongTensor(self.parlai_agent.dict.txt2vec(prefix_str))
            logging.debug(f"Decoding with prefix {prefix_str}")
            self.parlai_agent.set_prefix_tokens(prefix_toks)

        self.parlai_agent.observe(ex)
        act = self.parlai_agent.act()

        # reset candidate ranking
        self.parlai_agent.rank_candidates = old_rank_candidates
        self.parlai_agent.skip_generation = old_skip_generation

        # reset prefix tokens
        if prefix_str:
            self.parlai_agent.set_prefix_tokens(None)

        return act

    def _extract_beam_texts(
        self, input_seq, temp_beam_size: int, cached_text_vec: Optional[LongTensor] = None,
    ) -> Tuple[List[Tuple[str, float]], LongTensor]:
        """
        Extract beam texts from agent act.

        Return the beam texts as well as the text_vecs, which are also optionally taken as input.
        """
        ex = {"text": input_seq, "episode_done": True}
        if cached_text_vec is not None:
            # We cached the text_vec from a previous call;
            # We pass it in as input to the model so that it does not re-tokenize the text
            ex["text_vec"] = cached_text_vec
            ex["added_start_end_tokens"] = True

        initial_beam_size = self.parlai_agent.beam_size
        self.parlai_agent.beam_size = temp_beam_size
        self.parlai_agent.opt["beam_size"] = temp_beam_size

        obs = self.parlai_agent.observe(ex)
        cached_text_vec = obs["text_vec"]  # cache the text vec for next time
        act = self.parlai_agent.act()

        # return beam size to old beam size
        self.parlai_agent.beam_size = initial_beam_size
        self.parlai_agent.opt["beam_size"] = initial_beam_size

        return act["beam_texts"], cached_text_vec  # type: ignore

    def get_model_pred_many(
        self,
        input_seq: str,
        num_preds: int,
        batch_size: Optional[int] = None,
        prefix_str: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return the model's beam texts for a prediction.

        Args:
        - input_seq (str): input sequence for which we are getting a prediction
        - n (int): number of model predictions to return
        - batch_size (Optional[int]): batch size for model calls; used when memory
          imposes a constraint
        - prefix_str (Optional[str]): optional prefix string for decoding
        """
        inference = self.opt.get("inference")
        if inference is None:
            raise RuntimeError("ParlAI agent is not a generative model. Cannot sample.")

        beams = []
        if num_preds > 1:
            assert (
                inference != "greedy"
            ), f"Cannot do greedy sampling with a beam size > 1; requested beam size is {num_preds}"
        if batch_size is None:
            batch_size = num_preds

        if num_preds > batch_size:
            assert inference != "beam", "Cannot do beam search with batching"

        # decoding a prefix
        if prefix_str:
            prefix_toks = LongTensor(self.parlai_agent.dict.txt2vec(prefix_str))
            logging.debug(f"Decoding with prefix {prefix_str}")
            self.parlai_agent.set_prefix_tokens(prefix_toks)

        cached_text_vec = None
        beams = []
        for i in range(0, num_preds, batch_size):
            batch_size = min(num_preds - i, batch_size)
            # cache text_vec in between calls
            partial_beams, cached_text_vec = self._extract_beam_texts(
                input_seq, batch_size, cached_text_vec,
            )
            beams += partial_beams

        # reset prefix tokens
        if prefix_str:
            self.parlai_agent.set_prefix_tokens(None)

        return beams

    @abstractmethod
    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Given a game object and possibly other args, return a string
        to feed to the model as input.

        Args:
        game: dipcc game object
        view_of_power: game from view of power
        target_power: optional power that may be used in formatting (e.g. could represent message recipient)
        timestamp: optional current timestamp
        """
        pass

    def maybe_show_input_example(self, formatted_input: str):
        """
        Display input example
        """
        if not self.seen_input_example:
            logging.info(
                f"First input example for model trained on task {self.task_name}: {formatted_input}"
            )
            self.seen_input_example = True

    def validate_input(self, input_seq: str):
        if "input_validation_regex" in self.opt:
            input_validation.validate(self.opt["input_validation_regex"], input_seq)

    def update_game_metadata(self, game: pydipcc.Game):
        """
        Dynamically update player metadata based on game metadata

        Currently, updates:
        - pot_type
        - phase_minutes
        """
        # Update pot type
        game_pot_type = game.get_scoring_system()
        if game_pot_type == pydipcc.Game.SCORING_DSS:
            new_model_pot_type = POT_TYPE_CONVERSION["Winner-takes-all"]
        elif game_pot_type == pydipcc.Game.SCORING_SOS:
            new_model_pot_type = POT_TYPE_CONVERSION["Sum-of-squares"]
        else:
            raise ValueError(f"pydipcc.Game has unknown scoring system: {game_pot_type}")

        if "pot_type" in self.metadata and self.metadata["pot_type"] != new_model_pot_type:
            logging.warning(
                f"Model pot type is currently {self.metadata['pot_type']}, but updated to {new_model_pot_type}"
            )
        self.metadata["pot_type"] = new_model_pot_type

        # Phase minutes
        phase_minutes = game.get_metadata("phase_minutes")
        if not phase_minutes:
            # Default to a 24 hour game
            phase_minutes = 1440
            logging.warning(
                "phase_minutes was not set in the game metadata. Defaulting to a 24 hour game."
            )

        if "phase_minutes" in self.metadata and self.metadata["phase_minutes"] != phase_minutes:
            logging.warning(
                f"Model phase_minutes is currently {self.metadata['phase_minutes']}, but updated to {phase_minutes}"
            )
        self.metadata["phase_minutes"] = phase_minutes

    def format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Before calling child method, we assert that the game view is correct; e.g., that the power
        is unable to see messages it would not be privy to.
        """
        # first assert that we have the correct game view
        assert_game_from_view_of(game, view_of_power)

        self.update_game_metadata(game)

        # next, call child class implementation
        formatted_input = self._format_input_seq(game, view_of_power, target_power, timestamp)
        self.maybe_show_input_example(formatted_input)
        with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
            self.validate_input(formatted_input)

        return formatted_input

    def format_output_seq(self, output_seq: str, *args, **kwargs) -> Any:
        """
        Given a text sequence returned by a model and a power, return the output
        format expected by the game engine

        Args:
        output_seq: output sequence of text returned by the model
        """
        raise NotImplementedError("Child classes must implement `format_output_seq`")

    def format_candidate_seqs(self, candidates) -> List[str]:
        """
        Format a list of candidates for a given power.

        These are candidates for re-scoring, so depending on the type of model,
        they could be orders, dialogue utterances, or whatever output type the
        model might expect.

        Returns:
        A list of formatted candidates
        """
        raise NotImplementedError(
            "In order to re-score candidates, "
            "child classes must implement `format_candidate_seqs`"
        )

    def score_candidate_seqs(
        self,
        game: pydipcc.Game,
        candidate_seqs: List[str],
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
        skip_prefix: bool = False,
        skip_end_token: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Given a game object and a list of formatted candidate output seqs, score the candidates
        """
        seq = self.format_input_seq(
            game, view_of_power, target_power=target_power, timestamp=timestamp
        )
        if candidate_seqs:
            act = self.get_model_pred(seq, candidates=candidate_seqs)
            output_logprobs = self.get_candidate_logprobs(act, skip_prefix, skip_end_token)

            return [(a, score) for a, score in zip(act["text_candidates"], output_logprobs)]
        else:
            # candidates are empty
            return []

    def expects_pseudo_orders(self) -> bool:
        return "pseudoorder" in self.opt["task"]

    def expects_actual_orders(self) -> bool:
        return "actualorders" in self.opt["task"]

    def expects_single_view_pseudo_orders(self) -> bool:
        return self.opt.get("single_view_pseudo_orders", False)

    def expected_rollout_type(self) -> RolloutType:
        return expects_rollout_type(self.opt)

    def expects_rollout_pseudo_orders(self) -> bool:
        return self.expected_rollout_type() != RolloutType.NONE

    def expects_extended_rollout_pseudo_orders(self) -> bool:
        return self.expected_rollout_type() == RolloutType.EXTENDED

    def expects_bilateral_pseudo_orders(self) -> bool:
        return expects_bilateral_pseudo_orders(self.opt)

    def update_pseudo_orders(self, phase: Phase, speaker: Power, pseudo_orders: PseudoOrders):
        """
        Update metadata with appropriate pseudo orders.
        """
        assert self.expects_pseudo_orders()

        assert pseudo_orders.check_rollout(
            self.expected_rollout_type(),
        ), f"Must pass rollout pseudo orders to the wrapper IFF expects_rollout_pseudo_orders is True. rollout_type= {self.expected_rollout_type().name}; pseudo_orders={pseudo_orders}"

        if not self.expects_bilateral_pseudo_orders() and pseudo_orders.is_bilateral():
            # Here we must throw an exception: there's not enough info
            raise ValueError("model expects all-powers pseudo orders")

        self.metadata.setdefault("pseudo_orders", {})
        self.metadata["pseudo_orders"].setdefault(phase, {})
        self.metadata["pseudo_orders"][phase][speaker] = pseudo_orders

    def update_actual_orders(self, phase: Phase, speaker: Power, rollout_action: RolloutAction):
        assert self.expects_actual_orders()
        self.metadata.setdefault("actual_orders", {})
        self.metadata["actual_orders"].setdefault(phase, {})
        self.metadata["actual_orders"][phase][speaker] = rollout_action
