#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
from typing import Any, Optional, Dict, Tuple

from fairdiplomacy.typedefs import Phase, Power

import parlai_diplomacy.utils.datapath_constants as constants
from parlai_diplomacy.tasks.base_diplomacy_agent import BaseDiplomacyTeacher
from parlai_diplomacy.utils.game2seq.format_helpers.misc import load_json
from parlai_diplomacy.utils.game2seq.format_helpers.orders import OrdersUnflattener
from parlai_diplomacy.utils.game2seq.typing import Metadata
from parlai_diplomacy.utils.game2seq.dialogue_prediction import TrainingDialoguePredictionFormatter
import parlai_diplomacy.utils.justification_generation_helpers as justification

from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, SumMetric
from parlai.utils import logging
from parlai.utils.misc import warn_once


# This is a temporary piece of code anyway, so might as well just leave it in this file (will be removed Nov 15 2021)
class DeprecatedOptAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        raise RuntimeError(
            f"The option {option_string} is deprecated. Failing you so that you remove option from call to teacher."
        )


class BaseDialogueChunkTeacher(BaseDiplomacyTeacher):
    """
    Streaming data base dialogue teacher for messages/orders.

    Label is next message
    """

    @staticmethod
    def add_cmdline_args(
        argparser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        argparser.register("action", "ignore", DeprecatedOptAction)

        argparser.add_argument(
            "--dialogue-single-turn",
            type=bool,
            default=True,
            help="Only predict one dialogue message at a time",
            action="ignore",
        )

        argparser.add_argument(
            "--include-silence-messages",
            type=bool,
            default=False,
            help="(Deprecated: Has no effect)",
            action="ignore",
        )
        argparser.add_argument(
            "--calculate-year-metrics",
            type=bool,
            default=False,
            help="Calculate metrics per-year. Helpful for debugging.",
        )
        argparser.add_argument(
            "--calculate-ppl-by-rating-metrics",
            type=bool,
            default=False,
            help=(
                "Include a bunch of additional metrics about perplexity by rating and other minor metadata properties"
            ),
        )
        argparser.add_argument(
            "--include-sleep-messages",
            type=bool,
            default=False,
            help="Include sleep messages in the dialogue teacher",
        )
        argparser.add_argument(
            "--output-draw-messages",
            type=bool,
            default=False,
            help="Include draw vote messages as target outputs",
        )
        argparser.add_argument(
            "--add-sleep-times",
            type=bool,
            default=False,
            help="Include sleep times in the dialogue history",
        )
        argparser.add_argument(
            "--add-recipient-to-prompt",
            type=bool,
            default=False,
            help="Add the recipient to the prompt token at the end of the input",
        )

        argparser.add_argument(
            "--include-style",
            type=bool,
            default=False,
            help="(Deprecated: Has no effect)",
            action="ignore",
        )
        argparser.add_argument(
            "--mark-bad-messages",
            type=str,
            default=None,
            help=(
                "Comma separated list of possible things to mark as BAD. Current options include: "
                + ", ".join(
                    TrainingDialoguePredictionFormatter.MARK_BAD_OR_FILTER_MESSAGES_OPTIONS
                )
            ),
        )
        argparser.add_argument(
            "--filter-bad-messages",
            type=str,
            default=None,
            help=(
                "Comma separated list of possible things to filter and not train on. Current options include: "
                + ", ".join(
                    TrainingDialoguePredictionFormatter.MARK_BAD_OR_FILTER_MESSAGES_OPTIONS
                )
            ),
        )
        argparser.add_argument(
            "--edit-bad-messages",
            type=str,
            default=None,
            help=(
                "Comma separated list of possible things to edit. Current options include: "
                + ", ".join(TrainingDialoguePredictionFormatter.EDIT_MESSAGES_OPTIONS)
            ),
        )
        argparser.add_argument(
            "--filter-bad-messages-about-draws",
            type=bool,
            default=False,
            help="Deprecated, use --filter-bad-messages draws instead",
        )
        argparser.add_argument(
            "--min-speaker-rating",
            type=float,
            default=None,
            help="Only train on examples with speaker rating at least this",
        )
        argparser.add_argument(
            "--max-game-redacted-words-percent",
            type=float,
            default=None,
            help="Only train on games that have <= this percent of redacted words",
        )
        argparser.add_argument(
            "--response-view-dialogue-model",
            type=bool,
            default=False,
            help=(
                "This normally should be False. "
                "If this is True, the input is game view from player A, "
                "the output is response from other players to player A, "
                "which is different from traditional teacher where the input is game view from player A, "
                "and the output is response from A to other players"
            ),
        )
        argparser.add_argument(
            "--extend-order-history-since-last-n-movement-phase",
            type=int,
            default=1,
            help="If teacher includes orderhistorysincelastmovementphase, use this to extend order history to additional movement phases",
        )
        argparser.add_argument(
            "--extend-state-history-since-last-n-movement-phase",
            type=int,
            default=0,
            help="If teacher includes state or shortstate, use this to extend state history to additional movement phases",
        )
        # pseudo order generation only flags
        argparser.add_argument(
            "--pseudo-order-generation",
            type=bool,
            default=False,
            help="Pseudo order generation teacher",
        )
        argparser.add_argument(
            "--pseudo-order-generation-future-message",
            type=bool,
            default=True,
            help="Include future sentence in pseudo orders",
        )
        argparser.add_argument(
            "--pseudo-order-generation-injected-sentence",
            type=str,
            default=None,
            help="Injected sentence for pseudo order generation; the response 'all' will result in a random selection of pre-approved responses for all powers",
        )
        argparser.add_argument(
            "--pseudo-order-generation-inject-all",
            type=bool,
            default=True,
            help="Inject sentence for all players that received a message from current speaking power",
        )
        argparser.add_argument(
            "--pseudo-order-generation-partner-view",
            type=bool,
            default=False,
            help="Partner view for annotating recipient pseudo orders with a 'single order' prediction model",
        )
        argparser.add_argument(
            "--pseudo-order-generation-current-phase-prefix",
            type=bool,
            default=False,
            help="When generating pseudo orders, use the pre-computed current phase pseudo orders as a prefix",
        )
        argparser.add_argument(
            "--two-party-dialogue",
            type=bool,
            default=False,
            help="Only show dialogue history betwen sender and recipient of current message",
        )
        argparser.add_argument(
            "--no-speaker-dialogue-history",
            type=bool,
            default=False,
            help="Omit all of the speaker's own messages in the dialogue history",
        )
        argparser.add_argument(
            "--remove-n-latest-messages-from-dialogue-history",
            type=int,
            default=None,
            help="To catch non-sequiturs, randomly remove n latest messages from message history",
        )
        # Pseudo order arguments
        argparser.add_argument(
            "-appo",
            "--all-power-pseudo-orders",
            type=bool,
            default=True,
            help="Show pseudo orders for all powers, as opposed to only for the speaker and recipient(s)",
        )
        argparser.add_argument(
            "--single-view-pseudo-orders",
            type=bool,
            default=False,
            help="Use pseudo orders compiled from a single POV, self and partner ",
        )
        argparser.add_argument(
            "--rollout-pseudo-orders",
            type=bool,
            default=False,
            help=(
                "Use rollout pseudo orders on retreats and build phases (i.e. rollout to the next movement phase). "
                "Must have --single-view-pseudo-orders True"
            ),
        )
        argparser.add_argument(
            "--rollout-except-movement",
            type=bool,
            default=True,
            help="Condition on pseudo orders which only rollout on non-movement phases",
        )
        argparser.add_argument(
            "--rollout-phasemajor",
            type=bool,
            default=False,
            help=(
                "Key rollout pseudo orders by (phase, power), not the default (power, phase). "
                "Must have --single-view-pseudo-orders True --rollout-pseudo-orders True"
            ),
        )
        argparser.add_argument(
            "--rollout-actual-orders",
            type=bool,
            default=False,
            help=(
                "Use rollout actual orders on retreats and build phases (i.e. rollout to the next movement phase). "
            ),
        )
        argparser.add_argument(
            "--justification-generation",
            type=bool,
            default=False,
            help=("Generating justifications"),
        )
        argparser = BaseDiplomacyTeacher.add_cmdline_args(argparser, partial_opt)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self._check_is_pseudo_order_generation(opt)
        self._check_incompatible_opt(opt)
        if self.requires_pseudo_orders():
            # Set pseudo orders dir
            self._set_pseudo_orders_dir()
        super().__init__(opt, shared)

    def requires_pseudo_orders(self):
        # Requires loading pseudo orders
        if "pseudoorder" in self.opt["task"]:
            return True

        if self.opt.get("pseudo_order_generation_current_phase_prefix", False):
            return True

        return False

    def _check_incompatible_opt(self, opt) -> None:
        if opt.get("include_sleep_messages"):
            assert self.output_type in (
                "sleepclassifier",
                "sleepsix",
            ), "Do not use sleep messages outside of sleep classifier"

        if "dialogue_single_turn" in opt:
            assert opt.get("dialogue_single_turn")

        if opt.get("output_draw_messages"):
            assert (
                self.output_type == "recipientclassifier" or self.output_type == "drawclassifier"
            ), "Draw messages are currently only viewable by the recipient or draw classifier. Must have `--output-draw-messages False`"
            assert opt["task_version"] >= 2, "Draw messages only supported in version 2 or higher"

        if (opt.get("two_party_dialogue") or opt.get("no_speaker_dialogue_history")) and opt.get(
            "truncation", 0
        ) > 0:
            logging.warning(
                "WARNING: Message history will not be truncated when you set `--two-party-dialogue True` or `--no-speaker-dialogue-history True`"
            )

    def _set_pseudo_orders_dir(self):
        """
        Set the pseudo orders dir that we should pull pseudo orders from given the specified args.
        """
        if not self.opt.get("rollout_except_movement", True):
            # Rollout orders on every phase
            assert self.opt.get("rollout_pseudo_orders")
            assert self.opt.get("single_view_pseudo_orders")
            assert constants.PSEUDO_ORDER_PREFIX_ROLLOUT_DIR is not None
            self.pseudo_orders_dir = constants.PSEUDO_ORDER_PREFIX_ROLLOUT_DIR
        elif self.opt.get("single_view_pseudo_orders", False):
            # Single view pseudo orders
            assert not self.opt.get(
                "include_sleep_messages"
            ), "Cannot include sleep messages with single view pseudo orders"
            assert (
                constants.PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR is not None
            ), "Single turn pseudo orders have not been compiled for the latest data dump"
            self.pseudo_orders_dir = constants.PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR
        else:
            # Non-single view pseudo orders (i.e., compiled from a JointActions model)
            assert (
                constants.PSEUDO_ORDER_SINGLETURN_DIR is not None
            ), "Single turn pseudo orders have not been compiled for the latest data dump"
            self.pseudo_orders_dir = constants.PSEUDO_ORDER_SINGLETURN_DIR

        if self.opt.get("rollout_pseudo_orders", False):
            assert self.opt[
                "single_view_pseudo_orders"
            ], "To use rollout pseudo orders, you must set --single-view-pseudo-orders True"

        logging.info(f"Pseudo orders dir set to: {self.pseudo_orders_dir}")

    def _load_pseudo_orders(self, game_id: int):
        """
        Returns dict of pseudo orders given the game ID
        """
        path = os.path.join(self.pseudo_orders_dir, f"game_{game_id}_pseudo_orders.json")

        if not os.path.isfile(path):
            logging.error(f"Pseudo orders missing for game ID: {game_id}")
            return {}

        return load_json(path)

    def _check_is_pseudo_order_generation(self, opt):
        if opt.get("pseudo_order_generation"):
            assert self.output_type == "dialogue", "Pseudo order generation is for dialogue only"
            assert (
                "valid" in opt["datatype"]
                or "test" in opt["datatype"]
                or "evalmode" in opt["datatype"]
            ), "Pseudo order generation is for evaluation only"
            self.pseudo_order_gen = True
            warn_once("Loading a teacher for PSEUDO ORDER GENERATION.")

            assert not self.opt.get(
                "add_recipient_to_prompt"
            ), "Pseudo order generation should not add recipient to the prompt"
        else:
            self.pseudo_order_gen = False

        if opt.get("pseudo_order_generation_injected_sentence"):
            assert self.pseudo_order_gen

        if opt.get("pseudo_order_generation_partner_view"):
            assert self.pseudo_order_gen

        if opt.get("pseudo_order_generation_current_phase_prefix"):
            assert self.pseudo_order_gen

    def get_player_metadata(self, game, game_id) -> Dict:
        metadata = super().get_player_metadata(game, game_id)
        metadata["pseudo_order_gen"] = self.pseudo_order_gen

        if self.requires_pseudo_orders():
            metadata["pseudo_orders"] = self._load_pseudo_orders(game_id)

        return metadata

    def _set_game_metadata(self):
        super()._set_game_metadata()

    def _get_data_folder(self):
        return constants.FULLPRESS_GAME_JSONS

    def _get_pseudo_order_prefix_for_example(
        self, phase: Phase, ex: Dict[str, str], metadata: Metadata
    ) -> Optional[str]:
        """
        For a given example, get the pseudo orders associated with that example
        """
        example_id = ex["example_id"]
        pseudo_order_dct = metadata["pseudo_orders"].get(example_id, {})
        if not pseudo_order_dct:
            return None
        if self.opt.get("pseudo_order_generation_partner_view"):
            # Get the recipient pseudo orders
            if "partner" not in pseudo_order_dct:
                return None
            pseudo_orders = pseudo_order_dct["partner"]
        else:
            # Speaker pseudo orders
            if "self" not in pseudo_order_dct:
                return None
            pseudo_orders = pseudo_order_dct["self"]
        unflattened_pseudo_orders = OrdersUnflattener(
            constants.PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION  # Version the pseudo orders were compiled with
        ).unflatten_rollout_action(pseudo_orders, current_phase=phase)
        flattened_pseudo_orders = self.formatter.orders_flattener.flatten_rollout_action(
            unflattened_pseudo_orders, strip_current_phase=False
        )
        return flattened_pseudo_orders

    def build_example(
        self, game_id: int, speaker: Power, phase: Phase, ex: Dict[str, str], metadata: Metadata
    ) -> Optional[Dict[str, Any]]:
        """
        Override build example to add the pseudo order prefix if necessary

        Additionally, we add the timestamp for improved logging
        """
        example = super().build_example(game_id, speaker, phase, ex, metadata)
        if example is None:
            return example

        # Log the timestamp
        example["timestamp"] = ex.get("timestamp")

        if self.opt.get("pseudo_order_generation_current_phase_prefix", False):
            # Log the pseudo orders in the example ID
            example["pseudo_orders_prefix"] = self._get_pseudo_order_prefix_for_example(
                phase, ex, metadata
            )

        if self.opt.get("justification_generation", False):
            example = justification.modify_example_for_justification_generation(
                example, version=self.opt["task_version"]
            )

        return example

    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        """
        Custom evaluation to determine how the model is performing on examples from different
        phases.
        """
        if self.opt.get("calculate_year_metrics", False) and "metrics" in model_response:
            year = teacher_action["phase_id"][1:-1]
            loss = model_response["metrics"]["loss"].value()
            tokenized_length = model_response["metrics"]["clen"].value()

            self.metrics.add(f"{year}_loss", AverageMetric(loss, 1))
            self.metrics.add(f"{year}_count", SumMetric(1))
            self.metrics.add(f"{year}_length", AverageMetric(tokenized_length, 1))
        if (
            self.opt.get("calculate_ppl_by_rating_metrics", False)
            and "metadata_partial" in teacher_action
            and "metrics" in model_response
        ):
            metadata = teacher_action["metadata_partial"]
            metrics = model_response["metrics"]
            power_metadata = metadata.get("power_metadata")
            power = teacher_action.get("player")
            ppl = metrics.get("ppl")
            if ppl is not None:
                if power and power_metadata:
                    rating = power_metadata[power].get("rating")
                    if rating is not None:
                        if rating == 5:
                            self.metrics.add("ppl_r5", ppl)
                            self.metrics.add("count_r5", SumMetric(1))
                        elif rating == 4:
                            self.metrics.add("ppl_r4", ppl)
                            self.metrics.add("count_r4", SumMetric(1))
                        elif rating == 3:
                            self.metrics.add("ppl_r3", ppl)
                            self.metrics.add("count_r3", SumMetric(1))
                        elif rating == 2:
                            self.metrics.add("ppl_r2", ppl)
                            self.metrics.add("count_r2", SumMetric(1))
                        elif rating == 1:
                            self.metrics.add("ppl_r1", ppl)
                            self.metrics.add("count_r1", SumMetric(1))
                        else:
                            # There shouldn't be any other ratings, if somehow there is
                            # we will report stats for it
                            self.metrics.add("ppl_rother", ppl)
                            self.metrics.add("count_rother", SumMetric(1))

                        if "anon" in metadata:
                            self.metrics.add(f"anon_{metadata['anon']}_count", SumMetric(1))
                            if rating == 5:
                                self.metrics.add(
                                    f"anon_{metadata['anon']}__r5_count", SumMetric(1)
                                )
                                self.metrics.add(f"anon_{metadata['anon']}__r5_ppl", ppl)
                        if "phase_minutes" in metadata:
                            # For these metrics, we consider a game fast if there is less than 2h/phase
                            isfast = metadata["phase_minutes"] < 120
                            self.metrics.add(f"isfast{isfast}_count", SumMetric(1))
                            if rating == 5:
                                self.metrics.add(f"isfast{isfast}__r5_count", SumMetric(1))
                                self.metrics.add(f"isfast{isfast}__r5_ppl", ppl)
                    else:
                        # There shouldn't be any unknown, if somehow there is
                        # we will report stats for it
                        self.metrics.add("ppl_runknown", ppl)
                        self.metrics.add("count_runknown", SumMetric(1))
