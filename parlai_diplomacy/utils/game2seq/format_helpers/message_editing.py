#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# hack to resolve circular imports
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Optional

from collections import defaultdict
import copy
from enum import Enum
import logging
import re
import math
import json
import torch
from parlai.utils.safety import OffensiveStringMatcher

from fairdiplomacy.typedefs import MessageDict, OutboundMessageDict, Timestamp
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.typedefs import increment_last_message_time, with_time_sent
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.pseudo_orders import PseudoOrders

from parlai_diplomacy.utils.game2seq.typing import PhaseMsgs
from parlai_diplomacy.utils.game2seq.format_helpers import common_names
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageObjectPart,
    add_message_to_game_copy,
    is_draw_msg,
    is_unvote_draw_msg,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import uncorrupt_newlines

if TYPE_CHECKING:
    from parlai_diplomacy.wrappers.dialogue import BaseDialogueWrapper
    from parlai_diplomacy.wrappers.orders import ParlAIAllOrderIndependentRolloutWrapper
    from parlai_diplomacy.wrappers.classifiers import EnsembleNonsenseClassifierWrapper


class FilterReasons(Enum):
    """
    Reasons for filtering a message
    """

    NO_FILTER = "Unfiltered"
    OFF_LANG = "Offensive language"
    PHASE_REPEAT = "Phase repeats"
    CONSEC_SHORT = "Consecutive short messages"
    REDACTED = "Excess redacted tokens"
    ANY_REDACTED = "Contains any redacted tokens"
    AMPERSANDS = "Message contains ampersands"
    NAMES = "Contains what seem to be real-life names of people"
    URLS_OR_EMAILS = "Contains URLs or emails"
    MESSAGE_ABOUT_DRAWS = "Message contains the word draw"
    ZSHOT_NONSENSE = "Zero-shot nonsense detected"
    NONSENSE = "Discriminative-model nonsense detected"
    PSEUDO_ORDERS = "Pseudo orders correspondence"
    MUTES = "Contains webdip message for talking to someone who muted you"
    LOW_RATING = "Low Rating"
    GROUNDING = "Grounding"
    INSULTS = "Insults and rude language"
    GROUNDING_END_OF_GAME = "Grounding issues relating to game ending year"


class MessageFiltering:
    """
    Class that handles filtering of bad messages with various helper methods.
    """

    def __init__(
        self,
        filter_offensive_language: bool,
        filter_phase_repeats: bool,
        filter_consecutive_short: bool,
        filter_excess_redacted: bool,
        filter_any_redacted: bool,
        filter_ampersands: bool,
        filter_names: bool,
        filter_urls_emails: bool,
        filter_draw_discussion_when_missing_votes: bool,
        filter_mutes: bool,
        filter_grounding: bool,
        filter_insults: bool,
        grounding_last_playable_year: Optional[int],
        rating_threshold_first_message: float = 1.0,
        rating_threshold_other: float = 1.0,
        zshot_nonsense_classifier: Optional[Any] = None,
        ensemble_nonsense_classifier: Optional["EnsembleNonsenseClassifierWrapper"] = None,
        pseudo_orders_correspondence_threshold: Optional[float] = None,
        orders_model: Optional["ParlAIAllOrderIndependentRolloutWrapper"] = None,
        dialogue_model: Optional["BaseDialogueWrapper"] = None,
    ):
        self.filter_offensive_language = filter_offensive_language
        if self.filter_offensive_language:
            logging.info("Offensive language will be filtered using a word list.")
            self.osm = OffensiveStringMatcher()

        # filtering
        self.filter_phase_repeats = filter_phase_repeats
        self.filter_consecutive_short = filter_consecutive_short
        self.filter_excess_redacted = filter_excess_redacted
        self.filter_any_redacted = filter_any_redacted
        self.filter_ampersands = filter_ampersands
        self.filter_names = filter_names
        self.filter_urls_emails = filter_urls_emails
        self.filter_draw_discussion_when_missing_votes = filter_draw_discussion_when_missing_votes
        self.filter_mutes = filter_mutes
        self.filter_grounding = filter_grounding
        self.filter_insults = filter_insults

        self.grounding_last_playable_year = grounding_last_playable_year

        # zshot nonsense classifier for nonsense detection
        self.zshot_nonsense_classifier = zshot_nonsense_classifier

        # nonsense classifier for nonsense detection
        self.ensemble_nonsense_classifier = ensemble_nonsense_classifier

        # dialogue model, which can be used for rating-based filtering
        self.dialogue_model = dialogue_model
        self.rating_threshold_first_message = rating_threshold_first_message
        self.rating_threshold_other = rating_threshold_other
        assert (
            math.isclose(rating_threshold_first_message, 1.0)
            and math.isclose(rating_threshold_other, 1.0)
        ) or dialogue_model is not None, "Need to supply dialogue model for rating-based filtering"

        # orders model for pseudo orders-based lie filtering
        self.orders_model = orders_model
        # threshold for orders-based lie filtering
        self.pseudo_orders_correspondence_threshold = pseudo_orders_correspondence_threshold

        # collect statistics regarding how many messages are filtered and why
        self.statistics = defaultdict(int)

        # filtering annotations that will be displayed in game view
        self.filtering_annotations: List[Tuple[MessageDict, str, Dict]] = []

        self.filtered_messages: List[Tuple[MessageDict, str, Dict]] = []

    def clear_statistics(self):
        """
        Reset filtering statistics to zero
        """
        self.statistics = defaultdict(int)

    def get_statistics(self):
        """
        Return a dict of filter reason -> count
        """
        return self.statistics.copy()

    def report_statistics(self):
        """
        Report filtering statistics
        """

        def pct(num, tot):
            return round(num / tot, 4) * 100

        unfiltered = self.statistics[FilterReasons.NO_FILTER]
        filtered = sum([v for k, v in self.statistics.items() if k != FilterReasons.NO_FILTER])
        total = unfiltered + filtered
        if total == 0:
            return  # No need to report, no messages have been generated yet
        logging.info(f"{filtered} / {total} ({pct(filtered, total)}%) messages have been filtered")
        if filtered > 0:
            # Report the kinds of messages that have been filtered
            pcts = [
                (k, pct(v, filtered))
                for k, v in self.statistics.items()
                if k != FilterReasons.NO_FILTER
            ]
            pcts_str = "\n".join([f"\t{k.value}: {v}%" for k, v in pcts])
            logging.info(f"Types of messages filtered:\n{pcts_str}")

    def report_filtering_annotations(self):
        """
        When a message would have been filtered (by a filter that is not active), let's display this information in the dashboard for analysis
        """
        for msg, reason, extra_data in self.filtering_annotations:
            sender, recipient = msg[MessageObjectPart.SENDER], msg[MessageObjectPart.RECIPIENT]
            logging.info(
                f"Filter annotation from {sender} to {recipient}: {reason} EXTRA_INFO {json.dumps(extra_data)}"
            )
            meta_annotations.add_filtered_msg(
                (reason, json.dumps(extra_data)), msg[MessageObjectPart.TIME_SENT],
            )

        for msg, reason, _ in self.filtered_messages:
            logging.info(f"Filtered a message {json.dumps(msg)} for reason: {reason}")
            meta_annotations.add_filtered_msg(
                (reason, json.dumps(msg)), msg[MessageObjectPart.TIME_SENT],
            )

        # clear filter annotations after logging
        self.filtering_annotations = []
        self.filtered_messages = []

    def report_filtered_message(
        self, msg: MessageDict, reason: FilterReasons, extra_info: Dict = {}
    ):
        """
        Convenience method: When a message is filtered, let's display it in the dashboard for analysis
        """
        self.filtered_messages.append(
            (msg, f"Filtered message found (reason: {reason.value})", extra_info)
        )

    def should_filter_message(
        self,
        msg: MessageDict,
        curr_phase_messages: PhaseMsgs,
        game: Optional[pydipcc.Game] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
        game_is_missing_draw_votes: bool = False,
        timings: Optional[TimingCtx] = None,
    ) -> bool:
        """
        Maybe filter the message depending on criteria.

        Currently, filtering is done based on:
        - offensive language
        - exact message repeats in the same phase
        - consecutive short messages
        - too many redacted tokens
        - zero shot nonsense classifier

        We currently have logging for but do not filter based on:
        - nonsense classifiers
        - checking whether the message makes the pseudo orders more likely


        Returns a bool indicating whether the message should be filtered.
        """
        if timings is None:
            timings = TimingCtx()

        # do not filter draw messages
        timings.start("is_draw_msg")
        if is_draw_msg(msg) or is_unvote_draw_msg(msg):
            return False

        # filter offensive language
        if self.filter_offensive_language:
            with timings("simple_filter"):
                if self._contains_offensive_language(msg[MessageObjectPart.MESSAGE]):
                    self.statistics[FilterReasons.OFF_LANG] += 1
                    self.report_filtered_message(msg, FilterReasons.OFF_LANG)
                    return True

        # check for repeats
        if self.filter_phase_repeats:
            with timings("simple_filter"):
                if _contains_phase_repeat(msg, curr_phase_messages):
                    self.statistics[FilterReasons.PHASE_REPEAT] += 1
                    self.report_filtered_message(msg, FilterReasons.PHASE_REPEAT)
                    return True

        # check for consecutive short messages
        if self.filter_consecutive_short:
            with timings("simple_filter"):
                if _contains_consecutive_short(msg, curr_phase_messages):
                    self.statistics[FilterReasons.CONSEC_SHORT] += 1
                    self.report_filtered_message(msg, FilterReasons.CONSEC_SHORT)
                    return True

        # filter when messages contain too many redacted tokens
        if self.filter_excess_redacted:
            with timings("simple_filter"):
                if _contains_too_many_redacted(msg):
                    self.statistics[FilterReasons.REDACTED] += 1
                    self.report_filtered_message(msg, FilterReasons.REDACTED)
                    return True

        # filter when messages contain any redacted tokens
        if self.filter_any_redacted:
            with timings("simple_filter"):
                if _contains_redacted(msg):
                    self.statistics[FilterReasons.ANY_REDACTED] += 1
                    self.report_filtered_message(msg, FilterReasons.ANY_REDACTED)
                    return True

        # filter messages containing ampersands
        if self.filter_ampersands:
            with timings("simple_filter"):
                if _contains_ampersands(msg):
                    self.statistics[FilterReasons.AMPERSANDS] += 1
                    self.report_filtered_message(msg, FilterReasons.AMPERSANDS)
                    return True

        # filter messages containing names
        if self.filter_names:
            with timings("simple_filter"):
                if _contains_names(msg):
                    self.statistics[FilterReasons.NAMES] += 1
                    self.report_filtered_message(msg, FilterReasons.NAMES)
                    return True

        # filter messages containing urls and emails
        if self.filter_urls_emails:
            with timings("simple_filter"):
                if _contains_url_or_email(msg):
                    self.statistics[FilterReasons.URLS_OR_EMAILS] += 1
                    self.report_filtered_message(msg, FilterReasons.URLS_OR_EMAILS)
                    return True

        # filter for grounding issues
        if self.filter_grounding:
            with timings("simple_filter"):
                filter_matched = _should_filter_grounding(msg)
                if filter_matched:
                    self.statistics[FilterReasons.GROUNDING] += 1
                    self.report_filtered_message(
                        msg, FilterReasons.GROUNDING, {"matched": filter_matched}
                    )
                    return True

        # filter for insulting/rude things
        if self.filter_insults:
            with timings("simple_filter"):
                filter_matched = _should_filter_insults(msg)
                if filter_matched:
                    self.statistics[FilterReasons.INSULTS] += 1
                    self.report_filtered_message(
                        msg, FilterReasons.INSULTS, {"matched": filter_matched}
                    )
                    return True

        # maybe filter messages about draws
        if self.filter_draw_discussion_when_missing_votes and game_is_missing_draw_votes:
            with timings("simple_filter"):
                if _contains_draw(msg):
                    self.statistics[FilterReasons.MESSAGE_ABOUT_DRAWS] += 1
                    self.report_filtered_message(msg, FilterReasons.MESSAGE_ABOUT_DRAWS)
                    return True

        if self.filter_mutes:
            with timings("simple_filter"):
                if _contains_mute(msg):
                    self.statistics[FilterReasons.MUTES] += 1
                    self.report_filtered_message(msg, FilterReasons.MUTES)
                    return True

        # maybe filter grounding issues related to game ending year
        if (
            self.grounding_last_playable_year is not None
            and self.grounding_last_playable_year >= 1901
            and game is not None
        ):
            with timings("simple_filter"):
                if _should_filter_end_of_game_grounding(
                    game, self.grounding_last_playable_year, msg
                ):
                    self.statistics[FilterReasons.GROUNDING_END_OF_GAME] += 1
                    self.report_filtered_message(msg, FilterReasons.GROUNDING_END_OF_GAME)
                    return True

        # filter out nonsense messages detected by the zshot nonsense classifier
        if self.zshot_nonsense_classifier is not None:
            with timings("zshot_nonsense_classifier"):
                assert (
                    game is not None
                ), "Game must be provided if zshot_nonsense_classifier is set"
                if self._contains_zshot_nonsense(msg, game):
                    self.statistics[FilterReasons.ZSHOT_NONSENSE] += 1
                    self.report_filtered_message(msg, FilterReasons.ZSHOT_NONSENSE)
                    return True

        if self.ensemble_nonsense_classifier is not None:
            with timings("ensemble_nonsense_classifier"):
                assert game is not None, "Game must be provided if nonsense_classifier is set"
                (
                    is_nonsense,
                    verbose_nonsense_status,
                ) = self.ensemble_nonsense_classifier.get_verbose_nonsense_status(game, msg)
                if is_nonsense:
                    self.statistics[FilterReasons.NONSENSE] += 1
                    logging.info("Filtering message due to nonsense")
                    self.report_filtered_message(
                        msg, FilterReasons.NONSENSE, extra_info=verbose_nonsense_status
                    )
                    return True
                else:
                    # Add nonsense ensemble scores for all messages into filtering annotations so that we can view it in the viz tool
                    nonsense_probs = [
                        verbose_nonsense_status[c]["p_nonsense"]
                        for c in verbose_nonsense_status.keys()
                    ]
                    nonsense_probs = ",".join([f"{n:.2f}" for n in nonsense_probs])
                    self.filtering_annotations.append(
                        (
                            msg,
                            f"Nonsense Ensemble Scores:{nonsense_probs}",
                            verbose_nonsense_status,
                        )
                    )

        # filter out messages based on their correspondence to pseudo orders
        if self.pseudo_orders_correspondence_threshold is not None:
            with timings("pseudo_orders_correspondence"):
                assert (
                    self.orders_model is not None
                ), "Need an orders model for pseudo-orders correspondence lie filtering"
                assert (
                    pseudo_orders is not None
                ), "Pseudo orders must be provided if we want to filter on pseudo orders"
                assert (
                    game is not None
                ), "Game must be provided if we want to filter on pseudo orders"
                (corresponds_to_pseudo, extra_corr_info,) = self._corresponds_to_pseudo_orders(
                    msg, game, pseudo_orders,
                )

                if extra_corr_info:
                    self.filtering_annotations.append(
                        (
                            msg,
                            f"Pseudo correspondence diff: {extra_corr_info['diff']:.4g}",
                            extra_corr_info,
                        )
                    )
                if not corresponds_to_pseudo:
                    logging.info("Filtering message due to pseudo orders correspondence")
                    logging.info(f"Pseudo-orders: {pseudo_orders}")
                    logging.info(f"Message: {msg[MessageObjectPart.MESSAGE]}")

                    self.report_filtered_message(
                        msg, FilterReasons.PSEUDO_ORDERS, extra_info=extra_corr_info,
                    )
                    self.statistics[FilterReasons.PSEUDO_ORDERS] += 1
                    return True

        if self.rating_threshold_first_message < 1.0 or self.rating_threshold_other < 1.0:
            with timings("rating_threshold"):
                assert game is not None, "Game must be provided to filter on rating"
                if self._has_low_rating(game, msg):
                    self.statistics[FilterReasons.LOW_RATING] += 1
                    self.report_filtered_message(msg, FilterReasons.LOW_RATING)
                    return True

        self.statistics[FilterReasons.NO_FILTER] += 1
        return False

    def filter_messages(
        self,
        msg_lst: List[MessageDict],
        curr_phase_messages: PhaseMsgs,
        game: Optional[pydipcc.Game] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
        game_is_missing_draw_votes: bool = False,
    ):
        """
        Filter a list of messages.
        """
        new_msgs = []
        for msg in msg_lst:
            should_filter = self.should_filter_message(
                msg=msg,
                curr_phase_messages=curr_phase_messages,
                game=game,
                pseudo_orders=pseudo_orders,
                game_is_missing_draw_votes=game_is_missing_draw_votes,
            )
            if not should_filter:
                new_msgs.append(msg)

        if new_msgs:
            return new_msgs

        return None

    def _has_low_rating(self, game: pydipcc.Game, msg: MessageDict) -> bool:
        scores = []
        assert self.dialogue_model is not None
        actual_rating = self.dialogue_model.metadata["power_metadata"][
            msg[MessageObjectPart.SENDER]
        ]["rating"]
        is_first_message = game.current_short_phase == "S1901M" and not any(
            [
                m[MessageObjectPart.SENDER] == msg[MessageObjectPart.SENDER]
                for m in game.messages.values()
            ]
        )

        if not is_first_message and self.rating_threshold_other == 1.0:
            logging.info(
                "Not first message, and self.rating_threshold_other == 1.0, so skipping computing rating posteriors"
            )
            return False
        logging.info(f"Computing rating likelihoods for message: {msg}")
        for r in [1, 2, 3, 4, 5]:
            self.dialogue_model.metadata["power_metadata"][msg[MessageObjectPart.SENDER]][
                "rating"
            ] = r
            with torch.no_grad():
                scores.append(
                    float(
                        self.dialogue_model.score_candidate_messages(
                            game,
                            [msg[MessageObjectPart.MESSAGE]],
                            msg[MessageObjectPart.SENDER],
                            msg[MessageObjectPart.TIME_SENT],
                            msg[MessageObjectPart.RECIPIENT],
                            skip_end_token=False,
                        )[0][1]
                    )
                )

        self.dialogue_model.metadata["power_metadata"][msg[MessageObjectPart.SENDER]][
            "rating"
        ] = actual_rating
        probs = torch.tensor(scores).softmax(dim=-1)
        logging.info("Posterior over ratings: " + str(probs.tolist()))
        if is_first_message and probs[0] > self.rating_threshold_first_message:
            logging.info(
                f"Likelihood of being by low-rated player ({probs[0]}) above first message threshold ({self.rating_threshold_first_message})"
            )
            return True
        elif not is_first_message and probs[0] > self.rating_threshold_other:
            logging.info(
                f"Likelihood of being by low-rated player ({probs[0]}) above threshold ({self.rating_threshold_other})"
            )
            return True
        return False

    def _contains_offensive_language(self, msg: str):
        """
        Optionally check messages for offensive language

        Only checks if attribute `check_offensive_language` is True. Messages containing
        offensive language are removed from the data.
        """
        if not msg:
            # message is an empty string
            return False

        if msg in self.osm:
            logging.info(f"Bot message flagged by offensive word list: {msg}")
            return True

        return False

    def _contains_zshot_nonsense(self, msg: MessageDict, game: pydipcc.Game) -> bool:
        """
        Check if message contains nonsense using the zero shot nonsense classifier
        """
        game_view_of_sender = game_from_view_of(game, msg[MessageObjectPart.SENDER])
        assert self.zshot_nonsense_classifier is not None
        is_nonsense = self.zshot_nonsense_classifier.get_nonsense_status(
            game=game_view_of_sender, potential_msg=msg, add_eom_token=False, skip_prefix=True,
        )
        if is_nonsense:
            return True

        return False

    def _contains_nonsense(self, msg: MessageDict, game: pydipcc.Game) -> bool:
        """
        Check if message contains nonsense using the nonsense classifier
        """
        game_view_of_sender = game_from_view_of(game, msg[MessageObjectPart.SENDER])
        assert self.ensemble_nonsense_classifier is not None
        is_nonsense = self.ensemble_nonsense_classifier.get_nonsense_status(
            game=game_view_of_sender, potential_msg=msg
        )
        return is_nonsense

    def _corresponds_to_pseudo_orders(
        self, msg: OutboundMessageDict, game: pydipcc.Game, pseudo_orders: PseudoOrders,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Computes the log likelihood of the agent pseudo-orders under the orders model
        before and after the message.

        The provided orders model should be trained on phase dialogue prefixes
        to be "sound".

        - msg: The potential generated message
        - game: game object
        - pseudo_orders: Joint action used to condition the dialogue model
        """
        if self.pseudo_orders_correspondence_threshold is None:
            return True, {}

        if not [m for p in game.get_all_phases() for m in p.messages.values()]:
            # If there are no messages so far this game, bail
            # This is because the "before" state will assume a no-press game which
            # induces a radically different policy, so these pseudo order correspondence
            # checks will randomly pass or fail
            logging.info(f"Disabled PO correspondence because no messages yet.")
            return True, {}

        assert (
            self.orders_model is not None
        ), "Need an orders model for pseudo-orders correspondence lie filtering"
        assert game is not None, "Game must be provided if we want to filter on pseudo orders"

        assert self.orders_model is not None
        sender = msg[MessageObjectPart.SENDER].upper()
        recipient = msg[MessageObjectPart.RECIPIENT].upper()
        msg_body = msg[MessageObjectPart.MESSAGE]
        game_view_of_sender = game_from_view_of(game, sender)
        msg_dict = with_time_sent(
            msg, increment_last_message_time(game_view_of_sender, Timestamp.from_seconds(10))
        )
        game_view_of_sender_with_msg = add_message_to_game_copy(  # add message to the history to get a pseudo order now
            game=game_view_of_sender,
            phase=msg[MessageObjectPart.PHASE],
            sender=sender,
            recipient=recipient,
            body=msg_body,
            time_sent=msg_dict["time_sent"],
        )
        sender_pseudo = pseudo_orders.first_joint_action()[sender]

        def get_pseudo_logprob(game):
            return self.orders_model.score_candidate_actions(
                game=game, candidates=[sender_pseudo], view_of_power=sender, target_power=sender,
            )[0][1]

        before_logprob = get_pseudo_logprob(game_view_of_sender)
        after_logprob = get_pseudo_logprob(game_view_of_sender_with_msg)
        before_prob = math.exp(before_logprob)
        after_prob = math.exp(after_logprob)
        diff = after_prob - before_prob
        logging.info(
            f"Pseudo correspondence: prob before= {before_prob:.4g} after= {after_prob:.4g} diff= {diff:.4g}"
        )
        extra_info = {
            "before_prob": before_prob,
            "after_prob": after_prob,
            "diff": diff,
            "thresh": self.pseudo_orders_correspondence_threshold,
            "pseudo_orders": sender_pseudo,
        }

        corresponds_to_pseudo = diff >= self.pseudo_orders_correspondence_threshold
        return corresponds_to_pseudo, extra_info

    def _edit_newlines(self, msg_txt: str) -> str:
        """
        Corrupted new line characters appear as "~N~" in text.

        We replace with "\n" for readability.
        """
        return uncorrupt_newlines(msg_txt)


class MessageEditing:
    """
    Class that handles editing messages with various helper methods.
    """

    def __init__(
        self, edit_newlines: bool, edit_names: bool, edit_weird_capitalization: bool,
    ):
        # editing
        self.edit_newlines = edit_newlines
        self.edit_names = edit_names
        self.edit_weird_capitalization = edit_weird_capitalization

    def maybe_edit_message(self, msg: MessageDict,) -> MessageDict:
        """
        Maybe edit the message.
        """
        if self.edit_newlines or self.edit_names or self.edit_weird_capitalization:
            msg_txt = msg[MessageObjectPart.MESSAGE]

            if self.edit_newlines:
                msg_txt = _edit_newlines(msg_txt)
            if self.edit_names:
                msg_txt = _edit_names(msg_txt)
            if self.edit_weird_capitalization:
                msg_txt = _edit_weird_capitalization(msg_txt)

            if msg_txt != msg[MessageObjectPart.MESSAGE]:
                new_msg = copy.deepcopy(msg)
                new_msg[MessageObjectPart.MESSAGE] = msg_txt
                msg = new_msg
        return msg

    def edit_messages(
        self, msg_lst: List[MessageDict],
    ):
        """
        Edit a list of messages.
        """
        return [self.maybe_edit_message(msg) for msg in msg_lst]


def _contains_phase_repeat(msg: MessageDict, curr_phase_messages: PhaseMsgs) -> bool:
    """
    Return true or false corresponding to whether the
    sender sent the *exact same* message content to the same
    or another person *during this phase*.

    NOTE: we may want to only filter repeats in a row,
    or repeats sent to the same player.
    """
    sender = msg[MessageObjectPart.SENDER].upper()
    msg_txt = msg[MessageObjectPart.MESSAGE]
    prev_sender_messages = [
        x[MessageObjectPart.MESSAGE]
        for x in curr_phase_messages
        if x[MessageObjectPart.SENDER] == sender
    ]
    # NOTE: might want to do some sort of fuzzy match here;
    # leaving it as exact match for now
    if msg_txt in prev_sender_messages:
        logging.info(f"Repeat detected from {sender}: {msg_txt}")
        return True

    return False


def _contains_consecutive_short(
    msg: MessageDict, curr_phase_messages: PhaseMsgs, short_threshold=20
) -> bool:
    """
    Block short messages from a player if the previous message was also short, to reduce feedback loops. "Short" by default is 20 characters. This is ~2% of human messages, disproportionaly those with lower ratings.
    """
    if len(msg[MessageObjectPart.MESSAGE]) >= short_threshold:
        return False
    sender = msg[MessageObjectPart.SENDER]
    recipient = msg[MessageObjectPart.RECIPIENT]
    last_sent_msg = None
    for prev_msg in reversed(list(curr_phase_messages)):
        if (
            prev_msg[MessageObjectPart.SENDER] == sender
            and prev_msg[MessageObjectPart.RECIPIENT] == recipient
        ):
            last_sent_msg = prev_msg[MessageObjectPart.MESSAGE]
            break
    if last_sent_msg is None or len(last_sent_msg) < short_threshold:
        # This also blocks short messages from being the first in the phase, which is probably helpful
        logging.info(
            f'Filtering consecutive short message from {sender}: "{msg[MessageObjectPart.MESSAGE]}" Previous was: "{last_sent_msg}"'
        )
        return True
    return False


def _contains_too_many_redacted(msg: MessageDict, threshold: float = 0.2) -> bool:
    """
    Checks to see whether >20% of a message is comprised of redacted tokens.

    Currently, ~5% of the train set has >20% of tokens redacted.
    """
    text = msg[MessageObjectPart.MESSAGE]
    num_redacted_toks = len(re.findall(r"\[\d+\]", text))
    total_toks = len(text.split(" "))  # rough approximation of number of tokens
    pct_redacted = num_redacted_toks / total_toks

    return pct_redacted > threshold


def _contains_redacted(msg: MessageDict) -> bool:
    """
    Return whether the word 'redacted' appears in the message, with any capitalization
    or square brackets.
    """
    text = msg[MessageObjectPart.MESSAGE]
    if "redacted" in text.lower():
        return True

    if "[" in text and "]" in text:
        return True

    return False


def _contains_ampersands(msg: MessageDict) -> bool:
    """
    Check to see whether a message contains ampersands.

    In the training data, messages with ampersands are abundant because
    the characters $, *, ^, `, {, |, } have all seemingly been converted to
    ampersands.
    """
    text = msg[MessageObjectPart.MESSAGE]
    return "&" in text


NAME_SEPARATOR_PATTERN = re.compile("[^-a-zA-Z0-9_]+")
NAME_PATTERN = re.compile("[-a-zA-Z0-9_]+")


def _contains_names(msg: MessageDict) -> bool:
    """
    Check to see whether a message contains common names of people.
    """
    text = msg[MessageObjectPart.MESSAGE]
    # Explicitly include hyphens as part of the word, to help avoid filtering
    # expressions like round-robin, or hail-mary, or scott-free.
    for word in re.split(NAME_SEPARATOR_PATTERN, text.lower()):
        if word in common_names.common_names:
            return True
    return False


def _edit_names(msg_txt: str) -> str:
    """
    Redact common names of people within a message.
    """
    bad_name_ranges = []
    for match in re.finditer(NAME_PATTERN, msg_txt):
        word = match.group(0)
        if word.lower() in common_names.common_names:
            bad_name_ranges.append((match.start(), match.end()))
    if bad_name_ranges:
        new_msg_txt_pieces = []
        msg_is_good_start = 0
        for (bad_left, bad_right) in bad_name_ranges:
            new_msg_txt_pieces.append(msg_txt[msg_is_good_start:bad_left])
            new_msg_txt_pieces.append("[1]")
            msg_is_good_start = bad_right
        if msg_is_good_start < len(msg_txt):
            new_msg_txt_pieces.append(msg_txt[msg_is_good_start:])
        return "".join(new_msg_txt_pieces)
    else:
        return msg_txt


WEIRD_CAP_PATTERN = re.compile(r"\b[a-z]+-[a-z]*[A-Z]+\b")


def _edit_weird_capitalization(msg_txt: str) -> str:
    """
    Webdip has an oddity where its redaction algorithm also modifies the text,
    converting things like "anti-English" to "anti-eNGLISH",
    or anti-Mediterranean -> anti-mediTERRANEAN or
    anti-Italy -> anti-ITALY

    Attempt to fix this.
    """
    bad_cap_replacements = []
    for match in re.finditer(WEIRD_CAP_PATTERN, msg_txt):
        word = match.group(0)
        pieces = word.split("-")
        assert len(pieces) == 2
        bad_cap_replacements.append(
            (match.start(), match.end(), pieces[0] + "-" + pieces[1].capitalize())
        )
    if bad_cap_replacements:
        new_msg_txt_pieces = []
        msg_is_good_start = 0
        for (bad_left, bad_right, replacement) in bad_cap_replacements:
            new_msg_txt_pieces.append(msg_txt[msg_is_good_start:bad_left])
            new_msg_txt_pieces.append(replacement)
            msg_is_good_start = bad_right
        if msg_is_good_start < len(msg_txt):
            new_msg_txt_pieces.append(msg_txt[msg_is_good_start:])
        return "".join(new_msg_txt_pieces)
    else:
        return msg_txt


# Detects URLs with a "://". Does NOT require the protocol in front since webdip redacts that
# even when it doesn't redact the rest of the URL.
URLPATTERN = re.compile(r"://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
# Detects some common domains with a "." and a "/". Doesn't require the :// so we can't make this
# too aggressive or else we'll have false positives when peopule use slashes and don't put spaces
# after their periods.
URLPATTERN2 = re.compile(r"[-a-zA-Z0-9]+\.(com|org|edu|net|uk|ru|au|de)/")
# Detects some email addresses
EMAILPATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _contains_url_or_email(msg: MessageDict) -> bool:
    """
    Check to see whether a message contains common names of people.
    """
    text = msg[MessageObjectPart.MESSAGE]
    if (
        re.search(URLPATTERN, text)
        or re.search(URLPATTERN2, text)
        or re.search(EMAILPATTERN, text)
    ):
        return True
    return False


def _contains_draw(msg: MessageDict) -> bool:
    """
    Check to see whether a message contains the word "draw".

    In certain circumstances, we may want to filter messages about draws or mark
    them as "BAD", such as when the game does not contain any draw info
    """
    text = msg[MessageObjectPart.MESSAGE]
    return " draw" in text.lower()


def _edit_newlines(msg_txt: str) -> str:
    """
    Corrupted new line characters appear as "~N~" in text.

    We replace with "\n" for readability.
    """
    return uncorrupt_newlines(msg_txt)


def _contains_mute(msg: MessageDict) -> bool:
    """
    In our training data there are a bunch copies of a message that webdip sends
    when a player tries to talk to another player that muted them. We obviously
    want to filter our bot from sending these.
    """
    text = msg[MessageObjectPart.MESSAGE]
    return "this country has muted you" in text.lower()


def _should_filter_grounding(msg: MessageDict) -> Optional[str]:
    """Filter for keywords based on grounding issues"""
    m = msg["message"].lower()
    substrs = [
        "afternoon",
        "backstabbr",
        "banned",
        "beginner",
        "cancel",
        "cd",
        "cheater",
        "cheating",
        "christmas",
        "deadline",
        "discord",
        "disorder",
        "drop",
        "email",
        "evening",
        "first game",
        "first time",
        "1st game",
        "1st time",
        "forum",
        "game histor",
        "game stat",
        "girlfriend",
        "gmail",
        "hacked",
        "hacker",
        "hacking",
        "holiday",
        "hour",
        "inactiv",
        "internet",
        "last game",
        "league",
        "mobile",
        "morning",
        "next game",
        "nmr",
        "old game",
        "older game",
        "online",
        "password",
        "pause",
        "pausing",
        "phone",
        "playdiplom",
        "points",
        "profile",
        "previous game",
        "prior game",
        # ghost rating, monthly rating, reliability rating.
        # Leading space avoids false-positiving "cooperating", "frustrating", etc.
        " rating",
        "ranked",
        "ranking",
        "replac",
        "resign",
        "second game",
        "second time",
        "2nd game",
        "2nd time",
        "sleep",
        "stakes",
        "submit",
        "third game",
        "third time",
        "3rd game",
        "3rd time",
        "time zone",
        "timezone",
        "tomorrow",
        "tonight",
        "tournament",
        "username",
        "vacation",
        "wager",
        "webdip",
        "website",
        "week",
        "xmas",
        "x-mas",
        "yesterday",
        # Cannot just block the substring "account" due to phrases like "On account of"
        # phrases like or "accounting for", but we can still block some common phrases about
        # people talking about accounts.
        "multi-account",
        "multi account",
        "multiple account",
        "new account",
        "old account",
        "other account",
        "one account",
        "real account",
        "this account",
        "user account",
        # Weekdays
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        # Months
        "january",
        "february",
        # "march", "march" is sometimes a verb in Diplomacy for army movement.
        "april",
        # "may",  "may" is too common a word to block.
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    regexes = [
        r"[^a-z]days?[^a-z]",
        r"\d+ ?[ap]\.?m",
        r"enter.{0,16}(order|mov)",
        r"(order|mov).{0,16}enter",
        r"put.{0,16}(order|mov)",
        r"(order|mov).{0,16}put",
        r"miss.*turn",
        r"new .{0,16}(austria|england|france|germany|italy|russia|turkey)",
        r"take.{0,16}over.{0,16}(austria|england|france|germany|italy|russia|turkey)",
        r"[^a-z]ready",
        r"^ready",
        r"[0-9] point",
        r"\bsave",
        r"\bbed",
        r"\b[1-2]?[0-9]:[0-9][0-9]",  # times, like 12:30
        r"[1-2]?[0-9] ?(am|pm)\b",  # times, like 6 pm
        r"[0-9] minute",  # times, like 45 minutes
    ]

    # do the checks
    for x in substrs:
        if x in m:
            return x
    for r in regexes:
        if re.search(r, m):
            return r
    return None


def _should_filter_insults(msg: MessageDict) -> Optional[str]:
    """Filter for things that might be insulting or rude language"""
    m = msg["message"].lower()
    substrs = [
        "abomination",
        "airhead",
        "asshat",
        "asshead",
        "asshol",
        "birdbrain",
        "bird-brain",
        "bird brain",
        "blockhead",
        "bullcrap",
        "chump",
        "clown",
        "crackbrain",
        "crackhead",
        "crackpot",
        "dick",
        "dimwit",
        "dingus",
        "disgrace",
        "disgusting",
        "dolt",
        "doofus",
        "dope",
        "dumb",
        "dummy",
        "dweeb",
        "fuck",
        "halfwit",
        "half-wit",
        "idiot",
        "ignoramus",
        "imbecil",
        "incompetent",
        "inept",
        "jackass",
        "jerk",
        "kill yourself",
        "knob",
        "liar",
        "loser",
        "lunatic",
        "mediocre",
        "moron",
        "newbie",
        "nimrod",
        "nincompoop",
        "ninny",
        "nitwit",
        "noob",
        "numskull",
        "numbskull",
        "nutcase",
        "pawn",
        "pissed",
        "pitiful",
        "pleb",
        "psychopath",
        "pushover",
        "putz",
        "pwned",
        "schizo",
        "schitzo",
        "schmuck",
        "screw you",
        "shit",
        "shut up",
        "sissy",
        "stupid",
        "stfu",
        "tool",
        "troll",
        "tyke",
        "weakling",
        "whiner",
        "wtf",
    ]
    regexes = [
        r"\blousy",
        r"\bpathetic",
        r"\bcrap",
        r"\bhalf wit\b",
        r"\bhead case\b",
        r"[-a-z]sucker",
        r"sucking",
        r"psycho\b",
        r"\bretard(ed)?\b",
        r"(bad|worst|horrible|terrible|foolish|poor|weak|novice|new|naive|worthless).{0,16}(player|person|boy|girl|ally)",
        r"\b(you|he|she|they|austria|england|france|germany|italy|turkey|russia).{0,30}(bad|weak|poor|terrible|horrible|worst|awful).{0,10} at.{0,10}(playing|game|diplomacy)\b",
        r"you.{0,20}how to play",
        r"you.{0,36}ridiculous",
        r"ridiculous.{0,25}you",
        r"awful (player|person)",
    ]

    # do the checks
    for x in substrs:
        if x in m:
            return x
    for r in regexes:
        if re.search(r, m):
            return r

    r = r"\b(you)((.{0,20}are)|r|'re) .{0,25}(terrible|horrible|worst|foolish|fool|looney|loony|freaking)\b"
    rnot = r"\b(you)((.{0,20}are)|r|'re).{0,20}(\bnot|n't)\b.{0,15}(terrible|horrible|worst|foolish|fool)\b"
    if re.search(r, m) and not re.search(rnot, m):
        return r

    return None


def _should_filter_end_of_game_grounding(
    game: pydipcc.Game, grounding_last_playable_year: int, msg: MessageDict
) -> Optional[str]:
    """Filter for some words or phrases that are likely to reveal that the model is ignorant
    of the game having a predetermined end_year"""

    m = msg["message"].lower()

    # always block discussion of years after the game ends
    substrs = list(map(str, range(grounding_last_playable_year + 1, 2000)))

    # block several substrings if the game is near completion
    if game.current_year >= grounding_last_playable_year - 2:
        substrs.extend(
            ["solo", "draw",]
        )

    if game.current_year >= grounding_last_playable_year - 1:
        substrs.extend(
            [
                "long-term",
                "long term",
                "longterm",
                "longer term",
                "longer-term",
                "in the future",
                "eventually",
                "years",
            ]
        )

    if game.current_year >= grounding_last_playable_year:
        substrs.extend(
            ["next year", "next spring", "later", "build", "seasons", "phases", "turns",]
        )

    if game.current_year >= grounding_last_playable_year and (
        game.current_short_phase.startswith("F") or game.current_short_phase.startswith("W")
    ):
        substrs.extend(
            ["next fall", "next autumn", "next turn", "next season", "next phase", "the spring",]
        )

    # do the checks
    for x in substrs:
        if x in m:
            return x
    return None
