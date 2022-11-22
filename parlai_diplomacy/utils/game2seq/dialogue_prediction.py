#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Format helper for dialogue prediction
"""

from typing import List, Optional, Tuple
import parlai.utils.logging as logging
import random
import re

from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.typedefs import CurrentDrawState, GameJson, Power, Phase, Timestamp
from fairdiplomacy.pseudo_orders import PseudoOrders

from parlai_diplomacy.tasks.common_task_utils import CORRUPTED, REAL
from parlai_diplomacy.tasks.discriminator.change_entity_utils import CONJUNCTIONS_AND_PUNCTUATION
from parlai_diplomacy.utils.game2seq import input_validation
from parlai_diplomacy.utils.game2seq.typing import (
    DialogueSequencePart,
    DialoguePredictionOutput,
    MessageDict,
    MsgHistoryList,
    TrainingDialoguePredictionOutput,
    Metadata,
    PhaseMsgs,
    FlatState,
    FlatOrders,
)
from parlai_diplomacy.utils.game2seq.base_prediction import BaseDiplomacyPredictionFormatter
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryUnflattener,
    MessageObjectPart,
    concate_msghistory_with_curmsg,
    get_elapsed_time,
    get_gamejson_draw_state,
    get_last_timestamp_gamejson,
    get_last_timestamp_msg_history,
    is_draw_msg,
    is_unvote_draw_msg,
)
from parlai_diplomacy.utils.game2seq.format_helpers.message_editing import (
    MessageEditing,
    MessageFiltering,
)
import parlai_diplomacy.utils.game2seq.format_helpers.orders as order_helper
import parlai_diplomacy.utils.game2seq.format_helpers.state as state_helper
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    COUNTRY_ID_TO_POWER,
    ParlaiDecodingError,
    add_recipient_to_prompt_token,
    modify_input_prompt_for_power,
    format_player_prompt_token,
    get_example_key,
)
from parlai_diplomacy.utils.game2seq.format_helpers.opt_utils import (
    expects_bilateral_pseudo_orders,
    expects_rollout_type,
)


class DialoguePredictionFormatter(BaseDiplomacyPredictionFormatter):
    """
    Dialogue sequence formatter for INFERENCE.

    Note that this is different from training (see below) as during inference we use the entire
    dialogue history available, rather than iteratively building up examples.
    """

    def _is_train_time(self) -> bool:
        """Overridden in TrainDialoguePredictionFormatter"""
        return False

    def _check_incompatible_opt(
        self, metadata: Metadata, format_parts: List[DialogueSequencePart]
    ) -> None:
        """
        check incompatible opt in dialogue teachers
        """
        if metadata["opt"].get("response_view_dialogue_model", False):
            assert not metadata["opt"].get(
                "pseudo_order_generation", False
            ), "response view dialogue model only support non-pseudo-order teachers"
            assert (
                DialogueSequencePart.PSEUDO_ORDERS not in format_parts
            ), "response view dialogue model only support non-pseudo-order teachers"
            assert (
                DialogueSequencePart.ELAPSED_TIME not in format_parts
            ), "response view dialogue model only support non-pseudo-order teachers"

    def get_format_parts(self, fmt: str) -> List[DialogueSequencePart]:
        """
        Override from parent class
        """
        fmt_str = fmt.replace("message_history", "messagehistory")
        fmt_keys = {
            "messagehistory": DialogueSequencePart.HISTORY,
            "state": DialogueSequencePart.STATE,
            "lastorder": DialogueSequencePart.LAST_PHASE_ORDER,
            "lastmovementorder": DialogueSequencePart.LAST_MOVEMENT_ORDER,
            "orderhistorysincelastmovementphase": DialogueSequencePart.ORDER_HISTORY_SINCE_LAST_MOVE,
            "actualorders": DialogueSequencePart.ACTUAL_ORDERS,
            "pseudoorder": DialogueSequencePart.PSEUDO_ORDERS,
        }

        fmt_parts = [fmt_keys[x] for x in fmt_str.split("_")]

        return fmt_parts

    def generate_input_output_pairs(
        self,
        game_json: GameJson,
        metadata: Metadata,
        format_parts: List[DialogueSequencePart],
        speaker: Power,
        recipient: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> DialoguePredictionOutput:
        """
        Generates the dialogue input sequence on the requested format.

        Receives the game data via json, speaker, and metadata parameters.
        Outputs the formatted message and orders history as dictionary,
        according to extras parameter.

        Formatted messages is formed from a list of of DialogueSequencePart enums
        that are passed in by `format_parts`

        For example if
        format_parts=[DialogueSequencePart.STATE, DialogueSequencePart.PSEUDO_ORDERS], then

        <state> <pseudoorder> <player_prompt_token>

        For default format_parts=None we have

        <player_prompt_token>
        """
        if not format_parts:
            # to skip the iterations on for loop
            format_parts = []

        if metadata["opt"].get("add_sleep_times", False) or "sleep" in metadata["opt"].get(
            "task", ""
        ):
            # add elapsed time to the input
            format_parts.append(DialogueSequencePart.ELAPSED_TIME)

        seqs = {}
        phase = sorted(list(game_json.keys()), key=lambda x: sort_phase_key(x))[-1]
        seqs[phase] = {}
        formatted_str = ""
        for fmt_part in format_parts:
            format_part_str = self._format_part(
                fmt_part,
                game_json,
                speaker,
                phase,
                metadata,
                timestamp=timestamp,
                recipient=recipient,
            )
            if not formatted_str:
                formatted_str = format_part_str
            else:
                formatted_str += f"{self.delimiter}{format_part_str}"

        player_prompt_token = self._maybe_edit_prompt_token(
            format_player_prompt_token(phase, speaker, metadata), speaker, recipient, metadata,
        )
        seqs[phase]["input"] = f"{formatted_str}{self.delimiter}{player_prompt_token}"

        return seqs

    def _maybe_edit_prompt_token(
        self,
        curr_prompt_token: str,
        sender: Power,
        recipient: Optional[Power],
        metadata: Metadata,
    ) -> str:
        """
        Left overridable by child classes. Potentially edit the prompt token.

        Currently, if `--add-recipient-to-prompt` is True, the recipient is appended to the prompt token.
        """
        if metadata["opt"].get("add_recipient_to_prompt", False):
            assert (
                recipient is not None
            ), "--add-recipient-to-prompt is True for dialogue model, must know recipients a priori"
            curr_prompt_token = add_recipient_to_prompt_token(
                curr_prompt_token, sender, recipient, metadata, self.version,
            )
        elif "sleepsix" in metadata["opt"]["task"] and not self._is_train_time():
            # at train time we modify prompt manually because we create
            # multiple examples per base example
            assert recipient is not None
            curr_prompt_token = modify_input_prompt_for_power(
                curr_prompt_token, recipient, self.version
            )

        return curr_prompt_token

    def _get_dialogue_state(
        self, game_json: GameJson, phase: Phase, metadata: Metadata
    ) -> FlatState:
        num_movement_phases = metadata["opt"].get(
            "extend_state_history_since_last_n_movement_phase", 0
        )
        assert isinstance(num_movement_phases, int), (
            num_movement_phases,
            type(num_movement_phases),
        )

        if num_movement_phases > 0:
            state_history_dct = state_helper.build_state_history_dct(game_json, phase)

            return self.state_flattener.flatten_state_since_last_n_movement_phases(
                num_movement_phases,
                state_history_dct,
                short_version=metadata.get("shortstate", False),
                opt=metadata["opt"],
            )
        else:
            return self.state_flattener.flatten_state(
                game_json[phase]["state"],
                phase,
                short_version=metadata.get("shortstate", False),
                opt=metadata["opt"],
            )

    def _filter_orders(self, str_orders: FlatOrders, relevant_powers: List[Power]):
        power_to_order_str = {x.split(":")[0].upper(): x for x in str_orders.split("\n")}
        if self.version <= 1:
            str_orders = "\n".join([power_to_order_str.get(x, "") for x in relevant_powers])
        else:
            # In V2, we still need to represent empty orders so that the recipient is visible
            str_orders = "\n".join([power_to_order_str.get(x, f"{x}: ") for x in relevant_powers])
        return str_orders

    def _get_inference_pseudo_orders(
        self, phase: Phase, speaker: Power, metadata: Metadata, recipient: Optional[Power] = None,
    ) -> FlatOrders:
        assert (
            "pseudo_orders" in metadata
            and phase in metadata["pseudo_orders"]
            and speaker in metadata["pseudo_orders"][phase]
        ), f"Pseudo orders not found for {speaker} in {phase}"
        pseudo_orders: PseudoOrders = metadata["pseudo_orders"][phase][speaker]
        model_expects_rollout_pseudo_orders = metadata["opt"].get("rollout_pseudo_orders", False)
        assert pseudo_orders.check_rollout(expects_rollout_type(metadata["opt"]))

        # Rollout pseudo orders
        if model_expects_rollout_pseudo_orders:
            assert metadata["opt"].get(
                "single_view_pseudo_orders", False
            ), "Must have single view with rollout pseudo orders"
            assert recipient is not None
            str_orders = self.orders_flattener.flatten_rollout_joint_action_bilateral_powermajor(
                pseudo_orders.val, speaker, recipient
            )
        else:
            # Regular pseudo orders
            str_orders = self.orders_flattener.flatten_joint_action(
                pseudo_orders.first_joint_action(), speaker
            )
            if str_orders and expects_bilateral_pseudo_orders(metadata["opt"]):
                # Only include pseudo orders for the sender and recipients
                assert (
                    recipient is not None
                ), "Must include recipients --all-power-pseudo-orders is False"
                relevant_powers = [recipient, speaker]
                str_orders = self._filter_orders(str_orders, relevant_powers)

        return str_orders

    def _get_actual_orders(self, phase: Phase, speaker: Power, metadata: Metadata) -> FlatOrders:
        if metadata["opt"].get("rollout_actual_orders", False):
            return self.orders_flattener.flatten_rollout_action(
                metadata["actual_orders"][phase][speaker]
            )
        else:
            return self.orders_flattener.flatten_only_first_action_of_rollout_action(
                metadata["actual_orders"][phase][speaker]
            )

    def _get_last_movement_order(self, game_json: GameJson, phase: Phase) -> FlatOrders:
        order_history_dct = order_helper.build_order_history_dct(game_json, phase)
        return self.orders_flattener.flatten_last_movement_phase_order(order_history_dct)

    def _get_last_phase_order(self, game_json: GameJson, phase: Phase) -> FlatOrders:
        order_history_dct = order_helper.build_order_history_dct(game_json, phase)
        return self.orders_flattener.flatten_last_phase_order(order_history_dct)

    def _get_order_history_since_last_move(
        self, game_json: GameJson, phase: Phase, metadata: Metadata
    ) -> FlatOrders:

        order_history_dct = order_helper.build_order_history_dct(game_json, phase)

        maybe_num_movement_phases = metadata["opt"].get(
            "extend_order_history_since_last_n_movement_phase", 1
        )
        assert isinstance(maybe_num_movement_phases, int), (
            maybe_num_movement_phases,
            type(maybe_num_movement_phases),
        )

        return self.orders_flattener.flatten_order_history_since_last_n_movement_phases(
            order_history_dct, maybe_num_movement_phases,
        )

    def _get_history(
        self,
        game_json: GameJson,
        phase: Phase,
        speaker: Power,
        recipient: Optional[Power],
        metadata: Metadata,
    ) -> str:
        """
        Extract and flatten message history from game JSO
        """
        truncation = metadata["opt"].get("message_history_truncation", 2048)
        two_party_dialogue = metadata["opt"].get("two_party_dialogue", False)
        if "2person_dialogue" in metadata["opt"]:
            # Deprecated argument
            two_party_dialogue = metadata["opt"]["2person_dialogue"]

        no_speaker_dialogue = metadata["opt"].get("no_speaker_dialogue_history", False)

        if two_party_dialogue or no_speaker_dialogue:
            truncation = truncation * 100  # Effectively do not truncate the message history

        message_history = self.messagehistory_builder.extract_message_history_from_game_json(
            game_json, phase, speaker, truncation=truncation,
        )
        last_timestamp = get_last_timestamp_gamejson(game_json)
        draw_state = get_gamejson_draw_state(game_json, last_timestamp, metadata)

        if two_party_dialogue:
            assert recipient, f"recipient must be set"
            message_history = self.messagehistory_builder._to_2person_dialogue(
                message_history, speaker, recipient
            )
        if metadata["opt"].get("no_speaker_dialogue_history", False):
            message_history = self.messagehistory_builder.to_no_speaker_dialogue(
                message_history, speaker
            )

        if no_speaker_dialogue:
            message_history = self.messagehistory_builder.to_no_speaker_dialogue(
                message_history, speaker
            )

        return self.messagehistory_flattener.flatten_message_history(
            message_history, draw_state, metadata=metadata
        )

    def _get_elapsed_time(
        self, game_json: GameJson, phase: Phase, speaker: Power, timestamp: Timestamp,
    ) -> str:
        """
        Return a stringified version of the elapsed time.
        """
        # time the last message was sent (which is visible to the speaker)
        message_history = self.messagehistory_builder.extract_message_history_from_game_json(
            game_json, phase, speaker,
        )

        return get_elapsed_time(timestamp, message_history)

    def _format_part(
        self,
        fmt_part: DialogueSequencePart,
        game_json: GameJson,
        speaker: Power,
        phase: Phase,
        metadata: Metadata,
        timestamp: Optional[Timestamp] = None,
        recipient: Optional[Power] = None,
    ) -> str:
        if fmt_part == DialogueSequencePart.HISTORY:
            return self._get_history(game_json, phase, speaker, recipient, metadata)
        elif fmt_part == DialogueSequencePart.STATE:
            return self._get_dialogue_state(game_json, phase, metadata)
        elif fmt_part == DialogueSequencePart.PSEUDO_ORDERS:
            return self._get_inference_pseudo_orders(phase, speaker, metadata, recipient=recipient)
        elif fmt_part == DialogueSequencePart.LAST_MOVEMENT_ORDER:
            return self._get_last_movement_order(game_json, phase)
        elif fmt_part == DialogueSequencePart.LAST_PHASE_ORDER:
            return self._get_last_phase_order(game_json, phase)
        elif fmt_part == DialogueSequencePart.ORDER_HISTORY_SINCE_LAST_MOVE:
            return self._get_order_history_since_last_move(game_json, phase, metadata)
        elif fmt_part == DialogueSequencePart.ELAPSED_TIME:
            if "sleep" in metadata["opt"].get("task", ""):
                return "0"
            else:
                assert timestamp is not None
                return self._get_elapsed_time(game_json, phase, speaker, timestamp)
        elif fmt_part == DialogueSequencePart.ACTUAL_ORDERS:
            return self._get_actual_orders(phase, speaker, metadata)
        else:
            raise RuntimeError(f'Unrecognized item "{fmt_part}" in dialogue format request.')


class TrainingDialoguePredictionFormatter(DialoguePredictionFormatter):
    """
    Format helper for dialogue during TRAINING.

    This requires a format slightly different than during inference, as during training, we iteratively
    build up the message history to form different examples. We also build examples for every speaker instead of
    a single speaker.
    """

    def __init__(self, version: int):
        super().__init__(version)
        # Override init to track malformed pseudo orders
        self.malformed_pseudo_orders_cnt = 0
        self.total_pseudos_cnt = 0
        self.messages_passed_through_filter_cnt = 0
        self.filtered_due_to_game_redaction_word_pct = 0
        self.filtered_due_to_speaker_rating = 0

    def _is_train_time(self) -> bool:
        return True

    MARK_BAD_OR_FILTER_MESSAGES_OPTIONS = [
        "offensive_language",
        "phase_repeats",
        "consecutive_short",
        "redacted",
        "ampersands",
        "names",
        "urls_emails",
        "draws",
        "mutes",
        "grounding",
        "insults",
    ]

    EDIT_MESSAGES_OPTIONS = [
        "names_redact_name_only",
    ]

    def _build_message_marker_and_filterer_and_editor(
        self, metadata: Metadata
    ) -> Tuple[MessageFiltering, MessageFiltering, MessageEditing]:
        marks = metadata["opt"].get("mark_bad_messages", "")
        filters = metadata["opt"].get("filter_bad_messages", "")
        edits = metadata["opt"].get("edit_bad_messages", "")

        marks = "" if marks is None else marks
        filters = "" if filters is None else filters
        edits = "" if edits is None else edits

        marks = marks.split(",")
        filters = filters.split(",")
        edits = edits.split(",")

        marks = [opt.strip() for opt in marks if len(opt.strip()) > 0]
        filters = [opt.strip() for opt in filters if len(opt.strip()) > 0]
        edits = [opt.strip() for opt in edits if len(opt.strip()) > 0]

        if metadata["opt"].get("filter_bad_messages_about_draws", False):
            assert (
                False
            ), "Now you should use --filter-bad-messages draws instead of --filter-bad-messages-about-draws"

        for opt in marks:
            if opt and opt not in self.MARK_BAD_OR_FILTER_MESSAGES_OPTIONS:
                assert False, f"Unrecognized markbad/filter messages option: {opt}"
        for opt in filters:
            if opt and opt not in self.MARK_BAD_OR_FILTER_MESSAGES_OPTIONS:
                assert False, f"Unrecognized markbad/filter messages option: {opt}"
        for opt in edits:
            if opt and opt not in self.EDIT_MESSAGES_OPTIONS:
                assert False, f"Unrecognized edit messages option: {opt}"

        for opt in marks + filters:
            if opt in marks and opt in filters:
                assert False, f"Cannot both mark bad and filter the same thing: {opt}"

        if "names_redact_name_only" in edits and "names" in (marks + filters):
            assert False, f"Cannot both mark bad or filter and redact names"

        if not hasattr(self, "message_editor"):
            self.message_marker = MessageFiltering(
                filter_offensive_language="offensive_language" in marks,
                filter_phase_repeats="phase_repeats" in marks,
                filter_consecutive_short="consecutive_short" in marks,
                filter_excess_redacted="redacted" in marks,
                filter_any_redacted=False,  # inference only
                filter_ampersands="ampersands" in marks,
                filter_names="names" in marks,
                filter_urls_emails="urls_emails" in marks,
                filter_draw_discussion_when_missing_votes="draws" in marks,
                filter_mutes="mutes" in marks,
                filter_grounding="grounding" in marks,
                filter_insults="insults" in marks,
                grounding_last_playable_year=None,
            )
            self.message_filterer = MessageFiltering(
                filter_offensive_language="offensive_language" in filters,
                filter_phase_repeats="phase_repeats" in filters,
                filter_consecutive_short="consecutive_short" in filters,
                filter_excess_redacted="redacted" in filters,
                filter_any_redacted=False,  # inference only
                filter_ampersands="ampersands" in filters,
                filter_names="names" in filters,
                filter_urls_emails="urls_emails" in filters,
                filter_draw_discussion_when_missing_votes="draws" in filters,
                filter_mutes="mutes" in filters,
                filter_grounding="grounding" in filters,
                filter_insults="insults" in filters,
                grounding_last_playable_year=None,
            )
            self.message_editor = MessageEditing(
                # No editing newlines at training time, we only post-process them
                # at inference.
                edit_newlines=False,
                # If names_redact_name_only, we train the net to predict redaction
                # tokens instead of names in the output examples we give it to mimic.
                # Then our usual methods of redaction tokens blocking at inference works.
                edit_names="names_redact_name_only" in edits,
                # No editing weird capitalization at training time, we only post-process them
                # at inference.
                edit_weird_capitalization=False,
            )

        return self.message_marker, self.message_filterer, self.message_editor

    def _get_output_str(
        self,
        partial_history_output: PhaseMsgs,
        speaker: Power,
        recipient: Power,
        phase: Phase,
        metadata: Metadata,
        ind: int,
    ) -> str:
        """
        Format the target output sequence.
        Most args are unused to allow flexibility with overriding.
        """
        return self.messagehistory_flattener.flatten_model_output_messages(partial_history_output,)

    def _get_alternate_game(self, metadata: Metadata, phase: Phase) -> Optional[GameJson]:
        """
        When training models to distinguish between corrupted input/output and real input/output,
        we sometimes want to pull a random game to swap pieces of information (such as state or order history).

        This function tries to find a game with the same phase in the cache, and otherwise returns None.
        """
        if phase == "S1901M":
            # We skip phase S1901M because the state is the same here for every game
            return None

        game_cache = metadata["game_cache"]
        if not game_cache:
            return None

        if len(game_cache) > 0:
            rand_iter = list(range(len(game_cache)))
            random.shuffle(rand_iter)
            for i in rand_iter:
                game = game_cache[i]
                if phase in game.keys():
                    return game

        return None

    def generate_input_output_pairs(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[DialogueSequencePart],
    ) -> TrainingDialoguePredictionOutput:
        """
        Generates an standard dialogue input sequence on the requested format.

        Receives the game data via json, speaker, and metadata parameters.
        Outputs the formatted message and orders history as dictionary,
        according to `format_parts` parameter.

        For example if
        extras=[DialogueSequencePart.STATE, DialogueSequencePart.PSEUDO_ORDERS], then

        <state> <pseudoorder> <player_prompt_token>

        For default extras=None we have

        <player_prompt_token>
        """
        # build message editor
        (
            message_marker,
            message_filterer,
            message_editor,
        ) = self._build_message_marker_and_filterer_and_editor(metadata)

        if not format_parts:
            format_parts = []

        if metadata["opt"].get("add_sleep_times", False):
            # add elapsed time to the input
            format_parts.append(DialogueSequencePart.ELAPSED_TIME)

        self._check_incompatible_opt(metadata, format_parts)

        max_game_redacted_words_percent = metadata["opt"].get("max_game_redacted_words_percent")
        min_speaker_rating = metadata["opt"].get("min_speaker_rating")
        power_metadata = metadata.get("power_metadata")

        seqs = {}
        for phase in game_json:
            seqs[phase] = {}

            # Maybe get game from cache to swap state with
            if "swapstate" in metadata["opt"]["task"]:
                assert (
                    DialogueSequencePart.HISTORY not in format_parts
                ), "Swapping game state is not compatible with order history"
                assert (
                    DialogueSequencePart.PSEUDO_ORDERS not in format_parts
                ), "Swapping game state is not compatible with pseudo orders"

                alternate_game = self._get_alternate_game(metadata, phase)
                if alternate_game is None:
                    continue
            else:
                alternate_game = None

            for speaker in COUNTRY_ID_TO_POWER.values():
                player_prompt_token = format_player_prompt_token(phase, speaker, metadata)
                seqs[phase][speaker] = []
                output_draw_messages = metadata["opt"].get("output_draw_messages", False)
                truncation = metadata["opt"].get("message_history_truncation", 2048)
                two_party_dialogue = metadata["opt"].get("two_party_dialogue", False)
                remove_n_latest_messages_from_dialogue_history = metadata["opt"].get(
                    "remove_n_latest_messages_from_dialogue_history", None
                )
                no_speaker_dialogue = metadata["opt"].get("no_speaker_dialogue_history", False)
                if two_party_dialogue or no_speaker_dialogue:
                    truncation = (
                        truncation * 100
                    )  # Effectively do not truncate the message history

                all_message_histories = self.messagehistory_builder.build_all_possible_message_histories(
                    phase,
                    speaker,
                    game_json,
                    truncation=truncation,
                    include_sleep=metadata["opt"].get("include_sleep_messages", False),
                    output_draw_messages=output_draw_messages,
                    response_view=metadata["opt"].get("response_view_dialogue_model", False),
                )

                cacheable_extras = {}
                # iterate through partial history input and output examples
                for i, (partial_history_input, partial_history_output) in enumerate(
                    all_message_histories
                ):
                    # Apply some filtering criteria
                    if (
                        max_game_redacted_words_percent is not None
                        and metadata["message_stats"]["num_words"] > 0
                    ):
                        if (
                            100.0
                            * metadata["message_stats"]["num_redactions"]
                            / metadata["message_stats"]["num_words"]
                            > max_game_redacted_words_percent
                        ):
                            self.filtered_due_to_game_redaction_word_pct += 1
                            continue
                    if min_speaker_rating is not None:
                        assert (
                            power_metadata is not None
                        ), "power_metadata is None but min_speaker_rating is used"
                        rating = power_metadata[speaker]["rating"]
                        if rating < min_speaker_rating:
                            self.filtered_due_to_speaker_rating += 1
                            continue

                    # possibly mark bad, filter, and edit output messages
                    # NOTE: we are not using the zshot-nonsense classifier to edit the
                    # training messages, if we want in the future, we need to put game here
                    partial_history_output_passing_mark_check = message_marker.filter_messages(
                        partial_history_output,
                        partial_history_input[-1],
                        game=None,
                        game_is_missing_draw_votes=(not metadata.get("has_draw_votes", True)),
                    )
                    if partial_history_output_passing_mark_check is None:
                        mark_bad = True
                    else:
                        mark_bad = False

                    partial_history_output = message_filterer.filter_messages(
                        partial_history_output,
                        partial_history_input[-1],
                        game=None,
                        game_is_missing_draw_votes=(not metadata.get("has_draw_votes", True)),
                    )
                    if partial_history_output is None:
                        # skip this example if we filtered it out
                        continue

                    partial_history_output = message_editor.edit_messages(partial_history_output)

                    self.messages_passed_through_filter_cnt += 1
                    if self.messages_passed_through_filter_cnt % 100000 == 0:
                        logging.info(
                            f"Number of messages checked for markbad/filter/edit: {self.messages_passed_through_filter_cnt}"
                        )
                        logging.info(f"Mark bad stats: {message_marker.get_statistics()}")
                        logging.info(f"Filter bad stats: {message_filterer.get_statistics()}")
                        logging.info(
                            f"Filtered due to game redaction word percent: {self.filtered_due_to_game_redaction_word_pct}"
                        )
                        logging.info(
                            f"Filtered due to speaker rating: {self.filtered_due_to_speaker_rating}"
                        )

                    # determine recipients
                    recipients = [x[MessageObjectPart.RECIPIENT] for x in partial_history_output]
                    assert len(recipients) == 1
                    recipient = recipients[0]

                    # Check that recipient is not ALL unless draw messages
                    if not output_draw_messages or (
                        output_draw_messages
                        and not (
                            is_draw_msg(partial_history_output[0])
                            or is_unvote_draw_msg(partial_history_output[0])
                        )
                    ):
                        assert (
                            recipient != "ALL"
                        ), f"Game {metadata['game_id']}: {partial_history_output[0]}"  # don't train the bot to send messages to all

                    curr_timestamp = partial_history_output[0][MessageObjectPart.TIME_SENT]

                    ind = i + 1
                    formatted_str = ""
                    for fmt_part in format_parts:
                        if fmt_part == DialogueSequencePart.PSEUDO_ORDERS:
                            fmt_part_str = self._get_train_pseudo_orders(
                                speaker, recipient, phase, metadata, ind, curr_timestamp,
                            )
                        elif fmt_part == DialogueSequencePart.ACTUAL_ORDERS:
                            fmt_part_str = self._get_train_actual_orders(
                                game_json, speaker, phase, metadata,
                            )
                        elif fmt_part == DialogueSequencePart.HISTORY:
                            # add message history to the input
                            last_message_time = get_last_timestamp_msg_history(
                                partial_history_input
                            )
                            draw_state = get_gamejson_draw_state(
                                game_json, last_message_time, metadata
                            )

                            if two_party_dialogue:
                                partial_history_input = self.messagehistory_builder._to_2person_dialogue(
                                    partial_history_input, speaker, recipient
                                )
                            if no_speaker_dialogue:
                                partial_history_input = self.messagehistory_builder.to_no_speaker_dialogue(
                                    partial_history_input, speaker
                                )
                            if remove_n_latest_messages_from_dialogue_history is not None:
                                partial_history_input = self.messagehistory_builder._to_remove_n_latest_messages_from_dialogue(
                                    partial_history_input,
                                    speaker,
                                    recipient,
                                    remove_n_latest_messages_from_dialogue_history,
                                )

                            history = self.messagehistory_flattener.flatten_message_history(
                                partial_history_input, draw_state, metadata=metadata,
                            )

                            if metadata is not None and metadata.get("pseudo_order_gen", False):
                                # for pseudo order generation, we edit the history to add
                                # future messages and possible injected messages
                                history = self._edit_history_for_pseudo_order_gen(
                                    speaker,
                                    history,
                                    partial_history_input,
                                    partial_history_output,
                                    phase,
                                    metadata,
                                    draw_state,
                                )
                            fmt_part_str = history
                        elif fmt_part == DialogueSequencePart.ELAPSED_TIME:
                            # add elapsed time
                            if not partial_history_output:
                                # no message, wait time is infinite
                                fmt_part_str = "inf"
                                continue
                            fmt_part_str = get_elapsed_time(curr_timestamp, partial_history_input,)
                        elif fmt_part in cacheable_extras:
                            fmt_part_str = cacheable_extras[fmt_part]
                        else:
                            if alternate_game is not None and fmt_part in {
                                DialogueSequencePart.STATE,
                                DialogueSequencePart.LAST_PHASE_ORDER,
                                DialogueSequencePart.LAST_MOVEMENT_ORDER,
                                DialogueSequencePart.ORDER_HISTORY_SINCE_LAST_MOVE,
                            }:
                                # In this case, we want to corrupt the input by swapping
                                # orders/state with a randomly chosen alternate game
                                game_json_to_use = alternate_game
                            else:
                                # Otherwise, use the designated game JSON
                                game_json_to_use = game_json
                            fmt_part_str = self._format_part(
                                fmt_part, game_json_to_use, speaker, phase, metadata
                            )
                            cacheable_extras[fmt_part] = fmt_part_str

                        if not formatted_str:
                            formatted_str = fmt_part_str
                        else:
                            formatted_str += f"{self.delimiter}{fmt_part_str}"

                    # now add prompt token
                    prompt_token = self._maybe_edit_prompt_token(
                        player_prompt_token, speaker, recipient, metadata
                    )
                    formatted_str += f"{self.delimiter}{prompt_token}"

                    # Leaving this debug code here because it is useful when trying to figure out much why an input fails validation.
                    # Sometimes (for example running diplom dd) when something fails validation it is buried in so much other output that
                    # it is hard to figure out what message goes with what error. Failing locally here, immediately after the relevant input
                    # makes it much easier.
                    # print("----------------------")
                    # print(formatted_str)
                    # regex = input_validation.InputValidator(format_parts, "recipientclassifier", metadata["opt"], self.version).get_input_validation_regex()  # type: ignore
                    # input_validation.validate(
                    #     regex,
                    #     formatted_str,
                    #     throw_on_failure=True
                    # )

                    if mark_bad:
                        formatted_str += " BAD"

                    ex_d = {
                        "input": formatted_str,
                        "output": self._get_output_str(
                            partial_history_output, speaker, recipient, phase, metadata, ind,
                        ),
                        "example_id": get_example_key(metadata["game_id"], speaker, phase, ind),
                        "timestamp": curr_timestamp,
                    }

                    seqs[phase][speaker].append(ex_d)

        return seqs

    def _maybe_edit_prompt_token(
        self, curr_prompt_token: str, sender: Power, recipient: Power, metadata: Metadata
    ) -> str:
        """
        Maybe edit the prompt token.

        Current the prompt is edited:
        (1) When --add-recipient-to-prompt is set to True, we add the recipient to the prompt
        (2) When we are generating pseudo orders and --pseudo-order-generation-partner-view is set to True,
            we swap the sender in the prompt with the recipient, to change the "view" of the input
        """
        curr_prompt_token = super()._maybe_edit_prompt_token(
            curr_prompt_token, sender, recipient, metadata
        )

        if not metadata.get("pseudo_order_gen", False) or not metadata["opt"].get(
            "pseudo_order_generation_partner_view", False
        ):
            # not pseudo order generation
            return curr_prompt_token

        if self.version <= 1:
            # In version 2+, powers are uppercase
            sender = sender.capitalize()
            recipient = recipient.capitalize()

        assert sender in curr_prompt_token
        new_prompt_token = curr_prompt_token.replace(sender, recipient)

        return new_prompt_token

    def _get_partner_view_history(
        self,
        speaker: Power,
        recipient: Power,
        partial_history_input: MsgHistoryList,
        metadata: Metadata,
        draw_state: Optional[CurrentDrawState],
    ) -> str:
        """
        For pseudo order annotation ONLY

        Get the 2 party message history view for annotating the pseudo orders for the partner
        """

        def _valid_msg(msg: MessageDict) -> bool:
            curr_sender = msg[MessageObjectPart.SENDER]
            curr_recipient = msg[MessageObjectPart.RECIPIENT]
            if curr_sender not in {speaker, recipient}:
                return False

            if curr_recipient not in {speaker, recipient, "ALL"}:
                return False

            return True

        edited_history = [[x for x in phase if _valid_msg(x)] for phase in partial_history_input]
        return self.messagehistory_flattener.flatten_message_history(
            edited_history, draw_state, metadata=metadata
        )

    def _edit_history_for_pseudo_order_gen(
        self,
        speaker: Power,
        history: str,
        partial_history_input: MsgHistoryList,
        partial_history_output: PhaseMsgs,
        phase: Phase,
        metadata: Metadata,
        draw_state: Optional[CurrentDrawState],
    ) -> str:
        """
        Add future message and possible injected sentences into
        the dialogue history for the purpose of pseudo order generation.

        Essentially, when we are predicting pseudo orders for a message
        (e.g. France -> England: Let's stay out of the English channel),
        we additionally inject a reply (e.g. England -> France: ok.) so
        that the pseudo orders are predicated on our speaking partner
        agreeing to our plans
        """
        # NOTE: we do not add sleep times for pseudo order generation
        assert not metadata["opt"].get("add_sleep_times", False)

        if metadata["opt"].get("pseudo_order_generation_partner_view", False):
            # limit the input to the partner view
            output = partial_history_output[0]  # only get first message
            history = self._get_partner_view_history(
                output[MessageObjectPart.SENDER],
                output[MessageObjectPart.RECIPIENT],
                partial_history_input,
                metadata,
                draw_state,
            )

        output = self.messagehistory_flattener.flatten_phase_messages(
            partial_history_output, add_sleep_times=False
        )
        # we include the "future message" (e.g. the message that will condition
        # on those orders) when annotating the train set; we turn this flag
        # off if we want to see pseudo orders just prior to the message being sent
        # in order to better understand how they change (for debugging)
        include_future_message = metadata["opt"].get(
            "pseudo_order_generation_future_message", True
        )
        if include_future_message:
            history = concate_msghistory_with_curmsg(
                history=history, output=output, phase_id=phase
            )

        # form of injected sentence, like 'ok.'; see explanation in docstring
        injected_sentence = metadata["opt"].get("pseudo_order_generation_injected_sentence")
        if injected_sentence is None:
            return history

        # whether or not to add an injected sentence from every message recipient
        # this is irrelevant in the single turn dialogue case
        if metadata["opt"].get("pseudo_order_generation_inject_all", True):
            to_inject_msgs = partial_history_output
        else:
            to_inject_msgs = [partial_history_output[-1]]

        # when we don't include future messages, injected messages are handled
        # slightly differently:
        # - if prev sender is not speaker, we do not inject a sentence
        # - if prev sender is speaker, we inject a reply to *that* recipient, rather
        #   than the recipient of the future message
        if not include_future_message:
            # need to check who prev sender was
            last_phase = partial_history_input[-1]
            if not last_phase:
                return history
            last_msg = last_phase[-1]
            last_sender = last_msg[MessageObjectPart.SENDER]
            if speaker != last_sender:
                # no need to inject reply, prev sender was not speaker
                return history
            else:
                # inject reply to recipent of last message
                to_inject_msgs = [last_msg]

        receivers = []
        all_inject_msgs = []
        # for every message we want to add a "fake reply" (injected message from)
        # we construct the injected messages
        for to_inject_msg in to_inject_msgs:
            if to_inject_msg[MessageObjectPart.RECIPIENT] in receivers:
                # don't send multiple injected messages from
                # same player
                continue
            receivers.append(to_inject_msg[MessageObjectPart.RECIPIENT])
            injected_msg = {
                MessageObjectPart.MESSAGE: injected_sentence,
                MessageObjectPart.RECIPIENT: to_inject_msg[MessageObjectPart.SENDER],
                MessageObjectPart.SENDER: to_inject_msg[MessageObjectPart.RECIPIENT],
                MessageObjectPart.PHASE: phase,
            }
            all_inject_msgs.append(injected_msg)

            # Add an identical response
            injected_msg = {
                MessageObjectPart.MESSAGE: injected_sentence,
                MessageObjectPart.RECIPIENT: to_inject_msg[MessageObjectPart.RECIPIENT],
                MessageObjectPart.SENDER: to_inject_msg[MessageObjectPart.SENDER],
                MessageObjectPart.PHASE: phase,
            }
            all_inject_msgs.append(injected_msg)
        # now flatten injected messages
        fmt_injected_message = self.messagehistory_flattener.flatten_phase_messages(
            all_inject_msgs, add_sleep_times=False,
        )
        # concatenate existing history with injected messages
        history = concate_msghistory_with_curmsg(
            history=history, output=fmt_injected_message, phase_id=phase,
        )

        return history

    def _get_train_actual_orders(
        self, game_json: GameJson, speaker: Power, phase: Phase, metadata: Metadata,
    ) -> FlatOrders:
        phases_to_include = order_helper.get_phases_until_next_movement_phase(
            phase,
            game_json,
            including_current=metadata["opt"].get("rollout_except_movement", True),
        )
        rollout_action = {
            _phase: game_json[_phase]["orders"].get(speaker, ()) for _phase in phases_to_include
        }
        if metadata["opt"].get("rollout_actual_orders", False):
            return self.orders_flattener.flatten_rollout_action(rollout_action)
        else:
            return self.orders_flattener.flatten_only_first_action_of_rollout_action(
                rollout_action
            )

    def _get_train_pseudo_orders(
        self,
        speaker: Power,
        recipient: Power,
        phase: Phase,
        metadata: Metadata,
        ind: int,
        curr_timestamp: Timestamp,
    ) -> FlatOrders:
        """
        Retrieve the relevant pseudo orders for the example based on the metadata
        """
        # get the example key; the pseudo orders JSON is indexed on these keys
        ex_key = get_example_key(metadata["game_id"], speaker, phase, ind)
        # now select training pseudo orders
        if metadata["opt"].get("single_view_pseudo_orders", False):
            # Single view pseudo orders; i.e., self/partner view are compiled separately,
            self.total_pseudos_cnt += 1
            try:
                pseudo_orders = metadata["pseudo_orders"][ex_key]
                if "timestamp" in pseudo_orders:
                    # Perform an extra check that pseudo orders exist
                    timestamp = Timestamp.from_centis(
                        pseudo_orders["timestamp"]
                    )  # Extra assurance that we are mapping to the correct example
                    assert timestamp == curr_timestamp, "Pseudo orders timestamp mismatch"
                str_orders = self.orders_flattener.flatten_train_singleview_pseudo_orders(
                    pseudo_orders,
                    speaker,
                    recipient,
                    phase,
                    rollout=metadata["opt"]["rollout_pseudo_orders"],
                    rollout_except_movement=metadata["opt"].get("rollout_except_movement", True),
                )
            except (ValueError, AssertionError, KeyError) as e:
                # Occasionally, pseudo orders are malformed
                self.malformed_pseudo_orders_cnt += 1
                # Raise error if incidence rate is greater than 1%
                if (
                    self.malformed_pseudo_orders_cnt > 100
                    and (self.malformed_pseudo_orders_cnt / self.total_pseudos_cnt) >= 0.01
                ):
                    raise e
                else:
                    logging.warning(
                        f"Malformed or missing pseudo orders, continuing training. Key was {ex_key}, error was {e}"
                    )
                    str_orders = ""
        else:
            pseudo_orders = metadata["pseudo_orders"].get(ex_key, "")
            if pseudo_orders:
                # unflatten & re-flatten orders in order to sort them
                joint_action = self.orders_unflattener.unflatten_joint_action(pseudo_orders)
                # re-sort the order_dct
                str_orders = self.orders_flattener.flatten_joint_action(joint_action, speaker)
            else:
                str_orders = ""
            if str_orders and not metadata["opt"].get("all_power_pseudo_orders", True):
                # only include pseudo orders for relevant powers, which is currently defined
                # as the sender and recipient of the outgoing dialogue message
                relevant_powers = [recipient, speaker]
                str_orders = self._filter_orders(str_orders, relevant_powers)

        return str_orders


class HumanVsModelDiscriminatorFormatter(DialoguePredictionFormatter):
    def generate_input(self, context: str, potential_msg: MessageDict) -> str:
        flattened_potential_msg = self.messagehistory_flattener.flatten_message(potential_msg)
        flattened_input = self.delimiter.join([context, flattened_potential_msg])
        """
        Expected format for the context and potential_msg
        PHASE
        A -> B message1 [EO_M]
        B -> A message2 [EO_M]
        ...
        A -> B messagen PHASE B 5:
        PHASE
        B -> A potential message [EO_M]
        """
        return flattened_input


class TrainingHumanVsModelDiscriminatorFormatter(TrainingDialoguePredictionFormatter):
    """
    Training examples for human vs. model discriminator
    """

    MODEL_GEN_MESSAGES_VERSION = 1
    MODEL_DENOISING_GEN_MESSAGES_VERSION = 3

    def generate_input_output_pairs(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[DialogueSequencePart],
    ) -> TrainingDialoguePredictionOutput:
        seqs = super().generate_input_output_pairs(game_json, metadata, format_parts)
        _skipped_examples = 0
        _same_examples = 0
        _examples = 0
        new_seqs = {}
        for phase in seqs:
            new_seqs[phase] = {}
            for speaker in seqs[phase]:
                new_seqs[phase][speaker] = []
                for example in seqs[phase][speaker]:
                    # V3 of the data was missing annotations for some of the games
                    # Maybe V4 is as well?
                    if any(
                        game_msgs_single_dir.get(example["example_id"]) is None
                        for game_msgs_single_dir in metadata["model_generated_messages"]
                    ):
                        _skipped_examples += 1
                        # logging.info(f"Missing annotation for {example['example_id']}")
                        continue
                    if len(metadata.get("model_generated_messages", [])) == 0:
                        # logging.info(f"Missing annotations for whole game at {example['example_id']}")
                        _skipped_examples += 1
                        continue

                    # Get real output
                    real_output = example["output"]
                    # Unflatten sender message
                    real_message = self.messagehistory_unflattener.unflatten_model_output_messages(
                        real_output, speaker, phase,
                    )[0]

                    # Reflatten as a history message
                    flattened_real_message = self.messagehistory_flattener.flatten_message(real_message)  # type: ignore

                    # Format real example
                    real_example = {
                        "example_id": example["example_id"] + "-R",
                        "input": self.delimiter.join([example["input"], flattened_real_message]),
                        "output": REAL,
                    }
                    _examples += 1

                    # Now format model generated message

                    # We may have multiple sources for generated messages. In this case, let's randomly sample a source.
                    model_output = random.choice(metadata["model_generated_messages"]).get(
                        example["example_id"], None
                    )
                    if model_output is None:
                        # For targeted denoising, there may not be a denoised example for a given human example
                        _skipped_examples += 1
                        continue
                    try:
                        model_gen_unflattener = MessageHistoryUnflattener(
                            self.MODEL_GEN_MESSAGES_VERSION
                        )
                        model_generated_message = model_gen_unflattener.unflatten_model_output_messages(
                            model_output, speaker, phase
                        )
                        if not model_generated_message:
                            model_gen_unflattener = MessageHistoryUnflattener(
                                self.MODEL_DENOISING_GEN_MESSAGES_VERSION
                            )
                            model_generated_message = model_gen_unflattener.unflatten_model_output_messages(
                                model_output, speaker, phase
                            )

                        model_generated_message = model_generated_message[0]
                    except ParlaiDecodingError:
                        # Some model output is not formatted correctly
                        _skipped_examples += 1
                        logging.error(
                            f"Model output: {model_output} formatted incorrectly, skipping example..."
                        )
                        continue
                    except IndexError:
                        _skipped_examples += 1
                        logging.error(f"unflatten returned 0 messages...")
                        continue

                    if metadata["opt"].get("blend_generations", False):
                        frankenstein_message = None
                        for tok in random.sample(
                            CONJUNCTIONS_AND_PUNCTUATION, len(CONJUNCTIONS_AND_PUNCTUATION)
                        ):
                            if tok not in model_generated_message["message"]:
                                continue
                            if tok not in real_message["message"]:
                                continue

                            model_gen_tok_idxs = [
                                m.start()
                                for m in re.finditer(
                                    re.escape(tok), model_generated_message["message"]
                                )
                            ]
                            real_gen_tok_idxs = [
                                m.start()
                                for m in re.finditer(re.escape(tok), real_message["message"])
                            ]

                            num_found = min(len(model_gen_tok_idxs), len(real_gen_tok_idxs))
                            common_idx_attempt = random.randint(
                                0, num_found - 1
                            )  # - 1 because it samples from closed interval
                            model_gen_tok_idx = model_gen_tok_idxs[common_idx_attempt]
                            real_gen_tok_idx = real_gen_tok_idxs[common_idx_attempt]

                            # randomly choose whether to have model-gen part before or after tok
                            if random.randint(0, 1) == 0:
                                # should skip if there is no second bit to add (e.g. the tok is a question mark at end of message)
                                # 3 token soft threshold is a heuristic here
                                if (
                                    len(model_generated_message["message"][model_gen_tok_idx:])
                                    <= len(tok) + 3
                                ):
                                    continue
                                frankenstein_message = (
                                    real_message["message"][:real_gen_tok_idx]
                                    + model_generated_message["message"][model_gen_tok_idx:]
                                )
                            else:
                                if len(real_message["message"][real_gen_tok_idx:]) <= len(tok) + 3:
                                    continue
                                frankenstein_message = (
                                    model_generated_message["message"][:model_gen_tok_idx]
                                    + real_message["message"][real_gen_tok_idx:]
                                )

                            model_generated_message["message"] = frankenstein_message
                            break

                        # skip examples with no option to blend
                        if frankenstein_message is None:
                            continue

                    # Reflatten as history message with current version
                    flattened_model_generated_message = self.messagehistory_flattener.flatten_message(model_generated_message)  # type: ignore

                    # Format model generated example
                    model_example = {
                        "example_id": example["example_id"] + "-C",
                        "input": self.delimiter.join(
                            [example["input"], flattened_model_generated_message]
                        ),
                        "output": CORRUPTED,
                    }
                    if metadata.get("blend_generations", False):
                        model_example["splitting_tok"] = tok  # type: ignore

                    if (
                        model_generated_message["message"].lower()
                        != real_message["message"].lower()
                    ):
                        new_seqs[phase][speaker].append(real_example)
                        new_seqs[phase][speaker].append(model_example)
                    # else:
                    #    new_seqs[phase][speaker].append(real_example)

        logging.warn(f"Processed {_examples} examples and Skipped {_skipped_examples} examples")
        return new_seqs
