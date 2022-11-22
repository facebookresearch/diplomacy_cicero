#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Helper functions related to formatting message history
"""
from collections import defaultdict
import copy
import re
from typing import Dict, List, Optional, Tuple
import json
import random

from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.data.build_dataset import DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN
from fairdiplomacy.typedefs import (
    CurrentDrawState,
    GameJson,
    Phase,
    Power,
    Timestamp,
    MessageDict,
    OutboundMessageDict,
)
from fairdiplomacy.utils.typedefs import is_phase_name
from fairdiplomacy import pydipcc

from parlai_diplomacy.utils.game2seq.typing import Metadata, PhaseMsgs, MsgHistoryList
from parlai_diplomacy.utils.game2seq.format_helpers.base_helper import BaseFormatHelper
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    add_end_token,
    ParlaiDecodingError,
    remove_trailing_carriage_return,
    uncorrupt_ampersands,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import corrupt_newlines


# Message object parts
class MessageObjectPart:
    SENDER = "sender"
    RECIPIENT = "recipient"
    MESSAGE = "message"
    TIME_SENT = "time_sent"
    PHASE = "phase"


REDACTED_TOKEN = "[REDACTED]"
SLEEP_RECIPIENT = "sleep"


class MessageHistoryFlattener(BaseFormatHelper):
    """
    Flattening messages to strings
    """

    @property
    def use_generic_redacted_token(self):
        # In Version 3, we replace the redacted token inputs
        # with generic [REDACTED] tokens
        return self.version >= 3

    def maybe_replace_redacted_tokens_with_generic_token(self, msg: str) -> str:
        """
        Replace redacted tokens with a special [REDACTED] token
        """
        if self.use_generic_redacted_token:
            msg = re.sub("\\[\d+\\]", REDACTED_TOKEN, msg)

        return msg

    def flatten_message(
        self,
        msg_dct: MessageDict,
        elapsed_time: Optional[Timestamp] = None,
        include_sender: bool = True,
    ) -> str:
        """
        Flatten single message.

        - msg_dict: Message to flatten
        - elapsed_time: Optional timestamp. If it is not None, this is prepended to the message
        - include_sender: Whether to include the sender in the message prefix. In v2, we do not do this
            for the target sequence
        """
        sender = msg_dct[MessageObjectPart.SENDER]
        recipient = msg_dct[MessageObjectPart.RECIPIENT]

        # No game master messages should be included
        assert sender != "GameMaster" and recipient != "GameMaster"

        # In v1, all powers are capitalized
        # In v2, all powers are uppercase
        if self.version <= 1:
            sender = sender.capitalize()
            recipient = recipient.capitalize()

        message = msg_dct[MessageObjectPart.MESSAGE]
        # Message editing
        # (1) Maybe replace redacted tokens with a generic token
        message = self.maybe_replace_redacted_tokens_with_generic_token(message)
        # (2) Corrupt newlines (\n --> ~N~)
        message = corrupt_newlines(message)
        # (3) Replace corrupted ampersands
        message = uncorrupt_ampersands(message)
        # (4) In the latest version of the dataset (as of April 2022) there is a "\r" at the
        # end of messages. Remove it.
        message = remove_trailing_carriage_return(message)

        # Add the recipient
        seq = f"{recipient}: {message}"

        # Maybe add the sender
        if include_sender:
            seq = f"{sender} -> " + seq

        # Maybe add the end token -- in v2 we do not add the end token
        if self.version <= 1:
            seq = add_end_token(seq, "[EO_M]")

        # Maybe add the elpased time
        if elapsed_time is not None:
            seq = f"{elapsed_time.to_seconds_int()} {seq}"

        return seq

    def flatten_outbound_message_candidate(self, msg: OutboundMessageDict) -> str:
        """
        Flatten possible outbound candidate messages
        """
        if self.version <= 1:
            return self.flatten_phase_messages(
                [msg],  # type: ignore
                add_sleep_times=False,
            )

        # Version 2 or greater
        return self.flatten_messages(
            [msg],  # type: ignore
            include_sender=False,
            add_sleep_times=False,
        )

    def flatten_messages(
        self, msg_lst: PhaseMsgs, include_sender: bool = True, add_sleep_times: bool = False
    ) -> str:
        """
        Flatten a list of messages:

        Args:
            msg_lst: list[dict]

        Returns:
            England -> Turkey: msg [EO_M]
            Turkey -> England: msg [EO_M]
        """
        if not msg_lst:
            # empty phase
            return ""

        flat_msgs = []
        phase_name = msg_lst[0][MessageObjectPart.PHASE]
        sleep_time = None
        last_message_time = Timestamp.from_centis(0)
        if add_sleep_times and len(msg_lst) > 0:
            last_message_time = msg_lst[0][MessageObjectPart.TIME_SENT]
        for msg_dct in msg_lst:
            # just checking if they are from the same phase
            assert phase_name == msg_dct[MessageObjectPart.PHASE]
            # check for elapsed time
            if add_sleep_times:
                curr_msg_time: Timestamp = msg_dct[MessageObjectPart.TIME_SENT]
                assert isinstance(curr_msg_time, Timestamp), type(curr_msg_time)
                assert isinstance(last_message_time, Timestamp), type(last_message_time)
                sleep_time = curr_msg_time - last_message_time
                last_message_time = curr_msg_time  # update
            flat_msg = self.flatten_message(
                msg_dct, elapsed_time=sleep_time, include_sender=include_sender
            )
            flat_msgs.append(flat_msg)

        return "\n".join(flat_msgs)

    def flatten_phase_messages(self, msg_lst: PhaseMsgs, add_sleep_times: bool = False) -> str:
        """
        flatten messages in one phase, called in self.format_msg in the teacher
        for the current phase msg

        Args:
            msg_lst: list[dict]

        Returns:
            S1901M
            England -> Turkey: msg [EO_M]
            Turkey -> England: msg [EO_M]
        """
        # flatten messages
        flat_msgs = self.flatten_messages(
            msg_lst, include_sender=True, add_sleep_times=add_sleep_times
        )
        if not flat_msgs:
            return ""

        # add phase info
        phase_name = msg_lst[0]["phase"]
        flat_msgs_str = "\n".join([phase_name, flat_msgs])

        return flat_msgs_str

    def flatten_message_history(
        self, msg_his_lst: MsgHistoryList, draw_state: Optional[CurrentDrawState], metadata=None
    ):
        """flatten the message history,
        message history = all msgs that happen before the prediction
        e.g.
        1) if predicting an order, message history = all msgs that happen before the end of the phase
        2) if predicting a message, message history = all msgs that happen before the time to generate a new message

        Args:
            msg_his_lst (list[list[dict]]): a list of phase messages, each phase message is a list of messages

        Returns:
            if len(msg_his_lst) == 0:
                return ""
            else:
                return:
                    S1901M
                    England -> Turkey: msg [EO_M]
                    Turkey -> England: msg [EO_M]
                    F1901M
                    England -> Turkey: msg [EO_M]
                    Turkey -> England: msg [EO_M]
        """
        phase_msgs = []
        for phase_msg_lst in msg_his_lst:
            phase_msg = self.flatten_phase_messages(
                phase_msg_lst, add_sleep_times=metadata["opt"].get("add_sleep_times", False)
            )
            phase_msgs.append(phase_msg)

        # remove empty phase
        phase_msgs = [msg for msg in phase_msgs if msg != ""]

        # maybe add draw state summary
        maybe_draw_state = (
            get_draw_state_suffix(draw_state, metadata) if draw_state is not None else ""
        )

        return "\n".join(phase_msgs) + maybe_draw_state

    def flatten_model_output_messages(self, msg_lst: PhaseMsgs) -> str:
        """
        Flatten dialogue model output messages
        """
        if self.version <= 1:
            return self.flatten_phase_messages(
                msg_lst, add_sleep_times=False  # sleep times are not added to output
            )

        # In V2, we remove the redundant speaker and phase at the beginning of the message
        return self.flatten_messages(
            msg_lst,
            include_sender=False,
            add_sleep_times=False,  # sleep times are not added to output
        )


def get_draw_state_suffix(draw_state: CurrentDrawState, metadata: Metadata) -> str:
    hide_empty = metadata["opt"].get("hide_empty_draw_state", False)
    if not hide_empty and not metadata["has_draw_votes"]:
        return "\nDRAWS: Unavailable"

    draw_powers = set(power for power, status in draw_state.items() if status)

    if hide_empty and not draw_powers:
        # Don't show empty draw state
        return ""

    return "\nDRAWS: " + " ".join(sorted(draw_powers, key=POWERS.index))


class MessageHistoryUnflattener(BaseFormatHelper):
    """
    Unflattening messages from strings
    """

    def create_message_object(
        self, sender: Power, recipient: Power, message: str, phase: Phase
    ) -> OutboundMessageDict:
        """
        Create a message object from the parts:

        - sender: sender of the message
        - recipient: recipient of the message
        - message: text of the message
        - phase: phase the message was sent in
        """
        assert sender in POWERS, f"{sender} is not a power"
        assert recipient in POWERS, f"{recipient} is not a power"
        assert is_phase_name(phase), f"{phase} is not a phase name"

        return {
            MessageObjectPart.SENDER: sender,
            MessageObjectPart.RECIPIENT: recipient,
            MessageObjectPart.MESSAGE: message,
            MessageObjectPart.PHASE: phase,
        }

    def unflatten_single_message(self, message_str: str, phase: Phase) -> OutboundMessageDict:
        """
        Unflatten a single message that was parsed as text
        """
        split_msg = message_str.split(": ")
        powers = split_msg[0]
        text = ": ".join(split_msg[1:])
        text = text.replace("[EO_M]", "").strip()  # Get rid of [EO_M] token
        sender, recipient = powers.split(" -> ")

        return self.create_message_object(sender.upper(), recipient.upper(), text, phase)

    def unflatten_messages(self, output_seq, phase: Phase) -> List[OutboundMessageDict]:
        """
        Unflatten a string sequence containing possibly multiple messages.
        """
        message_lst = []
        try:
            lines = output_seq.split("\n")[1:]  # first line is the phase
            curr_msg = []
            for line in lines:
                if " -> " in line:  # we use the arrow to determine whether a new message was sent
                    # parse new msg
                    if curr_msg:
                        message_lst.append(
                            self.unflatten_single_message("\n".join(curr_msg), phase)
                        )
                    curr_msg = []
                curr_msg.append(line)

            # clear cache
            if curr_msg:
                message_lst.append(self.unflatten_single_message("\n".join(curr_msg), phase))
        except Exception as e:
            raise ParlaiDecodingError(
                f"Could not decipher model output: {output_seq};\nError message: {e}"
            )

        return message_lst

    def unflatten_single_sender_message(
        self, message_str, sender: Power, phase: Phase
    ) -> OutboundMessageDict:
        """
        Unflatten a single message that was sent by `sender`.

        Used in version V2 -- messages are of the form
        <RECIPIENT>: <message>
        """
        assert self.version >= 2
        split_msg = message_str.split(": ")
        recipient = split_msg[0].upper()
        text = ": ".join(split_msg[1:])
        text = text.replace("[EO_M]", "").strip()  # Get rid of [EO_M] token

        return self.create_message_object(sender.upper(), recipient.upper(), text, phase)

    def unflatten_sender_messages(
        self, output_seq, sender: Power, phase: Phase
    ) -> List[OutboundMessageDict]:
        """
        Unflatten a possible list of messages sent by a specific power.

        Used in version V2 -- messages are of the form
        <RECIPIENT>: <message>
        """
        assert self.version >= 2
        receivers = [f"{power.capitalize()}" for power in POWERS]

        def new_msg(line: str) -> bool:
            for receiver in receivers:
                if line.startswith(receiver):
                    return True

            return False

        message_lst = []
        try:
            lines = output_seq.split("\n")
            curr_msg = []
            for line in lines:
                if new_msg(line):
                    # parse new msg
                    if curr_msg:
                        message_lst.append(
                            self.unflatten_single_sender_message(
                                "\n".join(curr_msg), sender, phase
                            )
                        )
                    curr_msg = []
                curr_msg.append(line)

            # clear cache
            if curr_msg:
                message_lst.append(
                    self.unflatten_single_sender_message("\n".join(curr_msg), sender, phase)
                )
        except Exception as e:
            raise ParlaiDecodingError(
                f"Could not decipher model output: {output_seq};\nError message: {e}"
            )

        return message_lst

    def unflatten_model_output_messages(
        self, output_seq, sender: Power, phase: Phase
    ) -> List[OutboundMessageDict]:
        """
        Unflatten dialogue model output messages.

        Output dialogue messages are formatted differently in different versions.
        """
        if self.version <= 1:
            msg_lst = self.unflatten_messages(output_seq, phase)
        else:
            # In V2, we remove the redundant phase and sender in the output
            msg_lst = self.unflatten_sender_messages(output_seq, sender, phase)

        return msg_lst


class MessageHistoryBuilder(BaseFormatHelper):
    """
    Utilities for building message history from a game JSON.
    """

    @property
    def include_messages_to_all(self):
        # include messages sent to/from "ALL"
        return self.version >= 2

    def extract_message_history_from_game_json(
        self,
        game_json: GameJson,
        curr_phase: Phase,
        speaker: Power,
        truncation: int = 2048,
        include_prev_phase: bool = True,
        include_curr_phase: bool = True,
    ) -> MsgHistoryList:
        """
        Given a game JSON in pydipcc format, extract all of the messages that would be seen by
        a given speaker and convert them to ParlAI format.

        - game json: game JSON
        - curr_phase: current phase
        - speaker: view of power
        - truncation: # of words the message history should be truncated to, default 2048
        - include_prev_phase: include messages from previous phases
        - include_curr_phase: include messages from the current phase
        """

        def _valid_msg_for_speaker(
            speaker: Power, msg_sender: Power, msg_recipient: Power
        ) -> bool:
            # Don't include messages to/from ALL
            if not self.include_messages_to_all and (
                msg_sender == "ALL" or msg_recipient == "ALL"
            ):
                return False

            # return True/False whether this message is 'viewable' by the speaker power
            if msg_sender == speaker:
                return True
            if msg_recipient == speaker:
                return True
            if msg_recipient == "ALL":
                # Include messages sent by any power to "ALL"
                return True

            return False

        all_phases = sorted(list(game_json.keys()), key=lambda x: sort_phase_key(x))
        curr_phase_idx = all_phases.index(curr_phase)
        all_message_history = [
            (phase, game_json[phase]["messages"])
            for phase in all_phases[: curr_phase_idx + 1]  # don't include future messages
        ]

        messages_to_keep = []
        total_words = 0
        # note we iterate in reverse order here for truncation purposes
        for phase, message_lst in reversed(all_message_history):
            if total_words > truncation:
                # we hit the max # of words, don't include previous phases
                break
            if not include_curr_phase and phase == curr_phase:
                # do not include messages from current phase
                continue
            if not include_prev_phase and phase != curr_phase:
                # skip messages not from this phase
                break
            phase_lst = []
            prev_msg = None  # Track to see duplicate messages
            for message in message_lst:
                if _valid_msg_for_speaker(
                    speaker,
                    message[MessageObjectPart.SENDER],
                    message[MessageObjectPart.RECIPIENT],
                ) and not is_duplicate_msg(message, prev_msg):
                    total_words += len(message["message"].split(" "))  # rough count of num words
                    phase_lst.append(message)
                    if message[MessageObjectPart.RECIPIENT] != "ALL":
                        # Don't consider messages to "ALL" as previous messages
                        prev_msg = message
            messages_to_keep.insert(0, phase_lst)  # we are iterating in reverse

        return messages_to_keep

    def _extract_prev_phase_messages_from_game_json(
        self, game_json: GameJson, curr_phase: Phase, speaker: Power, truncation: int = 2048
    ) -> MsgHistoryList:
        """
        Given a game JSON in pydipcc format, extract all of the messages from the current phase
        """
        return self.extract_message_history_from_game_json(
            game_json,
            curr_phase,
            speaker,
            truncation=truncation,
            include_prev_phase=True,
            include_curr_phase=False,
        )

    def _extract_curr_phase_messages_from_game_json(
        self, game_json: GameJson, curr_phase: Phase, speaker: Power
    ) -> PhaseMsgs:
        """
        Given a game JSON in pydipcc format, extract all of the messages from the current phase
        """
        return self.extract_message_history_from_game_json(
            game_json, curr_phase, speaker, include_prev_phase=False, include_curr_phase=True
        )[0]

    def _extract_phase_start_time_from_game_json(
        self, game_json, curr_phase
    ) -> Optional[Timestamp]:
        """
        Given a game JSON in pydipcc format, extract time of the first message from anyone in a phase, or None if
        no messages were sent in a phase.
        """
        curr_phase_messages = game_json[curr_phase]["messages"]
        return curr_phase_messages[0]["time_sent"] if len(curr_phase_messages) > 0 else None

    @staticmethod
    def _add_sleep_messages(
        phase: str, power: str, msgs: PhaseMsgs, phase_start_time: Optional[Timestamp]
    ):
        """
        # before each interrupt event (send/recieve message, or phase ends),
        # add a sleep message for how long the player slept since the last
        # interrupt event. This is exact when a player is sending a message, or
        # a lower bound when recieving a message. For example, if a player
        # sends a message at timestamps 5 and 20, and then recieves a message
        # at timestamp 100, we add a sleep of 15 after the first message, and a
        # sleep of >80 after the second.

        if we sleep 10 and send to ENGLAND
        -> sleep: 10 ENGLAND
        -> sleep: >10 everybody else

        if we sleep 10 and receive a message
        -> sleep: >10 everybody
        """
        updated_msgs = []
        last_msg_time = phase_start_time
        if phase_start_time is None:
            # This currently happens when no messages are sent in a phase
            return msgs
        have_sent_message = False

        for msg in msgs:
            assert last_msg_time is not None
            is_sender = msg["sender"] == power
            have_sent_message = have_sent_message or is_sender
            elapsed_time = str((msg["time_sent"] - last_msg_time).to_seconds_int())
            text = f"OUT {elapsed_time} {msg['recipient']}" if is_sender else f"IN {elapsed_time}"
            updated_msgs.append(
                {
                    "sender": power,
                    "recipient": SLEEP_RECIPIENT,
                    "message": text,
                    "time_sent": last_msg_time,
                    "phase": phase,
                }
            )
            updated_msgs.append(msg)
            last_msg_time = msg["time_sent"]

        if phase != "S1901M" or have_sent_message:
            # Add a final infinite wait after the last interrupt event. We skip this for players who don't sent any messages on the first turn, analogously to the all-holds filtering.
            updated_msgs.append(
                {
                    "sender": power,
                    "recipient": SLEEP_RECIPIENT,
                    "message": "inf",
                    "time_sent": last_msg_time,
                    "phase": phase,
                }
            )
        return updated_msgs

    def _to_2person_dialogue(
        self, partial_msg_history_input: MsgHistoryList, sender: Power, recipient: Power
    ) -> MsgHistoryList:
        def _valid_msg(msg: MessageDict) -> bool:
            msg_sender = msg[MessageObjectPart.SENDER]
            msg_recipient = msg[MessageObjectPart.RECIPIENT]

            # Message sender must be sender or recipient
            if msg_sender not in {sender, recipient}:
                return False

            # Message sender can be sender/recipient/ALL
            if msg_recipient not in {sender, recipient, "ALL"}:
                return False

            return True

        new_message_history_input = []
        for phase_msgs in partial_msg_history_input:
            new_phase_msgs = []
            for phase_msg in phase_msgs:
                if _valid_msg(phase_msg):
                    new_phase_msgs.append(phase_msg)

            new_message_history_input.append(new_phase_msgs)

        return new_message_history_input

    def to_no_speaker_dialogue(
        self, partial_msg_history_input: MsgHistoryList, speaker: Power
    ) -> MsgHistoryList:
        def _phase_to_no_speaker_dialogue(phase_msg_list: PhaseMsgs) -> PhaseMsgs:
            return [msg_dct for msg_dct in phase_msg_list if speaker != msg_dct["sender"]]

        # remove dialogue from speaker
        partial_msg_history_input = [
            _phase_to_no_speaker_dialogue(partial_phase_history_input)
            for partial_phase_history_input in partial_msg_history_input
        ]
        # remove empty phases
        partial_msg_history_input = [
            partial_phase_history_input
            for partial_phase_history_input in partial_msg_history_input
            if len(partial_phase_history_input) > 0
        ]
        # if no messages, make sure there's empty list for current phse
        if len(partial_msg_history_input) == 0:
            partial_msg_history_input = [[]]

        return partial_msg_history_input

    def _to_remove_n_latest_messages_from_dialogue(
        self,
        partial_msg_history_input: MsgHistoryList,
        sender: Power,
        recipient: Power,
        remove_n_latest_messages_from_dialogue_history: int,
    ):
        assert (
            remove_n_latest_messages_from_dialogue_history > 0
        ), remove_n_latest_messages_from_dialogue_history

        num_msgs_to_remove = random.sample(
            range(1, remove_n_latest_messages_from_dialogue_history + 1), 1
        )[0]
        msgs_removed = 0

        def _valid_msg(msg: MessageDict) -> bool:
            msg_sender = msg[MessageObjectPart.SENDER]
            msg_recipient = msg[MessageObjectPart.RECIPIENT]

            # We only filter messages between sender and recipient
            if {msg_sender, msg_recipient} != {sender, recipient}:
                return True

            if msgs_removed < num_msgs_to_remove:
                return False

            return True

        new_message_history_input = []
        for phase_msgs in reversed(partial_msg_history_input):
            new_phase_msgs = []
            for phase_msg in reversed(phase_msgs):
                if _valid_msg(phase_msg):
                    new_phase_msgs.append(phase_msg)
                else:
                    msgs_removed += 1

            new_message_history_input.append(list(reversed(new_phase_msgs)))

        return list(reversed(new_message_history_input))

    def _build_single_turn_dialogue_message_histories(
        self,
        prev_phase_message_history: List[List[MessageDict]],
        all_cur_phase_msgs: List[MessageDict],
        speaker: str,
        output_draw_messages: bool = False,
    ):
        """
        In the single-turn case, we predict all messages sent by a player one at a time.

        Given the previous message history and the current phase messages, we build a list
        of tuples as possible dialogue examples.
        """
        # in this case, we predict every dialogue message one at a time

        def valid_speaker_msg(msg: MessageDict) -> bool:
            # return whether or not a message is a valid target for a speaker
            # speaker messages only
            if not msg[MessageObjectPart.SENDER] == speaker:
                return False

            # include draw messages only if output_draw_messages = True
            if output_draw_messages and (is_draw_msg(msg) or is_unvote_draw_msg(msg)):
                return True

            # otherwise, we ignore messages directed to ALL in the output
            if msg[MessageObjectPart.RECIPIENT] == "ALL":
                return False

            return True

        all_possible_histories = []
        cur_phase_msgs = []
        for msg in all_cur_phase_msgs:
            if valid_speaker_msg(msg):
                # current speaker, so we add an example
                # NOTE: we don't add examples when the recipient is "ALL" as we don't want to teach
                # the bot to send messages to all, UNLESS the message is a draw message, and we specifically
                # want to output draw messagess
                ex_history = [
                    *[copy.copy(ms) for ms in prev_phase_message_history],
                    copy.copy(cur_phase_msgs),
                ]
                all_possible_histories.append((ex_history, [msg]))
                # now add to the message history ONLY if the recipient != sleep
                if msg[MessageObjectPart.RECIPIENT] != SLEEP_RECIPIENT:
                    cur_phase_msgs.append(msg)
            else:
                # different speaker, so we simply append to the message history
                cur_phase_msgs.append(msg)

        return all_possible_histories

    def _build_single_turn_dialogue_message_histories_from_response_view(
        self,
        prev_phase_message_history: MsgHistoryList,
        curr_phase_messages: PhaseMsgs,
        speaker: str,
    ):
        """
        In the single-turn case, we predict all messages sent by a player one at a time, from the response view.

        Given the previous message history and the current phase messages, we build a list
        of tuples as possible dialogue examples.
        """
        # in this case, we predict every dialogue message one at a time
        all_possible_histories = []
        curr_all_msg_history = copy.deepcopy(prev_phase_message_history)
        curr_all_msg_history.append([])  # for this phase
        for msg in curr_phase_messages:
            if msg["recipient"] == speaker:
                # current speaker, so we add an example
                all_possible_histories.append((copy.deepcopy(curr_all_msg_history), [msg]))
                # now add to the message history
                curr_all_msg_history[-1].append(msg)
            else:
                # different speaker, so we simply append to the message history
                curr_all_msg_history[-1].append(msg)

        return all_possible_histories

    def build_all_possible_message_histories(
        self,
        phase: str,
        speaker: str,
        game_json: Dict,
        truncation: int = 2048,
        include_sleep: bool = False,
        output_draw_messages: bool = False,
        response_view: bool = False,
    ) -> List[Tuple]:
        """
        Given a list of messages for a given power, build up all possible message histories through time.

        - phase: current phase
        - speaker: view of power
        - game json: game JSON
        - truncation: # of words the message history should be truncated to, default 2048
        - include_sleep: include sleep times as messages, only used by the sleep classifier
        - output_draw_messages: whether or not to include draw messages in the output
        - response_view: input = message from game view of 'speaker', output = Other players' response to 'speaker'
        """

        prev_phase_message_history = self._extract_prev_phase_messages_from_game_json(
            game_json, phase, speaker, truncation=truncation
        )
        # get all messages from previous messages
        curr_phase_messages = self._extract_curr_phase_messages_from_game_json(
            game_json, phase, speaker
        )  # get messages from the current phase

        if include_sleep:
            first_message_in_phase_time = self._extract_phase_start_time_from_game_json(
                game_json, phase
            )
            # add sleep messages
            curr_phase_messages = self._add_sleep_messages(
                phase, speaker, curr_phase_messages, first_message_in_phase_time
            )

        if response_view:
            # in this case, we predict every dialogue message one at a time
            return self._build_single_turn_dialogue_message_histories_from_response_view(
                prev_phase_message_history, curr_phase_messages, speaker
            )
        else:
            # in this case, we predict every dialogue message one at a time
            return self._build_single_turn_dialogue_message_histories(
                prev_phase_message_history, curr_phase_messages, speaker, output_draw_messages
            )


##########################################
#  UTILITIES
##########################################


def concate_msghistory_with_curmsg(history, output, phase_id) -> str:
    """
    concate message_history and current_msg to form future msg for the pseudo-order generation
    msg[:t] + msg[t] --> order[t-1]
    :param queue_output: the queue_output with all the data used in the streaming teacher
    """
    # First remove draw state from history
    history_lines = history.split("\n")
    draw_state_lines = [line for line in history_lines if line.startswith("DRAWS: ")]
    draw_state = draw_state_lines[0] if draw_state_lines else None
    history_without_draws = [line for line in history_lines if not line.startswith("DRAWS: ")]
    history = "\n".join(history_without_draws)

    if phase_id in history:
        msg_history_and_future_msg = (
            history
            + "\n"
            + output.replace(phase_id, "").strip("\n")  # remove the leading phase_id in output
        )
    else:
        msg_history_and_future_msg = history + "\n" + output
    msg_history_and_future_msg = msg_history_and_future_msg.strip("\n")

    # Now re-add draw state
    if draw_state is not None:
        msg_history_and_future_msg += f"\n{draw_state}"

    return msg_history_and_future_msg


def is_duplicate_msg(curr_message, prev_message):
    """Some messages from the same speaker in the same phase are exactly the same but with different time stamps in the original data, and it's
    a pretty easy pattern for the model to pick up, so we want to remove these duplicate messages (we suspect it's because of some logging issue on webdip)

    For example, the last two sentences in the example below are exactly the same (the same sender, receipt and message content)

    [id]: Base Diplomacy Teacher
    [game_id]: 120860
    [phase_id]: S1902M
    [player]: ENGLAND
    [text]: S1901M
    England -> Germany: i would like to suggest an alliance against france [EO_M]
    England -> Italy: i would like to suggest an alliance against france [EO_M]
    Italy -> England: All right, let fight France. [EO_M]
    Germany -> England: okay, that sounds good to me ~N~ are you moving to north sea or directly to english channel? ~N~ are you completely ignoring norway? ~N~ I would like hol this turn, you can have bel [EO_M]
    England -> Germany: i am moving to both the north sea and the english channel [EO_M]
    England -> Germany: i agree with the holland and belgium idea, is fine by me [EO_M]
    England -> Germany: i agree with the holland and belgium idea, is fine by me [EO_M]

    """
    if curr_message is None or prev_message is None:
        return False

    if (
        curr_message["message"] == prev_message["message"]
        and curr_message["phase"] == prev_message["phase"]
        and curr_message["sender"] == prev_message["sender"]
        and curr_message["recipient"] == prev_message["recipient"]
    ):
        return True
    else:
        return False


def is_draw_msg(msg: MessageDict) -> bool:
    """
    Return True/False corresponding to whether a message is a draw vote
    """
    return msg[MessageObjectPart.MESSAGE] == DRAW_VOTE_TOKEN


def is_unvote_draw_msg(msg: MessageDict) -> bool:
    """
    Return True/False corresponsding to whether a messages is an unvote draw vote
    """
    return msg[MessageObjectPart.MESSAGE] == UNDRAW_VOTE_TOKEN


def get_last_speaker(input_text: str) -> Tuple[Optional[Power], Optional[str]]:
    """Given a formatted input sequence containing message history, return
       the speaker and phase of the last message sent
    """
    last_speaker, last_phase, current_phase = None, None, None
    for line in input_text.split("\n"):
        line = line.strip()
        if is_phase_name(line):
            current_phase = line
            continue
        if "->" in line and "[EO_M]" in line:
            # this is a message
            last_speaker = line.split()[0].upper()
            last_phase = current_phase
            assert last_speaker in POWERS, last_speaker
    return last_speaker, last_phase


def get_elapsed_time(curr_time: Timestamp, message_history: MsgHistoryList) -> str:
    """
    Get the time since the last message sent or received, given the curr time, in stringified format
    """
    last_message_time = Timestamp.from_centis(0)

    if message_history:
        for phase_messages in reversed(message_history):
            if phase_messages:
                last_message_time = phase_messages[-1][MessageObjectPart.TIME_SENT]
                break

    # get the stringified elapsed time
    return str((curr_time - last_message_time).to_seconds_int())


def add_message_to_game_copy(
    game: pydipcc.Game,
    *,
    phase: str,
    sender: Power,
    recipient: Power,
    body: str,
    time_sent: Timestamp,
) -> pydipcc.Game:
    """
    This function recreates the game from scratch phase by phase and then inserts one message according to `time_sent` of the message
    and returns a new game object with the original one untouched.
    The reason why we have this instead of using `add_messages_to_game_object` in fairdiplomacy.env is that
    `add_messages_to_game_object` adds the message to the current phase, while this function adds the message to the specified phase in the argument
    Note:
    The copied game may have different name for the last phase. If you load a human game then last could be called "COMPLETED" instead of something like "W1905M".
    It shouldn't matter as the model should not do predictions at the last phase, it doesn't break any test as of now.
    """
    game_json = json.loads(game.to_json())
    phases = [phase["name"] for phase in game_json["phases"]]
    assert phase in phases, "Unknown phase: %s. Known: %s" % (phase, phases)

    phase_json = game_json["phases"][phases.index(phase)]
    if time_sent is not None:
        assert not any(
            msg["time_sent"] == time_sent for msg in phase_json["messages"]
        ), f"Timestamp {time_sent} already exists"
    else:
        # insert a fake time stamp
        time_sent = Timestamp.from_centis(0)
        for phase_json in game_json["phases"]:
            for message in phase_json["messages"]:
                # Set the time sent to at least one second longer
                # than the last message that was sent
                time_sent = message["time_sent"] + 1

    phase_json["messages"].append(
        dict(sender=sender, recipient=recipient, time_sent=time_sent, message=body, phase=phase)
    )
    phase_json["messages"].sort(key=lambda x: x["time_sent"])

    return pydipcc.Game.from_json(json.dumps(game_json))


def add_message_to_game_copy_increment_on_collision(
    game: pydipcc.Game,
    *,
    phase: str,
    sender: Power,
    recipient: Power,
    body: str,
    time_sent: Timestamp,
) -> pydipcc.Game:
    """
    This method is identical to add_message_to_game_copy, except it increments on timestamp collision rather than throwing on error in this case.
    """
    game_json = json.loads(game.to_json())
    phases = [phase["name"] for phase in game_json["phases"]]
    assert phase in phases, "Unknown phase: %s. Known: %s" % (phase, phases)
    phase_json = game_json["phases"][phases.index(phase)]
    assert phase_json["name"] == phase, (phase_json["name"], phase)
    num_msgs = len(phase_json["messages"])

    if time_sent is not None:
        # increment on collision
        updated = True
        while updated != False:
            updated = False
            for message in phase_json["messages"]:
                if time_sent == Timestamp.from_centis(int(message["time_sent"])):
                    time_sent += Timestamp.from_centis(1)
                    updated = True
        assert not any(
            msg["time_sent"] == time_sent for msg in phase_json["messages"]
        ), f"Timestamp {time_sent} already exists"
    else:
        # insert a fake time stamp
        time_sent = Timestamp.from_centis(0)
        for phase_json_iter in game_json["phases"]:
            for message in phase_json_iter["messages"]:
                # Set the time sent to at least one second longer
                # than the last message that was sent
                time_sent = message["time_sent"] + 1

    print(
        f"Injected message: {dict(sender=sender, recipient=recipient, time_sent=time_sent, message=body, phase=phase)}"
    )

    phase_json["messages"].append(
        dict(sender=sender, recipient=recipient, time_sent=time_sent, message=body, phase=phase)
    )
    phase_json["messages"].sort(key=lambda x: x["time_sent"])

    # test
    new_phase_json = game_json["phases"][phases.index(phase)]
    assert new_phase_json["name"] == phase, (new_phase_json["name"], phase)
    assert len(phase_json["messages"]) == num_msgs + 1, (len(phase_json["messages"]), num_msgs)
    assert len(new_phase_json["messages"]) == len(phase_json["messages"]), (
        len(new_phase_json["messages"]),
        len(phase_json["messages"]),
    )

    return pydipcc.Game.from_json(json.dumps(game_json))


def get_gamejson_draw_state(
    game_json: GameJson, until_time: Optional[Timestamp], metadata: Metadata
) -> Optional[CurrentDrawState]:
    """
    Returns dict with boolean flag corresponding to which players have currently voted for a draw
    up through and including time until_time.
    """
    if not metadata["opt"].get("include_draw_state"):
        return None

    drawstate: CurrentDrawState = defaultdict(lambda: False)

    if until_time is None:
        # No previous messages; no one has drawn
        return drawstate

    for phase_data in game_json.values():
        for message in phase_data["messages"]:
            if message[MessageObjectPart.TIME_SENT] > until_time:
                # we already exceed this time, break before updating
                # the draw state
                break
            if is_draw_msg(message):
                drawstate[message[MessageObjectPart.SENDER]] = True
            elif is_unvote_draw_msg(message):
                drawstate[message[MessageObjectPart.SENDER]] = False

    return drawstate


def get_last_timestamp_msg_history(msg_history: MsgHistoryList) -> Optional[Timestamp]:
    """
    Returns the last possible timestamp in the message history list, else None
    """
    timestamp = None
    for phase_messages in reversed(msg_history):
        if phase_messages:
            timestamp = phase_messages[-1][MessageObjectPart.TIME_SENT]
            assert isinstance(timestamp, Timestamp)
            break

    return timestamp


def get_last_timestamp_gamejson(game_json: GameJson) -> Optional[Timestamp]:
    """
    Returns the last possible timestamp in the GameJson, else None

    """
    all_message_history: MsgHistoryList = [
        game_json[phase].get("messages", []) for phase in game_json.keys()
    ]
    return get_last_timestamp_msg_history(all_message_history)
