#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
ParlAI utils related to WebDip games.
"""
from fairdiplomacy.timestamp import Timestamp
from parlai.utils import logging
from typing import List, Tuple, Optional

from fairdiplomacy.typedefs import OutboundMessageDict, Phase, Power, RolloutJointAction
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryUnflattener,
    MessageObjectPart,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    CORRUPTED_NEWLINE,
    uncorrupt_newlines,
)
from parlai_diplomacy.utils.game2seq.format_helpers.orders import OrdersUnflattener

TASK_VERSION = 3


def _get_dialogue_input_output_sequences(game_lines: List[str]) -> List[Tuple[str, str]]:
    input_to_output_sequences = []
    curr_input_lines = []
    curr_output_lines = []
    input = False
    output = False
    for line in game_lines:
        if "(ParlAIDialogueWrapper) Input sequence:" in line:
            input = True
            continue
        if "(ParlAIDialogueWrapper) Output sequence:" in line:
            output = True
            continue
        if input:
            if "[dialogue:" in line:
                # Input ended
                input = False
            else:
                curr_input_lines.append(line)
        if output:
            if not line:
                # Output ended
                output = False
                # Append existing example
                input_to_output_sequences.append(
                    ("\n".join(curr_input_lines).strip(), "\n".join(curr_output_lines))
                )
                curr_input_lines = []
                curr_output_lines = []
            else:
                curr_output_lines.append(line)

    return input_to_output_sequences


def _extract_pseudo_orders_from_dialogue_input(dialogue_input: str) -> RolloutJointAction:
    dialogue_input_lines = dialogue_input.split("\n")
    prompt = dialogue_input_lines[-1]
    dialogue_input_lines = dialogue_input_lines[:-2]  # remove prompt and sleep time
    state_end = 0
    for i, line in enumerate(dialogue_input_lines):
        for state_type in ["units:", "centers:", "builds:", "retreats:"]:
            if line.startswith(state_type):
                state_end = i

    pseudo_orders_str = "\n".join(dialogue_input_lines[state_end + 1 :])
    phase = prompt.split(" ")[0]
    unflattener = OrdersUnflattener(TASK_VERSION)
    pseudo_orders = unflattener.unflatten_rollout_joint_action_bilateral_powermajor(
        pseudo_orders_str, phase
    )

    return pseudo_orders


def _clean_msg(raw_message: str) -> str:
    clean_msg = uncorrupt_newlines(raw_message)

    return clean_msg


def _extract_last_ts_msg_from_dialogue_input(
    dialogue_input: str,
) -> Optional[Tuple[Timestamp, str]]:
    dialogue_input_lines = dialogue_input.split("\n")
    ts, last_msg = None, None
    for line in dialogue_input_lines:
        if "->" in line:
            ts = Timestamp.from_seconds(int(line.split(" ")[0]))
            last_msg = line.split(": ")[1]

    if ts and last_msg:
        return ts, _clean_msg(last_msg)
    else:
        return None


def _extract_message_from_dialogue_input_and_output(dialogue_input: str, dialogue_output: str):
    prompt = dialogue_input.split("\n")[-1]
    phase = prompt.split(" ")[0]
    power = prompt.split(" ")[1]
    unflattener = MessageHistoryUnflattener(TASK_VERSION)
    message = unflattener.unflatten_model_output_messages(dialogue_output, power, phase)[0]
    message["message"] = _clean_msg(message["message"])

    return message


def extract_all_pseudo_orders_from_webdip_logs(
    game_log_path: str,
) -> List[Tuple[OutboundMessageDict, RolloutJointAction, Optional[Tuple[Timestamp, str]]]]:
    """
    ***VERY HACKY: USE WITH CAUTION***
    """
    with open(game_log_path, "r") as f:
        lines = f.read().splitlines()

    # Get string input and output sequences
    input_to_output_sequences = _get_dialogue_input_output_sequences(lines)

    # Extract pseudo orders from input and message content from output
    message_to_pseudo_orders = []
    for inp, out in input_to_output_sequences:
        try:
            maybe_last_ts_msg = _extract_last_ts_msg_from_dialogue_input(inp)
            pseudo_orders = _extract_pseudo_orders_from_dialogue_input(inp)
            message = _extract_message_from_dialogue_input_and_output(inp, out)
            message_to_pseudo_orders.append((message, pseudo_orders, maybe_last_ts_msg))
        except Exception as e:
            continue

    return message_to_pseudo_orders


def extract_pseudo_orders_for_webdip_message(
    game_log_path: str, message_str: str, sender: Power, recipient: Power, phase: Phase,
) -> Optional[RolloutJointAction]:
    """
    ***VERY HACKY: USE WITH CAUTION***
    """
    all_pseudo_orders = extract_all_pseudo_orders_from_webdip_logs(game_log_path)
    logging.info(f"Found {len(all_pseudo_orders)} messages from game path: {game_log_path}")
    for message, pseudo_orders, maybe_last_ts_msg in all_pseudo_orders:
        if (
            sender == message[MessageObjectPart.SENDER]
            and recipient == message[MessageObjectPart.RECIPIENT]
            and phase == message[MessageObjectPart.PHASE]
            and message_str == message[MessageObjectPart.MESSAGE]
        ):
            return pseudo_orders

    logging.warning("Could not find pseudo orders associated with message")
    return None
