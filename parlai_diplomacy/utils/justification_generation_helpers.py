#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Utilities for generating 'justifications' from human prefixes
"""
from typing import Optional, Dict, Any
from parlai.utils import logging

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryUnflattener,
    MessageObjectPart,
)


JUSTIFICATION_KEY_PHRASES = [
    "so that ",
    "because ",
    "so i ",
    "so he ",
    "so she ",
    "so you ",
    "so russia ",
    "so austria ",
    "so italy ",
    "so england ",
    "so france ",
    "so germany ",
    "so turkey ",
    "but i ",
    "but you ",
    "but he ",
    "but she ",
    "but i'm ",
    "but russia ",
    "but austria ",
    "but italy ",
    "but england ",
    "but france ",
    "but germany ",
    "but turkey ",
    "but if ",
    "but it ",
    "if i ",
    "if you ",
    "if you're ",
    "if he ",
    "if she ",
    "if russia ",
    "if austria ",
    "if italy ",
    "if england ",
    "if france ",
    "if germany ",
    "if turkey ",
    "in order to ",
    "in case ",
    "since i ",
    "since he ",
    "since you ",
    "since russia ",
    "since austria ",
    "since italy ",
    "since england ",
    "since france ",
    "since germany ",
    "since turkey ",
    "however i ",
    "which will ",
    "which would ",
    "want to risk ",
    "you will ",
    "you would ",
    "that will ",
    "that would ",
]


def modify_example_for_justification_generation(
    example: Dict[str, Any], version: int,
) -> Optional[Dict[str, Any]]:
    """
    In the special case of `--justification-generation True`, we
    """
    label_key = "labels" if "labels" in example else "eval_labels"
    example_label = example[label_key][0]
    message_history_unflattener = MessageHistoryUnflattener(version=version)
    try:
        message_obj = message_history_unflattener.unflatten_model_output_messages(
            example_label, sender=example["player"], phase=example["phase_id"]
        )[0]
    except:
        logging.info(f"Malformatted message: {example_label}")
        return example

    recipient = message_obj[MessageObjectPart.RECIPIENT]  # type: ignore
    label_lower = example_label.lower()
    prefix = None

    # Check justification key phrases
    for kw in JUSTIFICATION_KEY_PHRASES:
        if kw in label_lower:
            prefix_lower = label_lower.split(kw)[0] + kw
            prefix = example_label[: len(prefix_lower)].rstrip()  # This rstrip is important here
            example["justification_kw"] = kw
            break

    # Also check if a question mark is featured in the last message from the recipient in the bilateral history
    if prefix is None:
        try:
            msg_hist = [
                message_history_unflattener.unflatten_single_message(x, phase="S1901M")
                for x in example["text"].split("\n")
                if " -> " in x and "ALL" not in x
            ]
            if msg_hist:
                sender_recip_pairs = [
                    x
                    for x in msg_hist
                    if recipient in {x[MessageObjectPart.SENDER], x[MessageObjectPart.RECIPIENT]}
                ]
                if sender_recip_pairs:
                    last_sender = sender_recip_pairs[-1][MessageObjectPart.SENDER]
                    if (
                        last_sender == recipient
                        and "?" in sender_recip_pairs[-1][MessageObjectPart.MESSAGE]
                    ):
                        prefix = example_label.split(":")[0] + ":"
                        example[
                            "justification_kw"
                        ] = f"Question from {last_sender}: {sender_recip_pairs[-1]}"
        except:
            pass

    if prefix is None:
        # No justification words in here, skip this example
        return None

    example["justification_prefix"] = prefix
    return example
