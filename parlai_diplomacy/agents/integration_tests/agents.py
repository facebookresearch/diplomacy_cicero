#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from parlai.agents.test_agents.test_agents import MockTorchAgent
from parlai.core.loader import register_agent
from parlai.core.torch_agent import Output

from fairdiplomacy.models.consts import POWERS

import random
import torch
import numpy as np


ALL_ORDERS_TASK = "message_history_shortstate_allorder_chunk"
ALL_ORDERS_INDEPENDENT_TASK = (
    "message_history_orderhistorysincelastmovementphase_shortstate_allorderindependent_chunk"
)
DIALOGUE_TASK = "message_history_pseudoorder_dialogue_chunk"
SLEEP_CLASSIFIER_TASK = "message_history_sleepclassifier_chunk"
RECIPIENT_CLASSIFIER_TASK = (
    "message_history_orderhistorysincelastmovementphase_shortstate_recipientclassifier_chunk"
)


@register_agent("mock_all_orders_agent")
class MockOrdersAgent(MockTorchAgent):
    beam_size = 1

    def eval_step(self, batch):
        """
        Return fake orders, or score candidates by their length.
        """
        if self.rank_candidates:
            ret_candidates = [e[::-1] for e in batch.candidates]
            return Output(
                text_candidates=ret_candidates,
                cand_scores=[
                    [[(len(c), len(c)) for _ in range(3)] for c in e] for e in ret_candidates
                ],
            )

        # get string input to the model
        # inputs should be of form: "{model input} {power we are predicting}:"
        inputs = [batch.observations[i]["text"] for i in range(len(batch.text_vec))]
        powers = [text.split(" ")[-1].replace(":", "") for text in inputs]
        all_other_powers = []
        for power in powers:
            other_powers = [other.capitalize() for other in POWERS if other.capitalize() != power]
            other_powers.append(power)
            all_other_powers.append(other_powers)

        responses = [
            "\n".join(f"{power}: [EO_O]" for power in other_powers)
            for i, other_powers in enumerate(all_other_powers)
        ]

        return Output(responses)


@register_agent("mock_dialogue_agent")
class MockDialogueAgent(MockTorchAgent):
    beam_size = 1
    DIALOGUE_RESPONSES = [
        "Do you want to form an alliance with me?",
        "OK.",
        "Sounds good!",
        "Good luck!",
    ]

    def eval_step(self, batch):
        """
        Return fake dialogue.
        """
        inputs = [batch.observations[i]["text"] for i in range(len(batch.text_vec))]
        powers = [text.split(" ")[-1].replace(":", "") for text in inputs]
        phases = [text.split(" ")[-2] for text in inputs]
        responses = [random.choice(self.DIALOGUE_RESPONSES) for _ in inputs]
        other_powers = []
        for power in powers:
            other_power = power
            while other_power == power:
                other_power = random.choice(POWERS)
            other_powers.append(other_power.capitalize())

        formatted_responses = [
            f"{phases[i]}\n{powers[i]} -> {other_powers[i]}: {resp}"
            for i, resp in enumerate(responses)
        ]

        return Output(formatted_responses)

    def set_prefix_tokens(self, prefix):
        pass


@register_agent("mock_sleep_classifier_agent")
class MockSleepClassiferAgent(MockTorchAgent):
    SLEEP_CLASSES = "pad 0 1 2 3 inf".split()
    PROBS = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def eval_step(self, batch):
        return Output(
            [f"{random.choice(self.SLEEP_CLASSES)}" for i in range(len(batch.text_vec))],
            class_list=[self.SLEEP_CLASSES],
            probs=torch.tensor([self.PROBS] * len(batch.text_vec)),
        )


@register_agent("mock_recipient_classifier_agent")
class MockRecipientClassiferAgent(MockTorchAgent):
    RECIPIENT_CLASSES = ["England", "France", "Italy", "Germany", "Austria", "Turkey", "Russia"]
    PROBS = [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.0,
        0.0,
    ]  # there is no reasoning behind these choices for probabilities

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def eval_step(self, batch):
        return Output(
            [f"{random.choice(self.RECIPIENT_CLASSES)}" for i in range(len(batch.text_vec))],
            class_list=[self.RECIPIENT_CLASSES],
            probs=torch.tensor([self.PROBS] * len(batch.text_vec)),
        )
