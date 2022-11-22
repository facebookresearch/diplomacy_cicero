#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from types import MethodDescriptorType
from typing import List, Optional, Dict, Tuple
from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.typedefs import Message, Phase, RolloutJointAction
from fairdiplomacy.utils.typedefs import is_rollout_joint_action
from parlai_diplomacy.wrappers.dialogue import ParlAIDialogueWrapper
from fairdiplomacy import pydipcc
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.format_helpers.orders import is_movement_phase
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.agents.parlai_message_handler import ParlaiMessageHandler
from fairdiplomacy.models.consts import POWERS


def get_prob_under_nucleus_sampling(
    message_handler: ParlaiMessageHandler,
    game: pydipcc.Game,
    timestamp: int,
    pseudoorders_per_phase_time_sent: Dict[Tuple[str, int], RolloutJointAction],
) -> Optional[float]:
    """
    This function determines if the message at a given timestamp in a given game coul
    be produced by a given agent under nucleus scoring. It returns true of the message could
    not be produced.

    It does this by rolling back the game to right before the message in question was produced,
    generating pseudo orders if relevant, and then calculating the probability of the message
    in question under nucleus scoring according the agent's conditional distribution over messages.

    Since nucleus scoring truncates the probability of unlikely messages to 0, we say the message
    "in nucleus" by the agent if its probability under nucleus scoring is nonzero.
    """
    game_at_time_end = game.rolled_back_to_timestamp_end(timestamp)
    game_at_time_start = game.rolled_back_to_timestamp_start(timestamp)

    message_data = game_at_time_end.messages[timestamp]

    sender = message_data[MessageObjectPart.SENDER]
    recipient = message_data[MessageObjectPart.RECIPIENT]
    curr_phase = message_data[MessageObjectPart.PHASE]

    game_at_time_start_with_perspective = game_from_view_of(game_at_time_start, sender)

    if message_handler.expects_pseudo_orders():
        pseudo_orders = pseudoorders_per_phase_time_sent[
            (message_data[MessageObjectPart.PHASE], message_data[MessageObjectPart.TIME_SENT])
        ]

        # We need the eval dataset for computing the nonsense filtering metric to be the same
        # for all models to enable comparison, so we make sure metric can be computed for rollout pseudo orders model
        if not is_movement_phase(curr_phase) and len(pseudo_orders.keys()) == 1:
            return None
        else:
            pseudo_orders = sorted(
                pseudo_orders.items(), key=lambda psuedo_order: sort_phase_key(psuedo_order[0])
            )[0][1]

        # convert to rollout pseudo orders if need be
        if message_handler.model_dialogue.expects_rollout_pseudo_orders() and not is_rollout_joint_action(
            pseudo_orders
        ):
            pseudo_orders = {curr_phase: pseudo_orders}

        if not message_handler.model_dialogue.expects_rollout_pseudo_orders() and is_rollout_joint_action(
            pseudo_orders
        ):
            pseudo_orders = pseudo_orders[curr_phase]

        message_handler.model_dialogue.update_pseudo_orders(
            game_at_time_start_with_perspective.current_short_phase, sender, pseudo_orders
        )

    game_at_time_start_with_perspective = game_from_view_of(game_at_time_start, sender)
    scores = message_handler.model_dialogue.score_candidate_messages(
        game_at_time_start_with_perspective,
        [message_data[MessageObjectPart.MESSAGE]],
        sender=sender,
        timestamp=message_data[MessageObjectPart.TIME_SENT],
        recipient=recipient,
    )

    return scores[0][1]
