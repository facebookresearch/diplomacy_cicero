#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
from parlai.utils import logging
from typing import Optional, Dict, Tuple, Union

from fairdiplomacy import pydipcc
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.typedefs import Phase

from parlai_diplomacy.utils.datapath_constants import (
    PSEUDO_ORDER_PREFIX_ROLLOUT_DIR,
    PSEUDO_ORDER_PREFIX_ROLLOUT_DIR_VERSION,
    PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR,
    PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION,
)
from parlai_diplomacy.utils.game2seq.typing import MessageDict, TrainRolloutPseudoOrderDict
from parlai_diplomacy.utils.game2seq.format_helpers.orders import (
    OrdersUnflattener,
    OrdersFlattener,
)
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.format_helpers.misc import get_example_key

"""
Utils related to pseudo orders
"""


def get_pseudo_orders_directory(
    single_view_pseudo_orders: Optional[bool] = True,
    rollout_except_movement: bool = True,
    load_from_previous_version: bool = False,
) -> str:
    """
    Get directory corresponding to pseudo orders
    """
    if not rollout_except_movement:
        return PSEUDO_ORDER_PREFIX_ROLLOUT_DIR
    elif single_view_pseudo_orders:
        return PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR
    elif load_from_previous_version:
        raise NotImplementedError("Previous versions of pseudo orders are not available")
    else:
        raise RuntimeError("Pseudo orders not compiled for this version")


def load_pseudo_orders_json(
    game_id: int,
    single_view_pseudo_orders: Optional[bool] = True,
    rollout_except_movement: bool = True,
) -> Dict[str, Union[str, TrainRolloutPseudoOrderDict]]:
    """
    Load the pseudo orders JSON for a particular game ID.
    - game ID: game ID
    - single_view_pseudo_orders: load the single_view_pseudo_orders
    """
    pseudo_orders_dir = get_pseudo_orders_directory(
        single_view_pseudo_orders, rollout_except_movement, load_from_previous_version=True
    )
    pseudo_orders_path = os.path.join(pseudo_orders_dir, f"game_{game_id}_pseudo_orders.json")

    with open(pseudo_orders_path, "r") as f:
        pseudo_orders_json = json.load(f)

    return pseudo_orders_json


def _get_deprecated_joint_action_train_pseudo_orders(
    flattened_train_pseudo_orders_str: str, phase: Phase
) -> PseudoOrders:
    return PseudoOrders.from_joint_action(
        OrdersUnflattener(1).unflatten_joint_action(flattened_train_pseudo_orders_str), phase,  # type: ignore
    )


def get_train_pseudo_orders_for_message(
    game: pydipcc.Game,
    game_id: int,
    human_message: MessageDict,
    single_view_pseudo_orders: bool = True,
    rollout_pseudo_orders: bool = True,
    rollout_except_movement: bool = True,
    pseudo_orders_json: Dict[str, Union[str, TrainRolloutPseudoOrderDict]] = None,
) -> Optional[PseudoOrders]:
    """
    Load the train pseudo orders corresponding to a particular message in a particular game.
    - game: full Game object
    - game ID: game ID
    - human_message: message in the game for which we are retrieving pseudo orders
    - single_view_pseudo_orders (bool): True/False corresponding to whether we should retrieve single view pseudo orders
    - rollout_pseudo_orders (bool): True/False corresponding to whether we should retrieve rollout pseudo orders or not
    - rollout_except_movement (bool): True/False corresponding to whether we should retrieve pseudo orders which rollout
        on phases not including the movement phase, or on every phase
    - pseudo orders JSON (optional): optional pseudo orders JSON, if it has been loaded already
    Returns the pseudo orders.
    """
    power = human_message[MessageObjectPart.SENDER]
    recipient = human_message[MessageObjectPart.RECIPIENT]
    phase = human_message[MessageObjectPart.PHASE]

    if pseudo_orders_json is None:
        pseudo_orders_json = load_pseudo_orders_json(
            game_id,
            single_view_pseudo_orders=single_view_pseudo_orders,
            rollout_except_movement=rollout_except_movement,
        )

    phase_messages = game.rolled_back_to_phase_end(phase).messages
    msgs_from_sender = [
        x
        for x in phase_messages.values()
        if x[MessageObjectPart.SENDER] == power and x[MessageObjectPart.RECIPIENT] != "ALL"
    ]
    msg_ind = msgs_from_sender.index(human_message) + 1
    ex_key = get_example_key(game_id, speaker=power, phase=phase, ind=msg_ind)

    if ex_key not in pseudo_orders_json:
        logging.error(
            f"{ex_key} not in pseudo orders JSON for GAME ID {game_id} for message:\n{human_message}"
        )
        return None

    # Rollout except movement being False here implies that we are in the rollout_pseudo_orders setting
    if not rollout_except_movement:
        single_view_pseudo_orders = True
        rollout_pseudo_orders = True

    if not single_view_pseudo_orders:
        # WARNING: DEPRECATED
        return _get_deprecated_joint_action_train_pseudo_orders(pseudo_orders_json[ex_key], phase)  # type: ignore

    if rollout_pseudo_orders:
        version = (
            PSEUDO_ORDER_PREFIX_ROLLOUT_DIR_VERSION
            if not rollout_except_movement
            else PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION
        )
        pseudo_orders = PseudoOrders.from_rollout_joint_action(
            OrdersUnflattener(version).train_pseudo_orders_to_rollout_jointaction(
                pseudo_orders_json[ex_key],  # type: ignore
                power,
                recipient,
                phase,
                rollout_except_movement=rollout_except_movement,
            )
        )
    else:
        # Non-rollout single view pseudo orders
        # (1) First flatten
        flat_pseudo_orders = OrdersFlattener(
            PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION
        ).flatten_train_singleview_pseudo_orders(
            pseudo_orders_json[ex_key], power, recipient, phase, rollout=False,  # type: ignore
        )
        # (2) Then unflatten
        pseudo_orders = PseudoOrders.from_joint_action(
            OrdersUnflattener(
                PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION
            ).unflatten_joint_action(flat_pseudo_orders),
            phase,
        )

    return pseudo_orders


def get_all_pseudo_order_variations_for_message(
    game: pydipcc.Game, game_id: int, human_message: MessageDict,
) -> Tuple[
    Optional[PseudoOrders], Optional[PseudoOrders], Optional[PseudoOrders], Optional[PseudoOrders]
]:
    deprecated_train_joint_action_pseudo_orders = get_train_pseudo_orders_for_message(
        game,
        game_id,
        human_message,
        single_view_pseudo_orders=False,
        rollout_pseudo_orders=False,
        rollout_except_movement=True,
    )
    train_single_view_pseudo_orders = get_train_pseudo_orders_for_message(
        game,
        game_id,
        human_message,
        single_view_pseudo_orders=True,
        rollout_pseudo_orders=False,
        rollout_except_movement=True,
    )
    train_rollout_pseudo_orders = get_train_pseudo_orders_for_message(
        game,
        game_id,
        human_message,
        single_view_pseudo_orders=True,
        rollout_pseudo_orders=True,
        rollout_except_movement=True,
    )
    train_extended_rollout_pseudo_orders = get_train_pseudo_orders_for_message(
        game,
        game_id,
        human_message,
        single_view_pseudo_orders=True,
        rollout_pseudo_orders=True,
        rollout_except_movement=False,
    )

    return (
        deprecated_train_joint_action_pseudo_orders,
        train_single_view_pseudo_orders,
        train_rollout_pseudo_orders,
        train_extended_rollout_pseudo_orders,
    )
