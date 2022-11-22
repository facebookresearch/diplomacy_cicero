#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Callable, List, Optional

import numpy as np

import fairdiplomacy.action_exploration
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.typedefs import Action, JointAction, Policy, Power, PowerPolicies
from fairdiplomacy.utils.zipn import unzip2


def _get_stepped_games(
    game: Game, own_actions: List[Action], power: Power, power_orders: JointAction
) -> List[Game]:
    stepped_games = []
    for action in own_actions:
        stepped_game = Game(game)
        stepped_game.set_orders(power, action)
        for other_power, other_action in power_orders.items():
            if other_power != power:
                stepped_game.set_orders(other_power, other_action)
        stepped_game.process()
        stepped_games.append(stepped_game)
    return stepped_games


def _compute_variance_reduction_offsets(
    get_values: Callable[[List[Game], Power], np.ndarray],
    game: Game,
    power: Power,
    policy: Policy,
    power_orders: JointAction,
) -> float:
    # For now, something fairly simple - for each power, we hold every other power's action
    # fixed and compute how lucky or unlucky we were simply among our own actions.

    own_actions, own_probs = unzip2(policy.items())
    own_actions = list(own_actions)
    stepped_games = _get_stepped_games(game, own_actions, power, power_orders)
    values = get_values(stepped_games, power)
    probs = np.array(own_probs)
    sum_probs = np.sum(probs)
    # Something weird - perhaps we got passed an empty policy or something like that
    assert (
        abs(sum_probs - 1.0) < 0.01
    ), f"Phase {game.current_short_phase} variance reduction {power} got a non-probability-distribution: {policy}"

    mean_value = np.sum(values * probs) / np.sum(probs)
    offset = -(values[own_actions.index(power_orders[power])] - mean_value)
    logging.info(
        f"Phase {game.current_short_phase} variance reduction for {power}: values {values}, probs {probs}, mean {mean_value}, final offset {offset}"
    )

    return offset


def compute_variance_reduction_offsets(
    variance_reduction_model: BaseStrategyModelWrapper,
    game: Game,
    power: Power,
    policy: Policy,
    power_orders: JointAction,
) -> float:
    """Return a value to add to the square_score of each power to hopefully reduce variance.

    To guarantee unbiasedness, it MUST be the case that power_orders was sampled
    from power_policies according to the probabilities in power_policies.

    Each entry in the returned dictionary will be zero-mean in expectation. It is NOT guaranteed
    that they will sum to 0.
    """

    def get_values(stepped_games: List[Game], power: Power) -> np.ndarray:
        values = fairdiplomacy.action_exploration.get_values_from_base_strategy_model(
            variance_reduction_model.value_model, power, stepped_games, power,
        )
        values = values.cpu().numpy()
        assert len(values.shape) == 1
        assert values.shape[0] == len(stepped_games)
        return values

    return _compute_variance_reduction_offsets(get_values, game, power, policy, power_orders)


def _get_actions_if_all_obvious(policies_by_power: PowerPolicies) -> Optional[JointAction]:
    # For purpose of searchbot evaluation, consider an action obvious and rolloutable-past if
    # the probablity in the policy is at least this high.
    OBVIOUS_ACTION_THRESHOLD = 0.98

    actions_by_power = {}
    for power, policy in policies_by_power.items():
        if not policy:
            continue
        top_action, top_prob = max(policy.items(), key=lambda x: x[1])
        if top_prob >= OBVIOUS_ACTION_THRESHOLD:
            actions_by_power[power] = top_action
        else:
            return None
