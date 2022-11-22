#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
import logging
import itertools
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler

from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import SearchResult
from fairdiplomacy.agents.bilateral_stats import WeightedAverager
from fairdiplomacy.models.base_strategy_model.load_model import SomeBaseStrategyModel
from fairdiplomacy.agents.base_strategy_model_wrapper import compute_action_logprobs_from_state
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import (
    Action,
    JointAction,
    BilateralConditionalValueTable,
    Power,
    PowerPolicies,
    Policy,
)
from fairdiplomacy.utils.base_strategy_model_multi_gpu_wrappers import (
    MultiProcessBaseStrategyModelExecutor,
)
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY_TO_IDX
from fairdiplomacy.utils.game import game_from_two_party_view


class BRCorrBilateralSearchResult(SearchResult):
    def __init__(
        self,
        agent_power: Power,
        bp_policies: PowerPolicies,
        bilateral_search_policies: PowerPolicies,
        power_value_matrices: Dict[Power, BilateralConditionalValueTable],
    ):
        self.agent_power = agent_power
        self.bp_policies = bp_policies
        assert len(self.bp_policies) == len(POWERS)
        self.policies: PowerPolicies = bilateral_search_policies
        self.power_value_matrices = power_value_matrices
        self.value_to_me: Dict[Tuple[Power, Action], WeightedAverager] = {}

        # search results do not contain dead powers and our policies
        for power, policy in self.bp_policies.items():
            if power not in self.policies and power != agent_power:
                assert len(bilateral_search_policies) == 0 or (
                    len(policy) == 1 and list(policy.keys())[0] == ()
                )
                self.policies[power] = policy

    def set_policy_and_value_for_power(self, power: Power, best_action: Action, best_value: float):
        """Set our policy and decide the value_to_me for my and opponents' actions.

        Our policy is simply set as {best_action: 1.0}
        The value_to_me[pwr, action'] is the value we will get if we play the best_action and pwr plays action',
        which is ectracted from the corresponding joint action value matrix between agent_power and pwr.
        """
        assert power == self.agent_power
        assert len(self.value_to_me) == 0, self.value_to_me
        self.policies[power] = {best_action: 1.0}
        agent_power_idx = POWERS.index(self.agent_power)
        for power, policy in self.bp_policies.items():
            if power == self.agent_power:
                self.value_to_me[power, best_action] = WeightedAverager()
                self.value_to_me[power, best_action].accum(best_value, 1)
                continue

            for action in policy:
                self.value_to_me[power, action] = WeightedAverager()
                if power not in self.power_value_matrices:
                    assert len(action) == 0
                    # dead power, value not matter
                    self.value_to_me[power, action].accum(0, 1)
                    continue

                # our value when we select the best_action and the opponent select this action
                value = self.power_value_matrices[power][(best_action, action)][
                    agent_power_idx
                ].item()
                self.value_to_me[power, action].accum(value, 1)

    def get_agent_policy(self) -> PowerPolicies:
        return self.policies

    def get_population_policy(self) -> PowerPolicies:
        return self.policies

    def get_bp_policy(self) -> PowerPolicies:
        return self.bp_policies

    def sample_action(self, power) -> Action:
        action = sample_p_dict(self.policies[power])
        return action

    def avg_utility(self, power: Power) -> float:
        """Returns the average utility for this power, if everyone plays the population policy."""
        raise NotImplementedError

    def avg_action_utility(self, power: Power, a: Action) -> float:
        raise NotImplementedError

    def is_early_exit(self) -> bool:
        return False


def extract_bp_policy_for_powers(bp_policy: PowerPolicies, powers: List[Power]):
    pair_bp: PowerPolicies = {}
    for power, policy in bp_policy.items():
        if power in powers:
            pair_bp[power] = policy
        else:
            # as if they have been eliminated
            pair_bp[power] = {(): 1.0}
    return pair_bp


def _sample_conditional_joint_actions(
    all_power_base_strategy_model: MultiProcessBaseStrategyModelExecutor,
    game: pydipcc.Game,
    agent_power: Power,
    other_power_per_joint_action: List[Power],
    bilateral_joint_actions: List[Tuple[Action, Action]],
    num_sample: int,
    has_press: bool,
) -> List[JointAction]:
    """sample a list of joint actions conditioning on the partial_joint_actions
    return 1-D list of joint actions that can be reshaped as [num_sample x len(partial_joint_actions)]

    other_power_per_joint_action must be of the same size as bilateral_joint_actions, and it indicates
    for each of those items, who is the other power that the second action in the tuple belongs to.
    """
    conditional_orders: List[Action] = [
        tuple(order for action in (agent_action, other_action) for order in action)
        for (agent_action, other_action) in bilateral_joint_actions
    ]
    other_power_per_joint_action = [
        x for x in other_power_per_joint_action for _ in range(num_sample)
    ]
    bilateral_joint_actions = [x for x in bilateral_joint_actions for _ in range(num_sample)]

    orders_per_batch = 1 + len(conditional_orders) // all_power_base_strategy_model.num_workers()
    orders_per_batch = min(
        all_power_base_strategy_model.base_strategy_model_wrapper_kwargs["max_batch_size"]
        // num_sample,
        orders_per_batch,
    )
    assert orders_per_batch > 0

    futures = []
    logging.info(
        f"total orders to condition on {len(conditional_orders)}, per batch: {orders_per_batch}"
    )
    for i in range(0, len(conditional_orders), orders_per_batch):
        conditional_orders_per_worker = conditional_orders[i : i + orders_per_batch]
        if len(conditional_orders_per_worker) == 0:
            continue
        futures.append(
            all_power_base_strategy_model.compute(
                "forward_policy",
                [game],
                agent_power=None,
                has_press=has_press,
                temperature=1.0,
                top_p=1.0,
                conditional_orders=conditional_orders_per_worker,
                batch_repeat_interleave=num_sample,
            )
        )
    cond_joint_actions: List[List[Action]] = []
    for future in futures:
        cond_joint_actions += future.result()[0]

    joint_actions = []
    assert len(bilateral_joint_actions) == len(cond_joint_actions)
    mismatch = 0
    match = 0
    for i in range(len(bilateral_joint_actions)):
        (agent_action, other_action) = bilateral_joint_actions[i]
        joint_action = {agent_power: agent_action, other_power_per_joint_action[i]: other_action}
        assert len(cond_joint_actions[i]) == len(POWERS)
        for power, action in zip(POWERS, cond_joint_actions[i]):
            if power not in joint_action:
                joint_action[power] = action
            else:
                match += action == joint_action[power]
                mismatch += action != joint_action[power]
        joint_actions.append(joint_action)

    logging.info(f"sample conditional joint action, mismatch: {mismatch}, match: {match}")
    if mismatch > 0.1 * match:
        logging.warning(f"Too many mismatches, mismatch: {mismatch}, match: {match}")
    return joint_actions


def sample_joint_actions(power_policies: PowerPolicies, num_sample: int) -> List[JointAction]:
    joint_actions: List[JointAction] = []
    for i in range(num_sample):
        joint_action = {}
        for power, policy in power_policies.items():
            action = sample_p_dict(policy)
            joint_action[power] = action
        joint_actions.append(joint_action)
    return joint_actions


def compute_weights_for_opponent_joint_actions(
    joint_actions: List[JointAction],
    my_power: Power,
    game: pydipcc.Game,
    base_strategy_model: SomeBaseStrategyModel,
    bp_policy: PowerPolicies,
    has_press: bool,
    min_unnormalized_weight: float,
    max_unnormalized_weight: float,
) -> List[float]:
    """Compute the weight of each joint action of opponents (a1, a2, ..., a6)

    Assume that each joint action (are sampled from prod_i P_cfr(a_i), this function
    computes [P_joint (a1, a2, ..., a6) + min_prob] / [prod_i P_marginal (a_i)]

    joint_action: list of joint action for other powers excluding my_power
    base_strategy_model: model to evaluate probability of joint actions for other powers (a1, a2, ..., a6)
    bp_policy: probability of actions rescored by independent base_strategy_model
    """
    # compute P_joint(a1, a2, ..., a6) as sum_{a0'} P_joint(a0', a1, ..., a6)
    logprob_joints = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        joint_actions,
        agent_power=None,
        has_press=has_press,
        batch_size=len(joint_actions),
    )

    assert len(logprob_joints) == len(joint_actions), (len(logprob_joints), len(joint_actions))
    unnormed_weights = []
    weights = []
    stats = []
    for i, joint_action in enumerate(joint_actions):
        joint_logp = logprob_joints[i]
        independent_logp = 0
        stat = {}
        for power, action in joint_action.items():
            assert power != my_power
            indep_p = bp_policy[power][action]
            independent_logp += np.log(indep_p)
            stat[f"{power}, {action}"] = f"bp: {indep_p:.5f}, log bp: {np.log(indep_p):.5f}"

        stat["indep logp"] = independent_logp
        stat["joint logp"] = logprob_joints[i]
        weight = np.exp(joint_logp) / np.exp(independent_logp)
        unnormed_weights.append(weight)
        if min_unnormalized_weight > 0:
            weight = max(min_unnormalized_weight, weight)
        if max_unnormalized_weight > 0:
            weight = min(max_unnormalized_weight, weight)
        weights.append(weight)
        stat["weight"] = weight
        stats.append(stat)

    logging.info(
        f">> max unnormed weight before clip: {max(unnormed_weights):.6f}, "
        f"min unnormed weight before clip: {min(unnormed_weights):.6f}"
    )
    logging.info(
        f">> max_clipped_importance_weight: {max(weights)}, min_clipped_importance_weight: {min(weights)}"
    )
    hist_counts, log_values = np.histogram(np.log(weights), bins=8)
    logging.info("      weight | count")
    for cc, vv in zip(hist_counts, np.exp(log_values)):
        logging.info(f"    {vv:8.2g} | {cc}")
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    stats = sorted(stats, key=lambda x: -x["weight"])
    logging.debug("top 5 joint actions with the largest un-normalized weight:")
    for i in range(5):
        logging.debug(f"joint action No.{i}")
        for k, v in stats[i].items():
            logging.debug(f"{k}, {v}")
    stats2 = sorted(stats, key=lambda x: -x["weight"])
    logging.debug("bottom 5 joint actions with the smallest un-normalized weight:")
    for i in range(5):
        logging.debug(f"joint action No.{i}")
        for k, v in stats2[i].items():
            logging.debug(f"{k}, {v}")

    return weights


def compute_best_action_against_reweighted_opponent_joint_actions(
    game: pydipcc.Game,
    agent_power: Power,
    agent_policy: Policy,
    opponent_joint_actions: List[JointAction],
    weights: List[float],
    all_power_base_strategy_model: MultiProcessBaseStrategyModelExecutor,
    player_rating: Optional[float],
    regularize_lambda: float,
) -> Tuple[Action, float]:
    action_values: List[Tuple[Action, float, float, float]] = []
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    assert abs(weights_tensor.sum().item() - 1) < 1e-5

    full_joint_actions: List[JointAction] = []
    for action, prob in agent_policy.items():
        for partial in opponent_joint_actions:
            assert agent_power not in partial
            full = partial.copy()
            full[agent_power] = action
            full_joint_actions.append(full)

    rollout_results = _multi_gpu_base_strategy_model_rollouts(
        game, all_power_base_strategy_model, agent_power, full_joint_actions, player_rating
    )
    assert rollout_results.size(0) == len(
        full_joint_actions
    ), f"{rollout_results.size(0)}, {len(full_joint_actions)}"
    assert rollout_results.size(0) == len(opponent_joint_actions) * len(agent_policy)

    for i, (action, prob) in enumerate(agent_policy.items()):
        values = rollout_results[
            i * len(opponent_joint_actions) : (i + 1) * len(opponent_joint_actions),
            POWERS.index(agent_power),
        ]
        values = values.squeeze(1).cpu()
        assert weights_tensor.size() == values.size()
        value = (weights_tensor * values).sum().item()
        pice_value = value + regularize_lambda * np.log(max(prob, 1e-6))
        action_values.append((action, value, prob, pice_value))

    action_values = sorted(action_values, key=lambda x: -x[-1])

    logging.info(f"<> best response results using pice lambda {regularize_lambda}")
    effective_num_sample = sum(weights) ** 2 / sum([w ** 2 for w in weights])
    logging.info(f">> effective num sample: {effective_num_sample:.3f} / {len(weights)}")
    logging.info(
        f">> max weight: {max(weights) * len(weights):.4f}, min weight: {min(weights) * len(weights):.4f}"
    )
    logging.info(f"      {'pice_v':8s}  {'v':8s}  {'bp_p':8s}  orders")
    for action, value, prob, pice_value in action_values:
        logging.info(f"|>:  {pice_value:8.5f}  {value:8.5f}  {prob:8.5f}  {action}")

    best_action, _, _, best_value = action_values[0]
    return best_action, best_value


def compute_payoff_matrix_for_all_opponents(
    game: pydipcc.Game,
    all_power_base_strategy_model: MultiProcessBaseStrategyModelExecutor,
    bp_policy: PowerPolicies,
    agent_power: Power,
    num_sample: int,
    has_press: bool,
    player_rating: Optional[float],
    value_table_cache: Optional[DefaultDict[Power, BilateralConditionalValueTable]],
) -> Dict[Power, BilateralConditionalValueTable]:
    """Compute payoff matrix for all (agent_power, opponent) pairs

    returns a dictionary of opponent -> ConditionalValueTable(agent_power, opponent)
    ConditionalVable Table is a dict that maps each partial joint action of (agent_power, opponent)
    to a [7, 1] tensor that store the value for each power averaged over num_sample joint actions
    conditioning on the partial bilateral joint action.
    For example, given N actions for agent_power and M actions for other_power,
    the dictionary contains:

    (agent_action_0, other_action_0) -> Tensor [7, 1]
    (agent_action_0, other_action_1) -> Tensor [7, 1]
    ...
    (agent_action_N, other_action_M) -> Tensor [7, 1]
    """
    if value_table_cache is None:
        value_table_cache = defaultdict(dict)

    num_bilateral_joint_action_per_opponent: List[Tuple[Power, int]] = []
    cache_hits = 0
    # list of all non-cached partial joint actions that we need to condition on
    bilateral_joint_actions: List[Tuple[Action, Action]] = []
    other_power_per_joint_action: List[Power] = []
    for opponent in bp_policy:
        if opponent == agent_power:
            continue

        num_joint_action = 0
        power_actions_list: List[List[Action]] = [
            [action for action in bp_policy[power]] for power in [agent_power, opponent]
        ]
        for bilateral_joint_action in itertools.product(*power_actions_list):
            if bilateral_joint_action in value_table_cache[opponent]:
                cache_hits += 1
                continue
            bilateral_joint_actions.append(bilateral_joint_action)
            other_power_per_joint_action.append(opponent)
            num_joint_action += 1

        num_bilateral_joint_action_per_opponent.append((opponent, num_joint_action))

    logging.info(
        f"payoff_matrix: {cache_hits}/{len(bilateral_joint_actions) + cache_hits} joint actions cached"
    )

    if len(bilateral_joint_actions) == 0:
        # call cached, nothing to compute
        return value_table_cache

    joint_actions = _sample_conditional_joint_actions(
        all_power_base_strategy_model,
        game,
        agent_power,
        other_power_per_joint_action,
        bilateral_joint_actions,
        num_sample,
        has_press,
    )
    # compute values for these joint actions
    rollout_results = _multi_gpu_base_strategy_model_rollouts(
        game, all_power_base_strategy_model, agent_power, joint_actions, player_rating
    )
    rollout_results = rollout_results.view(
        num_sample, len(bilateral_joint_actions), len(POWERS), 1
    ).mean(0)
    logging.info(f"len joint_actions: {len(joint_actions)}, {len(bilateral_joint_actions)}")
    start = 0
    for opponent, count in num_bilateral_joint_action_per_opponent:
        if count == 0:
            continue

        opponent_bilateral_joint_actions = bilateral_joint_actions[start : start + count]
        opponent_rollout_results = rollout_results[start : start + count]
        for idx, bilateral_joint_action in enumerate(opponent_bilateral_joint_actions):
            value_table_cache[opponent][bilateral_joint_action] = opponent_rollout_results[idx]

        start += count
    assert start == len(
        bilateral_joint_actions
    ), f"{start * num_sample}, {len(bilateral_joint_actions)}"

    return value_table_cache


def filter_invalid_actions_from_policy(
    power_policies: PowerPolicies, game: pydipcc.Game
) -> PowerPolicies:
    """Remove actions that cannot be evaluated by base_strategy_model and renormalize policy

    We only consider movement phase as br_corr_search only works in movement phase
    These invalid actions include:
      - wrong number of order
      - order out of base_strategy_model vocab
      - order impossible given current game state
    """
    assert "MOVEMENT" in game.phase, game.phase
    filtered_power_policies: PowerPolicies = {}
    orderable_locations = game.get_orderable_locations()
    all_possible_orders = game.get_all_possible_orders()

    for power, policy in power_policies.items():
        if len(policy) == 1 and list(policy.keys())[0] == ():
            filtered_power_policies[power] = policy
            continue

        possible_orders = []
        for loc in orderable_locations[power]:
            for order in all_possible_orders[loc]:
                possible_orders.append(order)

        sum_prob = 0
        filtered_policy = {}
        for action, prob in policy.items():
            keep = True
            if len(action) != len(orderable_locations[power]):
                num_missing = len(orderable_locations[power]) - len(action)
                logging.warning(
                    f"WARNING, INVALID ACTION (maybe parlai gave an action base_strategy_model doesn't like, or we have a bug in order formatting): Remove {action}: missing {num_missing} orders"
                )
                continue

            for order in action:
                if order not in ORDER_VOCABULARY_TO_IDX:
                    logging.warning(
                        f"WARNING, INVALID ACTION (maybe parlai gave an action base_strategy_model doesn't like, or we have a bug in order formatting): Remove {action}: {order} is not in order vocab"
                    )
                    keep = False
                    break

                if order not in possible_orders:
                    logging.warning(
                        f"WARNING, INVALID ACTION (maybe parlai gave an action base_strategy_model doesn't like, or we have a bug in order formatting): Remove {action}: {order} is not in possible orders"
                    )
                    keep = False
                    break

            if keep:
                filtered_policy[action] = prob
                sum_prob += prob

        if len(filtered_policy) == 0:
            filtered_policy = {(): 1.0}
        elif abs(sum_prob - 1) > 1e-5:
            for action, prob in filtered_policy.items():
                filtered_policy[action] /= sum_prob

        filtered_power_policies[power] = filtered_policy
    return filtered_power_policies


def rescore_bp_from_bilateral_views(
    game: pydipcc.Game,
    bp_policy: PowerPolicies,
    agent_power: Power,
    order_sampler: PlausibleOrderSampler,
) -> Dict[Power, PowerPolicies]:
    """Rescore bp from all bilateral views between (agent_power, pwr) for pwr in bp_policy

    Return value ret[pwr] contains the rescored policy from (agent_power, pwr)'s view.
    For each power_policy in ret[pwr], only the policies of agent_power and pwr are rescored.
    Policies for the rest of the powers are the same as their bp_policy.
    """
    speaking_power = []
    game_views = []
    list_bp_policy: List[PowerPolicies] = []
    list_include_powers: List[List[Power]] = []
    living_opponents = []
    for opponent, policy in bp_policy.items():
        if opponent == agent_power:
            continue
        if len(policy) == 1 and list(policy.keys())[0] == ():
            continue

        living_opponents.append(opponent)
        speaking_power.append(agent_power)
        game_views.append(
            game_from_two_party_view(game, agent_power, opponent, add_message_to_all=False)
        )
        list_include_powers.append([agent_power, opponent])
        list_bp_policy.append(bp_policy)

    rescored_policies = order_sampler.rescore_actions_parlai_multi_games(
        game_views, speaking_power, list_bp_policy, list_include_powers
    )
    return dict(zip(living_opponents, rescored_policies))


def _multi_gpu_base_strategy_model_rollouts(
    game: pydipcc.Game,
    all_power_base_strategy_model: MultiProcessBaseStrategyModelExecutor,
    agent_power: Power,
    joint_actions: List[JointAction],
    player_rating: Optional[float],
) -> torch.Tensor:
    """Compute base_strategy_model rollouts on for joint_actions with MultiProcessBaseStrategyModelExecutor

    Return tensor of shape [len(joint_actions), 7, 1]
    """
    futures = []
    num_workers = all_power_base_strategy_model.num_workers()
    num_actions = len(joint_actions)
    logging.info(f"total rollouts {num_actions}, num workers: {num_workers}")
    for i in range(num_workers):
        joint_actions_per_worker = joint_actions[
            i * num_actions // num_workers : (i + 1) * num_actions // num_workers
        ]
        if len(joint_actions_per_worker) == 0:
            continue
        futures.append(
            all_power_base_strategy_model.rollout(
                game,
                agent_power=agent_power,
                set_orders_dicts=joint_actions_per_worker,
                player_rating=player_rating,
            )
        )

    rollout_results = []
    for future in futures:
        rollout_results.append(future.result())
    return torch.cat(rollout_results, 0)
