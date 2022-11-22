#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Various useful functions for analysing and printing policies and values"""
from collections import defaultdict
from dataclasses import dataclass
import typing

import numpy as np
from fairdiplomacy.agents.player import Player
from fairdiplomacy.game import sort_phase_key
import pathlib
from fairdiplomacy.agents.base_search_agent import BaseSearchAgent, SearchResult
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
import logging
from typing import Any, Optional, Dict, List, Sequence, Set, Tuple, Union

from fairdiplomacy.agents.searchbot_agent import SearchBotAgent, CFRData, CFRResult
from fairdiplomacy.typedefs import Phase, Power, PowerPolicies, Policy, Order
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.game import game_from_view_of, year_of_phase


def get_order_probs(policy: Policy) -> Dict[Order, float]:
    """Flatten a policy down to probs on individual orders"""
    order_probs = defaultdict(float)
    for action, prob in policy.items():
        for order in action:
            order_probs[order] += prob
    return order_probs


def get_top_order_probs(
    policy: Policy, desired_per_location_prob_mass: float = 0.98
) -> Dict[Order, float]:
    """Flatten a policy down to probs on individual orders.
    Keep only the top probability mass of orders per board location."""
    order_probs = get_order_probs(policy)
    unit_probs = defaultdict(list)
    for order, prob in order_probs.items():
        unit = " ".join(order.split()[:2])
        unit_probs[unit].append((order, prob))

    filtered_orders_and_probs = {}
    for unit, orders_and_probs in unit_probs.items():
        orders_and_probs = sorted(orders_and_probs, key=(lambda order_prob: -order_prob[1]))
        probsum = 0
        for order, prob in orders_and_probs:
            filtered_orders_and_probs[order] = prob
            probsum += prob
            if probsum >= desired_per_location_prob_mass:
                break
    return filtered_orders_and_probs


def compute_cross_regret_from_cfrdata(
    cfr_data1: CFRData, cfr_data2: CFRData, power: Power
) -> float:
    """How much worse does power i do if they play pi2^i in equilibrium pi1.

    This one is computed from the historical cfr_data from the equilibrium computations.
    """
    plaus = cfr_data1.power_plausible_orders[power]
    prob1 = cfr_data1.avg_strategy(power)
    prob2 = cfr_data2.avg_strategy(power)
    cross_regret = 0
    for a, p1, p2 in zip(plaus, prob1, prob2):
        u1 = cfr_data1.avg_action_utility(power, a)
        cross_regret += (p1 - p2) * u1
    return cross_regret


def compute_cross_regret(
    game,
    agent: SearchBotAgent,
    pi1: PowerPolicies,
    pi2: PowerPolicies,
    u1: Optional[float],
    power: Power,
) -> float:
    """How much worse does power i do if they play pi2^i in equilibrium pi1.

    This one is computed by doing independent rollouts.
    """

    u21 = agent.eval_policy_values(game, {**pi1, power: pi2[power]}, agent_power=None)[power]
    if u1 is None:
        u1 = agent.eval_policy_values(game, pi1, agent_power=None)[power]

    return u1 - u21


class CFRResultSummary:
    """Summarizes the results of one or more runs of searchbot to be printed out or logged.

    For compactness and easier understanding, the summary flattens down the policies
    to just the top per-order probabilities, but it preserves the different probabilities for
    each searchbot run separately, to make it easy to see the variability of the policies
    due to seed between separate searchbot runs in the same position."""

    bp_top_order_probs_per_power: Dict[Power, Dict[Order, float]]
    # For each power, dict maps order -> probability of that order in each CFR run.
    cfr_top_order_probs_per_power: Dict[Power, Dict[Order, List[float]]]
    utilities: Dict[Power, List[float]]
    cross_regrets: Dict[Power, List[float]]

    bp_argmax_orders: Dict[Power, Dict[Order, float]]
    cfr_argmax_orders: Dict[Power, Dict[Order, float]]
    plausible_orders: Dict[Power, Set[Order]]

    def __init__(
        self,
        bp_policy: PowerPolicies,
        cfr_results: List[SearchResult],
        filter_to_powers: Optional[List[Power]] = None,
        bp_desired_per_location_prob_mass=0.97,
        cfr_desired_per_location_prob_mass=0.94,
        game: Optional[Game] = None,
        compact_print=False,
        use_population_policy=False,
    ):
        """
        Arguments:
        bp_policy: Blueprint policy for each power.
        cfr_results: List of CFRResults returned by different independent SearchBotAgent runs
        filter_to_powers: If specified, record and summarize only the policies for these powers.
        """

        self.bp_top_order_probs_per_power = {
            power: get_top_order_probs(
                policy, desired_per_location_prob_mass=bp_desired_per_location_prob_mass
            )
            for (power, policy) in bp_policy.items()
        }
        if use_population_policy:
            cfr_policies = [result.get_population_policy() for result in cfr_results]
        else:
            cfr_policies = [result.get_agent_policy() for result in cfr_results]

        # Find all orders where in at least one run of CFR the order was a top order.
        all_top_locs_and_orders = defaultdict(set)
        for powerpolicies in cfr_policies:
            for power in POWERS:
                top_orders = get_top_order_probs(
                    powerpolicies[power],
                    desired_per_location_prob_mass=cfr_desired_per_location_prob_mass,
                )
                for order in top_orders:
                    loc = order.split()[:2][-1]
                    all_top_locs_and_orders[power].add((loc, order))

        # Break down all powerpolicies to individual orders
        all_order_probs = [
            {power: get_order_probs(policy) for (power, policy) in powerpolicies.items()}
            for powerpolicies in cfr_policies
        ]

        self.bp_argmax_orders = {
            pwr: {order: 1.0 for order in max(policy.items(), key=lambda item: item[1])[0]}
            for pwr, policy in bp_policy.items()
        }
        self.cfr_argmax_orders = {pwr: defaultdict(float) for pwr in POWERS}
        for powerpolicies in cfr_policies:
            for pwr, policy in powerpolicies.items():
                argmax_action = max(policy.items(), key=lambda item: item[1])[0]
                for order in argmax_action:
                    self.cfr_argmax_orders[pwr][order] += 1.0 / len(cfr_policies)

        self.plausible_orders = {
            pwr: set([o for a in policy.keys() for o in a]) for pwr, policy in bp_policy.items()
        }

        # Accumulate final dictionary
        self.cfr_top_order_probs_per_power = {}
        for power in POWERS:
            top_order_probs_for_power = {}
            # Have one entry for each top order, sorted first by location and then by order
            for (loc, order) in sorted(list(all_top_locs_and_orders[power])):
                probs = []
                for order_probs in all_order_probs:
                    probs.append(order_probs[power][order] if order in order_probs[power] else 0.0)
                top_order_probs_for_power[order] = probs
            self.cfr_top_order_probs_per_power[power] = top_order_probs_for_power

        self.raw_scores = None
        if game is not None:
            self.raw_scores = {}
            raw_scores = game.get_scores()
            for power in POWERS:
                self.raw_scores[power] = raw_scores[POWERS.index(power)]

        self.utilities = {}
        self.cross_regrets = {}
        for power in POWERS:
            utilities_for_power = []
            cross_regrets_for_power = []
            for i in range(len(cfr_results)):
                sum_cross_regret = 0
                count = 0
                for j in range(len(cfr_results)):
                    if j == i:
                        continue
                    this_cross_regret = compute_cross_regret_from_cfrdata(
                        cfr_results[j].cfr_data, cfr_results[i].cfr_data, power  # type: ignore
                    )
                    sum_cross_regret += this_cross_regret
                    count += 1
                cross_regrets_for_power.append(sum_cross_regret / (1e-30 + count))
                utilities_for_power.append(cfr_results[i].avg_utility(power))
            self.cross_regrets[power] = cross_regrets_for_power
            self.utilities[power] = utilities_for_power

        if filter_to_powers is not None:

            def filter_dict(d: Dict[Power, Any]):
                assert filter_to_powers is not None
                return {power: data for power, data in d.items() if power in filter_to_powers}

            self.bp_top_order_probs_per_power = filter_dict(self.bp_top_order_probs_per_power)
            self.cfr_top_order_probs_per_power = filter_dict(self.cfr_top_order_probs_per_power)
            self.bp_argmax_orders = filter_dict(self.bp_argmax_orders)
            self.cfr_argmax_orders = filter_dict(self.cfr_argmax_orders)
            self.utilities = filter_dict(self.utilities)
            self.cross_regrets = filter_dict(self.cross_regrets)
        self.compact_print = compact_print

    def get_bp_single_order_prob(self, order: Order) -> float:
        for power in POWERS:
            probs = self.bp_top_order_probs_per_power.get(power, {})
            if order in probs:
                return probs[order]
        return 0.0

    def get_cfr_single_order_probs(self, order: Order) -> List[float]:
        for power in POWERS:
            probs = self.cfr_top_order_probs_per_power.get(power, {})
            if order in probs:
                return probs[order]
        return [0.0]

    def get_bp_single_order_argmax_prob(self, order: Order) -> float:
        for power in POWERS:
            probs = self.bp_argmax_orders.get(power, {})
            if order in probs:
                return probs[order]
        return 0.0

    def get_cfr_single_order_argmax_prob(self, order: Order) -> float:
        for power in POWERS:
            probs = self.cfr_argmax_orders.get(power, {})
            if order in probs:
                return probs[order]
        return 0.0

    def is_in_plausible_orders(self, order: Order) -> bool:
        for power in POWERS:
            if order in self.plausible_orders[power]:
                return True
        return False

    def __str__(self):
        s = ""

        def top_order_probs_to_str(top_order_probs):
            s = ""
            if self.compact_print:
                orders_by_unit = defaultdict(list)
                for order in top_order_probs:
                    orders_by_unit[tuple(order.split()[:2])].append(order)
                for _unit, orders in orders_by_unit.items():
                    line = ""
                    for order in orders:
                        prob = top_order_probs[order]
                        if isinstance(prob, list) or isinstance(prob, tuple):
                            line += f"  {order:22s}"
                            for p in prob:
                                line += f"  {p:.3f}"
                        else:
                            line += f"  {order:22s} {prob:.3f}"
                    s += line
                    s += "\n"
            else:
                for order, prob in top_order_probs.items():
                    if isinstance(prob, list) or isinstance(prob, tuple):
                        s += f"  {order:22s}"
                        for p in prob:
                            s += f"  {p:.3f}"
                    else:
                        s += f"  {order:22s} {prob:.3f}"
                    s += "\n"
            return s

        s += "BLUEPRINT -------------------------------------------------\n"
        for (power, top_order_probs) in self.bp_top_order_probs_per_power.items():
            s += power + "\n"
            s += top_order_probs_to_str(top_order_probs)

        s += "CFR -------------------------------------------------\n"
        if self.raw_scores is not None:
            s += "Raw Scores: "
            for (power, top_order_probs) in self.cfr_top_order_probs_per_power.items():
                utility = self.raw_scores[power]
                s += f"  {power} {utility:.3f}"
            s += "\n"

        s += "Utilites: "
        for (power, top_order_probs) in self.cfr_top_order_probs_per_power.items():
            s += f"  {power}"
            for utility in self.utilities[power]:
                s += f" {utility:.3f}"
        s += "\n"
        for (power, top_order_probs) in self.cfr_top_order_probs_per_power.items():
            s += power + "\n"
            if len(self.cross_regrets[power]) > 1:
                s += "Cross Regrets:          "
                for cross_regret in self.cross_regrets[power]:
                    s += f"  {cross_regret:.3f}"
                s += "\n"
            s += top_order_probs_to_str(top_order_probs)
        return s


@dataclass
class MultipleSearchResults:
    bp_policy: PowerPolicies
    cfr_results: List[SearchResult]


def run_multiple_cfrs(
    agent: BaseSearchAgent, n: int, game: Game, agent_power: Optional[Power], label: str = "",
) -> MultipleSearchResults:
    """Run CFR multiple times and return all resulting policies

    Arguments:
    agent: The agent to use
    n: The number of times to run
    game: Game to use
    filter_to_powers: If specified, record and summarize only the policies for these powers.
    label: label for logging
    """

    logging.info(f"Summarize multiple cfrs {label}: Computing blueprint")
    player = Player(agent, agent_power or "AUSTRIA")
    bp_policy = player.get_plausible_orders_policy(game)

    cfr_results = []
    for i in range(n):
        logging.info(f"Summarize multiple cfrs {label}: computing search policy {i + 1} / {n}")
        result = player.run_search(game, bp_policy=bp_policy)
        cfr_results.append(result)

    return MultipleSearchResults(bp_policy=bp_policy, cfr_results=cfr_results)


def _average_values(
    games_and_powers: List[Tuple[Game, List[Power]]],
    model: BaseStrategyModelWrapper,
    has_press: bool,
) -> Tuple[float, Dict[Power, float]]:
    values = model.forward_values(
        [game for (game, _) in games_and_powers], has_press=has_press, agent_power=None,
    ).numpy()
    agent_power_values = []
    for (game, powers), values in zip(games_and_powers, values):
        if not game.is_game_done:
            for power in powers:
                agent_power_values.append((power, values[POWERS.index(power)]))
        else:
            square_scores = game.get_scores()
            for power in powers:
                agent_power_values.append((power, square_scores[POWERS.index(power)]))

    average_value = float(np.mean([value for _, value in agent_power_values]))
    average_value_by_power = {
        power: float(np.mean([value for p, value in agent_power_values if p == power]))
        for power in POWERS
    }
    return average_value, average_value_by_power


def average_value_by_phase(
    games_and_powers: List[Tuple[Game, List[Power]]],
    model: Union[BaseStrategyModelWrapper, str, pathlib.Path],
    movement_only: bool,
    spring_only: bool,
    up_to_year: Optional[int],
    has_press: bool,
) -> Tuple[Dict[Phase, float], Dict[Phase, Dict[Power, float]]]:
    """Compute the average values that an agent achieved by phase over a set of games according to a model.

    Arguments:
    games_and_powers: A list of the games played and which powers that agent played in that game.
    model: The value model to use.
    movement_only: Only movement phases.
    spring_only: Only spring phases.
    up_to_year: Only up to (and including) this year.
    has_press: has_press

    Returns: A dictionary of the agent's average value by phase, and a dictionary of the
    agent's average value by phase broken down by when that agent was each power.
    """

    all_phases = set()
    for game, _ in games_and_powers:
        all_phases = all_phases.union(set(game.get_all_phase_names()))
    all_phases = sorted(list(all_phases), key=sort_phase_key)

    model = (
        model if isinstance(model, BaseStrategyModelWrapper) else BaseStrategyModelWrapper(model)
    )

    value_by_phase = {}
    value_by_power_by_phase = {}
    for phase in all_phases:
        if movement_only and not phase.endswith("M"):
            continue
        if spring_only and not phase.startswith("S"):
            continue
        if up_to_year is not None and year_of_phase(phase) > up_to_year:
            break
        rollback_games = []
        for game, powers in games_and_powers:
            try:
                game = game.rolled_back_to_phase_start(phase)
            except RuntimeError:
                pass  # if game ended before this phase, use the last phase
            rollback_games.append((game, powers))
        average_value, average_value_by_power = _average_values(rollback_games, model, has_press)
        value_by_phase[phase] = average_value
        value_by_power_by_phase[phase] = average_value_by_power
    return value_by_phase, value_by_power_by_phase


def average_value_by_year(
    games_and_powers: List[Tuple[Game, List[Power]]],
    model: Union[BaseStrategyModelWrapper, str, pathlib.Path],
    up_to_year: int,
    has_press: bool,
) -> Tuple[Dict[int, float], Dict[int, Dict[Power, float]]]:
    """Compute the average values that an agent achieved by year over a set of games according to a model.

    Arguments:
    games_and_powers: A list of the games played and which powers that agent played in that game.
    model: The value model to use.
    up_to_year: Maximum year
    has_press: has_press

    Returns: A dictionary of the agent's average value by year, and a dictionary of the
    agent's average value by year broken down by when that agent was each power.
    """
    value_by_phase, value_by_power_by_phase = average_value_by_phase(
        games_and_powers=games_and_powers,
        model=model,
        movement_only=True,
        spring_only=True,
        up_to_year=up_to_year,
        has_press=has_press,
    )

    value_by_year = {year_of_phase(phase): value for (phase, value) in value_by_phase.items()}
    value_by_power_by_year = {
        year_of_phase(phase): value_by_power
        for (phase, value_by_power) in value_by_power_by_phase.items()
    }
    return value_by_year, value_by_power_by_year
