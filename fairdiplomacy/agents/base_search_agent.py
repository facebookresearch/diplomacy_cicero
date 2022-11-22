#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Action, JointAction, PlausibleOrders, Power, PowerPolicies
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.sampling import sample_p_list

if TYPE_CHECKING:
    from fairdiplomacy.agents.searchbot_agent import CFRData
    from fairdiplomacy.agents.bqre1p_agent import BRMData


class SearchResult(abc.ABC):
    """
    Interface for the return type of BaseSearchAgent.run_search
    """

    @abc.abstractmethod
    def is_early_exit(self) -> bool:
        """Returns true if this is an early-exit search where we didn't actually compute anything."""
        pass

    @abc.abstractmethod
    def get_bp_policy(self) -> PowerPolicies:
        """Returns the blueprint policy that was used to generate the agent policy.

        Agents that don't have a blueprint policy can return an empty policy, or throw.
        """
        pass

    @abc.abstractmethod
    def get_agent_policy(self) -> PowerPolicies:
        """Returns the policy that the agent action is sampled from.

        IMPORTANT: This may not be exactly the policy that the action is sampled from,
        e.g. CFR may use the final iter to sample an action while this will return the average policy.
        So it is not safe to use this policy for e.g. control variate.
        """
        pass

    @abc.abstractmethod
    def get_population_policy(self) -> PowerPolicies:
        """Returns the predicted policy for the population of players, sorted by probability descending.
        """
        pass

    @abc.abstractmethod
    def sample_action(self, power: Power) -> Action:
        """Sample an action based on the search procedure, sorted by probability descending."""
        pass

    @abc.abstractmethod
    def avg_utility(self, power: Power) -> float:
        """Returns the average utility for this power, if everyone plays the population policy."""
        pass

    @abc.abstractclassmethod
    def avg_action_utility(self, power: Power, a: Action) -> float:
        """Returns the average utility of a for this power, if everyone plays the population policy."""
        raise NotImplementedError()

    if TYPE_CHECKING:
        # The properties here are not actually defined, instead CFRResult and BRMResult subclasses
        # of SearchResult will just have these properties. We simply put them here to make the
        # type checker happy because otherwise *every* location we access them needs us to write
        # something like cast(CFRResult,search_result).cfr_data in order to make the type checker
        # happy. The only cost is that accessing an attribute that doesn't exist will raise
        # an exception at runtime that the type checker won't catch.
        @property
        def cfr_data(self) -> "CFRData":
            """Convenience property to get cfr_data if this SearchResult is actually a CFRResult.
            Makes type checker happy without casting."""
            return getattr(self, "cfr_data")

        @property
        def brm_data(self) -> "BRMData":
            """Convenience property to get brm_data if this SearchResult is actually a BRMResult.
            Makes type checker happy without casting."""
            return getattr(self, "brm_data")


class BaseSearchAgent(BaseAgent):
    """
    Base class for equilibrium search agents that pick an action by computing
    equilibrium policies for each player.
    """

    def __init__(self, cfg):
        pass

    @abc.abstractmethod
    def run_search(
        self,
        game: Game,
        *,
        bp_policy: PowerPolicies = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],  # No default helps type-checker catch Agent vs Player
    ) -> SearchResult:
        """Run this agent's search algorithm directly on the given position.

        Currently, doing this may skip certain kinds of caching or incremental bp logic
        in some agents.

        agent_state is required so that the call signature of this method differs from that of
        fairdiplomacy.agents.player.Player.run_search, so that type checking in vscode is
        more effective. If you are sure the agent you are working with doesn't care about
        the state and don't mind raising an exception if that turns out not to be the case,
        and you don't want to use the Player interface, you can pass None for it.
        """
        pass

    @abc.abstractmethod
    def get_plausible_orders_policy(
        self,
        game: Game,
        *,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],  # No default helps type-checker catch Agent vs Player
    ) -> PowerPolicies:
        pass


def n_move_phases_later(from_phase, n):
    if n == 0:
        return from_phase
    year_idx = int(from_phase[1:-1]) - 1901
    season = from_phase[0]
    from_move_phase_idx = 2 * year_idx + (1 if season in "FW" else 0)
    to_move_phase_idx = from_move_phase_idx + n
    to_move_phase_year = to_move_phase_idx // 2 + 1901
    to_move_phase_season = "S" if to_move_phase_idx % 2 == 0 else "F"
    return f"{to_move_phase_season}{to_move_phase_year}M"


def num_orderable_units(game_state, power):
    if game_state["name"][-1] == "A":
        return abs(game_state["builds"].get(power, {"count": 0})["count"])
    if game_state["name"][-1] == "R":
        return len(game_state["retreats"].get(power, []))
    else:
        return len(game_state["units"].get(power, []))


def sample_orders_from_policy(
    power_actions: PlausibleOrders,
    power_action_probs: Union[
        Dict[Power, List[float]], Dict[Power, torch.Tensor]
    ],  # make typechecker happy (ask Adam)
) -> Tuple[Dict[Power, int], JointAction]:
    """
    Sample orders for each power from an action distribution (i.e. policy).

    Arguments:
        - power_actions: a list of plausible orders for each power
        - power_action_probs: probabilities for each of the power_actions

    Returns:
        - A dictionary of order indices for each power sampled out of the action distribution
        - A dictionary of orders for each power sampled out of the action distribution
    """
    sampled_idxs = {}
    power_sampled_orders = {}
    for power, action_probs in power_action_probs.items():
        if len(action_probs) <= 0:
            power_sampled_orders[power] = ()
        else:
            idx = sample_p_list(action_probs)
            sampled_idxs[power] = idx
            power_sampled_orders[power] = power_actions[power][idx]

    return sampled_idxs, power_sampled_orders


def make_set_orders_dicts(
    power_actions: PlausibleOrders,
    power_sampled_orders: JointAction,
    traverser_powers: List[Power] = None,
) -> List[JointAction]:
    """
    Construct a list of set_orders dicts for CFR traversal, that can be
    used as an input to BaseSearchAgent.do_rollout.

    Arguments:
        - power_actions: a list of plausible orders for each power
        - power_sampled_orders: orders for each power
        - traverser_powers: a list of powers for whom each plausible order should be
          sampled in the output dict

    Returns:
        - A list of Power -> Action dicts, where each one has one of the plausible orders
        for one of the traverser_powers, and the sampled orders for all other powers.
        Outputs are ordered by traverser_power, then by index in power_plausible_order[pwr].
    """

    if traverser_powers is None:
        traverser_powers = POWERS

    # for each power: compare all actions against sampled opponent action
    return [
        {**{p: a for p, a in power_sampled_orders.items()}, pwr: action}
        for pwr in traverser_powers
        for action in power_actions[pwr]
    ]
