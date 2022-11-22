#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict, Generic, List, Optional, TypeVar
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.base_search_agent import BaseSearchAgent, SearchResult
from fairdiplomacy.typedefs import (
    Action,
    MessageDict,
    PlausibleOrders,
    Power,
    PowerPolicies,
    Timestamp,
)
from fairdiplomacy.utils.timing_ctx import TimingCtx

from fairdiplomacy.pseudo_orders import PseudoOrders


AnyAgent = TypeVar("AnyAgent", bound=BaseAgent)


class Player(Generic[AnyAgent]):
    """Wraps together BaseAgent and AgentState. Also handles BaseSearchAgent.

    A player is like an Agent (i.e. subclass of BaseAgent), except it also holds the
    the state of that agent, possibly per-power.

    Exactly one of these should be created per game that this agent is involved in.
    It supports holding states for multiple powers, so creating a single one for
    an agent_six for a game is fine, for example.

    It should NOT be reused for multiple games, instead a new Player should be created
    for each new game. It should also be be saved and reloaded via state_dict and
    load_state_dict in any situation where state needs to be preserved across
    preemption of a job.

    Generally, a player expects its methods to be called only on a single game,
    and only on successive chronologically increasing phases of that game, except
    multiple or repeated method calls on the same phase is okay.
    """

    def __init__(self, agent: AnyAgent, power: Power):
        self.agent = agent
        self.power = power
        self.state = agent.initialize_state(power)

    def state_dict(self) -> Any:
        return {"power": self.power, "state": self.state}

    def load_state_dict(self, state: Any):
        self.power = state["power"]
        self.state = state["state"]

    def get_orders(self, game: Game) -> Action:
        return self.agent.get_orders(game, self.power, self.state)

    def generate_message(
        self,
        game: Game,
        timestamp: Optional[Timestamp] = None,
        recipient: Optional[Power] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
    ) -> Optional[MessageDict]:
        assert self.power is not None
        return self.agent.generate_message(
            game,
            self.power,
            timestamp,
            self.state,
            recipient=recipient,
            pseudo_orders=pseudo_orders,
        )

    def can_share_strategy(self) -> bool:
        return self.agent.can_share_strategy()

    def can_sleep(self) -> bool:
        return self.agent.can_sleep()

    def get_sleep_time(self, game: Game, recipient: Optional[Power]) -> Timestamp:
        assert self.power is not None
        return self.agent.get_sleep_time(game, self.power, self.state, recipient=recipient)

    def run_search(
        self,
        game: Game,
        *,
        bp_policy: PowerPolicies = None,
        allow_early_exit: bool = False,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
    ) -> SearchResult:
        assert isinstance(
            self.agent, BaseSearchAgent
        ), f"run_search called on player that uses non-search agent: {type(self.agent)}"
        return self.agent.run_search(
            game=game,
            bp_policy=bp_policy,
            early_exit_for_power=(self.power if allow_early_exit else None),
            timings=timings,
            extra_plausible_orders=extra_plausible_orders,
            agent_power=self.power,
            agent_state=self.state,
        )

    def get_plausible_orders_policy(self, game: Game) -> PowerPolicies:
        assert isinstance(
            self.agent, BaseSearchAgent
        ), f"get_plausible_orders_policy called on player that uses non-search agent: {type(self.agent)}"
        return self.agent.get_plausible_orders_policy(
            game=game, agent_power=self.power, agent_state=self.state,
        )

    def get_pseudo_orders(self, game: Game, recipient: Power) -> Optional[PseudoOrders]:
        return self.agent.get_pseudo_orders(
            game=game, power=self.power, state=self.state, recipient=recipient
        )
