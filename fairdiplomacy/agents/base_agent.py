#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.typedefs import Action, MessageDict, Power, Timestamp, MessageHeuristicResult

from fairdiplomacy.pseudo_orders import PseudoOrders


class AgentState(ABC):
    """A class that agents should subclass to store state that they wish
    to maintain across turns of a game, and can mutate the contents of freely
    as that state changes.

    AgentStates must be picklable.

    Generally, an AgentState expects to be used only with a single game,
    and only on successive chronologically increasing phases of that game, except
    multiple or repeated method calls with the same state on the same phase is okay.

    An agent should NOT expect to be called on all of the phases of a game's history.
    For example, we might start in the middle of a game at S1903 during analysis or
    situation checks or other things, and never run the agent on 1901-1902, and so
    an agent's state logic should tolerate things like this.

    See also the Player module in fairdiplomacy.agents.player, which wraps
    together a BaseAgent *with* its AgentState(s) for convenience.
    """

    pass


class NoAgentState(AgentState):
    """A subclass of AgentState for agents that are stateless."""

    pass


class BaseAgent(ABC):
    """
    BaseAgents are expected to store any game-specific power-specific state
    in the AgentState object.

    See also the Player module in fairdiplomacy.agents.player, which wraps
    together a BaseAgent *with* its AgentState(s) for convenience.
    """

    def initialize_state(self, power: Power) -> AgentState:
        """Override this method to give an agent state."""
        return NoAgentState()

    @abstractmethod
    def get_orders(self, game: Game, power: Power, state: AgentState) -> Action:
        """Return a list of orders that should be taken based on the game state

        Arguments:
        - game: a fairdiplomacy.Game object
        - power: str, one of {'AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY',
                              'ITALY', 'RUSSIA', 'TURKEY'}
        - state: whatever game-specific cross-turn state the agent is maintaining.

        Returns a list of order strings, e.g.
            ["A TYR - TRI", "F ION - TUN", "A VEN S A TYR - TRI"]
        """
        raise NotImplementedError("Subclasses must implement")

    def get_orders_many_powers(self, game: Game, powers: Sequence[Power]) -> Dict[Power, Action]:
        """Return a set of orders that should be taken based on the game state per power.
        Agents that implement this method should do very carefully, and only in cases where
        all agents see the same information (e.g. not in full-press with private messages),
        and only when the agent itself doesn't depend on differing per-power state.

        Users of this method should similarly take care to understand the corner cases,
        and do so only very carefully.

        Arguments:
        - game: a fairdiplomacy.Game object
        - powers: a dict whose keys are the powers that need orders

        Returns a dict of orders for each power.
        """
        raise NotImplementedError("This agent doesn't get_orders_many_powers")

    def generate_message(
        self,
        game: Game,
        power: Power,
        timestamp: Optional[Timestamp],
        state: AgentState,
        recipient: Optional[Power] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
    ) -> Optional[MessageDict]:
        """Return a complete MessageDict, or None on failure

        Implementations must choose a recipient within.

        Implement a default here which full-press agents should override.
        """
        return None

    def can_share_strategy(self) -> bool:
        """Return true if this agent type can reasonably generate orders for all powers.
        Should be false for full-press agents because they can't observe other agents' dialogue.
        """
        return False

    def can_sleep(self) -> bool:
        return False

    def get_sleep_time(
        self, game: Game, power: Power, state: AgentState, recipient: Optional[Power],
    ) -> Timestamp:
        """
        For dialogue with a sleep classifier, determines how long the agent should sleep
        before sending another message, in seconds.
        """
        raise NotImplementedError()

    def get_pseudo_orders(
        self, game: Game, power: Power, state: AgentState, recipient: Optional[Power],
    ) -> Optional[PseudoOrders]:
        """
        For dialogue with pseudo-orders, computes pseudo-orders for a hypothetical message sent
        to this recipient as of the current state of the game.
        Otherwise, return None
        """
        return None

    def postprocess_sleep_heuristics_should_trigger(
        self, msg: MessageDict, game: Game, state: AgentState,
    ) -> MessageHeuristicResult:
        """
        Does post-processing checks on whether a message should be force-sent
        or blocked based on properties of the message, pseudo-orders, etc.
        """
        return MessageHeuristicResult.NONE
