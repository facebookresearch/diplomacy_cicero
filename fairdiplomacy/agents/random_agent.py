#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from fairdiplomacy.pydipcc import Game

from fairdiplomacy.typedefs import Power

from .base_agent import AgentState, BaseAgent, NoAgentState


class RandomAgent(BaseAgent):
    def __init__(self, cfg):
        del cfg  # Not used.

    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        possible_orders = {
            loc: orders
            for loc, orders in game.get_all_possible_orders().items()
            if loc in game.get_orderable_locations()[power]
        }
        return [random.choice(orders) for orders in possible_orders.values()]
