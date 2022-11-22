#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent, NoAgentState
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power
from .parlai_order_handler import ParlaiOrderHandler
from parlai_diplomacy.wrappers.factory import load_order_wrapper


class ParlaiNoPressAgent(BaseAgent):
    def __init__(self, cfg: agents_cfgs.ParlaiNoPressAgent):
        self.model_orders = load_order_wrapper(cfg.model_orders)
        self.order_handler = ParlaiOrderHandler(self.model_orders)

    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        return self.order_handler.get_orders(game, power)
