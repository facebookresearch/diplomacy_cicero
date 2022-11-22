#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Dict, Optional, Sequence

import torch
import torch.cuda

from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent, NoAgentState
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power
from fairdiplomacy.utils.parse_device import device_id_to_str


class BaseStrategyModelAgent(BaseAgent):
    def __init__(self, cfg: agents_cfgs.BaseStrategyModelAgent):
        self.device = device_id_to_str(cfg.device)
        if not torch.cuda.is_available() and self.device != "cpu":
            logging.warning("Using cpu because cuda not available")
            self.device = "cpu"
        self.model = BaseStrategyModelWrapper(
            cfg.model_path, device=self.device, half_precision=cfg.half_precision
        )
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.has_press = cfg.has_press

    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return tuple()
        return self._get_orders_many_powers(game, [power], agent_power=power)[power]

    def get_orders_many_powers(self, game: Game, powers: Sequence[Power]):
        return self._get_orders_many_powers(game, powers, agent_power=None)

    def _get_orders_many_powers(
        self, game: Game, powers: Sequence[Power], agent_power: Optional[Power]
    ):
        assert self.temperature is not None
        temperature = self.temperature
        top_p = self.top_p
        actions, _logprobs = self.model.forward_policy(
            [game],
            has_press=self.has_press,
            agent_power=None,
            temperature=temperature,
            top_p=top_p,
        )
        actions = actions[0]  # batch dim
        return {p: a for p, a in zip(POWERS, actions) if p in powers}

    def can_share_strategy(self) -> bool:
        return True
