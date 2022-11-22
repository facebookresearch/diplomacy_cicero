#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent, NoAgentState
from fairdiplomacy.typedefs import Action, Power
import json
import logging
from collections import Counter
from typing import Dict, List, Tuple

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_strategy_model_rollouts import BaseStrategyModelRollouts
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler
from fairdiplomacy.utils.parse_device import device_id_to_str


class BRSearchAgent(BaseAgent):
    """One-ply search with base_strategy_model-policy rollouts

    ## Policy
    1. Consider a set of orders that are suggested by the base_strategy_model policy network.
    2. For each set of orders, perform a number of rollouts using the base_strategy_model
    policy network for each power.
    3. Score each order set by the average supply center count at the end
    of the rollout.
    4. Choose the order set with the highest score.
    """

    def __init__(self, cfg: agents_cfgs.BRSearchAgent):
        self.base_strategy_model = BaseStrategyModelWrapper(
            cfg.model_path, device_id_to_str(cfg.device), cfg.value_model_path, cfg.max_batch_size
        )
        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg, base_strategy_model=self.base_strategy_model
        )
        self.base_strategy_model_rollouts = BaseStrategyModelRollouts(
            self.base_strategy_model, cfg.rollouts_cfg, has_press=False
        )

    def get_orders(self, game: pydipcc.Game, power: Power, state: AgentState) -> Action:
        assert isinstance(state, NoAgentState)
        assert isinstance(game, pydipcc.Game)

        plausible_orders = list(
            self.order_sampler.sample_orders(game, agent_power=power).get(power, {}).keys()
        )
        logging.info("Plausible orders: {}".format(plausible_orders))

        if len(plausible_orders) == 0:
            return tuple()
        if len(plausible_orders) == 1:
            return plausible_orders.pop()

        results = self.base_strategy_model_rollouts.do_rollouts(
            game,
            agent_power=power,
            set_orders_dicts=[{power: orders} for orders in plausible_orders],
        )

        return self.best_order_from_results(results, power)

    @classmethod
    def best_order_from_results(cls, results: List[Tuple[Dict, Dict]], power) -> Action:
        """Given a set of rollout results, choose the move to play

        Arguments:
        - results: List[Tuple[set_orders_dict, all_scores]], where
            -> set_orders_dict: Dict[power, orders] on first turn
            -> all_scores: Dict[power, supply count], e.g. {'AUSTRIA': 6, 'ENGLAND': 3, ...}
        - power: the power making the orders, e.g. "ITALY"

        Returns:
        - the orders with the highest average score for power
        """
        order_scores = Counter()
        order_counts = Counter()

        for set_orders_dict, all_scores in results:
            orders = set_orders_dict[power]
            order_scores[orders] += all_scores[power]
            order_counts[orders] += 1

        order_avg_score = {
            orders: order_scores[orders] / order_counts[orders] for orders in order_scores
        }
        logging.info("order_avg_score: {}".format(order_avg_score))
        return max(order_avg_score.items(), key=lambda kv: kv[1])[0]
