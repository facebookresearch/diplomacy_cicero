#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import numpy as np
import random

from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import AgentState, NoAgentState
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power

from .br_search_agent import BRSearchAgent
from .plausible_order_sampling import PlausibleOrderSampler
from parlai_diplomacy.wrappers.factory import load_order_wrapper


class ParlAIBestResponseOrderHandler(BRSearchAgent):
    def __init__(self, cfg: agents_cfgs.ParlAIBestResponseOrderHandler):
        super().__init__(cfg)
        if cfg.parlai_model.model_path:
            parlai_model = load_order_wrapper(cfg.parlai_model)
        else:
            parlai_model = None

        self.order_sampler = PlausibleOrderSampler(cfg.plausible_orders, parlai_model=parlai_model)
        logging.info("Finished init ParlAIBestResponseOrderHandler")

    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        logging.info(
            f"Getting parlai plausible orders, req_size={self.plausible_orders_req_size} batch_size={self.plausible_orders_batch_size}"
        )
        all_plausible_orders_and_scores = self.order_sampler.sample_orders(game, agent_power=power)
        plausible_orders = list(all_plausible_orders_and_scores[power].keys())

        if len(plausible_orders) == 0:
            return []
        if len(plausible_orders) == 1:
            return list(plausible_orders.pop())

        # other powers play their sampled distribution
        set_orders_dicts = [{} for _ in range(self.rollouts_per_plausible_order)]
        for opp_power, action_to_logp in all_plausible_orders_and_scores.items():
            if power == opp_power or len(action_to_logp) == 0:
                continue

            # get opponent action distribution
            actions, logps = list(zip(*action_to_logp.items()))
            p = np.exp(logps)
            p /= p.sum()

            # set opponent action according to distribution
            rollouts_per_opp_action = quantize(p, self.rollouts_per_plausible_order)
            base = 0
            for action, n in zip(actions, rollouts_per_opp_action):
                for i in range(base, base + n):
                    set_orders_dicts[i][opp_power] = action
                base += n

            # shuffle opponent action matchups
            random.shuffle(set_orders_dicts)

        # fill in this power's plausible orders
        set_orders_dicts = [
            {**opp_dict, power: action}
            for opp_dict in set_orders_dicts
            for action in plausible_orders
        ]

        # do rollouts, possibly in chunks to reduce batch size
        n_chunks, chunk_size = self.get_chunk_size(
            len(plausible_orders), self.rollouts_per_plausible_order
        )
        results = []
        for i in range(n_chunks):
            results.extend(
                self.do_rollouts(
                    game,
                    agent_power=power,
                    set_orders_dicts=set_orders_dicts[(i * chunk_size) : ((i + 1) * chunk_size)],
                )
            )

        return self.best_order_from_results(results, power)


def quantize(p, n=256):
    """Return a list of integers that follow distribution p and sum to exactly n"""
    p = np.array(p)
    r = (p * n).round().astype(np.int)
    rsum = r.sum()
    if rsum == n:
        return r

    rerr = np.abs((p * n) - r)

    delta = 1 if rsum < n else -1
    needed = abs(rsum - n)

    for idx in list(reversed(np.argsort(rerr)))[:needed]:
        r[idx] += delta

    assert r.sum() == n
    return r
