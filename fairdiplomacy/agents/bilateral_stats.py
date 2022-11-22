#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
from typing import Any, Dict, TYPE_CHECKING

from fairdiplomacy.game import POWERS
import logging
from fairdiplomacy.typedefs import Action, JointAction, PlausibleOrders, Power, RolloutResults
from fairdiplomacy import pydipcc
import tabulate

if TYPE_CHECKING:
    from fairdiplomacy.agents.searchbot_agent import CFRData
else:
    CFRData = Any


def hcat(strs, sep="    |    "):
    """Concatenate multi-line strings horizontally."""

    all_lines = []
    for s in strs:
        lines = s.split("\n")
        max_len = max(len(line) for line in lines)
        lines = [line.ljust(max_len) for line in lines]
        if all_lines:
            assert len(lines) == len(all_lines[0])
        all_lines.append(lines)
    return "\n".join(sep.join(fields) for fields in zip(*all_lines))


class WeightedAverager:
    def __init__(self):
        self._cum = 0
        self._weight = 0
        self._count = 0

    def __repr__(self) -> str:
        return f"{self.get_avg()}"

    def accum(self, val, weight):
        self._cum += val * weight
        self._weight += weight
        self._count += 1

    def get_avg(self):
        return self._cum / (self._weight + 1e-8)

    def get_weight(self):
        return self._weight

    def get_count(self):
        return self._count


class BilateralStats:
    def __init__(
        self, game: pydipcc.Game, agent_power: Power, plausible_orders: PlausibleOrders,
    ):
        self.phase = game.phase
        self.agent_power = agent_power
        self.plausible_orders = plausible_orders

        # the value to me of each of pwr's actions
        self.value_to_me = defaultdict(WeightedAverager)
        # the value to me of each bilateral action (a_me, a_pwr)
        self.bilateral_value_me = defaultdict(WeightedAverager)
        # the value to pwr of each bilateral action (a_me, a_pwr)
        self.bilateral_value_pwr = defaultdict(WeightedAverager)
        # the probability of playing a joint bilateral action
        self.bilateral_action_prob = defaultdict(WeightedAverager)

    def accum_bilateral_probs(self, sampled_action: JointAction, weight: float):
        """Accumulate bilateral probabilities from the joint sampled action at each CFR iter."""

        agent_power = self.agent_power
        for other_pwr in sampled_action.keys():
            if other_pwr == agent_power:
                continue

            for my_idx, my_action in enumerate(self.plausible_orders[agent_power]):
                for pwr_idx, pwr_action in enumerate(self.plausible_orders[other_pwr]):
                    key = (other_pwr, my_action, pwr_action)
                    was_sampled = (
                        1
                        if my_action == sampled_action[agent_power]
                        and pwr_action == sampled_action[other_pwr]
                        else 0
                    )
                    self.bilateral_action_prob[key].accum(was_sampled, weight=weight)

    def accum_bilateral_values(self, pwr: Power, cfr_iter: int, rollout_results: RolloutResults):
        """Accumulate bilateral values from the RolloutResults for pwr's actions, at each CFR iteration."""
        # orders : JointAction, e.g. {"AUSTRIA": ["VIE H", ...], ...}
        # values : JointActionValue e.g. {"AUSTRIA": 0.343, ...}
        agent_power = self.agent_power
        for orders, values in rollout_results:
            self.value_to_me[pwr, orders[pwr]].accum(values[agent_power], weight=cfr_iter)
            if pwr == agent_power:
                for other_pwr in values:
                    if pwr == other_pwr:
                        continue
                    key = (other_pwr, orders[agent_power], orders[other_pwr])
                    self.bilateral_value_me[key].accum(values[agent_power], weight=cfr_iter)
                    self.bilateral_value_pwr[key].accum(values[other_pwr], weight=cfr_iter)
            else:
                key = (pwr, orders[agent_power], orders[pwr])
                self.bilateral_value_me[key].accum(values[agent_power], weight=cfr_iter)
                self.bilateral_value_pwr[key].accum(values[pwr], weight=cfr_iter)

    def log(self, cfr_data: CFRData, min_order_prob: float) -> None:
        """Log the bilateral values and joint action probabilities between agent_power and each other power."""
        for power in POWERS:
            self.log_power(cfr_data, min_order_prob, power)

    def log_power(self, cfr_data: CFRData, min_order_prob: float, other_pwr: Power) -> None:
        """Log the bilateral values and joint action probabilities between agent_power and other_pwr"""
        agent_power = self.agent_power

        if len(self.plausible_orders[other_pwr]) == 0:
            return

        logging.info(
            f"<B> {self.phase}: Value to {agent_power} of {other_pwr} actions (base {cfr_data.avg_utility(agent_power):8.5f}):"
        )
        logging.info(f"<B>    {'idx':5s} {'prob':8s}  {'bp_p':8s}  {'value':8s}  {'orders':8s}")

        pwr_eq_strat = cfr_data.avg_policy(other_pwr)
        pwr_bp_strat = cfr_data.bp_policy(other_pwr)
        my_eq_strat = cfr_data.avg_policy(agent_power)

        # 3a. Compute valid pwr_actions and log the values of each pwr to agent_power
        valid_pwr_actions = []
        for idx, (pwr_action, sort_prob) in enumerate(pwr_eq_strat.items()):
            if sort_prob < min_order_prob:
                break
            valid_pwr_actions.append(pwr_action)

            eq_prob = pwr_eq_strat[pwr_action]
            bp_prob = pwr_bp_strat[pwr_action]
            averager = self.value_to_me[other_pwr, pwr_action]
            logging.info(
                f"<B>    [{idx}] {eq_prob:8.5f}  {bp_prob:8.5f} {averager.get_avg():8.5f}  {pwr_action}"
            )

        # 3b. log the bilateral action-value and order-prob matrices for this power and agent_power
        if other_pwr != agent_power:

            # the expected marginal probability for an action
            bilateral_action_me = defaultdict(WeightedAverager)
            bilateral_action_pwr = defaultdict(WeightedAverager)
            my_marg = cfr_data.avg_strategy(agent_power)
            pwr_marg = cfr_data.avg_strategy(other_pwr)
            for my_idx, my_action in enumerate(self.plausible_orders[agent_power]):
                for pwr_idx, pwr_action in enumerate(self.plausible_orders[other_pwr]):
                    key = (other_pwr, my_action, pwr_action)
                    bilateral_action_me[key].accum(my_marg[my_idx], weight=1)
                    bilateral_action_pwr[key].accum(pwr_marg[pwr_idx], weight=1)

            def make_bilateral_table(averager):
                table = []
                for pwr_action, sort_prob in pwr_eq_strat.items():
                    if sort_prob < min_order_prob:
                        break
                    table.append([])
                    for my_action, my_prob in my_eq_strat.items():
                        if my_prob < min_order_prob:
                            break
                        key = (other_pwr, my_action, pwr_action)
                        table[-1].append(averager[key].get_avg())
                return table

            bilateral_table_value_me = make_bilateral_table(self.bilateral_value_me)
            bilateral_table_value_pwr = make_bilateral_table(self.bilateral_value_pwr)
            bilateral_table_probs = make_bilateral_table(self.bilateral_action_prob)
            bilateral_table_probs_me = make_bilateral_table(bilateral_action_me)
            bilateral_table_probs_pwr = make_bilateral_table(bilateral_action_pwr)

            if len(bilateral_table_value_me) and len(bilateral_table_value_me[0]):
                headers = [f"{other_pwr[:3]} \\ {agent_power[:3]}"] + [
                    f"[{idx}]" for idx in range(len(bilateral_table_value_me[0]))
                ]
                u_avg_me = cfr_data.avg_utility(agent_power)
                u_avg_pwr = cfr_data.avg_utility(other_pwr)

                def construct_tabular(tables, title, F=lambda e: e):
                    T = [
                        [f"[{idx}]"] + [F(*cells) for cells in zip(*rows)]
                        for idx, rows in enumerate(zip(*tables))
                    ]
                    return f"{title}\n" + tabulate.tabulate(T, headers, floatfmt=".3f")

                tbl_me = construct_tabular(
                    (bilateral_table_value_me,),
                    F=lambda u: u - u_avg_me,
                    title=f"Change in value to {agent_power} (base {u_avg_me:.3f}):",
                )
                tbl_pwr = construct_tabular(
                    (bilateral_table_value_pwr,),
                    F=lambda u: u - u_avg_pwr,
                    title=f"Change in value to {other_pwr}  (base {u_avg_pwr:.3f}):",
                )
                tbl_sum = construct_tabular(
                    (bilateral_table_value_me, bilateral_table_value_pwr),
                    F=lambda u_m, u_p: u_m + u_p - u_avg_me - u_avg_pwr,
                    title="Change in value (sum):",
                )
                tbl_min = construct_tabular(
                    (bilateral_table_value_me, bilateral_table_value_pwr),
                    F=lambda u_m, u_p: max(min(u_m - u_avg_me, u_p - u_avg_pwr), 0),
                    title="Min positive change in value:",
                )

                logging.info("\n" + hcat((tbl_me, tbl_pwr)))
                logging.info("\n" + hcat((tbl_sum, tbl_min)))

                tbl_probs = construct_tabular((bilateral_table_probs,), title="Joint probs:")
                # n.b. my conditional is joint divided by pwr marginal, and vice versa
                EPS = 1e-10
                tbl_conditional_me = construct_tabular(
                    (bilateral_table_probs, bilateral_table_probs_pwr),
                    F=lambda p, pi: p / (pi + EPS),
                    title=f"Conditional probs {agent_power}:",
                )
                tbl_conditional_pwr = construct_tabular(
                    (bilateral_table_probs, bilateral_table_probs_me),
                    F=lambda p, pi: p / (pi + EPS),
                    title=f"Conditional probs {other_pwr}:",
                )
                logging.info("\n" + hcat((tbl_probs, tbl_conditional_me, tbl_conditional_pwr)))
