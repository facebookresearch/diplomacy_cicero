#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.agents.base_search_agent import BaseSearchAgent
import numpy as np

import torch
import conf.conf_cfgs
from conf import misc_cfgs
from fairdiplomacy.agents.player import Player
from fairdiplomacy.typedefs import Action, Order, Power, PowerPolicies
from typing import Optional, Dict, List
import logging
from collections import defaultdict
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.models.state_space import get_order_vocabulary

import heyhi

from fairdiplomacy.situation_check_press import run_pseudoorder_annotation_test
from fairdiplomacy.timestamp import Timestamp
from fairdiplomacy.utils.game import game_from_view_of


def order_prob(prob_distributions: PowerPolicies, *expected_orders) -> float:
    # sanity check for stupidity
    for o in expected_orders:
        assert o in get_order_vocabulary(), o

    total = 0
    for pd in prob_distributions.values():
        for orders, prob in pd.items():
            if all(x in orders for x in expected_orders):
                total += prob
    return total


def fragment_prob(prob_distributions: PowerPolicies, power: Power, fragment) -> float:
    total = 0
    for orders, prob in prob_distributions[power].items():
        if any(fragment in x for x in orders):
            total += prob
    return total


def considers_orders(prob_distributions: PowerPolicies, *expected_orders) -> bool:
    # sanity check for stupidity
    for o in expected_orders:
        assert o in get_order_vocabulary(), o

    seen = False
    for pd in prob_distributions.values():
        for orders in pd:
            if all(x in orders for x in expected_orders):
                seen = True
    return seen


def _parse_extra_plausible_orders(string: str) -> Dict[Power, List[Action]]:
    plausible_orders = {}
    for power_orders_str in string.split(";"):
        power_orders_str = power_orders_str.strip()
        # Ignore ")'(" so one can copy things from how we print order in the terminal.
        power_orders_str = power_orders_str.replace("'", "")
        power_orders_str = power_orders_str.replace("(", "")
        power_orders_str = power_orders_str.replace(")", "")
        if not power_orders_str:
            continue
        try:
            power, rest = power_orders_str.upper().split(":")
        except ValueError:
            raise ValueError(f"Excpected '<power>: orders'. Got: {power_orders_str}")
        assert power in POWERS, power
        if power not in plausible_orders:
            plausible_orders[power] = []
        plausible_orders[power].append(
            tuple(order.strip() for order in rest.split(",") if order.strip())
        )
    return plausible_orders


def run_situation_checks(situations: List[misc_cfgs.GameSituation], agent, **kwargs):
    policy_situations = [s for s in situations if "policy" in s.tags]
    policy_results = run_policy_situation_checks(policy_situations, agent, **kwargs)
    orders_situations = [s for s in situations if "orders" in s.tags]
    orders_results = run_orders_situation_checks(orders_situations, agent)
    pseudo_situations = [s for s in situations if "pseudoorders" in s.tags]
    pseudo_results = run_pseudo_situation_checks(pseudo_situations, agent)
    return {**policy_results, **orders_results, **pseudo_results}


def _get_game_from_situation_and_log(situation: misc_cfgs.GameSituation, dss: bool):
    logging.info("=" * 80)
    comment = situation.comment
    logging.info(f"{situation.name}: {comment} (phase={situation.phase})")
    game_path = situation.game_path
    assert game_path is not None
    # If path is not absolute, treat as relative to code root.
    game_path = heyhi.PROJ_ROOT / game_path
    logging.info(f"path: {game_path}")
    with open(game_path) as f:
        game = Game.from_json(f.read())
    if situation.phase is not None:
        game = game.rolled_back_to_phase_end(situation.phase)

    if situation.pov_power:
        game = game_from_view_of(game, situation.pov_power)

    if situation.time_sent is not None and situation.time_sent != -1:
        game = game.rolled_back_to_timestamp_end(Timestamp(situation.time_sent))

    if dss:
        game.set_scoring_system(Game.SCORING_DSS)

    return game


def record_test_result(results: Dict, test_idx: int, sit_name: str, test, passed: int) -> None:
    logging.info(f"Test: {test.test}")
    res_string = "PASSED" if passed else "FAILED"
    logging.info(f"Result: {res_string:8s}  {sit_name:20s} {test.name}")
    logging.info(f"        {test.test}")
    results[f"{sit_name}.{test_idx}"] = int(passed)


def run_policy_situation_checks(
    situations: List[misc_cfgs.GameSituation],
    agent,
    extra_plausible_orders_str: Optional[str] = None,
    n_samples: Optional[int] = None,
    single_power: Optional[Power] = None,
    dss: bool = False,
) -> Dict[str, int]:
    extra_plausible_orders_str = (
        "" if extra_plausible_orders_str is None else extra_plausible_orders_str
    )
    n_samples = 100 if n_samples is None else n_samples

    extra_plausible_orders: Optional[Dict[Power, List[Action]]]
    if extra_plausible_orders_str:
        assert isinstance(agent, SearchBotAgent)
        extra_plausible_orders = _parse_extra_plausible_orders(extra_plausible_orders_str)
    else:
        extra_plausible_orders = None

    powers = [single_power] if single_power else POWERS
    results = {}
    for situation in situations:
        if "policy" not in situation.tags:
            continue
        name = situation.name or ""
        game = _get_game_from_situation_and_log(situation, dss)

        if single_power:
            assert not situation.pov_power or situation.pov_power == single_power, (
                situation.pov_power,
                single_power,
            )
            game = game_from_view_of(game, single_power)

        agent_power = single_power or situation.pov_power or "AUSTRIA"

        if isinstance(agent, BaseSearchAgent):
            # NOTE: State for stateful agents is created fresh here every time.
            # For situation checks we don't go back through the whole game history.
            player = Player(agent, agent_power)
            if (
                agent.br_corr_bilateral_search_cfg is not None
                and agent.br_corr_bilateral_search_cfg.enable_for_final_order
            ):
                search_result = agent.run_best_response_against_correlated_bilateral_search(
                    game=game,
                    bp_policy=None,
                    early_exit_for_power=None,
                    timings=None,
                    extra_plausible_orders=extra_plausible_orders,
                    agent_power=player.power,
                    agent_state=player.state,
                )
            else:
                search_result = player.run_search(
                    game, extra_plausible_orders=extra_plausible_orders
                )

            prob_distributions = search_result.get_agent_policy()
        else:
            # this is a supervised agent, sample N times to get a distribution
            prob_distributions = {p: defaultdict(float) for p in powers}
            for power in powers:
                # NOTE: State for stateful agents is created fresh here every time.
                # For situation checks we don't go back through the whole game history.
                player = Player(agent, power)
                for N in range(n_samples):
                    orders = player.get_orders(game)
                    prob_distributions[power][tuple(orders)] += 1 / n_samples

        if hasattr(agent, "base_strategy_model"):
            logging.info(
                "Values (if no press): %s",
                " ".join(
                    f"{p}={v:.3f}"
                    for p, v in zip(
                        powers,
                        agent.base_strategy_model.get_values(
                            game, has_press=False, agent_power=agent_power
                        ),
                    )
                ),
            )

            logging.info(
                "Values (if press): %s",
                " ".join(
                    f"{p}={v:.3f}"
                    for p, v in zip(
                        powers,
                        agent.base_strategy_model.get_values(
                            game, has_press=True, agent_power=agent_power
                        ),
                    )
                ),
            )
        for power in powers:
            pd = prob_distributions[power]
            pdl = sorted(list(pd.items()), key=lambda x: -x[1])
            logging.info(f"   {power}")

            for order, prob in pdl:
                if prob < 0.02:
                    break
                logging.info(f"       {prob:5.2f} {order}")

        for i, test in enumerate(situation.tests):
            logging.info(f"Test: {test.test}")
            test_func = eval(f"lambda r: {test.test}")
            passed = test_func(prob_distributions)
            record_test_result(results, i, name, test, passed)

    if results:
        logging.info("Passed: %d/%d", sum(results.values()), len(results))
        logging.info("JSON: %s", results)
    return results


def run_orders_situation_checks(
    situations: List[misc_cfgs.GameSituation], agent, dss: bool = False,
):
    results = {}
    for situation in situations:
        if "orders" not in situation.tags:
            continue
        name = situation.name or ""
        game = _get_game_from_situation_and_log(situation, dss)

        agent_power = situation.pov_power or "AUSTRIA"

        player = Player(agent, agent_power)
        orders = player.get_orders(game)
        logging.info(f"orders= {orders}")
        for i, test in enumerate(situation.tests):
            passed = run_pseudoorder_annotation_test(test.test or "???", list(orders))
            record_test_result(results, i, name, test, passed)

    if results:
        logging.info("Passed: %d/%d", sum(results.values()), len(results))
        logging.info("JSON: %s", results)
    return results


def run_pseudo_situation_checks(
    situations: List[misc_cfgs.GameSituation], agent, dss: bool = False,
):
    results = {}
    for situation in situations:
        if "pseudoorders" not in situation.tags:
            continue
        name = situation.name or ""
        game = _get_game_from_situation_and_log(situation, dss)

        agent_power = situation.pov_power or "AUSTRIA"

        player = Player(agent, agent_power)

        # now we have to find the relevant message that produced the error
        time_sent = situation.time_sent
        assert time_sent is not None
        timestamp = Timestamp.from_centis(time_sent)
        try:
            msg = game.messages[timestamp]
        except KeyError:
            raise RuntimeError(
                f"Didn't find a message corresponding to pseudo-order test: {situation}"
            )

        game_premsg = game.rolled_back_to_timestamp_start(timestamp)
        recipient = msg["recipient"]

        pseudo_orders = player.get_pseudo_orders(game_premsg, recipient)
        pseudoorders_flattened: List[Order] = [
            o
            for phase_joint_action in pseudo_orders.val.values()
            for _pwr, action in phase_joint_action.items()
            for o in action
        ]
        logging.info(f"Flattened pseudo orders: {pseudoorders_flattened}")
        for i, test in enumerate(situation.tests):
            passed = run_pseudoorder_annotation_test(test.test or "???", pseudoorders_flattened)
            record_test_result(results, i, name, test, passed)

    if results:
        logging.info("Passed: %d/%d", sum(results.values()), len(results))
        logging.info("JSON: %s", results)
    return results


def run_situation_check_from_cfg(cfg: conf.conf_cfgs.SituationCheckTask):
    # NEED TO SET THIS BEFORE CREATING THE AGENT!
    if cfg.seed >= 0:
        logging.info(f"Set seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)  # type:ignore

    agent = build_agent_from_cfg(cfg.agent)

    if cfg.single_game:
        # Creating fake situations from a single game.
        situations = [
            misc_cfgs.GameSituation(
                game_path=cfg.single_game,
                phase=cfg.single_phase,
                time_sent=cfg.single_timestamp,
                tags=["policy"],
            )
        ]
        logging.info("Created fake test situation JSON: %s", situations)
    else:
        assert cfg.situation_proto is not None
        situation_set: misc_cfgs.GameSituationSet = heyhi.load_config(
            heyhi.PROJ_ROOT / cfg.situation_proto, msg_class=misc_cfgs.GameSituationSet
        )

        selection = None
        if cfg.selection is not None and cfg.selection != "":
            selection = cfg.selection.split(",")
            situations = [s for s in situation_set.situations if s.name in selection]
        else:
            situations = [s for s in situation_set.situations]

    result = run_situation_checks(
        situations,
        agent,
        extra_plausible_orders_str=cfg.extra_plausible_orders,
        n_samples=cfg.n_samples,
        single_power=cfg.single_power,
    )

    print(result)

    return result
