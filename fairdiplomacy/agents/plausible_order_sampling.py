#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import math
from collections import Counter, defaultdict
from math import ceil
from typing import Dict, List, Optional
from parlai_diplomacy.wrappers.factory import load_order_wrapper

import torch
import torch.cuda

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import num_orderable_units
from fairdiplomacy.agents.base_strategy_model_wrapper import (
    BaseStrategyModelWrapper,
    compute_action_logprobs_from_state,
)
from fairdiplomacy.agents.parlai_order_handler import filter_orders
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Action, PlausibleOrders, Power, PowerPolicies, PlayerRating
from fairdiplomacy.utils.order_idxs import filter_out_of_vocab_orders, is_valid_build_or_destroy
from fairdiplomacy.utils.timing_ctx import TimingCtx
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper
from parlai_diplomacy.wrappers.orders import (
    ParlAIAllOrderIndependentWrapper,
    ParlAIAllOrderIndependentRolloutWrapper,
)

from fairdiplomacy.utils.slack import GLOBAL_SLACK_EXCEPTION_SWALLOWER
from fairdiplomacy.utils.parlai_multi_gpu_wrappers import (
    load_wrapper_executor,
    wrap_parlai_model_to_executor,
)


def renormalize_policy(policy: PowerPolicies) -> None:
    totals = {}
    for power, orders_to_probs in policy.items():
        assert orders_to_probs, f"Empty policy for {power} in {policy}"
        total_prob = sum(orders_to_probs.values())
        assert total_prob > 0, orders_to_probs
        totals[power] = total_prob
        for orders in orders_to_probs:
            orders_to_probs[orders] /= total_prob
    logging.info(
        "Policy probability masses before normalization: %s",
        " ".join(f"{k}={v:.2f}" for k, v in sorted(totals.items())),
    )


def cutoff_policy(policy: PowerPolicies, limits: List[int]) -> PowerPolicies:
    assert list(policy.keys()) == POWERS
    return {
        power: {
            orders: probs
            for orders, probs in sorted(orders_to_probs.items(), key=lambda x: -x[1])[:limit]
        }
        for limit, (power, orders_to_probs) in zip(limits, policy.items())
    }


class PlausibleOrderSampler:
    def __init__(
        self,
        cfg: agents_cfgs.PlausibleOrderSampling,
        base_strategy_model: Optional[BaseStrategyModelWrapper] = None,
        parlai_model: Optional[BaseWrapper] = None,
        parlai_model_cfg: Optional[agents_cfgs.ParlaiModel] = None,
    ):
        assert (
            parlai_model is None or parlai_model_cfg is None or parlai_model_cfg.model_path is None
        ), "Flags are mutually exclusive"
        self.cfg = cfg
        assert cfg.n_plausible_orders is not None
        self.n_plausible_orders = cfg.n_plausible_orders
        assert self.n_plausible_orders > 0
        self.max_actions_units_ratio = cfg.max_actions_units_ratio
        self.exclude_n_holds = cfg.exclude_n_holds
        self.req_size = cfg.req_size
        self.batch_size = cfg.batch_size or self.req_size
        self.base_strategy_model = base_strategy_model
        if parlai_model_cfg is not None and parlai_model_cfg.model_path is not None:
            self.parlai_model_executor = load_wrapper_executor(
                parlai_model_cfg, load_order_wrapper, cfg.allow_multi_gpu, True
            )
        elif parlai_model is not None:
            self.parlai_model_executor = wrap_parlai_model_to_executor(parlai_model)
        else:
            self.parlai_model_executor = None
        self.parlai_model = (
            self.parlai_model_executor.get_model()
            if self.parlai_model_executor is not None
            else None
        )

        self.parlai_req_size: int = cfg.parlai_req_size
        self.parlai_batch_size = cfg.parlai_batch_size
        self.parlai_take_first = cfg.parlai_take_first
        self.do_parlai_rescoring = cfg.do_parlai_rescoring
        self.n_rescore = cfg.n_rescore
        self.augment_base_strategy_model_frac = cfg.augment_base_strategy_model_frac

    def get_plausible_order_limits(self, game: pydipcc.Game) -> List[int]:
        """Returns the max # plausible actions that should be sampled for each power
        in the state specified by `game`, based on the specified config and the number

        of units for that power.
        Returns:
            - A list of 7 ints corresponding to the max number of plausible actions that
            should be sampled for each power in POWERS.
        """
        limits = [self.n_plausible_orders] * len(POWERS)
        assert self.n_plausible_orders > 0
        if self.max_actions_units_ratio > 0:
            game_state = game.get_state()
            power_n_units = [num_orderable_units(game_state, p) for p in POWERS]
            limits = [
                max(min(limit, ceil(u * self.max_actions_units_ratio)), 1)
                for limit, u in zip(limits, power_n_units)
            ]
        return limits

    def log_orders(self, game: pydipcc.Game, policies: PowerPolicies, label: str = "") -> None:
        logging.info(f"Plausible orders {label} :")
        limit = self.get_plausible_order_limits(game)
        for power, orders_to_probs in policies.items():
            logging.info(
                f"    {power} ( found {len(orders_to_probs)} / {limit[POWERS.index(power)]} )"
            )
            logging.info("        prob,order")
            for orders, probs in orders_to_probs.items():
                logging.info(f"        {probs:10.5f}  {orders}")

    def sample_orders(
        self,
        game: pydipcc.Game,
        *,
        agent_power: Optional[Power],
        speaking_power: Optional[Power] = None,
        player_rating: Optional[PlayerRating] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        force_base_strategy_model_has_press: bool = False,
        timings: Optional[TimingCtx] = None,
    ) -> PowerPolicies:
        """
        Sample a set of orders for each power. Return the distribution over these orders (policies).

        force_base_strategy_model_has_press: if set, base_strategy_model will always be queried with
            has_press=True. If not set, then it depends...

        Returns: A dictionary of Action -> Prob(Action) for each power.
        """
        if timings is None:
            timings = TimingCtx()
        logging.info("Starting sample_orders...")
        base_strategy_model_policy = None

        dialogue_phase = game.get_metadata("last_dialogue_phase")
        cur_phase = game.current_short_phase

        if dialogue_phase != game.current_short_phase and dialogue_phase.endswith("M"):
            logging.info(
                f"HACK: falling back to base_strategy_model for extended rollout order sampling of {cur_phase} for {dialogue_phase} dialogue."
            )
            timings.start("sample_base_strategy_model")
            ret = self._sample_orders_base_strategy_model(
                game,
                has_press=force_base_strategy_model_has_press,
                agent_power=agent_power,
                player_rating=player_rating,
            )
        ####################################################################################
        elif self.do_parlai_rescoring:
            assert self.base_strategy_model and self.parlai_model
            assert speaking_power is not None

            logging.info("Sampling base_strategy_model orders...")
            timings.start("sample_base_strategy_model")
            base_strategy_model_policy = self._sample_orders_base_strategy_model(
                game, has_press=True, agent_power=agent_power
            )
            if self.n_rescore > 0:
                base_strategy_model_policy = cutoff_policy(
                    base_strategy_model_policy, [self.n_rescore] * len(POWERS)
                )

            if self.parlai_req_size > 0:
                timings.start("sample_parlai")
                logging.info("Sampling parlai orders...")
                parlai_policy = self._sample_orders_parlai(game, speaking_power)
                # filter out parlai orders that are not invalid under base_strategy_model
                parlai_policy = filter_out_of_vocab_orders(parlai_policy)
            else:
                parlai_policy = {pwr: {} for pwr in POWERS}

            if extra_plausible_orders is None:
                extra_plausible_orders = {}
            combined_policy = {
                pwr: {
                    **base_strategy_model_policy[pwr],
                    **parlai_policy[pwr],
                    # don't worry about the 0s, the extra plausible orders will be rescored immediately below
                    **{a: 0.0 for a in extra_plausible_orders.get(pwr, [])},
                }
                for pwr in POWERS
            }
            logging.info(
                f"Combined: {[len(p) for p in base_strategy_model_policy.values()]} (base_strategy_model) + {[len(p) for p in parlai_policy.values()]} (parlai) --> {[len(p) for p in combined_policy.values()]} combined actions for rescoring..."
            )
            timings.start("rescore_parlai")
            ret = self.rescore_actions_parlai(game, speaking_power, combined_policy)
            logging.info("Done rescoring.")

        elif self.parlai_model:
            timings.start("sample_parlai")
            assert speaking_power is not None
            assert extra_plausible_orders is None
            ret = self._sample_orders_parlai(game, speaking_power)
        elif self.base_strategy_model:
            assert extra_plausible_orders is None
            timings.start("sample_base_strategy_model")
            ret = self._sample_orders_base_strategy_model(
                game,
                has_press=force_base_strategy_model_has_press,
                agent_power=agent_power,
                player_rating=player_rating,
            )
        else:
            raise RuntimeError()
        timings.stop()

        # take the limit most common orders per power
        limits = self.get_plausible_order_limits(game)
        ret_limit = cutoff_policy(ret, limits)

        # renormalize after cutting off
        renormalize_policy(ret_limit)

        if base_strategy_model_policy and self.augment_base_strategy_model_frac:
            for pwr, limit in zip(POWERS, limits):
                max_extra = round(limit * self.augment_base_strategy_model_frac)
                ret_limit[pwr] = {
                    **{
                        tuple(sorted(orders)): 0
                        for orders in list(base_strategy_model_policy[pwr])[:max_extra]
                    },
                    **ret_limit[pwr],
                }

        self.log_orders(game, ret_limit)
        return ret_limit

    def incremental_update_policy(
        self,
        game: pydipcc.Game,
        input_policy: PowerPolicies,
        speaking_power: Power,
        powers: List[Power],
        parlai_req_size: int,
    ) -> PowerPolicies:
        """
        Incrementally update policies
        """
        logging.info("Starting incremental_update_policy...")
        assert self.do_parlai_rescoring

        if parlai_req_size > 0:
            logging.info(f"Sampling {parlai_req_size} parlai orders...")
            parlai_policy = self._sample_orders_parlai(
                game, speaking_power, powers=powers, num_preds=parlai_req_size
            )
            # filter out parlai orders that are not invalid under base_strategy_model
            parlai_policy = filter_out_of_vocab_orders(parlai_policy)
            num_new_actions = sum(
                [
                    len([a for a in pi if a not in input_policy[pwr]])
                    for pwr, pi in parlai_policy.items()
                ]
            )
            logging.info(f"Found {num_new_actions} in incremental update.")
        else:
            parlai_policy = {pwr: {} for pwr in POWERS}

        combined_policy = {
            pwr: {**input_policy[pwr], **parlai_policy.get(pwr, {})} for pwr in powers
        }
        logging.info(
            f"Combined: {[len(input_policy[p]) for p in powers]} (input) + {[len(parlai_policy[p]) for p in powers]} (parlai) --> {[len(combined_policy[p]) for p in powers]} combined actions for rescoring..."
        )

        updated_policy_for_powers = self.rescore_actions_parlai(
            game, speaking_power, combined_policy
        )
        renormalize_policy(updated_policy_for_powers)

        updated_policy = {**input_policy, **updated_policy_for_powers}
        logging.info("Done rescoring.")

        # # take the limit most common orders per power
        # limits = self.get_plausible_order_limits(game)
        # ret_limit = cutoff_policy(updated_policy, limits)

        self.log_orders(game, updated_policy)
        return updated_policy

    def _sample_orders_base_strategy_model(
        self,
        game,
        *,
        has_press: bool,
        agent_power: Optional[Power],
        player_rating=None,
        temperature=1.0,
        top_p=1.0,
    ) -> PowerPolicies:
        n_samples = self.req_size
        batch_size = self.batch_size
        assert n_samples % batch_size == 0, f"{n_samples}, {batch_size}"

        counters = {p: Counter() for p in POWERS}

        game_rating_dict = None if player_rating is None else {game.game_id: player_rating}

        orders_to_logprobs = {}
        for _ in range(n_samples // batch_size):
            # Use batch_repeat_interleave so that the model behaves as if we'd duplicated
            # the input batch_size many times - taking that many policy samples.
            assert self.base_strategy_model is not None
            batch_orders, batch_order_logprobs = self.base_strategy_model.forward_policy(
                [game],
                has_press=has_press,
                agent_power=agent_power,
                temperature=temperature,
                game_rating_dict=game_rating_dict,
                top_p=top_p,
                batch_repeat_interleave=batch_size,
            )
            batch_orders = list(zip(*batch_orders))  # power -> list[orders]
            batch_order_logprobs = batch_order_logprobs.t()  # [7 x B]
            for p, power in enumerate(POWERS):
                counters[power].update(batch_orders[p])

            # slow and steady
            for power_orders, power_scores in zip(batch_orders, batch_order_logprobs):
                for order, score in zip(power_orders, power_scores):
                    if order not in orders_to_logprobs:
                        orders_to_logprobs[order] = score
                    else:
                        assert (
                            abs(orders_to_logprobs[order] - score)
                            < 0.2  # very loose tolerance, for fp16
                        ), f"{order} : {orders_to_logprobs[order]} != {score}"

        def sort_key(order_count_pair):
            order, _ = order_count_pair
            return (-int(are_supports_coordinated(order)), -orders_to_logprobs[order])

        most_common = {
            power: sorted(counter.most_common(), key=sort_key)
            for power, counter in counters.items()
        }

        logging.info(
            "get_plausible_orders(n={}, t={}) found {} unique sets, n_0={}".format(
                n_samples,
                temperature,
                list(map(len, counters.values())),
                [safe_idx(most_common[p], 0, default=(None, None))[1] for p in POWERS],
            )
        )

        orders_to_probs = {}
        for pwr, orders_and_counts in most_common.items():
            logprobs = torch.tensor(
                [orders_to_logprobs[orders] for orders, _ in orders_and_counts]
            )
            # Make sure that return a dict of action -> float rather than a dict of action -> tensor
            # Singleton float tensors have less precision than floats and the lower precision can
            # cause problems with np.choice or other things that require probabilities summing to 1.
            probs = logprobs.softmax(dim=0).cpu().numpy()
            orders_to_probs[pwr] = {
                orders: prob for (orders, _), prob in zip(orders_and_counts, probs)
            }

        return orders_to_probs

    def _sample_orders_parlai(
        self,
        game: pydipcc.Game,
        speaking_power: Power,
        num_preds: Optional[int] = None,
        powers: List[Power] = POWERS,
    ) -> PowerPolicies:
        assert speaking_power is not None
        assert self.parlai_model is not None

        if num_preds is None:
            num_preds = self.parlai_req_size

        power_orders: PowerPolicies = {}
        if isinstance(self.parlai_model, ParlAIAllOrderIndependentWrapper) or isinstance(
            self.parlai_model, ParlAIAllOrderIndependentRolloutWrapper
        ):
            #  Case 2: ParlAI models that return orders for a single specified power
            #  We will sample orders for each power independently.
            logging.info(f"Sampling orders from {type(self.parlai_model)}")
            power_orders = {}
            game_state = game.get_state()
            power2context = {}
            for power in powers:
                if num_orderable_units(game_state, power) == 0:
                    power_orders[power] = {tuple(): 1.0}
                    continue

                assert self.parlai_model_executor is not None
                pairs = self.parlai_model_executor.compute(
                    "produce_many_order_for_target_power",
                    game,
                    view_of_power=speaking_power,
                    target_power=power,
                    num_preds=num_preds,
                    batch_size=self.parlai_batch_size,
                )
                power2context[power] = pairs

            for power, pairs_future in power2context.items():
                pairs = pairs_future.result()
                logging.info(f"Finished sampling {len(pairs)} plausible orders for {power}.")
                possible_orders = [
                    order
                    for loc, orders in game.get_all_possible_orders().items()
                    for order in orders
                    if loc in game.get_orderable_locations().get(power, [])
                ]
                # filter orders
                good_pairs = []
                num_builds = game_state["builds"][power]["count"]
                for orders, score in pairs:
                    _, bad_orders = filter_orders(orders, possible_orders)
                    valid_build_destroy = is_valid_build_or_destroy(orders, num_builds)
                    no_duplicates = len(set(orders)) == len(orders)
                    if not any(bad_orders) and valid_build_destroy and no_duplicates:
                        good_pairs.append((orders, score))

                if good_pairs:
                    orders_list, order_scores = zip(*good_pairs)
                    order_probs = torch.tensor(order_scores).softmax(dim=0)
                else:
                    orders_list, order_probs = [], []
                power_orders[power] = {o: p for o, p in zip(orders_list, order_probs)}
        else:
            raise RuntimeError(f"{self.parlai_model} not supported.")

        if self.exclude_n_holds > 0:
            for _, actions in power_orders.items():
                for action in list(actions):
                    is_all_holds = all(order.endswith("H") for order in action)
                    if len(action) >= self.exclude_n_holds and is_all_holds:
                        del actions[action]
        return power_orders

    def rescore_actions_base_strategy_model(
        self,
        game: pydipcc.Game,
        *,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
        input_policy: PowerPolicies,
        model=None,
    ) -> PowerPolicies:

        power_action_dicts: List[Dict[Power, Action]] = [
            {pwr: action} for pwr, policy in input_policy.items() for action in policy.keys()
        ]
        if model is None:
            assert self.base_strategy_model is not None
            model = self.base_strategy_model.model

        logprobs = compute_action_logprobs_from_state(
            base_strategy_model=model,
            game=game,
            power_action_dicts=power_action_dicts,
            has_press=has_press,
            agent_power=agent_power,
            game_rating_dict=game_rating_dict,
            batch_size=self.batch_size,
        )
        rescored_policy: PowerPolicies = {power: {} for power in input_policy}
        for power_action_dict, logprob in zip(power_action_dicts, logprobs):
            for power, action in power_action_dict.items():
                rescored_policy[power][action] = math.exp(logprob)

        renormalize_policy(rescored_policy)

        return rescored_policy

    def rescore_actions_parlai(
        self,
        game: pydipcc.Game,
        speaking_power: Power,
        input_policy: PowerPolicies,
        include_powers: Optional[List[Power]] = None,
    ) -> PowerPolicies:
        return self.rescore_actions_parlai_multi_games(
            [game],
            [speaking_power],
            [input_policy],
            None if include_powers is None else [include_powers],
        )[0]

    def rescore_actions_parlai_multi_games(
        self,
        games: List[pydipcc.Game],
        speaking_powers: List[Power],
        input_policies: List[PowerPolicies],
        list_include_powers: Optional[List[List[Power]]] = None,
    ) -> List[PowerPolicies]:
        assert self.parlai_model is not None
        assert len(games) == len(speaking_powers), f"{len(games)}, {len(speaking_powers)}"
        assert len(games) == len(input_policies), f"{len(games)}, {len(input_policies)}"
        if list_include_powers is not None:
            assert len(games) == len(
                list_include_powers
            ), f"{len(games)}, {len(list_include_powers)}"

        if isinstance(self.parlai_model, ParlAIAllOrderIndependentWrapper) or isinstance(
            self.parlai_model, ParlAIAllOrderIndependentRolloutWrapper
        ):
            assert self.parlai_model_executor is not None
            rescored_policies = []
            list_futures = []
            for game_idx, (game, speaking_power, input_policy) in enumerate(
                zip(games, speaking_powers, input_policies)
            ):
                rescored_policies.append({})
                list_futures.append({})
                include_powers = (
                    list_include_powers[game_idx] if list_include_powers is not None else None
                )
                for pwr, policy in input_policy.items():
                    if include_powers is not None and pwr not in include_powers:
                        rescored_policies[-1][pwr] = input_policy[pwr]
                        continue

                    candidates = list(policy)
                    pwr_logprobs = {}
                    # batch to avoid GPU OOM from evaluating too many candidates
                    for i in range(0, len(candidates), self.parlai_batch_size):
                        list_futures[-1][pwr, i] = self.parlai_model_executor.compute(
                            "score_candidate_actions",
                            game,
                            candidates[i : i + self.parlai_batch_size],
                            view_of_power=speaking_power,
                            target_power=pwr,
                        )

            for game_idx, (game, speaking_power, input_policy) in enumerate(
                zip(games, speaking_powers, input_policies)
            ):
                include_powers = (
                    list_include_powers[game_idx] if list_include_powers is not None else None
                )

                for pwr, policy in input_policy.items():
                    if include_powers is not None and pwr not in include_powers:
                        continue

                    candidates = list(policy)
                    pwr_logprobs = {}

                    # batch to avoid GPU OOM from evaluating too many candidates
                    for i in range(0, len(candidates), self.parlai_batch_size):
                        pwr_logprobs.update(list_futures[game_idx][pwr, i].result())

                    # lets add a sanity check that the proposed actions don't all
                    # have extremely low probability. We might need to remove this
                    # for some use cases (or make configurable)
                    #
                    # This assert sometimes (rarely) hits when current_short_phase != dialogue_phase
                    # because the rollout model can't really reason about future phases very well.
                    # Since this is just used for rollout pseudo-orders, it's not really worth debugging
                    # forever; to fix we should use prefix dialogue models rather than rollout models.
                    with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
                        assert (
                            max(pwr_logprobs.values()) >= -20
                            or self.parlai_model.opt["is_debug"]
                            or game.current_short_phase != game.get_metadata("last_dialogue_phase")
                        ), f"WTF?\n\nspeaking_power: {speaking_power}\npwr: {pwr}\nphase: {game.current_short_phase}\ndialogue_phase: {game.get_metadata('last_dialogue_phase')}\npolicy:{policy}\n\ngame:\n{game.to_json()}\n\npwr_logprobs:\n{pwr_logprobs}\n\nsamples:\n{self.parlai_model.produce_many_order_for_target_power(game, view_of_power=speaking_power, target_power=pwr, num_preds=4, batch_size=4)}"

                    total_prob = sum(math.exp(logprob) for _, logprob in pwr_logprobs.items())
                    logging.info(
                        f"Plausible orders for {pwr} captured {total_prob:.3f} of probability mass (at temperature 1)"
                    )

                    temperature = self.parlai_model.opt.get("temperature", 1.0)
                    max_logprob = max(pwr_logprobs.values())
                    rescored_policies[game_idx][pwr] = {
                        a: math.exp((logprob - max_logprob) / temperature)
                        for a, logprob in pwr_logprobs.items()
                    }
        else:
            raise RuntimeError(f"Unexpected model type: {type(self.parlai_model)}")

        for rescored_policy in rescored_policies:
            renormalize_policy(rescored_policy)

        return rescored_policies

    def rescore_actions(
        self,
        game: pydipcc.Game,
        *,
        has_press: bool,
        agent_power: Optional[Power],
        game_rating_dict: Optional[Dict[str, PlayerRating]] = None,
        input_policy: PowerPolicies,
    ) -> PowerPolicies:
        if self.parlai_model is not None:
            assert agent_power is not None
            return self.rescore_actions_parlai(
                game, speaking_power=agent_power, input_policy=input_policy
            )
        else:
            return self.rescore_actions_base_strategy_model(
                game,
                has_press=has_press,
                agent_power=agent_power,
                input_policy=input_policy,
                game_rating_dict=game_rating_dict,
            )


def is_n_holds(orders: Action, max_holds) -> bool:
    return len(orders) >= max_holds and all([o.endswith(" H") for o in orders])


def filter_keys(d, fn, log_warn=False):
    """Return a copy of a dict-like input containing the subset of keys where fn(k) is truthy"""
    r = type(d)()
    for k, v in d.items():
        if fn(k):
            r[k] = v
        elif log_warn:
            logging.warning(f"filtered bad key: {k}")
    return r


def are_supports_coordinated(orders: Action) -> bool:
    """Return False if any supports or convoys are not properly coordinated

    e.g. if "F BLA S A SEV - RUM", return False if "A SEV" is not ordered "A SEV - RUM"
             0  1  2 3  4  5  6
    """
    required = {}
    ordered = {}

    for order in orders:
        split = order.split()
        ordered[split[1]] = split  # save by location
        if split[2] in ("S", "C"):
            if split[4] in required and required[split[4]] != split[3:]:
                # an order is already required of this unit, but it contradicts this one
                return False
            else:
                required[split[4]] = split[3:]

    for req_loc, req_order in required.items():
        if req_loc not in ordered:
            # supporting a foreign unit is always allowed, since we can't
            # control the coordination
            continue

        actual_order = ordered[req_loc]

        if len(req_order) == 2 and actual_order[2] == "-":
            # we supported a hold, but it tried to move
            return False
        elif (
            len(req_order) > 2
            and req_order[2] == "-"
            and (actual_order[2] != "-" or actual_order[3][:3] != req_order[3][:3])
        ):
            # we supported a move, but the order given was (1) not a move, or
            # (2) a move to the wrong destination
            return False

    # checks passed, return True
    return True


def safe_idx(seq, idx, default=None):
    try:
        return seq[idx]
    except IndexError:
        return default
