#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict

import math
from typing import Callable, DefaultDict, Dict, List, Set, Tuple, Optional, Any
import collections
import copy
import functools
import itertools
import json
import logging
import random
import tabulate
import time
from termcolor import colored

import numpy as np
from fairdiplomacy.utils.agent_interruption import raise_if_should_stop
from fairdiplomacy.utils.typedefs import get_last_message
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME
import torch

from conf import agents_cfgs
from fairdiplomacy.pydipcc import Game, CFRStats
from fairdiplomacy.agents.base_agent import AgentState
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    SearchResult,
    make_set_orders_dicts,
    sample_orders_from_policy,
)
from fairdiplomacy.agents.bilateral_stats import BilateralStats
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.utils.temp_redefine import temp_redefine
from parlai_diplomacy.utils.misc import last_dict_key
from parlai.utils.logging import _is_interactive

# fairdiplomacy.action_generation and fairdiplomacy.action_exploration
# both circularly refer fairdiplomacy.agents, so we import those modules whole
# instead of from "fairdiplomacy.action_generation import blah".
# That way, we break the circular initialization issue by not requiring the symbols
# *within* those modules to exist at import time, since they very well might not exist
# if we were only halfway through importing those when we began importing this file.
import fairdiplomacy.action_generation
import fairdiplomacy.action_exploration
from fairdiplomacy.agents.base_strategy_model_rollouts import (
    BaseStrategyModelRollouts,
    RolloutResultsCache,
)
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import (
    PlausibleOrderSampler,
    cutoff_policy,
    renormalize_policy,
)
from fairdiplomacy.agents.br_corr_bilateral_search import (
    BRCorrBilateralSearchResult,
    compute_payoff_matrix_for_all_opponents,
    extract_bp_policy_for_powers,
)
from fairdiplomacy.models.consts import POWER2IDX, POWERS
from fairdiplomacy.utils.game import game_from_two_party_view, get_last_message_from, next_M_phase
from fairdiplomacy.utils.parse_device import device_id_to_str
from fairdiplomacy.utils.sampling import sample_p_dict, argmax_p_dict
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.order_idxs import is_action_valid
from fairdiplomacy.utils.base_strategy_model_multi_gpu_wrappers import (
    MultiProcessBaseStrategyModelExecutor,
)
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.typedefs import (
    Action,
    BilateralConditionalValueTable,
    JointAction,
    MessageDict,
    MessageHeuristicResult,
    ConditionalValueTable,
    Phase,
    PlausibleOrders,
    PlayerRating,
    Policy,
    Power,
    PowerPolicies,
    Timestamp,
)
from parlai_diplomacy.wrappers.dialogue import TOKEN_DETAILS_TAG
from parlai_diplomacy.wrappers.factory import load_order_wrapper
from parlai_diplomacy.wrappers.orders import ParlAIPlausiblePseudoOrdersWrapper
from parlai_diplomacy.wrappers.base_wrapper import RolloutType

from fairdiplomacy.agents.parlai_message_handler import (
    ParlaiMessageHandler,
    ParlaiMessagePseudoOrdersCache,
    SleepSixTimesCache,
    pseudoorders_initiate_sleep_heuristics_should_trigger,
    joint_action_contains_xpower_support_or_convoy,
)


ActionDict = Dict[Tuple[Power, Action], float]


def color_order_logprobs(a: Action, lp: Optional[Dict[Action, List[float]]]) -> str:
    if not _is_interactive() or lp is None:
        return ", ".join(a)

    alp = lp[a]
    assert len(a) == len(alp)
    ret = []
    for o, olp in zip(a, alp):
        op = math.exp(olp)
        if op < 1e-2:
            o = colored(o, "red")
        elif op < 1e-1:
            o = colored(o, "yellow")
        elif op < 0.5:
            o = colored(o, "green")
        ret.append(o)

    return ", ".join(ret)


def filter_logprob_ratio(action_logprobs: Policy, min_ratio: float) -> Set[Action]:
    """Returns the set of actions in the logprobs whose probability is at least
    min_ratio times the probability of the most likely action"""

    assert all(
        p <= 1e-8 for p in action_logprobs.values()
    ), f"action_logprobs should be negative; got {action_logprobs}"
    assert 0 < min_ratio < 1, min_ratio
    max_logprob = max(action_logprobs.values())
    return {a for a, p in action_logprobs.items() if p - max_logprob >= math.log(min_ratio)}


class CFRResult(SearchResult):
    def __init__(
        self,
        bp_policies: PowerPolicies,
        avg_policies: PowerPolicies,
        final_policies: PowerPolicies,
        cfr_data: Optional["CFRData"],
        use_final_iter: bool,
        bilateral_stats: Optional[BilateralStats] = None,
    ):
        self.bp_policies = bp_policies
        self.avg_policies = avg_policies
        self.final_policies = final_policies
        self.cfr_data = cfr_data  # type: ignore
        self.use_final_iter = use_final_iter
        self.bilateral_stats = bilateral_stats

    def get_agent_policy(self) -> PowerPolicies:
        return self.avg_policies

    def get_population_policy(self) -> PowerPolicies:
        return self.avg_policies

    def get_bp_policy(self) -> PowerPolicies:
        return self.bp_policies

    def sample_action(self, power) -> Action:
        policies = self.final_policies if self.use_final_iter else self.avg_policies
        return sample_p_dict(policies[power])

    def avg_utility(self, pwr: Power) -> float:
        return self.cfr_data.avg_utility(pwr) if self.cfr_data is not None else 0

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        return self.cfr_data.avg_action_utility(pwr, a) if self.cfr_data is not None else 0

    def get_bilateral_stats(self) -> BilateralStats:
        assert self.bilateral_stats is not None
        return self.bilateral_stats

    def is_early_exit(self) -> bool:
        return self.cfr_data is None


def _early_quit_cfr_result(power: Power, *, action: Action = tuple()) -> CFRResult:
    policies = {power: {action: 1.0}}
    return CFRResult(
        bp_policies=policies,
        avg_policies=policies,
        final_policies=policies,
        cfr_data=None,
        use_final_iter=False,
    )


def sorted_policy(plausible_orders: List[Action], probs: List[float]) -> Policy:
    return dict(sorted(zip(plausible_orders, probs), key=lambda ac_p: -ac_p[1]))


class CFRData:
    def __init__(
        self,
        bp_policy: PowerPolicies,
        use_optimistic_cfr: bool,
        qre: Optional[agents_cfgs.SearchBotAgent.QRE] = None,
        agent_power=None,
        scale_lambdas_by_power: Optional[Dict[Power, float]] = None,
    ):
        # Make sure that all powers have some actions. This guarantees that
        # we run utility computation for every power and so state values will
        # be computed correctly. In theory, only alive powers should have
        # non-zero utility, ano so we only need to augment alive powers without
        # orders.  But in practice as the state values are computed by a value
        # network some dead power may have non-zero utilities.
        # Note, that the plausible order sampler by default adds empty policies
        # for all powers.
        for power in bp_policy:
            assert bp_policy[
                power
            ], f"Power {power} doesn't have policy. Add an empty action policy."

        self.use_optimistic_cfr = use_optimistic_cfr
        use_linear_weighting = True
        if qre:
            use_qre = True
            qre_target_blueprint = qre.target_pi == "BLUEPRINT"
            qre_eta = qre.eta
            assert qre.qre_lambda is not None, "qre_lambda must be set."
            qre_lambda = qre.qre_lambda
            power_qre_lambda = {p: qre_lambda for p in POWERS}
            if qre.agent_qre_lambda is not None:
                assert (
                    qre.agent_qre_lambda != -1
                ), "Use None for specifying unset agent_qre_lambda, rather than -1"
                assert agent_power is not None, "agent_power must be set"
                power_qre_lambda[agent_power] = qre.agent_qre_lambda
            qre_entropy_factor = qre.qre_entropy_factor
            power_qre_entropy_factor = {p: qre_entropy_factor for p in POWERS}
            if qre.agent_qre_entropy_factor is not None:
                assert (
                    qre.agent_qre_entropy_factor != -1
                ), "Use None for specifying unset agent_qre_entropy_factor, rather than -1"
                assert agent_power is not None, "agent_power must be set"
                power_qre_entropy_factor[agent_power] = qre.agent_qre_entropy_factor
        else:
            use_qre = False
            qre_target_blueprint = False
            qre_eta = 0.0
            qre_lambda = 0.0
            qre_entropy_factor = 1.0
            power_qre_lambda = {p: qre_lambda for p in POWERS}
            power_qre_entropy_factor = {p: qre_entropy_factor for p in POWERS}

        if scale_lambdas_by_power is not None:
            for power in scale_lambdas_by_power:
                power_qre_lambda[power] *= scale_lambdas_by_power[power]

        self.power_qre_lambda = power_qre_lambda
        self.power_plausible_orders: PlausibleOrders = {p: sorted(v) for p, v in bp_policy.items()}
        power_plausible_action_probs = {
            p: [bp_policy[p][a] for a in self.power_plausible_orders[p]] for p in POWERS
        }

        self.stats = CFRStats(
            use_linear_weighting,
            use_optimistic_cfr,
            use_qre,
            qre_target_blueprint,
            qre_eta,
            power_qre_lambda,
            power_qre_entropy_factor,
            power_plausible_action_probs,
        )

    def cur_iter_strategy(self, pwr: Power) -> List[float]:
        return self.stats.cur_iter_strategy(pwr)

    def cur_iter_policy(self, pwr: Power) -> Policy:
        return sorted_policy(self.power_plausible_orders[pwr], self.cur_iter_strategy(pwr))

    def avg_strategy(self, pwr: Power) -> List[float]:
        return self.stats.avg_strategy(pwr)

    def avg_policy(self, pwr: Power) -> Policy:
        return sorted_policy(self.power_plausible_orders[pwr], self.avg_strategy(pwr))

    def avg_utility(self, pwr: Power) -> float:
        return self.stats.avg_utility(pwr)

    def avg_action_utilities(self, pwr: Power) -> List[float]:
        return self.stats.avg_action_utilities(pwr)

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        return self.stats.avg_action_utility(pwr, self.power_plausible_orders[pwr].index(a))

    def avg_action_regret(self, pwr: Power, a: Action) -> float:
        return self.stats.avg_action_regret(pwr, self.power_plausible_orders[pwr].index(a))

    def avg_action_prob(self, pwr: Power, a: Action) -> float:
        return self.stats.avg_action_prob(pwr, self.power_plausible_orders[pwr].index(a))

    def cur_iter_action_prob(self, pwr: Power, a: Action) -> float:
        return self.stats.cur_iter_action_prob(pwr, self.power_plausible_orders[pwr].index(a))

    def bp_strategy(self, pwr: Power, temperature=1.0) -> List[float]:
        return self.stats.bp_strategy(pwr, temperature)

    def bp_policy(self, pwr: Power, temperature=1.0) -> Policy:
        return sorted_policy(self.power_plausible_orders[pwr], self.bp_strategy(pwr, temperature))

    def update(
        self,
        pwr: Power,
        actions: List[Action],
        state_utility: float,
        action_utilities: List[float],
        which_strategy_to_accumulate: int,
        cfr_iter: int,
    ) -> None:
        self.stats.update(
            pwr, state_utility, action_utilities, which_strategy_to_accumulate, cfr_iter
        )


PhaseKey = Tuple[Phase, Phase]  # (dialogue_phase, rollout_phase)


class SearchBotAgentState(AgentState):
    """Cached state for a particular agent power."""

    def __init__(self, agent_power):
        self.last_ts: Timestamp = Timestamp.from_seconds(-1)
        self.agent_power = agent_power
        # hide these because we have to check that we're on the same phase!
        self._last_search_result: Dict[PhaseKey, Optional[SearchResult]] = {}
        self._last_pseudo_orders: Dict[PhaseKey, Optional[JointAction]] = {}
        self._value_table_cache: DefaultDict[
            PhaseKey, DefaultDict[Power, BilateralConditionalValueTable]
        ] = defaultdict(functools.partial(defaultdict, dict))

        self.pseudo_orders_cache = ParlaiMessagePseudoOrdersCache()

    def _get_phase_key(self, game: Game) -> PhaseKey:
        return (
            game.get_metadata("last_dialogue_phase") or game.current_short_phase,
            game.current_short_phase,
        )

    def update(
        self,
        game: Game,
        agent_power: Power,
        search_result: Optional[SearchResult],
        pseudo_orders: Optional[JointAction],
    ) -> None:
        assert self.agent_power == agent_power
        self.last_ts = (
            last_dict_key(game.messages) if game.messages else Timestamp.from_seconds(-1)
        )
        self.agent_power = agent_power
        if search_result and search_result.is_early_exit():
            # don't include early-exit search results, because they
            # don't have the actual search policies for anyone
            search_result = None

        phase_key = self._get_phase_key(game)
        self._last_search_result[phase_key] = search_result
        self._last_pseudo_orders[phase_key] = pseudo_orders

    def get_last_search_result(self, game: Game) -> Optional[SearchResult]:
        phase_key = self._get_phase_key(game)
        return self._last_search_result.get(phase_key, None)

    def get_last_pseudo_orders(self, game: Game) -> Optional[JointAction]:
        phase_key = self._get_phase_key(game)
        return self._last_pseudo_orders.get(phase_key, None)

    def get_new_messages(self, game: Game):
        """Return all new messages in the game since this state"""
        return [m for m in game.messages.values() if m["time_sent"] > self.last_ts]

    def get_sleepsix_cache(self) -> SleepSixTimesCache:
        # backwards compatibility
        if not hasattr(self, "sleepsix_cache"):
            self.sleepsix_cache = SleepSixTimesCache()
        return self.sleepsix_cache

    def get_cached_value_tables(
        self, game: Game
    ) -> DefaultDict[Power, BilateralConditionalValueTable]:
        phase_key = self._get_phase_key(game)
        return self._value_table_cache[phase_key]


class SearchBotAgent(BaseSearchAgent):
    """One-ply cfr with base_strategy_model-policy rollouts"""

    def __init__(self, cfg: agents_cfgs.SearchBotAgent, *, skip_base_strategy_model_cache=False):
        super().__init__(cfg)
        base_strategy_model_wrapper_kwargs = dict(
            device=device_id_to_str(cfg.device),
            max_batch_size=cfg.max_batch_size,
            half_precision=cfg.half_precision,
            skip_base_strategy_model_cache=skip_base_strategy_model_cache,
        )
        self.base_strategy_model = BaseStrategyModelWrapper(
            cfg.rollout_model_path or cfg.model_path,
            value_model_path=cfg.value_model_path,
            force_disable_all_power=True,
            **base_strategy_model_wrapper_kwargs,
        )
        self.cfg = cfg
        self.has_press = cfg.dialogue is not None
        self.set_player_rating = cfg.set_player_rating
        self.player_rating = cfg.player_rating
        if self.set_player_rating:
            if cfg.cache_rollout_results:
                logging.warning("Undefined behaviour if searchbot.cache_rollout_results is set")
            assert (
                self.player_rating is not None and 0 <= self.player_rating <= 1.0
            ), "Player rating needs to be a float between 0 and 1.0"
            logging.info(f"Setting player rating to {self.player_rating}")
        else:
            if self.player_rating is not None:
                logging.warning(
                    "searchbot.player_rating is set but searchbot.set_player_rating is not set"
                )
            self.player_rating = None

        self.base_strategy_model_rollouts = BaseStrategyModelRollouts(
            self.base_strategy_model,
            cfg=cfg.rollouts_cfg,
            has_press=self.has_press,
            set_player_ratings=self.set_player_rating,
        )
        self.bilateral_cfg = cfg.bilateral_dialogue
        assert cfg.n_rollouts >= 0, "Set searchbot.n_rollouts"

        self.qre = cfg.qre
        if self.qre is not None:
            logging.info(
                f"Performing qre regret minimization with eta={self.qre.eta} "
                f"and lambda={self.qre.qre_lambda} with target pi={self.qre.target_pi}"
            )
            if self.qre.qre_lambda == 0.0:
                logging.info("Using lambda 0.0 simplifies qre to regular hedge")

        self.n_rollouts = cfg.n_rollouts
        self.cache_rollout_results = cfg.cache_rollout_results
        self.precompute_cache = cfg.precompute_cache
        self.enable_compute_nash_conv = cfg.enable_compute_nash_conv
        self.n_plausible_orders = cfg.plausible_orders_cfg.n_plausible_orders
        self.use_optimistic_cfr = cfg.use_optimistic_cfr
        self.use_final_iter = cfg.use_final_iter
        self.bp_iters = cfg.bp_iters
        self.bp_prob = cfg.bp_prob
        self.loser_bp_iter = cfg.loser_bp_iter
        self.loser_bp_value = cfg.loser_bp_value
        self.reset_seed_on_rollout = cfg.reset_seed_on_rollout
        self.max_seconds = cfg.max_seconds
        self.br_corr_bilateral_search_cfg = cfg.br_corr_bilateral_search
        self.message_search_cfg = cfg.message_search

        self.all_power_base_strategy_model_executor = None
        if self.br_corr_bilateral_search_cfg is not None:
            assert (
                self.base_strategy_model.model.is_all_powers()
            ), "br_corr_bilateral_search requires an all-powers base_strategy_model model."
            allpower_wrapper_kwargs = {
                **base_strategy_model_wrapper_kwargs,
                "model_path": self.base_strategy_model.model_path,
                "value_model_path": cfg.value_model_path,
            }
            allpower_rollouts_kwargs = {
                "cfg": cfg.rollouts_cfg,
                "has_press": self.has_press,
                "set_player_ratings": self.set_player_rating,
            }
            # if we allow multi gpu for plausible orders,
            # then we also allow multi gpu for the base_strategy_model to avoid adding redundant flags
            self.all_power_base_strategy_model_executor = MultiProcessBaseStrategyModelExecutor(
                allow_multi_gpu=cfg.plausible_orders_cfg.allow_multi_gpu,
                base_strategy_model_wrapper_kwargs=allpower_wrapper_kwargs,
                base_strategy_model_rollouts_kwargs=allpower_rollouts_kwargs,
            )

        if cfg.parlai_model_orders.model_path:
            if cfg.rescoring_blueprint_model_path is not None:
                raise RuntimeError(
                    "You probably don't want to rescore parlai policy with a base_strategy_model BP"
                )

            logging.info("Setting up parlai orders model...")
            self.parlai_model_orders = load_order_wrapper(cfg.parlai_model_orders)
        else:
            self.parlai_model_orders = None

        self.cfr_messages = cfg.cfr_messages

        if cfg.dialogue is not None:
            self.message_handler = ParlaiMessageHandler(
                cfg.dialogue,
                model_orders=self.parlai_model_orders,
                base_strategy_model=self.base_strategy_model,
            )
        else:
            self.message_handler = None
            assert not self.cfr_messages

        if cfg.rollout_model_path and cfg.model_path != cfg.rollout_model_path:
            self.proposal_base_strategy_model = BaseStrategyModelWrapper(
                cfg.model_path, **base_strategy_model_wrapper_kwargs
            )
        else:
            self.proposal_base_strategy_model = self.base_strategy_model
        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg,
            base_strategy_model=self.proposal_base_strategy_model,
            parlai_model_cfg=cfg.parlai_model_orders,
        )
        self.order_aug_cfg = cfg.order_aug

        if cfg.rescoring_blueprint_model_path:
            assert self.parlai_model_orders is None
            assert (
                cfg.order_aug.do is None
            ), "Cannot use DO with rescoring_blueprint_model_path. Use multibp rescoring instead"
            self.rescoring_blueprint_model = BaseStrategyModelWrapper(
                cfg.rescoring_blueprint_model_path, **base_strategy_model_wrapper_kwargs
            )
        else:
            self.rescoring_blueprint_model = None

        self.exploited_agent = None
        self.exploited_agent_power = None
        self.exploited_agent_num_samples = 1
        if cfg.exploited_searchbot_cfg is not None and cfg.exploited_agent_power:
            # When exploiting an agent, we run a one-sided regret minimization with full knowledge of the fixed exploited
            # policy. So we need to make sure we have their actual policy.
            exploited_searchbot_cfg = cfg.exploited_searchbot_cfg.to_editable()
            logging.info(
                f"Replacing exploited agent device {exploited_searchbot_cfg.device} -> {cfg.device}"
            )
            exploited_searchbot_cfg.device = cfg.device
            exploited_searchbot_cfg = exploited_searchbot_cfg.to_frozen()
            assert not exploited_searchbot_cfg.use_final_iter
            assert cfg.exploited_agent_power in POWERS, cfg.exploited_agent_power
            self.exploited_agent = SearchBotAgent(exploited_searchbot_cfg)
            self.exploited_agent_power = cfg.exploited_agent_power
            self.exploited_agent_num_samples = cfg.exploited_agent_num_samples
            logging.info(
                f"Exploited agent: {self.exploited_agent_power} {exploited_searchbot_cfg}"
            )

        self.log_intermediate_iterations = cfg.log_intermediate_iterations
        self.log_bilateral_values = cfg.log_bilateral_values

        self.do_incremental_search = cfg.do_incremental_search
        logging.info(f"Initialized SearchBot Agent: {self.__dict__}")

    def initialize_state(self, power: Power) -> AgentState:
        return SearchBotAgentState(power)

    def get_exploited_agent_power(self) -> Optional[Power]:
        return self.exploited_agent_power

    def override_has_press(self, has_press: bool):
        self.has_press = has_press
        self.base_strategy_model_rollouts.override_has_press(has_press)

    # Overrides BaseAgent
    def can_share_strategy(self) -> bool:
        # It's only safe to share strategy if the strategy is not conditional on dialogue.
        # If qre uses different params per power, then its not symmetric or safe
        search_can_share_strategy = (self.qre is None) or (
            self.qre is not None
            and self.qre.agent_qre_lambda is None
            and self.qre.agent_qre_entropy_factor is None
        )
        return self.parlai_model_orders is None and search_can_share_strategy

    # Overrides BaseAgent
    def get_orders(self, game: Game, power: Power, state: AgentState) -> Action:
        assert isinstance(state, SearchBotAgentState)
        cfr_result = self.try_get_cached_search_result(game, state)

        if not cfr_result:
            bp_policy = self.maybe_get_incremental_bp(game, agent_power=power, agent_state=state)

            if self.use_br_correlated_search(game.phase, "final_order"):
                cfr_result = self.run_best_response_against_correlated_bilateral_search(
                    game,
                    bp_policy=bp_policy,
                    agent_power=power,
                    early_exit_for_power=power,
                    agent_state=state,
                )
            else:
                cfr_result = self.run_search(
                    game,
                    bp_policy=bp_policy,
                    agent_power=power,
                    early_exit_for_power=power,
                    agent_state=state,
                )
        state.update(game, power, cfr_result, None)
        return cfr_result.sample_action(power)

    # Overrides BaseAgent
    def get_orders_many_powers(self, game: Game, powers: List[Power],) -> JointAction:
        assert (
            self.message_handler is None
        ), "This searchbot agent appears to be a full-press agent. Do not use get_orders_many_powers in full-press since not all agents see the same info."
        assert not self.use_final_iter, "unsafe: use_final_iter"
        cfr_result = self.run_search(game, agent_state=None,)
        return {power: cfr_result.sample_action(power) for power in powers}

    # Overrides BaseSearchAgent
    def get_plausible_orders_policy(
        self,
        game: Game,
        *,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
        player_rating: Optional[PlayerRating] = None,
        allow_augment: bool = True,
    ) -> PowerPolicies:
        """Compute blueprint policy for all agents.

        If exploiting an agent, the blueprint will be the agent's computed average policy.
        If allow_augment is false, will not attempt to apply augmentations like double oracle.
        """

        # Determine the set of plausible actions to consider for each power
        policy = self.order_sampler.sample_orders(
            game, agent_power=agent_power, speaking_power=agent_power, player_rating=player_rating
        )

        if self.rescoring_blueprint_model is not None:
            policy = self.order_sampler.rescore_actions_base_strategy_model(
                game,
                has_press=self.has_press,
                agent_power=agent_power,
                input_policy=policy,
                model=self.rescoring_blueprint_model.model,
            )
            self.order_sampler.log_orders(game, policy, label="AFTER BP-rescoring")

        # If we are exploiting an agent, compute their policy and replace the blueprint
        # with their known average policy.
        if self.exploited_agent_power is not None:
            exploited_policy = collections.defaultdict(float)
            per_sample_weight = 1.0 / self.exploited_agent_num_samples
            for i in range(self.exploited_agent_num_samples):
                # ahh this is terrible!
                assert self.exploited_agent is not None
                assert not self.exploited_agent.use_final_iter
                sample_policy = self.exploited_agent.run_search(
                    game, agent_power=self.exploited_agent_power, agent_state=None
                ).avg_policies[self.exploited_agent_power]
                for action in sample_policy:
                    exploited_policy[action] += per_sample_weight * sample_policy[action]
            # Convert defaultdict -> ordinary dict
            exploited_policy = dict(exploited_policy)
            # Replace the blueprint for the power being exploited
            policy[self.exploited_agent_power] = exploited_policy

        # Inference time double oracle or other augmentation.
        if allow_augment:
            with temp_redefine(self.base_strategy_model_rollouts, "max_rollout_length", 0):
                with temp_redefine(self, "cache_rollout_results", True):
                    original_policy, policy = (
                        policy,
                        augment_plausible_orders(
                            game,
                            policy,
                            self,
                            self.order_aug_cfg,
                            agent_power=agent_power,
                            limits=self.order_sampler.get_plausible_order_limits(game),
                        ),
                    )

            for power in sorted(policy):
                new_actions = set(policy[power]).difference(original_policy[power])
                for i, action in enumerate(sorted(new_actions)):
                    logging.info(
                        "Order augmentation. New order for %s (%d/%d): %s",
                        power,
                        i + 1,
                        len(new_actions),
                        action,
                    )

        return policy

    def use_br_correlated_search(self, phase: str, mode: str):
        assert mode in ["final_order", "pseudo_order"], mode
        if self.br_corr_bilateral_search_cfg is None:
            return False
        if "MOVEMENT" not in phase:
            return False
        if mode == "final_order" and self.br_corr_bilateral_search_cfg.enable_for_final_order:
            return True
        if mode == "pseudo_order" and self.br_corr_bilateral_search_cfg.enable_for_pseudo_order:
            return True
        return False

    def run_search(
        self,
        game: Game,
        *,
        bp_policy: Optional[PowerPolicies] = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
    ) -> CFRResult:
        """Computes an equilibrium policy for all powers.

        Arguments:
            - game: Game object encoding current game state.
            - bp_policy: If set, overrides the plausible order set and blueprint policy for initialization.
                         Values should be probabilities, but can be set to -1 to simply specify plausible orders;
                         in that case, this function will raise an error if any feature uses the BP distribution (e.g. bp_iters > 0)
            - early_exit_for_power: If set, then if this power has <= 1 plausible order, will exit early without computing a full equilibrium.
            - timings: A TimingCtx object to measure timings
            - extra_plausible_orders: Extra plausible orders to add to the base_strategy_model-computed set.
            - agent_power: Optionally, specify which agent is computing the equilibrium.
                           Used by parlai plausible order generation, as well as advanced features like bilateral strategy.

        Returns:
            - CFRResult object:
                - avg_policies: {pwr: avg_policy} for each power
                - final_policies: {pwr: avg_policy} for each power
                - cfr_data: detailed internal information from the CFR procedure
        """
        if timings is None:
            timings = TimingCtx()
        timings.start("one-time")

        deadline: Optional[float] = (
            time.monotonic() + self.max_seconds if self.max_seconds > 0 else None
        )

        # If there are no locations to order, bail
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            if agent_power is not None:
                assert early_exit_for_power == agent_power
            return _early_quit_cfr_result(early_exit_for_power)

        logging.info(f"BEGINNING CFR run_search, agent_power={agent_power}")

        maybe_rollout_results_cache = (
            self.base_strategy_model_rollouts.build_cache() if self.cache_rollout_results else None
        )

        if bp_policy is None:
            bp_policy = self.get_plausible_orders_policy(
                game,
                agent_power=agent_power,
                agent_state=agent_state,
                player_rating=self.player_rating if self.set_player_rating else None,
            )

        if extra_plausible_orders:
            for p, orders in extra_plausible_orders.items():
                for order in orders:
                    bp_policy[p].setdefault(order, 0.0)
                logging.info(f"Adding extra plausible orders {p}: {orders}")

        cfr_data = CFRData(
            bp_policy,
            use_optimistic_cfr=self.use_optimistic_cfr,
            qre=self.qre,
            agent_power=agent_power,
        )

        # If there a single plausible action, no need to search.
        if (
            early_exit_for_power
            and len(cfr_data.power_plausible_orders[early_exit_for_power]) == 1
        ):

            [the_action] = cfr_data.power_plausible_orders[early_exit_for_power]
            return _early_quit_cfr_result(early_exit_for_power, action=the_action)

        # run rollouts or get from cache
        if self.cache_rollout_results and self.precompute_cache:
            num_active_powers = sum(
                len(actions) > 1 for actions in cfr_data.power_plausible_orders.values()
            )
            if num_active_powers > 2:
                logging.warning(
                    "Disabling precomputation of the CFR cache as have %d > 2 active powers",
                    num_active_powers,
                )
            else:
                joint_orders = sample_all_joint_orders(cfr_data.power_plausible_orders)
                self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                    game,
                    agent_power=agent_power,
                    set_orders_dicts=joint_orders,
                    cache=maybe_rollout_results_cache,
                    timings=timings,
                )

        if agent_power is not None:
            bilateral_stats = BilateralStats(game, agent_power, cfr_data.power_plausible_orders)
        else:
            bilateral_stats = None

        logging.info("Starting CFR iters...")
        last_search_iter = False
        for cfr_iter in range(self.n_rollouts):
            if last_search_iter:
                logging.info(f"Early exit from CFR after {cfr_iter} iterations by timeout")
                break
            elif deadline is not None and time.monotonic() >= deadline:
                last_search_iter = True
            timings.start("start")
            # do verbose logging on 2^x iters
            verbose_log_iter = self.is_verbose_log_iter(cfr_iter) or last_search_iter

            timings.start("query_policy")
            # get policy probs for all powers

            power_action_ps = self.get_cur_iter_strategies(cfr_data, cfr_iter)

            timings.start("apply_orders")
            # sample policy for all powers
            _, power_sampled_orders = sample_orders_from_policy(
                cfr_data.power_plausible_orders, power_action_ps
            )
            if bilateral_stats is not None:
                bilateral_stats.accum_bilateral_probs(power_sampled_orders, weight=cfr_iter)
            set_orders_dicts = make_set_orders_dicts(
                cfr_data.power_plausible_orders, power_sampled_orders
            )

            timings.stop()

            all_rollout_results = self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                game,
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                cache=maybe_rollout_results_cache,
                timings=timings,
                player_rating=self.player_rating,
            )
            timings.start("cfr")

            for pwr, actions in cfr_data.power_plausible_orders.items():
                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )
                if bilateral_stats is not None:
                    bilateral_stats.accum_bilateral_values(pwr, cfr_iter, results)
                # logging.info(f"Results {pwr} = {results}")
                # calculate regrets
                action_utilities: List[float] = [r[1][pwr] for r in results]
                state_utility: float = np.dot(power_action_ps[pwr], action_utilities)  # type: ignore

                # log some action values
                if verbose_log_iter:
                    self.log_cfr_iter_state(
                        game=game,
                        pwr=pwr,
                        actions=actions,
                        cfr_data=cfr_data,
                        cfr_iter=cfr_iter,
                        state_utility=state_utility,
                        action_utilities=action_utilities,
                        power_sampled_orders=power_sampled_orders,
                    )

                # update cfr data structures
                cfr_data.update(
                    pwr=pwr,
                    actions=actions,
                    state_utility=state_utility,
                    action_utilities=action_utilities,
                    which_strategy_to_accumulate=CFRStats.ACCUMULATE_PREV_ITER,
                    cfr_iter=cfr_iter,
                )

            if self.enable_compute_nash_conv and verbose_log_iter:
                logging.info(f"Computing nash conv for iter {cfr_iter}")
                self.compute_nash_conv(
                    cfr_data,
                    f"cfr iter {cfr_iter}",
                    game,
                    cfr_data.avg_strategy,
                    maybe_rollout_results_cache,
                    agent_power=agent_power,
                )

            if maybe_rollout_results_cache is not None and verbose_log_iter:
                logging.info(f"{maybe_rollout_results_cache}")

        timings.start("to_dict")

        # return prob. distributions for each power
        avg_ret, final_ret = {}, {}
        power_is_loser = self.get_power_loser_dict(cfr_data, self.n_rollouts)
        for p in POWERS:
            if power_is_loser[p] or p == self.exploited_agent_power:
                avg_ret[p] = final_ret[p] = cfr_data.bp_policy(p)
            else:
                avg_ret[p] = cfr_data.avg_policy(p)
                final_ret[p] = cfr_data.cur_iter_policy(p)

        if agent_power is not None:
            logging.info(f"Final avg strategy: {avg_ret[agent_power]}")

        logging.info(
            "Raw Values: %s",
            {
                p: f"{x:.3f}"
                for p, x in zip(
                    POWERS,
                    self.base_strategy_model.get_values(
                        game, has_press=self.has_press, agent_power=agent_power
                    ),
                )
            },
        )
        logging.info("CFR Values: %s", {p: f"{cfr_data.avg_utility(p):.3f}" for p in POWERS})

        timings.stop()

        if bilateral_stats is not None and self.log_bilateral_values:
            bilateral_stats.log(cfr_data, min_order_prob=self.bilateral_cfg.min_order_prob)

        timings.pprint(logging.getLogger("timings").info)

        return CFRResult(
            bp_policies=bp_policy,
            avg_policies=avg_ret,
            final_policies=final_ret,
            cfr_data=cfr_data,
            use_final_iter=self.use_final_iter,
            bilateral_stats=bilateral_stats,
        )

    def run_bilateral_search_with_conditional_evs(
        self, game: Game, *args, **kwargs,
    ):
        raise NotImplementedError

    def run_best_response_against_correlated_bilateral_search(
        self, game: Game, *args, **kwargs,
    ):
        raise NotImplementedError

    def _eval_action_under_bilateral_search(
        self,
        game: Game,
        agent_power: Power,
        agent_state: SearchBotAgentState,
        agent_action: Action,
        recipient_power: Power,
        bp_policy: PowerPolicies,
        pair_value_table: BilateralConditionalValueTable,
        timings: Optional[TimingCtx] = None,
    ) -> float:
        if timings is None:
            timings = TimingCtx()

        timings.start("run_search")
        eq_policy = self.run_bilateral_search_with_conditional_evs(
            game,
            bp_policy=bp_policy,
            agent_power=agent_power,
            other_power=recipient_power,
            agent_state=agent_state,
            conditional_evs=pair_value_table,
        ).get_population_policy()

        for policy_name, policy in [("bp_policy", bp_policy), ("eq_policy", eq_policy)]:
            policy_str = "\n".join(
                f"{power}\n{tabulate.tabulate(list(policy[power].items())[:6], tablefmt='plain')}"
                for power in (agent_power, recipient_power)
            )
            logging.info(f"\n{policy_name}:\n{policy_str}")

        payoffs = torch.zeros((len(POWERS), 1))  # type: ignore
        for recipient_action, prob in eq_policy[recipient_power].items():
            payoffs += prob * pair_value_table[(agent_action, recipient_action)]

        agent_payoff = float(payoffs[POWER2IDX[agent_power]])
        recipient_payoff = float(payoffs[POWER2IDX[recipient_power]])
        logging.info(
            f"_eval_action_under_bilateral_search payoffs:\n"
            f"{agent_power}: {float(agent_payoff)}\n"
            f"{recipient_power}: {float(recipient_payoff)}"
        )
        timings.stop()

        return agent_payoff

    def _generate_message_via_message_search(
        self,
        game: Game,
        agent_power: Power,
        agent_state: SearchBotAgentState,
        timestamp: Timestamp,
        recipient_power: Power,
        pseudo_orders: Optional[PseudoOrders],
        timings: Optional[TimingCtx] = None,
        additional_msg_dict: Optional[MessageDict] = None,
    ) -> Optional[MessageDict]:
        if timings is None:
            timings = TimingCtx()

        assert self.message_handler is not None

        with timings.create_subcontext("generate_candidate_messages") as subtimings:
            candidate_messages = self.message_handler.generate_multiple_messages_with_annotations(
                game,
                agent_power,
                timestamp,
                recipient_power,
                pseudo_orders,
                self.message_search_cfg.n_messages,
                timings=subtimings,
                skip_filtering=True,
            )

        if additional_msg_dict is not None:
            candidate_messages.append((additional_msg_dict, None))

        game_phase = game.current_short_phase
        value_table_cache = agent_state.get_cached_value_tables(game)

        game_2p = game_from_two_party_view(
            game, agent_power, recipient_power, add_message_to_all=False
        )

        timings.start("construct_base_policy")

        # Extract base policy from blueprint
        base_policy = self.maybe_get_incremental_bp(
            game_2p, agent_power=agent_power, agent_state=agent_state,
        )
        if base_policy is None:
            with timings.create_subcontext("bp") as subtimings:
                base_policy = self.order_sampler.sample_orders(
                    game_2p,
                    agent_power=agent_power,
                    speaking_power=agent_power,
                    player_rating=self.player_rating if self.set_player_rating else None,
                    timings=subtimings,
                )
        base_policy = extract_bp_policy_for_powers(base_policy, [agent_power, recipient_power])

        # Ensure pseudoorders represented in base_policy
        for power, action in pseudo_orders[game_phase].items():
            assert power in (agent_power, recipient_power)
            if action not in base_policy[power]:
                base_policy[power][action] = 0.0

        timings.start("compute_payoff_matrix")
        assert self.all_power_base_strategy_model_executor is not None
        pair_value_table = compute_payoff_matrix_for_all_opponents(
            game_2p,
            self.all_power_base_strategy_model_executor,
            base_policy,
            agent_power,
            self.br_corr_bilateral_search_cfg.bilateral_search_num_cond_sample,
            self.has_press,
            self.player_rating,
            value_table_cache=value_table_cache,
        )[recipient_power]

        timings.start("rescore_base_policy")
        logging.info("Running base policy rescorings for message search")
        cf_games = []
        for msg_dict, _ in candidate_messages:
            cf_game = Game(game_2p)
            cf_game.add_message(
                msg_dict["sender"],
                msg_dict["recipient"],
                msg_dict["message"],
                msg_dict["time_sent"],
                # increment_on_collision=True,  # Uncomment if we want to support 0 sleep time messages!
            )
            cf_games.append(cf_game)
        rescored_bp_policies = self.order_sampler.rescore_actions_parlai_multi_games(
            cf_games,
            [agent_power for _ in cf_games],
            [base_policy for _ in cf_games],
            [[agent_power, recipient_power] for _ in cf_games],
        )
        timings.stop()

        with timings.create_subcontext("eval_messages") as subtimings:
            candidate_message_values: List[float] = []
            for cf_game, bp_policy, (msg_dict, _) in zip(
                cf_games, rescored_bp_policies, candidate_messages
            ):
                cf_value = self._eval_action_under_bilateral_search(
                    cf_game,
                    agent_power,
                    agent_state,
                    pseudo_orders[game_phase][agent_power],
                    recipient_power,
                    bp_policy,
                    pair_value_table,
                    timings=subtimings,
                )
                candidate_message_values.append(cf_value)
                logging.info(f"Generated message \"{msg_dict['message']}\" (score: {cf_value})")

            # `additional_msg_dict` should only be set for debug purposes
            # If it is set, we evaluate not sending a message as well
            if additional_msg_dict is not None:
                cf_value = self._eval_action_under_bilateral_search(
                    game_2p,
                    agent_power,
                    agent_state,
                    pseudo_orders[game_phase][agent_power],
                    recipient_power,
                    base_policy,
                    pair_value_table,
                    timings=subtimings,
                )
                logging.info(f"<NO MESSAGE> (score: {cf_value})")

        selected_msg_dict = None
        if len(candidate_message_values) > 0:
            idx_ranking, strategy = self._rank_candidate_messages(candidate_message_values)
            assert len(set(idx_ranking)) == len(candidate_message_values)
            for idx in idx_ranking:
                msg_score = candidate_message_values[idx]
                msg_dict, annotator = candidate_messages[idx]
                if annotator is not None:
                    meta_annotations.append_annotator(annotator)

                if selected_msg_dict is None:
                    logging.info(
                        f"Message search maybe selecting message \"{msg_dict['message']}\" (score: {msg_score}; strategy: {strategy})"
                    )
                    with timings.create_subcontext("filter_messages") as subtimings:
                        logging.info(
                            f"Running message filtering on message \"{msg_dict['message']}\"."
                        )
                        should_filter = self.message_handler.message_filterer.should_filter_message(
                            msg_dict,
                            list(game.messages.values()),
                            game,
                            pseudo_orders,
                            game_is_missing_draw_votes=False,
                            timings=subtimings,
                        )
                    if not should_filter:
                        data = (
                            f"Message selected by message search (score: {msg_score}; strategy: {strategy})",
                            json.dumps(msg_dict),
                        )
                        meta_annotations.add_filtered_msg(data, msg_dict["time_sent"])
                        selected_msg_dict = msg_dict
                else:
                    # Mark remaining messages as rejected by message search
                    data = (
                        f"Message filtered by message search (score: {msg_score}; strategy: {strategy})",
                        json.dumps(msg_dict),
                    )
                    logging.info(
                        f"Message search did not select message \"{msg_dict['message']}\" (score: {msg_score}; strategy: {strategy})"
                    )
                    meta_annotations.add_filtered_msg(data, msg_dict["time_sent"])
                    meta_annotations.after_message_generation_failed(bad_tags=[TOKEN_DETAILS_TAG])

        return selected_msg_dict

    def _rank_candidate_messages(
        self, candidate_message_values: List[float]
    ) -> Tuple[List[int], str]:
        assert len(candidate_message_values) > 0

        max_score_diff = max(candidate_message_values) - min(candidate_message_values)
        max_rel_score_diff = max_score_diff / min(candidate_message_values)

        # Only run message search if the potential for improvement (in either absolute or relative
        # terms) meets particular thresholds
        if self.message_search_cfg.strategy == "NONE" or (
            max_score_diff < self.message_search_cfg.max_score_diff_threshold
            and max_rel_score_diff < self.message_search_cfg.max_rel_score_diff_threshold
        ):
            strategy = "NONE"
            ranking = list(range(len(candidate_message_values)))
            np.random.shuffle(ranking)
        elif self.message_search_cfg.strategy == "BEST":
            strategy = "BEST"
            ranking = list(reversed(np.argsort(candidate_message_values)))
        elif self.message_search_cfg.strategy == "SOFTMAX":
            strategy = "SOFTMAX"
            noisy_candidate_message_values = [
                x + np.random.gumbel() * self.message_search_cfg.softmax_temperature
                for x in candidate_message_values
            ]
            ranking = list(reversed(np.argsort(noisy_candidate_message_values)))
        elif self.message_search_cfg.strategy == "FILTER":
            strategy = "FILTER"
            ranking_sorted = list(reversed(np.argsort(candidate_message_values)))
            ranking_good = ranking_sorted[: self.message_search_cfg.filter_top_k]
            ranking_bad = ranking_sorted[self.message_search_cfg.filter_top_k :]
            np.random.shuffle(ranking_good)
            np.random.shuffle(ranking_bad)
            ranking = ranking_good + ranking_bad
        else:
            raise RuntimeError(f"Unknown strategy: {self.message_search_cfg.strategy}")

        return ranking, strategy

    def get_pseudo_orders(
        self,
        game: Game,
        power: Power,
        state: SearchBotAgentState,
        timings: Optional[TimingCtx] = None,
        recipient: Optional[Power] = None,
    ) -> PseudoOrders:
        logging.info(f"SearchBotAgent.get_pseudo_orders: {power} -> {recipient}")
        assert self.message_handler is not None
        cache = getattr(state, "pseudo_orders_cache")
        pseudo_orders = None

        if cache is not None:
            pseudo_orders = state.pseudo_orders_cache.maybe_get(
                game,
                power,
                self.message_handler.reuse_pseudo_for_consecutive_messages,
                self.message_handler.reuse_pseudo_for_phase,
                recipient=recipient,
            )

        if pseudo_orders is None:
            if self.cfr_messages:
                pseudo_orders = self.get_search_pseudo_orders(
                    game, power, state, recipient=recipient, timings=timings
                )
            else:
                pseudo_orders = self.message_handler.get_pseudo_orders_many_powers(
                    game, power, recipient=recipient
                )

        if cache is not None and pseudo_orders is not None:
            state.pseudo_orders_cache.set(game, pseudo_orders, recipient=recipient)

        assert pseudo_orders is not None
        return pseudo_orders

    def log_pseudoorder_consistency(
        self, game: Game, power: Power, pseudo_orders: JointAction, state: SearchBotAgentState
    ):
        last_pseudo_orders = state.get_last_pseudo_orders(game)
        if last_pseudo_orders is None:
            return

        self_consistency = get_action_consistency_frac(
            pseudo_orders[power], last_pseudo_orders[power]
        )

        logging.info(f"Pseudo-orders for {power} are {self_consistency:.3%} consistent for agent")
        if pseudo_orders[power] != last_pseudo_orders[power]:
            logging.info("After messages: ")
            for msg in state.get_new_messages(game):
                logging.info(f"  {msg['sender']} -> {msg['recipient']}: {msg['message']}")
            logging.info(f"My old pseudo: {last_pseudo_orders[power]}")
            logging.info(f"My new pseudo: {pseudo_orders[power]}")

    def generate_message(
        self,
        game: Game,
        power: Power,
        timestamp: Optional[Timestamp],
        state: AgentState,
        timings: Optional[TimingCtx] = None,
        recipient: Optional[Power] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
    ) -> Optional[MessageDict]:
        assert isinstance(state, SearchBotAgentState), state

        if timings is None:
            timings = TimingCtx()
        timings.start("init")

        if self.message_handler is None:
            return None

        if recipient is None:
            timings.start("get_recipient")
            recipient = self.message_handler.get_recipient(
                game, power, timestamp, state.get_sleepsix_cache()
            )
            timings.stop()
        assert recipient is not None

        # Fancy message re-generation only works with sleepsix code, not legacy code
        if not timestamp:
            timings.start("get_sleep_time")
            sleep_time = self.get_sleep_time(game, power, state, recipient)

            # To keep model in-distribution when force-sending messages with inf sleep time, condition on sleep time of 1 hour instead
            if sleep_time >= INF_SLEEP_TIME:
                if game.get_metadata("phase_minutes") == "5":
                    sleep_time = Timestamp.from_seconds(15)
                else:
                    sleep_time = Timestamp.from_seconds(60 * 60)

            last_msg_dct = get_last_message(game)
            last_message_ts = (
                last_msg_dct["time_sent"] if last_msg_dct else Timestamp.from_seconds(0)
            )

            timestamp = last_message_ts + sleep_time
            timings.stop()

        if pseudo_orders is None and self.message_handler.expects_pseudo_orders():
            with timings.create_subcontext("po") as subtimings:
                pseudo_orders = self.get_pseudo_orders(
                    game, power=power, state=state, recipient=recipient, timings=subtimings,
                )

        with timings("interrupt"):
            raise_if_should_stop(post_pseudoorders=True)

        assert (
            self.bilateral_cfg.strategy == "NONE" or self.cfr_messages
        ), "Bilateral strategy is a no-op if cfr_messages=false"

        should_perform_message_search = False
        if self.message_search_cfg is not None:
            in_movement_phase = game.current_short_phase.endswith("M")
            have_message_from_us = get_last_message_from(game, power) is not None
            have_message_from_them = get_last_message_from(game, recipient) is not None
            should_perform_message_search = (
                in_movement_phase and have_message_from_us and have_message_from_them
            )

        if should_perform_message_search:
            with timings.create_subcontext("message_search") as subtimings:
                maybe_msg_dict = self._generate_message_via_message_search(
                    game, power, state, timestamp, recipient, pseudo_orders, timings=subtimings,
                )
        else:
            with timings.create_subcontext("generate_message") as subtimings:
                maybe_msg_dict = self.message_handler.generate_message(
                    game,
                    power,
                    timestamp,
                    recipient,
                    pseudo_orders=pseudo_orders,
                    timings=subtimings,
                )

        if self.message_handler.model_sleep_classifier.is_sleepsix():
            if maybe_msg_dict is None:
                logging.info(f"Blocking messages to power: {recipient}")
                state.sleepsix_cache.block_messages_to_power(game, recipient)

        timings.stop()
        timings.pprint(logging.getLogger("timings").info)
        return maybe_msg_dict

    def _get_phase_pseudo_orders(
        self,
        game: Game,
        agent_power: Power,
        state: SearchBotAgentState,
        recipient: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
    ) -> Tuple[JointAction, JointAction]:
        """
        Pseudo-order generation, combining search and a plausible pseudo-orders (PPO) model.

        Returns: (phase_pseudo_orders, rollout_joint_action)
        """
        if timings is None:
            timings = TimingCtx()

        assert self.message_handler is not None
        assert recipient is not None
        is_ppo_compatible = isinstance(
            self.message_handler.model_pseudo_orders, ParlAIPlausiblePseudoOrdersWrapper
        ) and (
            game.get_metadata("last_dialogue_phase").endswith("M")
            or self.message_handler.model_pseudo_orders.opt.get("rollout_phasemajor")
        )
        assert is_ppo_compatible, (
            "Cannot use legacy pseudo-orders code. Use a model of type"
            " ParlAIPlausiblePseudoOrdersWrapper with rollout_phasemajor flag"
        )

        assert self.message_handler is not None
        model_pseudo_orders = self.message_handler.model_pseudo_orders
        model_pseudo_orders_executor = self.message_handler.model_pseudo_orders_executor
        assert model_pseudo_orders is not None
        assert model_pseudo_orders_executor is not None
        assert isinstance(
            model_pseudo_orders, ParlAIPlausiblePseudoOrdersWrapper
        ), model_pseudo_orders

        ##################################
        # 1. Compute "default" most likely pseudo-orders
        ##################################
        timings.start("greedy")
        # we want to get the most likely pseudo-orders as the "default", and as
        # a baseline probability to score other orders against.
        # For now, I will do it with greedy generation because it has the lowest compute cost.
        # But a more sound way would be beam search or finding the most likely out of N samples.
        old_args = model_pseudo_orders.set_generation_args("greedy")
        pseudo_orders = model_pseudo_orders.produce_joint_action_bilateral(
            game, agent_power, recipient=recipient
        )
        model_pseudo_orders.set_generation_args(**old_args)
        logging.info(f"greedy model pseudo orders: {pseudo_orders}")
        extra_plausible_orders = {
            pwr: ([a] if is_action_valid(game, pwr, a) else []) for pwr, a in pseudo_orders.items()
        }
        with timings("interruption"):
            raise_if_should_stop(post_pseudoorders=False)

        ##################################
        # 2. Compute the blueprint policy
        ##################################
        timings.start("inc_bp")
        bp_policy = self.maybe_get_incremental_bp(
            game,
            agent_power=agent_power,
            agent_state=state,
            extra_plausible_orders=extra_plausible_orders,
        )

        with timings("interruption"):
            raise_if_should_stop(post_pseudoorders=False)

        if bp_policy is None:
            with timings.create_subcontext("bp") as subtimings:
                bp_policy = self.order_sampler.sample_orders(
                    game,
                    agent_power=agent_power,
                    speaking_power=agent_power,
                    player_rating=self.player_rating if self.set_player_rating else None,
                    extra_plausible_orders=extra_plausible_orders,
                    timings=subtimings,
                )

        ##################################
        # 3. Compute the search policy
        ##################################
        if self.use_br_correlated_search(game.phase, "pseudo_order"):
            assert self.cfg.use_truthful_pseudoorders
            with timings.create_subcontext("br_against_bilateral_search") as subtimings:
                search_result = self.run_best_response_against_correlated_bilateral_search(
                    game,
                    agent_state=state,
                    bp_policy=bp_policy,
                    agent_power=agent_power,
                    timings=subtimings,
                )
        else:
            with timings.create_subcontext("run_search") as subtimings:
                search_result = self.run_search(
                    game,
                    agent_power=agent_power,
                    agent_state=state,
                    bp_policy=bp_policy,
                    timings=subtimings,
                )
        state.update(game, agent_power, search_result, None)
        with timings("interruption"):
            raise_if_should_stop(post_pseudoorders=False)

        # the policy for pseudo-orders should have the agent playing their policy,
        # and everyone else playing according to the population average
        search_pseudo_orders_policy = search_result.get_population_policy()
        search_pseudo_orders_policy[agent_power] = search_result.get_agent_policy()[agent_power]

        ##################################
        # 4. Find a pseudo-order for the agent and the recipient.
        #    - For the agent: most likely agent search action in the "support" of PPO model
        #    - For the recipient: best action for the agent in the "support" of the PPO model and the search population policy
        ##################################

        timings.start("selecting")
        # n.b.: We have to predict the recipient pseudo-order first, because the
        # plausible pseudo-order model predicts recipient then agent pseudo-orders. Thus,
        # it's crucial that we choose the recipient pseudo-order before using it to predict
        # plausible agent pseudo-orders
        # n.b.: We have to predict the speaker vs recipient pseudo-order in the same order
        # as the PPO model, to get the model auto-regressive dependencies correct.
        ppo_powers = (
            (agent_power, recipient)
            if model_pseudo_orders.is_speaker_first()
            else (recipient, agent_power)
        )
        for cur_power in ppo_powers:
            policy = search_pseudo_orders_policy[cur_power]
            policy_valid = policy

            # 4a. plausible pseudo orders stuff
            # We are relying on the fact that the agent orders are predicted last,
            # so they don't matter in computing logprobs for the recipient.
            need_ppo = True
            if self.cfg.use_truthful_pseudoorders and cur_power == agent_power:
                logging.info("Ignoring PPO for sender (truthful).")
                need_ppo = False
            if self.cfg.use_truthful_pseudoorders_recipient and cur_power != agent_power:
                logging.info("Ignoring PPO for recipient (truthful).")
                need_ppo = False

            if need_ppo or not self.cfg.skip_policy_evaluation_for_truthful_pseudoorders:
                score_batch_size = 10  # chunk into 10 at a time to avoid OOM
                score_list_batches_futures = []
                candidates = [{**pseudo_orders, cur_power: a} for a in policy]
                for i in range(0, len(candidates), score_batch_size):
                    chunk = candidates[i : i + score_batch_size]
                    score_list_batches_futures.append(
                        model_pseudo_orders_executor.compute(
                            "score_candidate_rollout_order_tokens_bilateral",
                            game,
                            chunk,
                            agent_power,
                            target_power=recipient,
                            batch_size=score_batch_size,
                        )
                    )
                score_list = sum([result.result() for result in score_list_batches_futures], [])

                ppo_order_logprobs = {a[cur_power]: lp[cur_power] for a, lp in score_list}
                ppo_logprobs = {a: float(sum(lp)) for a, lp in ppo_order_logprobs.items()}
                ppo_probs = {a: math.exp(lp) for a, lp in ppo_logprobs.items()}
                filtered_ppos = filter_logprob_ratio(ppo_logprobs, 0.1)
                if need_ppo:
                    policy_valid = {a: p for a, p in policy_valid.items() if a in filtered_ppos}
            else:
                score_list = ppo_order_logprobs = filtered_ppos = ppo_probs = None

            # 4b. find a recipient action that's "good for the agent"
            if isinstance(search_result, BRCorrBilateralSearchResult):
                value_to_me = search_result.value_to_me
            else:
                assert search_result.bilateral_stats is not None
                value_to_me = search_result.bilateral_stats.value_to_me
            if self.bilateral_cfg.strategy != "NONE" and cur_power != agent_power:
                assert self.bilateral_cfg.strategy == "BEST_POP"
                filtered_policy = filter_logprob_ratio(
                    {a: math.log(p) for a, p in policy.items()}, 0.1
                )
                policy_valid = {a: p for a, p in policy_valid.items() if a in filtered_policy}

                if policy_valid:  # else go with PPO default
                    logging.info(f"Choosing BEST_POP pseudo-order for {cur_power}")
                    pseudo_orders[cur_power] = max(
                        policy_valid, key=lambda a: value_to_me[cur_power, a].get_avg()
                    )
                else:
                    logging.info(f"Sticking with default pseudo-order for {cur_power}")
            else:
                filtered_policy = None
                if policy_valid:  # else go with PPO default
                    logging.info(f"Choosing SEARCH pseudo-order for {cur_power}")

                    if max(policy_valid.values()) > 0.02:
                        pseudo_orders[cur_power] = argmax_p_dict(policy_valid)
                    else:
                        # there's no valid action in the support of the BQRE equilibrium
                        # so just pick the action with highest QRE value (i.e. what we would have played
                        # if only the valid actions were availableq).
                        agent_type = search_result.agent_type  # type:ignore
                        type_data = search_result.brm_data.type_cfr_data[agent_type]
                        agent_qre_values = {
                            a: type_data.avg_action_utility(agent_power, a)
                            + type_data.power_qre_lambda[agent_power]
                            * np.log(bp_policy[agent_power][a])  # type: ignore
                            for a in policy_valid
                        }
                        pseudo_orders[cur_power] = argmax_p_dict(agent_qre_values)

                else:
                    logging.info(f"Sticking with default pseudo-order for {cur_power}")

            # =================== logging ==================
            def x_in(a, pi):
                return "-" if pi is None else ("x" if a in pi else "")

            rows = [
                (
                    bp_policy[cur_power][action],
                    policy[action],
                    x_in(action, filtered_policy),
                    ppo_probs[action] if ppo_probs else "",
                    x_in(action, filtered_ppos),
                    value_to_me[cur_power, action].get_avg(),
                    "x" if action == pseudo_orders[cur_power] else "",
                    color_order_logprobs(action, ppo_order_logprobs),
                )
                for action in policy
            ]
            logging.info(
                "\n"
                + tabulate.tabulate(
                    rows,
                    headers=["bp_p", "search_p", "val", "ppo_p", "val", "v_me", "sel", "action"],
                    floatfmt=".2g",
                )
            )
            # ===========================================

        logging.info(f"Pseudo orders for {agent_power}: {pseudo_orders}")

        self.log_pseudoorder_consistency(game, agent_power, pseudo_orders, state)

        state.update(game, agent_power, search_result, pseudo_orders)

        # joint action for rollout is pseudo-orders with other powers filled in with
        # their most likely action
        population_policy = search_result.get_population_policy()
        argmax_orders = {pwr: argmax_p_dict(population_policy[pwr]) for pwr in POWERS}
        # make sure pseudo-orders are valid actions
        valid_pseudo_orders = {
            pwr: a if is_action_valid(game, pwr, a) else argmax_orders[pwr]
            for pwr, a in pseudo_orders.items()
        }
        # valid_pseudo_orders = pseudo_orders
        if valid_pseudo_orders != pseudo_orders:
            logging.info(
                f"Updated pseudo-orders because some weren't valid: {valid_pseudo_orders}"
            )
        joint_action = {pwr: valid_pseudo_orders.get(pwr, argmax_orders[pwr]) for pwr in POWERS}
        timings.stop()

        return valid_pseudo_orders, joint_action

    def get_search_pseudo_orders(
        self,
        game: Game,
        power: Power,
        agent_state: SearchBotAgentState,
        recipient: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
    ) -> PseudoOrders:
        if timings is None:
            timings = TimingCtx()
        assert self.message_handler is not None
        assert self.message_handler.model_pseudo_orders is not None

        rollout_pseudo_orders = {}
        game_future = Game(game)
        last_dialogue_phase = game.current_short_phase
        game_future.set_metadata("last_dialogue_phase", last_dialogue_phase)
        rollout_type = self.message_handler.model_pseudo_orders.expected_rollout_type()
        has_seen_move_phase = False
        assert recipient is not None
        while True:
            logging.info(
                f"SearchBotAgent.get_search_pseudo_orders {power} -> {recipient} : dialogue= {last_dialogue_phase} search= {game_future.current_short_phase}"
            )

            if has_seen_move_phase and self.cfg.use_greedy_po_for_rollout:
                timings.start("compute_greedy_pos")
                logging.info("Skipping search and doing greedy PO for the rest of the phases")
                model_pseudo_orders = self.message_handler.model_pseudo_orders
                model_pseudo_orders._use_fast_prefix_decoding = True  # type: ignore
                old_args = model_pseudo_orders.set_generation_args("greedy")
                precomputed_greedy_rollout_pseudo_orders = self.message_handler.model_pseudo_orders.produce_rollout_joint_action_bilateral(
                    game_future, power, recipient
                )
                model_pseudo_orders.set_generation_args(**old_args)
                logging.info(f"Computed greedy POs for {precomputed_greedy_rollout_pseudo_orders}")
                rollout_pseudo_orders.update(precomputed_greedy_rollout_pseudo_orders)
                break

            with timings.create_subcontext("phase_pos") as subtimings:
                pseudo_current, joint_action_current = self._get_phase_pseudo_orders(
                    game_future, power, agent_state, recipient=recipient, timings=subtimings
                )

            if game_future.current_short_phase.endswith("M"):
                has_seen_move_phase = True

            rollout_pseudo_orders[game_future.current_short_phase] = pseudo_current

            if (
                game_future.current_short_phase.endswith("M")
                and (
                    game_future.current_short_phase != last_dialogue_phase
                    or rollout_type == RolloutType.RA_ONLY
                )
                or rollout_type == RolloutType.NONE
            ):
                break

            game_future.set_all_orders(joint_action_current)
            game_future.process()

            if game_future.current_short_phase == "COMPLETED":
                # ensure that this is still a valid rollout order
                rollout_pseudo_orders[next_M_phase(game.current_short_phase)] = {
                    p: tuple() for p in POWERS
                }

                break
        timings.stop()

        return PseudoOrders(rollout_pseudo_orders)

    def maybe_get_incremental_bp(
        self,
        game: Game,
        agent_power: Power,
        agent_state: SearchBotAgentState,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        parlai_req_size: int = 10,
        policy_top_n: int = -1,
    ) -> Optional[PowerPolicies]:

        if extra_plausible_orders is None:
            extra_plausible_orders = {}

        last_search_result = agent_state.get_last_search_result(game)
        if last_search_result is None or not self.do_incremental_search:
            return None

        # If this is a "rollout" phase we can't guarantee that the game state is the same as
        # last time. We could try to be careful but lets just bail.
        last_dialogue_phase = game.get_metadata("last_dialogue_phase")
        if last_dialogue_phase and last_dialogue_phase != game.current_short_phase:
            return None

        recent_messages = agent_state.get_new_messages(game)

        powers_to_update = set([m["sender"] for m in recent_messages]) | set(
            [m["recipient"] for m in recent_messages]
        )
        powers_to_update &= set(POWERS)  # don't allow ALL
        last_bp = last_search_result.get_bp_policy()
        last_agent_policy = last_search_result.get_agent_policy()
        last_pop_policy = last_search_result.get_population_policy()

        # only take the top N actions by probability in the search population
        if policy_top_n > 0:
            last_bp_thinned = {
                pwr: {
                    a: last_bp[pwr][a]
                    for a in (
                        list(last_bp[pwr])[:policy_top_n]
                        + list(last_agent_policy[pwr])[:policy_top_n]
                        + list(last_pop_policy[pwr])[:policy_top_n]
                    )
                }
                for pwr in POWERS
            }
        else:
            last_bp_thinned = copy.deepcopy(last_bp)

        # in theory, incremental updates could keep increasing the #plausible
        # if policy_top_n is not set. We can't cut it off after incremental_update
        # because we don't want to remove the extra plausible orders.
        # # So lets just cut it off here.
        limits = self.order_sampler.get_plausible_order_limits(game)
        last_bp_thinned = cutoff_policy(last_bp_thinned, limits)

        for pwr, actions in extra_plausible_orders.items():
            for a in actions:
                if a in last_bp[pwr] and not recent_messages:
                    # can use the cached BP prob
                    last_bp_thinned[pwr][a] = last_bp[pwr][a]
                else:
                    # need to recompute the BP prob
                    last_bp_thinned[pwr][a] = 0.0
                    powers_to_update.add(pwr)

        logging.info(f"Incremental update will update orders for: {powers_to_update}")

        return self.order_sampler.incremental_update_policy(
            game,
            last_bp_thinned,
            agent_power,
            powers=list(powers_to_update),
            parlai_req_size=parlai_req_size if recent_messages else 0,
        )

    def try_get_cached_search_result(
        self, game: Game, state: SearchBotAgentState
    ) -> Optional[SearchResult]:
        if not state.get_new_messages(game):
            return state.get_last_search_result(game)
        return None

    def can_sleep(self) -> bool:
        return self.message_handler is not None and self.message_handler.has_sleep_classifier()

    def get_sleep_time(
        self, game: Game, power: Power, state: AgentState, recipient: Optional[Power] = None,
    ) -> Timestamp:
        if not self.can_sleep():
            raise RuntimeError("This agent doesn't know how to sleep.")
        assert self.message_handler is not None
        assert isinstance(state, SearchBotAgentState)

        return self.message_handler.get_sleep_time(
            game, power, state.get_sleepsix_cache(), recipient
        )

    def log_cfr_iter_state(
        self,
        *,
        game,
        pwr,
        actions,
        cfr_data,
        cfr_iter,
        state_utility,
        action_utilities,
        power_sampled_orders,
        ptype=None,
    ):
        power_is_loser = self.get_power_loser_dict(cfr_data, cfr_iter)
        ptype_str = f":{ptype}" if ptype else ""
        logging.info(
            f"<> [ {cfr_iter+1} / {self.n_rollouts} ] {pwr}{ptype_str} {game.phase} avg_utility={cfr_data.avg_utility(pwr):.5f} cur_utility={state_utility:.5f} "
            f"is_loser= {int(power_is_loser[pwr])}"
        )
        logging.info(f">> {pwr} cur action at {cfr_iter+1}: {power_sampled_orders[pwr]}")
        logging.info(f"     {'probs':8s}  {'bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders")
        action_probs: List[float] = cfr_data.avg_strategy(pwr)
        bp_probs: List[float] = cfr_data.bp_strategy(pwr)
        avg_utilities: List[float] = cfr_data.avg_action_utilities(pwr)
        sorted_metrics = sorted(
            zip(actions, action_probs, bp_probs, avg_utilities, action_utilities),
            key=lambda ac: -ac[1],
        )
        for orders, p, bp_p, avg_u, cur_u in sorted_metrics:
            logging.info(f"|>  {p:8.5f}  {bp_p:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}")

    def compute_nash_conv(
        self,
        cfr_data: CFRData,
        label: str,
        game: Game,
        strat_f: Callable[[Power], List[float]],
        maybe_rollout_results_cache: Optional[RolloutResultsCache],
        *,
        agent_power: Optional[Power],
        br_iters: int = 1000,
        verbose: bool = True,
    ):
        """For each power, compute EV of each action assuming opponent ave policies"""

        # get policy probs for all powers
        power_action_ps: Dict[Power, List[float]] = {
            pwr: strat_f(pwr) for (pwr, actions) in cfr_data.power_plausible_orders.items()
        }
        if verbose:
            logging.info("Policies: {}".format(power_action_ps))

        total_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        temp_action_utilities: Dict[Tuple[Power, Action], float] = defaultdict(float)
        total_state_utility: Dict[Power, float] = defaultdict(float)
        max_state_utility: Dict[Power, float] = defaultdict(float)
        for pwr, actions in cfr_data.power_plausible_orders.items():
            total_state_utility[pwr] = 0
            max_state_utility[pwr] = 0
        # total_state_utility = [0 for u in idxs]
        nash_conv = 0
        for _ in range(br_iters):
            # sample policy for all powers
            idxs, power_sampled_orders = sample_orders_from_policy(
                cfr_data.power_plausible_orders, power_action_ps
            )

            # for each power: compare all actions against sampled opponent action
            set_orders_dicts = make_set_orders_dicts(
                cfr_data.power_plausible_orders, power_sampled_orders
            )

            all_rollout_results = self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                game,
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                cache=maybe_rollout_results_cache,
            )

            for pwr, actions in cfr_data.power_plausible_orders.items():
                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )
                assert len(results) == len(actions)

                for r in results:
                    action = r[0][pwr]
                    val = r[1][pwr]
                    temp_action_utilities[(pwr, action)] = val
                    total_action_utilities[(pwr, action)] += val
                # logging.info("results for power={}".format(pwr))
                # for i in range(len(cfr_data.power_plausible_orders[pwr])):
                #     action = cfr_data.power_plausible_orders[pwr][i]
                #     util = action_utilities[i]
                #     logging.info("{} {} = {}".format(pwr,action,util))

                # for action in cfr_data.power_plausible_orders[pwr]:
                #     logging.info("{} {} = {}".format(pwr,action,action_utilities))
                # logging.info("action utilities={}".format(action_utilities))
                # logging.info("Results={}".format(results))
                # state_utility = np.dot(power_action_ps[pwr], action_utilities)
                # action_regrets = [(u - state_utility) for u in action_utilities]
                # logging.info("Action utilities={}".format(temp_action_utilities))
                # for action in actions:
                #     total_action_utilities[(pwr,action)] += temp_action_utilities[(pwr,action)]
                # logging.info("Total action utilities={}".format(total_action_utilities))
                # total_state_utility[pwr] += state_utility
            assert len(all_rollout_results) == 0
        # total_state_utility[:] = [x / 100 for x in total_state_utility]
        for pwr, actions in cfr_data.power_plausible_orders.items():
            # ps = self.avg_strategy(pwr, cfr_data.power_plausible_orders[pwr])
            for i in range(len(actions)):
                action = actions[i]
                total_action_utilities[(pwr, action)] /= br_iters
                if total_action_utilities[(pwr, action)] > max_state_utility[pwr]:
                    max_state_utility[pwr] = total_action_utilities[(pwr, action)]
                total_state_utility[pwr] += (
                    total_action_utilities[(pwr, action)] * power_action_ps[pwr][i]
                )

        for pwr, actions in cfr_data.power_plausible_orders.items():
            state_regret = max_state_utility[pwr] - total_state_utility[pwr]
            logging.info(
                f"results for power={pwr} value={total_state_utility[pwr]:.6g} diff={state_regret:.6g} [cfr_data value={cfr_data.avg_utility(pwr):.6g}]"
            )
            nash_conv += state_regret
            if verbose:
                for i in range(len(actions)):
                    action = actions[i]
                    logging.info(
                        f"{pwr} {action} = {total_action_utilities[(pwr, action)]:.6g} (prob {power_action_ps[pwr][i]:.6g}) (cfr_util= {cfr_data.avg_action_utility(pwr, action):.6g})"
                    )

        logging.info(f"Nash conv for {label} = {nash_conv}")
        return nash_conv

    def eval_policy_values(
        self,
        game: Game,
        policy: PowerPolicies,
        *,
        agent_power: Optional[Power],
        n_rollouts: int = 1000,
    ) -> Dict[Power, float]:
        """Compute the EV of a {pwr: policy} dict at a state by running `n_rollouts` rollouts.

        Returns:
            - {power: avg_sos}
        """

        power_actions = {pwr: list(p.keys()) for pwr, p in policy.items()}
        power_action_probs = {pwr: list(p.values()) for pwr, p in policy.items()}

        set_orders_dicts = [
            sample_orders_from_policy(power_actions, power_action_probs)[1]
            for _i in range(n_rollouts)
        ]

        rollout_results = self.base_strategy_model_rollouts.do_rollouts(
            game, agent_power=agent_power, set_orders_dicts=set_orders_dicts
        )

        utilities = {
            pwr: mean([values[pwr] for order, values in rollout_results]) for pwr in POWERS
        }
        return utilities

    def get_power_loser_dict(self, cfr_data, cfr_iter) -> Dict[Power, bool]:
        "Determine which powers are 'losers' and should therefore play BP"
        if cfr_iter >= self.loser_bp_iter and self.loser_bp_value > 0:
            return {
                pwr: all(u < self.loser_bp_value for u in cfr_data.avg_action_utilities(pwr))
                for pwr in POWERS
            }
        else:
            return {pwr: False for pwr in POWERS}

    def is_verbose_log_iter(self, cfr_iter) -> bool:
        "Return true if we should do verbose logging on this search iteration."
        return (
            (
                self.log_intermediate_iterations
                and cfr_iter & (cfr_iter + 1) == 0
                and cfr_iter > self.n_rollouts / 8
            )
            or cfr_iter == self.n_rollouts - 1
            or (self.log_intermediate_iterations and (cfr_iter + 1) == self.bp_iters)
        )

    def get_cur_iter_strategies(
        self, cfr_data: CFRData, cfr_iter: int
    ) -> Dict[Power, List[float]]:
        "Get the current strategy for each power; either CFR"
        power_is_loser = self.get_power_loser_dict(cfr_data, cfr_iter)
        return {
            pwr: (
                cfr_data.bp_strategy(pwr)
                if (
                    cfr_iter < self.bp_iters
                    or np.random.rand() < self.bp_prob  # type:ignore
                    or power_is_loser[pwr]
                    or pwr == self.exploited_agent_power
                )
                else cfr_data.cur_iter_strategy(pwr)
            )
            for pwr in cfr_data.power_plausible_orders
        }

    def postprocess_sleep_heuristics_should_trigger(
        self, msg: MessageDict, game: Game, state: AgentState,
    ) -> MessageHeuristicResult:
        logging.info("Running postprocess_sleep_heuristics_should_trigger")
        assert isinstance(state, SearchBotAgentState)
        sender, recipient = msg["sender"], msg["recipient"]
        if self.message_handler and self.message_handler.use_pseudoorders_initiate_sleep_heuristic:
            pseudo_orders = state.pseudo_orders_cache.maybe_get(
                game, sender, True, False, recipient
            )
            if pseudo_orders is not None:
                # Should we block messages in the last season?
                if (
                    self.message_handler.use_last_phase_silence_except_coordination_heuristic
                    and game.current_year == self.message_handler.grounding_last_playable_year
                    and game.current_short_phase.startswith("F")
                    and not joint_action_contains_xpower_support_or_convoy(
                        game, sender, recipient, pseudo_orders.first_joint_action()
                    )
                ):
                    return MessageHeuristicResult.BLOCK
                # Should we force-initiate a message based on pseudos?
                if pseudoorders_initiate_sleep_heuristics_should_trigger(
                    game, sender, recipient, pseudo_orders
                ):
                    return MessageHeuristicResult.FORCE
        return MessageHeuristicResult.NONE


def augment_plausible_orders(
    game: Game,
    power_plausible_orders: PowerPolicies,
    agent: SearchBotAgent,
    cfg: agents_cfgs.SearchBotAgent.PlausibleOrderAugmentation,
    agent_power: Optional[Power] = None,
    *,
    limits: List[int],
) -> PowerPolicies:
    policy_model = agent.base_strategy_model.model
    augmentation_type = cfg.which_augmentation_type
    if augmentation_type is None:
        return power_plausible_orders
    if not game.current_short_phase.endswith("M"):

        return power_plausible_orders

    if augmentation_type == "do":
        aug_cfg = cfg.do
        policy, _, _ = fairdiplomacy.action_exploration.double_oracle(
            game,
            agent,
            double_oracle_cfg=aug_cfg,
            initial_plausible_orders_policy=power_plausible_orders,
            agent_power=agent_power,
        )
        return policy

    assert augmentation_type == "random"
    aug_cfg = cfg.random

    # Creating a copy.
    power_plausible_orders = dict(power_plausible_orders)

    for power in game.get_alive_powers():
        if power == agent.exploited_agent_power:
            continue
        actions = fairdiplomacy.action_generation.generate_order_by_column_from_base_strategy_model(
            policy_model, game, selected_power=power, agent_power=agent_power
        )
        logging.info(
            "Found %s actions for %s. Not in plausible: %s",
            len(actions),
            power,
            len(frozenset(actions).difference(power_plausible_orders[power])),
        )
        max_actions = limits[POWERS.index(power)]
        # Creating space for new orders.
        orig_size = len(power_plausible_orders[power])
        power_plausible_orders[power] = dict(
            collections.Counter(power_plausible_orders[power]).most_common(
                max(aug_cfg.min_actions_to_keep, max_actions - aug_cfg.max_actions_to_drop)
            )
        )
        random.shuffle(actions)
        logging.info("Addding extra plausible orders for %s", power)
        if orig_size != len(power_plausible_orders[power]):
            logging.info(
                " (deleted %d least probable actions)",
                orig_size - len(power_plausible_orders[power]),
            )
        for action in actions:
            if len(power_plausible_orders[power]) >= max_actions:
                break
            if action not in power_plausible_orders[power]:
                power_plausible_orders[power][action] = 0
                logging.info("       %s", action)

    renormalize_policy(power_plausible_orders)

    return power_plausible_orders


def sample_all_joint_orders(power_actions: Dict[Power, List[Action]]) -> List[Dict[Power, Action]]:
    power_actions = dict(power_actions)
    for pwr in list(power_actions):
        if not power_actions[pwr]:
            power_actions[pwr] = [tuple()]

    all_orders = []
    powers, action_sets = zip(*power_actions.items())
    for joint_action in itertools.product(*action_sets):
        all_orders.append(dict(zip(powers, joint_action)))
    return all_orders


def mean(L: List[float], eps=0.0):
    return sum(L) / (len(L) + eps)


def get_action_consistency_frac(a1: Action, a2: Action) -> float:
    """Returns the fraction of orders that are consistent between a1 and a2"""
    # assert len(a1) == len(a2), f"{a1} {a2}"  # in R/A phases, actions may have different len
    if len(a1) == 0:
        return 1.0
    common = set(a1) & set(a2)
    return len(common) / max(len(a1), len(a2))


if __name__ == "__main__":
    import pathlib
    import heyhi

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    np.random.seed(0)  # type:ignore
    torch.manual_seed(0)  # type: ignore

    game = Game()
    cfg = heyhi.load_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "conf/common/agents/searchbot_03_fastbot_loser.prototxt",
        overrides=["searchbot.n_rollouts=64"],
    )
    agent = SearchBotAgent(cfg.searchbot)
    print(agent.get_orders(game, power="AUSTRIA", state=agent.initialize_state(power="AUSTRIA")))
