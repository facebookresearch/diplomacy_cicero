#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import copy
import dataclasses
import logging
import math
import time
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from conf import agents_cfgs
from fairdiplomacy import pydipcc

from fairdiplomacy.agents.base_agent import AgentState
from fairdiplomacy.agents.base_search_agent import (
    SearchResult,
    make_set_orders_dicts,
    sample_orders_from_policy,
)
from fairdiplomacy.agents.bilateral_stats import BilateralStats
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent, SearchBotAgentState, CFRData
from fairdiplomacy.agents.base_strategy_model_rollouts import RolloutResultsCache
from fairdiplomacy.agents.br_corr_bilateral_search import (
    extract_bp_policy_for_powers,
    compute_weights_for_opponent_joint_actions,
    compute_best_action_against_reweighted_opponent_joint_actions,
    sample_joint_actions,
    filter_invalid_actions_from_policy,
    rescore_bp_from_bilateral_views,
    compute_payoff_matrix_for_all_opponents,
    BRCorrBilateralSearchResult,
)
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import (
    Action,
    BilateralConditionalValueTable,
    JointAction,
    Phase,
    PlausibleOrders,
    PlayerType,
    PlayerTypePolicies,
    Policy,
    Power,
    PowerPolicies,
    RolloutResults,
)
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.agent_interruption import raise_if_should_stop


RescoringPolicyName = str


class BRMResult(SearchResult):
    def __init__(
        self,
        bp_policies: PowerPolicies,
        ptype_avg_policies: PlayerTypePolicies,
        ptype_final_policies: PlayerTypePolicies,
        beliefs: Dict[Power, np.ndarray],
        agent_type: PlayerType,
        use_final_iter: bool,
        brm_data: Optional["BRMData"],
        bilateral_stats: Optional[BilateralStats] = None,
    ):
        self.bp_policies = bp_policies
        self.ptype_avg_policies = ptype_avg_policies
        self.ptype_final_policies = ptype_final_policies
        self.brm_data = brm_data  # type: ignore
        # make sure we don't hold on to mutable belief arrays
        self.beliefs = copy.deepcopy(beliefs)
        self.agent_type = agent_type
        self.use_final_iter = use_final_iter
        self.bilateral_stats = bilateral_stats

    def get_bp_policy(self) -> PowerPolicies:
        return self.bp_policies

    def get_agent_policy(self) -> PowerPolicies:
        return self.ptype_avg_policies[self.agent_type]

    def get_population_policy(self, power: Optional[Power] = None) -> PowerPolicies:
        return belief_weighted_policy(self.beliefs, self.ptype_avg_policies, power)

    def sample_action(self, power) -> Action:
        ptype_policies = (
            self.ptype_final_policies if self.use_final_iter else self.ptype_avg_policies
        )
        return sample_p_dict(ptype_policies[self.agent_type][power])

    def avg_utility(self, pwr: Power) -> float:
        assert self.brm_data is not None
        return sum(
            cfr_data.avg_utility(pwr) * self.beliefs[pwr][ptype]
            for ptype, cfr_data in self.brm_data.type_cfr_data.items()
        )

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        assert self.brm_data is not None
        return sum(
            cfr_data.avg_action_utility(pwr, a) * self.beliefs[pwr][ptype]
            for ptype, cfr_data in self.brm_data.type_cfr_data.items()
        )

    def is_early_exit(self) -> bool:
        return self.brm_data is None

    def get_bilateral_stats(self) -> BilateralStats:
        assert self.bilateral_stats is not None
        return self.bilateral_stats


class BRMData:
    def __init__(self, type_cfr_data: Dict[PlayerType, CFRData]):
        self.type_cfr_data: Dict[PlayerType, CFRData] = type_cfr_data

    def get_best_type_data(self):
        return self.type_cfr_data[0]

    def get_type_data(self, ptype: PlayerType):
        return self.type_cfr_data[ptype]


class BeliefState:
    def __init__(self, player_types: List[PlayerType]):
        self.player_types = player_types
        self.updated_phases: Set[str]
        self.beliefs: Dict[Power, np.ndarray]
        self.reset()

    @property
    def num_player_types(self):
        return len(self.player_types)

    def __str__(self):
        return "\n".join(
            [
                f"{pwr}: {np.array2string(belief, precision=2, suppress_small=True)}\n"  # type: ignore
                for pwr, belief in self.beliefs.items()
            ]
        )

    def reset(self):
        self.updated_phases = set()
        self.beliefs = {
            pwr: np.ones(self.num_player_types) / self.num_player_types for pwr in POWERS
        }


def belief_weighted_policy(
    beliefs: Dict[Power, np.ndarray],
    ptype_policies: Dict[PlayerType, PowerPolicies],
    power: Optional[Power],
) -> PowerPolicies:
    ret = {}
    for pwr in beliefs:
        if power is not None and pwr != power:
            continue
        ret[pwr] = {}
        for ptype, ptype_prob in enumerate(beliefs[pwr].tolist()):
            ptype_policy = ptype_policies[ptype][pwr]
            for action, action_prob in ptype_policy.items():
                if action not in ret[pwr]:
                    ret[pwr][action] = 0
                ret[pwr][action] += ptype_prob * action_prob

    ret_sorted = {
        pwr: dict(sorted(pi.items(), key=lambda ac_p: -ac_p[1])) for pwr, pi in ret.items()
    }
    return ret_sorted


class BayesAgentState(SearchBotAgentState):
    def __init__(self, power: Power, player_types: List[PlayerType]):
        super().__init__(power)
        self.belief_state = BeliefState(player_types)
        self.dynamic_lambda_scale_cache: Dict[Phase, Dict[Power, float]] = {}


@dataclasses.dataclass
class PlayerTypeSpec:
    # To be used to identify blueprint policy.
    BLUEPRINT_POLICY_NAME: ClassVar[RescoringPolicyName] = "BP"

    qre_lambda: float
    # Model path or BLUEPRINT_POLICY_NAME. Used to debug print and such.
    policy_name: RescoringPolicyName
    # Optional for BLUEPRINT_POLICY_NAME. Required otherwise.
    rescore_policy_path: Optional[str]

    # Arbitrary extra string to identify this type. The collection of all type should have unique names.
    extra_tag: str = ""

    @property
    def name(self) -> str:
        name = "{:}*{:.2e}".format(self.policy_name, self.qre_lambda)
        if self.extra_tag:
            name = f"{name}_{self.extra_tag}"
        return name

    def assert_valid(self) -> None:
        try:
            assert 0 <= self.qre_lambda
            assert (self.policy_name == self.BLUEPRINT_POLICY_NAME) == (
                self.rescore_policy_path is None
            )
        except AssertionError:
            logging.error("Invalid spec: %s", self)
            raise


def _make_bqre_data(
    policies: Dict[RescoringPolicyName, PowerPolicies],
    specs: Dict[PlayerType, PlayerTypeSpec],
    qre: agents_cfgs.SearchBotAgent.QRE,
    scale_lambdas: float,
    pow_lambdas: float,
    scale_lambdas_by_power: Dict[Power, float],
    agent_power: Optional[Power],
) -> BRMData:
    logging.info(f"Making BQRE data with scale_lambdas {scale_lambdas} pow_lambdas {pow_lambdas}")
    return BRMData(
        {
            player_type: CFRData(
                bp_policy=policies[spec.policy_name],
                use_optimistic_cfr=False,
                qre=agents_cfgs.SearchBotAgent.QRE(
                    **{
                        **qre.to_dict(),
                        "qre_lambda": math.pow(spec.qre_lambda, pow_lambdas) * scale_lambdas,
                    }
                ),
                agent_power=agent_power,
                scale_lambdas_by_power=scale_lambdas_by_power,
            )
            for player_type, spec in specs.items()
        }
    )


def _migrate_old_lambdas(cfg: agents_cfgs.BQRE1PAgent) -> agents_cfgs.BQRE1PAgent:
    # Next 2 lines -> deep copy.
    cfg_proto = type(cfg.to_editable())()
    cfg_proto.CopyFrom(cfg.to_editable())

    if cfg.player_types.which_player_types is not None:
        assert (
            cfg.lambda_min is None and cfg.lambda_multiplier is None
        ), "Do not specify lambda_min and lambda_multiplier when using player_types"
    else:
        assert cfg.lambda_min is not None and cfg.lambda_multiplier is not None
        assert cfg.lambda_min > 0.0, "qre_lambda needs to be > 0"
        assert (
            cfg.lambda_multiplier > 0.0 and cfg.lambda_multiplier != 1.0
        ), "lambda_multiplier needs to be > 0 and not equal to 1"

        cfg_proto.player_types.log_uniform.min_lambda = cfg.lambda_min * cfg.lambda_multiplier
        assert cfg.num_player_types is not None
        cfg_proto.player_types.log_uniform.max_lambda = cfg.lambda_min * (
            cfg.lambda_multiplier ** cfg.num_player_types
        )

        cfg_proto.ClearField("lambda_min")
        cfg_proto.ClearField("lambda_multiplier")

        logging.info("Migrated lambda-player-type configuration:\n%s", cfg_proto)

    return cfg_proto.to_frozen()


def _compute_player_specs(
    cfg: agents_cfgs.BQRE1PAgent.PlayerTypes, num_player_types: int
) -> List[PlayerTypeSpec]:
    assert num_player_types >= 1
    if cfg.log_uniform is not None:
        subcfg = cfg.log_uniform
        assert subcfg.min_lambda is not None
        assert subcfg.max_lambda is not None

        num_policies = max(1, len(subcfg.policies))

        assert (num_player_types - int(subcfg.include_zero_lambda)) % num_policies == 0, (
            "num_player_types should account for the zero type and all policies",
        )

        num_lambdas = (num_player_types - int(subcfg.include_zero_lambda)) // num_policies
        if num_lambdas == 0:
            lambdas = []
        elif num_lambdas == 1:
            assert subcfg.max_lambda == subcfg.min_lambda
            lambdas = [subcfg.min_lambda]
        else:
            multiplier = (subcfg.max_lambda / subcfg.min_lambda) ** (1.0 / (num_lambdas - 1))
            # This slightly weird way of computing the lambda is to maintain exact consistency
            # with earlier behavior
            lambdas = [
                float(np.float32(subcfg.min_lambda / multiplier)) * (multiplier ** x)
                for x in range(1, num_lambdas + 1)
            ]
        specs = []
        if subcfg.include_zero_lambda:
            # We don't care wichh policy to user for lambda=0, as we will ignore
            # it anyways. So we just use the BP.
            specs.append(
                PlayerTypeSpec(
                    qre_lambda=0.0,
                    policy_name=PlayerTypeSpec.BLUEPRINT_POLICY_NAME,
                    rescore_policy_path=None,
                )
            )
        if not subcfg.policies:
            # Using "blueprint" policy.
            specs.extend(
                PlayerTypeSpec(
                    qre_lambda=qre_lambda,
                    policy_name=PlayerTypeSpec.BLUEPRINT_POLICY_NAME,
                    rescore_policy_path=None,
                )
                for qre_lambda in lambdas
            )
        else:
            for policy in subcfg.policies:
                if policy.model_path is None:
                    assert policy.name is None
                    specs.extend(
                        PlayerTypeSpec(
                            qre_lambda=qre_lambda,
                            policy_name=PlayerTypeSpec.BLUEPRINT_POLICY_NAME,
                            rescore_policy_path=None,
                        )
                        for qre_lambda in lambdas
                    )
                else:
                    assert policy.name is not None
                    specs.extend(
                        PlayerTypeSpec(
                            qre_lambda=qre_lambda,
                            policy_name=policy.name,
                            rescore_policy_path=policy.model_path,
                        )
                        for qre_lambda in lambdas
                    )
    else:
        raise RuntimeError("Unknown player types")
    assert specs, "No player types!"
    # Names must be unique.
    [[most_common_name, freq]] = collections.Counter(x.name for x in specs).most_common(1)
    assert freq == 1, f"Name duplication for player type: {most_common_name}"
    return specs


class BQRE1PAgent(SearchBotAgent):
    """One-ply bayes qre rm with base_strategy_model-policy rollouts"""

    def __init__(self, cfg: agents_cfgs.BQRE1PAgent, *, skip_base_strategy_model_cache=False):
        #################  refactored from base class ###########################
        super().__init__(
            cfg.base_searchbot_cfg, skip_base_strategy_model_cache=skip_base_strategy_model_cache
        )
        assert cfg.num_player_types is not None
        self.num_player_types = cfg.num_player_types
        assert cfg.agent_type is not None
        # The rest of the code uses 0-based indexing of the types. This should
        # be the only place we convert 1-based to 0-based.
        self.agent_type = cfg.agent_type - 1
        self.player_types = list(range(self.num_player_types))
        self.agent_type_is_public = cfg.agent_type_is_public

        assert self.exploited_agent_power is None, "exploited agent power is not supported"
        ##########################################################################

        self.qre_type2spec: Dict[PlayerType, PlayerTypeSpec] = {}
        self.qre_extra_rescoring_models: Dict[RescoringPolicyName, PlausibleOrderSampler] = {}
        self.pow_lambdas_1901 = cfg.pow_lambdas_1901
        self.scale_lambdas_1901 = cfg.scale_lambdas_1901
        self.pow_lambdas_1901_spring = cfg.pow_lambdas_1901_spring
        self.scale_lambdas_1901_spring = cfg.scale_lambdas_1901_spring

        self.dynamic_lambda_stdev_espilon = cfg.dynamic_lambda_stdev_espilon
        self.dynamic_lambda_stdev_baseline = cfg.dynamic_lambda_stdev_baseline
        self.dynamic_lambda_stdev_num_samples = cfg.dynamic_lambda_stdev_num_samples
        if self.dynamic_lambda_stdev_num_samples > 0:
            assert (
                self.dynamic_lambda_stdev_espilon is not None
                and self.dynamic_lambda_stdev_espilon > 0
            ), "Must specify self.dynamic_lambda_stdev_epsilon if using dynamic lambda"
            assert (
                self.dynamic_lambda_stdev_baseline is not None
                and self.dynamic_lambda_stdev_baseline > 0
            ), "Must specify self.dynamic_lambda_stdev_baseline if using dynamic lambda"

        cfg = _migrate_old_lambdas(cfg)
        self.set_player_type_spec(_compute_player_specs(cfg.player_types, len(self.player_types)))
        if cfg.base_searchbot_cfg.qre.target_pi != "BLUEPRINT":
            assert cfg.base_searchbot_cfg.qre.target_pi == "UNIFORM", cfg.base_searchbot_cfg.qre
            assert all(
                spec.rescore_policy_path is None for spec in self.qre_type2spec.values()
            ), "When target_pi is uniform extra policies are not supported"
        logging.info(
            "Performing bayes quantal response equilibrium hedge with different player types:\n%s",
            "\n".join(
                f"{i}: {spec.name} {spec}" for i, spec in enumerate(self.qre_type2spec.values())
            ),
        )
        self.log_intermediate_iterations = cfg.base_searchbot_cfg.log_intermediate_iterations
        assert self.qre is not None, "base_searchbot_cfg qre needs to be set."
        logging.info(f"Initialized BQRE1P Agent: {self.__dict__}")

    def initialize_state(self, power: Power) -> AgentState:
        state = BayesAgentState(power=power, player_types=self.player_types)
        if self.agent_type_is_public:
            pure_type_belief = np.zeros(self.num_player_types)
            pure_type_belief[self.agent_type] = 1
            state.belief_state.beliefs[power] = pure_type_belief[:]
        return state

    def get_exploited_agent_power(self) -> Optional[Power]:
        assert self.exploited_agent is None
        return None

    def set_player_type_spec(
        self,
        specs: List[PlayerTypeSpec],
        *,
        skip_base_strategy_model_cache=False,
        allow_reuse_model=True,
    ) -> None:
        """(Re-)initializes type configuration from a list of specs."""
        assert len(self.player_types) == len(specs), (len(self.player_types), len(specs))
        base_strategy_model_wrapper_kwargs = dict(
            device=self.base_strategy_model.device,
            max_batch_size=self.base_strategy_model.max_batch_size,
            half_precision=self.base_strategy_model.half_precision,
        )
        self.qre_type2spec = dict(zip(self.player_types, specs))
        self.qre_extra_rescoring_models, old_qre_extra_rescoring_models = (
            {},
            self.qre_extra_rescoring_models,
        )
        for spec in self.qre_type2spec.values():
            spec.assert_valid()
            if (
                spec.rescore_policy_path is not None
                and spec.rescore_policy_path not in self.qre_extra_rescoring_models
            ):
                if (
                    spec.rescore_policy_path in old_qre_extra_rescoring_models
                    and allow_reuse_model
                ):
                    self.qre_extra_rescoring_models[
                        spec.policy_name
                    ] = old_qre_extra_rescoring_models[spec.rescore_policy_path]
                else:
                    base_strategy_model = BaseStrategyModelWrapper(
                        spec.rescore_policy_path, **base_strategy_model_wrapper_kwargs
                    )
                    assert self.parlai_model_orders is None
                    self.qre_extra_rescoring_models[spec.policy_name] = PlausibleOrderSampler(
                        self.order_sampler.cfg, base_strategy_model=base_strategy_model
                    )

    def get_player_type_strings(self) -> List[str]:
        return [self.qre_type2spec[x].name for x in self.player_types]

    def _get_scale_pow_lambdas(self, game: pydipcc.Game) -> Tuple[float, float]:
        scale_lambdas = 1.0
        pow_lambdas = 1.0
        if game.current_year == 1901:
            if self.scale_lambdas_1901 is not None:
                scale_lambdas = self.scale_lambdas_1901
            if self.pow_lambdas_1901 is not None:
                pow_lambdas = self.pow_lambdas_1901
            if game.current_short_phase == "S1901M":
                if self.scale_lambdas_1901_spring is not None:
                    scale_lambdas = self.scale_lambdas_1901_spring
                if self.pow_lambdas_1901_spring is not None:
                    pow_lambdas = self.pow_lambdas_1901_spring
        return (scale_lambdas, pow_lambdas)

    def _get_dynamic_lambda_scale(
        self, game: pydipcc.Game, agent_power: Optional[Power], agent_state: Optional[AgentState]
    ) -> Dict[Power, float]:
        if not self.dynamic_lambda_stdev_num_samples:
            return {power: 1.0 for power in POWERS}

        assert agent_state is not None, "If using dynamic lambda, agent_state should not be None"
        assert self.dynamic_lambda_stdev_espilon is not None
        assert self.dynamic_lambda_stdev_baseline is not None

        phase = game.get_current_phase()
        assert isinstance(agent_state, BayesAgentState)
        if phase not in agent_state.dynamic_lambda_scale_cache:
            # Rollout *through* exactly one set of movement phase orders.
            override_max_rollout_length = 1 if phase.endswith("M") else 2
            # Shape [len(set_orders_dicts), num_powers, num_value_functions].
            rollout_results = self.base_strategy_model_rollouts.do_rollouts_multi(
                game,
                agent_power=agent_power,
                set_orders_dicts=[{}] * self.dynamic_lambda_stdev_num_samples,
                override_max_rollout_length=override_max_rollout_length,
            )
            rollout_results = rollout_results.squeeze(2)
            means_by_power = torch.mean(rollout_results, dim=0)
            variances_by_power = torch.mean(
                torch.square(rollout_results - means_by_power.unsqueeze(0)), dim=0
            )

            means_by_power_list = means_by_power.cpu().tolist()
            variances_by_power_list = variances_by_power.cpu().tolist()
            assert len(means_by_power_list) == len(POWERS)
            assert len(variances_by_power_list) == len(POWERS)

            epsilon = self.dynamic_lambda_stdev_espilon
            stdevs_by_power_list = [
                math.sqrt(variance + epsilon * epsilon) for variance in variances_by_power_list
            ]
            logging.info(
                f"Dynamic lambda: power means: {list(zip(POWERS,['%.3f' % x for x in means_by_power_list]))}"
            )
            logging.info(
                f"Dynamic lambda: power stdevs (after epsilon {epsilon}): {list(zip(POWERS,['%.3f' % x for x in stdevs_by_power_list]))}"
            )

            dynamic_lambda_scale_by_power_list = [
                stdev / self.dynamic_lambda_stdev_baseline for stdev in stdevs_by_power_list
            ]
            dynamic_lambda_scale_by_power = dict(zip(POWERS, dynamic_lambda_scale_by_power_list))
            agent_state.dynamic_lambda_scale_cache[phase] = dynamic_lambda_scale_by_power

        logging.info(
            f"Dynamic lambda final scale: {[(power,'%.3f' % x) for (power,x) in agent_state.dynamic_lambda_scale_cache[phase].items()]}"
        )
        return agent_state.dynamic_lambda_scale_cache[phase]

    def _handle_extra_plausible_and_rescoring(
        self,
        game: pydipcc.Game,
        bp_policy: PowerPolicies,
        extra_plausible_orders: Optional[PlausibleOrders],
        agent_power: Optional[Power],
    ):
        if extra_plausible_orders:
            for pwr, policy in extra_plausible_orders.items():
                for action in policy:
                    if action not in bp_policy[pwr]:
                        logging.info(f"Adding extra plausible orders {pwr}: {action}")
                        bp_policy[pwr][action] = 0.0

            bp_policy = self.order_sampler.rescore_actions(
                game, has_press=self.has_press, agent_power=agent_power, input_policy=bp_policy
            )

        biasing_policies = {PlayerTypeSpec.BLUEPRINT_POLICY_NAME: bp_policy}
        for name, sampler in self.qre_extra_rescoring_models.items():
            logging.info("Querying %s", name)
            biasing_policies[name] = sampler.rescore_actions(
                game, has_press=self.has_press, agent_power=agent_power, input_policy=bp_policy
            )
        return bp_policy, biasing_policies

    def run_search(
        self,
        game: pydipcc.Game,
        *,
        bp_policy: Optional[PowerPolicies] = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
        rollout_results_cache: Optional[RolloutResultsCache] = None,
        pre_rescored_bp: Optional[PowerPolicies] = None,
    ) -> BRMResult:
        """Same as get_all_power_prob_distributions but also returns the stats about the bcfr"""
        # If there are no locations to order, bail
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            if agent_power is not None:
                assert early_exit_for_power == agent_power
            return self._early_quit_bcfr_result(
                power=early_exit_for_power, agent_state=agent_state
            )

        if timings is None:
            timings = TimingCtx()
        timings.start("one-time")

        deadline: Optional[float] = (
            time.monotonic() + self.max_seconds if self.max_seconds > 0 else None
        )

        if agent_state is not None:
            assert isinstance(agent_state, BayesAgentState)
            belief_state = agent_state.belief_state
        else:
            belief_state = BeliefState(self.player_types)

        logging.info(f"BEGINNING BQRE run_search, agent_power={agent_power}")
        if rollout_results_cache is not None:
            maybe_rollout_results_cache = rollout_results_cache
        else:
            maybe_rollout_results_cache = (
                self.base_strategy_model_rollouts.build_cache()
                if self.cache_rollout_results
                else None
            )

        # Compute blueprint policy for each of the different types
        with timings.create_subcontext() as sub_timings, sub_timings("get_plausible_order"):
            if bp_policy is None:
                bp_policy = self.get_plausible_orders_policy(
                    game, agent_power=agent_power, agent_state=agent_state
                )

            bp_policy, biasing_policies = self._handle_extra_plausible_and_rescoring(
                game, bp_policy, extra_plausible_orders, agent_power
            )

        scale_lambdas, pow_lambdas = self._get_scale_pow_lambdas(game)
        scale_lambdas_by_power = self._get_dynamic_lambda_scale(game, agent_power, agent_state)

        bqre_data = _make_bqre_data(
            biasing_policies,
            specs=self.qre_type2spec,
            qre=self.qre,
            pow_lambdas=pow_lambdas,
            scale_lambdas=scale_lambdas,
            scale_lambdas_by_power=scale_lambdas_by_power,
            agent_power=agent_power,
        )

        if agent_power is not None:
            bilateral_stats = BilateralStats(
                game, agent_power, bqre_data.get_best_type_data().power_plausible_orders
            )
        else:
            bilateral_stats = None
        power_is_loser = {}  # make typechecker happy
        last_search_iter = False
        for bqre_iter in range(self.n_rollouts):
            if last_search_iter:
                logging.info(f"Early exit from BCFR after {bqre_iter} iterations by timeout")
                break
            elif deadline is not None and time.monotonic() >= deadline:
                last_search_iter = True
            timings.start("start")
            # do verbose logging on 2^x iters
            verbose_log_iter = self.is_verbose_log_iter(bqre_iter) or last_search_iter

            # Sample player types
            ptypes = {
                pwr: np.random.choice(self.player_types, 1, p=belief_state.beliefs[pwr])[
                    0
                ]  # type:ignore
                for pwr in POWERS
            }

            timings.start("query_policy")

            # check if the *best* type is a loser
            power_is_loser = self.get_power_loser_dict(bqre_data.get_best_type_data(), bqre_iter)

            ptype_power_action_ps = {
                ptype: self.get_cur_iter_strategies(cfr_data, bqre_iter)
                for ptype, cfr_data in bqre_data.type_cfr_data.items()
            }

            power_action_ps = {
                pwr: ptype_power_action_ps[ptype][pwr] for pwr, ptype in ptypes.items()
            }

            timings.start("apply_orders")
            # sample policy for all powers

            plausible_orders = {
                pwr: bqre_data.get_type_data(ptype).power_plausible_orders[pwr]
                for pwr, ptype in ptypes.items()
            }

            idxs, power_sampled_orders = sample_orders_from_policy(
                plausible_orders, power_action_ps
            )
            if bilateral_stats is not None:
                bilateral_stats.accum_bilateral_probs(power_sampled_orders, weight=bqre_iter)
            set_orders_dicts = make_set_orders_dicts(plausible_orders, power_sampled_orders)

            timings.stop()

            all_set_orders_dicts = list(set_orders_dicts)
            with timings.create_subcontext("rollout") as inner_timings, inner_timings("rollout"):
                all_rollout_results_tensor = self.base_strategy_model_rollouts.do_rollouts_multi_maybe_cached(
                    game,
                    agent_power=agent_power,
                    set_orders_dicts=set_orders_dicts,
                    cache=maybe_rollout_results_cache,
                    timings=inner_timings,
                )

            timings.start("bcfr")

            for pwr, actions in plausible_orders.items():
                # pop this power and player type's results
                scores, all_rollout_results_tensor = (
                    all_rollout_results_tensor[: len(actions)],
                    all_rollout_results_tensor[len(actions) :],
                )
                pwr_set_order_dicts, all_set_orders_dicts = (
                    all_set_orders_dicts[: len(actions)],
                    all_set_orders_dicts[len(actions) :],
                )
                # Results for the first value function as RolloutResults.
                default_results: RolloutResults = [
                    (orders, dict(zip(POWERS, scores[..., 0].tolist())))
                    for orders, scores in zip(pwr_set_order_dicts, scores)
                ]

                if bilateral_stats is not None:
                    bilateral_stats.accum_bilateral_values(pwr, bqre_iter, default_results)

                # logging.info(f"Results {pwr} = {results}")
                # calculate regrets
                # Shape: [len(pwr_set_order_dicts), num value functions].
                all_action_utilities = scores[:, POWERS.index(pwr)]

                for cur_ptype in self.player_types:
                    spec = self.qre_type2spec[cur_ptype]
                    action_utilities = all_action_utilities[..., 0].tolist()

                    cfr_data = bqre_data.get_type_data(cur_ptype)
                    state_utility: float = np.dot(
                        ptype_power_action_ps[cur_ptype][pwr], action_utilities
                    )  # type: ignore

                    # update bcfr data structures for particular player type and power
                    cfr_data.update(
                        pwr=pwr,
                        actions=actions,
                        state_utility=state_utility,
                        action_utilities=action_utilities,
                        which_strategy_to_accumulate=pydipcc.CFRStats.ACCUMULATE_PREV_ITER,
                        cfr_iter=bqre_iter,
                    )

                    # log some action values
                    if verbose_log_iter and cur_ptype == self.player_types[0] and actions[0] != ():
                        self.log_bcfr_iter_state(
                            game=game,
                            pwr=pwr,
                            actions=actions,
                            brm_data=bqre_data,
                            brm_iter=bqre_iter,
                            state_utility=state_utility,
                            action_utilities=action_utilities,
                            power_sampled_orders=power_sampled_orders,
                            beliefs=belief_state.beliefs[pwr],
                            pre_rescore_bp=pre_rescored_bp[pwr] if pre_rescored_bp else None,
                        )
                        if maybe_rollout_results_cache is not None:
                            logging.info(f"{maybe_rollout_results_cache}")

            assert (
                all_rollout_results_tensor.shape[0] == 0
            ), "all_rollout_results_tensor should be empty"
        timings.start("to_dict")

        # return prob. distributions for each power, player type
        ptype_avg_ret, ptype_final_ret = {}, {}
        power_is_loser = self.get_power_loser_dict(bqre_data.get_best_type_data(), self.n_rollouts)
        for player_type in self.player_types:
            avg_ret, final_ret = {}, {}
            cfr_data = bqre_data.get_type_data(player_type)
            for p in POWERS:
                if power_is_loser[p] or p == self.exploited_agent_power:
                    avg_ret[p] = final_ret[p] = cfr_data.bp_policy(p)
                else:
                    avg_ret[p] = cfr_data.avg_policy(p)
                    final_ret[p] = cfr_data.cur_iter_policy(p)

            ptype_avg_ret[player_type] = avg_ret
            ptype_final_ret[player_type] = final_ret

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
        logging.info(
            "BQRE Values (Agent Type: %s): %s",
            self.agent_type,
            {p: f"{bqre_data.get_type_data(self.agent_type).avg_utility(p):.3f}" for p in POWERS},
        )

        timings.stop()
        timings.pprint(logging.getLogger("timings").info)

        if bilateral_stats is not None:
            cfr_data = bqre_data.get_best_type_data()
            if self.log_bilateral_values:
                bilateral_stats.log(cfr_data, min_order_prob=self.bilateral_cfg.min_order_prob)

        # Update bcfr result cache
        bcfr_result = BRMResult(
            bp_policies=bp_policy,
            ptype_avg_policies=ptype_avg_ret,
            ptype_final_policies=ptype_final_ret,
            brm_data=bqre_data,
            beliefs=belief_state.beliefs,
            agent_type=self.agent_type,
            use_final_iter=self.use_final_iter,
            bilateral_stats=bilateral_stats,
        )

        return bcfr_result

    def run_bilateral_search_with_conditional_evs(
        self,
        game: pydipcc.Game,
        *,
        bp_policy: PowerPolicies,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Power,
        other_power: Power,
        agent_state: Optional[AgentState],
        conditional_evs: BilateralConditionalValueTable,
        pre_rescored_bp: Optional[PowerPolicies] = None,
    ) -> BRMResult:
        """Same as run_search but with all conditional evs precomputed, and specialized to 2 players.

        This function is mostly a copypasta of run_search for performance, where a lot of options
        have been removed and the specialization to 2 players allows computing things much faster.
        """
        # If there are no locations to order, bail
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            assert early_exit_for_power == agent_power
            return self._early_quit_bcfr_result(
                power=early_exit_for_power, agent_state=agent_state
            )

        if timings is None:
            timings = TimingCtx()
        timings.start("rswce_one-time")

        assert self.loser_bp_value <= 0
        assert self.exploited_agent_power is None

        if agent_state is not None:
            assert isinstance(agent_state, BayesAgentState)
            belief_state = agent_state.belief_state
        else:
            belief_state = BeliefState(self.player_types)

        logging.info(f"BEGINNING BQRE run_search_with_conditional_evs, agent_power={agent_power}")

        # Compute blueprint policy for each of the different types
        with timings.create_subcontext() as sub_timings, sub_timings("rswce_get_plausible_order"):
            bp_policy, biasing_policies = self._handle_extra_plausible_and_rescoring(
                game, bp_policy, extra_plausible_orders, agent_power
            )

        scale_lambdas, pow_lambdas = self._get_scale_pow_lambdas(game)
        scale_lambdas_by_power = self._get_dynamic_lambda_scale(game, agent_power, agent_state)

        bqre_data = _make_bqre_data(
            biasing_policies,
            specs=self.qre_type2spec,
            qre=self.qre,
            pow_lambdas=pow_lambdas,
            scale_lambdas=scale_lambdas,
            scale_lambdas_by_power=scale_lambdas_by_power,
            agent_power=agent_power,
        )

        for bqre_iter in range(self.n_rollouts):
            timings.start("rswce_start")
            # do verbose logging on 2^x iters
            verbose_log_iter = self.is_verbose_log_iter(bqre_iter)

            # Sample player types
            # Written as an iteration over POWERS specifically to preserve the ordering
            # of the items within the dictionary to be the same order as POWERS, which
            # affects the ordering of many downstream dictionaries and the order in which
            # our logs will record things.
            ptypes = {
                pwr: np.random.choice(self.player_types, 1, p=belief_state.beliefs[pwr])[
                    0
                ]  # type:ignore
                for pwr in POWERS
                if pwr in (agent_power, other_power)
            }

            timings.start("rswce_query_policy")

            ptype_power_action_ps = {
                ptype: self.get_cur_iter_strategies(cfr_data, bqre_iter)
                for ptype, cfr_data in bqre_data.type_cfr_data.items()
            }

            power_action_ps = {
                pwr: ptype_power_action_ps[ptype][pwr] for pwr, ptype in ptypes.items()
            }

            timings.start("rswce_apply_orders")
            # sample policy for all powers

            plausible_orders = {
                pwr: bqre_data.get_type_data(ptype).power_plausible_orders[pwr]
                for pwr, ptype in ptypes.items()
            }

            _idxs, power_sampled_orders = sample_orders_from_policy(
                plausible_orders, power_action_ps
            )

            timings.stop()

            timings.start("rswce_bcfr")

            for pwr, actions in plausible_orders.items():
                if pwr == agent_power:
                    other_sampled_action = power_sampled_orders[other_power]
                    scores_list = [
                        conditional_evs[(agent_action, other_sampled_action)]
                        for agent_action in actions
                    ]
                else:
                    agent_sampled_action = power_sampled_orders[agent_power]
                    scores_list = [
                        conditional_evs[(agent_sampled_action, other_action)]
                        for other_action in actions
                    ]

                # scores_list is a list of tensors of shape (7,), indicating the value estimate for all 7 players
                # conditional on each plausible action of pwr and the sampled action of the other power besides pwr.
                # Stack into an (N,7) tensor.
                scores = torch.stack(scores_list, 0)

                # logging.info(f"Results {pwr} = {results}")
                # calculate regrets
                # Shape: [len(pwr_set_order_dicts), num value functions].
                all_action_utilities = scores[:, POWERS.index(pwr)]

                for cur_ptype in self.player_types:
                    spec = self.qre_type2spec[cur_ptype]
                    action_utilities = all_action_utilities[..., 0].tolist()

                    cfr_data = bqre_data.get_type_data(cur_ptype)
                    state_utility: float = np.dot(
                        ptype_power_action_ps[cur_ptype][pwr], action_utilities
                    )  # type: ignore

                    # update bcfr data structures for particular player type and power
                    cfr_data.update(
                        pwr=pwr,
                        actions=actions,
                        state_utility=state_utility,
                        action_utilities=action_utilities,
                        which_strategy_to_accumulate=pydipcc.CFRStats.ACCUMULATE_PREV_ITER,
                        cfr_iter=bqre_iter,
                    )

                    # log some action values
                    if verbose_log_iter and cur_ptype == self.player_types[0] and actions[0] != ():
                        self.log_bcfr_iter_state(
                            game=game,
                            pwr=pwr,
                            actions=actions,
                            brm_data=bqre_data,
                            brm_iter=bqre_iter,
                            state_utility=state_utility,
                            action_utilities=action_utilities,
                            power_sampled_orders=power_sampled_orders,
                            beliefs=belief_state.beliefs[pwr],
                            pre_rescore_bp=pre_rescored_bp[pwr] if pre_rescored_bp else None,
                        )

        timings.start("rswce_to_dict")

        # return prob. distributions for each power, player type
        ptype_avg_ret, ptype_final_ret = {}, {}
        for player_type in self.player_types:
            avg_ret, final_ret = {}, {}
            cfr_data = bqre_data.get_type_data(player_type)
            for p in POWERS:
                avg_ret[p] = cfr_data.avg_policy(p)
                final_ret[p] = cfr_data.cur_iter_policy(p)

            ptype_avg_ret[player_type] = avg_ret
            ptype_final_ret[player_type] = final_ret

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
        logging.info(
            "BQRE Values (Agent Type: %s): %s",
            self.agent_type,
            {p: f"{bqre_data.get_type_data(self.agent_type).avg_utility(p):.3f}" for p in POWERS},
        )

        timings.stop()

        # Update bcfr result cache
        bcfr_result = BRMResult(
            bp_policies=bp_policy,
            ptype_avg_policies=ptype_avg_ret,
            ptype_final_policies=ptype_final_ret,
            brm_data=bqre_data,
            beliefs=belief_state.beliefs,
            agent_type=self.agent_type,
            use_final_iter=self.use_final_iter,
            bilateral_stats=None,
        )

        return bcfr_result

    def run_best_response_against_correlated_bilateral_search(
        self,
        game: pydipcc.Game,
        *,
        bp_policy: Optional[PowerPolicies] = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Power,
        agent_state: Optional[AgentState] = None,
    ) -> SearchResult:
        assert agent_power is not None
        assert self.all_power_base_strategy_model_executor is not None

        # If there are no locations to order, bail
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            if agent_power is not None:
                assert early_exit_for_power == agent_power
            return self._early_quit_bcfr_result(
                power=early_exit_for_power, agent_state=agent_state
            )

        if timings is None:
            timings = TimingCtx()

        if bp_policy is None:
            timings.start("corr_search_allpower_bp")
            # this is the bp considering our (agent_power's) conversation with every other power
            bp_policy = self.get_plausible_orders_policy(
                game, agent_power=agent_power, agent_state=agent_state
            )
        bp_policy = filter_invalid_actions_from_policy(bp_policy, game)
        # 1) get pairwise bqre policies for all our opponents
        bqre_policies: PowerPolicies = {}
        timings.start("corr_search_rescore_bp")
        if self.order_sampler.do_parlai_rescoring:
            opponent_rescored_policies = rescore_bp_from_bilateral_views(
                game, bp_policy, agent_power, self.order_sampler
            )
        else:
            # bp was not rescored by parlai model, i.e. no-press game
            opponent_rescored_policies = {
                power: bp_policy for power in bp_policy if power != agent_power
            }

        timings.start("corr_search_payoff_matrix")
        value_table_cache = None
        if agent_state is not None:
            assert isinstance(agent_state, SearchBotAgentState)
            value_table_cache = agent_state.get_cached_value_tables(game)

        # power_value_matrices[pwr] stores the values of the joint actions between agent_power and pwr
        # each value matrix is a map from frozenset((pwr, action), (agent_pwr, action)) -> Tensor [7, 1]
        power_value_matrices: Dict[
            Power, BilateralConditionalValueTable
        ] = compute_payoff_matrix_for_all_opponents(
            game,
            all_power_base_strategy_model=self.all_power_base_strategy_model_executor,
            bp_policy=bp_policy,
            agent_power=agent_power,
            num_sample=self.br_corr_bilateral_search_cfg.bilateral_search_num_cond_sample,
            has_press=self.has_press,
            player_rating=self.player_rating,
            value_table_cache=value_table_cache,
        )

        timings.start("corr_search_pairwise_cfr")
        for opponent, policy in bp_policy.items():
            if opponent == agent_power:
                continue
            if len(policy) == 1 and list(policy.keys())[0] == ():
                continue

            logging.info(f"BQRE1P.run_search between {agent_power}, {opponent}")
            if timings is not None:
                timings.start("corr_search_run_search")
            rescored_pair_policy = extract_bp_policy_for_powers(
                opponent_rescored_policies[opponent], [agent_power, opponent],
            )
            search_result = self.run_bilateral_search_with_conditional_evs(
                game,
                bp_policy=rescored_pair_policy,
                early_exit_for_power=early_exit_for_power,
                extra_plausible_orders=extra_plausible_orders,
                agent_power=agent_power,
                other_power=opponent,
                agent_state=agent_state,
                conditional_evs=power_value_matrices[opponent],
                pre_rescored_bp=bp_policy,
            )
            bqre_policies[opponent] = search_result.get_population_policy(opponent)[opponent]

        result = BRCorrBilateralSearchResult(
            agent_power, bp_policy, bqre_policies, power_value_matrices
        )
        if len(bp_policy[agent_power]) == 1:
            [action] = list(bp_policy[agent_power])
            result.set_policy_and_value_for_power(agent_power, action, best_value=0)
            return result

        timings.start("corr_search_br_sample")
        # 2) now we have both bp_policy and others uncorr_policy
        # next we sample k(=10 default) joint actions to use as candidates for reweighting
        opponent_joint_actions = sample_joint_actions(
            result.policies, self.br_corr_bilateral_search_cfg.br_num_sample
        )

        timings.start("corr_search_br_weight")
        base_strategy_model_rescored_bp_policy = self.order_sampler.rescore_actions_base_strategy_model(
            game,
            has_press=self.has_press,
            agent_power=None,  # agent_power,
            input_policy=bp_policy,
            model=self.base_strategy_model.model,
        )

        if self.br_corr_bilateral_search_cfg.use_all_power_for_p_joint:
            joint_action_weights = compute_weights_for_opponent_joint_actions(
                opponent_joint_actions,
                agent_power,
                game,
                self.all_power_base_strategy_model_executor.get_model().model,
                base_strategy_model_rescored_bp_policy,
                self.has_press,
                self.br_corr_bilateral_search_cfg.min_unnormalized_weight,
                self.br_corr_bilateral_search_cfg.max_unnormalized_weight,
            )
        else:
            assert self.br_corr_bilateral_search_cfg.joint_action_min_prob == 0
            joint_action_weights = [
                1 / len(opponent_joint_actions) for _ in opponent_joint_actions
            ]

        with timings("interruption"):
            raise_if_should_stop(post_pseudoorders=False)

        # 3) compute weighted best response
        br_regularize_lambda = self.br_corr_bilateral_search_cfg.br_regularize_lambda
        scale_lambdas, pow_lambdas = self._get_scale_pow_lambdas(game)
        scale_lambdas_by_power = self._get_dynamic_lambda_scale(game, agent_power, agent_state)

        logging.info(
            f"Config br_regularize_lambda {br_regularize_lambda}, scale_lambdas {scale_lambdas}, pow_lambdas {pow_lambdas}"
        )
        br_regularize_lambda = (
            scale_lambdas_by_power[agent_power]
            * scale_lambdas
            * (br_regularize_lambda ** pow_lambdas)
        )

        timings.start("corr_search_br_get_action")
        best_action, best_value = compute_best_action_against_reweighted_opponent_joint_actions(
            game,
            agent_power,
            bp_policy[agent_power],
            opponent_joint_actions,
            joint_action_weights,
            self.all_power_base_strategy_model_executor,
            self.player_rating,
            br_regularize_lambda,
        )
        result.set_policy_and_value_for_power(agent_power, best_action, best_value)

        timings.stop()
        timings.pprint(logging.getLogger("timings").info)
        return result

    def log_bcfr_iter_state(
        self,
        *,
        game: pydipcc.Game,
        pwr: Power,
        actions: List[Action],
        brm_data: BRMData,
        brm_iter: int,
        state_utility: float,
        action_utilities: List[float],
        power_sampled_orders: JointAction,
        beliefs: np.ndarray,
        pre_rescore_bp: Optional[Policy] = None,
    ):
        agent_cfr_data = brm_data.get_type_data(self.agent_type)
        logging.info(
            f"<> [ {brm_iter+1} / {self.n_rollouts} ] {pwr} {game.phase} avg_utility={agent_cfr_data.avg_utility(pwr):.5f} cur_utility={state_utility:.5f} "
        )
        logging.info(f">> {pwr} cur action at {brm_iter+1}: {power_sampled_orders[pwr]}")
        if pre_rescore_bp is None:
            logging.info(
                f"     {'agent_p':8s}  {'pop_p':8s}  {'bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders"
            )
        else:
            logging.info(
                f"     {'agent_p':8s}  {'pop_p':8s}  {'bp_p':8s}  {'pre_bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders"
            )
        agent_type_action_probs: List[float] = agent_cfr_data.avg_strategy(pwr)
        population_action_probs = np.zeros(len(agent_type_action_probs))
        for player_type in self.player_types:
            population_action_probs += beliefs[player_type] * np.array(
                brm_data.get_type_data(player_type).avg_strategy(pwr)
            )
        bp_probs: List[float] = agent_cfr_data.bp_strategy(pwr)
        avg_utilities: List[float] = agent_cfr_data.avg_action_utilities(pwr)
        sorted_metrics = sorted(
            zip(
                actions,
                agent_type_action_probs,
                population_action_probs,
                bp_probs,
                avg_utilities,
                action_utilities,
            ),
            key=lambda ac: -ac[1],
        )
        for orders, a_p, p_p, bp_p, avg_u, cur_u in sorted_metrics:
            if pre_rescore_bp is None:
                logging.info(
                    f"|>  {a_p:8.5f}  {p_p:8.5f}  {bp_p:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}"
                )
            else:
                logging.info(
                    f"|>  {a_p:8.5f}  {p_p:8.5f}  {bp_p:8.5f}  {pre_rescore_bp[orders]:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}"
                )

    def _early_quit_bcfr_result(
        self, power: Power, agent_state: Optional[AgentState], *, action: Action = tuple()
    ) -> BRMResult:
        ptype_policies = {ptype: {power: {action: 1.0}} for ptype in self.player_types}
        if agent_state is not None:
            assert isinstance(agent_state, BayesAgentState)
            belief_state = agent_state.belief_state
        else:
            belief_state = BeliefState(self.player_types)
        return BRMResult(
            bp_policies=ptype_policies[self.player_types[0]],
            ptype_avg_policies=ptype_policies,
            ptype_final_policies=ptype_policies,
            brm_data=None,
            beliefs=belief_state.beliefs,
            agent_type=self.agent_type,
            use_final_iter=self.use_final_iter,
        )


if __name__ == "__main__":
    import pathlib
    import heyhi
    import sys

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    np.random.seed(0)  # type:ignore
    torch.manual_seed(0)

    game = pydipcc.Game()
    cfg = heyhi.load_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "conf/common/agents/bqre1p_20210723.prototxt",
        overrides=["bqre1p.base_searchbot_cfg.n_rollouts=64"] + sys.argv[1:],
    )
    print(cfg.bqre1p)
    agent = BQRE1PAgent(cfg.bqre1p)
    print(agent.get_orders(game, power="AUSTRIA", state=agent.initialize_state(power="AUSTRIA")))
