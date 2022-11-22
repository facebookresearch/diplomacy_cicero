#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import itertools
import random
from typing import Dict, List, Optional, Set, Tuple, Union
import collections
import logging
import math

import numpy as np
import scipy.special
import torch

from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import AgentState, NoAgentState
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    SearchResult,
)
from fairdiplomacy.agents.base_strategy_model_rollouts import (
    BaseStrategyModelRollouts,
    RolloutResultsCache,
)
from fairdiplomacy.agents.base_strategy_model_wrapper import (
    BaseStrategyModelWrapper,
    compute_action_logprobs_from_state,
    create_conditional_teacher_force_orders,
)
from fairdiplomacy.agents.plausible_order_sampling import (
    PlausibleOrderSampler,
    cutoff_policy,
    renormalize_policy,
)
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.models.base_strategy_model.base_strategy_model import BaseStrategyModelV2
from fairdiplomacy.typedefs import (
    Action,
    JointAction,
    JointPolicy,
    PlayerRating,
    Power,
    PowerPolicies,
)
from fairdiplomacy.utils.parse_device import device_id_to_str
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.zipn import unzip2, zip2


_JoinedPolicyData = collections.namedtuple("_JoinedPolicyData", "power_action_dicts,probs")


class TheBestResult(SearchResult):
    def __init__(
        self,
        power_action_dicts: List[Dict[Power, Action]],
        joint_probs: np.ndarray,
        is_early_exit: bool = False,
        values: Optional[np.ndarray] = None,
    ):
        self.power_action_dicts = power_action_dicts
        self.joint_probs = joint_probs
        self._is_early_exit = is_early_exit
        self.values = values

        # Compute marginal probabilities
        avg_policies = {power: collections.defaultdict(float) for power in POWERS}
        for action_dict, prob in zip(power_action_dicts, joint_probs):
            for power, action in action_dict.items():
                avg_policies[power][action] += prob
        self.avg_policies = {power: dict(policy) for power, policy in avg_policies.items()}

    @property
    def joint_policy_size(self) -> int:
        return len(self.power_action_dicts)

    def get_agent_policy(self) -> PowerPolicies:
        return self.avg_policies

    def get_population_policy(self) -> PowerPolicies:
        return self.get_agent_policy()

    def get_bp_policy(self) -> PowerPolicies:
        raise RuntimeError("TheBestResult.get_bp_policy is not supported")

    def sample_action(self, power) -> Action:
        return self.sample_joint_action()[power]

    def sample_joint_action(self) -> JointAction:
        return self.power_action_dicts[np.random.choice(len(self.joint_probs), p=self.joint_probs)]

    def get_joint_policy(self, max_actions: int) -> JointPolicy:
        return [(self.sample_joint_action(), 1.0 / max_actions) for _ in range(max_actions)]

    def avg_utility(self, pwr: Power) -> float:
        assert self.values is not None
        return self.values[POWERS.index(pwr)]

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        raise RuntimeError("Not implemented. Do you even mean here?")

    def is_early_exit(self) -> bool:
        return self._is_early_exit


# def _early_quit_search_result(power: Power, *, action: Action = tuple()) -> TheBestResult:
#     policies = {power: {action: 1.0}}
#     return TheBestResult(
#         bp_policies=policies,
#         avg_policies=policies,
#         final_policies=policies,
#         cfr_data=None,
#         use_final_iter=False,
#     )


class TheBestAgent(BaseSearchAgent):
    """One-ply cfr with base_strategy_model-policy rollouts"""

    def __init__(self, cfg: agents_cfgs.TheBestAgent, *, skip_base_strategy_model_cache=False):
        super().__init__(cfg)
        self.has_press = True
        self.cfg = cfg
        self.eta = cfg.qre_eta
        self.compute_inside_ratio = cfg.compute_inside_ratio

        qre_lambda = self.cfg.qre_lambda
        assert qre_lambda is not None
        assert 0 < qre_lambda, qre_lambda
        self.qre_lambda = qre_lambda  # Extra step to fix typing
        # self.set_player_rating = cfg.set_player_rating

        num_br_samples = self.cfg.num_br_samples
        assert num_br_samples is not None, "Must be set"
        self.num_br_samples = num_br_samples
        self.num_importance_samples = self.cfg.num_importance_samples or (self.num_br_samples * 10)

        if self.num_importance_samples > self.cfg.num_value_computation_samples:
            logging.warning(
                "The max of the policy (%d) is bigger than the number of samples allowed"
                " to compute value function (%d). Therefore, value approximation will be used.",
                self.num_importance_samples,
                self.cfg.num_value_computation_samples,
            )
        else:
            logging.info("Will use exact EV computation over the joint policy")

        # Load base_strategy_model models.
        base_strategy_model_wrapper_kwargs = dict(
            device=device_id_to_str(cfg.device),
            max_batch_size=cfg.max_batch_size,
            half_precision=cfg.half_precision,
            skip_base_strategy_model_cache=skip_base_strategy_model_cache,
        )
        self.conditional_max_batch_size: int = cfg.conditional_max_batch_size or cfg.max_batch_size
        assert self.conditional_max_batch_size is not None
        assert cfg.value_model_path
        self.rollout_base_strategy_model = BaseStrategyModelWrapper(
            cfg.conditional_policy_model_path,
            value_model_path=cfg.value_model_path,
            **base_strategy_model_wrapper_kwargs,
        )
        self.anchor_base_strategy_model = BaseStrategyModelWrapper(
            cfg.anchor_joint_policy_model_path or cfg.conditional_policy_model_path,
            **base_strategy_model_wrapper_kwargs,
        )
        assert self.rollout_base_strategy_model.is_all_powers()
        if cfg.plausible_model_path:
            self.proposal_base_strategy_model = BaseStrategyModelWrapper(
                cfg.plausible_model_path,
                force_disable_all_power=True,  # To allow anypower models.
                **base_strategy_model_wrapper_kwargs,
            )
        else:
            self.proposal_base_strategy_model = BaseStrategyModelWrapper(
                cfg.conditional_policy_model_path,
                force_disable_all_power=True,  # To allow anypower models.
                **base_strategy_model_wrapper_kwargs,
            )
            # Hack. We re-use the model from self.rollout_base_strategy_model so that it's
            # possible to update models in both wrappers during RL training.
            self.proposal_base_strategy_model.model = self.rollout_base_strategy_model.model

        # Construct things on top of base_strategy_model models
        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg,
            base_strategy_model=self.proposal_base_strategy_model,
            parlai_model=None,
        )
        assert cfg.rollouts_cfg.max_rollout_length == 0, "Cannot do rollouts in TheBestAgent"
        self.base_strategy_model_rollouts = BaseStrategyModelRollouts(
            self.rollout_base_strategy_model,
            cfg.rollouts_cfg,
            has_press=self.has_press,
            # set_player_ratings=self.set_player_rating,
        )

        logging.info(f"Initialized TheBestAgent Agent: {self.__dict__}")

    @property
    def n_plausible_orders(self) -> int:
        # Accessor for RL
        return self.order_sampler.n_plausible_orders

    @property
    def base_strategy_model(self) -> BaseStrategyModelWrapper:
        # Accessor for RL
        return self.rollout_base_strategy_model

    def override_has_press(self, has_press: bool) -> None:
        # For RL
        self.has_press = has_press

    def get_exploited_agent_power(self) -> Optional[Power]:
        # For RL
        return None

    def can_share_strategy(self) -> bool:
        return True

    # Overrides BaseAgent
    def get_orders(self, game: Game, power: Power, state: AgentState) -> Action:
        del state  # Not used
        return self.get_orders_many_powers(game, [power])[power]

    # Overrides BaseAgent
    def get_orders_many_powers(self, game: Game, powers: List[Power]) -> JointAction:
        search_result = self.run_search(game, agent_state=None,)
        joint_action = search_result.sample_joint_action()
        return {power: joint_action.get(power, tuple()) for power in powers}

    # Overrides BaseSearchAgent
    def get_plausible_orders_policy(
        self,
        game: Game,
        *,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
        player_rating: Optional[PlayerRating] = None,
    ) -> PowerPolicies:
        policy = self.order_sampler.sample_orders(
            game,
            agent_power=agent_power,
            player_rating=player_rating,
            force_base_strategy_model_has_press=True,
        )
        return policy

    def _compute_power_action_utilities(
        self,
        *,
        game: Game,
        bp_policy: PowerPolicies,
        timings: TimingCtx,
        maybe_rollout_results_cache: Optional[RolloutResultsCache] = None,
    ) -> Dict[Power, Dict[Action, float]]:
        """Compute utilities for (power, action) pairs from PowerPolicies."""
        power_evs: Dict[Power, Dict[Action, float]] = {}

        if maybe_rollout_results_cache is None:
            maybe_rollout_results_cache = self.base_strategy_model_rollouts.build_cache()

        def dict_to_device(x: DataFields) -> DataFields:
            if self.rollout_base_strategy_model.half_precision:
                x = x.to_half_precision()
            return DataFields(
                {k: v.to(self.rollout_base_strategy_model.device) for k, v in x.items()}
            )

        timings.start("compute_feats")

        feature_encoder = self.rollout_base_strategy_model.feature_encoder
        assert self.rollout_base_strategy_model.is_all_powers()
        features = dict_to_device(
            DataFields(
                feature_encoder.encode_inputs_all_powers(
                    [game], self.rollout_base_strategy_model.get_policy_input_version()
                )
            )
        )

        rollout_base_strategy_model_model = self.rollout_base_strategy_model.model
        assert isinstance(rollout_base_strategy_model_model, BaseStrategyModelV2), type(
            rollout_base_strategy_model_model
        )
        encoder_features = DataFields(
            **{
                k: v
                for k, v in features.items()
                if k not in {"x_possible_actions", "x_loc_idxs", "x_power"}
            }
        )
        encoder_features = self.rollout_base_strategy_model.add_stuff_to_datafields(
            encoder_features, has_press=True, agent_power=None, game_rating_dict=None
        )
        encoder_features = dict_to_device(encoder_features)

        # For each action we condition on, we sample num_br_samples sample.
        # Hence we measure the batch size in the number of output sequences
        # rather than the number of input sequences.
        inputs_per_batch_size = self.conditional_max_batch_size // self.num_br_samples + 1
        batch_repeat_interleave = self.num_br_samples
        flat_plausible_actions = [
            power_action
            for marginal_human_policy in bp_policy.values()
            for power_action in list(marginal_human_policy)
        ]

        # Actual sampling starts here.
        all_sampled_actions = []
        for power_action_batch in groupby(flat_plausible_actions, inputs_per_batch_size):
            timings.start("s.feats")
            x_current_orders = []
            teacher_force_orders = []
            for power_action in power_action_batch:
                x_current_orders.append(
                    feature_encoder.encode_orders_single_tolerant(
                        game,
                        power_action,
                        self.rollout_base_strategy_model.get_policy_input_version(),
                    ).unsqueeze(0)
                )
                teacher_force_orders.append(
                    create_conditional_teacher_force_orders(
                        DataFields(x_current_orders=x_current_orders[-1], **features)
                    )
                )
            with torch.no_grad():
                conditional_encoder_features = DataFields(
                    **{
                        k: v.repeat_interleave(len(x_current_orders), dim=0)
                        for k, v in encoder_features.items()
                    },
                    x_current_orders=torch.cat(x_current_orders, 0),
                )
                encoded = rollout_base_strategy_model_model.encode_state(
                    **dict_to_device(conditional_encoder_features)
                )
            batch = DataFields()
            batch["encoded"] = encoded
            batch["teacher_force_orders"] = torch.cat(teacher_force_orders, 0).repeat_interleave(
                batch_repeat_interleave, dim=0
            )
            # Input features for decoder.
            batch.update(
                {
                    k: features[k].repeat_interleave(len(x_current_orders), dim=0)
                    for k in ["x_loc_idxs", "x_possible_actions", "x_power", "x_in_adj_phase"]
                }
            )
            # We don't need it as we use encoded so passing some dummy tensor.
            dummy_tensor = torch.zeros((len(x_current_orders), 1))
            batch.update(
                {
                    k: dummy_tensor
                    for k in [
                        "x_board_state",
                        "x_prev_state",
                        "x_prev_orders",
                        "x_season",
                        "x_year_encoded",
                        "x_build_numbers",
                    ]
                }
            )
            batch = self.rollout_base_strategy_model.add_stuff_to_datafields(
                batch, has_press=True, agent_power=None, game_rating_dict=None
            )
            batch = dict_to_device(batch)

            timings.stop()
            with timings.create_subcontext("s") as subtimings:
                (
                    chunk_sampled_actions,
                    _,
                ) = self.rollout_base_strategy_model.forward_policy_from_datafields(
                    batch,
                    temperature=1.0,
                    top_p=1.0,
                    batch_repeat_interleave=batch_repeat_interleave,
                    timings=subtimings,
                )
                all_sampled_actions.extend(chunk_sampled_actions)
                del batch

        timings.start("compute_values")
        sampled_actions_iterator = groupby(all_sampled_actions, batch_repeat_interleave)
        for power, marginal_human_policy in bp_policy.items():
            power_evs[power] = {}
            for power_action, _ in list(marginal_human_policy.items()):
                sampled_actions = next(sampled_actions_iterator)
                for joint_action in sampled_actions:
                    joint_action[POWERS.index(power)] = power_action  # Just in case.

                joint_sampled_actions = [dict(zip(POWERS, a)) for a in sampled_actions]
                with timings.create_subcontext("v") as subtimings:
                    _, joint_values = unzip2(
                        self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                            game,
                            set_orders_dicts=joint_sampled_actions,
                            agent_power=power,
                            cache=maybe_rollout_results_cache,
                            timings=subtimings,
                        )
                    )
                power_values = [x[power] for x in joint_values]
                ev = sum(power_values) / len(power_values)
                power_evs[power][power_action] = ev
        return power_evs

    def run_search(
        self,
        game: Game,
        *,
        bp_policy: Optional[PowerPolicies] = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
    ) -> TheBestResult:
        del agent_power  # Not used.
        del agent_state  # Not used.
        del early_exit_for_power  # Not used.

        if timings is None:
            timings = TimingCtx()
        timings.start("one-time")

        # If there are no locations to order, bail
        # if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
        #     return _early_quit_search_result(early_exit_for_power)

        logging.info("BEGINNING TheBestAgent run_search (%s)", game.current_short_phase)

        if game.current_short_phase.endswith("A"):
            num_adj_phase_samples = 16  # Some random number that is small enough to be fast, but not 1 to reduce variance of values.
            sampled_actions, _ = self.rollout_base_strategy_model.forward_policy(
                [game],
                has_press=self.has_press,
                agent_power=None,
                temperature=1.0,
                top_p=1.0,
                batch_repeat_interleave=num_adj_phase_samples,
            )
            joint_policy = _JoinedPolicyData(
                power_action_dicts=[
                    dict(zip(POWERS, joint_action)) for joint_action in sampled_actions
                ],
                probs=np.ones([len(sampled_actions)]) / len(sampled_actions),
            )
        else:
            if self.cfg.sampling_type == "INDEPENDENT_PIKL":
                joint_policy = self._run_search_sample_independent_pikl(
                    game, bp_policy=bp_policy, timings=timings
                )
            elif self.cfg.sampling_type == "JOINT_CONDITIONAL":
                joint_policy = self._run_search_sample_joint_conditional(game, timings=timings)
            elif self.cfg.sampling_type == "HYBRID_JOINT_AND_INDEP_PIKL":
                joint_policy = self._run_search_sample_hybrid(
                    game, bp_policy=bp_policy, timings=timings
                )
            else:
                raise RuntimeError(f"Unknown sampling type: {self.cfg.sampling_type}")
        joint_policy = _normalize_policy(joint_policy)

        with timings("final_values"):
            values = self._compute_joint_policy_values(game, joint_policy,)

        timings.pprint(logging.getLogger("timings").info)

        logging.info(
            "Raw--- Values: %s",
            {
                p: f"{x:.3f}"
                for p, x in zip(
                    POWERS,
                    self.base_strategy_model.get_values(
                        game, has_press=self.has_press, agent_power=None
                    ),
                )
            },
        )
        logging.info(
            "Search Values: %s", {p: f"{v:.3f}" for p, v in zip2(POWERS, values.tolist())}
        )

        return TheBestResult(
            power_action_dicts=joint_policy.power_action_dicts,
            joint_probs=joint_policy.probs,
            values=values,
        )

    def _compute_joint_policy_values(self, game: Game, policy: _JoinedPolicyData) -> np.ndarray:
        """Sample some actions from a joint poilicy, step and compute EVs"""
        assert self.cfg.num_value_computation_samples is not None
        num_value_computation_samples = min(
            len(policy.power_action_dicts), self.cfg.num_value_computation_samples
        )
        if len(policy.power_action_dicts) <= self.cfg.num_value_computation_samples:
            # Doing exact computation.
            maybe_rollout_results_cache = self.base_strategy_model_rollouts.build_cache()
            _, joint_values = unzip2(
                self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                    game,
                    set_orders_dicts=policy.power_action_dicts,
                    agent_power=None,
                    cache=maybe_rollout_results_cache,
                )
            )
            joint_values_array = np.array([[values[p] for p in POWERS] for values in joint_values])
            return np.dot(joint_values_array.T, policy.probs)
        else:
            # Resort to uniform sampling.
            selected_power_action_indices = np.random.choice(
                len(policy.probs), replace=True, size=num_value_computation_samples, p=policy.probs
            )
            selected_power_action_dict = [
                policy.power_action_dicts[i] for i in selected_power_action_indices
            ]
            maybe_rollout_results_cache = self.base_strategy_model_rollouts.build_cache()
            _, joint_values = unzip2(
                self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                    game,
                    set_orders_dicts=selected_power_action_dict,
                    agent_power=None,
                    cache=maybe_rollout_results_cache,
                )
            )
            joint_values_array = np.array([[values[p] for p in POWERS] for values in joint_values])
            return joint_values_array.mean(0)

    def _run_search_sample_independent_pikl(
        self, game: Game, *, bp_policy: Optional[PowerPolicies] = None, timings: TimingCtx,
    ) -> _JoinedPolicyData:
        """Sampling from independent pikl policies for each power and rescoring these with joint bp."""

        if bp_policy is None:
            bp_policy = self.get_plausible_orders_policy(
                game,
                agent_power=None,
                agent_state=NoAgentState(),
                # player_rating=self.player_rating if self.set_player_rating else None,
            )

        # If there a single plausible action, no need to search.
        # if early_exit_for_power and len(bp_policy[early_exit_for_power]) == 1:
        #     [the_action] = bp_policy[early_exit_for_power]
        #     return _early_quit_search_result(early_exit_for_power, action=the_action)

        power_evs = self._compute_power_action_utilities(
            game=game, bp_policy=bp_policy, timings=timings
        )

        if self.compute_inside_ratio:
            # Estimating probability of a joint action to be within the
            # plausible orders box.
            timings.start("estimate_inside")
            num_inside_sample = max(self.num_br_samples * 10, 10000)
            sampled_actions, _ = self.anchor_base_strategy_model.forward_policy(
                [game],
                batch_repeat_interleave=num_inside_sample,
                has_press=self.has_press,
                agent_power=None,
                temperature=1.0,
                top_p=1.0,
            )
            bp_action_sets = [frozenset(bp_policy.get(power, [])) for power in POWERS]
            inside_share = np.mean(
                [
                    all(
                        orders == tuple() or orders in action_set
                        for (orders, action_set) in zip(joint_action, bp_action_sets)
                    )
                    for joint_action in sampled_actions
                ]
            )
            logging.info("Share of joint inside the box: %.2e", inside_share)

        bp_log_policy = {
            power: {action: math.log(prob + 1e-10) for action, prob in power_policy.items()}
            for power, power_policy in bp_policy.items()
        }

        timings.start("compute_qre_policies")
        denominator = 1.0 / self.eta + self.qre_lambda
        power_pikl_log_policies = {}
        for power, marginal_human_policy in bp_policy.items():
            power_pikl_log_policies[power] = {}
            for power_action in marginal_human_policy:
                log_bp_prob = bp_log_policy[power][power_action]
                value = power_evs[power][power_action]
                power_pikl_log_policies[power][power_action] = (
                    value + self.qre_lambda * log_bp_prob
                ) / denominator

        renormalize_log_policy(power_pikl_log_policies)
        for power in POWERS:
            self.log_independent_policies(
                game=game,
                pwr=power,
                bp_policy=bp_policy,
                pikl_log_policy=power_pikl_log_policies,
                power_evs=power_evs,
            )

        timings.start("sample_from_independent")

        per_power_log_probs = []
        per_power_pikl_log_probs = []
        per_power_actions = []
        joint_evs = []
        for power in POWERS:
            # we sample from pikl-probs, but interested to save the independent
            # BP prob for each action instead.
            actions, pikl_log_probs = unzip2(power_pikl_log_policies[power].items())
            sampled_ids = np.random.choice(
                len(pikl_log_probs),
                size=self.num_importance_samples,
                p=np.exp(np.array(pikl_log_probs)),
            )

            _, bp_log_probs = unzip2(list(bp_log_policy[power].items()))
            assert len(bp_log_probs) == len(actions)
            bp_log_probs = np.array(bp_log_probs)
            pikl_log_probs = np.array(pikl_log_probs)
            # print(bp_probs.shape, len(pikl_probs), sampled_ids.shape)
            per_power_log_probs.append(bp_log_probs[sampled_ids])
            per_power_pikl_log_probs.append(pikl_log_probs[sampled_ids])
            joint_evs.append([power_evs[power][actions[i]] for i in sampled_ids])
            per_power_actions.append([actions[i] for i in sampled_ids])
        independent_log_probs = np.stack(per_power_log_probs, 0).sum(0)
        independent_pikl_log_probs = np.stack(per_power_pikl_log_probs, 0).sum(0)
        joint_evs = np.array(joint_evs).sum(0)
        power_action_tuples = list(zip(*per_power_actions))
        power_action_dicts = [
            dict(zip(POWERS, power_orders)) for power_orders in power_action_tuples
        ]

        timings.start("weight_by_joint")
        joint_anchor_log_probs = np.array(
            compute_action_logprobs_from_state(
                self.anchor_base_strategy_model.model,
                game,
                power_action_dicts,
                agent_power=None,
                has_press=self.has_press,
                batch_size=self.anchor_base_strategy_model.max_batch_size,
            )
        )

        timings.start("compute_the_fake_policy")
        # p = exp( (joint_ev + lambda * joint_bp(a)) / denominator)  # unnormalized
        # q = exp( (joint_ev + lambda * indep_bp(a)) / denominator)  # unnormalized
        # log(p) - log(q) = (joined_bp(a) - indep_bp(a)) * lambda / denominator
        log_ratios = (joint_anchor_log_probs - independent_log_probs) * (
            self.qre_lambda / denominator
        )
        ratios = scipy.special.softmax(np.array(log_ratios))

        self.log_joined(
            game=game,
            power_action_tuples=power_action_tuples,
            joint_policy=ratios,
            joint_evs=joint_evs,
            extra_logprobs={
                "joint_anchor": joint_anchor_log_probs,
                "indep_anchor": independent_log_probs,
                "indep_pikl": independent_pikl_log_probs,
            },
        )

        timings.stop()

        return _JoinedPolicyData(power_action_dicts=power_action_dicts, probs=ratios)

    def _run_search_sample_joint_conditional(
        self, game: Game, *, timings: TimingCtx,
    ) -> _JoinedPolicyData:
        """Sampling from "conditional" joint policy and rescoring these with joint bp."""
        timings.start("sample_proposal")

        the_computational_budget = sum(self.order_sampler.get_plausible_order_limits(game))
        logging.info(f"the_computational_budget={the_computational_budget}")

        def _iterate_over_samples_from_conditional():
            while True:
                sampled_actions, logprobs = self.rollout_base_strategy_model.forward_policy(
                    [game],
                    batch_repeat_interleave=self.rollout_base_strategy_model.max_batch_size,
                    has_press=self.has_press,
                    agent_power=None,
                    temperature=1.0,
                    top_p=1.0,
                )
                assert (logprobs[:, 1:] < 1e-6).all(), "Not all powers?"
                yield from zip2(sampled_actions, logprobs[:, 0].tolist())

        proposal_joint_actions: List[JointAction] = []
        proposal_joint_actions_logprobs: List[float] = []
        seen_actions: Set[Action] = set()
        for power_actions, logprob in itertools.islice(
            _iterate_over_samples_from_conditional(), self.num_importance_samples
        ):
            seen_actions.update(power_actions)
            proposal_joint_actions.append(dict(zip(POWERS, power_actions)))
            proposal_joint_actions_logprobs.append(logprob)
            if len(seen_actions) > the_computational_budget:
                break
        logging.info(
            "Samples %d joint actions resulting in %d unique actions",
            len(proposal_joint_actions),
            len(seen_actions),
        )

        # Values are fake. We just want to collect all per-power actions we need to compute utilities for.
        selected_marginal_actions: PowerPolicies = {
            p: {ja[p]: -1 for ja in proposal_joint_actions} for p in POWERS
        }
        power_evs = self._compute_power_action_utilities(
            game=game, bp_policy=selected_marginal_actions, timings=timings
        )

        action_evs: Dict[Action, float] = {
            action: value
            for power_values in power_evs.values()
            for action, value in power_values.items()
        }
        joint_evs: List[float] = [
            sum(action_evs[action] for action in joint_action.values())
            for joint_action in proposal_joint_actions
        ]

        timings.start("weight_by_joint")
        joint_anchor_log_probs = compute_action_logprobs_from_state(
            self.anchor_base_strategy_model.model,
            game,
            proposal_joint_actions,
            agent_power=None,
            has_press=self.has_press,
            batch_size=self.anchor_base_strategy_model.max_batch_size,
        )

        timings.start("compute_the_fake_policy")
        # ratio = p(a) /  exp ( ( U(a) + lambda log tau(a)   ) / (1/eta + lambda))
        # log_ratio = log(p(a)) - ( u(a) + lambda log(tau)) / (1/eta + lambda))
        denominator = 1.0 / self.eta + self.qre_lambda
        joint_ev_max = max(joint_evs)  # To make exp(p/q) not insane.
        target_logprobs = [
            (joint_ev - joint_ev_max + self.qre_lambda * anchor_logprob) / denominator
            for anchor_logprob, joint_ev in zip2(joint_anchor_log_probs, joint_evs)
        ]

        log_ratios = [
            target_logprob - proposal_logprob
            for proposal_logprob, target_logprob in zip2(
                proposal_joint_actions_logprobs, target_logprobs
            )
        ]
        ratios = scipy.special.softmax(np.array(log_ratios))

        power_action_tuples = [
            tuple(joint_action.values()) for joint_action in proposal_joint_actions
        ]
        self.log_joined(
            game=game,
            power_action_tuples=power_action_tuples,
            joint_policy=ratios,
            joint_evs=joint_evs,
            extra_logprobs={
                "joint_pikl": np.array(target_logprobs),
                "learned_joint": np.array(proposal_joint_actions_logprobs),
            },
        )

        timings.stop()

        return _JoinedPolicyData(power_action_dicts=proposal_joint_actions, probs=ratios)

    def _run_search_sample_hybrid(
        self, game: Game, *, bp_policy: Optional[PowerPolicies] = None, timings: TimingCtx,
    ) -> _JoinedPolicyData:

        if bp_policy is None:
            bp_policy = self.get_plausible_orders_policy(
                game,
                agent_power=None,
                agent_state=NoAgentState(),
                # player_rating=self.player_rating if self.set_player_rating else None,
            )

        temp = self.cfg.hybrid_joint_temp

        the_computational_budget = sum(self.order_sampler.get_plausible_order_limits(game))
        logging.info(f"the_computational_budget={the_computational_budget}")

        # ---------------------------
        # We use half the budget to compute independent pikl policies
        bp_actions_per_power = 1 + the_computational_budget // 2 // len(game.get_alive_powers())
        logging.info(f"Truncationg BP to at most {bp_actions_per_power} actions per power")
        bp_policy = cutoff_policy(bp_policy, [bp_actions_per_power] * len(POWERS))

        renormalize_policy(bp_policy)
        logging.info("BP sizes: %s", {power: len(policy) for power, policy in bp_policy.items()})

        maybe_rollout_results_cache = self.base_strategy_model_rollouts.build_cache()
        power_evs = self._compute_power_action_utilities(
            game=game,
            bp_policy=bp_policy,
            timings=timings,
            maybe_rollout_results_cache=maybe_rollout_results_cache,
        )

        bp_log_policy = {
            power: {action: math.log(prob + 1e-10) for action, prob in power_policy.items()}
            for power, power_policy in bp_policy.items()
        }

        timings.start("compute_qre_policies")
        denominator = 1.0 / self.eta + self.qre_lambda
        power_pikl_log_policies = {}
        for power, marginal_human_policy in bp_policy.items():
            power_pikl_log_policies[power] = {}
            for power_action in marginal_human_policy:
                log_bp_prob = bp_log_policy[power][power_action]
                value = power_evs[power][power_action]
                power_pikl_log_policies[power][power_action] = (
                    value + self.qre_lambda * log_bp_prob
                ) / denominator

        renormalize_log_policy(power_pikl_log_policies)
        for power in POWERS:
            self.log_independent_policies(
                game=game,
                pwr=power,
                bp_policy=bp_policy,
                pikl_log_policy=power_pikl_log_policies,
                power_evs=power_evs,
            )
        power_pikl_policies: PowerPolicies = {
            power: {action: np.exp(logprob) for action, logprob in power_log_policies.items()}
            for power, power_log_policies in power_pikl_log_policies.items()
        }

        # ---------------------------
        # Now we start sample joint actions either from the idependt pikl or
        # from the conditional model until we reach the computational budget.
        def _iterate_over_samples_from_conditional():
            while True:
                sampled_actions, logprobs = self.rollout_base_strategy_model.forward_policy(
                    [game],
                    batch_repeat_interleave=self.rollout_base_strategy_model.max_batch_size,
                    has_press=self.has_press,
                    agent_power=None,
                    temperature=temp,
                    top_p=1.0,
                )
                assert (logprobs[:, 1:] < 1e-6).all(), "Not all powers?"
                yield from zip2(sampled_actions, logprobs[:, 0].tolist())

        probability_of_conditional_policy = 1 - self.cfg.hybrid_independent_pikl_prob

        def _iterate_over_samples_from_mixed():
            conditional_iterator = iter(_iterate_over_samples_from_conditional())
            while True:
                if random.random() < probability_of_conditional_policy:
                    action, _ = next(conditional_iterator)
                else:
                    action = [sample_p_dict(policy) for policy in power_pikl_policies.values()]
                yield action

        proposal_joint_actions: List[JointAction] = []
        # Pre-populate seen action with all action from BP policy - we already
        # computed values for these actions.
        seen_actions: Set[Action] = set(
            action for policy in bp_policy.values() for action in policy
        )
        for power_actions in itertools.islice(
            _iterate_over_samples_from_mixed(), self.num_importance_samples
        ):
            seen_actions.update(power_actions)
            proposal_joint_actions.append(dict(zip(POWERS, power_actions)))
            if len(seen_actions) > the_computational_budget:
                break
        logging.info(
            "Samples %d joint actions resulting in %d unique actions",
            len(proposal_joint_actions),
            len(seen_actions),
        )

        # ---------------------------
        # Now we have action that sampled from a mixture of the conditional
        # policy and independent policy. For each action we need to compute
        # probability under both policies to compute esimate of the prior
        # probability.
        timings.start("compute_mixture_probs")

        # First, compute the probabilites under the conditional model.
        joined_policy_logprobs = np.array(
            compute_action_logprobs_from_state(
                self.rollout_base_strategy_model.model,
                game,
                proposal_joint_actions,
                has_press=self.has_press,
                agent_power=None,
                batch_size=self.rollout_base_strategy_model.max_batch_size,
                temperature=temp,
            )
        )

        # Second, compute the probabilities under the independent pikl policy.
        indep_pikl_logprobs = np.array(
            [
                sum(
                    power_pikl_log_policies[power].get(action, -1e100)
                    for power, action in joint_action.items()
                )
                for joint_action in proposal_joint_actions
            ]
        )
        # Mix the two together.
        safe_log = lambda x: math.log(max(x, 1e-100))
        proposal_joint_actions_logprobs: np.ndarray = scipy.special.logsumexp(
            np.stack(
                [
                    joined_policy_logprobs + safe_log(probability_of_conditional_policy),
                    indep_pikl_logprobs + safe_log(1 - probability_of_conditional_policy),
                ]
            ),
            axis=0,
        )

        # ---------------------------
        # In order to compute q(action) = joint-pikl-probability(action), we
        # need to know utilities of all actions. We already computed utilities
        # for some of the actions from independent pikl. Need to take diff in
        # order to not-recompute.
        timings.start("compute_extra_evs")

        # Values are fake. We just want to collect all per-power actions we need to compute utilities for.
        new_marginal_actions: PowerPolicies = {
            p: {ja[p]: -1 for ja in proposal_joint_actions} for p in POWERS
        }
        for power, policy in bp_policy.items():
            for action in policy:
                if action in new_marginal_actions[power]:
                    del new_marginal_actions[power][action]
        new_power_evs = self._compute_power_action_utilities(
            game=game,
            bp_policy=new_marginal_actions,
            maybe_rollout_results_cache=maybe_rollout_results_cache,
            timings=TimingCtx(),
        )
        logging.info(
            "Novel power-action pairs: %s", sum(len(x) for x in new_marginal_actions.values())
        )

        for power in new_power_evs:
            for action in new_power_evs[power]:
                power_evs[power][action] = new_power_evs[power][action]

        action_evs: Dict[Action, float] = {
            action: value
            for power_values in power_evs.values()
            for action, value in power_values.items()
        }
        joint_evs = np.array(
            [
                sum(action_evs[action] for action in joint_action.values())
                for joint_action in proposal_joint_actions
            ]
        )

        timings.start("weight_by_joint")
        joint_anchor_log_probs = np.array(
            compute_action_logprobs_from_state(
                self.anchor_base_strategy_model.model,
                game,
                proposal_joint_actions,
                agent_power=None,
                has_press=self.has_press,
                batch_size=self.anchor_base_strategy_model.max_batch_size,
            )
        )

        timings.start("compute_the_fake_policy")
        denominator = 1.0 / self.eta + self.qre_lambda
        joint_ev_max = joint_evs.max()  # To make exp(p/q) not insane.
        target_logprobs = (
            joint_evs - joint_ev_max + self.qre_lambda * joint_anchor_log_probs
        ) / denominator

        log_ratios = target_logprobs - proposal_joint_actions_logprobs
        ratios = scipy.special.softmax(log_ratios)

        power_action_tuples = [
            tuple(joint_action.values()) for joint_action in proposal_joint_actions
        ]
        print(
            "indep_pikl_logprobs",
            indep_pikl_logprobs.min(),
            indep_pikl_logprobs.max(),
            np.exp(indep_pikl_logprobs).min(),
            np.exp(indep_pikl_logprobs).max(),
            min([math.exp(x) for x in indep_pikl_logprobs]),
            max([math.exp(x) for x in indep_pikl_logprobs]),
        )
        self.log_joined(
            game=game,
            power_action_tuples=power_action_tuples,
            joint_policy=ratios,
            joint_evs=joint_evs,
            extra_logprobs={
                "joint_pikl*": np.array(target_logprobs),
                "indep_pikl": indep_pikl_logprobs,
                "joint_learned": joined_policy_logprobs,
                "prop_mix": proposal_joint_actions_logprobs,
                "joint_anchor": joint_anchor_log_probs,
            },
        )

        timings.stop()

        return _JoinedPolicyData(power_action_dicts=proposal_joint_actions, probs=ratios)

    def log_independent_policies(
        self, *, game, pwr, bp_policy, power_evs, pikl_log_policy,
    ):
        logging.info(f"<> [  ] {pwr} {game.phase} ")
        logging.info(f"     {'pikl_p':8s}  {'bp_p':8s}  {'ev_p':8s}  {'ev':8s}  orders")

        actions = list(bp_policy[pwr])
        pikl_probs: List[float] = [math.exp(pikl_log_policy[pwr][action]) for action in actions]
        bp_probs: List[float] = [bp_policy[pwr][action] for action in actions]
        utilities: List[float] = [power_evs[pwr][action] for action in actions]
        utilities_policy: List[float] = torch.softmax(
            torch.FloatTensor(utilities) / self.qre_lambda, -1
        ).tolist()
        sorted_metrics = sorted(
            zip(actions, pikl_probs, bp_probs, utilities_policy, utilities), key=lambda ac: -ac[1],
        )
        for orders, pikl_p, bp_p, u_p, u in sorted_metrics[:10]:
            logging.info(f"|>  {pikl_p:8.5f}  {bp_p:8.5f}  {u_p:8.5f}  {u:8.5f}  {orders}")

    def log_joined(
        self,
        *,
        game,
        power_action_tuples: List[Tuple[Action, ...]],
        joint_policy,
        joint_evs: Union[List[float], np.ndarray],
        extra_logprobs: Dict[str, np.ndarray],
    ) -> None:
        action_counter = collections.Counter(power_action_tuples)
        top_n = 10

        # print(action_counter)
        metrics = {}
        for i, j_action in enumerate(power_action_tuples):
            n = action_counter[j_action]
            metrics[j_action] = dict(
                prob=joint_policy[i] * n,
                ev=joint_evs[i],
                extra_logprobs={key: lp[i] for key, lp in extra_logprobs.items()},
            )
        sorted_metrics = sorted(metrics.items(), key=lambda ac: -ac[1]["prob"],)

        chunks = []
        sub_top_n = 10
        probs_list = np.array([metrics["prob"] for _, metrics in sorted_metrics])
        while sub_top_n < len(sorted_metrics):
            weight = probs_list[:sub_top_n].sum()
            chunks.append("p(top-{})={:.2f}".format(sub_top_n, weight))
            sub_top_n *= 2
        logging.info(f"<> [  ] TOP-{top_n}/{len(action_counter)} JOINT {game.phase} ")
        ent = -(probs_list * np.log(probs_list + 1e-10)).sum()
        logging.info("Distribution: ent=%.1e ppl=%.1f %s", ent, np.exp(ent), " ".join(chunks))
        logging.info(
            f"    {'prob':>8s}  {'ev':>8s}  %s", "  ".join(f"{key:>12s}" for key in extra_logprobs)
        )

        for (orders, order_metrics) in sorted_metrics[:top_n]:
            logging.info(
                f"|>  {order_metrics['prob']:8.5f}  {order_metrics['ev']:8.5f}  %s",
                "  ".join(
                    f"{math.exp(v):12.2e}" for v in order_metrics["extra_logprobs"].values()
                ),
            )
            # logging.info(
            #     f"|>  {'':8s}  {'log()':8s}  %s",
            #     "  ".join(f"{v:12.6f}" for v in order_metrics["extra_logprobs"].values()),
            # )
            for o in orders:
                logging.info(f"    {'':18s}   {o}")


def groupby(a_list: List, n: int):
    i = 0
    while i < len(a_list):
        yield a_list[i : i + n]
        i += n


def renormalize_log_policy(log_policy: PowerPolicies) -> None:
    for power, orders_to_log_probs in list(log_policy.items()):
        assert orders_to_log_probs, f"Empty policy for {power} in {log_policy}"
        orders, log_probs = unzip2(orders_to_log_probs.items())
        log_probs = scipy.special.log_softmax(np.array(log_probs))
        log_policy[power] = dict(zip(orders, log_probs.tolist()))


def _normalize_policy(policy: _JoinedPolicyData) -> _JoinedPolicyData:
    """Sum probabilities for duplicate actions."""
    accumulator = collections.defaultdict(float)
    for power_dict, prob in zip2(policy.power_action_dicts, policy.probs):
        accumulator[tuple(power_dict.values())] += prob
    power_dicts = [dict(zip2(POWERS, actions)) for actions in accumulator]
    probs = np.array(list(accumulator.values()))
    assert (
        abs(sum(probs) - 1.0) < 1e-2
    ), "Joint policy probs do not sum to one at all! sum={}".format(sum(probs))
    return _JoinedPolicyData(power_action_dicts=power_dicts, probs=probs)
