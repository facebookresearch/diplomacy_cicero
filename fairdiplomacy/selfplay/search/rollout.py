#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import dataclasses
import datetime
import io
import logging
import math
import pathlib
import random
import queue as queue_lib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, TypeVar

import nest
import numpy as np
import postman
import torch
import torch.multiprocessing as mp_module

import heyhi
import conf.agents_cfgs
import conf.agents_pb2
import conf.conf_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.action_exploration import compute_evs_fva, double_oracle
from fairdiplomacy.agents.base_search_agent import SearchResult
from fairdiplomacy.agents.the_best_agent import TheBestAgent
from fairdiplomacy.agents.base_strategy_model_wrapper import compute_action_logprobs_from_state
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS
from fairdiplomacy.selfplay.ckpt_syncer import build_searchbot_like_agent_with_syncs
from fairdiplomacy.selfplay.search.search_utils import (
    power_prob_distributions_to_tensors,
    compute_marginal_policy,
    create_research_targets_single_rollout,
    perform_retry_loop,
    unparse_device,
)
from fairdiplomacy.typedefs import Action, JointPolicy, Policy, Power, PowerPolicies
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY
from fairdiplomacy.utils.sampling import sample_p_dict, sample_p_joint_list
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.zipn import unzip2


QUEUE_PUT_TIMEOUT = 1.0

ReSearchRolloutBatch = collections.namedtuple(
    "ReSearchRolloutBatch",
    # is_search_policy_valid, search_policy_probs, and search_policy_orders
    # have different shapes depending on wheather the model is all-powers
    [
        "observations",
        "rewards",  # Tensor [T, 7]. Final rewards for the episode.
        "done",  # Tensor bool [T]. False everywhere but last phase.
        "is_explore",  # Tensor bool [T, 7]. True if the agent deviated at this phase.
        "explored_on_the_right",  # Tensor bool [T, 7]. True if the agent will deviate.
        "scores",  # [T, 7], SoS scores at the end of each phase.
        "values",  # [T, 7], Values for all states from the value net.
        "targets",  # [T, 7], bootstapped targets.
        "is_search_policy_valid",  # [T, 7] or [T, 1]. search_policy_probs, search_policy_orders, and
        # blueprint_probs should be ingored if this is 0.
        "state_hashes",  # [T] long hash of the state
        "years",  # [T]
        "phase_type",  # Byte tensor [T], contains ord("M"), ord("A"), or ord("R").
        "search_policy_evs",  # Float tensor [T, 7, max_actions] or zeros of [T] if not collect_search_evs.
        "search_policy_probs",  # [T, 7, max_actions] or [T, 1, max_actions] or zeros of [T].
        "search_policy_orders",  # [T, 7, max_actions, MAX_SEQ_LEN] or [T, 1, max_actions, N_SCS] or zeros of [T].
        "blueprint_probs",  # [T, 7, max_actions] or zeros of [T]. Probabilities of search_policy_orders under blueprint.
    ],
)

# A ByPowerList is a List of length 7 whose indices correspond to the powers
# This type alias doesn't enforce any typesafety, it just helps make the code more self-documenting
T = TypeVar("T")
ByPowerList = List[T]

JOINT_ENTROPY_METRIC_NAMES: Tuple[str, ...] = (
    "joint_entropy_move",
    "joint_entropy_adj",
    "joint_entropy",
)


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    batch: ReSearchRolloutBatch
    # Info about the initial state rollout produced on
    game_meta: Dict  # Info about the produced game and initial state.
    # Info about the checkpoint that was used for the rollout. One per syncer.
    last_ckpt_metas: Optional[Dict[str, Dict]]


def _load_game_with_phase(path: str) -> pydipcc.Game:
    if ":" in path:
        path, phase = path.rsplit(":", 1)
    else:
        path, phase = path, None
    with open(path) as stream:
        game_serialized = stream.read()
    game = pydipcc.Game.from_json(game_serialized)
    if phase is not None:
        game = game.rolled_back_to_phase_start(phase)
    return game


def yield_game(
    seed: int,
    game_json_paths: Optional[Sequence[str]],
    game_kwargs: Optional[Dict],
    sample_game_json_phases: bool,
    *,
    logger=None,
):

    if logger is not None and game_json_paths:
        logger.info(f"Using {len(game_json_paths)} game json paths!")
        logger.info(f"extra_params_cfg.sample_game_json_phases = {sample_game_json_phases}")

    game_sample_weight = None
    if game_json_paths is not None:
        game_sample_weight_list = []
        if not sample_game_json_phases:
            # Sample everything equally likely
            game_sample_weight_list = [1.0 for _ in game_json_paths]
        else:
            # Sample proportional to length of game, so each phase is equally likely
            for game_json_path in game_json_paths:
                game = _load_game_with_phase(game_json_path)
                game_length = len(game.get_all_phase_names())
                game_sample_weight_list.append(game_length)
        game_sample_weight = np.array(game_sample_weight_list, dtype=np.float32)
        game_sample_weight /= np.sum(game_sample_weight)

    rng = np.random.RandomState(seed=seed)  # type:ignore

    # If we cannot sample a game this many times, just die.
    MAX_RETRIES = 100

    retries_left = MAX_RETRIES
    while True:
        if game_json_paths is None:
            game = pydipcc.Game()
            game_id = "std"
        else:
            p = game_json_paths[rng.choice(len(game_json_paths), p=game_sample_weight)]
            game = _load_game_with_phase(p)
            if sample_game_json_phases:
                phase_names = game.get_all_phase_names()
                phase = phase_names[rng.choice(len(phase_names))]
                game = game.rolled_back_to_phase_start(phase)
            game_id = pathlib.Path(p).name.rsplit(".", 1)[0]
            # Sometimes we may sample a game state that is already done from
            # game json. If so, skip it and do another.
            if game.is_game_done:
                if retries_left > 0:
                    if logger is not None:
                        logger.info(
                            f"Skipping game {game_id} since already done. Attempts"
                            f" left: {retries_left}"
                        )
                    # Just in case, avoid hot loop where we print out a million
                    # log messages per second when there are no games
                    retries_left -= 1
                    continue
                else:
                    raise RuntimeError(
                        "Invalid game_json_paths file provided. Failed to find"
                        f" non-done game after {MAX_RETRIES} atempts"
                    )
            else:
                retries_left = MAX_RETRIES
        if game_kwargs is not None:
            for k, v in game_kwargs.items():
                getattr(game, f"set_{k}")(v)

        yield game_id, game


def yield_rollouts(
    *,
    device: str,
    game_json_paths: Optional[Sequence[str]],
    agent_cfg: conf.agents_cfgs.Agent,
    seed: Optional[int] = None,
    ckpt_sync_path: Optional[str] = None,
    eval_mode: bool = False,
    game_kwargs: Optional[Dict] = None,
    stats_server: Optional[str] = None,
    extra_params_cfg: conf.conf_cfgs.ExploitTask.SearchRollout.ExtraRolloutParams,
    collect_game_logs: bool = False,
) -> Generator[RolloutResult, None, None]:
    """Do non-stop rollout for 7 CFR agents.

    This method can safely be called in a subprocess

    Arguments:
    - model_path: str
    - game_jsons: either None or a list of paths to json-serialized games.
    - agent_cfg: message of type Agent with oneof set to SearchBotAgent or BQREAgent
    - seed: seed for _some_ randomness within the rollout.
    - ckpt_sync_path: if not None, will load the model from this folder.
    - eval_mode: whether rollouts are for train or eval, e.g., to exploration in eval.
    - game_kwargs: Optional a dict of modifiers for the game.
    - stats_server: if not None, a postman server where to send stats about the game
    - extra_params_cfg: Misc flags straight from the proto.
    - collect_game_logs: if True, the game json will collect logs for each phase.

    yields a RolloutResult.

    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = int(torch.randint(1_000_000_000, [1]))

    fake_gen = extra_params_cfg.fake_gen or 1

    if extra_params_cfg.randomize_best_agent_sampling_type:
        agent_cfg = _randomize_best_agent_sampling(agent_cfg)

    # ---- Creating SearchBot config to use in this worker.
    agent, do_sync_fn = build_searchbot_like_agent_with_syncs(
        agent_cfg,
        ckpt_sync_path=ckpt_sync_path,
        use_trained_value=extra_params_cfg.use_trained_value,
        use_trained_policy=extra_params_cfg.use_trained_policy,
        device_id=unparse_device(device),
    )
    exploited_agent_power = agent.get_exploited_agent_power()
    is_all_powers = isinstance(agent, TheBestAgent)

    assert not (
        extra_params_cfg.independent_explore and extra_params_cfg.explore_all_but_one
    ), "Mutually exclusive flags"

    input_version = agent.base_strategy_model.get_policy_input_version()
    assert input_version == agent.base_strategy_model.get_value_input_version()
    input_encoder = FeatureEncoder()

    if extra_params_cfg.max_year is not None:
        logging.warning("Will simulate only up to %s", extra_params_cfg.max_year)
    if extra_params_cfg.max_episode_length is None:
        max_episode_length = 100000000
    else:
        assert extra_params_cfg.max_episode_length > 0
        max_episode_length = extra_params_cfg.max_episode_length
        logging.warning(f"Will simulate only {max_episode_length} steps")

    assert (extra_params_cfg.min_max_episode_movement_phases is None) == (
        extra_params_cfg.max_max_episode_movement_phases is None
    )
    if (
        extra_params_cfg.min_max_episode_movement_phases is None
        or extra_params_cfg.max_max_episode_movement_phases is None
    ):
        min_max_episode_movement_phases = 100000000
        max_max_episode_movement_phases = 100000000
    else:
        min_max_episode_movement_phases = extra_params_cfg.min_max_episode_movement_phases
        max_max_episode_movement_phases = extra_params_cfg.max_max_episode_movement_phases

    max_policy_size = extra_params_cfg.max_policy_size or agent.n_plausible_orders
    assert max_policy_size is not None

    if stats_server:
        stats_client = postman.Client(stats_server)
        stats_client.connect()
    else:
        stats_client = None

    # ---- Logging.
    logger = logging.getLogger("")
    if collect_game_logs:
        logger.info("Collecting logs into game json")
        _set_all_logger_handlers_to_level(logger, logging.WARNING)
    game_log_handler = None

    rng = np.random.RandomState(seed=seed)  # type:ignore
    last_ckpt_metas = None
    for rollout_id, (game_id, game) in enumerate(
        yield_game(
            seed,
            game_json_paths,
            game_kwargs,
            extra_params_cfg.sample_game_json_phases,
            logger=logger,
        )
    ):
        if extra_params_cfg.max_year:
            if extra_params_cfg.randomize_max_year:
                game_max_year = rng.randint(1902, extra_params_cfg.max_year + 1)
            else:
                game_max_year = extra_params_cfg.max_year
        else:
            game_max_year = None

        if rollout_id < 1 or (rollout_id & (rollout_id + 1)) == 0:
            _set_all_logger_handlers_to_level(logger, logging.INFO)
        else:
            logging.info("Skipping logs for rollout %s", rollout_id)
            _set_all_logger_handlers_to_level(logger, logging.WARNING)
            # Keep the game log handler at info level for writing into game json
            if game_log_handler is not None:
                game_log_handler.setLevel(logging.INFO)
        game_meta: Dict[str, Any] = {"start_phase": game.current_short_phase, "game_id": game_id}
        timings = TimingCtx()
        with timings("ckpt_sync"):
            if ckpt_sync_path is not None:
                last_ckpt_metas = do_sync_fn()

        has_press = extra_params_cfg.default_has_press
        if extra_params_cfg.randomize_has_press and random.choice([False, True]):
            has_press = True
        agent.override_has_press(has_press)

        if extra_params_cfg.randomize_sosdss and random.choice([False, True]):
            game.set_scoring_system(pydipcc.Game.SCORING_DSS)

        max_episode_movement_phases_left = rng.randint(
            min_max_episode_movement_phases, max_max_episode_movement_phases + 1
        )
        agent_power = None
        if extra_params_cfg.use_random_agent_power:
            agent_power = rng.choice(POWERS)

        with timings("rl.init"):
            # Need to do some measurement to make Lost time tracking work.
            stats = collections.defaultdict(float)
            list_stats: Dict[str, list] = {}
            if is_all_powers:
                list_stats.update((k, []) for k in JOINT_ENTROPY_METRIC_NAMES)
            observations: List[DataFields] = []
            is_explore: List[ByPowerList[bool]] = []
            scores: List[ByPowerList[float]] = []
            phases: List[str] = []
            years: List[int] = []
            phase_type: List[int] = []
            state_hashes: List[int] = []

            is_search_policy_valid_flags: List[
                ByPowerList[bool]
            ] = []  # Only filled if collect_search_policies.
            search_policies: List[Dict[str, Any]] = []  # Only filled if collect_search_policies.
            search_policy_evs: List[torch.Tensor] = []  # Only filled if collect_search_evs.
            target_evs: List[torch.Tensor] = []  # Only filled if collect_search_evs.

            # If true, we consider the game to be ended here (e.g. draw by agreement)
            # even if not ended by the normal pydipcc rules.
            force_treat_game_as_done = False

        # Truncate the portion of the rollout we train on to
        # just [0,right)
        def truncate_rollout_keeping_final_score(*, right):
            if right >= len(observations):
                return
            # Preserve final score for the rollout
            final_scores = scores[-1]

            del observations[right:]
            del is_explore[right:]
            del scores[right:]
            del phases[right:]
            del years[right:]
            del phase_type[right:]
            del state_hashes[right:]

            del is_search_policy_valid_flags[right:]
            del search_policies[right:]
            del search_policy_evs[right:]
            del target_evs[right:]

            # Reassign final score after truncation
            scores[-1] = final_scores

        while not game.is_game_done:
            current_year = int(game.current_short_phase[1:-1])
            if game_max_year and current_year > game_max_year:
                break
            if len(phases) >= max_episode_length:
                break
            if game.current_short_phase.endswith("M"):
                if max_episode_movement_phases_left <= 0:
                    break
                max_episode_movement_phases_left -= 1

            assert not force_treat_game_as_done

            if collect_game_logs:
                if game_log_handler is not None:
                    logger.removeHandler(game_log_handler)
                log_stream = io.StringIO()
                game_log_handler = logging.StreamHandler(log_stream)
                game_log_handler.setLevel(logging.INFO)
                logger.addHandler(game_log_handler)
            else:
                log_stream = None

            state_hashes.append(game.compute_board_hash() % 2 ** 63)
            phases.append(game.current_short_phase)
            years.append(int(game.current_short_phase[1:-1]))
            phase_type.append(ord(game.current_short_phase[-1]))
            observations.append(
                [input_encoder.encode_inputs, input_encoder.encode_inputs_all_powers][
                    agent.base_strategy_model.is_all_powers()
                ]([game], input_version=input_version)
            )

            x_board_state = observations[-1]["x_board_state"]
            if has_press:
                observations[-1]["x_has_press"] = x_board_state.new_ones(x_board_state.shape[0], 1)
            else:
                observations[-1]["x_has_press"] = x_board_state.new_zeros(
                    x_board_state.shape[0], 1
                )

            with timings("plausible_orders"):
                if extra_params_cfg.random_policy:
                    plausible_orders_policy = _random_plausible_orders(
                        observations[-1]["x_possible_actions"], max_policy_size
                    )
                elif extra_params_cfg.always_play_blueprint:
                    with torch.no_grad():
                        actions_by_power, _logprobs = agent.base_strategy_model.forward_policy(
                            [game],
                            has_press=has_press,
                            agent_power=agent_power,
                            temperature=extra_params_cfg.always_play_blueprint.temperature,
                            top_p=extra_params_cfg.always_play_blueprint.top_p,
                        )
                    plausible_orders_policy = {
                        power: {actions_by_power[0][power_idx]: 1.0}
                        for power_idx, power in enumerate(POWERS)
                    }
                else:
                    # Explicitly pass agent_state=None, right now we don't handle agents
                    # that need per-power per-game state because we rely on a single call
                    # to the agent to generate actions for everyone
                    plausible_orders_policy = agent.get_plausible_orders_policy(
                        game, agent_state=None, agent_power=agent_power
                    )
                    # In the case of an exploited power that has order augmentation, it is
                    # possible the order augmentation adds more orders to their policy.
                    # So we truncate to max_policy_size.
                    plausible_orders_policy = {
                        power: truncate_policy(policy, max_policy_size)
                        for power, policy in plausible_orders_policy.items()
                    }

            run_double_oracle: bool
            is_search_policy_valid: bool
            if eval_mode or extra_params_cfg.do is None:
                run_double_oracle = False
                is_search_policy_valid = True
            else:
                should_run_do = random.random() <= extra_params_cfg.run_do_prob
                is_search_policy_valid = (
                    should_run_do or extra_params_cfg.allow_policy_tragets_without_do
                )
                run_double_oracle = should_run_do and game.current_short_phase.endswith("M")
            inner_timings = TimingCtx()

            all_power_prob_distributions: PowerPolicies  # Marginal polcieis
            all_power_joint_prob_distributions: Optional[JointPolicy] = None  # Joint policies.
            search_result: Optional[SearchResult]
            if is_all_powers:
                assert isinstance(agent, TheBestAgent), type(agent)
                assert not extra_params_cfg.always_play_blueprint
                assert not run_double_oracle
                search_result = agent.run_search(
                    game,
                    timings=inner_timings,
                    bp_policy=plausible_orders_policy,
                    agent_state=None,
                    agent_power=agent_power,
                )
                all_power_joint_prob_distributions = search_result.get_joint_policy(
                    max_policy_size
                )
                all_power_prob_distributions = compute_marginal_policy(
                    all_power_joint_prob_distributions
                )
                entropy = (-search_result.joint_probs * np.log(search_result.joint_probs)).sum()
                if game.current_short_phase.endswith("M"):
                    list_stats["joint_entropy_move"].append(entropy)
                if game.current_short_phase.endswith("A"):
                    list_stats["joint_entropy_adj"].append(entropy)
                list_stats["joint_entropy"].append(entropy)
            elif extra_params_cfg.always_play_blueprint:
                search_result = None
                all_power_prob_distributions = plausible_orders_policy
            elif not run_double_oracle:
                # Explicitly pass agent_state=None, right now we don't handle agents
                # that need per-power per-game state because we rely on a single call
                # to the agent to generate actions for everyone
                search_result = agent.run_search(
                    game,
                    timings=inner_timings,
                    bp_policy=plausible_orders_policy,
                    agent_state=None,
                    agent_power=agent_power,
                )
                all_power_prob_distributions = search_result.get_population_policy()
            else:
                all_power_prob_distributions, do_stats, search_result = double_oracle(
                    game,
                    agent,
                    double_oracle_cfg=extra_params_cfg.do,
                    need_actions_only=False,
                    initial_plausible_orders_policy=plausible_orders_policy,
                    agent_power=agent_power,
                    timings=inner_timings,
                )
                stats["do_attempts"] += 1
                stats["do_successes"] += do_stats["num_changes"]
                # Truncating to max_policy_size.
                all_power_prob_distributions = {
                    power: truncate_policy(policy, max_policy_size)
                    for power, policy in all_power_prob_distributions.items()
                }

            timings += inner_timings
            is_power_exploring = _decide_who_explores(
                game,
                rng,
                eval_mode=eval_mode,
                explore_eps=extra_params_cfg.explore_eps,
                explore_s1901m_eps=extra_params_cfg.explore_s1901m_eps,
                explore_f1901m_eps=extra_params_cfg.explore_f1901m_eps,
                independent_explore=extra_params_cfg.independent_explore,
                explore_all_but_one=extra_params_cfg.explore_all_but_one,
                exploited_agent_power=exploited_agent_power,
            )
            # Making a move: sampling from the policies.
            if all_power_joint_prob_distributions is not None:
                power_orders = sample_p_joint_list(all_power_joint_prob_distributions)
                for i, power in enumerate(POWERS):
                    if all_power_prob_distributions[power] and is_power_exploring[i]:
                        # Uniform sampling from plausible actions. Note, for
                        # joint policy this may contain power-actions outside of
                        # the joint policy as this things samples from the
                        # marginal policies.
                        power_orders[power] = _choice(
                            rng, list(plausible_orders_policy[power].keys()) or [tuple()]
                        )
            else:
                power_orders = {}
                for i, power in enumerate(POWERS):
                    if not all_power_prob_distributions[power]:
                        power_orders[power] = tuple()
                    elif is_power_exploring[i]:
                        # Uniform sampling from plausible actions.
                        power_orders[power] = _choice(
                            rng, list(plausible_orders_policy[power].keys()) or [tuple()]
                        )
                    else:
                        power_orders[power] = sample_p_dict(
                            all_power_prob_distributions[power], rng=rng
                        )

            if extra_params_cfg.collect_search_policies:
                if is_all_powers:
                    is_search_policy_valid_flags.append([is_search_policy_valid])
                else:
                    is_search_policy_valid_flags.append(
                        [
                            (is_search_policy_valid if power != exploited_agent_power else False)
                            for power in POWERS
                        ]
                    )

                if is_search_policy_valid:
                    with timings("rl.collect_policies"):
                        try:
                            phase_orders, phase_probs = power_prob_distributions_to_tensors(
                                all_power_joint_prob_distributions or all_power_prob_distributions,
                                max_policy_size,
                                observations[-1]["x_possible_actions"].squeeze(0),
                                observations[-1]["x_in_adj_phase"].item(),
                                observations[-1].get("x_power"),
                            )
                        except AssertionError:
                            fname = "debug.pt.%s" % datetime.datetime.now().isoformat()
                            torch.save(
                                dict(
                                    game_json=game.to_json(),
                                    all_power_joint_prob_distributions=all_power_joint_prob_distributions,
                                    all_power_prob_distributions=all_power_prob_distributions,
                                    max_policy_size=max_policy_size,
                                    observations=observations,
                                ),
                                fname,
                            )
                            logging.error("Saving to %s", fname)
                            raise
                        # This may be bullshit for joint policy.
                        phase_blueprint_probs = _compute_action_probs(
                            agent.base_strategy_model.model,
                            game,
                            all_power_prob_distributions,
                            max_actions=max_policy_size,
                            batch_size=agent.base_strategy_model.max_batch_size,
                            half_precision=agent.base_strategy_model.half_precision,
                            agent_power=agent_power,
                        )
                        search_policies.append(
                            dict(
                                probs=phase_probs,
                                orders=phase_orders,
                                blueprint_probs=phase_blueprint_probs,
                            )
                        )
                else:
                    search_policies.append(
                        dict(
                            probs=torch.empty((len(POWERS), max_policy_size), dtype=torch.long),
                            orders=torch.empty((len(POWERS), max_policy_size, MAX_SEQ_LEN)),
                            blueprint_probs=torch.empty((len(POWERS), max_policy_size)),
                        )
                    )
            if extra_params_cfg.collect_search_evs:
                with timings("rl.collect_evs"):
                    if not extra_params_cfg.use_cfr_evs:
                        assert (
                            agent_power is None
                        ), "agent_power handling not implemented for not use_cfr_evs"
                        assert not is_all_powers, "Cannot use not use_cfr_evs with is_all_powers"
                        phase_evs, phase_action_evs = _compute_evs_as_tensors(
                            game,
                            agent,
                            agent_power,
                            all_power_prob_distributions,
                            max_actions=max_policy_size,
                        )
                    else:
                        assert search_result is not None
                        phase_evs, phase_action_evs = _compute_evs_as_tensors_from_cfr_data(
                            all_power_prob_distributions,
                            search_result,
                            max_actions=max_policy_size,
                            compute_per_power_evs=not is_all_powers,
                        )
                    # Hack: set EVs for dead powers to zero and renormalize.
                    # In theory the agent should learn it, but we can help it a little bit.
                    alive_power_ids = game.get_alive_power_ids()
                    dead_power_ids = list(set(range(len(POWERS))).difference(alive_power_ids))
                    if dead_power_ids:
                        phase_evs[dead_power_ids] = 0.0
                        phase_evs /= phase_evs.sum()
                target_evs.append(phase_evs)
                search_policy_evs.append(phase_action_evs)

            is_explore.append(is_power_exploring)
            for power, orders in power_orders.items():
                game.set_orders(power, orders)
            if collect_game_logs:
                assert log_stream is not None
                game.add_log(log_stream.getvalue())
            game.process()
            scores.append(game.get_scores())

            # After any spring movement phase check if the agent is configured to evaluate
            # that the game may randomly end during spring. If true, then also randomly end during
            # after spring with the same chance.
            if game.current_short_phase.startswith("S") and game.current_short_phase.endswith("M"):
                prob_ending = agent.base_strategy_model_rollouts.get_prob_of_spring_ending(
                    int(game.current_short_phase[1:-1])
                )
                if rng.uniform() < prob_ending:
                    force_treat_game_as_done = True
                    break

        # If the game itself was terminated without finishing, then replace the last scores
        # of the game with a value net evaluation.
        if not game.is_game_done and not force_treat_game_as_done:
            if len(observations) > 0:
                value = agent.base_strategy_model.forward_values(
                    [game], has_press=False, agent_power=agent_power
                )
                assert value.shape == (1, 7)
                final_value_net = value.squeeze(0).tolist()

                # If this parameter is specified, we chop off the ends of the episode
                # for a game that wasn't finished normally, since we want a minimum amount of
                # rollout length between us and the final values from the net we are trying to train on.
                if (
                    extra_params_cfg.min_undone_episode_length is not None
                    and extra_params_cfg.min_undone_episode_length > 0
                ):
                    right = len(observations) - extra_params_cfg.min_undone_episode_length
                    truncate_rollout_keeping_final_score(right=right)

                if len(observations) > 0:
                    scores[-1] = final_value_net

        # If rollout is longer than the maximum episode length that we want to train on
        # then truncate to only train on the desired initial segment.
        if extra_params_cfg.max_training_episode_length is not None:
            right = extra_params_cfg.max_training_episode_length
            truncate_rollout_keeping_final_score(right=right)

        timings.start("rl.aggregate")
        # Must do before merging.
        rollout_length = len(observations)
        # Convert everything to tensors.
        observations_tensor = DataFields.cat(observations)
        scores_tensor = torch.as_tensor(scores)
        rewards_tensor = scores_tensor[-1].unsqueeze(0).repeat(rollout_length, 1)
        is_explore_tensor = torch.as_tensor(is_explore)
        done_tensor = torch.zeros(rollout_length, dtype=torch.bool)
        done_tensor[-1] = True
        # Unbind things converted to tensors
        del observations
        del scores
        del is_explore

        if extra_params_cfg.collect_search_policies:
            is_search_policy_valid_flags_tensor = torch.as_tensor(
                is_search_policy_valid_flags, dtype=torch.bool
            )
            search_policies_stacked: Dict[str, torch.Tensor] = nest.map_many(
                lambda x: torch.stack(x, 0), *search_policies
            )
            search_policy_probs = search_policies_stacked["probs"]
            search_policy_orders = search_policies_stacked["orders"]
            blueprint_probs = search_policies_stacked["blueprint_probs"]
            del is_search_policy_valid_flags
            del search_policies
        else:
            assert not is_search_policy_valid_flags
            assert not search_policies
            is_search_policy_valid_flags_tensor = torch.zeros(
                (rollout_length, 1 if is_all_powers else len(POWERS)), dtype=torch.bool
            )
            search_policy_probs = search_policy_orders = blueprint_probs = torch.zeros(
                rollout_length
            )
            del is_search_policy_valid_flags
            del search_policies

        if extra_params_cfg.collect_search_evs:
            search_policy_evs_tensor = torch.stack(search_policy_evs)
            del search_policy_evs
        else:
            assert not search_policy_evs
            assert not target_evs
            search_policy_evs_tensor = torch.zeros(rollout_length)
            del search_policy_evs

        values_datafields = agent.base_strategy_model.add_stuff_to_datafields(
            observations_tensor.copy(), has_press=has_press, agent_power=agent_power
        )
        values_tensor = torch.as_tensor(
            agent.base_strategy_model.forward_values_from_datafields(x=values_datafields)
        )
        values_tensor = values_tensor.float()

        # [False, True, False, False] -> [ True, True, False, False].
        explored_on_the_right = (
            torch.flip(torch.cumsum(torch.flip(is_explore_tensor.long(), [0]), 0), [0]) > 0
        )

        if extra_params_cfg.use_ev_targets:
            assert extra_params_cfg.collect_search_evs
            targets = torch.stack(target_evs)
            del target_evs
        else:
            targets = create_research_targets_single_rollout(
                is_explore_tensor,
                scores_tensor[-1],
                values_tensor,
                scores_tensor > 1e-3,
                extra_params_cfg.discounting,
            )
            del target_evs

        timings.stop()
        timings.pprint(logging.getLogger("timings.search_rollout").info)

        if stats_client is not None and last_ckpt_metas is not None:
            # Also here be tolerarant of the other side hanging due to things like
            # torch.distributed.init_process_group
            aggregated = _aggregate_stats(
                game, stats, list_stats, last_ckpt_metas, timings=timings
            )

            def send_to_client():
                assert stats_client is not None
                stats_client.add_stats(aggregated)

            perform_retry_loop(send_to_client, max_tries=20, sleep_seconds=10)

        game_meta["game"] = game

        research_batch = ReSearchRolloutBatch(
            observations=observations_tensor,
            rewards=rewards_tensor,
            done=done_tensor,
            is_explore=is_explore_tensor,
            explored_on_the_right=explored_on_the_right,
            scores=scores_tensor,
            values=values_tensor,
            targets=targets,
            state_hashes=torch.tensor(state_hashes, dtype=torch.long),
            years=torch.tensor(years),
            phase_type=torch.ByteTensor(phase_type),
            is_search_policy_valid=is_search_policy_valid_flags_tensor,
            search_policy_evs=search_policy_evs_tensor,
            search_policy_probs=search_policy_probs,
            search_policy_orders=search_policy_orders,
            blueprint_probs=blueprint_probs,
        )
        # _debug_print_batch(research_batch)
        for key, value in research_batch._asdict().items():
            if key != "observations":
                assert value.shape[0] == rollout_length, nest.map(
                    lambda x: x.shape, research_batch._asdict()
                )
        for _ in range(fake_gen):
            yield RolloutResult(
                batch=research_batch, game_meta=game_meta, last_ckpt_metas=last_ckpt_metas
            )


def _debug_print_batch(research_batch: ReSearchRolloutBatch):
    torch.set_printoptions(threshold=10000)
    print("rewards", research_batch.rewards)
    print("done", research_batch.done)
    print("is_explore", research_batch.is_explore)
    print("explored_on_the_right", research_batch.explored_on_the_right)
    print("scores", research_batch.scores)
    print("values", research_batch.values)
    print("targets", research_batch.targets)
    print("years", research_batch.years)
    print("phase_type", research_batch.phase_type)
    print("is_search_policy_valid", research_batch.is_search_policy_valid)


def _decide_who_explores(
    game: pydipcc.Game,
    rng: np.random.RandomState,  # type:ignore
    *,
    eval_mode: bool,
    explore_eps: float,
    explore_s1901m_eps: float,
    explore_f1901m_eps: float,
    independent_explore: bool,
    explore_all_but_one: bool,
    exploited_agent_power: Optional[Power],
) -> List[bool]:
    alive_power_ids = game.get_alive_power_ids()
    if game.current_short_phase == "S1901M" and explore_s1901m_eps > 0:
        phase_explore_eps = explore_s1901m_eps
    elif game.current_short_phase == "F1901M" and explore_f1901m_eps > 0:
        phase_explore_eps = explore_f1901m_eps
    else:
        phase_explore_eps = explore_eps

    if independent_explore:
        do_explore = [
            not eval_mode and phase_explore_eps > 0 and phase_explore_eps > rng.uniform()
            for _ in POWERS
        ]
    else:
        someone_explores = (
            not eval_mode and phase_explore_eps > 0 and phase_explore_eps > rng.uniform()
        )
        if someone_explores:
            do_explore = [True] * len(POWERS)
            if explore_all_but_one:
                do_explore[random.choice(alive_power_ids)] = False
        else:
            do_explore = [False] * len(POWERS)

    for i in range(len(POWERS)):
        # Dead never wander.
        if i not in alive_power_ids:
            do_explore[i] = False
        # When exploiting a fixed agent, that agent never explores.
        if exploited_agent_power is not None and POWERS[i] == exploited_agent_power:
            do_explore[i] = False
    return do_explore


def _set_all_logger_handlers_to_level(logger, level):
    for handler in logger.handlers[:]:
        handler.setLevel(level)


def _compute_evs_as_tensors(
    game, agent, agent_power: Optional[Power], policies, *, max_actions: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function computes EVs for FvA game and format it as 7p EVs.

    It will die if the game is not FvA.

    Returns tuple (tensor [7], tensor [7, max_actions])
    """
    ev_target_aus, ev_aus, ev_fra = compute_evs_fva(
        game, agent, agent_power, policies, min_action_prob=0.01
    )
    ev_target = ev_aus.new_zeros([len(POWERS)])
    ev_target[POWERS.index("AUSTRIA")] = ev_target_aus
    ev_target[POWERS.index("FRANCE")] = 1 - ev_target_aus

    result = ev_aus.new_full((len(POWERS), max_actions), -1)
    result[POWERS.index("AUSTRIA"), : len(ev_aus)] = ev_aus
    result[POWERS.index("FRANCE"), : len(ev_fra)] = ev_fra
    return ev_target, result


def _compute_evs_as_tensors_from_cfr_data(
    policies: PowerPolicies,
    search_result: SearchResult,
    *,
    max_actions: int,
    compute_per_power_evs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This extract EVs per action per power from CFRData.

    Returns tuple (tensor [7], tensor [7, max_actions])
    """
    ev_target = torch.zeros([len(POWERS)])
    for i, power in enumerate(POWERS):
        ev_target[i] = search_result.avg_utility(power)

    result = ev_target.new_full((len(POWERS), max_actions), -1)
    if compute_per_power_evs:
        for power_id, power in enumerate(POWERS):
            for action_id, action in enumerate(policies[power]):
                if action == tuple():
                    assert action_id == 0, action_id
                    # This is a fake no-op action for a power without real action.
                    # Skipping it altogether.
                    continue
                if action_id < max_actions:
                    result[power_id, action_id] = search_result.avg_action_utility(power, action)
    return ev_target, result


def _aggregate_stats(
    game, raw_stats: Dict, raw_list_stats: Dict, last_ckpt_metas: Dict, timings: TimingCtx
) -> Dict[str, torch.Tensor]:
    stats = {}
    # Metrics.
    if "do_attempts" in raw_stats:
        stats["rollouter/do_success_rate"] = raw_stats["do_successes"] / raw_stats["do_attempts"]
    stats["rollouter/phases"] = len(game.get_phase_history())
    for entropy_key in JOINT_ENTROPY_METRIC_NAMES:
        if raw_list_stats.get(entropy_key):
            stats[f"rollouter/{entropy_key}"] = np.mean(raw_list_stats[entropy_key])
    # Timings.
    total = sum(v for k, v in timings.items())
    for k, v in timings.items():
        stats[f"rollouter_timings/{k}"] = v
        stats[f"rollouter_timings_pct/{k}"] = v / (total + 1e-100)
    stats.update((f"rollouter/epoch_{k}", d["epoch"]) for k, d in last_ckpt_metas.items())
    # Global step.
    stats["epoch"] = max(d["epoch"] for d in last_ckpt_metas.values())
    # PostMan hack.
    stats = {k: torch.tensor(v).view(1, 1) for k, v in stats.items()}
    return stats


def truncate_policy(policy: Policy, max_items: int) -> Policy:
    if len(policy) <= max_items:
        return policy
    actions, probs = unzip2(collections.Counter(policy).most_common(max_items))
    total_prob = sum(probs)
    probs = [x / total_prob for x in probs]
    return dict(zip(actions, probs))


def _compute_action_probs(
    base_strategy_model,
    game,
    plausible_actions: PowerPolicies,
    max_actions: int,
    batch_size: int,
    half_precision: bool,
    agent_power: Optional[Power],
    has_press=False,
) -> torch.Tensor:
    power_action_pairs = []
    power_action_dicts_strings: List[Dict[Power, Action]] = []
    for power, actions in plausible_actions.items():
        power_id = POWERS.index(power)
        power_action_pairs.extend((power_id, i) for i in range(len(actions)))
        power_action_dicts_strings.extend({power: a} for a in actions)

    logprobs = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        power_action_dicts_strings,
        has_press=has_press,
        agent_power=agent_power,
        game_rating_dict=None,
        batch_size=batch_size,
    )

    prob_tensor = torch.zeros((len(POWERS), max_actions))
    for (power_id, action_id), logprob in zip(power_action_pairs, logprobs):
        prob_tensor[power_id, action_id] = math.exp(logprob)
    return prob_tensor


def _choice(rng, sequence):
    return sequence[rng.randint(0, len(sequence))]


def _random_plausible_orders(x_possible_actions, max_policy_size):
    assert x_possible_actions.shape[1] == 7, x_possible_actions.shape
    x_possible_actions = x_possible_actions.squeeze(0)
    plausible_orders = {}
    for i, power in enumerate(POWERS):
        power_possible_actions = x_possible_actions[i]
        if (power_possible_actions == -1).all():
            plausible_orders[power] = {tuple(): -1.0}
            continue
        plausible_orders[power] = {}
        for _ in range(max_policy_size):
            action = []
            for row in power_possible_actions:
                if (row == -1).all():
                    continue
                row = row[row != -1]
                action.append(ORDER_VOCABULARY[random.choice(row)])
            if len(action) == 1 and ";" in action[0]:
                action = action[0].split(";")
            plausible_orders[power][tuple(action)] = -1.0

    return plausible_orders


def average_policies(policies: List[PowerPolicies]) -> PowerPolicies:
    averaged: PowerPolicies = {}
    for power in policies[0]:
        summed = collections.defaultdict(float)
        for policy in policies:
            for a, prob in policy[power].items():
                summed[a] += prob
        averaged[power] = {a: prob / len(policies) for a, prob in summed.items()}
    return averaged


def queue_rollouts(out_queue: mp_module.Queue, log_path, log_level, need_games, **kwargs) -> None:
    if log_path:
        heyhi.setup_logging(console_level=None, fpath=log_path, file_level=log_level)
    try:
        rollout_result: RolloutResult
        for rollout_result in yield_rollouts(**kwargs):
            if need_games:
                game_meta = rollout_result.game_meta.copy()
                game_meta["game"] = game_meta["game"].to_json()
                item = (rollout_result.batch, game_meta, rollout_result.last_ckpt_metas)
            else:
                item = rollout_result.batch
            try:
                out_queue.put(item, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def _randomize_best_agent_sampling(agent_cfg):
    assert agent_cfg.which_agent == "best_agent", agent_cfg
    agent_proto = agent_cfg.to_editable()
    agent_proto.best_agent.sampling_type = [
        conf.agents_pb2.TheBestAgent.INDEPENDENT_PIKL,
        conf.agents_pb2.TheBestAgent.JOINT_CONDITIONAL,
    ][int(random.random() < 0.5)]
    return agent_proto.to_frozen()
