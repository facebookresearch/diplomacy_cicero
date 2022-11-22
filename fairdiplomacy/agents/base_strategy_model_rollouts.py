#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

import logging
import numpy as np
import torch
from typing import List, Tuple

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import n_move_phases_later

from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import (
    JointAction,
    JointActionValues,
    PlayerRating,
    Power,
    RolloutResults,
)
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.yearprob import (
    get_prob_of_latest_year_leq,
    parse_year_spring_prob_of_ending,
)

T = TypeVar("T")


class BaseStrategyModelRollouts:
    def __init__(
        self,
        base_strategy_model: BaseStrategyModelWrapper,
        cfg: agents_cfgs.BaseStrategyModelRollouts,
        has_press: bool,
        set_player_ratings: bool = False,
    ):
        self.base_strategy_model = base_strategy_model
        self.feature_encoder = FeatureEncoder(num_threads=cfg.n_threads)

        assert cfg.temperature >= 0, "Set rollout_cfg.temperature"
        assert cfg.max_rollout_length >= 0, "Set rollout_cfg.max_rollout_length"

        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_rollout_length = cfg.max_rollout_length
        self.mix_square_ratio_scoring = cfg.mix_square_ratio_scoring
        self.clear_old_all_possible_orders = cfg.clear_old_all_possible_orders
        self.average_n_rollouts = cfg.average_n_rollouts
        self.has_press = has_press

        self.set_player_ratings = set_player_ratings
        self.use_player_ratings = (
            self.base_strategy_model.model.use_player_ratings
            if hasattr(self.base_strategy_model.model, "use_player_ratings")
            else False
        )

        self.use_agent_power = (
            self.base_strategy_model.model.use_agent_power
            if hasattr(self.base_strategy_model.model, "use_agent_power")
            else False
        )

        self.year_spring_prob_of_ending = parse_year_spring_prob_of_ending(
            cfg.year_spring_prob_of_ending
        )
        self.has_year_spring_prob_of_ending = self.year_spring_prob_of_ending is not None

        if self.set_player_ratings:
            assert (
                self.use_player_ratings
            ), "BaseStrategyModel model needs to be trained with player ratings if player ratings is set"

    def get_prob_of_spring_ending(self, year: int) -> float:
        """Return the probability that we are assuming that the game ends after spring of this year
        when computing expected values of rollouts."""
        if self.year_spring_prob_of_ending is not None:
            return get_prob_of_latest_year_leq(self.year_spring_prob_of_ending, year)
        return 0.0

    def do_rollouts(
        self,
        game_init,
        *,
        agent_power: Optional[Power],
        set_orders_dicts: List[JointAction],
        player_ratings: Optional[List[PlayerRating]] = None,
        timings=None,
        log_timings=False,
    ) -> RolloutResults:
        if timings is None:
            timings = TimingCtx()
        # Shape: [num_order_sets, 7, 1].
        scores = self.do_rollouts_multi(
            game_init=game_init,
            agent_power=agent_power,
            set_orders_dicts=set_orders_dicts,
            player_ratings=player_ratings,
            extra_base_strategy_models=None,
            timings=timings,
            log_timings=False,
        )
        # Shape: [num_order_sets, 7].
        scores = scores.squeeze(-1).numpy()
        with timings("unpack"):
            r = [
                (set_orders_dict, dict(zip(POWERS, scores_array)))
                for set_orders_dict, scores_array in zip(set_orders_dicts, scores)
            ]
        if log_timings:
            timings.pprint(logging.getLogger("timings").info)
        return r

    def do_rollouts_multi(
        self,
        game_init,
        *,
        agent_power: Optional[Power],
        set_orders_dicts: List[JointAction],
        player_ratings: Optional[List[PlayerRating]] = None,
        extra_base_strategy_models: Optional[List[BaseStrategyModelWrapper]] = None,
        override_max_rollout_length: Optional[int] = None,
        timings=None,
        log_timings=False,
    ) -> torch.Tensor:
        """Computes actions of state-action pairs for a bunch of value functions.

        By default compute values only for the self.base_strategy_model value function.
        Rollouts are only performed using self.base_strategy_model value function.

        Returns array of shape [len(set_orders_dicts), num_powers, num_value_functions].
        """

        all_value_functions = [self.base_strategy_model] + (
            [] if extra_base_strategy_models is None else extra_base_strategy_models
        )

        if timings is None:
            timings = TimingCtx()

        if self.clear_old_all_possible_orders:
            with timings("clear_old_orders"):
                game_init = pydipcc.Game(game_init)
                game_init.clear_old_all_possible_orders()
        with timings("clone"):
            games = game_init.clone_n_times(len(set_orders_dicts) * self.average_n_rollouts)
        with timings("setup"):
            game_ids = [game.game_id for game in games]

            # set orders if specified
            for game, set_orders_dict in zip(
                games, repeat(set_orders_dicts, self.average_n_rollouts)
            ):
                for power, orders in set_orders_dict.items():
                    game.set_orders(power, list(orders))

            # for each game, a list of powers whose orders need to be generated
            # by the model on the first phase.
            missing_start_orders = {
                game.game_id: frozenset(p for p in POWERS if p not in set_orders_dict)
                for game, set_orders_dict in zip(
                    games, repeat(set_orders_dicts, self.average_n_rollouts)
                )
            }

            # Construct game_id -> player_rating dict
            if self.set_player_ratings:
                assert player_ratings is not None, "Player ratings have not been provided"
                assert len(set_orders_dicts) == len(player_ratings)
                game_rating_dict = dict(
                    zip(
                        game_ids,
                        [
                            ptype
                            for ptype in player_ratings
                            for _ in range(self.average_n_rollouts)
                        ],
                    )
                )
            else:
                game_rating_dict = None

        max_rollout_length = (
            override_max_rollout_length
            if override_max_rollout_length is not None
            else self.max_rollout_length
        )
        if max_rollout_length > 0:
            rollout_end_phase_id = sort_phase_key(
                n_move_phases_later(game_init.current_short_phase, max_rollout_length)
            )
            max_steps = 1000000
        else:
            # Really far ahead.
            rollout_end_phase_id = sort_phase_key(
                n_move_phases_later(game_init.current_short_phase, 10)
            )
            max_steps = 1

        # Accumulates the expected value contribution of the game ending early in spring
        # and being scored immediately (already multiplied by the probability of it happening)
        spring_ending_ev = torch.zeros((len(games), len(POWERS), 1))
        # Accumulates the probability of ending in spring and being scored immediately.
        cumulative_spring_ending_prob = torch.zeros((len(games), 1, 1))

        # This loop steps the games until one of the conditions is true:
        #   - all games are done
        #   - at least one game was stepped for max_steps steps
        #   - all games are either completed or reach a phase such that
        #     sort_phase_key(phase) >= rollout_end_phase_id
        for step_id in range(max_steps):
            ongoing_game_phases = [
                game.current_short_phase for game in games if not game.is_game_done
            ]

            if len(ongoing_game_phases) == 0:
                # all games are done
                break

            # step games together at the pace of the slowest game, e.g. process
            # games with retreat phases alone before moving on to the next move phase
            min_phase = min(ongoing_game_phases, key=sort_phase_key)

            if sort_phase_key(min_phase) >= rollout_end_phase_id:
                break

            # Processing the spring movement phase of a year
            if step_id > 0 and min_phase.startswith("S") and min_phase.endswith("M"):
                year = int(min_phase[1:-1])
                end_prob = self.get_prob_of_spring_ending(year)
                if end_prob > 0.0:
                    # Because we aren't filtering by game.is_game_done, this will also get scores
                    # for games that stopped early due to being completed,
                    # but this is fine since it will just be averaging in the same score again.
                    # Shape: [num_games, num_powers, 1]
                    scores = torch.FloatTensor([game.get_scores() for game in games]).unsqueeze(-1)
                    spring_ending_ev += scores * (end_prob * (1.0 - cumulative_spring_ending_prob))
                    cumulative_spring_ending_prob += end_prob * (
                        1.0 - cumulative_spring_ending_prob
                    )

            games_to_step = [
                game
                for game in games
                if not game.is_game_done and game.current_short_phase == min_phase
            ]

            if step_id > 0 or any(missing_start_orders.values()):
                games_to_step_rating_dict = (
                    {game.game_id: game_rating_dict[game.game_id] for game in games_to_step}
                    if game_rating_dict is not None
                    else None
                )
                batch_orders, _logprobs = self.base_strategy_model.forward_policy(
                    games_to_step,
                    has_press=self.has_press,
                    agent_power=agent_power,
                    game_rating_dict=games_to_step_rating_dict,
                    feature_encoder=self.feature_encoder,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timings=timings,
                )

                with timings("env.set_orders"):
                    assert len(games_to_step) == len(batch_orders)
                    for game, orders_per_power in zip(games_to_step, batch_orders):
                        for power, orders in zip(POWERS, orders_per_power):
                            if step_id == 0 and power not in missing_start_orders[game.game_id]:
                                continue
                            game.set_orders(power, list(orders))

            with timings("env.step"):
                self.feature_encoder.process_multi([game for game in games_to_step])

        # Shape: [num_games, num_powers, num_value_functions].
        final_scores = torch.zeros((len(games), len(POWERS), len(all_value_functions)))

        # Compute SoS for done game and query the net for not-done games.
        not_done_games = [game for game in games if not game.is_game_done]
        if not_done_games:
            timings.start("encoding")
            not_done_game_rating_dict = (
                {game.game_id: game_rating_dict[game.game_id] for game in not_done_games}
                if game_rating_dict is not None
                else None
            )
            # Note, we assume that all base_strategy_models share settings of the first wrapper.
            value_net_inputs = self.base_strategy_model.create_datafield_for_values(
                not_done_games,
                game_rating_dict=not_done_game_rating_dict,
                has_press=self.has_press,
                agent_power=agent_power,
                feature_encoder=self.feature_encoder,
            )
            timings.start("v_model")
            # Shape: [num_undone_games, num_values]
            final_scores_per_base_strategy_model = torch.stack(
                [
                    base_strategy_model.forward_values_from_datafields(value_net_inputs)
                    for base_strategy_model in all_value_functions
                ],
                -1,
            )
            not_done_games_mask = torch.BoolTensor([not game.is_game_done for game in games])
            # Extra float() to handle half().
            final_scores[not_done_games_mask] = final_scores_per_base_strategy_model.float().cpu()

        timings.start("final_scores")
        for i, game in enumerate(games):
            if game.is_game_done:
                final_scores[i] = torch.FloatTensor(game.get_scores()).unsqueeze(-1)

        # mix in current sum of squares ratio to encourage losing powers to try hard
        # get GameScores objects for current game state
        if self.mix_square_ratio_scoring > 0:
            # Shape: [num_games, num_powers, 1]
            sos_scores = torch.FloatTensor([game.get_scores() for game in games]).unsqueeze(-1)
            final_scores = (1 - self.mix_square_ratio_scoring) * final_scores + (
                self.mix_square_ratio_scoring * sos_scores
            )

        if self.has_year_spring_prob_of_ending:
            # For the final game positions, they may be on different phases, check each one individually
            final_spring_ending_ev = np.zeros((len(games), len(POWERS)))
            final_spring_ending_prob = np.zeros((len(games),))
            for i, game in enumerate(games):
                game_current_phase = game.current_short_phase
                if game_current_phase.startswith("S") and game_current_phase.endswith("M"):
                    p = self.get_prob_of_spring_ending(game.current_year)
                    final_spring_ending_ev[i, :] = [score * p for score in game.get_scores()]
                    final_spring_ending_prob[i] = p

            spring_ending_ev += torch.FloatTensor(final_spring_ending_ev).unsqueeze(-1)
            cumulative_spring_ending_prob += (
                torch.FloatTensor(final_spring_ending_prob).unsqueeze(-1).unsqueeze(-1)
            )

            final_scores = (1.0 - cumulative_spring_ending_prob) * final_scores + spring_ending_ev

        if self.average_n_rollouts != 1:
            final_scores = final_scores.view(
                (len(set_orders_dicts), -1, len(POWERS), len(all_value_functions))
            ).mean(1)

        timings.stop()

        if log_timings:
            timings.pprint(logging.getLogger("timings").info)

        return final_scores

    def override_has_press(self, has_press: bool):
        self.has_press = has_press

    def do_rollouts_maybe_cached(
        self,
        game: pydipcc.Game,
        *,
        agent_power: Optional[Power],
        set_orders_dicts: List[JointAction],
        cache: Optional["RolloutResultsCache"],
        timings: Optional[TimingCtx] = None,
        player_rating: Optional[PlayerRating] = None,
    ) -> List[Tuple[JointAction, JointActionValues]]:
        """A version of do_rollouts that may try to use the cache if provided."""

        def on_miss(
            set_orders_dicts: List[JointAction],
        ) -> List[Tuple[JointAction, JointActionValues]]:
            nonlocal timings
            inner_timings = TimingCtx()
            if player_rating is not None:
                player_ratings = [player_rating] * len(set_orders_dicts)
            else:
                player_ratings = None
            result = self.do_rollouts(
                game,
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                timings=inner_timings,
                player_ratings=player_ratings,
            )
            if timings is not None:
                timings += inner_timings
            return result

        all_rollout_results = (
            cache.get(set_orders_dicts, on_miss)
            if cache is not None
            else on_miss(set_orders_dicts)
        )

        return all_rollout_results

    def do_rollouts_multi_maybe_cached(
        self,
        game: pydipcc.Game,
        *,
        agent_power: Optional[Power],
        set_orders_dicts: List[JointAction],
        cache: Optional["RolloutResultsCache"],
        extra_base_strategy_models: Optional[List[BaseStrategyModelWrapper]] = None,
        timings: Optional[TimingCtx] = None,
        player_rating: Optional[PlayerRating] = None,
    ) -> torch.Tensor:
        """A version of do_rollouts_multi that may try to use the cache if provided."""

        def on_miss(set_orders_dicts: List[JointAction]) -> torch.Tensor:
            nonlocal timings
            if player_rating is not None:
                player_ratings = [player_rating] * len(set_orders_dicts)
            else:
                player_ratings = None
            result = self.do_rollouts_multi(
                game,
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                timings=timings,
                player_ratings=player_ratings,
                extra_base_strategy_models=extra_base_strategy_models,
            )
            return result

        if timings is None:
            timings = TimingCtx()
        if cache is not None:
            timings.start("cache")

        all_rollout_results = (
            cache.get_multi(set_orders_dicts, on_miss)
            if cache is not None
            else on_miss(set_orders_dicts)
        )

        return all_rollout_results

    @staticmethod
    def build_cache() -> "RolloutResultsCache":
        return RolloutResultsCache()


def repeat(seq, n):
    """Yield each element in seq 'n' times"""
    for e in seq:
        for _ in range(n):
            yield e


def groups_of(seq, n):
    """Yield len(seq)/n groups of `n` elements each"""
    for i in range(0, len(seq), n):
        yield seq[i : (i + n)]


class RolloutResultsCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.calls = 0

    def _get(
        self,
        set_orders_dicts: List[JointAction],
        onmiss_fn: Callable[[List[JointAction]], Iterable[T]],
    ) -> List[T]:
        joint_actions = tuple(frozenset(d.items()) for d in set_orders_dicts)
        n_unique = len(frozenset(joint_actions))
        self.calls += n_unique

        unknown_order_dicts = [
            set_orders_dicts[i]
            for i, joint_action in enumerate(joint_actions)
            if joint_action not in self.cache
        ]
        # Minor optimization. Orders may have duplicates.
        unknown_order_dicts = list({frozenset(x.items()): x for x in unknown_order_dicts}.values())
        self.hits += n_unique - len(unknown_order_dicts)
        for set_order_dict, r in zip(unknown_order_dicts, onmiss_fn(unknown_order_dicts)):
            joint_action = frozenset(set_order_dict.items())
            self.cache[joint_action] = r
        results = [self.cache[joint_action] for joint_action in joint_actions]
        return results

    def get(
        self,
        set_orders_dicts: List[JointAction],
        onmiss_fn: Callable[[List[JointAction]], RolloutResults],
    ) -> RolloutResults:
        return self._get(set_orders_dicts, onmiss_fn)

    def get_multi(
        self,
        set_orders_dicts: List[JointAction],
        onmiss_fn: Callable[[List[JointAction]], torch.Tensor],
    ) -> torch.Tensor:
        return torch.stack(self._get(set_orders_dicts, onmiss_fn), 0)

    def __repr__(self):
        return "RolloutResultsCache[hits/calls = {} / {} = {:.3f}]".format(
            self.hits, self.calls, 0 if self.calls == 0 else self.hits / self.calls
        )
