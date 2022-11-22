#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import collections
from functools import partial
import logging
from fairdiplomacy.utils.analysis import average_value_by_phase
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from fairdiplomacy.compare_agent_population_array import (
    get_games_from_folder,
    get_power_scores_from_folder,
    get_power_scores_from_folder_grouped_by_game,
)
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import GameScores, average_game_scores
from tabulate import tabulate

ELO_PER_LOGGAMMA = 400.0 * math.log10(math.exp(1))


def _print_stats(
    power_scores: List[Tuple[Power, str, GameScores]],
    power_scores_grouped_by_game: List[List[Tuple[Power, str, GameScores]]],
    compute_elos: bool,
):
    stats_per_agent = collections.defaultdict(list)
    stats_per_power = collections.defaultdict(list)
    # (pwr, agent_name, game_scores)
    for power, agent_name, game_scores in power_scores:
        stats_per_agent[agent_name].append(game_scores)
        stats_per_power[power].append(game_scores)
    stats_per_agent = {
        agent: average_game_scores(game_scores) for agent, game_scores in stats_per_agent.items()
    }
    stats_per_power = {
        power: average_game_scores(game_scores) for power, game_scores in stats_per_power.items()
    }

    print(f"Number of games found: {len(power_scores_grouped_by_game)}")
    print("#" * 60)
    print("Average square scores:")
    print("#" * 60)
    stats_per_agent = dict(sorted(stats_per_agent.items(), key=lambda x: -x[1][0].square_score))
    for agent_name in stats_per_agent.keys():
        m = stats_per_agent[agent_name][0].square_score
        s = stats_per_agent[agent_name][1].square_score
        print(f"{agent_name:15s} = {m*100:4.1f}% ± {s*100:.1f}% (one stderr)")
    print("#" * 60)
    for power in stats_per_power.keys():
        m = stats_per_power[power][0].square_score
        s = stats_per_power[power][1].square_score
        print(f"{power:15s} = {m*100:4.1f}% ± {s*100:.1f}% (one stderr)")
    print("#" * 60)
    if not compute_elos:
        return

    agents_array = list(stats_per_agent.keys())
    num_agents = len(agents_array)
    num_players_per_game = len(power_scores_grouped_by_game[0])

    # Replace agent names with integers to get faster indexing
    power_scores_grouped_by_game_indexed = []
    for power_scores_this_game in power_scores_grouped_by_game:
        assert (
            len(power_scores_this_game) == num_players_per_game
        ), "Different games have different numbers of players!"
        power_scores_this_game_indexed = []
        for power, agent_name, game_scores in power_scores_this_game:
            power_scores_this_game_indexed.append(
                (POWERS.index(power), agents_array.index(agent_name), game_scores)
            )
        power_scores_grouped_by_game_indexed.append(power_scores_this_game_indexed)

    # Variables for fitting a multiplayer Elo model
    # Baseline strength of each player
    player_loggammas = torch.zeros((num_agents), dtype=torch.float32, requires_grad=True)
    # Baseline strength of each power
    power_loggammas = torch.zeros((len(POWERS)), dtype=torch.float32, requires_grad=True)

    def get_power_loggammas_zeromean():
        power_loggammas_zeromean = power_loggammas - torch.mean(power_loggammas)
        return power_loggammas_zeromean

    # Cross terms that measure how much each agent type
    # benefits from each other agent type being in the same game.
    player_cologgammas = torch.zeros(
        (num_agents, num_agents), dtype=torch.float32, requires_grad=True
    )

    def get_cologgammas_zeromean():
        player_cologgammas_zeromean = player_cologgammas - torch.mean(
            player_cologgammas, dim=1, keepdim=True
        )
        return player_cologgammas_zeromean

    # Cross-terms that scale the cologgammas based on the specific pair of powers,
    # intended to represent how much each pair of powers cares about the other.
    power_couplings_logits = torch.zeros(
        (len(POWERS), len(POWERS)), dtype=torch.float32, requires_grad=True
    )

    def get_power_couplings():
        couplings = power_couplings_logits
        # Force to be symmetric.
        couplings = 0.5 * (couplings + torch.transpose(couplings, 0, 1))
        # Convert logits to factors
        couplings = torch.reshape(
            F.softmax(torch.reshape(couplings, [-1]), dim=0), [len(POWERS), len(POWERS)]
        )
        diag = torch.eye(len(POWERS), dtype=torch.float32)
        off_diag = 1.0 - diag
        # Normalize by the mean of all off-diagonal entries
        couplings = couplings / (torch.sum(couplings * off_diag) / torch.sum(off_diag))
        # Set the diagonal to 1
        couplings = couplings * off_diag + diag
        return couplings

    def compute_loss():
        power_loggammas_zeromean = get_power_loggammas_zeromean()
        player_cologgammas_zeromean = get_cologgammas_zeromean()
        power_couplings = get_power_couplings()

        log_likelihoods = []
        for power_scores_this_game_indexed in power_scores_grouped_by_game_indexed:
            len_scores = len(power_scores_this_game_indexed)

            # First, compute the loggamma strength of each player
            all_loggammas = []
            for i in range(len_scores):
                loggammas_to_add = []
                (poweri, agenti, _) = power_scores_this_game_indexed[i]
                loggammas_to_add.append(player_loggammas[agenti])
                loggammas_to_add.append(power_loggammas_zeromean[poweri])
                for j in range(len_scores):
                    if i != j:
                        (powerj, agentj, _) = power_scores_this_game_indexed[j]
                        loggammas_to_add.append(
                            power_couplings[poweri, powerj]
                            * player_cologgammas_zeromean[agenti, agentj]
                        )
                all_loggammas.append(torch.sum(torch.stack(loggammas_to_add)))

            # The Elo model says that the probability a player i wins i
            # exp(loggamma_i) / sum_j (exp(logamma_j))
            # Taking logs, we hae that the log probability that player i wins is
            # loggamma_i - log(sum_j (exp(logamma_j)))
            #
            # total_resistance is the second term.
            total_resistance = torch.logsumexp(torch.stack(all_loggammas), dim=0)

            # For each player that won or participated in a draw, add in the
            # weighted term for that player
            for i in range(len_scores):
                (_, _, game_scores) = power_scores_this_game_indexed[i]
                square_score = game_scores.square_score
                if square_score > 0.0:
                    assert square_score <= 1.0
                    myself_loggamma = all_loggammas[i]
                    assert len(myself_loggamma.shape) == 0
                    log_likelihood_of_win = myself_loggamma - total_resistance
                    log_likelihoods.append(square_score * log_likelihood_of_win)
        # Add it all up!
        log_likelihood = torch.sum(torch.stack(log_likelihoods))

        # Add a gaussian prior that each player and power loggamma is 0, with stdev 2. Very weak prior.
        loggamma_prior_stdev = 2.0
        loggamma_prior_loss = (0.5 / loggamma_prior_stdev / loggamma_prior_stdev) * torch.sum(
            player_loggammas * player_loggammas
        )

        # Add a gaussian prior that power loggamma is 0, with stdev 1.
        power_loggamma_prior_stdev = 1.0
        power_loggamma_prior_loss = (
            0.5 / power_loggamma_prior_stdev / power_loggamma_prior_stdev
        ) * torch.sum(power_loggammas * power_loggammas)

        # Add a gaussian prior that each cologgamma is 0, with stdev 0.3333333.
        cologgamma_prior_stdev = 0.3333333
        cologgamma_prior_loss = (
            0.5 / cologgamma_prior_stdev / cologgamma_prior_stdev
        ) * torch.sum(player_cologgammas * player_cologgammas)

        # Add a gaussian prior that each power coupling logfactor is 0, with stdev 0.3333333
        power_coupling_logfactor_prior_stdev = 0.3333333
        power_coupling_logfactor_prior_loss = (
            0.5 / power_coupling_logfactor_prior_stdev / power_coupling_logfactor_prior_stdev
        ) * torch.sum(torch.log(power_couplings) * torch.log(power_couplings))

        loss = (
            -log_likelihood
            + loggamma_prior_loss
            + power_loggamma_prior_loss
            + cologgamma_prior_loss
            + power_coupling_logfactor_prior_loss
        )
        return (
            loss,
            loggamma_prior_loss,
            power_loggamma_prior_loss,
            cologgamma_prior_loss,
            power_coupling_logfactor_prior_loss,
        )

    print("Beginning cross-player Elo model fitting")
    print(f"{num_players_per_game} players per game")
    optimizer = torch.optim.Adam(
        [player_loggammas, power_loggammas, player_cologgammas, power_couplings_logits], lr=0.2
    )
    for lr in [0.200, 0.075, 0.025]:
        print(f"Setting lr {lr}")
        for g in optimizer.param_groups:
            g["lr"] = lr
        for iteration in range(60):
            optimizer.zero_grad()
            (
                loss,
                loggamma_prior_loss,
                power_loggamma_prior_loss,
                cologgamma_prior_loss,
                power_coupling_logfactor_prior_loss,
            ) = compute_loss()

            prev_player_loggammas = player_loggammas.detach().numpy().copy()
            prev_power_loggammas = power_loggammas.detach().numpy().copy()
            prev_cologgammas = player_cologgammas.detach().numpy().copy()
            loss.backward()
            optimizer.step()
            cur_player_loggammas = player_loggammas.detach().numpy()
            cur_power_loggammas = power_loggammas.detach().numpy()
            cur_cologgammas = player_cologgammas.detach().numpy()
            max_elo_change = ELO_PER_LOGGAMMA * np.max(
                np.abs(cur_player_loggammas - prev_player_loggammas)
            )
            max_power_elo_change = ELO_PER_LOGGAMMA * np.max(
                np.abs(cur_power_loggammas - prev_power_loggammas)
            )
            max_co_elo_change = ELO_PER_LOGGAMMA * np.max(
                np.abs(cur_cologgammas - prev_cologgammas)
            )
            max_elo_param_change = max(max_elo_change, max_power_elo_change, max_co_elo_change)
            print(
                f"Iteration {iteration}, loss {loss.item()}, max-elo-param-change {max_elo_param_change:.2f}, prior losses {loggamma_prior_loss.item():.6f}, {power_loggamma_prior_loss.item():.6f}, {cologgamma_prior_loss.item():.6f}, {power_coupling_logfactor_prior_loss.item():.6f}"
            )
    print("Done fitting cross-player Elo model")

    elos: np.ndarray = ELO_PER_LOGGAMMA * player_loggammas.detach().numpy().copy()
    power_elos: np.ndarray = ELO_PER_LOGGAMMA * get_power_loggammas_zeromean().detach().numpy().copy()
    coelos: np.ndarray = ELO_PER_LOGGAMMA * get_cologgammas_zeromean().detach().numpy().copy()
    # Also zero-mean the columns of coelos and record that as "average impact"
    impact = np.mean(coelos, axis=0, keepdims=True)
    coelos = coelos - impact
    power_couplings: np.ndarray = get_power_couplings().detach().numpy().copy()

    print(
        tabulate(
            [(agents_array[i], elos[i], impact[0, i]) for i in range(num_agents)],
            headers=["Agent", "Baseline Elo, Avg impact"],
            floatfmt=".1f",
        )
    )
    print("Elo adjustment based on which power an agent is playing")
    print(
        tabulate(
            [(POWERS[i], power_elos[i]) for i in range(len(POWERS))],
            headers=["Power", "Elo Adjustment"],
            floatfmt=".1f",
        )
    )
    print(
        "Elo gained by row agent per OTHER instance of column agent, before multiplying by power coupling"
    )
    print(
        tabulate(
            [[agents_array[i]] + coelos[i].tolist() for i in range(num_agents)],
            headers=[""] + agents_array,
            floatfmt=".1f",
        )
    )
    print("Power coupling factors for multiplying the above table by")
    print(
        tabulate(
            [[POWERS[i]] + power_couplings[i].tolist() for i in range(len(POWERS))],
            headers=[""] + POWERS,
            floatfmt=".2f",
        )
    )


def _print_stats_by_year(
    games_and_power_mappings: List[Tuple[Game, Dict[Power, str]]], model_paths: List[pathlib.Path],
):
    base_strategy_models = [
        BaseStrategyModelWrapper(model_path, device="cuda:0", max_batch_size=700)
        for model_path in model_paths
    ]
    all_agents = set()
    for _, mapping in games_and_power_mappings:
        all_agents = all_agents.union(set(mapping.values()))
    all_agents = sorted(list(all_agents))

    for agent_name in all_agents:
        games_and_powers = []
        for game, mapping in games_and_power_mappings:
            powers = [power for power in POWERS if mapping[power] == agent_name]
            if powers:
                games_and_powers.append((game, powers))

        headers = []
        values_by_phase = collections.defaultdict(list)

        for base_strategy_model in base_strategy_models:
            value_by_phase, value_by_power_by_phase = average_value_by_phase(
                games_and_powers,
                model=base_strategy_model,
                movement_only=True,
                spring_only=False,
                up_to_year=None,
                has_press=False,
            )

            headers = headers + POWERS + ["TOTAL"]
            for phase in value_by_phase:
                values_by_phase[phase].extend(
                    [value_by_power_by_phase[phase][power] for power in POWERS]
                    + [value_by_phase[phase]]
                )
        logging.info(
            "\nAGENT: "
            + agent_name
            + "\n"
            + tabulate(
                ([phase] + values for phase, values in values_by_phase.items()),
                headers=["phase"] + headers,
                floatfmt=".3f",
            )
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs", type=pathlib.Path, nargs="+", help="Directories containing game.json files"
    )
    parser.add_argument(
        "--with_base_strategy_model",
        type=pathlib.Path,
        action="append",
        help="Look at each agent's advantage level by phase using this value model.",
    )
    parser.add_argument("--include_partial", action="store_true", help="Include partial games.")
    parser.add_argument(
        "--compute_elos", action="store_true", help="Performs and prints out cross-elo computation"
    )

    args = parser.parse_args()
    print(args.with_base_strategy_model)
    if args.with_base_strategy_model:
        logging.info("Loading games...")
        games_and_power_mappings = sum(
            map(
                partial(get_games_from_folder, include_partial=args.include_partial),
                args.results_dirs,
            ),
            [],
        )
        logging.info(f"Loaded {len(games_and_power_mappings)} games...")
        _print_stats_by_year(games_and_power_mappings, model_paths=args.with_base_strategy_model)
    else:
        power_scores = sum(
            map(
                partial(get_power_scores_from_folder, include_partial=args.include_partial),
                args.results_dirs,
            ),
            [],
        )
        power_scores_grouped_by_game = sum(
            map(get_power_scores_from_folder_grouped_by_game, args.results_dirs), []
        )
        _print_stats(power_scores, power_scores_grouped_by_game, compute_elos=args.compute_elos)
