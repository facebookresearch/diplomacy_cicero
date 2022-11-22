#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import collections
import pathlib

import tabulate

from fairdiplomacy.utils.game_scoring import average_game_scores, GameScores
from fairdiplomacy.compare_agents_array import get_power_scores_from_folder


def print_rl_stats(power_scores, print_stderr=False, only_sos=False):
    stats_per_power = collections.defaultdict(list)
    for power, rl_stats in power_scores:
        stats_per_power[power].append(rl_stats)
        stats_per_power["_TOTAL"].append(rl_stats)
    stats_per_power = {
        power: average_game_scores(stats) for power, stats in stats_per_power.items()
    }

    cols = list(GameScores._fields)
    if only_sos:
        m = stats_per_power["_TOTAL"][0].square_score
        s = stats_per_power["_TOTAL"][1].square_score
        print(f"{m:.3f} ± {s:.3f}")
    else:
        table = [["-"] + cols]
        for power, (avgs, stderrs) in sorted(stats_per_power.items()):
            stats = stderrs if print_stderr else avgs
            table.append([power[:3]] + [getattr(stats, col) for col in cols])
        print(tabulate.tabulate(table, headers="firstrow"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs", type=pathlib.Path, nargs="+", help="Directories containing game.json files"
    )
    parser.add_argument("--stderr", action="store_true", default=False)
    parser.add_argument("--only-sos", action="store_true", default=False)
    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="If true, then accumulates statistics for *all* agents, not just agent_one. This is useful to see e.g. draw/solo stats in shared_agent games.",
    )
    args = parser.parse_args()

    power_scores = sum(
        [get_power_scores_from_folder(d, all_agents=args.all_agents) for d in args.results_dirs],
        [],
    )
    print_rl_stats(power_scores, print_stderr=args.stderr, only_sos=args.only_sos)
