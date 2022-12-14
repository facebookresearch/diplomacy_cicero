#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import tabulate
import torch

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.order_idxs import strip_coasts


def compute_xpower_supports_from_saved(path, max_year=None, cf_agent=None):
    """Computes cross-power supports from a JSON file in fairdiplomacy saved game format."""
    with open(path) as f:
        game = Game.from_json(f.read())

    return compute_xpower_supports(
        game, max_year=max_year, cf_agent=cf_agent, name=os.path.basename(path)
    )


def mean(L):
    return sum(L) / len(L)


@dataclass
class XPowerSupportStats:
    name: str
    num_orders: int = 0
    num_supports: int = 0  # total number of supports
    num_move: int = 0  # num support-move
    num_xpower: int = 0  # num xpower supports
    num_eff: int = 0  # num 'effective' xpower supports
    num_xpower_move: int = 0  # num xpower support-move
    num_xpower_move_coord: int = 0  # num coordinated xpower support-move (i.e. the underlying move was taken)


def compute_xpower_supports(
    game: Game,
    max_year: Optional[str] = None,
    cf_agent: Optional[BaseAgent] = None,
    only_power: Optional[str] = None,
    name: str = "",
) -> XPowerSupportStats:
    """Computes average cross-power supports for an entire game.

        Arguments:
        - game: a fairdiplomacy.Game object
        - max_year: If set, only compute cross-power supports up to this year.
        - cf_agent: If set, look at supports orders generated by `cf_agent`,  in the context of the game state
                    and game orders generated from the other powers.
        - only_power: if set, will use only orders for the power.
        - name: a label for this game, returned in the output.

        Returns a XPowerSupportStats of statistics for this game.
    """

    ret = XPowerSupportStats(name=name)

    for phase in game.get_phase_history():
        state = phase.state
        if phase.name[1:-1] == max_year:
            break
        if not phase.name.endswith("M"):
            # This is required as, e.g., in retreat phase a single location
            # could be occupied by several powers and everything is weird.
            continue
        loc_power = {
            unit.split()[1]: power for power, units in state["units"].items() for unit in units
        }
        # If power owns, e.g., BUL/SC, make it also own BUL. Support targets do
        # not use "/SC" so we need both.
        for loc, power in list(loc_power.items()):
            if "/" in loc:
                loc_land = loc.split("/")[0]
                assert loc_land not in loc_power, (loc_land, loc_power)
                loc_power[loc_land] = power

        if cf_agent is not None:
            phase_orders = cf_agent.get_orders_many_powers(
                game.rolled_back_to_phase_end(phase.name), powers=POWERS
            )
        else:
            phase_orders = phase.orders

        all_orders = [o for a in phase_orders.values() for o in a]
        all_order_roots = {strip_coasts(order).replace(" VIA", "") for order in all_orders}
        for power, power_orders in phase_orders.items():
            if only_power and power != only_power:
                continue

            for order in power_orders:
                ret.num_orders += 1
                order_tokens = order.split()
                is_support = (
                    len(order_tokens) >= 5
                    and order_tokens[2] == "S"
                    and order_tokens[3] in ("A", "F")
                )
                if not is_support:
                    continue
                ret.num_supports += 1
                src = order_tokens[4]
                if src not in loc_power:
                    torch.save(
                        dict(
                            game=game.to_json(),
                            power=power,
                            pwer_orders=power_orders,
                            order=order,
                            phase=phase,
                            loc_power=loc_power,
                        ),
                        "xpower_debug.pt",
                    )
                    raise RuntimeError(f"{order}: {src} not in {loc_power}")
                is_support_move = len(order_tokens) >= 6 and order_tokens[5] == "-"
                if is_support_move:
                    ret.num_move += 1

                if loc_power[src] == power:
                    continue

                ret.num_xpower += 1

                cf_states = []
                for do_support in (False, True):
                    g_cf = game.rolled_back_to_phase_end(phase.name)
                    g_cf.set_orders(power, power_orders)

                    assert g_cf.current_short_phase == phase.name
                    if not do_support:
                        hold_order = " ".join(order_tokens[:2] + ["H"])
                        g_cf.set_orders(power, [hold_order])

                    g_cf.process()
                    assert g_cf.current_short_phase != phase.name
                    s = g_cf.get_state()
                    cf_states.append((s["name"], s["units"], s["retreats"]))

                if cf_states[0] != cf_states[1]:
                    ret.num_eff += 1

                if is_support_move:
                    ret.num_xpower_move += 1

                    underlying_order = " ".join(order_tokens[3:])
                    if underlying_order in all_order_roots:
                        ret.num_xpower_move_coord += 1

    return ret


def compute_xpower_statistics(paths, max_year=None, num_jobs=1, cf_agent=None):

    if num_jobs == 1 or cf_agent is not None:
        # if running with CF-agent, can't use multiple cores (bc of model)
        stats = [
            compute_xpower_supports_from_saved(path, max_year=max_year, cf_agent=cf_agent)
            for path in paths
        ]
    else:
        stats = joblib.Parallel(num_jobs)(
            joblib.delayed(compute_xpower_supports_from_saved)(
                path, max_year=max_year, cf_agent=cf_agent
            )
            for path in paths
        )

    print(
        tabulate.tabulate(
            [
                (
                    s.name,
                    s.num_supports,
                    s.num_xpower,
                    s.num_eff,
                    s.num_xpower_move,
                    s.num_xpower_move_coord,
                )
                for s in stats[:10]
            ],
            headers=(
                "name",
                "supports",
                "xpower",
                "effective",
                "xpower_move",
                "xpower_move_coord",
            ),
        )
    )
    print("...\n")

    x_support_ratio = mean([s.num_xpower / s.num_supports for s in stats if s.num_supports > 0])
    eff_x_support_ratio = mean([s.num_eff / s.num_xpower for s in stats if s.num_xpower > 0])
    x_support_move_ratio = mean([s.num_xpower_move / s.num_move for s in stats if s.num_move > 0])
    coord_x_move_ratio = mean(
        [s.num_xpower_move_coord / s.num_xpower_move for s in stats if s.num_xpower_move > 0]
    )

    print(
        f"{len(paths)} games; x_support= {x_support_ratio:.4f}  eff_x_support= {eff_x_support_ratio:.4f} x_sup_move= {x_support_move_ratio:.4f} coord_x_move= {coord_x_move_ratio:.4f}"
    )


def get_game_paths(
    game_dir, metadata_path=None, metadata_filter=None, dataset_for_eval=None, max_games=None
):
    if metadata_path:
        with open(metadata_path) as mf:
            metadata = json.load(mf)
            if metadata_filter is not None:
                filter_lambda = eval(metadata_filter)
                game_ids = [k for k, g in metadata.items() if filter_lambda(g)]
                print(f"Selected {len(game_ids)} / {len(metadata)} games from metadata file.")
            else:
                game_ids = metadata.keys()

            if dataset_for_eval:
                train_cache, eval_cache = torch.load(dataset_for_eval)
                del train_cache
                game_ids = eval_cache.game_ids
                print(
                    f"Selected {len(game_ids)} / {len(metadata)} games from dataset cache eval set."
                )

            metadata_paths = [f"{game_dir}/game_{game_id}.json" for game_id in game_ids]
            paths = [p for p in metadata_paths if os.path.exists(p)]
            print(f"{len(paths)} / {len(metadata_paths)} from metadata exist.")
    else:
        # just use all the paths
        paths = sorted(glob.glob(f"{game_dir}/game*.json"))
        assert len(paths) > 0

    # reduce the number of games if necessary
    if len(paths) > max_games:
        print(f"Sampling {max_games} from dataset of size {len(paths)}")
        paths = [paths[i] for i in torch.randperm(len(paths))[:max_games]]

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_dir", help="Directory containing game.json files")
    parser.add_argument(
        "--max-games", type=int, default=1000000, help="Max # of games to evaluate"
    )
    parser.add_argument(
        "--max-year", help="Stop computing at this year (to avoid endless supports for draw)"
    )
    parser.add_argument(
        "--metadata-path", help="Path to metadata file for games, allowing for filtering"
    )
    parser.add_argument(
        "--metadata-filter", help="Lambda function to filter games based on metadata"
    )
    parser.add_argument("--dataset-for-eval", help="Dataset cache to select eval game IDs")
    args = parser.parse_args()

    paths = get_game_paths(
        args.game_dir,
        metadata_path=args.metadata_path,
        metadata_filter=args.metadata_filter,
        dataset_for_eval=args.dataset_for_eval,
        max_games=args.max_games,
    )

    compute_xpower_statistics(paths, max_year=args.max_year)
