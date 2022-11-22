#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib
import subprocess
import json

from fairdiplomacy.compare_agent_population_array import (
    DEFAULT_MODE,
    Mode,
    run_evals,
)
from fairdiplomacy.compare_agents_array import get_default_output_dir

import heyhi
from heyhi import checkpoint_repo
import conf.conf_cfgs


if __name__ == "__main__":
    MODES = {x.name.lower(): x for x in Mode}

    import argparse

    heyhi.setup_logging()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_games", type=int, required=True)
    # Still support json for backwards compatibility
    parser.add_argument(
        "--agent_mapping_json_path", type=str, required=False,
    )
    parser.add_argument(
        "--agent_mapping",
        type=str,
        required=False,
        help="Example: conf/ag_pop_cmp/mappings/example_agent_mapping.prototxt",
    )
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=DEFAULT_MODE.lower(),
        help="Run mode. gentle_start will start job if output not running."
        " start_missing will run jobs for seed without results. Note, it doesn't"
        " check whether another job for this seed is still running",
    )
    parser.add_argument(
        "--exp",
        type=pathlib.Path,
        required=True,
        help="Relative or absolute path for the experiment",
    )
    parser.add_argument(
        "--agent_overrides", nargs="+", default=[], help="Extra overrides to apply to each agent"
    )
    parser.add_argument(
        "--compute_elos", action="store_true", help="Performs and prints out cross-elo computation"
    )
    args, overrides = parser.parse_known_args()
    out_dir = (
        args.exp
        if args.exp.is_absolute()
        else get_default_output_dir() / "compare_agents_population" / args.exp
    )

    if not args.agent_mapping and not args.agent_mapping_json_path:
        raise ValueError("Must provide --agent_mapping")

    if args.agent_mapping:
        assert not args.agent_overrides, "Can only be used with --agent_mapping_json_path"
        with open(args.agent_mapping, "r") as f:
            agent_mapping = heyhi.load_config(
                args.agent_mapping, msg_class=conf.conf_cfgs.CompareAgentPopulationMapping
            )
    else:
        with open(args.agent_mapping_json_path, "r") as f:
            agent_mapping_json = json.load(f)
            agent_mapping = conf.conf_cfgs.CompareAgentPopulationMapping(
                agent=[
                    dict(
                        name=name,
                        cfg=entry["agent"],
                        overrides=entry["overrides"] + args.agent_overrides,
                    )
                    for name, entry in agent_mapping_json.items()
                ]
            )

    found_games = run_evals(
        agent_mapping=agent_mapping,
        mode=MODES[args.mode],
        num_games=args.num_games,
        seed=args.seed,
        out_dir=out_dir,
        overrides=overrides,
    ).done_evals
    if found_games:
        call_args = [
            "python",
            str(heyhi.PROJ_ROOT / "bin/get_compare_agent_population_results.py"),
            str(found_games[0].parent),
        ]
        if args.compute_elos:
            call_args.append("--compute_elos")
        subprocess.check_call(call_args)
