#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib
import subprocess

from fairdiplomacy.compare_agents_array import (
    Mode,
    DEFAULT_CFG,
    DEFAULT_MODE,
    DEFAULT_VARIANT,
    get_default_output_dir,
    run_evals,
)
from fairdiplomacy.data.build_dataset import GameVariant
import heyhi
from heyhi import checkpoint_repo


if __name__ == "__main__":
    MODES = {x.name.lower(): x for x in Mode}
    VARIANTS = {x.name.lower(): x for x in GameVariant}

    import argparse

    assert DEFAULT_CFG.exists(), f"Cannot find default config: {DEFAULT_CFG}"

    heyhi.setup_logging()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_path",
        type=pathlib.Path,
        help="Use this config as a base for overrides",
        default=DEFAULT_CFG,
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=DEFAULT_VARIANT.lower(),
        help="Run mode. gentle_start will start job if output not running."
        " start_missing will run jobs for seed without results. Note, it doesn't"
        " check whether another job for this seed is still running",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=DEFAULT_MODE.lower(),
        help="Run mode. gentle_start will start job if output not running."
        " start_missing will run jobs for seed without results. Note, it doesn't"
        " check whether another job for this seed is still running",
    )
    parser.add_argument("--num_seeds", type=int, required=True)
    parser.add_argument("--initial_seed", type=int, required=False, default=0)
    parser.add_argument("--power", help="If specified, only run for this power.")
    parser.add_argument(
        "--exp",
        type=pathlib.Path,
        required=True,
        help=f"Relative or absolute path for the experiment",
    )
    args, overrides = parser.parse_known_args()
    out_dir = args.exp if args.exp.is_absolute() else get_default_output_dir() / args.exp
    found_games = run_evals(
        cfg_path=args.cfg_path,
        variant=VARIANTS[args.variant],
        mode=MODES[args.mode],
        num_seeds=args.num_seeds,
        initial_seed=args.initial_seed,
        overrides=overrides,
        out_dir=out_dir,
        active_powers=[args.power] if args.power else None,
    ).done_evals
    if found_games:
        subprocess.check_call(
            [
                "python",
                str(heyhi.PROJ_ROOT / "bin/get_compare_results.py"),
                str(found_games[0].parent),
            ]
        )
