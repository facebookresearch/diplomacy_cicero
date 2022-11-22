#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Sequence, Optional
import argparse
import logging
import os
import pathlib
import pprint

import torch

from . import checkpoint_repo
from . import conf
from . import util

# This hardcoding is terrible. But as HeyHi is not a framework, it's fine. The
# reason for hardcoding, it that sweeps call `maybe_launch` here, while adhoc
# runs use `parse_args_and_maybe_launch` in user's run.py. And it's trickier to
# extract project name from the userland then from here.
# This constant is only used to define where to store logs.
PROJECT_NAME = "diplomacy"


def get_exp_dir(project_name) -> pathlib.Path:
    return pathlib.Path(
        os.environ.get(
            "HH_EXP_DIR", os.environ["HOME"] + f"/diplomacy_experiments/results/{project_name}"
        )
    )


def get_default_exp_dir():
    return get_exp_dir(PROJECT_NAME)


def parse_args_and_maybe_launch(main: Callable) -> None:
    """Main entrypoint to HeyHi.

    It does eveything HeyHi is for:
        * Finds a config.
        * Applies local and global includes and overridefs to the config.
        * Determindes a folder to put the experiment.
        * Detects the status of the experiment (if already running).
        * Depending on the status and mode, maybe start/restarts experiment
        locally or remotely.

    Args:
        task: dictionary of callables one of which will be called based on
            cfg.task.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cfg", required=True, type=pathlib.Path)
    parser.add_argument("--adhoc", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Do not ask confirmation in restart mode"
    )
    parser.add_argument(
        "--mode", choices=util.MODES, help="See heyhi/util.py for mode definitions."
    )
    checkpoint_repo.add_parser_arg(parser)
    parser.add_argument(
        "--exp_id_pattern_override",
        default=util.EXP_ID_PATTERN,
        help="A pattern to construct exp_id. Job's data is stored in <exp_root>/<exp_id>",
    )
    parser.add_argument("--out", help="Alias for exp_id_pattern_override. Overrides it if set.")
    parser.add_argument(
        "--print", help="If set, will print composed config and exit", action="store_true"
    )
    parser.add_argument(
        "--print-flat",
        help="If set, will print composed config as a list of redefines and exit. This command lists only explicitly defined flags",
        action="store_true",
    )
    parser.add_argument(
        "--print-flat-all",
        help="If set, will print composed config as a list of redefines and exit. This command lists all possible flags",
        action="store_true",
    )
    parser.add_argument("--log-level", default="INFO", choices=["ERROR", "WARN", "INFO", "DEBUG"])
    args, overrides = parser.parse_known_args()

    if args.out:
        args.exp_id_pattern_override = args.out
    del args.out
    if args.mode is None:
        args.mode = "restart" if args.adhoc else "gentle_start"

    overrides = [x.lstrip("-") for x in overrides]

    if args.print or args.print_flat or args.print_flat_all:
        task, meta_cfg = conf.load_root_proto_message(args.cfg, overrides)
        cfg = getattr(meta_cfg, task)
        if args.print:
            print(cfg)
        if args.print_flat or args.print_flat_all:
            for k, v in conf.flatten_cfg(cfg, with_all=args.print_flat_all).items():
                if v is None:
                    v = "NULL"
                elif isinstance(v, str) and (" " in v or not v):
                    v = repr(v)
                print("%s=%s" % (k, v))
        return

    kwargs = {}
    for k, v in vars(args).items():
        if k.startswith("print"):
            # If print was not handled.
            assert not v, "Bug!"
            continue
        kwargs[k] = v

    maybe_launch(main, exp_root=get_exp_dir(PROJECT_NAME), overrides=overrides, **kwargs)


def maybe_launch(
    main: Callable,
    *,
    exp_root: Optional[pathlib.Path],
    overrides: Sequence[str],
    cfg: pathlib.Path,
    mode: util.ModeType,
    checkpoint: str = checkpoint_repo.DEFAULT_CHECKPOINT,
    adhoc: bool = False,
    exp_id_pattern_override=None,
    force: bool = False,
    log_level: str = "INFO",
) -> util.ExperimentDir:
    """Computes the task locally or remotely if neeeded in the mode.

    This function itself is always executed locally.

    The function checks the exp_handle first to detect whether the experiment
    is running, dead, or dead. Depending on that and the mode the function
    may kill the job, wipe the exp_handle, start a computation or do none of
    this.

    See handle_dst() for how the modes and force are handled.

    The computation may run locally or on the cluster depending on the
    launcher config section. In both ways main(cfg) with me executed with the
    final config with all overrides and substitutions.
    """
    util.setup_logging(console_level=log_level)
    logging.info("Config: %s", cfg)
    logging.info("Overrides: %s", overrides)

    if exp_root is None:
        exp_root = get_exp_dir(PROJECT_NAME)

    exp_id = util.get_exp_id(cfg, overrides, adhoc, exp_id_pattern=exp_id_pattern_override)
    exp_handle = util.ExperimentDir(exp_root / exp_id, exp_id=exp_id)
    need_run = util.handle_dst(exp_handle, mode, force=force)
    logging.info("Exp dir: %s", exp_handle.exp_path)
    logging.info("Job status [before run]: %s", exp_handle.get_status())
    if need_run:
        # Only checkpoint if we actually need a new run.
        # Specially disable checkpointing by default in the case of adhoc runs, will checkpoint
        # if the user explicitly specifies a non default checkpoint path
        if adhoc and checkpoint == checkpoint_repo.DEFAULT_CHECKPOINT:
            ckpt_dir = ""
        else:
            ckpt_dir = checkpoint_repo.handle_parser_arg(checkpoint, exp_handle.exp_path)
        util.run_with_config(main, exp_handle, cfg, overrides, ckpt_dir, log_level)
    if exp_handle.is_done():
        result = torch.load(exp_handle.result_path)
        if result is not None:
            simple_result = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
            pprint.pprint(simple_result, indent=2)
    return exp_handle
