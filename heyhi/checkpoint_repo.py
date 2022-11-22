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

import argparse
import sys
import os
import logging
import pathlib
import subprocess

from . import util

# Name of a file storing a link to checkpoint.
LAST_CKPT_FILE = "ckpt.link"
# Special string value for command line argument that indicates to create a default checkpoint in
# tmp directory, instead of being interpreted as a path
DEFAULT_CHECKPOINT = "DEFAULT"
# Special string value for command line argument that indicates to not checkpoint
# (as of 2020-02, left over from older code, not clear why this is needed, maybe someone or some old scripts rely on it)
# The empty string also works to not checkpoint.
NONE_CHECKPOINT = "none"

PROJ_ROOT_DIR = pathlib.Path(__file__).parent.parent


def _create_checkpoint(checkpoint_arg: str) -> pathlib.Path:
    """Create a copy of all the code in the repo into a new tmp directory and return the directory path"""
    if checkpoint_arg == DEFAULT_CHECKPOINT:
        ckpt_dir = pathlib.Path(
            subprocess.check_output(str(PROJ_ROOT_DIR / "slurm" / "checkpoint_repo.sh"))
            .decode()
            .split()[-1]
        )
    else:
        ckpt_dir = pathlib.Path(
            subprocess.check_output(
                [str(PROJ_ROOT_DIR / "slurm" / "checkpoint_repo.sh"), checkpoint_arg]
            )
            .decode()
            .split()[-1]
        )

    logging.info("Created a code checkpoint: %s", ckpt_dir)
    return ckpt_dir


def _get_or_create_checkpoint_for_out_dir(
    checkpoint_arg: str, out_dir: pathlib.Path
) -> pathlib.Path:
    """If <out_dir>/ckpt.link already exists, return the path in it, else create a checkpoint and store the path in <out_dir>/ckpt.link"""
    last_ckpt_path = out_dir / LAST_CKPT_FILE
    if last_ckpt_path.exists():
        with open(last_ckpt_path) as stream:
            ckpt_dir = pathlib.Path(stream.read().strip())
        logging.info("Using previously used checkpoint for this experiment: %s", ckpt_dir)
        assert ckpt_dir.exists(), ckpt_dir
    else:
        ckpt_dir = _create_checkpoint(checkpoint_arg)
        last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(last_ckpt_path, "w") as stream:
            stream.write(str(ckpt_dir.absolute()))
    return ckpt_dir


def add_parser_arg(parser):
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Directory to automatically copy the code to keep an archive of it. For remote slurm jobs, will also run the code from there so that editing the local repo further doesn't screw up scheduled runs. Defaults to a tmp dir in ~/diplomacy_experiments/repo_checkpoints/ unless this is a --adhoc run, in which case no checkpointing is done",
    )


def handle_parser_arg(checkpoint_arg: str, out_dir: pathlib.Path) -> pathlib.Path:
    """
    If checkpoint_arg is falsy or NONE_CHECKPOINT or this process is on slurm already:
        no checkpointing, return code root

    If checkpoint_arg is DEFAULT_CHECKPOINT:
        peform get_or_create_checkpoint_for_out_dir

    If checkpoint_arg is set, use it.
    """
    if checkpoint_arg and checkpoint_arg != NONE_CHECKPOINT and not util.is_on_slurm():
        if checkpoint_arg == DEFAULT_CHECKPOINT:
            ckpt_dir = _get_or_create_checkpoint_for_out_dir(checkpoint_arg, out_dir)
        else:
            os.makedirs(checkpoint_arg, exist_ok=True)
            ckpt_dir = _get_or_create_checkpoint_for_out_dir(checkpoint_arg, out_dir)
    else:
        ckpt_dir = PROJ_ROOT_DIR
    return ckpt_dir


def is_nontrivial_checkpoint(path: pathlib.Path) -> bool:
    return path != PROJ_ROOT_DIR
