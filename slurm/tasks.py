#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import pathlib
from dataclasses import dataclass
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import heyhi
import submitit

from fairdiplomacy.utils.atomicish_file import atomicish_open_for_writing

DOT_SCHEDULED = ".scheduled"


@dataclass
class Task:
    """The information necessary to run a single task, whether locally on slurm. Must be pickleable.

    target_file_name: The name of the file intended to be produced as a result of the task.
      Should NOT include path. The task may also write other files as well, this is merely the
      file that will be used to determine task completion.

    task_kwargs: The args to pass to handler.

    handler: The handler for the task. Will be passed the task_kwargs, AND a kwarg "out" which is the file handle
      of the file to write results to, unless no_open_target_file is specified.

    no_open_target_file: If True, the caller is responsible for making sure the target file
      is produced. Using fairdiplomacy.utils.atomicish_file is strongly recommended.
    """

    target_file_name: str
    task_kwargs: Dict[str, Any]
    handler: Callable
    no_open_target_file: bool = False


def _run_task(task: Task, results_dir: pathlib.Path):
    target_path = results_dir / task.target_file_name
    # If somehow the file already exists, skip this task.
    if target_path.exists():
        logging.warning(f"Target file already exists, skipping this task: {target_path}")

    if task.no_open_target_file:
        task.handler(**task.task_kwargs)
    else:
        with atomicish_open_for_writing(target_path, binary=False) as f:
            task.handler(out=f, **task.task_kwargs)


def _run_locally(tasks: List[Task], results_dir: pathlib.Path):
    for task in tasks:
        # Write out a dummy file whose only purpose is to indicate that a task has been scheduled
        # to allow detection of unfinished tasks and such.
        out_file_todo = results_dir / (task.target_file_name + DOT_SCHEDULED)
        out_file_todo.touch()
    for task in tasks:
        out_file = results_dir / task.target_file_name
        if out_file.exists():
            logging.warning(f"Skipping {out_file}, it already exists")
            continue
        _run_task(task, results_dir)
    logging.info("Done running tasks locally")


def get_done_and_undone_files(
    results_dir: pathlib.Path,
) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    """Returns (files that are done, files that are scheduled but are failed or not done yet)."""
    logging.info(f"Checking results from: {results_dir}")
    done = []
    undone = []
    paths = results_dir.glob(f"*{DOT_SCHEDULED}")
    for path in paths:
        path = path.with_suffix("")
        if path.exists():
            done.append(path)
        else:
            undone.append(path)
    return (done, undone)


def run_locally_or_on_slurm(
    tasks: List[Task], results_dir: pathlib.Path, slurm_dir: pathlib.Path,
):
    """Run tasks locally or on slurm, based on command line args.
    Tasks whose target_file_name already exists will be skipped.
    Currently slurm not supported, local only.

    Arguments:
    tasks: List of tasks.
    results_dir: The directory to output produced files.
    slurm_dir: The directory to output slurm logs.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    # Also permanently log top-level info from this process,
    # the process launching the job in the first place, to here.
    heyhi.util.also_log_to_file(slurm_dir / "runner.log")

    command = " ".join(shlex.quote(s) for s in sys.argv)
    logging.info(f"Command line running this job was: {command}")
    with open(slurm_dir / "commandline.log", "a") as f:
        f.write(command + "\n")

    logging.info(f"Writing results to dir: {results_dir}")
    logging.info(f"Writing slurm logs to dir: {slurm_dir}")
    heyhi.log_git_status(base_path_for_logging=slurm_dir)

    _run_locally(tasks, results_dir)
