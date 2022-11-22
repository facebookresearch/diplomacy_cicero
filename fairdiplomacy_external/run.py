#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
import logging
import os
from typing import Dict
import socket
import torch
import numpy as np

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.compare_agents import run_1v6_trial, run_1v6_trial_multiprocess
from fairdiplomacy.compare_agent_population import run_population_trial
from fairdiplomacy.models.base_strategy_model import train_sl
from fairdiplomacy.models.consts import POWERS

from fairdiplomacy.situation_check import run_situation_check_from_cfg
from fairdiplomacy.typedefs import Power
from fairdiplomacy_external.webdip_api import play_webdip as play_webdip_impl

import heyhi


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def play_webdip(cfg):
    play_webdip_impl(cfg)


@heyhi.save_result_in_cwd
def main(task, cfg, log_level):
    logging.info(f"Machine IP Address: {socket.gethostbyname(socket.gethostname())}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not hasattr(cfg, "heyhi_patched"):
        raise RuntimeError("Run `make protos`")
    cfg = cfg.to_frozen()
    heyhi.setup_logging(console_level=log_level)
    logging.info("Cwd: %s", os.getcwd())
    logging.info("Task: %s", task)
    logging.info("Cfg:\n%s", cfg)
    logging.debug("Cfg (with defaults):\n%s", cfg.to_str_with_defaults())
    heyhi.log_git_status()
    logging.info("Is on slurm: %s", heyhi.is_on_slurm())
    logging.info("Job env: %s", heyhi.get_job_env())
    if heyhi.is_on_slurm():
        logging.info("Slurm job id: %s", heyhi.get_slurm_job_id())
    logging.info("Is master: %s", heyhi.is_master())
    if getattr(cfg, "use_default_requeue", False):
        heyhi.maybe_init_requeue_handler()

    if task not in TASKS:
        raise ValueError("Unknown task: %s. Known tasks: %s" % (task, sorted(TASKS)))
    return TASKS[task](cfg)


if __name__ == "__main__":
    heyhi.parse_args_and_maybe_launch(main)
