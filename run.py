#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

import heyhi


TASKS = {}


def _register(f):
    TASKS[f.__name__] = f
    return f


@_register
def compare_agents(cfg):
    # NEED TO SET THIS BEFORE CREATING THE AGENT!
    if cfg.seed >= 0:
        logging.info(f"Set seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    agent_one = build_agent_from_cfg(cfg.agent_one)
    agent_six = build_agent_from_cfg(cfg.agent_six)
    if cfg.cf_agent.WhichOneof("agent") is not None:
        cf_agent = build_agent_from_cfg(cfg.cf_agent)
    else:
        cf_agent = None

    power_string = cfg.power_one

    if cfg.num_processes > 0:
        assert cfg.num_trials > 0
        result = run_1v6_trial_multiprocess(
            agent_one, agent_six, power_string, cfg, cf_agent=cf_agent
        )
    else:
        result = run_1v6_trial(agent_one, agent_six, power_string, cfg, cf_agent=cf_agent)
        logging.warning("Result: {}".format(result))


@_register
def compare_agent_population(cfg):
    # NEED TO SET THIS BEFORE CREATING THE AGENT!
    if cfg.seed >= 0:
        logging.info(f"Set seed to {cfg.seed}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    agent_mappings: Dict[str, BaseAgent] = {
        agent.key: build_agent_from_cfg(agent.value) for agent in cfg.agents
    }
    power_agent_dict: Dict[Power, BaseAgent] = {
        pwr: agent_mappings[getattr(cfg, f"agent_{pwr}")] for pwr in POWERS
    }

    winning_power = run_population_trial(power_agent_dict, cfg)
    winning_agent = "NONE" if winning_power == "NONE" else getattr(cfg, f"agent_{winning_power}")
    logging.warning(
        "Result: Winning power - {}, Winning agent - {}".format(winning_power, winning_agent)
    )


@_register
def train(cfg):
    train_sl.run_with_cfg(cfg)


@_register
def exploit(cfg):
    # Do not load RL stuff by default.
    import fairdiplomacy.selfplay.exploit

    fairdiplomacy.selfplay.exploit.task(cfg)


@_register
def build_db_cache(cfg):
    from fairdiplomacy.data.build_db_cache import build_db_cache_from_cfg

    build_db_cache_from_cfg(cfg)


@_register
def situation_check(cfg):
    run_situation_check_from_cfg(cfg)


@_register
def benchmark_agent(cfg):
    import fairdiplomacy.benchmark_agent

    fairdiplomacy.benchmark_agent.run(cfg)


@_register
def compute_xpower_statistics(cfg):
    from fairdiplomacy.get_xpower_supports import compute_xpower_statistics, get_game_paths

    paths = get_game_paths(
        cfg.game_dir,
        metadata_path=cfg.metadata_path,
        metadata_filter=cfg.metadata_filter,
        dataset_for_eval=cfg.dataset_for_eval,
        max_games=cfg.max_games,
    )

    if cfg.cf_agent.WhichOneof("agent") is not None:
        cf_agent = build_agent_from_cfg(cfg.cf_agent)
    else:
        cf_agent = None

    compute_xpower_statistics(paths, max_year=cfg.max_year, cf_agent=cf_agent)


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
