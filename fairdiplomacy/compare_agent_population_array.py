#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Enables running a population evaluation across multiple agent_types"""
import dataclasses
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
from typing import List, Optional, Tuple, Dict, Union
import itertools

import conf.conf_cfgs
import conf.conf_pb2
import conf.agents_pb2
from fairdiplomacy.utils.yearprob import parse_year_spring_prob_of_ending

import heyhi
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.compare_agents_array import (
    Mode,
    STARTED_FILE,
    DEFAULT_MODE,
)
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Power
from fairdiplomacy.utils import game_scoring
from slurm.tasks import Task, run_locally_or_on_slurm


@dataclasses.dataclass
class EvalRun:
    cfg: conf.conf_cfgs.CompareAgentPopulationTask
    num_games: int
    seed: int
    out_dir: pathlib.Path
    done_evals: List[pathlib.Path]
    power_agent_mappings: List[Dict[Power, str]]


def get_eval_game_file_name(seed: int) -> str:
    return f"game.{seed}.json"


# 2022-02-09: Included here to support backwards compat for old compare agent pop runs
def get_deprecated_eval_game_file_name(seed: int, power_agent_mapping: Dict[Power, str]) -> str:
    game_name = "_".join([f"{power}-{agent[:8]}" for power, agent in power_agent_mapping.items()])
    return f"game.{game_name}.{seed}.json"


def get_games_from_folder(
    game_dir: pathlib.Path, include_partial: bool = False
) -> List[Tuple[Game, Dict[Power, str]]]:
    """Given a dir with game files like game_{game_nam}.{SEED}.{info,json,html} load games from it.

    Parameters:
    game_dir: Path to the dir that contains the game json and info files.

    Returns: A list of (game, agent_mapping), where agent_mapping is a dict
    {power: agent_name}.
"""
    games_and_mappings = []
    game_paths = list(game_dir.glob("game*.json"))
    if include_partial:
        game_paths += list(game_dir.glob("game*.json.partial"))
    for path in game_paths:
        power_mapping_json_path = os.path.join(
            str(path).replace(".partial", "").replace(".json", ""), "power_agent_mapping.json"
        )
        with open(power_mapping_json_path, "r") as f:
            power_agent_mapping = json.load(f)

        with path.open() as stream:
            game = Game.from_json(stream.read())
        games_and_mappings.append((game, power_agent_mapping))

    return games_and_mappings


def get_power_scores_from_folder_grouped_by_game(
    game_dir: pathlib.Path, include_partial: bool = False
) -> List[List[Tuple[Power, str, game_scoring.GameScores]]]:
    """Given a dir with game files like game_{game_name}.{SEED}.{info,json,html} load scores from it.

    Parameters:
    game_dir: Path to the dir that contains the game json and info files.

    Returns: A list with one power_scores per game, where power_scores is a list of
    (power, agent_name, scores for that power).
"""
    power_scores_by_game = []
    game_paths = list(game_dir.glob("game*.json"))
    if include_partial:
        game_paths += list(game_dir.glob("game*.json.partial"))
    for path in game_paths:
        power_mapping_json_path = os.path.join(
            str(path).replace(".json", ""), "power_agent_mapping.json"
        )
        with open(power_mapping_json_path, "r") as f:
            power_agent_mapping = json.load(f)

        info_path = pathlib.Path(str(path).replace(".json", ".info"))
        power_scores_this_game = []
        if info_path.exists():
            with info_path.open() as stream:
                info = json.load(stream)
            for pwr, agent_name in power_agent_mapping.items():
                game_scores = game_scoring.GameScores(**(info["game_scores"][pwr]))
                power_scores_this_game.append((pwr, agent_name, game_scores))
        else:
            with path.open() as stream:
                game = json.load(stream)
            for pwr, agent_name in power_agent_mapping.items():
                game_scores = game_scoring.compute_game_scores(POWERS.index(pwr), game)
                power_scores_this_game.append((pwr, agent_name, game_scores))

        power_scores_by_game.append(power_scores_this_game)

    return power_scores_by_game


def get_power_scores_from_folder(
    game_dir: pathlib.Path, include_partial: bool = False
) -> List[Tuple[Power, str, game_scoring.GameScores]]:
    """Given a dir with game files like game_{game_nam}.{SEED}.{info,json,html} load scores from it.

    Parameters:
    game_dir: Path to the dir that contains the game json and info files.

    Returns: A list of all the (power, agent_name, scores for that power)
    results across all games.
"""
    return list(
        itertools.chain.from_iterable(
            get_power_scores_from_folder_grouped_by_game(game_dir, include_partial=include_partial)
        )
    )


def _run_game(
    out_dir: pathlib.Path,
    power_agent_mapping: Dict[Power, str],
    seed: int,
    eval_cfg_path: pathlib.Path,
):
    game_json_name = get_eval_game_file_name(seed)
    log_dir = (out_dir / game_json_name.rsplit(".", 1)[0]).absolute()
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(log_dir, "power_agent_mapping.json"), "w") as fp:
        json.dump(power_agent_mapping, fp)

    tokens = [
        "python",
        f"{heyhi.PROJ_ROOT}/run.py",
        "--mode=start_continue",
        f"--cfg={eval_cfg_path}",
        f"--out={log_dir}",
        f"seed={seed}",
        f"out={out_dir}/{game_json_name}",
        ">>",
        f"{log_dir}/stdout.log",
        "2>>",
        f"{log_dir}/stderr.log",
    ]
    tokens.extend(
        [f'agent_{power}="{agent_name}"' for power, agent_name in power_agent_mapping.items()]
    )
    os.system(" ".join(tokens))


def launch_population(
    *,
    cfg: conf.conf_cfgs.MetaCfg,
    missing_jobs: List[Tuple[int, Dict[Power, str]]],
    out_dir: pathlib.Path,
):
    assert len(missing_jobs) > 0
    pid = os.getpid()
    eval_cfg_path = out_dir / f"default_eval_cfg.{pid}.prototxt"
    with eval_cfg_path.open("w") as stream:
        stream.write(str(cfg))

    tasks = []
    for (seed, power_agent_mapping) in missing_jobs:
        task = Task(
            target_file_name=get_eval_game_file_name(seed),
            task_kwargs=dict(
                out_dir=out_dir,
                power_agent_mapping=power_agent_mapping,
                seed=seed,
                eval_cfg_path=eval_cfg_path,
            ),
            handler=_run_game,
            # We don't want the file pre-opened for us, since _run_game itself
            # already is responsble for producing the game json file.
            no_open_target_file=True,
        )
        tasks.append(task)

    run_locally_or_on_slurm(
        tasks=tasks, results_dir=out_dir, slurm_dir=out_dir / "slurm",
    )


def run_evals(
    *,
    agent_mapping: conf.conf_cfgs.CompareAgentPopulationMapping,
    num_games: int,
    mode: Mode = getattr(Mode, DEFAULT_MODE),
    seed: Optional[int],
    out_dir: pathlib.Path,
    overrides: List[str],
    year_spring_prob_of_ending: Optional[str] = None,
    draw_on_stalemate_years: Optional[int] = None,
) -> EvalRun:
    # Load existing seed if we can and make sure it matches user's seed,
    # or generate a new one if not provided.
    seed_path = out_dir / "SEED.txt"
    if seed_path.exists():
        with seed_path.open() as f:
            existing_seed = int(f.read().strip())
        if seed is not None and existing_seed != seed:
            raise Exception(
                f"Specified seed {seed} but this compare agents population was already run with seed {existing_seed}"
            )
        seed = existing_seed
        logging.info(f"Using pre-existing seed: {seed}")
        used_existing_seed = True
    else:
        if seed is None:
            seed = abs(int.from_bytes(os.urandom(4), sys.byteorder)) % 100000000
        used_existing_seed = False
        if out_dir.exists() and out_dir.glob("game*.json"):
            raise Exception(
                "No SEED.txt file was found in compare agents dir, but games were found. This is probably an old compare agents population sweep performed before a refactor, run bin/get_compare_agent_population_results.py directly on it"
            )

    rand = random.Random(seed)
    power_agent_mappings = _generate_agent_populations(rand, agent_mapping, num_games)

    if year_spring_prob_of_ending:
        overrides.append(f"year_spring_prob_of_ending={year_spring_prob_of_ending}")
        # Parse so that we fail faster if it's invalid.
        parse_year_spring_prob_of_ending(year_spring_prob_of_ending)
    if draw_on_stalemate_years:
        overrides.append(f"draw_on_stalemate_years={draw_on_stalemate_years}")

    cfg: conf.conf_cfgs.MetaCfg = _create_default_meta_cfg(agent_mapping, overrides)
    known_files = frozenset(x.name for x in out_dir.iterdir()) if out_dir.exists() else frozenset()
    missing_jobs: List[Tuple[int, Dict[Power, str]]] = []
    found_games: List[pathlib.Path] = []
    for power_agent_mapping in power_agent_mappings:
        game_seed = rand.randint(0, 99999999)
        name = get_eval_game_file_name(game_seed)
        deprecated_name = get_deprecated_eval_game_file_name(game_seed, power_agent_mapping)
        if name in known_files or deprecated_name in known_files:
            found_games.append(out_dir / name)
        else:
            missing_jobs.append((game_seed, power_agent_mapping))

    out_dir.mkdir(exist_ok=True, parents=True)
    # At this point, now we write down the seed we used
    if not used_existing_seed:
        with seed_path.open("w") as f:
            f.write(str(seed))
        logging.info(f"Using new seed (wrote to {seed_path}): {seed}")

    started_file = out_dir / STARTED_FILE
    if not started_file.exists() or mode == mode.START_MISSING:
        if not missing_jobs:
            logging.info("No missing evals!")
        else:
            launch_population(
                cfg=cfg, missing_jobs=missing_jobs, out_dir=out_dir,
            )

            with started_file.open("w"):
                pass
    else:
        logging.info(
            "Eval in the folder exists. Not running again."
            " Use --mode=start_missing to compute missing"
        )

    logging.info("Out dir: %s", out_dir)
    logging.info("Logs:\n\ttail -n5 %s/*/*err.log", out_dir)
    logging.info("Done %d out of %d", len(found_games), len(found_games) + len(missing_jobs))
    return EvalRun(
        out_dir=out_dir,
        done_evals=found_games,
        cfg=cfg.compare_agent_population if cfg is not None else cfg,
        num_games=num_games,
        seed=seed,
        power_agent_mappings=power_agent_mappings,
    )


def _generate_agent_populations(
    rand: random.Random,
    agent_mapping: conf.conf_cfgs.CompareAgentPopulationMapping,
    num_games: int,
) -> List[Dict[Power, str]]:
    MAX_ATTEMPTS = 1000
    for agent in agent_mapping.agent:
        assert agent.name is not None
    agent_name_to_min_count = {
        agent.name: agent.min_count for agent in agent_mapping.agent if agent.name is not None
    }
    agent_names = list(agent_name_to_min_count)
    assert len(agent_names) == len(
        set(agent_names)
    ), f"Agents specified multiple times: {set([name for name in agent_names if agent_names.count(name) > 1])}"
    power_agent_mappings = []
    for _ in range(num_games):
        for _ in range(MAX_ATTEMPTS):
            sample = {pwr: rand.choice(agent_names) for pwr in POWERS}
            leftovers = agent_name_to_min_count.copy()
            for agent in sample.values():
                leftovers[agent] -= 1
            if all(x <= 0 for x in leftovers.values()):
                power_agent_mappings.append(sample)
                break
        else:
            raise RuntimeError(
                f"Failed to satisfy min_count requirements after {MAX_ATTEMPTS} attempts"
            )
    return power_agent_mappings


def _create_default_meta_cfg(
    agent_mapping: conf.conf_cfgs.CompareAgentPopulationMapping, overrides
):
    cfg = heyhi.load_config(
        heyhi.PROJ_ROOT / f"conf/ag_pop_cmp/template.prototxt",
        msg_class=conf.conf_pb2.MetaCfg,
        overrides=overrides,
    ).to_editable()

    last_agent_name = None
    for agent_entry in agent_mapping.agent:
        agent_name = agent_entry.name
        agent_cfg = agent_entry.cfg
        overrides = agent_entry.overrides
        assert agent_cfg is not None
        # Starts with slash: assume absolute filesystem path
        if os.path.isabs(agent_cfg):
            agent_path = pathlib.Path(agent_cfg)
        # User explicitly specifies a prototxt file but it's a relative path:
        # assume it's relative to the repo dir
        elif agent_cfg.endswith(".prototxt"):
            agent_path = heyhi.PROJ_ROOT / agent_cfg
        # No absolute path, no .prototxt in name, assume we want a default provided config
        else:
            agent_path = heyhi.PROJ_ROOT / f"conf/common/agents/{agent_cfg}.prototxt"
        agent_cfg = heyhi.load_config(
            agent_path, msg_class=conf.agents_pb2.Agent, overrides=overrides,
        ).to_editable()
        cfg.compare_agent_population.agents.append(
            conf.conf_cfgs.CompareAgentPopulationTask.NamedAgent(
                key=agent_name, value=agent_cfg
            ).to_editable()
        )
        last_agent_name = agent_name
    for pwr in POWERS:
        setattr(cfg.compare_agent_population, f"agent_{pwr}", last_agent_name)

    cfg = cfg.to_frozen()
    return cfg
