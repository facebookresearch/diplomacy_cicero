#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
from typing import Any, Dict, List, Optional, Union
import abc
import argparse
import dataclasses
import hashlib

import numpy as np
import pandas as pd

import conf.conf_cfgs
from conf import agents_pb2
import fairdiplomacy.compare_agents_array
import fairdiplomacy.compare_agent_population_array
import fairdiplomacy.utils.game_scoring
from fairdiplomacy.utils.timing_ctx import TimingCtx
from heyhi import load_config
import heyhi


@dataclasses.dataclass
class Agent:
    name: str
    cfg: str
    overrides: Dict[str, Any]

    def create_derived(self, new_name: str, more_overrides: Dict) -> "Agent":
        joined = {}
        joined.update(self.overrides)
        joined.update(more_overrides)
        return Agent(name=new_name, cfg=self.cfg, overrides=joined)


@dataclasses.dataclass
class Population:
    name: str
    agents: List[Agent]


@dataclasses.dataclass
class H2HPopItem:
    agent: Agent
    population: Population

    # Seed for the population
    population_seed: Optional[int] = 0

    # Override for num seeds and initial seed. By default will use default from the sweep.
    num_games: Optional[int] = None

    # Experiment tag. Defines the output folder. If not set, will generate one
    # automatically. See SWEEP_NAME for output paths.
    exp_tag: Optional[str] = None

    # Optional. A string of "year,prob;year,prob;..."
    # "year,prob" indicates that at the start of SPRING of that year or later years
    # there is a probability of the game ending instantly and being scored as-is,
    year_spring_prob_of_ending: Optional[str] = None

    # # Specified how the experiment will be placed in the final table. The
    # # easiest way to use agent name for row and population name for col.
    # row: Optional[str] = None
    # col: Optional[str] = None

    # Optional. A string of "year,prob;year,prob;..."
    # "year,prob" indicates that at the start of SPRING of that year or later years
    # there is a probability of the game ending instantly and being scored as-is,
    year_spring_prob_of_ending: Optional[str] = None

    # Optional. If not specified, will use the value in conf/c01_ag_cmp/cmp.prototxt
    draw_on_stalemate_years: Optional[int] = None

    def build_exp_tag(self):
        if self.exp_tag is not None:
            return self.exp_tag
        else:
            return f"{self.agent.name}_IN_{self.population.name}"


class H2HPopSweep:
    """Base class to run a grid of games against a fixed population.

    Users has to redefine `get_eval_grid` function to return a list of h2h
    evals to run.

    When `go()` is called, not yet launched evals will be launched, and a
    pivot tables of scores agent_one vs agent_six are printed. One can check
    results of eval ran by another user, my providing `user` argument.
    """

    # Name of the sweep.
    # If several sweeps shares the same SWEEP_NAME, then eval results with
    # matching exp_names will be shared as well.
    SWEEP_NAME: str

    NUM_GAMES: int = 10

    YEAR_SPRING_PROB_OF_ENDING = None

    # Flags that can go here: num_hours, num_gpus, partition.
    # See full list in Slurm proto message.
    SLURM_DEFAULTS = {}

    METRICS_TO_PRINT = {
        "folder",
        "square_score",
        "square_score_std",
        "progress",
    }

    # Digits after decimal point.
    PRECISION = 3

    # Optional. If not specified, will use the value in conf/c01_ag_cmp/cmp.prototxt
    DRAW_ON_STALEMATE_YEARS = None

    @abc.abstractmethod
    def get_eval_grid(self) -> List[H2HPopItem]:
        pass

    ########
    ######## . You probably do not need go to read below.
    ########

    def __init__(self, args):
        self.args = args

    def maybe_launch(
        self, exp: H2HPopItem
    ) -> fairdiplomacy.compare_agent_population_array.EvalRun:
        kwargs: Dict[str, Any] = dict(
            num_games=self.NUM_GAMES,
            year_spring_prob_of_ending=self.YEAR_SPRING_PROB_OF_ENDING,
            draw_on_stalemate_years=self.DRAW_ON_STALEMATE_YEARS,
        )
        if exp.num_games is not None:
            kwargs["num_games"] = exp.num_games
        if exp.population_seed is not None:
            kwargs["seed"] = exp.population_seed
        if self.args.user:
            kwargs["mode"] = fairdiplomacy.compare_agent_population_array.Mode.CHECK
        elif self.args.missing:
            kwargs["mode"] = fairdiplomacy.compare_agent_population_array.Mode.START_MISSING
        if exp.year_spring_prob_of_ending is not None:
            kwargs["year_spring_prob_of_ending"] = exp.year_spring_prob_of_ending
        if exp.draw_on_stalemate_years is not None:
            kwargs["draw_on_stalemate_years"] = exp.draw_on_stalemate_years
        overrides = []
        if exp.year_spring_prob_of_ending is not None:
            overrides.append(f"year_spring_prob_of_ending={exp.year_spring_prob_of_ending}")
        elif self.YEAR_SPRING_PROB_OF_ENDING is not None:
            overrides.append(f"year_spring_prob_of_ending={self.YEAR_SPRING_PROB_OF_ENDING}")

        population_agents = []
        population_agents.append(
            dict(
                name=exp.agent.name,
                cfg=exp.agent.cfg,
                overrides=redefines_from_dict(exp.agent.overrides),
                min_count=1,
            )
        )
        population_agents.extend(
            dict(name=agent.name, cfg=agent.cfg, overrides=redefines_from_dict(agent.overrides),)
            for agent in exp.population.agents
        )

        out_dir = self.get_out_dir(user=self.args.user) / exp.build_exp_tag()
        print("Out dir: %s" % out_dir)
        eval_run = fairdiplomacy.compare_agent_population_array.run_evals(
            agent_mapping=conf.conf_cfgs.CompareAgentPopulationMapping(agent=population_agents),
            out_dir=out_dir,
            overrides=overrides,
            **kwargs,
        )

        try:
            agent_cfg = load_config(
                heyhi.CONF_ROOT / "common" / "agents" / (exp.agent.cfg + ".prototxt"),
                redefines_from_dict(exp.agent.overrides),
                msg_class=agents_pb2.Agent,
            )
        except:
            print("Cannot load", exp.agent)
            agent_cfg = None
        if agent_cfg is not None:
            print("Saving agent cfg to", out_dir / "agent.prototxt")
            with (out_dir / "agent.prototxt").open("w") as stream:
                print(agent_cfg, file=stream)
        return eval_run

    def maybe_launch_and_get_metrics(
        self, exp: H2HPopItem
    ) -> Optional[Dict[str, Union[str, float]]]:
        timings = TimingCtx()
        timings.start("maybe_launch")
        eval_run = self.maybe_launch(exp)
        found_games = eval_run.done_evals
        metrics = {
            "total_games": eval_run.num_games,
            "num_missing": eval_run.num_games - len(found_games),
            "progress": "%s/%s" % (len(found_games), eval_run.num_games),
            "folder": str(eval_run.out_dir),
        }
        if found_games:
            timings.start("get_results")
            power_scores = fairdiplomacy.compare_agent_population_array.get_power_scores_from_folder(
                eval_run.out_dir
            )
            stats_per_agent = collections.defaultdict(list)
            for _, agent_name, game_scores in power_scores:
                stats_per_agent[agent_name].append(game_scores)
            stats_per_agent = {
                agent: fairdiplomacy.utils.game_scoring.average_game_scores(game_scores)
                for agent, game_scores in stats_per_agent.items()
            }

            square_score = 100 * stats_per_agent[exp.agent.name][0].square_score
            square_score_std = 100 * stats_per_agent[exp.agent.name][1].square_score
            in_progress = "*" if eval_run.num_games != len(found_games) else ""

            metrics.update(
                {
                    "total_games": eval_run.num_games,
                    "num_missing": eval_run.num_games - len(found_games),
                    "progress": "%s/%s" % (len(found_games), eval_run.num_games),
                    "folder": str(eval_run.out_dir),
                    "square_score": square_score,
                    "square_score_std": f"{square_score:.1f} +- {square_score_std:.1f}{in_progress}",
                }
            )
        else:
            metrics.update({"square_score": -1, "square_score_std": "*"})
        timings.stop()
        # timings.pprint(print)
        return metrics

    def maybe_launch_and_get_all_data(self) -> Optional[List[Dict]]:
        data = []
        for exp in self.get_eval_grid():
            print(exp)
            exp_metrics = self.maybe_launch_and_get_metrics(exp)
            if exp_metrics is None:
                continue
            if exp_metrics["num_missing"]:
                print("Missing: %s/%s" % (exp_metrics["num_missing"], exp_metrics["total_games"]))
            exp_metrics["agent"] = exp.agent.name
            exp_metrics["population"] = exp.population.name
            data.append(exp_metrics)

        if all(x["num_missing"] == x["total_games"] for x in data):
            return None

        return data

    def get_out_dir(self, user=None):
        assert self.SWEEP_NAME is not None, "SWEEP_NAME must be defined"
        return (
            fairdiplomacy.compare_agents_array.get_default_output_dir(user)
            / "h2h"
            / self.SWEEP_NAME
        )

    def go(self):
        self.data = data = self.maybe_launch_and_get_all_data()
        if data is None:
            print("No data")
            return

        if self.args.user is None:
            csv_path = self.get_out_dir() / "all_metrics.csv"
            print("Saving all metrics to", csv_path)
            pd.DataFrame(data).to_csv(csv_path)

        all_metrics = set(self.METRICS_TO_PRINT)

        def maybe_save_df(metric, df):
            if self.args.user is None:
                csv_path = self.get_out_dir() / f"{metric}.csv"
                print("Saving to", csv_path)
                df.to_csv(csv_path)

        for metric in all_metrics:
            print("-->", metric)
            df = pd.DataFrame(data).pivot(values=metric, index="agent", columns="population")
            if metric == "square_score":
                df = df.round(3)
            sort_order = ([x["agent"] for x in data] + [x["population"] for x in data]).index
            df = df[sorted(df.columns, key=sort_order)]
            df = df.loc[sorted(df.index, key=sort_order)]
            print(df)
            maybe_save_df(
                metric, df,
            )

        return data

    @classmethod
    def parse_args_and_go(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--missing",
            action="store_true",
            help="If set, will launch missing seeds for already started jobs",
        )
        parser.add_argument(
            "--user", help="If set, will get results from another user. No new evals are launched"
        )

        cls(parser.parse_args()).go()


def redefines_from_dict(redefines):
    chunks = []
    for k, v in redefines.items():
        chunks.append(f"{k}={v}")
    return chunks
