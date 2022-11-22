#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict, List, Tuple, Optional, Union
import abc
import argparse
import dataclasses
import hashlib

import pandas as pd
import torch

import fairdiplomacy.utils.game_scoring
import fairdiplomacy.compare_agents_array


@dataclasses.dataclass
class H2HItem:
    # A tuple (base_config_name, *overrides).
    # Something that you would pass to c01_ag_cmp/cmp.prototxt.
    # E.g., ("mila", "mila.temperature=0.1")
    #       ("base_strategy_model", "base_strategy_model.temperature=0.5", "base_strategy_model.model_path=/tmp/path")
    # Note, don't specify agent_one or agent_six here, it will be added for you.
    agent_one: Tuple[str, ...]
    agent_six: Tuple[str, ...]

    # Override for num seeds and initial seed. By default will use default from the sweep.
    # (seeds used will range from initial_seed to initial_seed + num_seeds - 1)
    num_seeds: Optional[int] = None
    initial_seed: Optional[int] = None

    # If set, will redefine time_per_phase in this h2h eval.
    time_per_phase: Optional[int] = None

    # Capture agent logs into the game json
    capture_logs: Optional[bool] = None

    # Compute and record variance reduction offsets for report_variance_reduced_scores.
    # This should be a path to a base_strategy_model value model.
    variance_reduction_model_path: Optional[str] = None
    # Report variance reduced square_scores instead of raw square_scores. Requires that
    # variance_reduction_model_path was set when generating the games.
    report_variance_reduced_scores: Optional[bool] = None

    # Optional. A string of "year,prob;year,prob;..."
    # "year,prob" indicates that at the start of SPRING of that year or later years
    # there is a probability of the game ending instantly and being scored as-is,
    year_spring_prob_of_ending: Optional[str] = None

    # Optional. If not specified, will use the value in conf/c01_ag_cmp/cmp.prototxt
    draw_on_stalemate_years: Optional[int] = None

    # Experiment tag. Defines the output folder. If not set, will generate one
    # automatically. See SWEEP_NAME for output paths.
    exp_tag: Optional[str] = None

    # Specified how the experiment will be placed in the final table. The
    # easiest way to use agent_one name for row and agent_six name for col.
    row: Optional[str] = None
    col: Optional[str] = None

    # If True, will enable optimization where policies for 6 opponents are computed within a single call.
    share_strategy_six: bool = False

    def build_exp_tag(self):
        if self.exp_tag is not None:
            return self.exp_tag

        def encode(agent):
            if isinstance(agent, str):
                return agent
            return "%s_%s" % (agent[0], hashlib.md5(str(tuple(agent)).encode()).hexdigest()[:5])

        shared_tag = "_shared" if self.share_strategy_six else False
        return "%s_X_%s%s" % (encode(self.agent_one), encode(self.agent_six), shared_tag)

    def get_row(self):
        if self.row is not None and self.row != "":
            return self.row
        return ":".join([self.agent_one] if isinstance(self.agent_one, str) else self.agent_one)

    def get_col(self):
        if self.col is not None and self.col != "":
            return self.col
        return ":".join([self.agent_six] if isinstance(self.agent_six, str) else self.agent_six)


class H2HSweep:
    """Base class to run a grid of head2head evals and gather results.

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

    NUM_SEEDS = 10
    INITIAL_SEED = 0
    VARIANT = "CLASSIC"
    ACTIVE_POWERS = None
    CAPTURE_LOGS = False
    VARIANCE_REDUCTION_MODEL_PATH = ""
    REPORT_VARIANCE_REDUCED_SCORES = False
    # Exit game once agent_one is out. Speeds up evals. Disable to collect a full game.
    STOP_ON_DEATH = True
    YEAR_SPRING_PROB_OF_ENDING = None
    # Optional. If not specified, will use the value in conf/c01_ag_cmp/cmp.prototxt
    DRAW_ON_STALEMATE_YEARS = None
    # If not None, will redefine time_per_phase in the game environment.
    TIME_PER_PHASE = None

    # Flags that can go here: num_hours, num_gpus, partition.
    # See full list in Slurm proto message.
    SLURM_DEFAULTS = {}

    # Column and row captions for the final pivot tables.
    ROW_NAME = "agent_one"
    COL_NAME = "agent_six"

    METRICS_TO_PRINT = (
        "square_score",
        "square_score_1sigma",
        "square_score_2sigma",
        "null_sigmas",
        "progress",
    )

    # Digits after decimal point.
    PRECISION = 3

    @abc.abstractmethod
    def get_eval_grid(self) -> List[H2HItem]:
        pass

    def postprocess(self, data: List[Dict]):
        """Users could redefine this mething to do something custom with the results."""
        pass

    ########
    ######## . You probably do not need go to read below.
    ########

    def __init__(self, args):
        self.args = args

    @classmethod
    def build_agent_overrides(cls, prefix, agent: Tuple[str, ...]) -> List[str]:
        cfg_name, *overrides = agent
        if "/" not in cfg_name:
            cfg_name = f"agents/{cfg_name}"
        prefixed_overrides = [f"I{prefix}={cfg_name}"]
        for over in overrides:
            # Allow overrides of entire subconfigs, rather than just single fields.
            if over.startswith("I"):
                prefixed_overrides.append(f"I{prefix}.{over[1:]}")
            else:
                prefixed_overrides.append(f"{prefix}.{over}")
        return prefixed_overrides

    def maybe_launch(self, exp: H2HItem) -> fairdiplomacy.compare_agents_array.EvalRun:
        kwargs = dict(
            variant=getattr(fairdiplomacy.compare_agents_array.GameVariant, self.VARIANT),
            num_seeds=self.NUM_SEEDS,
            initial_seed=self.INITIAL_SEED,
            active_powers=self.ACTIVE_POWERS,
            capture_logs=self.CAPTURE_LOGS,
            stop_on_death=self.STOP_ON_DEATH,
            variance_reduction_model_path=self.VARIANCE_REDUCTION_MODEL_PATH,
            year_spring_prob_of_ending=self.YEAR_SPRING_PROB_OF_ENDING,
            draw_on_stalemate_years=self.DRAW_ON_STALEMATE_YEARS,
        )
        if self.TIME_PER_PHASE is not None:
            kwargs["time_per_phase"] = self.TIME_PER_PHASE
        if exp.num_seeds is not None:
            kwargs["num_seeds"] = exp.num_seeds
        if exp.initial_seed is not None:
            kwargs["initial_seed"] = exp.initial_seed
        if exp.capture_logs is not None:
            kwargs["capture_logs"] = exp.capture_logs
        if exp.variance_reduction_model_path is not None:
            kwargs["variance_reduction_model_path"] = exp.variance_reduction_model_path
        if exp.time_per_phase is not None:
            kwargs["time_per_phase"] = exp.time_per_phase
        if exp.year_spring_prob_of_ending is not None:
            kwargs["year_spring_prob_of_ending"] = exp.year_spring_prob_of_ending
        if exp.draw_on_stalemate_years is not None:
            kwargs["draw_on_stalemate_years"] = exp.draw_on_stalemate_years
        kwargs["share_strategy"] = exp.share_strategy_six
        if self.args.user:
            kwargs["mode"] = fairdiplomacy.compare_agents_array.Mode.CHECK
        elif self.args.missing:
            kwargs["mode"] = fairdiplomacy.compare_agents_array.Mode.START_MISSING
        overrides = []
        overrides.extend(self.build_agent_overrides("agent_one", exp.agent_one))
        overrides.extend(self.build_agent_overrides("agent_six", exp.agent_six))
        kwargs["overrides"] = overrides
        kwargs["out_dir"] = self.get_out_dir(user=self.args.user) / exp.build_exp_tag()
        print("Out dir: %s" % kwargs["out_dir"])
        eval_run = fairdiplomacy.compare_agents_array.run_evals(**kwargs)
        return eval_run

    def maybe_launch_and_get_metrics(self, exp: H2HItem) -> Optional[Dict[str, Union[str, float]]]:
        num_powers_in_game = {"CLASSIC": 7, "FVA": 2}[self.VARIANT]
        num_powers_being_tested = (
            len(self.ACTIVE_POWERS) if self.ACTIVE_POWERS else num_powers_in_game
        )
        print(exp)
        num_seeds = exp.num_seeds or self.NUM_SEEDS
        total_games = num_seeds * num_powers_being_tested
        eval_run = self.maybe_launch(exp)
        report_variance_reduced_scores = (
            exp.report_variance_reduced_scores or self.REPORT_VARIANCE_REDUCED_SCORES
        )

        # Caching aggregation of .json files. For runs that are already done,
        # we don't want to re-gather results.
        cache_path = eval_run.out_dir / "cache.pth"
        cache_key = (
            frozenset(p.name for p in eval_run.out_dir.iterdir() if p.name != cache_path.name),
            num_seeds,
            total_games,
            report_variance_reduced_scores,
        )

        power_scores_list = None
        if cache_path.exists():
            cache_content = torch.load(cache_path)
            if cache_content["key"] != cache_key:
                print("Invalidating", cache_path)
                cache_path.unlink()
            else:
                power_scores_list = cache_content["power_scores_list"]
        if power_scores_list is None:
            power_scores_list = fairdiplomacy.compare_agents_array.get_power_scores_from_folder(
                eval_run.out_dir, apply_variance_reduction=report_variance_reduced_scores,
            )

        if not power_scores_list:
            metrics = {}
            num_missing = total_games
        else:
            _, scores_list = zip(*power_scores_list)
            means, stds = fairdiplomacy.utils.game_scoring.average_game_scores(scores_list)
            num_missing = total_games - means.num_games
            metrics = means._asdict()
            metrics.update((f"{k}_err", v) for k, v in stds._asdict().items())
            metrics["square_score_std"] = f"%.{self.PRECISION}f+-%.{self.PRECISION}f" % (
                means.square_score,
                stds.square_score,
            )
            # +/- 1 standard_error confidence interval
            metrics["square_score_1sigma"] = f"%.{self.PRECISION}f:%.{self.PRECISION}f" % (
                means.square_score - stds.square_score,
                means.square_score + stds.square_score,
            )
            # +/- 2 standard_error confidence interval
            metrics["square_score_2sigma"] = f"%.{self.PRECISION}f:%.{self.PRECISION}f" % (
                means.square_score - stds.square_score * 2,
                means.square_score + stds.square_score * 2,
            )
            # number of standard errors away from null hypothesis of 1/num_powers
            if stds.square_score > 0:
                metrics["null_sigmas"] = f"%.{self.PRECISION}f" % (
                    (means.square_score - 1 / num_powers_in_game) / stds.square_score
                )
            else:
                metrics["null_sigmas"] = ""
            if num_missing:
                metrics["square_score_std"] += "*"
                metrics["square_score_1sigma"] += "*"
                metrics["square_score_2sigma"] += "*"
                metrics["null_sigmas"] += "*"
        if not num_missing and not cache_path.exists():
            print("Saving cache", cache_path)
            torch.save(dict(key=cache_key, power_scores_list=power_scores_list), cache_path)
        metrics["progress"] = "%s/%s" % (total_games - num_missing, total_games)
        metrics["num_missing"] = num_missing
        metrics["total_games"] = total_games
        metrics["folder"] = str(eval_run.out_dir)
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
            exp_metrics[self.ROW_NAME] = exp.get_row()
            exp_metrics[self.COL_NAME] = exp.get_col()
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
            df = pd.DataFrame(data).pivot(
                values=metric, index=self.ROW_NAME, columns=self.COL_NAME
            )
            if metric == "square_score":
                df = df.round(3)
            sort_order = (
                [x[self.ROW_NAME] for x in data] + [x[self.COL_NAME] for x in data]
            ).index
            df = df[sorted(df.columns, key=sort_order)]
            df = df.loc[sorted(df.index, key=sort_order)]
            print(df)
            maybe_save_df(
                metric, df,
            )

        maybe_df_dict = self.postprocess(data)
        if isinstance(maybe_df_dict, dict):
            for name, extra_df in maybe_df_dict.items():
                maybe_save_df(name, extra_df)
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
