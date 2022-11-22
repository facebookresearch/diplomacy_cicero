#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import html
import json
import os
import pathlib
import random
import types
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import heyhi
import IPython
import pandas as pd
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Action, JointAction
from fairdiplomacy.utils.game_scoring import get_game_result_from_json
from parlai_diplomacy.wrappers.dialogue import ParlAIDialogueWrapper

AnyPath = Union[str, Path]


def load_jsonl(filename: str):
    """Load a jsonl file from a path."""
    ret = []
    with open(filename) as f:
        lines = f.readlines()
        ret = [json.loads(line) for line in lines[:-1]]
        try:
            ret.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            # Using print so it shows up in jupyter
            print(f"WARNING: Could not decode last line from {filename}")
    return ret


def message_to_dict(cfg):
    return heyhi.conf_to_dict(cfg)


def flatten_dict(cfg: Dict, _prefix=""):
    ret = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            ret.update(**flatten_dict(v, _prefix=_prefix + k + "."))
        else:
            ret[_prefix + k] = v
    return ret


def get_sl_learning_curves(folders: List[AnyPath]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return supervised learning curves from a list of checkpoint folders.

    The results are stored in pandas dataframes with one row per logged
    learning curve point. The dataframes also contain the name of the folder
    and all the config parameters in each row.

    Arguments:
        - *folders: a set of strings, each one containing a path to a
                    supervised learning checkpoint dir.
    Returns:
        - train_curves: a pandas dataframe with training learning curves.
        - val_curves: a pandas dataframe with validation learning curves.
    """

    all_metrics = []
    for folder in folders:
        folder = Path(folder)
        metrics = load_jsonl(str(folder / "metrics.jsonl"))
        config = heyhi.load_root_config(folder / "config_meta.prototxt", overrides=[])
        config_dict = flatten_dict(message_to_dict(config.train))
        for m in metrics:
            m["path"] = str(folder)
            m["name"] = os.path.basename(folder)
            m.update(config_dict)
        all_metrics += metrics

    train_metrics = pd.DataFrame([m for m in all_metrics if "valid_loss" not in m])
    val_metrics = pd.DataFrame([m for m in all_metrics if "valid_loss" in m])
    return train_metrics, val_metrics


def get_compare_agents_results(folders: List[AnyPath]) -> pd.DataFrame:
    """Return compare agents results from a list of folders, each of which
    was produced from `slurm/compare_agents.py`.

    The results are stored in a pandas dataframe, with one row per game.
    Each row also contains the name of the run it came from, and a set of
    representative config parameters from that run.

    Arguments:
        - folders: a set of strings, each one containing a path to a
                    supervised learning checkpoint dir.
    Returns:
        - train_curves: a pandas dataframe with training learning curves.
        - val_curves: a pandas dataframe with validation learning curves.
    """

    results = []
    for folder in folders:
        one_config = Path(glob.glob(f"{folder}/*_ENG.0")[0]) / "config_meta.prototxt"
        config = heyhi.load_root_config(one_config, overrides=[])
        config_dict = flatten_dict(message_to_dict(config.compare_agents))
        paths = glob.glob(f"{folder}/game_*.json")
        for path in paths:
            result = get_game_result_from_json(path)
            if result is None:
                continue
            stats = result[3]._asdict()
            stats["power"] = result[1]
            stats["name"] = os.path.basename(folder)
            stats["path"] = path
            stats.update(config_dict)
            results.append(stats)

    return pd.DataFrame(results)


def make_notebook_wide():
    """Executing this in a notebook will adjust the css to use the full horizontal space."""
    IPython.core.display.display(
        IPython.core.display.HTML("<style>.container { width:96% !important; }</style>")
    )
    IPython.core.display.display(
        IPython.core.display.HTML("<style>.output_result { max-width:96% !important; }</style>")
    )


def iframe(raw_html: str, height: int = 1000):
    """Executing and rendering this within a notebook will produce iframe with the html in the output cell."""
    iframe = f'<iframe srcdoc="{html.escape(raw_html)}" width=100%% height={height}></iframe>'
    return IPython.display.HTML(iframe)


def get_game_after_joint_action(game: Game, joint_action: Dict[str, Sequence[str]]) -> Game:
    """Convenience function - returns a copy of the game with joint_action applied"""
    game = Game(game)
    for power in joint_action:
        game.set_orders(power, joint_action[power])
    game.process()
    return game


def load_game(path: Union[pathlib.Path, str]) -> Game:
    """Load a pydipcc game."""
    with open(path) as f:
        game = Game.from_json(f.read())
    return game


def set_joint_action(game: Game, joint_action: JointAction):
    for power, orders in joint_action.items():
        game.set_orders(power, orders)
