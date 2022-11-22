#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import logging
import os
import pathlib
import random

import torch

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.data.dataset import Dataset
from fairdiplomacy.models.consts import POWERS


def make_ratings_table(game_jsons):
    game_jsons = map(os.path.abspath, game_jsons)
    game_stats = {
        game_id: {
            "id": game_id,
            "press_type": "NoPress",
            **{
                pwr: {"id": -1, "points": -1, "status": "NoPress", "logit_rating": 0, "total": 1}
                for pwr in POWERS
            },
        }
        for game_id in game_jsons
    }
    return game_stats


def build_db_cache_from_cfg(cfg):
    assert cfg.glob, cfg
    assert cfg.out_path, cfg
    no_press_cfg = cfg.dataset_params

    logging.info("Expanding the glob")
    game_json_paths = sorted(glob.glob(cfg.glob))
    logging.info("Found games: %s", len(game_json_paths))

    logging.info("Building metadata")
    game_metadata = make_ratings_table(game_json_paths)
    game_ids = list(game_metadata.keys())

    random.seed(0)
    val_game_ids = sorted(random.sample(game_ids, int(len(game_ids) * cfg.val_set_pct)))
    train_game_ids = sorted(set(game_ids) - set(val_game_ids))

    kwargs = dict(
        # Using full paths as game ids.
        data_dir="SHOULD_NOT_BE_USED",
        game_metadata=game_metadata,
        only_with_min_final_score=no_press_cfg.only_with_min_final_score,
        num_dataloader_workers=no_press_cfg.num_dataloader_workers,
        min_rating=-1e9,
        cf_agent=(
            None
            if cfg.cf_agent.WhichOneof("agent") is None
            else build_agent_from_cfg(cfg.cf_agent)
        ),
        n_cf_agent_samples=cfg.n_cf_agent_samples,
    )

    if len(val_game_ids) > 0:
        logging.info("Building val dataset")
        val_dataset = Dataset(game_ids=val_game_ids, **kwargs)
        val_dataset.preprocess()
    else:
        val_dataset = None

    logging.info("Building train dataset")
    train_dataset = Dataset(game_ids=train_game_ids, **kwargs)
    train_dataset.preprocess()

    pathlib.Path(cfg.out_path).parent.mkdir(exist_ok=True, parents=True)

    for dataset in (val_dataset, train_dataset):
        if hasattr(dataset, "cf_agent"):
            del dataset.cf_agent

    logging.info("Saving")
    torch.save((train_dataset, val_dataset), cfg.out_path)
