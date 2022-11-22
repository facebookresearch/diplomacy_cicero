#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Union
import hashlib
import logging
import os
import pathlib

import heyhi
from conf import conf_cfgs

import wandb


def initialize_wandb_if_enabled(
    cfg: Union[conf_cfgs.TrainTask, conf_cfgs.ExploitTask], default_project_name: str
) -> bool:
    wandb_cfg = cfg.wandb
    if not wandb_cfg.enabled or heyhi.is_adhoc():
        return False
    logging.info("Using WandB")

    exp_dir = pathlib.Path(os.getcwd()).absolute()
    wandb_kwargs = dict(
        project=wandb_cfg.project or default_project_name,
        group=wandb_cfg.group or None,
        entity="fairdiplomacy",
        tags=(wandb_cfg.tags.split(",") if wandb_cfg.tags else []),
        config=heyhi.flatten_cfg(cfg, with_all=False),
        resume=True,
    )
    if wandb_cfg.name:
        wandb_kwargs["name"] = wandb_cfg.name
        logging.info("Setting explicit WandB run name: %s", wandb_kwargs["name"])
    else:
        wandb_kwargs["name"] = (
            exp_dir.name + "_" + hashlib.md5(str(exp_dir).encode()).hexdigest()[:4]
        )
        logging.info("Setting auto-generated WandB run name: %s", wandb_kwargs["name"])
    logging.info("Wandb params: %s", wandb_kwargs)
    wandb.init(**wandb_kwargs)
    wandb.config.exp_dir = exp_dir
    return True
