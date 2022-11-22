#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import logging
import pathlib
import time
from typing import Dict, Optional

import heyhi
import numpy as np
import torch
import torch.cuda
import torch.utils.tensorboard
from conf import misc_cfgs
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.selfplay.paths import get_tensorboard_folder
from fairdiplomacy.selfplay.search.search_utils import unparse_device
from fairdiplomacy.situation_check import run_situation_checks
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx

mp = get_multiprocessing_ctx()


class TestSituationEvaller:
    """A process that run situation_check on a single GPU."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.p = ExceptionHandlingProcess(
            target=self.worker, args=[], kwargs=self.kwargs, daemon=True
        )
        self.p.start()

    @classmethod
    def worker(
        cls, *, cfg, agent_cfg, ckpt_dir: pathlib.Path, device: str, log_file: Optional[str]
    ):
        if log_file is not None:
            heyhi.setup_logging(console_level=None, fpath=log_file, file_level=logging.INFO)

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=get_tensorboard_folder())  # type: ignore

        last_ckpt = None
        get_ckpt_id = lambda p: int(p.name.split(".")[0][5:])
        agents: Dict[str, Dict] = {}
        agents["bl"] = dict()
        agents["rol0_it256"] = {"rollouts_cfg.max_rollout_length": 0, "n_rollouts": 256}
        agents["rol1_it256"] = {"rollouts_cfg.max_rollout_length": 1, "n_rollouts": 256}
        agents["rol2_it256"] = {"rollouts_cfg.max_rollout_length": 2, "n_rollouts": 256}

        situation_set: misc_cfgs.GameSituationSet = heyhi.load_config(
            heyhi.PROJ_ROOT / "data" / "game_situations" / "no_press.auto.ascii_proto",
            msg_class=misc_cfgs.GameSituationSet,
        )

        while True:
            newest_ckpt = max(ckpt_dir.glob("epoch*.ckpt"), key=get_ckpt_id, default=None)
            if newest_ckpt == last_ckpt:
                time.sleep(20)
                continue
            logging.info("Loading checkpoint: %s", newest_ckpt)
            last_ckpt = newest_ckpt
            for name, agent_kwargs in agents.items():
                agent = build_agent_from_cfg(
                    agent_cfg,
                    device=unparse_device(device),
                    value_model_path=last_ckpt,
                    **agent_kwargs,
                )
                scores = run_situation_checks(list(situation_set.situations), agent)
                del agent
                writer.add_scalar(
                    f"eval_test_sit/{name}",
                    float(np.mean(list(scores.values()))),
                    global_step=get_ckpt_id(last_ckpt),
                )

    def terminate(self):
        if self.p is not None:
            self.p.kill()
            self.p = None
