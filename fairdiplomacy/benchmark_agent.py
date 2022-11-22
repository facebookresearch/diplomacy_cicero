#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional
import logging
import pathlib
import random

import numpy as np
import torch
from fairdiplomacy.agents.player import Player

from fairdiplomacy.pydipcc import Game
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.utils.timing_ctx import TimingCtx
import conf.conf_cfgs


def load_game(game_json: Optional[str], phase: Optional[str]) -> Game:
    if not game_json:
        return Game()

    with open(game_json) as f:
        game = Game.from_json(f.read())
    if phase:
        game = game.rolled_back_to_phase_start(phase)
    return game


def run(cfg: conf.conf_cfgs.BenchmarkAgent):
    assert cfg.power is not None, "power is required"

    logger = logging.getLogger("timing")
    logger.setLevel(logging.DEBUG)
    timing_outpath = pathlib.Path(".").resolve() / "timing.log"
    logger.info("Will write timing logs to %s", timing_outpath)
    logger.addHandler(logging.FileHandler(timing_outpath))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # type:ignore
    torch.manual_seed(cfg.seed)

    game = game_from_view_of(load_game(cfg.game_json, cfg.phase), cfg.power)
    agent = build_agent_from_cfg(cfg.agent)
    player = Player(agent, cfg.power)

    logger.info("Warmup")
    player.get_orders(game)

    logger.info("Running benchmark")
    timings = TimingCtx()
    assert cfg.repeats is not None
    for _ in range(cfg.repeats):
        # Re-build player to reset
        player = Player(agent, cfg.power)

        if cfg.generate_message:
            for i in range(cfg.num_messages):
                with timings(f"message.{i + 1}"):
                    player.generate_message(game)
        else:
            with timings("orders"):
                player.get_orders(game)
    logger.info("### RESULTS ###")
    timings.pprint(logger.info)
