#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import pathlib
from typing import Dict

import torch

from conf import conf_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.env import Env, PopulationPolicyProfile
from fairdiplomacy.typedefs import Power
from fairdiplomacy.utils.yearprob import parse_year_spring_prob_of_ending

from fairdiplomacy.viz.meta_annotations.api import maybe_kickoff_annotations


def run_population_trial(
    power_agent_dict: Dict[Power, BaseAgent], cfg: conf_cfgs.CompareAgentsTask, cf_agent=None,
):
    """Run a population trial

    Arguments:
    - power_agent_dict: mapping between power and the corresponding agent
    - cfg: see conf.proto

    Returns winning_power is a power wins and None, if no agent wins
    """
    torch.set_num_threads(1)

    if cfg.start_game:
        with open(cfg.start_game) as stream:
            game_obj = pydipcc.Game.from_json(stream.read())
        if cfg.start_phase:
            game_obj = game_obj.rolled_back_to_phase_start(cfg.start_phase)
    else:
        game_obj = pydipcc.Game()

    if cfg.draw_on_stalemate_years is not None and cfg.draw_on_stalemate_years > 0:
        game_obj.set_draw_on_stalemate_years(cfg.draw_on_stalemate_years)

    year_spring_prob_of_ending = parse_year_spring_prob_of_ending(cfg.year_spring_prob_of_ending)

    policy_profile = PopulationPolicyProfile(power_agent_dict=power_agent_dict, game=game_obj)

    env = Env(
        policy_profile=policy_profile,
        seed=cfg.seed,
        cf_agent=cf_agent,
        max_year=cfg.max_year,
        max_msg_iters=cfg.max_msg_iters,
        game=game_obj,
        capture_logs=cfg.capture_logs,
        time_per_phase=cfg.time_per_phase,
        year_spring_prob_of_ending=year_spring_prob_of_ending,
    )

    if cfg.out is not None:
        pathlib.Path(cfg.out).parent.mkdir(exist_ok=True, parents=True)

    partial_out_name = cfg.out + ".partial" if cfg.out else None
    annotations_out_name = (
        pathlib.Path(cfg.out.rsplit(".", 1)[0] + ".metann.jsonl") if cfg.out else None
    )
    with maybe_kickoff_annotations(env.game, annotations_out_name):
        scores = env.process_all_turns(max_turns=cfg.max_turns, partial_out_name=partial_out_name)

    if cfg.out:
        env.save(cfg.out)
    if all(s < 18 for s in scores.values()):
        winning_power = "NONE"
    else:
        winning_power = max(scores, key=lambda x: scores[x])

    logging.info(f"Scores: {scores} ; Winner: {winning_power} ;")
    return winning_power
