#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

import torch

from conf import agents_cfgs
from conf import conf_cfgs
from fairdiplomacy.selfplay.search.rollout import yield_rollouts

DEFAULT_CFR_CFG = dict(
    model_path="MOCK",
    n_rollouts=2,
    device=-1,
    use_final_iter=0,
    rollouts_cfg=dict(max_rollout_length=0),
    plausible_orders_cfg=dict(n_plausible_orders=10, batch_size=10, req_size=10),
)

DEFAULT_EXTRA_PARAMS = dict(use_trained_policy=False, use_trained_value=False, max_year=1901)


def get_default_configs():
    agent_cfg = agents_cfgs.Agent(searchbot=DEFAULT_CFR_CFG).to_editable()
    extra_params_cfg = conf_cfgs.ExploitTask.SearchRollout.ExtraRolloutParams(
        **DEFAULT_EXTRA_PARAMS
    ).to_editable()
    return agent_cfg, extra_params_cfg


def produce_rollout(
    agent_cfg, extra_params_cfg, **kwargs,
):
    torch.manual_seed(0)
    default_kwargs = dict(
        game_json_paths=None,
        seed=1,
        device="cpu",
        extra_params_cfg=extra_params_cfg.to_frozen(),
        agent_cfg=agent_cfg.to_frozen(),
    )
    default_kwargs.update(kwargs)
    return next(iter(yield_rollouts(**default_kwargs)))


class YieldRolloutsTest(unittest.TestCase):
    def test_no_sync(self):
        agent_cfg, extra_params_cfg = get_default_configs()
        result = produce_rollout(agent_cfg, extra_params_cfg)
        # At least 2 move phases are expected.
        self.assertGreaterEqual(len(result.batch.done), 2)

    def test_no_sync_explore(self):
        agent_cfg, extra_params_cfg = get_default_configs()
        extra_params_cfg.explore_eps = 1.0
        result = produce_rollout(agent_cfg, extra_params_cfg)
        # At least 2 move phases are expected.
        self.assertGreaterEqual(len(result.batch.done), 2)
