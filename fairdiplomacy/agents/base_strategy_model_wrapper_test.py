#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from typing import List

import numpy as np

import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_strategy_model_wrapper import compute_action_logprobs_from_state
from fairdiplomacy.models.base_strategy_model import load_model
from fairdiplomacy.typedefs import Action, Power


def test_compute_bilateralaction_logprobs_from_state():
    def _get_some_actions(power: Power, count: int) -> List[Action]:
        locs = game.get_orderable_locations()[power]
        actions = []
        for _ in range(count):
            actions.append(
                tuple(random.choice(game.get_all_possible_orders()[loc]) for loc in locs)
            )
        return actions

    sl_cfg = heyhi.load_config(
        heyhi.CONF_ROOT / "c02_sup_train" / "sl_20211119_base.prototxt",
        [
            "use_v2_base_strategy_model=1",
            "num_encoder_blocks=1",
            "all_powers=1",
            "all_powers_add_double_chances=0.5",
        ],
    ).train
    base_strategy_model = load_model.new_model(sl_cfg)
    base_strategy_model.eval()
    game = pydipcc.Game()
    powers = ["FRANCE", "AUSTRIA"]
    nactions = 10
    actions = [_get_some_actions(power, nactions) for i, power in enumerate(powers)]
    power_action_dicts = [
        {powers[0]: actions[0][i], powers[1]: actions[1][i]} for i in range(nactions)
    ]

    logprobs = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        power_action_dicts,
        batch_size=512,
        has_press=False,
        agent_power=None,
    )
    assert len(logprobs) == nactions
    assert all(logprob <= 0 for logprob in logprobs), "Expected logprobs"

    power_action_dicts = power_action_dicts * 2

    logprobs_again = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        power_action_dicts,
        batch_size=512,
        has_press=False,
        agent_power=None,
    )
    np.testing.assert_allclose(logprobs, logprobs_again[:nactions], rtol=1e-5)
    np.testing.assert_allclose(logprobs, logprobs_again[nactions:], rtol=1e-5)


def test_compute_single_logprobs_from_state():
    def _get_some_actions(power: Power, count: int) -> List[Action]:
        locs = game.get_orderable_locations()[power]
        actions = []
        for _ in range(count):
            actions.append(
                tuple(random.choice(game.get_all_possible_orders()[loc]) for loc in locs)
            )
        return actions

    sl_cfg = heyhi.load_config(
        heyhi.CONF_ROOT / "c02_sup_train" / "sl_20211119_base.prototxt",
        [
            "use_v2_base_strategy_model=1",
            "num_encoder_blocks=1",
            "all_powers=1",
            "all_powers_add_single_chances=0.5",
            "all_powers_add_double_chances=0.5",
        ],
    ).train
    base_strategy_model = load_model.new_model(sl_cfg)
    base_strategy_model.eval()
    game = pydipcc.Game()
    powers = ["FRANCE", "AUSTRIA"]
    nactions = 10
    actions = [_get_some_actions(power, nactions) for i, power in enumerate(powers)]
    # Alternative frace/austria powers.
    power_action_dicts = [{powers[i % 2]: actions[i % 2][i]} for i in range(nactions)]

    logprobs = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        power_action_dicts,
        batch_size=512,
        has_press=False,
        agent_power=None,
    )
    assert len(logprobs) == nactions
    assert all(logprob <= 0 for logprob in logprobs), "Expected logprobs"

    power_action_dicts = power_action_dicts * 2

    logprobs_again = compute_action_logprobs_from_state(
        base_strategy_model,
        game,
        power_action_dicts,
        batch_size=512,
        has_press=False,
        agent_power=None,
    )
    np.testing.assert_allclose(logprobs, logprobs_again[:nactions], rtol=1e-5)
    np.testing.assert_allclose(logprobs, logprobs_again[nactions:], rtol=1e-5)
