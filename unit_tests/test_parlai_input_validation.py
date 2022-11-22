#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import random
import unittest
from typing import Dict

from parlai_diplomacy.utils.game2seq import input_validation as R
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
from parlai_diplomacy.utils.game2seq.format_helpers.misc import get_input_format, get_example_key
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.models.consts import POWERS

UNIT_TEST_DIR = os.path.dirname(__file__)


class InputValidationTests(unittest.TestCase):
    """
    This is a comprehensive test which validates inputs for players/phases of a
    game for various (task, opt) pairs that we actually care about.

    For each set of opts [1] we generate all training examples from an
    anonymized game, build the regex validator, and validate all inputs.

    We test the opts for each model that is used in any config since August 2021:

        find conf/common \
            | grep 2021 | grep -v $(for d in $(seq 202101 202107) ; do echo -n " -e $d" ; done) \
            | xargs grep -w model_path \
            | grep  'model"$' \
            | rev | cut -d' ' -f1 | rev | sort | uniq | sed 's/"//g' \
            | sed 's/$/.opt/'

    And then filter out the opts with ','-separated tasks (not currently supported)
    """

    def test_input_validation(self):
        with open(os.path.join(UNIT_TEST_DIR, "data/game_100012_anonymized.json")) as f:
            game = Game.from_json(f.read()).rolled_back_to_phase_start("S1903M")

        with open(os.path.join(UNIT_TEST_DIR, "data/input_validation_opts_to_test.json")) as f:
            opts = json.load(f)

        for opt in opts:
            fmt = opt["task"]

            opt = {**opt, "extend_state_history_since_last_n_movement_phase": 0}

            print(fmt, opt.get("rollout_pseudo_orders"))
            _metadata = {
                "opt": opt,
                "power_metadata": {p: {"rating": i % 5 + 1} for i, p in enumerate(POWERS)},
                "game_id": 12345,
                "anon": "ANON",
                "phase_minutes": "1440",
                "pot_type": "SOS",
                "all_unknowns": random.choice(["ALL-UNK", None]),
            }
            for version in [1]:
                formatter = sequence_formatter_factory(fmt, version=version)
                pseudo_orders_metadata = get_pseudo_orders_metadata(
                    game, _metadata["game_id"], formatter, opt
                )
                metadata = {**_metadata, "pseudo_orders": pseudo_orders_metadata}
                regex = formatter.get_input_validation_regex(fmt, opt)
                all_exs = formatter.change_format(game, get_input_format(fmt), metadata)
                for phase in all_exs.keys():
                    for power in all_exs[phase].keys():
                        for recipient_i, ex in enumerate(all_exs[phase][power]):
                            try:
                                R.validate(regex, ex["input"])
                            except ValueError:
                                print("", fmt, type(formatter), opt, sep="\n\n")
                                raise


def get_pseudo_orders_metadata(game: Game, game_id: int, formatter, opt) -> Dict:
    """
    Formats which require pseudo orders expect pre-formatted pseudo order
    strings in the metadata field. At train time, the pre-formatted strings are
    read from disk. For these tests, we pre-format them here and return a dict
    that can be saved to metadata['pseudo_orders']
    """
    d = {}
    all_phase_data = game.get_all_phases()
    for phase_i, phase_data in enumerate(all_phase_data):
        d[phase_data.name] = {}
        for ind in range(len(phase_data.messages) + 1):
            for speaker in POWERS:
                ex_key = get_example_key(game_id, speaker, phase_data.name, ind + 1)
                if opt.get("single_view_pseudo_orders"):
                    if opt.get("rollout_pseudo_orders"):
                        # get rollout action from phase dicts
                        rollout_action = {}
                        for i in range(phase_i, min(phase_i + 3, len(all_phase_data))):
                            rollout_action[all_phase_data[i].name] = all_phase_data[i].orders.get(
                                speaker, ()
                            )
                            if all_phase_data[i].name.endswith("M"):
                                break
                        # Flatten and use for both self and partner (doesn't matter, we're just checking formatting).
                        # Keep current phase (in R/A phases) because flatten_train_singleview_pseudo_orders strips it :/
                        flattened = formatter.orders_flattener.flatten_rollout_action(
                            rollout_action, strip_current_phase=phase_data.name.endswith("M")
                        )
                        prefix = "" if phase_data.name.endswith("M") else "rollout_"
                        d[ex_key] = {
                            f"{prefix}self": flattened,
                            f"{prefix}partner": flattened,
                        }
                    else:
                        # flatten and use for both self and partner (doesn't matter, we're just checking formatting)
                        flattened = formatter.orders_flattener.flatten_action(
                            phase_data.orders.get(speaker, ())
                        )
                        d[ex_key] = {
                            "self": flattened,
                            "partner": flattened,
                        }
                else:
                    d[ex_key] = formatter.orders_flattener.flatten_joint_action(
                        phase_data.orders, speaker
                    )
    return d
