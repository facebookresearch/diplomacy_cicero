#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List
import pathlib

from fairdiplomacy.utils.h2h_sweep import H2HItem, H2HSweep

THIS_FILE = pathlib.Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent


class MyH2HSweep(H2HSweep):

    SWEEP_NAME = pathlib.Path(__file__).name.rsplit(".")[0]
    NUM_SEEDS = 1  # How many games to run per power per agent pair.
    INITIAL_SEED = (
        10000  # Random seed to use, incremented per game with the same agents and power.
    )

    VARIANT = "CLASSIC"  # 7-player diplomacy
    # VARIANT="FVA"     # 2-player France vs Austria

    CAPTURE_LOGS = True

    def get_sort_order(self) -> List[str]:
        return []

    def get_eval_grid(self) -> List[H2HItem]:
        print("Getting eval grid for: " + MyH2HSweep.SWEEP_NAME)
        eval_items = []

        opponents = {
            "base_strategy_model": ("base_strategy_model_20200827_iclr_v_humans",),
        }

        agentstotest = {
            "human_dnvi_npu": ("searchbot_neurips21_human_dnvi_npu",),
        }

        for key in agentstotest:
            opponents[key] = agentstotest[key]

        pairs = []
        pairs_used = set()
        for pla in agentstotest:
            for opp in opponents:
                if pla == opp:
                    continue
                if (pla, opp) not in pairs_used:
                    pairs_used.add((pla, opp))
                    pairs.append((pla, agentstotest, opp, opponents))

        for agent_one, agent_one_detail, agent_six, agent_six_detail in pairs:
            eval_items.append(
                H2HItem(
                    agent_one=agent_one_detail[agent_one],
                    agent_six=agent_six_detail[agent_six],
                    row=agent_one,
                    col=agent_six,
                    exp_tag=(agent_one + "_X_" + agent_six),
                )
            )

        return eval_items


if __name__ == "__main__":
    MyH2HSweep.parse_args_and_go()
