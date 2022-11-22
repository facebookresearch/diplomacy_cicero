#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional

import json
import pathlib

import pandas as pd
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.selfplay.search.rollout import ReSearchRolloutBatch


def _rollout_batch_to_dataframe(tensors: ReSearchRolloutBatch) -> pd.DataFrame:
    data = {}
    for i, p in enumerate(POWERS):
        data[f"reward_{p}"] = tensors.rewards[:, i].numpy()
    data["done"] = tensors.done.numpy()
    for i, p in enumerate(POWERS):
        data[f"is_explore_{p}"] = tensors.is_explore[:, i].numpy()
    df = pd.DataFrame(data)
    df.index.name = "timestamp"
    return df


def save_game(
    *,
    game_json: str,
    epoch: int,
    dst_dir: pathlib.Path,
    game_id: str,
    start_phase: str,
    tensors: Optional[ReSearchRolloutBatch] = None,
    agent_one_power: Optional[str] = None,
) -> None:
    counter = 0
    while True:
        name = f"game_{epoch:06d}_{counter:05d}_{game_id}"
        if agent_one_power:
            name += f"_{agent_one_power}"
        path = dst_dir / f"{name}.json"
        path_meta = dst_dir / f"{name}.meta.csv"
        if not path.exists():
            break
        counter += 1
    game_dict = json.loads(game_json)
    game_dict["viz"] = dict(game_id=game_id, start_phase=start_phase)
    with path.open("w") as stream:
        json.dump(game_dict, stream)
    if tensors is not None:
        _rollout_batch_to_dataframe(tensors).to_csv(path_meta)
