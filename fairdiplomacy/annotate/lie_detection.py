#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Tuple

import torch
from dacite.core import from_dict

from fairdiplomacy.annotate.data.lie_detector_annotations import LieDetectorGameAnnotations
from fairdiplomacy.typedefs import Power, Phase, GameID
from fairdiplomacy.utils.game_id import extract_game_id_int

GameLieDetectorScores = Dict[Tuple[Power, Power], Dict[Phase, float]]


def read_lie_detector_annotations(
    annotations_dir: str, normalize: bool = True, use_tqdm: bool = False
) -> Dict[GameID, GameLieDetectorScores]:
    """Read lie detector annotation jsonl file and return annotations and metadata
    Returns a nested dict of game_id -> (lyer, lyee) -> phase -> score
    """
    # For debugging: skip slow file parsing
    if os.environ.get("USE_FAKE_LIE_DETECTOR_ANNOTATIONS"):
        return make_fake_lie_detector_annotations()  # type: ignore

    # check for pre-processed cache
    annotations_cache = annotations_dir.rstrip("/") + ".pkl"
    if os.path.exists(annotations_cache):
        logging.info(f"Reading lie detector annotations from {annotations_cache}")
        return pickle.load(open(annotations_cache, "rb"))

    annotations = {}

    sum_logp, sum_logp_sq, n = 0, 0, 0  # used to compute mean/var

    globbed = glob.glob(f"{annotations_dir}/zshot_*.json")
    if use_tqdm:
        globbed = tqdm(globbed)
    for p in globbed:
        with open(p, "r") as f:
            data = from_dict(LieDetectorGameAnnotations, json.load(f))

        assert data.version == 3, data.version

        game_id = extract_game_id_int(data.game_path)
        if game_id not in annotations:
            annotations[game_id] = {}

        for annotation in data.annotations:
            logp = torch.logsumexp(torch.tensor(annotation.logps), 0).item()
            pair = (annotation.lyer, annotation.lyee)
            if pair not in annotations[game_id]:
                annotations[game_id][pair] = {}
            annotations[game_id][pair][annotation.m_phase] = logp

            sum_logp += logp
            sum_logp_sq += logp * logp
            n += 1

    mean = sum_logp / n
    variance = (sum_logp_sq / n) - (mean * mean)
    stdev = variance ** 0.5
    logging.info(f"read_lie_detector_annotations mean={mean} stdev={stdev}")

    def recursively_normalize(d: Dict):
        for k, v in d.items():
            if isinstance(v, dict):
                recursively_normalize(v)
            elif isinstance(v, float):
                d[k] = (v - mean) / stdev
            else:
                raise ValueError(f"Something wrong: {type(v)}")

    if normalize:
        recursively_normalize(annotations)

    # write cache file for next time
    with open(annotations_cache, "wb") as f:
        logging.info(f"Writing lie detector annotations to {annotations_cache}")
        pickle.dump(annotations, f)

    return annotations


# can't pickle lambdas or locals, functions annoyingly need to be named globals
def __a():
    return random.random() * 20 - 40


def __b():
    return defaultdict(__a)


def __c():
    return defaultdict(__b)


def make_fake_lie_detector_annotations():
    return defaultdict(__c)
