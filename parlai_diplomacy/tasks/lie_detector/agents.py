#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import json
import logging
import random
import re
from collections import defaultdict
from typing import Tuple, Dict, Optional

import torch
from parlai.core.loader import register_teacher
from tqdm import tqdm

from fairdiplomacy.typedefs import Power
from fairdiplomacy.models.consts import POWERS
from parlai_diplomacy.tasks.base_diplomacy_agent import BaseDiplomacyTeacher
from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.tasks.common_task_utils import LIE_TOKEN, NOT_LIE_TOKEN
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import get_last_speaker


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_liedetector_chunk"
)
class LieDetectorChunkTeacher(BaseDialogueChunkTeacher):
    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        BaseDiplomacyTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)
        argparser.add_argument(
            "--lie-detector-annotations-glob",
            help="DEPRECATED, use --lie-detector-annotations-dir",
        )
        argparser.add_argument(
            "--lie-detector-threshold",
            type=float,
            default=None,
            help="Train-time classification threshold",
        )
        argparser.add_argument(
            "--lie-detector-debug-no-annotations",
            action="store_true",
            help="If set, skip reading annotations, use random labels",
        )

    def __init__(self, opt, shared=None):
        self.id = "Lie Detector Chunk"

        if shared is None:
            self.annotations = self.read_annotations(opt)
        else:
            self.annotations = shared["annotations"]

        # N.B. this needs to go at the end to avoid strange behaviour where
        # _generate_example_tuples is called on this object in a thread
        # before this __init__ has completed. Here we make sure this
        # initialization is complete before calling the parent __init__ which
        # launches this thread.
        super().__init__(opt, shared)

    def read_annotations(self, opt) -> Dict:
        if opt.get("lie_detector_debug_no_annotations"):
            logging.warning("DEBUGGING ONLY skip reading lie detector annotations")
            return defaultdict(lambda: LIE_TOKEN if random.random() < 0.01 else NOT_LIE_TOKEN)

        assert opt.get(
            "lie_detector_annotations_glob"
        ), "Must specify --lie-detector-annotations-glob"
        assert opt.get("lie_detector_threshold"), "Must specify --lie-detector-threshold"

        logging.info("Reading lie detector annotations")
        annotations = {}
        globbed = glob.glob(opt["lie_detector_annotations_glob"])
        logging.warning(f"Reading {len(globbed)} lie detector annotation files")
        for fname in tqdm(globbed):
            with open(fname, "r") as f:
                for line in f:
                    d = json.loads(line)
                    if "version" in d:
                        raise NotImplementedError(
                            "parse new lie detector annotation format with metadata header"
                        )
                    key, score = parse_annotation(d)
                    annotations[key] = (
                        LIE_TOKEN if score > opt["lie_detector_threshold"] else NOT_LIE_TOKEN
                    )
        logging.warning(f"Finished reading lie detector annotation files")
        return annotations

    def share(self):
        shared = super().share()
        shared["annotations"] = self.annotations
        return shared

    @property
    def model_type(self) -> str:
        return "lie_detector"

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        ret = []
        for ex in examples:
            logging.info(ex)
            power = ex["player"]
            last_msg_sender, last_msg_phase = get_last_speaker(ex["text"])
            if (
                last_msg_sender == power  # only consider incoming messages
                or last_msg_sender not in POWERS
                or last_msg_phase != ex["phase_id"]  # only incoming messages this phase
            ):
                continue
            if not ex["phase_id"].endswith("M"):
                # only annotated M-phases for now
                continue
            key = (int(game_id), ex["phase_id"], last_msg_sender, power)
            try:
                ex["labels"] = [self.annotations[key]]
            except KeyError:
                continue
            ret.append(ex)

        return ret


# DEPRECATED, parsing is now done in fairdiplomacy.annotate.lie_detection
# Keeping around temporarily for some notebooks
def parse_annotation(d: Dict) -> Tuple[Tuple[int, str, Power, Power], float]:
    game_id = re.findall(r"game_([0-9]*)\.json", d["game_path"])[0]
    key = (int(game_id), d["m_phase"], d["lyer"], d["lyee"])
    if "scores" in d:
        # This is an old version of the annotations which logged NLL values. They are combined
        # using logsumexp(-1 * vals)
        logp = torch.logsumexp(
            torch.tensor([-neglogp for cand, neglogp in d["scores"]]), dim=0
        ).item()
    else:
        # This is the current version of the annotations which logs log probs. They are combined
        # using logsumexp(vals)
        logp = torch.logsumexp(torch.tensor([logp for cand, logp in d["logps"]]), dim=0).item()
    return key, float(logp)
