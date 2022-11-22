#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass
from typing import Optional, List, Dict

VERSION = 3


@dataclass
class LieDetectorAnnotation:
    # Annotations are keyed by M-phase. Lie scores are computed at the
    # beginning of the next phase.
    m_phase: str

    # Recipient of the "you lied to me!" message
    lyer: str

    # Sender of the "you lied to me!" message
    lyee: str

    # Logps of candidate sentences
    logps: List[float]

    # Next sentence actually uttered in-game by lyee to lyer
    actual_response: Optional[str] = None


@dataclass
class LieDetectorGameAnnotations:
    # Path to game.json being annotated
    game_path: str

    # Dialogue model used to score candidate sentences
    model_cfg: Dict

    # Candidate sentences being scored
    candidates: List[str]

    # List of annotations
    annotations: List[LieDetectorAnnotation]

    # Schema version
    version: int = VERSION
