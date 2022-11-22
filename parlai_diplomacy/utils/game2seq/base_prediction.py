#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import json
from parlai_diplomacy.utils.game2seq.format_helpers.state import StateFlattener
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageHistoryBuilder,
    MessageHistoryFlattener,
    MessageHistoryUnflattener,
)
from typing import Any, Tuple, Optional, List, Dict

from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import GameJson
from fairdiplomacy.data.build_dataset import DATASET_NODRAW_MESSAGE, UNDRAW_VOTE_TOKEN

from parlai_diplomacy.utils.game2seq.typing import Metadata, DiplomacySequencePart
from parlai_diplomacy.utils.game2seq.format_helpers.orders import (
    OrdersFlattener,
    OrdersUnflattener,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    organize_game_by_phase,
    convert_all_timestamps,
    get_output_type,
    get_input_format,
)
from parlai_diplomacy.utils.game2seq.input_validation import InputValidator


class BaseDiplomacyPredictionFormatter(ABC):
    """
    Base Diplomacy prediction formatter.

    This base class defines an API for taking in a game json and returning an
    object of input and output (output is optional, training only) string
    sequences.

    Child classes must define two functions:
    - get_format_parts: given a string (fmt) return a List of strings corresponding
        to the sequence part
    - generate_input_output_pairs: given a game_json, metadata, a list of format_parts,
        and possibly other args, return the object input/output sequences.
    """

    def __init__(self, version: int):
        self.version = version
        # Initialize order and dialogue and state helpers
        self.orders_flattener = OrdersFlattener(version)
        self.orders_unflattener = OrdersUnflattener(version)
        self.messagehistory_builder = MessageHistoryBuilder(version)
        self.messagehistory_flattener = MessageHistoryFlattener(version)
        self.messagehistory_unflattener = MessageHistoryUnflattener(version)
        self.state_flattener = StateFlattener(version)

        # In V2, we change the delimiter to be a "\n"
        self.delimiter = " " if self.version <= 1 else "\n"

    def change_format(self, game: pydipcc.Game, fmt: str, metadata: Metadata, *args, **kwargs):
        game_json, fmt, metadata = self._clean_game_fmt_metadata(game, fmt, metadata)
        format_parts = self.get_format_parts(fmt)
        return self.generate_input_output_pairs(game_json, metadata, format_parts, *args, **kwargs)

    def get_input_validation_regex(self, fmt: str, opt: Dict) -> str:
        opt["shortstate"] = "shortstate" in fmt
        fmt = fmt.split(",")[0].replace("shortstate", "state")
        format_parts = self.get_format_parts(get_input_format(fmt))
        return InputValidator(
            format_parts, get_output_type(fmt), opt, self.version
        ).get_input_validation_regex()

    @abstractmethod
    def get_format_parts(self, fmt: str) -> List[DiplomacySequencePart]:
        """
        Given a format string, like `message_history_state_order`, return a list
        of squence parts, like [MESSAGEHISTORY, STATE].

        Must be defined by child class.
        """
        pass

    @abstractmethod
    def generate_input_output_pairs(
        self,
        game_json: GameJson,
        metadata: Metadata,
        format_parts: List[DiplomacySequencePart],
        *args
    ) -> Any:
        """
        Given a game_json, metadata, a list of format_parts, and possibly other args,
        return an object of input/output sequences.

        Return object structure is defined by the child class
        """
        pass

    def _clean_game_fmt_metadata(
        self, game: pydipcc.Game, fmt: str, metadata: Optional[Metadata]
    ) -> Tuple[GameJson, str, Metadata]:
        """
        Do any pre-processing of the game object and metadata.

        Returns a game JSON (restructured to be keyed on phase), the format
        string, and metadata
        """
        game_json = json.loads(game.to_json())
        game_json_by_phase = organize_game_by_phase(game_json)
        game_json_by_phase = convert_all_timestamps(game_json_by_phase)

        # In data_20211214, no-vote draws are encoded as self-messages
        # "Un-voted for Draw" Here we hot-swap them for public <NODRAW> tokens,
        # which is how they are represented in later dataset versions.
        for phase in game_json_by_phase.values():
            for m in phase.get("messages", []):
                if m["sender"] == m["recipient"] and m["message"] == DATASET_NODRAW_MESSAGE:
                    m["message"] = UNDRAW_VOTE_TOKEN
                    if metadata and metadata.get("draw_type") == "PUBLIC":
                        m["recipient"] = "ALL"

        assert metadata is not None
        assert fmt is not None

        if "shortstate" in fmt:
            metadata["shortstate"] = True
            fmt = fmt.replace("shortstate", "state")

        return game_json_by_phase, fmt, metadata
