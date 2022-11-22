#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Function to read and write anotations for phases and messages.

The main granulariry level here is RawRecord. It knows where in the game the
stored information belongs, but it's agnostic of the data - the data represented
as `str` blob.

The file provides function to read and write some blobs.

MetaAnnotator is a class that allows to write annotations for a game as it's
being generated. It allows the callers to indirectly refer to the corresdping
points in the game by using LAST_MSG, NEXT_MSG and LAST_PHASE named constant
instead of providing the message timestamp and the phase directly.

read_raw_records reads RawRecord's from a file produced by MetaAnnotor. It
cleans up inconsistent annotations produced due to requeue.
"""
import copy
import dataclasses
import datetime
import json
import logging
import pathlib
import traceback
from typing import ClassVar, Dict, List, Optional, cast
from typing_extensions import TypedDict

from dacite.core import from_dict

from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import MessageDict
from fairdiplomacy.viz.meta_annotations import datatypes


class UnmatchedAnnotationsError(Exception):
    pass


# Special values for phase and timestamp to be resolved later.
CURRENT_PHASE = "CURRENT_PHASE"
LAST_MSG = -101
NEXT_MSG = -100


@dataclasses.dataclass
class GamePointer:
    phase: str
    timestamp: Optional[int]
    # ISO-format
    datetime: str
    phase_hash: str


@dataclasses.dataclass
class RawRecord:
    pointer: GamePointer
    tag: str
    version: int
    # Seralized data.
    json_data: str


class MessageFilteredData(TypedDict):
    """Data for MESSAGE_FILTERED_TAG tag."""

    bad_tags: Optional[List[str]]


class MetaAnnotator:
    START_TAG: ClassVar[str] = "CONTROL_START"
    COMMIT_TAG: ClassVar[str] = "COMMIT_TAG"
    MESSAGE_FILTERED_TAG: ClassVar[str] = "MESSAGE_FILTERED_TAG"
    AFTER_MESSAGE_ADDED_TAG: ClassVar[str] = "AFTER_MESSAGE_ADDED_TAG"

    CONTROL_TAGS_VERSION = 2

    def __init__(
        self,
        game: pydipcc.Game,
        outpath: pathlib.Path,
        silent: bool = False,
        skip_start_tag: bool = False,
    ):
        # Consts.
        self.outpath = outpath

        # State.
        self.last_msg_timestamp: Optional[int] = list(game.messages)[-1] if game.messages else None
        self.current_short_phase = game.current_short_phase
        self.game_phase_hash = game.compute_order_history_hash()
        self.records: List[RawRecord] = []

        self._assert_game_in_sync(game)

        if not skip_start_tag:
            self.add_annotation(
                encoded_data={},
                tag=self.START_TAG,
                phase=self.current_short_phase,
                timestamp=self.last_msg_timestamp,
                version=self.CONTROL_TAGS_VERSION,
            )
            if not silent:
                logging.info(
                    "Initialized MetaAnnotator at phase=%s game_obj=%s",
                    self.current_short_phase,
                    game,
                )

    def __eq__(self, other):
        return self.records == other.records

    def after_new_phase(self, game: pydipcc.Game):
        """Call this when orders added to the game AND the game was stepped."""
        assert (
            not game.get_orders()
        ), f"Expected to be called right after game.process_orders(). Found orders: {game.get_orders()}"
        self.last_msg_timestamp = None
        old_phase = self.current_short_phase
        self.current_short_phase = game.current_short_phase
        self.game_phase_hash = game.compute_order_history_hash()
        logging.info(
            "Calling after_new_phase for phase=%s (was %s)", self.current_short_phase, old_phase
        )

    def after_message_add(self, last_message: MessageDict) -> None:
        """Call this when a new message was added to the game."""
        self.add_annotation(
            encoded_data={"msg": last_message},
            tag=self.AFTER_MESSAGE_ADDED_TAG,
            phase=self.current_short_phase,
            timestamp=last_message["time_sent"],
            version=self.CONTROL_TAGS_VERSION,
        )
        self.last_msg_timestamp = last_message["time_sent"]

    def after_message_generation_failed(self, *, bad_tags: Optional[List[str]] = None) -> None:
        data: MessageFilteredData = dict(bad_tags=bad_tags)  # type: ignore
        self.add_annotation(
            encoded_data=data,
            tag=self.MESSAGE_FILTERED_TAG,
            phase=self.current_short_phase,
            timestamp=None,
            version=self.CONTROL_TAGS_VERSION,
        )

    def _assert_game_in_sync(self, game: pydipcc.Game) -> None:
        if game.current_short_phase != self.current_short_phase:
            logging.error(
                "Data for a wrong game phase: %s != %s. Forgot to call after_new_phase? TB:\n%s",
                game.current_short_phase,
                self.current_short_phase,
                "".join(traceback.format_stack()),
            )
        if game.compute_order_history_hash() != self.game_phase_hash:
            logging.error("Data for a wrong phase hash! Forgot to call after_new_phase?")

    def commit(self, game: pydipcc.Game) -> None:
        logging.info("Calling commit for phase=%s", self.current_short_phase)
        self._assert_game_in_sync(game)
        if len(self.records) == 1 and self.records[0].tag == self.START_TAG:
            # No reason to bump the annotation file.
            return
        self.add_annotation(
            encoded_data={},
            tag=self.COMMIT_TAG,
            phase=self.current_short_phase,
            timestamp=self.last_msg_timestamp,
            version=self.CONTROL_TAGS_VERSION,
        )
        with self.outpath.open("a") as stream:
            for record in self.records:
                assert record.pointer.timestamp != LAST_MSG, record
                print(json.dumps(dataclasses.asdict(record)), file=stream)
        self.records = []

    def add_annotation(
        self,
        *,
        encoded_data: datatypes.SerializableState,
        tag: str,
        phase: str,
        version: int,
        timestamp: Optional[int],
    ) -> None:
        if phase == CURRENT_PHASE:
            phase = self.current_short_phase
        if timestamp == LAST_MSG:
            assert (
                self.last_msg_timestamp is not None
            ), "Cannot annotate last message and there is not last message in this phase"
            timestamp = self.last_msg_timestamp
            assert phase == self.current_short_phase, (phase, self.current_short_phase)

        record = RawRecord(
            pointer=GamePointer(
                phase=phase,
                timestamp=timestamp,
                datetime=datetime.datetime.now().isoformat(),
                phase_hash=str(self.game_phase_hash),
            ),
            version=version,
            tag=tag,
            json_data=json.dumps(encoded_data),
        )
        logging.debug("Adding annotation tag=%s phase=%s timestamp=%s", tag, phase, timestamp)

        self.records.append(record)

    def to_dict(self) -> Dict:
        records_dict = {"data": []}
        for record in self.records:
            records_dict["data"].append(dataclasses.asdict(record))

        return records_dict

    def load_state_dict(self, record_dicts):
        records = []
        for record_dict in record_dicts["data"]:
            records.append(from_dict(data_class=RawRecord, data=record_dict))

        self.records = records


def read_raw_records(path: pathlib.Path) -> List[RawRecord]:
    records: List[RawRecord] = []
    tmp_records: List[RawRecord] = []
    with path.open() as stream:
        for line in stream:
            record = from_dict(data_class=RawRecord, data=json.loads(line))
            if record.tag == MetaAnnotator.START_TAG:
                if tmp_records:
                    logging.warning("Found START tag before COMMIT. Probably requeue.")
                    logging.warning("Dropped %d messages", len(tmp_records))
                tmp_records = []
            elif record.tag == MetaAnnotator.COMMIT_TAG:
                records.extend(_resolve_next_message(tmp_records))
                tmp_records = []
            else:
                tmp_records.append(record)
    return records


def _resolve_next_message(records: List[RawRecord]) -> List[RawRecord]:
    next_message_timestamp = None
    new_records = []
    # if a list of strings, then filter only these tags
    filter_tags = []
    # Iterating through records in reverse-chronological time.
    for record in reversed(records):
        if record.pointer.timestamp == NEXT_MSG:
            if next_message_timestamp is None:
                logging.warning("Cannot resolve timestamp %s", record)
            elif filter_tags is None or record.tag in filter_tags:
                # if bad_tags field doesn't exist or includes this message tag, then filter
                continue
            else:
                record = copy.copy(record)
                record.pointer = copy.deepcopy(record.pointer)
                record.pointer.timestamp = next_message_timestamp
        elif record.tag == MetaAnnotator.AFTER_MESSAGE_ADDED_TAG:
            next_message_timestamp = record.pointer.timestamp
            filter_tags = []
            continue
        elif record.tag == MetaAnnotator.MESSAGE_FILTERED_TAG:
            parsed_data = cast(MessageFilteredData, json.loads(record.json_data))
            filter_tags = parsed_data["bad_tags"]
            continue
        new_records.append(record)
    # Reversing again to get chronological time.
    return list(reversed(new_records))
