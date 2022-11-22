#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import dataclasses
import json
import pathlib
import logging
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from fairdiplomacy import pydipcc
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.typedefs import MessageDict
from fairdiplomacy.viz.meta_annotations import annotator, datatypes

T = TypeVar("T")

_GLOBAL_ANNOTATOR: Optional[annotator.MetaAnnotator] = None


@dataclasses.dataclass
class ParsedData:
    @dataclasses.dataclass
    class CommonData:
        phase: str
        timestamp_centis: Optional[int]
        tag: str

    # We store different types of data in separate fields for type-checking reasons
    nonsenses: List[Tuple[CommonData, datatypes.NonsenseAnnotation]] = dataclasses.field(
        default_factory=list
    )
    pseudoorders: List[Tuple[CommonData, PseudoOrders]] = dataclasses.field(default_factory=list)
    misc: List[Tuple[CommonData, Dict]] = dataclasses.field(default_factory=list)


############################################
##### INITIALIZATION API
############################################
def has_annotator() -> bool:
    global _GLOBAL_ANNOTATOR
    return _GLOBAL_ANNOTATOR is not None


def get_annotator() -> annotator.MetaAnnotator:
    global _GLOBAL_ANNOTATOR
    assert _GLOBAL_ANNOTATOR is not None
    return _GLOBAL_ANNOTATOR


def pop_annotator() -> annotator.MetaAnnotator:
    global _GLOBAL_ANNOTATOR
    assert _GLOBAL_ANNOTATOR is not None

    cur_annotator = _GLOBAL_ANNOTATOR
    _GLOBAL_ANNOTATOR = None
    return cur_annotator


def push_annotator(new_annotator: annotator.MetaAnnotator,) -> annotator.MetaAnnotator:
    global _GLOBAL_ANNOTATOR
    assert _GLOBAL_ANNOTATOR is None, "Annotator already set! Not going to overwrite."
    _GLOBAL_ANNOTATOR = new_annotator
    return _GLOBAL_ANNOTATOR


def append_annotator(new_annotator: annotator.MetaAnnotator):
    global _GLOBAL_ANNOTATOR
    assert _GLOBAL_ANNOTATOR is not None
    _GLOBAL_ANNOTATOR.records.extend(new_annotator.records)


def start_global_annotation(game: pydipcc.Game, outpath: pathlib.Path) -> annotator.MetaAnnotator:
    global _GLOBAL_ANNOTATOR
    if _GLOBAL_ANNOTATOR is not None:
        logging.error("Rewritting existing annotator!")
    _GLOBAL_ANNOTATOR = annotator.MetaAnnotator(game, outpath)
    return _GLOBAL_ANNOTATOR


def stop_global_annotation_and_commit(game: pydipcc.Game) -> None:
    global _GLOBAL_ANNOTATOR
    assert _GLOBAL_ANNOTATOR is not None
    _GLOBAL_ANNOTATOR.commit(game)
    _GLOBAL_ANNOTATOR = None


@contextlib.contextmanager
def maybe_kickoff_annotations(game: pydipcc.Game, outpath: Optional[pathlib.Path]):
    global _GLOBAL_ANNOTATOR
    if outpath is None:
        yield
    else:
        start_global_annotation(game, outpath)
        yield None
        try:
            stop_global_annotation_and_commit(game)
        finally:
            _GLOBAL_ANNOTATOR = None


def after_new_phase(game: pydipcc.Game):
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.after_new_phase(game)


def after_last_message_add(game: pydipcc.Game):
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.after_message_add(list(game.messages.values())[-1])


def after_message_add(last_message: MessageDict):
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.after_message_add(last_message)


def after_message_generation_failed(*, bad_tags: Optional[List[str]] = None):
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.after_message_generation_failed(bad_tags=bad_tags)


def commit_annotations(game: pydipcc.Game) -> None:
    """Call this when the game game is checkpointed and annotations should be kept."""
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.commit(game)


############################################
##### WRITE API
############################################
def add_pseudo_orders_next_msg(data: PseudoOrders) -> None:
    return _maybe_add_at_timestamp(
        datatypes.PseudoOrdersDataType, data, timestamp=annotator.NEXT_MSG
    )


def add_filtered_msg(data: datatypes.NonsenseAnnotation, timpestamp: Optional[int]) -> None:
    return _maybe_add_at_timestamp(datatypes.NonsenseDataType, data, timpestamp)


def add_dict_next_msg(data: datatypes.JsonnableDict, *, tag: str, version: int = 2) -> None:
    assert not any(
        tag.startswith(default_tag) for default_tag in datatypes.KNOWN_TAGS
    ), f"Are you trying to serialize a known type? ({datatypes.KNOWN_TAGS})"
    if _GLOBAL_ANNOTATOR is not None:
        # Legacy: for testing
        encoded_data = json.dumps(data) if version == 1 else data
        _GLOBAL_ANNOTATOR.add_annotation(
            encoded_data=encoded_data,
            version=version,
            phase=annotator.CURRENT_PHASE,
            timestamp=annotator.NEXT_MSG,
            tag=tag,
        )


def add_dict_this_phase(data: datatypes.JsonnableDict, *, tag: str, version: int = 2) -> None:
    assert not any(
        tag.startswith(default_tag) for default_tag in datatypes.KNOWN_TAGS
    ), f"Are you trying to serialize a known type? ({datatypes.KNOWN_TAGS})"
    if _GLOBAL_ANNOTATOR is not None:
        # Legacy: for testing
        encoded_data = json.dumps(data) if version == 1 else data
        _GLOBAL_ANNOTATOR.add_annotation(
            encoded_data=encoded_data,
            version=version,
            phase=annotator.CURRENT_PHASE,
            timestamp=None,
            tag=tag,
        )


############################################
##### READ API
############################################
def read_annotations(annotation_path: pathlib.Path) -> ParsedData:
    data = ParsedData()
    for record in annotator.read_raw_records(annotation_path):
        common = ParsedData.CommonData(
            phase=record.pointer.phase, timestamp_centis=record.pointer.timestamp, tag=record.tag,
        )
        version = record.version
        payload = json.loads(record.json_data)
        if record.tag.startswith(datatypes.PseudoOrdersDataType.DEFAULT_TAG):
            data.pseudoorders.append(
                (common, datatypes.PseudoOrdersDataType.load(payload, version))
            )
        elif record.tag.startswith(datatypes.NonsenseDataType.DEFAULT_TAG):
            data.nonsenses.append((common, datatypes.NonsenseDataType.load(payload, version)))
        else:
            if record.version == 1:
                # Legacy: add_dict_* functions used to double-encode data.
                payload = json.loads(payload)
            data.misc.append((common, payload))
    return data


############################################
##### Helper functions
############################################
def _maybe_add_at_timestamp(
    datatype: Type[datatypes.MetaType[T]], data: T, timestamp: Optional[int]
) -> None:
    if _GLOBAL_ANNOTATOR is not None:
        _GLOBAL_ANNOTATOR.add_annotation(
            encoded_data=datatype.dump(data),
            version=datatype.VERSION,
            phase=annotator.CURRENT_PHASE,
            timestamp=timestamp,
            tag=datatype.DEFAULT_TAG,
        )
