#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from contextlib import contextmanager
import os
import pathlib
from typing import BinaryIO, ContextManager, TextIO, Union
import uuid


def atomicish_open_for_writing_binary(path: Union[str, pathlib.Path]) -> ContextManager[BinaryIO]:
    # Ignore type since apparently pylance can't prove that it's binary, even though it is
    return atomicish_open_for_writing(path, binary=True)  # type: ignore


def atomicish_open_for_writing_text(path: Union[str, pathlib.Path]) -> ContextManager[TextIO]:
    # Ignore type since apparently pylance can't prove that it's text, even though it is
    return atomicish_open_for_writing(path, binary=False)  # type: ignore


@contextmanager
def atomicish_open_for_writing(path: Union[str, pathlib.Path], binary: bool):
    """Pseudo-atomic writing of a file to a given path.

    Opens and yields a file handle for writing that points to a temporary place.
    Upon exiting the 'with' block, renames it to the correct path.
    This ensures that consumers never see the file in a partially-written state,
    and that the final file exists only if writing it was entirely successful.

    Depending on the filesystem, this does not actually guarantee there are no races,
    but even so, it should greatly reduce the chance of issues when these properties
    are needed.

    Example usage:

    with atomicish_open_for_writing(path,binary=False) as f:
        f.write("foo\n")
    """

    # Write to tmp file and rename, so that on filesystem, likely an expected file
    # will never appear to the reader as partially written, it will be complete or not there at all.
    if isinstance(path, str):
        tmppath = pathlib.Path(path + "." + str(uuid.uuid4())[:8] + ".TMP")
    else:
        tmppath: pathlib.Path = path.parent / (path.name + "." + str(uuid.uuid4())[:8] + ".TMP")
    try:
        mode = "wb" if binary else "w"
        with tmppath.open(mode) as out:
            yield out
        tmppath.rename(path)
    except BaseException as e:
        try:
            tmppath.unlink()
        except FileNotFoundError:
            pass
        raise e
