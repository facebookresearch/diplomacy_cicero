#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
from os import error
import pathlib

ROOT = pathlib.Path(".").parent.resolve()
PYRIGHT_CONFIG = ROOT / "pyrightconfig.CI.json"


def main():
    print("Reading", PYRIGHT_CONFIG)
    assert PYRIGHT_CONFIG.exists(), PYRIGHT_CONFIG
    with PYRIGHT_CONFIG.open() as stream:
        cfg = json.load(stream)

    errors = []

    excludes = cfg["exclude"]
    for fpath in excludes:
        print(f"Checking that {fpath} exists")
        if not (ROOT / fpath).exists() and fpath != "local":
            errors.append(f"Can't find {fpath}")
            print("ERROR!", errors[-1])

    if errors:
        print("=====")
        print("Test failed. Here's the list of errors:")
        for err in errors:
            print(err)
        raise RuntimeError("Pyright config is not valid")
    else:
        print("ALL GOOD!")


if __name__ == "__main__":
    main()
