#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Execute includes and print configs.

Usage:
    %(cmd)s cfg_path [overrides]

"""
import pathlib

import heyhi


def main(cfg_paths):
    for cfg_path in cfg_paths:
        assert cfg_path.exists()
        cfg = heyhi.conf.load_config(cfg_path)
        print("===", cfg_path)
        print(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_paths", nargs="+", type=pathlib.Path)
    main(**vars(parser.parse_args()))
