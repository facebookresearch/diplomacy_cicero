#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib
import importlib.util

ROOT = pathlib.Path(__file__).parent.parent


def main(cfg_path, is_task_cfg):
    # Using ugly loading to avoid loading torch on CircleCI.
    spec = importlib.util.spec_from_file_location("conf", str(ROOT / "heyhi/conf.py"))
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    assert cfg_path.exists()
    if "DEPRECATED" in str(cfg_path):
        print("This CFG is deprecated. Skipping attempt to load.")
        return
    print("Trying to load", cfg_path)
    if is_task_cfg:
        conf.load_root_proto_message(cfg_path, [])
    else:
        conf.load_proto_message(cfg_path)
    print("Loaded!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=pathlib.Path)
    parser.add_argument(
        "--is_task_cfg",
        action="store_true",
        help="If set, will tree the config as a Task config. Otherwise will treat as a partial cfg.",
    )
    main(**vars(parser.parse_args()))
