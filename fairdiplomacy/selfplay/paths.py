#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pathlib


CKPT_SYNC_DIR = "ckpt_syncer/ckpt"


def get_rendezvous_path() -> pathlib.Path:
    return pathlib.Path("rendezvous").resolve()


def get_torch_ddp_init_fname() -> pathlib.Path:
    return get_rendezvous_path() / "torch_ddp_init"


def get_trainer_server_fname(training_ddp_rank: int) -> pathlib.Path:
    return get_rendezvous_path() / f"buffer_postman{training_ddp_rank}.txt"


def get_remote_logger_port_file() -> pathlib.Path:
    return get_rendezvous_path() / "remote_logger_addr.txt"


def get_tensorboard_folder() -> pathlib.Path:
    return pathlib.Path("tb")
