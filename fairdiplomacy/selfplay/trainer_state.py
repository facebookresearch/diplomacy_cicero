#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Optional
import argparse

import logging
import pathlib

import attr
import torch
import torch.nn
import torch.optim

TRAINER_STATE_VERSION = 2


@attr.s(auto_attribs=True)
class NetTrainingState:
    """
    Contains the state of the Trainer for a single model.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    # To be able to load the model in eval.
    args: argparse.Namespace

    # These are propagated from the TrainerState.
    epoch_id: int = 0
    global_step: int = 0

    def state_dict(self) -> Dict:
        data = attr.asdict(self)
        data["model"] = getattr(self.model, "module", self.model).state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["scheduler"] = self.scheduler.state_dict() if self.scheduler is not None else None
        return data

    @classmethod
    def from_dict(
        cls, state_dict: Dict, default: "NetTrainingState", device: str = "cpu"
    ) -> "NetTrainingState":
        state_dict = dict(state_dict)
        if frozenset(attr.asdict(default)) != frozenset(state_dict):
            logging.error(
                "Loading state from that has different set of keys.\n\tState keys: %s\n\tckpt keys:%s",
                sorted(attr.asdict(default)),
                sorted(state_dict),
            )
            raise ValueError("Bad checkpoint")
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(state_dict["model"])
        state_dict["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(state_dict["optimizer"])
        state_dict["optimizer"] = optimizer

        if state_dict["scheduler"] is not None:
            scheduler = default.scheduler
            assert scheduler is not None, "Got state for scheduler, but no scheduler exists"
            scheduler.load_state_dict(state_dict["scheduler"])
            state_dict["scheduler"] = scheduler

        return cls(**state_dict)  # type: ignore

    def save(self, filename: pathlib.Path) -> None:
        tmp_fpath = pathlib.Path(str(filename) + ".tmp")
        torch.save(self.state_dict(), tmp_fpath)
        tmp_fpath.rename(filename)


@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    def _set_in_nets(self, attrib, value):
        """Sets epoch_id and global_step in submodules if set in the trainer state."""
        name = attrib.name
        setattr(self.net_state, name, value)
        if self.value_net_state is not None:
            setattr(self.value_net_state, name, value)
        return value

    net_state: NetTrainingState
    value_net_state: Optional[NetTrainingState]
    version: int = TRAINER_STATE_VERSION

    # Whenever these methods are set, their values are updated in net_state and
    # value_net_state.
    epoch_id: int = attr.ib(default=0, on_setattr=_set_in_nets)
    global_step: int = attr.ib(default=0, on_setattr=_set_in_nets)

    # Accessors to make PG work without changes.
    @property
    def model(self):
        return self.net_state.model

    @property
    def optimizer(self):
        return self.net_state.optimizer

    @property
    def scheduler(self):
        return self.net_state.scheduler

    @property
    def value_model(self):
        return self.value_net_state.model if self.value_net_state is not None else self.model

    def state_dict(self) -> Dict:
        data = attr.asdict(self)
        data["net_state"] = self.net_state.state_dict()
        if self.value_net_state is not None:
            data["value_net_state"] = self.value_net_state.state_dict()
        return data

    def save(self, filename: pathlib.Path) -> None:
        tmp_fpath = pathlib.Path(str(filename) + ".tmp")
        torch.save(self.state_dict(), tmp_fpath)
        tmp_fpath.rename(filename)

    @classmethod
    def from_dict(
        cls, state_dict: Dict, default: "TrainerState", device: str = "cpu"
    ) -> "TrainerState":
        data = dict(state_dict)
        # Migrate old state.
        version = data.get("version", 1)
        assert version <= TRAINER_STATE_VERSION, version
        if version == 1:
            data["net_state"] = {k: data.pop(k) for k in ("model", "optimizer", "scheduler")}
        # The new state is always on the last version.
        data["version"] = TRAINER_STATE_VERSION

        # Check keys.
        if frozenset(attr.asdict(default)) != frozenset(data):
            logging.error(
                "Loading state that has different set of keys.\n\tState keys: %s\n\tckpt keys:%s",
                sorted(attr.asdict(default)),
                sorted(data),
            )
            raise ValueError("Bad checkpoint")

        # Recursive init for submodels.
        data["net_state"] = default.net_state.from_dict(
            data["net_state"], default.net_state, device=device
        )

        if data["value_net_state"] is not None:
            assert (
                default.value_net_state is not None
            ), "Got state for value net, but no value net exists"
            data["value_net_state"] = default.value_net_state.from_dict(
                data["value_net_state"], default.value_net_state, device=device
            )
        return cls(**data)  # type: ignore

    @classmethod
    def load(
        cls, filename: pathlib.Path, default: "TrainerState", device: str = "cpu"
    ) -> "TrainerState":
        logging.info("Loading TrainerState from %s", filename)
        data = torch.load(filename, map_location=device)
        self = cls.from_dict(data, default, device)
        logging.info("Loaded state from %s", filename)
        logging.info(
            "Loaded scalars: %s",
            {k: v for k, v in data.items() if isinstance(v, (int, float, str))},
        )

        return self
