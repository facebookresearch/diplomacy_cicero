#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Optional, Union
import collections
import datetime
import json
import time
import wandb

import heyhi

import torch
import torch.utils.tensorboard


Numeric = Union[float, int]
NumericOrTensor = Union[Numeric, torch.Tensor]


def _sanitize(value: NumericOrTensor) -> Numeric:
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


def rec_map(callable, dict_seq_nest):
    """Recursive map that goes into dics, lists, and tuples.

    This function tries to preserve named tuples and custom dics. It won't
    work with non-materialized iterators.
    """
    if isinstance(dict_seq_nest, list):
        return type(dict_seq_nest)(rec_map(callable, x) for x in dict_seq_nest)
    if isinstance(dict_seq_nest, tuple):
        return type(dict_seq_nest)(*[rec_map(callable, x) for x in dict_seq_nest])
    if isinstance(dict_seq_nest, dict):
        return type(dict_seq_nest)((k, rec_map(callable, v)) for k, v in dict_seq_nest.items())
    return callable(dict_seq_nest)


def recursive_tensor_item(tensor_nest):
    return rec_map(_sanitize, tensor_nest)


def flatten_dict(tensor_dict):
    if not isinstance(tensor_dict, dict):
        return tensor_dict
    dd = {}
    for k, v in tensor_dict.items():
        v = flatten_dict(v)
        if isinstance(v, dict):
            for subkey, subvaule in v.items():
                dd[f"{k}/{subkey}"] = subvaule
        else:
            dd[k] = v
    dd = dict(sorted(dd.items()))
    return dd


class StopWatchTimer:
    """Time something with ability to pause."""

    def __init__(self, auto_start=True):
        self._elapsed: float = 0
        self._start: Optional[float] = None
        if auto_start:
            self.start()

    def start(self) -> None:
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + time.time() - self._start
        else:
            return self._elapsed

    def pause(self) -> None:
        self._elapsed = self.elapsed
        self._start = None


class MultiStopWatchTimer:
    """Time several stages that go one after another."""

    def __init__(self):
        self._start: Optional[float] = None
        self._name = None
        self._timings = collections.defaultdict(float)

    def start(self, name) -> None:
        now = time.time()
        if self._name is not None and self._start is not None:
            self._timings[self._name] += now - self._start
        self._start = now
        self._name = name

    @property
    def timings(self) -> Dict[str, float]:
        if self._name is not None:
            self.start(self._name)
        return self._timings

    def items(self):
        return self.timings.items()

    def __repr__(self):
        timings = self.timings
        return dict(
            total=sum(timings.values()),
            **dict(sorted(timings.items(), key=lambda kv: kv[1], reverse=True)),
        ).__repr__()


class FractionCounter:
    def __init__(self):
        self.numerator = self.denominator = 0

    def update(self, top, bottom=1.0):
        self.numerator += _sanitize(top)
        self.denominator += _sanitize(bottom)

    def value(self):
        return self.numerator / max(self.denominator, 1e-6)


class SumCounter:
    def __init__(self):
        self.accumulated = 0.0

    def update(self, value):
        self.accumulated += value

    def value(self):
        return self.accumulated


class MaxCounter:
    def __init__(self, default=0):
        self._value = default

    def update(self, value):
        self._value = max(_sanitize(value), self._value)

    def value(self):
        return self._value


class Logger:
    def __init__(self, tag=None, is_master=None, log_wandb=False):
        self.log_wandb = log_wandb
        if is_master is None:
            self.is_master = heyhi.is_master()
        else:
            self.is_master = is_master
        if self.is_master:
            self.writer = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
            if tag:
                fpath = f"metrics.{tag}.jsonl"
            else:
                fpath = "metrics.jsonl"
            self.jsonl_writer = open(fpath, "a")

    def log_config(self, cfg):
        if not self.is_master:
            return
        self.writer.add_text("cfg", str(cfg))

    def log_metrics(self, metrics, step, sanitize=False):
        if not self.is_master:
            return
        if sanitize:
            metrics = {k: _sanitize(v) for k, v in metrics.items()}
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, global_step=step)
        created_at = datetime.datetime.utcnow().isoformat()
        json_metrics = dict(global_step=step, created_at=created_at)
        json_metrics.update(metrics)
        if self.log_wandb:
            wandb_metrics = metrics.copy()
            wandb_metrics["global_step"] = step
            wandb.log(wandb_metrics)
        print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if not self.is_master:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __del__(self):
        self.close()
