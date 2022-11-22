#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import Counter
from textwrap import indent
import time
import contextlib

from tabulate import tabulate


class TimingCtx:
    def __init__(self, init=None, init_ns=None, first_tic=None):
        self.timings = init if init is not None else Counter()
        self.ns = init_ns if init_ns is not None else Counter()
        self.arg = None
        self.last_clear = time.time()
        self.first_tic = first_tic

    def clear(self):
        self.timings.clear()
        self.last_clear = time.time()

    def start(self, arg):
        if self.arg is not None:
            self.__exit__()
        self.__call__(arg)
        self.__enter__()

    def stop(self):
        self.start(None)

    def __call__(self, arg):
        if self.arg is not None:
            self.__exit__()
        self.arg = arg
        return self

    def __enter__(self):
        self.tic = time.time()
        if self.first_tic is None:
            self.first_tic = self.tic

    def __exit__(self, *args):
        self.timings[self.arg] += time.time() - self.tic
        self.ns[self.arg] += 1
        self.arg = None

    def __repr__(self):
        return dict(
            total=time.time() - self.last_clear,
            **dict(sorted(self.timings.items(), key=lambda kv: kv[1], reverse=True)),
        ).__repr__()

    def __truediv__(self, other):
        return {k: v / other for k, v in self.timings.items()}

    def __iadd__(self, other):
        if other != 0:
            self.timings += other.timings
            self.ns += other.ns
        return self

    def items(self):
        return self.timings.items()

    @classmethod
    def pprint_multi(cls, timings, log_fn):
        data = []
        for k in {k for t in timings for k in t.timings}:
            t_mean = sum(t.timings[k] for t in timings) / len(timings)
            t_max = max(t.timings[k] for t in timings)
            n_mean = sum(t.ns[k] for t in timings) / len(timings)
            data.append((k, n_mean, t_mean, t_max))
        sum_t_mean = sum(sum(t.timings.values()) for t in timings) / len(timings)

        rows = [
            (k, f"{n_mean:.1f}", f"{t_mean * 1e3:.1f}", f"{t_max * 1e3:.1f}")
            for k, n_mean, t_mean, t_max in data
        ]
        rows = sorted(rows, key=lambda x: float(x[2]), reverse=True)
        rows.append(("", "", "", ""))
        rows.append(("Total", "", f"{sum_t_mean * 1e3:.1f}", ""))
        headers = ("Key", "N_mean", "t_mean (ms)", "t_max (ms)")

        tabulated = tabulate(rows, headers=headers, disable_numparse=True, stralign="right")

        log_fn(
            f"TimingCtx summary of {len(timings)} timings (mean/max shown over this group):\n"
            f"{indent(tabulated, '  ')}\n"
        )

    def pprint(self, log_fn):
        data = []
        for k in self.timings.keys():
            n = self.ns[k]
            t_total = self.timings[k]
            t_mean = t_total / n
            data.append((k, n, t_total, t_mean))
        elapsed = time.time() - self.first_tic
        sum_t_total = sum(self.timings.values())

        rows = [
            (k, n, f"{t_total * 1e3:.1f}", f"{t_mean * 1e3:.1f}") for k, n, t_total, t_mean in data
        ]
        rows = sorted(rows, key=lambda x: float(x[2]), reverse=True)
        rows.append(("", "", "", ""))
        rows.append(("Lost", "", f"{(elapsed - sum_t_total) * 1e3:.1f}", ""))
        rows.append(("Total", "", f"{sum_t_total * 1e3:.1f}", ""))
        headers = ("Key", "N", "t_total (ms)", "t_mean (ms)")

        tabulated = tabulate(rows, headers=headers, disable_numparse=True, stralign="right")

        log_fn(f"TimingCtx summary:\n{indent(tabulated, '  ')}\n")

    @contextlib.contextmanager
    def create_subcontext(self, prefix: str = ""):
        """Create an inner timer to pass into function calls.

        Should be used as a context.  On exit, will incorporate timings from
        the inner timings.

        Creating context managet stops current timer and resumes on context exit.
        """
        old_arg = self.arg
        self.stop()
        subtiming = TimingCtx()
        yield subtiming
        prefix = (prefix.rstrip(".") + ".") if prefix else ""
        self.timings += {f"{prefix}{k}": v for k, v in subtiming.timings.items()}
        self.ns += {f"{prefix}{k}": v for k, v in subtiming.ns.items()}
        if old_arg is not None:
            self.start(old_arg)


class DummyCtx:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def __call__(self, *args):
        return self
