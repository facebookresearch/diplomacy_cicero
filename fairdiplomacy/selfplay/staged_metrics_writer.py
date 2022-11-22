#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections

import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger


class StagedLogger:
    """A metric logger that aggregates metrics from many workers across many epochs.

    This is a base class and clients may want to override _update_counters or
    _reset_state.
    """

    def __init__(self, tag: str, min_samples: int):
        self._tag = tag
        self._logger = fairdiplomacy.selfplay.remote_metric_logger.get_remote_logger(tag=tag)
        self._num_aggregated_samples = min_samples
        self._reset_state()

    def close(self):
        self._logger.close()

    def _reset_state(self):
        self._counters = collections.defaultdict(fairdiplomacy.selfplay.metrics.FractionCounter)
        self._num_added = 0
        self._max_seen_step = -1

    def _update_counters(self, data):
        """Main function to update counters. Client may redefine."""
        for k, v in data.items():
            self._counters[k].update(v)

    def _finalize_metrics(self):
        data = {k: v.value() for k, v in self._counters.items()}
        data[f"{self._tag}/num_games"] = self._num_added
        return data

    def add_metrics(self, *, data, global_step):
        if self._max_seen_step < global_step:
            if self._num_added >= self._num_aggregated_samples:
                self._logger.log_metrics(self._finalize_metrics(), self._max_seen_step)
                self._reset_state()
            self._max_seen_step = global_step
        self._num_added += 1
        self._update_counters(data)
