#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, Optional
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, SumMetric


class ClassifierMetricMixin:
    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        if not labels:
            return
        for c in self.classes:
            x = int(c in labels)
            self.metrics.add(f"class_{c}_sum", SumMetric(x))
            self.metrics.add(f"class_{c}_avg", AverageMetric(x))
