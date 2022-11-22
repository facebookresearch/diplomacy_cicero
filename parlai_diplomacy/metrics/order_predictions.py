#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Metrics related to order predictions
"""
from typing import Optional, Tuple

from parlai.core.message import Message
from parlai.core.metrics import AverageMetric

from parlai_diplomacy.utils.game2seq.format_helpers.orders import OrdersUnflattener, order_is_empty
from fairdiplomacy.utils.typedefs import is_phase_name


class OrderPredMetricMixin:
    """
    Mixin to add metrics to teachers which return a single order prediction.
    """

    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        if "text" not in model_response or model_response["text"] is None:
            # model didn't speak, skip this example
            return

        orders_unflattener = OrdersUnflattener(teacher_action["task_version"])
        split_label = labels[0].split("\n")
        if is_phase_name(split_label[0]):
            # This is anorder rollout teacher; don't calculate metrics
            order_label_str = split_label[1]
            order_pred_str = model_response["text"].split("\n")[1]
            order_label = set(orders_unflattener.unflatten_action(order_label_str))
            order_pred = set(orders_unflattener.unflatten_action(order_pred_str))
            empty_order = order_is_empty(order_label_str)
        else:
            order_label = set(orders_unflattener.unflatten_action(labels[0]))
            order_pred = set(orders_unflattener.unflatten_action(model_response["text"]))
            empty_order = order_is_empty(labels[0])

        if not empty_order:
            # set intersection metric
            intersect = len(order_pred.intersection(order_label))
            denom = len(order_label)
            self.metrics.add("order_no_empty_avg", AverageMetric(intersect, denom))

        # exact match metric
        exact_match = int(order_label == order_pred)
        self.metrics.add("order_exact_avg", AverageMetric(exact_match, 1))
        if not empty_order:
            self.metrics.add("order_exact_no_empty_avg", AverageMetric(exact_match, 1))


class AllOrderPredMetricMixin:
    """
    Mixin to add metrics to teachers which return predictions for all orders
    """

    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        if model_response.get("text", None) is None:
            # model didn't speak, skip this example
            return

        orders_unflattener = OrdersUnflattener(teacher_action["task_version"])
        split_label = labels[0].split("\n")
        if is_phase_name(split_label[0]):
            # This is an all order rollout teacher; don't calculate metrics
            phase = split_label[0]
            orders_label = orders_unflattener.unflatten_rollout_joint_action(labels[0]).get(phase)
            orders_pred = orders_unflattener.unflatten_rollout_joint_action(
                model_response["text"]
            ).get(phase)
        else:
            orders_label = orders_unflattener.unflatten_joint_action(labels[0])
            orders_pred = orders_unflattener.unflatten_joint_action(model_response["text"])

        player = teacher_action["player"]

        for power, order_label in orders_label.items():
            order_label = set(order_label)
            maybe_order_pred = orders_pred.get(power)
            if maybe_order_pred is None:
                self.metrics.add("all_order_exact_avg", AverageMetric(0, 1))
                continue
            order_pred = set(maybe_order_pred)

            empty_order = not order_label

            # intersection metrics
            if not empty_order:
                intersect = len(order_pred.intersection(order_label))
                denom = len(order_label)
                self.metrics.add("all_order_no_empty_avg", AverageMetric(intersect, denom))
                if power == player:
                    self.metrics.add("order_no_empty_avg", AverageMetric(intersect, denom))

            # exact match metrics
            exact_match = int(order_label == order_pred)
            self.metrics.add("all_order_exact_avg", AverageMetric(exact_match, 1))
            if not empty_order:
                self.metrics.add("all_order_exact_no_empty_avg", AverageMetric(exact_match, 1))
            if power == player:
                # get metrics for predicting your OWN order
                self.metrics.add("order_exact_avg", AverageMetric(exact_match, 1))
                if not empty_order:
                    self.metrics.add("order_exact_no_empty_avg", AverageMetric(exact_match, 1))
