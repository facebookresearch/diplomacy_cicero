#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from parlai_diplomacy.utils.game2seq.order_prediction import (
    AllOrderIndependentPredictionFormatter,
    AllOrderIndependentRolloutPredictionFormatter,
    AllOrderPredictionFormatter,
    AllOrderRolloutPredictionFormatter,
    BaseDiplomacyPredictionFormatter,
    OrderPredictionFormatter,
    OrderRolloutPredictionFormatter,
    PlausiblePseudoOrderPredictionFormatter,
    TrainingPlausiblePseudoOrderPredictionFormatter,
)
from parlai_diplomacy.utils.game2seq.dialogue_prediction import (
    DialoguePredictionFormatter,
    TrainingDialoguePredictionFormatter,
    HumanVsModelDiscriminatorFormatter,
    TrainingHumanVsModelDiscriminatorFormatter,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    get_output_type,
    get_input_format,
)  # noqa: F401


def sequence_formatter_factory(
    fmt: str, version: int, training=True
) -> BaseDiplomacyPredictionFormatter:
    output_type = get_output_type(fmt)

    if output_type == "order":
        return OrderPredictionFormatter(version)
    elif output_type == "orderrollout":
        return OrderRolloutPredictionFormatter(version)
    elif output_type == "allorder":
        return AllOrderPredictionFormatter(version)
    elif output_type == "allorderindependent":
        return AllOrderIndependentPredictionFormatter(version)
    elif output_type == "allorderrollout":
        return AllOrderRolloutPredictionFormatter(version)
    elif output_type == "allorderindependentrollout":
        return AllOrderIndependentRolloutPredictionFormatter(version)
    elif output_type == "plausiblepseudoorder":
        if training:
            return TrainingPlausiblePseudoOrderPredictionFormatter(version)
        else:
            return PlausiblePseudoOrderPredictionFormatter(version)
    elif (
        output_type == "dialogue"
        or output_type == "dialoguediscriminator"
        or output_type == "sleepclassifier"
        or output_type == "sleepsix"
        or output_type == "recipientclassifier"
        or output_type == "drawclassifier"
        or output_type == "liedetector"
    ):
        if training:
            return TrainingDialoguePredictionFormatter(version)
        else:
            return DialoguePredictionFormatter(version)
    elif output_type == "humanvsmodeldiscriminator":
        if training:
            return TrainingHumanVsModelDiscriminatorFormatter(version)
        else:
            return HumanVsModelDiscriminatorFormatter(version)
    else:
        raise RuntimeError(f"Model type {output_type} not supported")
