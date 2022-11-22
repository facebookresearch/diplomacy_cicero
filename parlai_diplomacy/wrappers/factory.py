#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import torch
import torch.cuda as cuda
import heyhi
from typing import Dict, Any, Optional

from parlai.core.opt import Opt

from conf import agents_cfgs
from fairdiplomacy.utils.parlai_multi_gpu_wrappers import (
    ParlaiExecutor,
    load_wrapper_executor,
)

from parlai_diplomacy.utils.game2seq.factory import get_output_type
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper, load_opt
from parlai_diplomacy.wrappers.orders import (
    BaseOrderWrapper,
    ParlAIPlausiblePseudoOrdersWrapper,
    ParlAISingleOrderWrapper,
    ParlAIAllOrderWrapper,
    ParlAIAllOrderIndependentWrapper,
    ParlAIAllOrderIndependentRolloutWrapper,
)
from parlai_diplomacy.wrappers.annotated_pseudo_orders import (
    DevOnlyBaseAnnotatedPseudoOrdersWrapper,
    DevOnlyAnnotatedPseudoAllOrdersWrapper,
    DevOnlyAnnotatedPseudoSingleOrdersWrapper,
)
from parlai_diplomacy.wrappers.dialogue import (
    BaseDialogueWrapper,
    ParlAIDialogueWrapper,
)
from parlai_diplomacy.wrappers.classifiers import (
    BaseNonsenseClassifierWrapper,
    ParlAIDrawClassifierWrapper,
    ParlAINonsenseClassifierWrapper,
    ParlAIHumanVsModelClassifierWrapper,
    ParlAIRecipientClassifierWrapper,
    ParlAISleepClassifierWrapper,
    EnsembleNonsenseClassifierWrapper,
    SludgeDialogueAsNonsenseClassifierWrapper,
)


def get_cuda_device():
    if cuda.is_available():
        return torch.zeros(1).to("cuda").device
    else:
        return None


def set_cuda_device(device):
    if device is not None:
        cur_device = torch.zeros(1).to("cuda").device
        if device != cur_device:
            logging.warn(f"Changing device from {cur_device} to {device}.")
            cuda.set_device(device)


def diplomacy_specific_overrides(override_dct: Dict[Any, Any], model_opt: Opt) -> Dict[Any, Any]:
    """
    General overrides for all Diplomacy wrappers of ParlAI agents

    When you add an override here, please date and annotate your change.
    """
    # 2021-04-08: override agent to get custom inference options
    if model_opt["model"] == "bart":
        override_dct["model"] = "bart_custom_inference"

    return override_dct


def parlai_wrapper_factory(cfg: agents_cfgs.ParlaiModel) -> BaseWrapper:
    model_opt = load_opt(cfg.model_path)
    overrides = diplomacy_specific_overrides(heyhi.conf_to_dict(cfg.overrides), model_opt)
    wrapper_args = [
        cfg.model_path,
        {"overrides": overrides},
    ]
    task = model_opt["task"].split(":")[0]
    output_type = get_output_type(task)
    old_device = get_cuda_device()

    logging.info(
        f"Loading {output_type} wrapper for model trained on task: {task}"
        + (f", remote={cfg.remote_addr}" if cfg.remote_addr else "")
    )

    if output_type == "order":
        ret = ParlAISingleOrderWrapper(*wrapper_args)
    elif output_type == "allorder":
        ret = ParlAIAllOrderWrapper(*wrapper_args)
    elif output_type == "allorderindependent":
        ret = ParlAIAllOrderIndependentWrapper(*wrapper_args)
    elif output_type == "allorderindependentrollout":
        ret = ParlAIAllOrderIndependentRolloutWrapper(*wrapper_args)
    elif output_type == "plausiblepseudoorder":
        ret = ParlAIPlausiblePseudoOrdersWrapper(*wrapper_args)
    elif (
        output_type == "dialogue"
        and (not model_opt.get("response_view_dialogue_model", False))
        and cfg.overrides.threshold is None
    ):
        ret = ParlAIDialogueWrapper(*wrapper_args)
    elif (
        output_type == "dialogue"
        and (not model_opt.get("response_view_dialogue_model", False))
        and cfg.overrides.threshold is not None
    ):
        ret = SludgeDialogueAsNonsenseClassifierWrapper(*wrapper_args)
    elif output_type in ("sleepclassifier", "sleepsix"):
        ret = ParlAISleepClassifierWrapper(*wrapper_args)
    elif output_type == "recipientclassifier":
        ret = ParlAIRecipientClassifierWrapper(*wrapper_args)
    elif output_type == "drawclassifier":
        ret = ParlAIDrawClassifierWrapper(*wrapper_args)
    elif output_type == "dialoguediscriminator":
        return ParlAINonsenseClassifierWrapper(*wrapper_args)
    elif output_type == "humanvsmodeldiscriminator":
        return ParlAIHumanVsModelClassifierWrapper(*wrapper_args)
    else:
        raise RuntimeError(f"Task {output_type} does not have a corresponding wrapper!")

    set_cuda_device(old_device)

    return ret


def load_order_wrapper(cfg: agents_cfgs.ParlaiModel) -> BaseOrderWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, BaseOrderWrapper), type(model)
    return model


def load_dialogue_wrapper(cfg: agents_cfgs.ParlaiModel,) -> BaseDialogueWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, BaseDialogueWrapper), type(model)
    return model


def load_sleep_classifier_wrapper(cfg: agents_cfgs.ParlaiModel) -> ParlAISleepClassifierWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, ParlAISleepClassifierWrapper), type(model)
    return model


def load_recipient_classifier_wrapper(
    cfg: agents_cfgs.ParlaiModel,
) -> ParlAIRecipientClassifierWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, ParlAIRecipientClassifierWrapper)
    return model


def load_draw_classifier_wrapper(cfg: agents_cfgs.ParlaiModel,) -> ParlAIDrawClassifierWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, ParlAIDrawClassifierWrapper)
    return model


def load_nonsense_classifier_wrapper(
    cfg: agents_cfgs.ParlaiModel,
) -> BaseNonsenseClassifierWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, BaseNonsenseClassifierWrapper)
    return model


def load_humanvsmodel_classifier_wrapper(
    cfg: agents_cfgs.ParlaiModel,
) -> ParlAIHumanVsModelClassifierWrapper:
    model = parlai_wrapper_factory(cfg)
    assert isinstance(model, ParlAIHumanVsModelClassifierWrapper)
    return model


def load_ensemble_nonsense_classifier_wrapper(
    cfg: agents_cfgs.ParlaiNonsenseDetectionEnsemble,
) -> Optional[EnsembleNonsenseClassifierWrapper]:
    models: Dict[str, ParlaiExecutor] = {}

    logging.info(
        f"Loading nonsense ensemble. Attempting to parallelize. Found {cuda.device_count()} gpus."
        f"Ensemble has {len(cfg.nonsense_classifiers)} classifiers, so {len(cfg.nonsense_classifiers) // cuda.device_count() if cuda.device_count() > 0 else '??'}"
        f" will be loaded per gpu."
    )
    for i, nonsense_classifer_data in enumerate(cfg.nonsense_classifiers):
        name = nonsense_classifer_data.name
        assert name is not None

        if cuda.device_count() >= 4:
            model = load_wrapper_executor(
                nonsense_classifer_data.nonsense_classifier,
                load_nonsense_classifier_wrapper,
                allow_multi_gpu=True,
                load_model_on_main=False,
                gpu_id=((i % (cuda.device_count() - 1)) + 1),
            )
        else:
            model = load_wrapper_executor(
                nonsense_classifer_data.nonsense_classifier,
                load_nonsense_classifier_wrapper,
                allow_multi_gpu=False,
                load_model_on_main=True,
            )

        assert name not in models, (name, models)
        models[name] = model

    while not all([model.is_loaded() for model in models.values()]):
        pass

    if len(models) > 0:
        return EnsembleNonsenseClassifierWrapper(models)
    else:
        return None


def load_pseudo_orders_wrapper(
    cfg: agents_cfgs.ParlaiModel,
) -> DevOnlyBaseAnnotatedPseudoOrdersWrapper:
    """
    Pseudo orders wrappers are loaded separately from the factory.
    """
    model_opt = load_opt(cfg.model_path)
    overrides = diplomacy_specific_overrides(heyhi.conf_to_dict(cfg.overrides), model_opt)
    wrapper_args = [
        cfg.model_path,
        {"overrides": overrides},
    ]
    task = model_opt["task"].split(":")[0]
    output_type = get_output_type(task)
    old_device = get_cuda_device()

    logging.info(f"Loading pseudo orders {output_type} wrapper for model trained on task: {task}")

    if output_type == "order":
        ret = DevOnlyAnnotatedPseudoSingleOrdersWrapper(*wrapper_args)
    elif output_type == "allorder":
        ret = DevOnlyAnnotatedPseudoAllOrdersWrapper(*wrapper_args)
    else:
        raise RuntimeError(
            f"Task {output_type} does not have a corresponding pseud orderswrapper!"
        )

    set_cuda_device(old_device)
    return ret
