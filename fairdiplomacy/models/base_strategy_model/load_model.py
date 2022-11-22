#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.cuda

import heyhi
from fairdiplomacy.models.consts import ADJACENCY_MATRIX, LOCS, MASTER_ALIGNMENTS
from fairdiplomacy.models.state_space import get_order_vocabulary
from fairdiplomacy.models.base_strategy_model.base_strategy_model import (
    BaseStrategyModel,
    BaseStrategyModelV2,
)
from fairdiplomacy.models.base_strategy_model.mock_base_strategy_model import MockBaseStrategyModel
from fairdiplomacy.utils.thread_pool_encoding import get_board_state_size
import conf.conf_cfgs

CACHE_SIZE = 5

SEASON_EMB_SIZE = 20
ORDER_EMB_SIZE = 80
PREV_ORDER_EMB_SIZE = 20

SomeBaseStrategyModel = Union[BaseStrategyModel, BaseStrategyModelV2]


def new_model(args: conf.conf_cfgs.TrainTask) -> SomeBaseStrategyModel:
    assert not args.avg_embedding
    assert not args.learnable_A
    assert not args.learnable_alignments
    assert not args.use_global_pooling
    assert not args.separate_value_encoder
    if args.use_v2_base_strategy_model:
        base_strategy_model = BaseStrategyModelV2(
            inter_emb_size=args.inter_emb_size,
            board_map_size=len(LOCS),
            order_emb_size=ORDER_EMB_SIZE,
            prev_order_emb_size=PREV_ORDER_EMB_SIZE,
            orders_vocab_size=len(get_order_vocabulary()),
            lstm_size=args.lstm_size,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            value_dropout=args.value_dropout,
            value_decoder_init_scale=args.value_decoder_init_scale,
            value_decoder_activation=args.value_decoder_activation,
            value_decoder_use_weighted_pool=args.value_decoder_use_weighted_pool,
            value_decoder_extract_from_encoder=args.value_decoder_extract_from_encoder,
            featurize_output=args.featurize_output,
            relfeat_output=args.relfeat_output,
            featurize_prev_orders=args.featurize_prev_orders,
            value_softmax=args.value_softmax,
            encoder_cfg=args.encoder,
            pad_spatial_size_to_multiple=args.pad_spatial_size_to_multiple,
            all_powers=args.all_powers,
            has_single_chances=(
                args.all_powers_add_single_chances is not None
                and args.all_powers_add_single_chances > 0
            ),
            has_double_chances=(
                args.all_powers_add_double_chances is not None
                and args.all_powers_add_double_chances > 0
            ),
            has_policy=args.has_policy,
            has_value=args.has_value,
            use_player_ratings=args.use_player_ratings,
            use_year=args.use_year,
            use_agent_power=args.use_agent_power,
            num_scoring_systems=args.num_scoring_systems,
            input_version=args.input_version,
            training_permute_powers=args.training_permute_powers,
            transformer_decoder=args.transformer_decoder,
            with_order_conditioning=args.with_order_conditioning,
        )
        return base_strategy_model  # type: ignore
    else:
        assert args.value_decoder_activation is None or args.value_decoder_activation == "relu"
        assert not args.value_decoder_use_weighted_pool
        assert not args.value_decoder_extract_from_encoder
        assert not args.use_year
        assert not args.use_agent_power
        return BaseStrategyModel(
            board_state_size=get_board_state_size(args.input_version),
            prev_order_emb_size=PREV_ORDER_EMB_SIZE,
            inter_emb_size=args.inter_emb_size,
            power_emb_size=args.power_emb_size,
            season_emb_size=SEASON_EMB_SIZE,
            num_blocks=args.num_encoder_blocks,
            A=torch.from_numpy(ADJACENCY_MATRIX).float(),
            master_alignments=torch.from_numpy(MASTER_ALIGNMENTS).float(),
            orders_vocab_size=len(get_order_vocabulary()),
            lstm_size=args.lstm_size,
            lstm_layers=args.lstm_layers,
            order_emb_size=ORDER_EMB_SIZE,
            lstm_dropout=args.lstm_dropout,
            encoder_dropout=args.encoder_dropout,
            use_simple_alignments=args.use_simple_alignments,
            value_decoder_init_scale=args.value_decoder_init_scale,
            value_dropout=args.value_dropout,
            featurize_output=args.featurize_output,
            relfeat_output=args.relfeat_output,
            featurize_prev_orders=args.featurize_prev_orders,
            residual_linear=args.residual_linear,
            merged_gnn=args.merged_gnn,
            encoder_layerdrop=args.encoder_layerdrop,
            value_softmax=args.value_softmax,
            encoder_cfg=args.encoder,
            pad_spatial_size_to_multiple=args.pad_spatial_size_to_multiple,
            all_powers=args.all_powers,
            has_policy=args.has_policy,
            has_value=args.has_value,
            use_player_ratings=args.use_player_ratings,
            input_version=args.input_version,
            training_permute_powers=args.training_permute_powers,
        )


def load_base_strategy_model_model_and_args(
    checkpoint_path,
    map_location: str = "cpu",
    eval: bool = False,
    skip_weight_loading: bool = False,
    override_has_policy: Optional[bool] = None,
    override_has_value: Optional[bool] = None,
) -> Tuple[SomeBaseStrategyModel, Dict[str, Any]]:
    """Load a base_strategy_model model and its args, which should be a dict of the TrainTask prototxt.

    If override_has_policy or override_has_value are left as None, then the default from the model checkpoint will be used.
    If explicitly given as True, then we will fail immediately if the model does not support that output instead of at inference.
    If explicitly given as False, then we will omit loading any unnecessary model parameters, and the model will

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.
        eval: If true, set the model into inference mode.
        skip_weight_loading: If True, will return a freshly initialized model,
            i.e., only args from the checkpoint to be used.
        override_has_policy: Override the "has_policy" field in the TrainTask args.
        override_has_value: Override the "has_value" field in the TrainTask args.

    Returns:
        (base_strategy_model_model, dictified version of conf.conf_cfgs.TrainTask)
    """
    if map_location != "cpu" and not torch.cuda.is_available():
        logging.warning("No CUDA so will load model to CPU instead of %s", map_location)
        map_location = "cpu"

    # Loading model to gpu right away will load optimizer state we don't care about.
    logging.info(f"Loading base_strategy_model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    args = checkpoint["args"]
    if not isinstance(args, dict):
        args = heyhi.conf_to_dict(args)

    load_state_dict_strict = True
    if override_has_policy is not None:
        args_has_policy = args.get("has_policy", True)
        # If we are changing has_policy from True to False, avoid errors about
        # unused model parameters.
        if args_has_policy and not override_has_policy:
            load_state_dict_strict = False
        args["has_policy"] = override_has_policy
    if override_has_value is not None:
        args_has_value = args.get("has_value", True)
        # If we are changing has_value from True to False, avoid errors about
        # unused model parameters.
        if args_has_value and not override_has_value:
            load_state_dict_strict = False
        args["has_value"] = override_has_value

    if "single_power_conditioning_prob" in args:
        assert "min_num_conditioning_power" not in args
        args["power_conditioning"] = {
            "prob": args["single_power_conditioning_prob"],
            "min_num_power": 1,
            "max_num_power": 1,
        }
        args.pop("single_power_conditioning_prob")

    cfg = conf.conf_cfgs.TrainTask(**args)
    model = new_model(cfg)

    if not skip_weight_loading:
        # strip "module." prefix if model was saved with DistributedDataParallel wrapper
        state_dict = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in checkpoint["model"].items()
        }

        results = model.load_state_dict(state_dict, strict=load_state_dict_strict)
        if not load_state_dict_strict:
            # Even when not strict, still fail on keys that needed to be there but weren't.
            if len(results.missing_keys) > 0:
                raise RuntimeError(
                    f"Missing keys in state dict when loading base_strategy_model: {results.missing_keys}"
                )
            if len(results.unexpected_keys) > 0:
                logging.info(
                    f"This base_strategy_model supports outputs we aren't using, pruning extra keys: {results.unexpected_keys}"
                )

    model = model.to(map_location)

    if eval:
        model.eval()

    return model, args


def load_base_strategy_model_model(
    checkpoint_path, map_location="cpu", eval=False,
) -> SomeBaseStrategyModel:
    """Load a base_strategy_model model.

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.
        eval: If true, set the model into inference mode.

    Returns:
        base_strategy_model_model
    """
    if checkpoint_path == "MOCK" or checkpoint_path.startswith("MOCKV"):
        logging.warning("Loading a mock base_strategy_model")
        assert eval
        if checkpoint_path.startswith("MOCKV"):
            return MockBaseStrategyModel(input_version=int(checkpoint_path[len("MOCKV") :]))  # type: ignore
        else:
            return MockBaseStrategyModel(input_version=1)  # type: ignore

    model, _args = load_base_strategy_model_model_and_args(
        checkpoint_path, map_location=map_location, eval=eval,
    )
    return model


@functools.lru_cache(maxsize=CACHE_SIZE)
def load_base_strategy_model_model_cached(
    *, checkpoint_path: str, map_location: str,
) -> SomeBaseStrategyModel:
    """Load a base_strategy_model model in inference mode, with caching.

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.

    Returns:
        base_strategy_model_model
    """
    return load_base_strategy_model_model(checkpoint_path, map_location=map_location, eval=True,)
