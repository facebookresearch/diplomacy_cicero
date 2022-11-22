#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import concurrent.futures
import logging
import os
from typing import List
import torch.cuda

from fairdiplomacy import pydipcc
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.base_strategy_model_rollouts import BaseStrategyModelRollouts
from fairdiplomacy.utils.parlai_multi_gpu_wrappers import InstantFuture

mp = get_multiprocessing_ctx()

_THE_MODEL = None
_THE_ROLLOUT = None
_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


class MultiProcessBaseStrategyModelExecutor:
    def __init__(
        self,
        allow_multi_gpu,
        base_strategy_model_wrapper_kwargs,
        base_strategy_model_rollouts_kwargs,
    ):
        assert _THE_MODEL is None, "Expected the model to be non-loaded in the main process"
        self.base_strategy_model_wrapper_kwargs = base_strategy_model_wrapper_kwargs
        if allow_multi_gpu and torch.cuda.device_count() > 2:
            # Will not use GPU:0, >2 so that we don't run on devfair.
            self._num_workers = torch.cuda.device_count() - 1
            logging.info("Buillding MultiProcessParlaiExecutor for %d devices", self._num_workers)
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_workers, mp_context=mp,
            )
            for _ in self._executor.map(
                _load_base_strategy_model_model_to_global_var,
                [
                    (
                        base_strategy_model_wrapper_kwargs,
                        base_strategy_model_rollouts_kwargs,
                        i + 1,
                    )
                    for i in range(self._num_workers)
                ],
            ):
                pass
        else:
            self._num_workers = 1
            self._executor = None

        self._model = BaseStrategyModelWrapper(**base_strategy_model_wrapper_kwargs)  # type: ignore
        self._rollouts = BaseStrategyModelRollouts(self._model, **base_strategy_model_rollouts_kwargs)  # type: ignore

    def compute(
        self, func_name: str, games: List[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        """Call BaseStrategyModelWrapper.func_name on multiple GPUs"""
        if self._executor is None:
            return InstantFuture(getattr(self._model, func_name)(games, *args, **kwargs))
        else:
            return self._executor.submit(
                _compute, func_name, [game.to_json() for game in games], *args, **kwargs
            )

    def rollout(self, game: pydipcc.Game, agent_power, set_orders_dicts, player_rating):
        if player_rating is not None:
            player_ratings = [player_rating] * len(set_orders_dicts)
        else:
            player_ratings = None

        # despite not having multiple base_strategy_models,
        # we use do_rollouts_multi here just becasue of its Tensor return type
        if self._executor is None:
            return InstantFuture(
                self._rollouts.do_rollouts_multi(
                    game,
                    agent_power=agent_power,
                    set_orders_dicts=set_orders_dicts,
                    player_ratings=player_ratings,
                )
            )
        else:
            return self._executor.submit(
                _rollout,
                game.to_json(),
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                player_ratings=player_ratings,
            )

    def get_model(self) -> BaseStrategyModelWrapper:
        return self._model

    def num_workers(self) -> int:
        return self._num_workers

    def __del__(self):
        if self._executor is not None:
            logging.info(f"Sunsetting the process pool for {self._model.model_path}")
            self._executor.shutdown()


def _load_base_strategy_model_model_to_global_var(args):
    base_strategy_model_wrapper_kwargs, base_strategy_model_rollout_kwargs, gpu_id = args
    global _THE_MODEL
    global _THE_ROLLOUT
    assert _THE_MODEL is None, f"Double loading base_strategy_model? ({os.getpid()})"
    os.environ[_CUDA_VISIBLE_DEVICES] = str(gpu_id)
    assert torch.cuda.device_count() == 1, gpu_id
    heyhi.setup_logging(label=f"proc:{gpu_id}")
    _THE_MODEL = BaseStrategyModelWrapper(**base_strategy_model_wrapper_kwargs)
    _THE_ROLLOUT = BaseStrategyModelRollouts(_THE_MODEL, **base_strategy_model_rollout_kwargs)
    logging.info(f"Process {os.getpid()}: Done loading")


def _compute(func_name: str, game_jsons: List[str], *args, **kwargs):
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in this process ({os.getpid()})"
    games = [pydipcc.Game.from_json(game_json) for game_json in game_jsons]
    result = getattr(_THE_MODEL, func_name)(games, *args, **kwargs)
    torch.cuda.empty_cache()
    return result


def _rollout(game_json: str, *args, **kwargs):
    global _THE_ROLLOUT
    assert _THE_ROLLOUT is not None, "Model is not loaded in this process"
    game = pydipcc.Game.from_json(game_json)
    result = _THE_ROLLOUT.do_rollouts_multi(game, *args, **kwargs)
    torch.cuda.empty_cache()
    return result
