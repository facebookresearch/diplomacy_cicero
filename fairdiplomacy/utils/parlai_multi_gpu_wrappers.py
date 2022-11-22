#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import concurrent.futures
import functools
import logging
import os
import time
from typing import Callable, Dict, Optional
import torch.cuda

from conf import agents_cfgs, conf_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper

mp = get_multiprocessing_ctx()

_THE_MODEL = None
_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
CACHE_SIZE = 32  # we have lots of nonsense classifiers that each run in their own executor


class ParlaiExecutor(abc.ABC):
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abc.abstractmethod
    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        pass

    @abc.abstractmethod
    def get(self, attr_name: str) -> concurrent.futures.Future:
        pass

    @abc.abstractmethod
    def get_model(self) -> BaseWrapper:
        pass

    @abc.abstractmethod
    def get_model_opt(self) -> Dict:
        pass


@functools.lru_cache(maxsize=CACHE_SIZE)
def load_wrapper_executor(
    parlai_model_cfg: conf_cfgs.ParlaiModel,
    model_factory: Callable[[conf_cfgs.ParlaiModel], BaseWrapper],
    allow_multi_gpu: bool,
    load_model_on_main: bool,
    gpu_id: Optional[int] = None,
) -> ParlaiExecutor:
    # >2 so that we don't run on devfair.
    if (gpu_id is None and torch.cuda.device_count() > 2 and allow_multi_gpu) or (
        gpu_id is not None and gpu_id < torch.cuda.device_count()
    ):
        return MultiProcessParlaiExecutor(
            parlai_model_cfg, model_factory, load_model_on_main, gpu_id
        )
    else:
        return PseudoParlaiExecutor(model_factory(parlai_model_cfg))


def wrap_parlai_model_to_executor(parlai_model: BaseWrapper,) -> ParlaiExecutor:
    return PseudoParlaiExecutor(parlai_model)


class PseudoParlaiExecutor(ParlaiExecutor):
    def __init__(self, parlai_model: BaseWrapper):
        logging.info("Buillding PseudoParlaiExecutor (no parallelizm)")
        self._model = parlai_model

    def is_loaded(self) -> bool:
        return True

    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        func = getattr(self._model, func_name)

        if game is not None:
            result = func(game, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return InstantFuture(result)

    def get(self, attr_name: str) -> concurrent.futures.Future:
        return InstantFuture(getattr(self._model, attr_name))

    def get_model(self) -> BaseWrapper:
        return self._model

    def get_model_opt(self) -> Dict:
        return self._model.opt


class MultiProcessParlaiExecutor(ParlaiExecutor):
    def __init__(
        self,
        cfg: conf_cfgs.ParlaiModel,
        model_factory: Callable[[conf_cfgs.ParlaiModel], BaseWrapper],
        load_model_on_main: bool = True,
        gpu_id: Optional[int] = None,
    ):
        assert _THE_MODEL is None, "Expected the model to be non-loaded in the main process"
        self.cfg = cfg

        if gpu_id is None:
            assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
            # Will not use GPU:0
            self._num_workers = torch.cuda.device_count() - 1
            logging.info("Building MultiProcessParlaiExecutor for %d devices", self._num_workers)
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_workers, mp_context=mp,
            )
            for _ in self._executor.map(
                _load_model_to_global_var,
                [(cfg, i + 1, model_factory) for i in range(self._num_workers)],
            ):
                pass
            self._model_loading_fut = None
        else:
            assert gpu_id < torch.cuda.device_count(), (gpu_id, torch.cuda.device_count())
            self._num_workers = 1
            logging.info(
                "Building MultiProcessParlaiExecutor for model dispatched to device %d", gpu_id
            )
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_workers, mp_context=mp,
            )
            self._model_loading_fut = self._executor.submit(
                _load_model_to_global_var, (cfg, gpu_id, model_factory)
            )

        self._model = model_factory(cfg) if load_model_on_main else None

    def is_loaded(self) -> bool:
        if self._model_loading_fut is None:
            # This means that the subprocesses were waited on inside the constructor
            return True

        # If there is a future, wait for it.
        self._model_loading_fut.result()
        return True

    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        assert self.is_loaded()
        return self._executor.submit(
            _compute, func_name, game.to_json() if game is not None else None, *args, **kwargs,
        )

    def get(self, attr_name: str) -> concurrent.futures.Future:
        assert self.is_loaded()
        return self._executor.submit(_get, attr_name)

    def get_model(self) -> BaseWrapper:
        if self._model:
            return self._model
        else:
            return self._executor.submit(get_global_model).result()

    def get_model_opt(self) -> Dict:
        if self._model:
            return self._model.opt
        else:
            return self.get("opt").result()

    def __del__(self):
        print("Sunsetting the process pool for", self.cfg)
        self._executor.shutdown()


def _load_model_to_global_var(args):
    parlai_model_cfg, gpu_id, model_factory = args
    global _THE_MODEL
    assert _THE_MODEL is None, f"Double loading? ({os.getpid()})"
    # As soon as torch.cuda.XXX is called, we load ~1GB of CUDA context on GPU
    # To avoid this, we set CUDA_VISIBLE_DEVICES=gpu_id up front before torch.cuda
    # gets initialized. From the perspective of this process, there's just a single
    # GPU.
    os.environ[_CUDA_VISIBLE_DEVICES] = str(gpu_id)
    assert torch.cuda.device_count() == 1, gpu_id
    heyhi.setup_logging(label=f"proc:{gpu_id}")
    _THE_MODEL = model_factory(parlai_model_cfg)
    assert isinstance(_THE_MODEL, BaseWrapper)
    logging.info(f"Process {os.getpid()} : Done loading")


def get_global_model():
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"
    return _THE_MODEL


def _compute(func_name: str, game_json: Optional[str], *args, **kwargs):
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"

    func = getattr(_THE_MODEL, func_name)
    if game_json is not None:
        game = pydipcc.Game.from_json(game_json)
        result = func(game, *args, **kwargs)
    else:
        result = func(*args, **kwargs)

    torch.cuda.empty_cache()
    return result


def _get(attr_name: str):
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"

    return getattr(_THE_MODEL, attr_name)


class InstantFuture(concurrent.futures.Future):
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result
