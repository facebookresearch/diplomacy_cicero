#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""DataLoader supervises all non-train jobs."""
from typing import Dict, Optional, Tuple, List
import itertools
import logging
import pathlib
import socket
import time

import nest
import postman
import torch
import torch.cuda


import conf.conf_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.selfplay import rela
from fairdiplomacy.selfplay.ckpt_syncer import ValuePolicyCkptSyncer
from fairdiplomacy.selfplay.execution_context import ExecutionContext
from fairdiplomacy.selfplay.search.jobs.h2h_evaler import H2HEvaler
from fairdiplomacy.selfplay.search.jobs.game_situations_evaler import TestSituationEvaller
from fairdiplomacy.selfplay.search.jobs.rollouter import Rollouter
from fairdiplomacy.selfplay.search.jobs.selfplay_evaler import EvalPlayer
from fairdiplomacy.selfplay.search.rollout import ReSearchRolloutBatch
from fairdiplomacy.selfplay.paths import get_trainer_server_fname, CKPT_SYNC_DIR
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()

ScoreDict = Dict[str, float]
ScoreDictPerPower = Dict[Optional[str], ScoreDict]

X_POSSIBLE_ACTIONS = "observations/x_possible_actions"


def flatten_dict(nested_tensor_dict):
    def _recursive(nested_tensor_dict, prefix):
        for k, v in nested_tensor_dict.items():
            if isinstance(v, dict):
                yield from _recursive(v, prefix=f"{prefix}{k}/")
            else:
                yield f"{prefix}{k}", v

    return dict(_recursive(nested_tensor_dict, prefix=""))


def compress_and_flatten(nested_tensor_dict):
    d = flatten_dict(nested_tensor_dict)
    d[X_POSSIBLE_ACTIONS] = d[X_POSSIBLE_ACTIONS].to(torch.short)
    offsets = {}
    last_offset = 0
    for k, v in d.items():
        if v.dtype == torch.float32:
            offsets[k] = (last_offset, last_offset + v.numel())
            last_offset += v.numel()
    storage = torch.empty(last_offset)
    for key, (start, end) in offsets.items():
        storage[start:end] = d[key].view(-1)
        d[key] = storage[start:end].view(d[key].size())

    return d


def unflatten_dict(flat_tensor_dict):
    d = {}
    for k, v in flat_tensor_dict.items():
        parts = k.split("/")
        subd = d
        for p in parts[:-1]:
            if p not in subd:
                subd[p] = {}
            subd = subd[p]
        subd[parts[-1]] = v
    return d


def decompress_and_unflatten(flat_tensor_dict):
    d = flat_tensor_dict.copy()
    d[X_POSSIBLE_ACTIONS] = d[X_POSSIBLE_ACTIONS].to(torch.int32)
    d = unflatten_dict(d)
    return d


def _read_game_json_paths(initial_games_index_file: str) -> List[str]:
    """Return a list of games with optional phases."""

    if ":" in initial_games_index_file or initial_games_index_file.endswith(".json"):
        # Maybe we got a single game instead of a list of games?
        if ":" in initial_games_index_file:
            maybe_game_path, _ = initial_games_index_file.rsplit(":", 1)
        else:
            maybe_game_path = initial_games_index_file
        if maybe_game_path.endswith(".json"):
            try:
                with open(maybe_game_path) as stream:
                    pydipcc.Game.from_json(stream.read())
            except Exception:
                pass
            else:
                return [initial_games_index_file]

    game_json_paths = []
    with open(initial_games_index_file) as stream:
        for line in stream:
            line = line.split("#")[0].strip()
            if line:
                game_json_paths.append(line)
    assert game_json_paths, initial_games_index_file
    return game_json_paths


class DataLoader:
    def __init__(
        self,
        model_path,
        rollout_cfg: conf.conf_cfgs.ExploitTask.SearchRollout,
        *,
        num_train_gpus: int,
        ectx: ExecutionContext,
    ):
        del model_path  # Not used.
        self.rollout_cfg = rollout_cfg

        self._num_train_gpus = num_train_gpus
        self._ectx = ectx

        self._game_json_paths: Optional[List[str]]
        if self.rollout_cfg.initial_games_index_file:
            self._game_json_paths = _read_game_json_paths(
                self.rollout_cfg.initial_games_index_file
            )
        else:
            self._game_json_paths = None

        assert heyhi.is_master() == (ectx.training_ddp_rank is not None)

        self._rank = heyhi.get_job_env().global_rank

        if self.rollout_cfg.num_workers_per_gpu > 0:
            self._use_buffer = True
        else:
            self._use_buffer = False

        self._game_kwargs = dict(draw_on_stalemate_years=self.rollout_cfg.draw_on_stalemate_years)

        self._assign_devices()
        self._log_dir = pathlib.Path("rollout_logs").absolute()
        self._log_dir.mkdir(exist_ok=True, parents=True)
        if self._ectx.is_training_master:
            self._ckpt_syncer = ValuePolicyCkptSyncer(
                CKPT_SYNC_DIR,
                create_dir=True,
                linear_average_policy=rollout_cfg.linear_average_sync_policy_checkpoints,
            )
        else:
            self._ckpt_syncer = None

        if self._ectx.is_training_helper:
            self._rollouter = None
        else:
            self._start_eval_procs()
            self._rollouter = Rollouter(
                self.rollout_cfg,
                log_dir=self._log_dir,
                devices=self._rollout_devices,
                cores=self.cores,
                game_json_paths=self._game_json_paths,
                game_kwargs=self._game_kwargs,
                local_mode=not self._use_buffer,
                ddp_world_size=ectx.ddp_world_size,
            )

        if self._use_buffer:
            if self._ectx.is_training_master or self._ectx.is_training_helper:
                self._initialize_buffer()
                self._need_warmup = True
                self._initialize_postman_server()

            if not self._ectx.is_training_helper:
                assert self._rollouter is not None
                self._rollouter.start_rollout_procs()
                if not self._ectx.is_training_master:
                    # will stay here until the job is dead or the master is dead.
                    self._rollouter.run_communicator_loop()
                else:
                    self._rollouter.start_communicator_proc()
        else:
            assert self._ectx.is_training_master
            assert self._rollouter is not None
            self._local_episode_iterator = self._rollouter.get_local_batch_iterator()

    def _assign_devices(self):
        """Sets what CPUs and GPUs to use.

        Sets
            self.cores
            self._rollout_devices
            self._sitcheck_device
        """
        self.cores: Optional[Tuple[int, ...]]
        self._rollout_devices: List[str]
        self._sitcheck_device: Optional[str]
        self._eval_sp_device: Optional[str]
        self._eval_h2h_devices: Optional[List[str]]
        self._num_eval_procs: int

        if self.rollout_cfg.num_cores_to_reserve and self._ectx.is_training_master:
            self.cores = tuple(range(80 - self.rollout_cfg.num_cores_to_reserve, 80))
        else:
            self.cores = None

        if not torch.cuda.is_available():
            # CircleCI probably.
            self._rollout_devices = ["cpu"]
            self._sitcheck_device = "cpu"
            self._eval_sp_device = "cpu"
            self._eval_h2h_devices = ["cpu"] * len(self.h2h_evals)
            self._num_eval_procs = 1
        else:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if (
                len(devices) > 1
                and self._ectx.is_training_master
                and not self.rollout_cfg.benchmark_only
            ):
                # If the trainer and has more than 1 gpu, don't use GPU used for training.
                devices = devices[self._num_train_gpus :]
            full_machine = torch.cuda.device_count() == 8

            # Second machine runs evals if more than one machine.
            evaler_rank = 1 if full_machine and heyhi.get_job_env().num_nodes > 1 else 0
            is_evaler_machine = heyhi.get_job_env().global_rank == evaler_rank

            if self.rollout_cfg.test_situation_eval.do_eval and is_evaler_machine:
                self._sitcheck_device = devices.pop(0) if full_machine else devices[0]
            else:
                self._sitcheck_device = None

            if is_evaler_machine and not self.rollout_cfg.local_debug:
                self._eval_sp_device = devices.pop(0) if full_machine else devices[0]
                self._eval_h2h_devices = [
                    devices.pop(0) if full_machine else devices[0] for _ in self.h2h_evals
                ]
            else:
                self._eval_sp_device = None
                self._eval_h2h_devices = None

            self._rollout_devices = devices
            self._num_eval_procs = 5 if full_machine else 1

        logging.info(f"Sit check device {self._sitcheck_device}")
        logging.info(f"Eval SelfPlay device {self._eval_sp_device}")
        logging.info(f"Eval H2H devices {self._eval_h2h_devices}")
        logging.info(f"Procs to use for evals: {self._num_eval_procs}")
        logging.info(f"Rollout devices {self._rollout_devices}")

    def _initialize_buffer(self):
        assert self.rollout_cfg.buffer.capacity is not None, "buffer.capacity is required"
        replay_params = dict(
            seed=10001,
            alpha=1.0,
            beta=0.4,
            prefetch=self.rollout_cfg.buffer.prefetch or 3,
            capacity=self.rollout_cfg.buffer.capacity // self._ectx.ddp_world_size,
            shuffle=self.rollout_cfg.buffer.shuffle,
        )
        self._buffer = rela.NestPrioritizedReplay(**replay_params)

        self._save_buffer_at = None
        self._preloaded_size = 0
        if self.rollout_cfg.buffer.load_path:
            logging.info("Loading buffer from: %s", self.rollout_cfg.buffer.load_path)
            if self._ectx.ddp_world_size > 1:
                logging.warning("Buffers on all machines will load the same content")
            assert pathlib.Path(
                self.rollout_cfg.buffer.load_path
            ).exists(), f"Cannot find the buffer dump: {self.rollout_cfg.buffer.load_path}"
            self._buffer.load(self.rollout_cfg.buffer.load_path)
            logging.info("Loaded. New size: %s", self._buffer.size())
            self._preloaded_size = self._buffer.size()
            if self.rollout_cfg.buffer.save_at:
                logging.warning("buffer.save_at is ignored")
        elif self.rollout_cfg.buffer.save_at and self.rollout_cfg.buffer.save_at > 0:
            self._save_buffer_at = self.rollout_cfg.buffer.save_at

        # If true, will go noop on postman buffer add calls.
        self._skip_buffer_adds = False

        # Timestamps where get_buffer_stats is called.
        self._first_call = time.time()
        self._last_call = time.time()
        # For throttling and stats.
        self._num_sampled = 0
        self._last_size = self._buffer.num_add()
        self._last_num_sampled = 0

    def _initialize_postman_server(self):
        num_added = 0

        def add_replay(data):
            nonlocal num_added
            if self._skip_buffer_adds:
                return
            if num_added < 10:
                logging.info(
                    "adding:\n\tdata=%s\n\tbuffer sz=%s",
                    nest.map(lambda x: x.shape, data),
                    self._buffer.size(),
                )
                num_added += 1
            data = compress_and_flatten(nest.map(lambda x: x.squeeze(0), data))
            priority = 1.0
            self._buffer.add_one(data, priority)

        def heartbeat(arg):
            del arg  # Unused.

        host = socket.gethostname()
        server = postman.Server("0.0.0.0:0")
        server.bind("add_replay", add_replay, batch_size=1)
        server.bind("heartbeat", heartbeat, batch_size=1)
        server.run()
        self._buffer_server = server
        master_addr = f"{host}:{server.port()}"
        logging.info("Kicked of postman buffer server on %s", master_addr)
        assert self._ectx.training_ddp_rank is not None
        trainer_server_fname = get_trainer_server_fname(self._ectx.training_ddp_rank)
        with open(trainer_server_fname, "w") as stream:
            print(master_addr, file=stream)

    @property
    def h2h_evals(self):
        return [
            cfg for cfg in [self.rollout_cfg.h2h_eval_0, self.rollout_cfg.h2h_eval_1] if cfg.tag
        ]

    def _start_eval_procs(self):
        if self.rollout_cfg.local_debug:
            return
        eval_agent_cfg = self.rollout_cfg.eval_agent or self.rollout_cfg.agent
        if self._sitcheck_device is not None:
            log_file = self._log_dir / "eval_sitcheck.log"
            logging.info("Starting situation check process. Logs: %s", log_file)
            self._sitcheck = TestSituationEvaller(
                cfg=self.rollout_cfg.test_situation_eval.do_eval,
                agent_cfg=eval_agent_cfg,
                ckpt_dir=pathlib.Path("ckpt/"),
                device=self._sitcheck_device,
                log_file=log_file,
            )
        if self._eval_sp_device is not None:
            self._eval_player = EvalPlayer(
                log_dir=self._log_dir,
                rollout_cfg=self.rollout_cfg,
                device=self._eval_sp_device,
                num_procs=5,
                cores=self.cores,
                ckpt_sync_path=CKPT_SYNC_DIR,
                game_json_paths=self._game_json_paths,
                game_kwargs=self._game_kwargs,
            )
        if self._eval_h2h_devices:
            self._h2h_evalers = []
            assert len(self._eval_h2h_devices) == len(self.h2h_evals)
            for device, cfg in zip(self._eval_h2h_devices, self.h2h_evals):
                if cfg.initial_games_index_file:
                    json_paths = _read_game_json_paths(cfg.initial_games_index_file)
                else:
                    json_paths = self._game_json_paths
                self._h2h_evalers.append(
                    H2HEvaler(
                        h2h_cfg=cfg,
                        log_dir=self._log_dir,
                        agent_one_cfg=eval_agent_cfg,
                        cores=self.cores,
                        ckpt_sync_path=CKPT_SYNC_DIR,
                        num_procs=self._num_eval_procs,
                        device=device,
                        game_json_paths=json_paths,
                        game_kwargs=self._game_kwargs,
                    )
                )

    def extract_eval_scores(self) -> Optional[ScoreDict]:
        return None

    def terminate(self):
        logging.warning(
            "DataLoader is being destroyed. Any data read before this point may misbehave"
        )
        if (self._ectx.is_training_master or self._ectx.is_training_helper) and self._use_buffer:
            logging.info("Stopping buffer PostMan server")
            self._buffer_server.stop()
        logging.info("Killing Rollouter")
        if self._rollouter is not None:
            self._rollouter.terminate()
        if getattr(self, "_sitcheck", None) is not None:
            self._sitcheck.terminate()
        if getattr(self, "_eval_player", None) is not None:
            self._eval_player.terminate()

    def get_buffer_stats(self, *, prefix: str) -> Dict[str, float]:
        if not self._use_buffer:
            return {}
        # Note: added stores num_add only for this buffer.
        now, added = time.time(), self._buffer.num_add()
        t_between_calls = max(1e-3, now - self._last_call)
        num_buffers = self._ectx.ddp_world_size
        # Stores total added for all (ddp) buffers.
        added_with_preload_all = added * num_buffers + self._preloaded_size
        stats = {
            f"{prefix}size": self._buffer.size(),
            f"{prefix}num_add": self._buffer.num_add(),
            f"{prefix}speed_write": (added - self._last_size) / t_between_calls,
            f"{prefix}speed_read": (self._num_sampled - self._last_num_sampled) / t_between_calls,
            f"{prefix}overall_write_speed": added / (now - self._first_call),
            f"{prefix}overall_read_speed": self._num_sampled / (now - self._first_call),
        }
        stats = {k: v * self.rollout_cfg.chunk_length * num_buffers for k, v in stats.items()}
        stats[f"{prefix}size_bytes"] = self._buffer.total_bytes() * num_buffers
        stats[f"{prefix}size_numel"] = self._buffer.total_numel() * num_buffers
        if self._rollouter is not None:
            stats[f"{prefix}num_alive_workers"] = self._rollouter.num_alive_workers()
        stats[f"{prefix}overuse"] = (
            self._num_sampled * num_buffers / max(1, added_with_preload_all)
        )
        self._last_call, self._last_size = now, added
        self._last_num_sampled = self._num_sampled
        return stats

    def sample_raw_batch_from_buffer(self):
        if self.rollout_cfg.enforce_train_gen_ratio > 0:
            while (
                self._num_sampled / (self._buffer.num_add() + self._preloaded_size)
                > self.rollout_cfg.enforce_train_gen_ratio
            ):
                print("sleep")
                time.sleep(1)
        assert self.rollout_cfg.batch_size is not None
        per_dataloader_batch_size = self.rollout_cfg.batch_size // self._ectx.ddp_world_size
        batch, _ = self._buffer.sample(per_dataloader_batch_size)
        # all_rewards = torch.cat([x for x in batch["rewards"]])
        # print("YY", len(all_rewards), all_rewards.mean(0))
        self._num_sampled += per_dataloader_batch_size
        self._buffer.keep_priority()
        return batch

    def get_batch(self) -> ReSearchRolloutBatch:
        assert self._ectx.is_training_master or self._ectx.is_training_helper
        assert self.rollout_cfg.batch_size is not None
        per_dataloader_batch_size = self.rollout_cfg.batch_size // self._ectx.ddp_world_size
        if self._use_buffer:
            # list_of_dicts, _ = self._buffer.get_all_content()
            # if list_of_dicts:
            #     all_rewards = torch.cat([x["rewards"] for x in list_of_dicts])
            #     print("XX", len(all_rewards), all_rewards.mean(0))
            if self._need_warmup:
                warmup_size = per_dataloader_batch_size * self.rollout_cfg.warmup_batches
                while self._buffer.size() < warmup_size:
                    logging.info("Warming up the buffer: %d/%d", self._buffer.size(), warmup_size)
                    if self._rollout_devices and self._rollouter is not None:
                        assert self._rollouter.is_communicator_alive(), "Oh shoot!"
                        assert self._rollouter.num_alive_workers() > 0, "Oh shoot!"
                    time.sleep(10)
                self._need_warmup = False
            if self._save_buffer_at and self._save_buffer_at < self._buffer.num_add():
                save_path = pathlib.Path(f"buffer{self._ectx.training_ddp_rank}.bin").absolute()
                logging.info("Saving buffer to %s", save_path)
                # To avoid overflowing, set buffer to read only mode.
                self._skip_buffer_adds = True
                self._buffer.save(str(save_path))
                self._skip_buffer_adds = False
                logging.info("Done.")
                self._save_buffer_at = None

            batch = self.sample_raw_batch_from_buffer()
            batch = decompress_and_unflatten(batch)
            batch = ReSearchRolloutBatch(**batch)
        else:
            batch = list(itertools.islice(self._local_episode_iterator, per_dataloader_batch_size))
            batch = [x._asdict() for x in batch]
            batch = ReSearchRolloutBatch(**nest.map_many(lambda x: torch.stack(x, 1), *batch))
            if self.rollout_cfg.buffer.shuffle:
                batch = ReSearchRolloutBatch(
                    **nest.map(lambda x: x.flatten(end_dim=1).unsqueeze(0), batch._asdict())
                )
        return batch

    def update_model(self, model, *, as_policy=True, as_value=True, **kwargs):
        if self._ectx.is_training_helper:
            return
        assert self._ectx.is_training_master
        assert self._ckpt_syncer is not None
        assert as_policy or as_value, "Should update policy, value, or both"
        if as_policy:
            self._ckpt_syncer.policy.save_state_dict(model, **kwargs)
        if as_value:
            self._ckpt_syncer.value.save_state_dict(model, **kwargs)
