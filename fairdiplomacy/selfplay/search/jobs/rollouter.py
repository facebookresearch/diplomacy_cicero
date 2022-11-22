#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Generator, Optional, Sequence
import itertools
import logging
import pathlib
import socket
import time

import postman
import psutil
import torch


import conf.conf_cfgs
import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.selfplay.search.rollout import (
    ReSearchRolloutBatch,
    yield_rollouts,
    queue_rollouts,
)
from fairdiplomacy.selfplay.search.search_utils import perform_retry_loop
from fairdiplomacy.selfplay.staged_metrics_writer import StagedLogger
from fairdiplomacy.selfplay.paths import get_trainer_server_fname, CKPT_SYNC_DIR
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()


class Rollouter:
    """A supervisor for a group of processes that are doing rollouts.

    The group consist of a bunch of rollout workers and single communication
    worker that aggregates the data and sends it to the master. The user is
    expected to call start_rollout_procs and then either
    run_communicator_loop or start_communicator_loop.

    The class also provides functionality to query rollouts directly without
    launching any processes. In this case the client is expected to pass
    local_mode=True and call get_local_batch_iterator.
    """

    def __init__(
        self,
        rollout_cfg: conf.conf_cfgs.ExploitTask.SearchRollout,
        log_dir: pathlib.Path,
        *,
        game_json_paths: Optional[Sequence[str]] = None,
        local_mode: bool = False,
        game_kwargs: Dict,
        devices: Sequence[str],
        cores: Optional[Sequence[int]] = None,
        ddp_world_size: int,
    ):
        self.rollout_cfg = rollout_cfg
        self._cores = cores
        self._rollout_devices = devices
        job_env = heyhi.get_job_env()
        self._rank = job_env.global_rank
        self._local_mode = local_mode
        self._log_dir = log_dir
        self._game_json_paths = game_json_paths
        self._game_kwargs = game_kwargs
        self._communicator_proc = None
        self._rollout_procs = None
        self._ddp_world_size = ddp_world_size

        # Last machine. Rationale: if we have one machine, then write logs on
        # it. If we have many machines, than machines 0 (traineer) and 1
        # (evalers) are less represetative than the last machine.
        is_writing_stats = not local_mode and (self._rank == job_env.num_nodes - 1)
        if is_writing_stats:
            logging.info("This machine will write Rollouter logs")
            self._stats_server, self._stats_server_addr = self._initialize_postman_server()
            self._logger = StagedLogger(tag="rollouter", min_samples=100)
        else:
            self._stats_server = None
            self._logger = None

    def _initialize_postman_server(self):
        def add_stats(stats):
            stats = {k: v.item() for k, v in stats.items()}
            epoch = stats.pop("epoch")
            assert self._logger is not None
            self._logger.add_metrics(data=stats, global_step=epoch)

        host = socket.gethostname()
        server = postman.Server("0.0.0.0:0")
        server.bind("add_stats", add_stats, batch_size=1)
        server.run()
        master_addr = f"{host}:{server.port()}"
        logging.info("Kicked of postman stats server on %s", master_addr)
        return server, master_addr

    def terminate(self):
        if self._communicator_proc is not None:
            logging.info("Killing collector process")
            self._communicator_proc.kill()
        if self._rollout_procs is not None:
            logging.info("Killing rollout processes")
            for proc in self._rollout_procs:
                proc.kill()
        if self._stats_server is not None:
            logging.info("Stopping Rollouter stats PostMan server")
            self._stats_server.stop()
        if self._logger is not None:
            self._logger.close()

    def num_alive_workers(self):
        if self._local_mode:
            return 1
        if self._rollout_procs is None:
            return 0
        return sum(int(proc.is_alive()) for proc in self._rollout_procs)

    def is_communicator_alive(self):
        assert self._communicator_proc is not None
        return self._communicator_proc.is_alive()

    def get_local_batch_iterator(self):
        assert self._local_mode, "Not in local mode"
        rollout_generator = iter(
            _drop_meta(yield_rollouts(**self._build_rollout_kwargs(proc_id=0)))
        )
        return iter(
            self._yield_batches(
                chunk_length=self.rollout_cfg.chunk_length, rollout_iterator=rollout_generator  # type: ignore
            )
        )

    @classmethod
    def _connect_to_masters(cls, ddp_world_size: int):
        timeout_secs = 60 * 60  # Wait 1h and die.
        sleep_secs = 10

        addrs_and_clients = []
        for training_ddp_rank in range(ddp_world_size):
            trainer_server_fname = get_trainer_server_fname(training_ddp_rank)
            success = False
            for _ in range(timeout_secs // sleep_secs + 1):
                if not trainer_server_fname.exists():
                    logging.info("Waiting for %s to appear", trainer_server_fname)
                    time.sleep(5)
                    continue
                with open(trainer_server_fname) as stream:
                    master_addr = stream.read().strip()
                logging.info("Trying to connect to %s", master_addr)
                try:
                    buffer_client = postman.Client(master_addr)
                    buffer_client.connect()
                    buffer_client.heartbeat(torch.zeros(1))
                except Exception as e:
                    logging.error("Got error: %s", e)
                    time.sleep(5)
                    continue
                logging.info("Successfully connected to %s", master_addr)
                addrs_and_clients.append((master_addr, buffer_client))
                success = True
                break
            if not success:
                raise RuntimeError(
                    "Failed to connect to the trainer after %d minutes", timeout_secs // 60
                )
        return addrs_and_clients

    def _build_rollout_kwargs(self, proc_id: int) -> Dict:
        return dict(
            agent_cfg=self.rollout_cfg.agent,
            game_json_paths=self._game_json_paths,
            seed=proc_id,
            device=self._rollout_devices[proc_id % len(self._rollout_devices)],
            ckpt_sync_path=CKPT_SYNC_DIR,
            game_kwargs=self._game_kwargs,
            extra_params_cfg=self.rollout_cfg.extra_params,
        )

    def start_rollout_procs(self):
        logging.info("Rollout devices: %s", self._rollout_devices)
        if not self._rollout_devices:
            logging.warning("No devices available. No rollout workers will be launched")
            self._rollout_procs = []
            return

        rollout_cfg = self.rollout_cfg
        assert not self._local_mode
        assert rollout_cfg.num_workers_per_gpu > 0
        logging.info("Creating rollout queue")
        queue = mp.Queue(maxsize=40)
        logging.info("Creating rollout workers")
        procs = []
        for i in range(rollout_cfg.num_workers_per_gpu * len(self._rollout_devices)):
            kwargs = self._build_rollout_kwargs(i)
            kwargs["need_games"] = False
            if self._stats_server is not None:
                kwargs["stats_server"] = self._stats_server_addr
            if rollout_cfg.verbosity >= 1:
                log_path = self._log_dir / f"rollout_{self._rank:03d}_{i:03d}.log"
                kwargs["log_path"] = log_path
                kwargs["log_level"] = (
                    logging.INFO if i == 0 or rollout_cfg.verbosity >= 2 else logging.WARNING
                )
                logging.info(
                    f"Rollout process {i} will write logs to {log_path} at level %s",
                    kwargs["log_level"],
                )
            procs.append(
                ExceptionHandlingProcess(
                    target=queue_rollouts, args=[queue], kwargs=kwargs, daemon=True
                )
            )
        logging.info("Starting rollout workers")
        for p in procs:
            p.start()
        if self._cores:
            logging.info("Setting affinities")
            for p in procs:
                psutil.Process(p.pid).cpu_affinity(list(self._cores))
        logging.info("Done")
        self._rollout_generator = (queue.get() for _ in itertools.count())
        self._rollout_procs = procs
        self.queue = queue

    @classmethod
    def _communicator_worker(cls, chunk_length, queue, ddp_world_size: int):
        addrs_and_clients = cls._connect_to_masters(ddp_world_size)
        rollout_generator = (queue.get() for _ in itertools.count())
        data_generator = cls._yield_batches(
            chunk_length=chunk_length, rollout_iterator=iter(rollout_generator)  # type: ignore
        )

        num_clients = len(addrs_and_clients)
        job_env = heyhi.get_job_env()
        next_client_idx = job_env.global_rank % num_clients

        for i, batch in enumerate(data_generator):
            if i & (i + 1) == 0:
                logging.info("Collector got batch %d", i + 1)

            _master_addr, client = addrs_and_clients[next_client_idx]
            next_client_idx = (next_client_idx + 1) % num_clients

            # The master may be unresponsive while spawning and joining on training helper processes from
            # the multiprocessing module, or on torch.distributed.init_process_group
            # So try again a bunch of times before actually failing
            def send_to_client():
                client.add_replay(batch._asdict())

            perform_retry_loop(send_to_client, max_tries=20, sleep_seconds=10)

    def run_communicator_loop(self):
        """Run communication loop in the current process. Never returns."""
        assert self.rollout_cfg.num_workers_per_gpu > 0
        assert self._rollout_procs
        assert self.queue is not None
        self._communicator_worker(
            queue=self.queue,
            chunk_length=self.rollout_cfg.chunk_length,
            ddp_world_size=self._ddp_world_size,
        )

    def start_communicator_proc(self):
        """Starts a process that reads rollout_iterator and pushes into the buffer."""
        if not self._rollout_procs:
            logging.warning(
                "Not starting communcator process as no rollouts procs on this machine"
            )
            return

        assert self.queue is not None
        self._communicator_proc = ExceptionHandlingProcess(
            target=self._communicator_worker,
            kwargs=dict(
                queue=self.queue,
                chunk_length=self.rollout_cfg.chunk_length,
                ddp_world_size=self._ddp_world_size,
            ),
        )
        self._communicator_proc.start()

    @classmethod
    def _yield_batches(
        cls, chunk_length: int, rollout_iterator: Generator[ReSearchRolloutBatch, None, None]
    ) -> Generator[ReSearchRolloutBatch, None, None]:
        accumulated_batches = []
        size = 0
        assert chunk_length > 0
        for _ in itertools.count():
            rollout: ReSearchRolloutBatch
            rollout = next(rollout_iterator)
            size += len(rollout.rewards)
            accumulated_batches.append(rollout)
            if size > chunk_length:
                # Use strict > to simplify the code.
                joined_batch = _join_batches(accumulated_batches)  # type: ignore
                while size > chunk_length:
                    extracted_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[:chunk_length], joined_batch
                    )
                    joined_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[chunk_length:], joined_batch
                    )
                    size -= chunk_length
                    yield extracted_batch  # type: ignore
                accumulated_batches = [joined_batch]


def _drop_meta(generator):
    for rollout_result in generator:
        yield rollout_result.batch


def _join_batches(batches: Sequence[ReSearchRolloutBatch]) -> ReSearchRolloutBatch:
    merged = {}
    for k in ReSearchRolloutBatch._fields:
        values = []
        for b in batches:
            values.append(getattr(b, k))
        if k == "observations":
            values = DataFields.cat(values)
        else:
            values = torch.cat(values, 0)
        merged[k] = values
    return ReSearchRolloutBatch(**merged)
