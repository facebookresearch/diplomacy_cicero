#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Optional, Tuple, Sequence
import collections
import logging
import pathlib
import time

import psutil
import torch.multiprocessing as mp_module

import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.plausible_order_sampling import are_supports_coordinated
from fairdiplomacy.get_xpower_supports import compute_xpower_supports
from fairdiplomacy.selfplay.search.jobs.common import save_game
from fairdiplomacy.selfplay.search.rollout import queue_rollouts
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx

mp = get_multiprocessing_ctx()

# Do not dump a game on disk more often that this.
GAME_WRITE_TIMEOUT = 60
# Aggregate at least this many game before generating any stats.
MIN_GAMES_FOR_STATS = 50


class EvalPlayer:
    """A group of processes that run eval selfplay on one GPU."""

    def __init__(
        self,
        *,
        log_dir,
        rollout_cfg,
        device,
        ckpt_sync_path,
        num_procs=5,
        game_kwargs: Dict,
        cores: Optional[Tuple[int, ...]],
        game_json_paths: Optional[Sequence[str]],
    ):
        def _build_rollout_kwargs(proc_id):
            return dict(
                agent_cfg=rollout_cfg.agent,
                game_json_paths=game_json_paths,
                seed=proc_id,
                device=device,
                ckpt_sync_path=ckpt_sync_path,
                need_games=True,
                game_kwargs=game_kwargs,
                extra_params_cfg=rollout_cfg.extra_params,
            )

        logging.info("Creating eval rollout queue")
        self.queue = mp.Queue(maxsize=4000)
        tag = "sp"
        logging.info("Creating eval rollout workers")
        self.procs = []
        for i in range(num_procs):
            kwargs = _build_rollout_kwargs(i)
            log_path = log_dir / f"eval_rollout_{tag}_{i:03d}.log"
            kwargs["log_path"] = log_path
            kwargs["log_level"] = logging.WARNING
            logging.info(
                f"Eval Rollout process {i} will write logs to {log_path} at level %s",
                kwargs["log_level"],
            )
            kwargs["collect_game_logs"] = True
            kwargs["eval_mode"] = True
            self.procs.append(
                ExceptionHandlingProcess(
                    target=queue_rollouts, args=[self.queue], kwargs=kwargs, daemon=True
                )
            )
        logging.info("Adding saving worker")
        self.procs.append(
            ExceptionHandlingProcess(
                target=self.aggregate_worker,
                args=[],
                kwargs=dict(
                    queue=self.queue,
                    dst_dir=pathlib.Path(f"games_{tag}").absolute(),
                    tag=tag,
                    save_every_secs=GAME_WRITE_TIMEOUT,
                ),
                daemon=True,
            )
        )

        logging.info("Starting eval rollout workers")
        for p in self.procs:
            p.start()
        if cores:
            logging.info("Setting affinities")
            for p in self.procs:
                psutil.Process(p.pid).cpu_affinity(list(cores))
        logging.info("Done")

    @classmethod
    def aggregate_worker(
        cls, *, queue: mp_module.Queue, tag: str, dst_dir: pathlib.Path, save_every_secs: float
    ):
        dst_dir.mkdir(exist_ok=True, parents=True)
        logger = fairdiplomacy.selfplay.remote_metric_logger.get_remote_logger(tag="eval_sp")
        counters = collections.defaultdict(fairdiplomacy.selfplay.metrics.FractionCounter)
        max_seen_epoch = -1
        num_games = 0

        def process_metrics(epoch, game_json):
            nonlocal logger
            nonlocal counters
            nonlocal max_seen_epoch
            nonlocal num_games

            if max_seen_epoch < epoch:
                if num_games >= MIN_GAMES_FOR_STATS:
                    metrics = {
                        f"eval_{tag}/{key}": value.value() for key, value in counters.items()
                    }
                    metrics[f"eval_{tag}/num_games"] = num_games
                    logger.log_metrics(metrics, max_seen_epoch)
                    counters.clear()
                    num_games = 0
                max_seen_epoch = epoch
            num_games += 1
            game = pydipcc.Game.from_json(game_json)
            x_supports = compute_xpower_supports(game)
            counters["num_orders"].update(x_supports.num_orders)
            counters["sup_to_all_share"].update(x_supports.num_supports, x_supports.num_orders)
            counters["sup_xpower_to_sup_share"].update(
                x_supports.num_xpower, x_supports.num_supports
            )
            counters["episode_length"].update(len(game.get_phase_history()))
            n_orders, n_coordinated = 0, 0
            for phase_data in game.get_phase_history():
                for action in phase_data.orders.values():
                    n_orders += 1
                    n_coordinated += are_supports_coordinated(tuple(action))
            counters["coordindated_share"].update(n_coordinated, n_orders)
            counters["r_draw"].update(max(game.get_scores()) < 0.99)

        last_save = 0.0
        while True:
            tensors, game_meta, last_ckpt_metas = queue.get()
            try:
                epoch = max(meta["epoch"] for meta in last_ckpt_metas.values())
            except KeyError:
                logging.error("Bad Meta %s:", last_ckpt_metas)
                raise

            game_json = game_meta["game"]
            process_metrics(epoch, game_json)

            now = time.time()
            if now - last_save > save_every_secs:
                save_game(
                    tensors=tensors,
                    game_json=game_json,
                    epoch=epoch,
                    dst_dir=dst_dir,
                    game_id=game_meta["game_id"],
                    start_phase=game_meta["start_phase"],
                )
                last_save = now

    def terminate(self):
        logging.info("Killing eval processes")
        for proc in self.procs:
            proc.kill()
        self.procs = []
