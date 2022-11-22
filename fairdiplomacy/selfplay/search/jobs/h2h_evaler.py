#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Optional, Tuple, Sequence

import collections
import io
import logging
import pathlib
import random
import time

import psutil
import torch
import torch.multiprocessing as mp_module
from fairdiplomacy.agents.player import Player

import fairdiplomacy.selfplay.metrics
import fairdiplomacy.selfplay.remote_metric_logger
from fairdiplomacy import pydipcc
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.get_xpower_supports import compute_xpower_supports
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.selfplay.ckpt_syncer import build_searchbot_like_agent_with_syncs
from fairdiplomacy.selfplay.search.jobs.common import save_game
from fairdiplomacy.selfplay.search.search_utils import unparse_device
from fairdiplomacy.selfplay.search.rollout import yield_game
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()

# Do not dump a game on disk more often that this.
GAME_WRITE_TIMEOUT = 60


class H2HEvaler:
    def __init__(
        self,
        *,
        log_dir,
        h2h_cfg,
        agent_one_cfg,
        device,
        ckpt_sync_path,
        num_procs,
        game_kwargs: Dict,
        cores: Optional[Tuple[int, ...]],
        game_json_paths: Optional[Sequence[str]],
    ):

        logging.info(f"Creating eval h2h {h2h_cfg.tag} rollout workers")
        self.queue = mp.Queue(maxsize=4000)
        self.procs = []
        for i in range(num_procs):
            log_path = log_dir / f"eval_h2h_{h2h_cfg.tag}_{i:03d}.log"
            kwargs = dict(
                queue=self.queue,
                device=device,
                ckpt_sync_path=ckpt_sync_path,
                agent_one_cfg=agent_one_cfg,
                agent_six_cfg=h2h_cfg.agent_six,
                game_json_paths=game_json_paths,
                game_kwargs=game_kwargs,
                seed=i,
                num_zero_epoch_evals=h2h_cfg.min_games_for_stats // num_procs + 5,
                use_trained_value=h2h_cfg.use_trained_value,
                use_trained_policy=h2h_cfg.use_trained_policy,
                disable_exploit=h2h_cfg.disable_exploit,
                play_as_six=h2h_cfg.play_as_six,
            )
            kwargs["log_path"] = log_path
            kwargs["log_level"] = logging.INFO
            logging.info(
                f"H2H Rollout process {h2h_cfg.tag}/{i} will write logs to {log_path} at level %s",
                kwargs["log_level"],
            )
            self.procs.append(
                ExceptionHandlingProcess(target=self.eval_worker, kwargs=kwargs, daemon=True)
            )
        logging.info(f"Adding main h2h {h2h_cfg.tag} worker")
        self.procs.append(
            ExceptionHandlingProcess(
                target=self.aggregate_worker,
                kwargs=dict(
                    queue=self.queue,
                    tag=h2h_cfg.tag,
                    min_games_for_stats=h2h_cfg.min_games_for_stats,
                    save_every_secs=GAME_WRITE_TIMEOUT,
                ),
                daemon=True,
            )
        )

        logging.info(f"Starting h2h {h2h_cfg.tag} workers")
        for p in self.procs:
            p.start()
        if cores:
            logging.info("Setting affinities")
            for p in self.procs:
                psutil.Process(p.pid).cpu_affinity(list(cores))
        logging.info("Done")

    @classmethod
    def eval_worker(
        cls,
        *,
        seed,
        queue: mp_module.Queue,
        device: str,
        ckpt_sync_path: str,
        log_path: pathlib.Path,
        log_level,
        agent_one_cfg,
        agent_six_cfg,
        game_json_paths,
        game_kwargs: Dict,
        num_zero_epoch_evals: int,
        use_trained_value: bool,
        use_trained_policy: bool,
        disable_exploit: bool,
        play_as_six: bool,
    ):

        # We collect this many games for the first ckpt before loading new
        # ckpt. This is to establish an accurate BL numbers where RL agent net
        # in equivalent to the blueprint it's initialized from.
        num_evals_without_reload_left = num_zero_epoch_evals

        heyhi.setup_logging(console_level=None, fpath=log_path, file_level=log_level)

        device_id = unparse_device(device)
        agent_one, do_sync_fn = build_searchbot_like_agent_with_syncs(
            agent_one_cfg,
            ckpt_sync_path=ckpt_sync_path,
            use_trained_policy=use_trained_policy,
            use_trained_value=use_trained_value,
            device_id=device_id,
            disable_exploit=disable_exploit,
        )
        exploited_agent_power = agent_one.get_exploited_agent_power()
        agent_six = build_agent_from_cfg(agent_six_cfg, device=device_id)

        if play_as_six:
            assert exploited_agent_power is None, "Cannot mix exploit and play_as_six"
            agent_six, agent_one = agent_one, agent_six

        random.seed(seed)
        torch.manual_seed(seed)

        # Hack: using whatever syncer is listed first to detect epoch.
        main_meta = next(iter(do_sync_fn().values()))

        if main_meta["epoch"] > 0:
            # First ckpt is not on zero epoch. Disabling.
            num_evals_without_reload_left = 0

        logger = logging.getLogger("")
        logger.info("Collecting logs into game json")
        _set_all_logger_handlers_to_level(logger, logging.WARNING)
        game_log_handler = None

        for game_id, game in yield_game(
            seed, game_json_paths, game_kwargs, sample_game_json_phases=False, logger=logger,
        ):
            start_phase = game.current_short_phase
            if num_evals_without_reload_left > 0:
                num_evals_without_reload_left -= 1
            else:
                main_meta = next(iter(do_sync_fn().values()))

            # Agent one must be alive at the start of the game.
            starting_sos = game.get_scores()
            agent_one_power = random.choice(
                [
                    p
                    for p, score in zip(POWERS, starting_sos)
                    if score > 1e-3 and p != exploited_agent_power
                ]
            )
            agent_one_power_idx = POWERS.index(agent_one_power)
            player_one = Player(agent_one, agent_one_power)

            while not game.is_game_done and game.get_scores()[agent_one_power_idx] > 1e-3:
                if game_log_handler is not None:
                    logger.removeHandler(game_log_handler)
                log_stream = io.StringIO()
                game_log_handler = logging.StreamHandler(log_stream)
                game_log_handler.setLevel(logging.INFO)
                logger.addHandler(game_log_handler)

                power_orders = {}
                logger.info(f"Running agent one power {agent_one_power}")
                power_orders[agent_one_power] = player_one.get_orders(game)

                six_powers = [p for p in POWERS if p != agent_one_power]
                logger.info("Running agent six powers")
                power_orders.update(agent_six.get_orders_many_powers(game, six_powers))

                for power, orders in power_orders.items():
                    if not game.get_orderable_locations().get(power):
                        continue
                    game.set_orders(power, orders)
                game.add_log(log_stream.getvalue())
                game.process()

                # After any spring movement phase check if the agent is configured to evaluate
                # that the game may randomly end during spring. If true, then also randomly end during
                # after spring with the same chance.
                if game.current_short_phase.startswith("S") and game.current_short_phase.endswith(
                    "M"
                ):
                    current_year = int(game.current_short_phase[1:-1])
                    prob_ending = agent_one.base_strategy_model_rollouts.get_prob_of_spring_ending(
                        current_year
                    )
                    if random.random() < prob_ending:
                        break

            queue.put(
                {
                    "last_ckpt_meta": main_meta,
                    "game_json": game.to_json(),
                    "agent_one_power": agent_one_power,
                    "game_id": game_id,
                    "start_phase": start_phase,
                }
            )

    @classmethod
    def aggregate_worker(
        cls, *, queue: mp_module.Queue, tag: str, save_every_secs: float, min_games_for_stats: int
    ):
        logger = fairdiplomacy.selfplay.remote_metric_logger.get_remote_logger(
            tag=f"eval_h2h_{tag}"
        )
        counters = collections.defaultdict(fairdiplomacy.selfplay.metrics.FractionCounter)
        max_seen_epoch = -1
        num_games = 0

        def process_metrics(epoch, game_json, power):
            nonlocal logger
            nonlocal counters
            nonlocal max_seen_epoch
            nonlocal num_games

            if max_seen_epoch < epoch:
                if num_games >= min_games_for_stats:
                    metrics = {
                        f"eval_h2h_{tag}/{key}": value.value() for key, value in counters.items()
                    }
                    metrics[f"eval_h2h_{tag}/num_games"] = num_games
                    logger.log_metrics(metrics, max_seen_epoch)
                    counters.clear()
                    num_games = 0
                max_seen_epoch = epoch

            num_games += 1
            game = pydipcc.Game.from_json(game_json)
            counters["episode_length"].update(len(game.get_phase_history()))
            scores = game.get_scores()
            counters["r_draw_all"].update(max(scores) < 0.99)
            counters["r_solo"].update(scores[POWERS.index(power)] > 0.99)
            counters["r_draw"].update(0.001 < scores[POWERS.index(power)] < 0.99)
            counters["r_dead"].update(scores[POWERS.index(power)] < 0.001)
            counters["r_square_score"].update(scores[POWERS.index(power)])

            x_supports_power = compute_xpower_supports(game, only_power=power)
            counters["sup_to_all_share"].update(
                x_supports_power.num_supports, x_supports_power.num_orders
            )
            counters["sup_xpower_to_sup_share"].update(
                x_supports_power.num_xpower, x_supports_power.num_supports
            )

        last_save = 0
        game_dump_path = pathlib.Path(f"games_h2h_{tag}").absolute()
        game_dump_path.mkdir(exist_ok=True, parents=True)
        while True:
            data = queue.get()
            try:
                epoch = data["last_ckpt_meta"]["epoch"]
            except KeyError:
                logging.error("Bad Meta: %s", data["last_ckpt_meta"])
                raise

            process_metrics(epoch, data["game_json"], data["agent_one_power"])

            now = time.time()
            if now - last_save > save_every_secs:
                save_game(
                    game_json=data["game_json"],
                    epoch=epoch,
                    dst_dir=game_dump_path,
                    game_id=data["game_id"],
                    start_phase=data["start_phase"],
                    agent_one_power=data["agent_one_power"],
                )
                last_save = now

    def terminate(self):
        logging.info("Killing H2H processes")
        for proc in self.procs:
            proc.kill()
        self.procs = []


def _set_all_logger_handlers_to_level(logger, level):
    for handler in logger.handlers[:]:
        handler.setLevel(level)
