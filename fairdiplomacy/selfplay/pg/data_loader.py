#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Generator, Optional, Tuple, Sequence
import collections
import itertools
import json
import logging
import pathlib
import queue as queue_lib

import psutil
import torch
import torch.utils.tensorboard


import fairdiplomacy.selfplay.metrics
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import POWERS, POWER2IDX
from fairdiplomacy.selfplay.ckpt_syncer import CkptSyncer
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
from fairdiplomacy.utils import game_scoring
from fairdiplomacy.selfplay.pg.rollout import (
    ExploitRollout,
    InferencePool,
    RolloutMode,
    model_input_transform_blueprint,
    model_output_transform_blueprint,
    model_output_transform_exploit,
    yield_rollouts,
)
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi

mp = get_multiprocessing_ctx()

QUEUE_PUT_TIMEOUT = 1.0


ScoreDict = Dict[str, float]
ScoreDictPerPower = Dict[Optional[str], ScoreDict]


RolloutBatch = collections.namedtuple(
    "RolloutBatch", "power_ids, observations, rewards, actions, logprobs, done"
)


def get_ckpt_sync_dir():
    if heyhi.is_on_slurm():
        return f"/scratch/slurm_tmpdir/{heyhi.get_slurm_job_id()}/ckpt_syncer/ckpt"
    else:
        # Use NFS. Slow, but at least don't have to clean or resolve conflicts.
        return "ckpt_syncer/ckpt"


def yield_rewarded_rollouts(reward_kwargs, rollout_kwargs, output_games: bool):
    for item in yield_rollouts(**rollout_kwargs):
        game_json = item.game_json if output_games else None
        yield (rollout_to_batch(item, reward_kwargs), game_json)


def queue_rollouts(
    out_queue: mp.Queue, reward_kwargs, rollout_kwargs, output_games: bool = False
) -> None:
    """A rollout worker function to push RolloutBatch's into the queue."""
    try:
        for item in yield_rewarded_rollouts(reward_kwargs, rollout_kwargs, output_games):
            try:
                out_queue.put(item, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def queue_scores(out_queue: mp.Queue, reward_kwargs, rollout_kwargs) -> None:
    """A rollout worker function to push scores from eval into the queue."""
    try:
        for (_, scores), _ in yield_rewarded_rollouts(
            reward_kwargs, rollout_kwargs, output_games=False
        ):
            try:
                out_queue.put(scores, timeout=QUEUE_PUT_TIMEOUT)
            except queue_lib.Full:
                continue
    except Exception as e:
        logging.exception("Got an exception in queue_rollouts: %s", e)
        raise


def _compute_simpler_reward(
    power_id: int, game_json: Dict, *, differential_reward: bool, score_name: str
) -> torch.Tensor:
    N = len(game_json["phases"]) - 1
    if differential_reward:
        target_score_history = torch.FloatTensor(
            [
                getattr(game_scoring.compute_phase_scores(power_id, phase), score_name)
                for phase in game_json["phases"]
            ]
        )
        rewards = target_score_history[1:] - target_score_history[:-1]
    else:
        scores = game_scoring.compute_game_scores(power_id, game_json)._asdict()
        rewards = torch.zeros([N], dtype=torch.float)
        rewards[-1] = scores[score_name]
    return rewards


def build_alliance_map(alliance_type: int) -> torch.Tensor:
    def groups_to_map(*alliance_groups: Sequence[str]):
        alliance_map = torch.eye(len(POWERS))
        for group_names in alliance_groups:
            group = [POWER2IDX[x] for x in group_names]
            for i, j in itertools.combinations(group, 2):
                alliance_map[i, j] = alliance_map[j, i] = 1.0
        alliance_map /= alliance_map.sum(-1, keepdim=True)
        return alliance_map

    if alliance_type == 1:
        return groups_to_map(
            ["ENGLAND", "FRANCE", "GERMANY"], ["AUSTRIA", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        )
    if alliance_type == 2:
        return groups_to_map(
            ["ENGLAND", "FRANCE", "GERMANY", "ITALY"], ["AUSTRIA", "GERMANY", "RUSSIA", "TURKEY"]
        )
    if alliance_type == 3:
        return groups_to_map(
            ["ENGLAND", "FRANCE", "RUSSIA"], ["AUSTRIA", "GERMANY", "ITALY", "TURKEY"]
        )
    if alliance_type == 4:
        return groups_to_map(
            ["FRANCE"], ["ENGLAND", "RUSSIA", "AUSTRIA", "GERMANY", "ITALY", "TURKEY"]
        )
    raise ValueError(f"Unknown alliance group: {alliance_type}")


def compute_reward(
    rollout: ExploitRollout,
    *,
    score_name: str,
    delay_penalty=False,
    differential_reward=False,
    alliance_type: Optional[int] = None,
) -> torch.Tensor:
    if not alliance_type:
        rewards = _compute_simpler_reward(
            rollout.power_id,
            rollout.game_json,
            score_name=score_name,
            differential_reward=differential_reward,
        )
    else:
        alliance_map = build_alliance_map(alliance_type)
        per_power_rewards = [
            _compute_simpler_reward(
                i,
                rollout.game_json,
                score_name=score_name,
                differential_reward=differential_reward,
            )
            for i in range(len(POWERS))
        ]
        # Shape: [T, 7].
        per_power_rewards = torch.stack(per_power_rewards, -1)
        rewards = torch.mv(per_power_rewards, alliance_map[rollout.power_id])
    if delay_penalty:
        rewards -= delay_penalty
    if rollout.first_phase:
        # Nobody cares.
        rewards = rewards[rollout.first_phase :]
    return rewards


def rollout_to_batch(
    rollout: ExploitRollout, reward_kwargs: Dict
) -> Tuple[RolloutBatch, ScoreDict]:
    # In case rollout started from an existing game.
    offset = rollout.first_phase
    assert len(rollout.game_json["phases"]) == len(rollout.actions) + 1 + offset, (
        len(rollout.game_json["phases"]),
        len(rollout.actions),
    )
    N = len(rollout.actions)
    scores = game_scoring.compute_game_scores(rollout.power_id, rollout.game_json)._asdict()

    rewards = compute_reward(rollout, **reward_kwargs)

    is_final = torch.zeros([N], dtype=torch.bool)
    is_final[-1] = True

    # Prepare observation to be used for training. Drop information about all
    # powers, but current.
    obs = DataFields(rollout.observations)
    obs["x_loc_idxs"] = obs["x_loc_idxs"][:, rollout.power_id].clone()
    obs["x_possible_actions"] = obs["x_possible_actions"][:, rollout.power_id].clone()

    rollout_batch = RolloutBatch(
        power_ids=torch.full([N], rollout.power_id, dtype=torch.long),
        rewards=rewards,
        observations=obs,
        actions=rollout.actions,
        logprobs=rollout.logprobs,
        done=is_final,
    )
    return rollout_batch, scores


def get_default_rollout_scores() -> ScoreDict:
    scores = {k: 0.0 for k in fairdiplomacy.utils.game_scoring.GameScores._fields}
    scores["queue_size"] = 0.0
    return scores


def get_default_rollout_scores_per_power() -> ScoreDictPerPower:
    return {
        power_id: get_default_rollout_scores() for power_id in [None] + list(range(len(POWERS)))
    }


def _join_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    merged = {}
    for k in RolloutBatch._fields:
        values = []
        for b in batches:
            values.append(getattr(b, k))
        if k == "observations":
            values = DataFields.cat(values)
        else:
            values = torch.cat(values, 0)
        merged[k] = values
    return RolloutBatch(**merged)


class Evaler:
    def __init__(
        self,
        *,
        model_path,
        reward_cfg,
        num_procs,
        blueprint_hostports,
        exploit_hostports,
        temperature,
        cores: Optional[Tuple[int, ...]],
        game_json_paths: Optional[Sequence[str]],
        max_length=100,
    ):
        self.model_path = model_path

        def _build_rollout_kwargs(proc_id):
            return dict(
                reward_kwargs=heyhi.conf_to_dict(reward_cfg),
                rollout_kwargs=dict(
                    mode=RolloutMode.EVAL,
                    seed=proc_id,
                    blueprint_hostports=blueprint_hostports,
                    exploit_hostports=exploit_hostports,
                    game_json_paths=game_json_paths,
                    temperature=temperature,
                    max_rollout_length=max_length,
                    batch_size=1,
                ),
            )

        logging.info("Creating eval rollout queue")
        self.queue = mp.Queue(maxsize=4000)
        logging.info("Creating eval rollout workers")
        self.procs = [
            ExceptionHandlingProcess(
                target=queue_scores,
                args=[self.queue],
                kwargs=_build_rollout_kwargs(i),
                daemon=True,
            )
            for i in range(num_procs)
        ]
        logging.info("Starting eval rollout workers")
        for p in self.procs:
            p.start()
        if cores:
            logging.info("Setting affinities")
            for p in self.procs:
                psutil.Process(p.pid).cpu_affinity(cores)
        logging.info("Done")

    def extract_scores(self) -> ScoreDict:
        qsize = self.queue.qsize()
        aggregated_scores = get_default_rollout_scores()
        for i in range(qsize):
            try:
                scores = self.queue.get_nowait()
            except queue_lib.Empty:
                logging.warning(f"Expected {qsize} elements, got {i}. Kind of odd...")
                break
            for k, v in scores.items():
                aggregated_scores[k] += v
        for k in list(aggregated_scores):
            if k != "num_games":
                aggregated_scores[k] /= max(1, aggregated_scores["num_games"])
        del aggregated_scores["queue_size"]
        return aggregated_scores

    def terminate(self):
        logging.info("Killing eval processes")
        for proc in self.procs:
            proc.kill()


class DataLoader:
    """Starts rollout processes and inference servers to generate impala data.

    Yields rollout in chunks. Each chunk contains concatenated
    block_size rollouts.

    Yields tuples (RolloutBatch, metrics).

    Metrics is a dict of some end-of-rollout metrics summed over all rollouts.
    """

    def __init__(self, model_path, rollout_cfg: "conf.conf_cfgs.ExploitTask.Rollout"):
        self.model_path = model_path
        self.rollout_cfg = rollout_cfg

        if rollout_cfg.initial_games_index_file:
            self._game_json_paths = []
            with open(rollout_cfg.initial_games_index_file) as stream:
                for line in stream:
                    line = line.strip()
                    if line:
                        self._game_json_paths.append(line)
            assert self._game_json_paths, rollout_cfg.game_json_paths
        else:
            self._game_json_paths = None

        self.cores: Optional[Tuple[int, ...]]
        if rollout_cfg.num_cores_to_reserve:
            self.cores = tuple(range(80 - rollout_cfg.num_cores_to_reserve, 80))
        else:
            self.cores = None

        assert heyhi.get_job_env().num_nodes == 1, "Multinode is not supported"
        self._ckpt_syncer = CkptSyncer(get_ckpt_sync_dir(), create_dir=True)

        self._start_inference_procs()
        self._start_rollout_procs()
        self._maybe_start_eval_procs()
        self._batch_iterator = iter(self._yield_batches())

    def _start_inference_procs(self):
        if not torch.cuda.is_available():
            logging.warning("No GPUs found! Will run postman on CPUs.")
            # Running on CPUs.
            inference_gpus = [None]
        elif torch.cuda.device_count() == 2:
            if self.rollout_cfg.single_rollout_gpu:
                inference_gpus = [1]
            else:
                inference_gpus = [0, 1]
        else:
            assert torch.cuda.device_count() == 8, torch.cuda.device_count()
            if self.rollout_cfg.single_rollout_gpu:
                inference_gpus = [1, 2]
            else:
                inference_gpus = [1, 2, 3, 4, 5, 6, 7]

        if len(inference_gpus) == 1:
            exploit_gpus = blueprint_gpus = inference_gpus
        elif self.rollout_cfg.selfplay:
            # Assign only one GPU for eval.
            blueprint_gpus = inference_gpus[:1]
            exploit_gpus = inference_gpus[1:]
        else:
            exploit_gpus = inference_gpus[: len(inference_gpus) // 2]
            blueprint_gpus = inference_gpus[len(inference_gpus) // 2 :]

        logging.info("Starting exploit PostMan servers on gpus: %s", exploit_gpus)
        exploit_inference_pool = InferencePool(
            model_path=self.model_path,
            gpu_ids=exploit_gpus,
            ckpt_sync_path=get_ckpt_sync_dir(),
            ckpt_sync_every=self.rollout_cfg.inference_ckpt_sync_every,
            max_batch_size=self.rollout_cfg.inference_batch_size,
            server_procs_per_gpu=self.rollout_cfg.server_procs_per_gpu,
            model_output_transform=model_output_transform_exploit,
        )
        logging.info("Starting blueprint PostMan servers on gpus: %s", blueprint_gpus)
        blueprint_inference_pool = InferencePool(
            model_path=self.rollout_cfg.blueprint_model_path or self.model_path,
            gpu_ids=blueprint_gpus,
            ckpt_sync_path=None,
            max_batch_size=self.rollout_cfg.inference_batch_size,
            server_procs_per_gpu=self.rollout_cfg.server_procs_per_gpu,
            model_input_transform=model_input_transform_blueprint,
            model_output_transform=model_output_transform_blueprint,
        )

        # Must save reference to the pools to keep the resources alive.
        self.blueprint_inference_pool = blueprint_inference_pool
        self.exploit_inference_pool = exploit_inference_pool

        self.blueprint_hostports = blueprint_inference_pool.hostports
        self.exploit_hostports = exploit_inference_pool.hostports

    def _start_rollout_procs(self):
        rollout_cfg = self.rollout_cfg

        def _build_rollout_kwargs(proc_id):
            if not rollout_cfg.selfplay:
                mode = RolloutMode.EXPLOIT
            else:
                mode = RolloutMode.SELFPLAY
            assert rollout_cfg.blueprint_temperature > 0
            return dict(
                reward_kwargs=heyhi.conf_to_dict(rollout_cfg.reward),
                rollout_kwargs=dict(
                    mode=mode,
                    seed=proc_id,
                    blueprint_hostports=self.blueprint_hostports,
                    exploit_hostports=self.exploit_hostports,
                    temperature=rollout_cfg.blueprint_temperature,
                    game_json_paths=self._game_json_paths,
                    max_rollout_length=rollout_cfg.rollout_max_length,
                    batch_size=rollout_cfg.rollout_batch_size,
                    fast_finish=rollout_cfg.fast_finish,
                ),
                output_games=rollout_cfg.dump_games_every > 0,
            )

        if rollout_cfg.num_rollout_processes > 0:
            logging.info("Creating rollout queue")
            queue = mp.Queue(maxsize=40)
            logging.info("Creating rollout workers")
            procs = [
                ExceptionHandlingProcess(
                    target=queue_rollouts,
                    args=[queue],
                    kwargs=_build_rollout_kwargs(i),
                    daemon=True,
                )
                for i in range(rollout_cfg.num_rollout_processes)
            ]
            logging.info("Starting rollout workers")
            for p in procs:
                p.start()
            if self.cores:
                logging.info("Setting affinities")
                for p in procs:
                    psutil.Process(p.pid).cpu_affinity(self.cores)
            logging.info("Done")
            rollout_generator = (queue.get() for _ in itertools.count())
            # Keeping track of there to prevent garbage collection.
            self._rollout_procs = procs
        else:
            queue = None
            rollout_generator = yield_rewarded_rollouts(**_build_rollout_kwargs(0))
            self._rollout_procs = None

        self.rollout_iterator = iter(rollout_generator)
        self.queue = queue

    def _maybe_start_eval_procs(self):
        if not self.rollout_cfg.selfplay or self.rollout_cfg.num_eval_rollout_processes == 0:
            self.evaler = None
        else:
            self.evaler = Evaler(
                model_path=self.rollout_cfg.blueprint_model_path or self.model_path,
                reward_cfg=self.rollout_cfg.reward,
                num_procs=self.rollout_cfg.num_eval_rollout_processes,
                blueprint_hostports=self.blueprint_hostports,
                exploit_hostports=self.exploit_hostports,
                temperature=self.rollout_cfg.blueprint_temperature,
                game_json_paths=self._game_json_paths,
                cores=self.cores,
            )

    def extract_eval_scores(self) -> Optional[ScoreDict]:
        if self.evaler is None:
            return None
        else:
            return self.evaler.extract_scores()

    def terminate(self):
        logging.warning(
            "DataLoader is being destroyed. Any data read before this point may misbehave"
        )
        if self._rollout_procs is not None:
            logging.info("Killing rollout processes")
            for proc in self._rollout_procs:
                proc.kill()
        if self.evaler is not None:
            self.evaler.terminate()
        self.blueprint_inference_pool.terminate()
        self.exploit_inference_pool.terminate()

    def _yield_batches(self) -> Generator[Tuple[RolloutBatch, ScoreDictPerPower], None, None]:
        rollout_cfg = self.rollout_cfg
        accumulated_batches = []
        aggregated_scores = get_default_rollout_scores_per_power()
        size = 0
        batch_size = rollout_cfg.batch_size
        for rollout_id in itertools.count():
            rollout: RolloutBatch
            scores: ScoreDict
            (rollout, scores), game_json = next(self.rollout_iterator)
            power_id = rollout.power_ids[
                0
            ].item()  # Somewhat a hack - we know all power ids are the same.
            if rollout_cfg.dump_games_every and rollout_id % rollout_cfg.dump_games_every == 0:
                game_dump_folder = pathlib.Path("dumped_games")
                game_dump_folder.mkdir(exist_ok=True, parents=True)
                dump_path = game_dump_folder / f"game.{rollout_id:09d}.{POWERS[power_id]}.json"
                with (dump_path).open("w") as stream:
                    json.dump(game_json, stream)

            size += len(rollout.rewards)
            accumulated_batches.append(rollout)
            for k, v in scores.items():
                aggregated_scores[power_id][k] += v
                aggregated_scores[None][k] += v
            if rollout_cfg.do_not_split_rollouts:
                if size >= batch_size:
                    yield _join_batches(accumulated_batches), aggregated_scores
                    # Reset.
                    accumulated_batches = []
                    aggregated_scores = get_default_rollout_scores_per_power()
                    size = 0
            elif size > batch_size:
                # Use strict > to simplify the code.
                joined_batch = _join_batches(accumulated_batches)
                while size > batch_size:
                    extracted_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[:batch_size], joined_batch
                    )
                    joined_batch = fairdiplomacy.selfplay.metrics.rec_map(
                        lambda x: x[batch_size - rollout_cfg.batch_interleave_size :], joined_batch
                    )
                    size -= batch_size - rollout_cfg.batch_interleave_size
                    if self.queue is not None:
                        # Hack. We want queue size to be the size of the queue when batch is produced.
                        aggregated_scores[None]["queue_size"] = (
                            self.queue.qsize() * aggregated_scores[None]["num_games"]
                        )
                    yield extracted_batch, aggregated_scores
                    # Reset.
                    aggregated_scores = get_default_rollout_scores_per_power()
                accumulated_batches = [joined_batch]

    def get_batch(self) -> Tuple[RolloutBatch, ScoreDict]:
        return next(self._batch_iterator)

    def update_model(self, model, **kwargs):
        self._ckpt_syncer.save_state_dict(model, **kwargs)
