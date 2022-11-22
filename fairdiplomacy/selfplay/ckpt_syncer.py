#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Simple mechanism to keep rollout models up to date.

CkptSyncer is initialized with a path to store checkpoints.

The trainer calls save() to save model in the folder.
Rollout worker class get_last_version() to get last version id and path to
the checkpoint. If the version differs from the last call, the worker should
reload the model.

The class can save/load arbitrary stuff. To work with torch modules check
helper functions save_state_dict and maybe_load_state_dict.

To prevent race conditions on NFS, the class maintains last `models_to_keep`
on the disk AND does atomic writes.
"""
import glob
import logging
import os
import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn
from conf import agents_cfgs, agents_pb2
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent
from fairdiplomacy.agents.the_best_agent import TheBestAgent

ModelVersion = int
ModelStateDict = Dict[str, torch.Tensor]


class CkptSyncer:
    def __init__(
        self,
        prefix: str,
        models_to_keep: int = 10,
        create_dir=False,
        ckpt_extra=None,
        linear_avaraging: bool = False,
    ):
        self.prefix = prefix
        self.models_to_keep = models_to_keep
        if create_dir:
            pathlib.Path(prefix).parent.mkdir(exist_ok=True, parents=True)
        self._last_loaded_model_meta = {}
        self._last_versioned_model: Optional[ModelStateDict] = None
        self._linear_averaging = linear_avaraging
        if self._linear_averaging:
            self._init_model_averaging()

    def _init_model_averaging(self) -> None:
        versions = self.get_all_versions()
        if versions:
            *_, (last_version, last_path) = versions
            logging.info("Initializing model averaging with %s %s", last_version, last_path)
            model = torch.load(last_path, map_location="cpu")["model"]
            self._last_versioned_model = model

    def get_all_versions(self) -> List[Tuple[ModelVersion, str]]:
        versions = []
        for path in glob.glob(f"{self.prefix}_*"):
            if path.endswith(".tmp"):
                continue
            try:
                idx = int(path.split("_")[-1])
            except ValueError:
                logging.error("Bad file: %s", path)
                continue
            versions.append((idx, path))
        return sorted(versions)

    def save(self, obj) -> None:
        versions = self.get_all_versions()
        if versions:
            new_id = versions[-1][0] + 1
        else:
            new_id = 0
        if self._linear_averaging:
            if new_id:
                assert self._last_versioned_model is not None
                obj["model"] = _average_linearly_model_states(
                    old_model=self._last_versioned_model,
                    new_model=obj["model"],
                    model_number=new_id + 1,
                )
            self._last_versioned_model = obj["model"]

        path = f"{self.prefix}_{new_id:08d}"
        torch.save(obj, path + ".tmp")
        os.rename(path + ".tmp", path)
        models_to_delete = (len(versions) + 1) - self.models_to_keep
        if models_to_delete > 0:
            for _, path in versions[:models_to_delete]:
                os.remove(path)

    def get_last_version(self) -> Tuple[ModelVersion, str]:
        """Get last checkpoint and its version. Blocks if no checkpoints found."""
        while True:
            versions = self.get_all_versions()
            if not versions:
                logging.info("Waiting for checkpoint to appear (%s*)...", self.prefix)
                time.sleep(5)
                continue
            return versions[-1]

    def save_state_dict(self, torch_module, args=None, **meta) -> None:
        """Helper function to save model state."""
        torch_module = getattr(torch_module, "module", torch_module)
        assert torch_module is not None
        state = {"model": torch_module.state_dict(), "meta": meta, "args": args}
        return self.save(state)

    def maybe_load_state_dict(
        self, torch_module: torch.nn.Module, last_version: Optional[ModelVersion]
    ) -> ModelVersion:
        """Load model state if needed and return latest model version."""
        version, path = self.get_last_version()
        if version != last_version:
            pickle = torch.load(path, map_location="cpu")
            torch_module.load_state_dict(pickle["model"])
            self._last_loaded_model_meta = pickle.get("meta", {})
        return version

    def maybe_load_meta_only(self, last_version: Optional[ModelVersion]) -> ModelVersion:
        """Load meta from model state and return latest model version."""
        version, path = self.get_last_version()
        if version != last_version:
            pickle = torch.load(path, map_location="cpu")
            self._last_loaded_model_meta = pickle.get("meta", {})
        return version

    def get_meta(self) -> Dict:
        return self._last_loaded_model_meta


def _average_linearly_model_states(
    old_model: ModelStateDict, new_model: ModelStateDict, model_number: int
) -> ModelStateDict:
    """Compute a new average model state given the previous models state.

    Assumes that:
      - the new_model is model_number's model that is being averages.
      - the weight if model number N is N, i.e., newer models have higher weights.
    """
    merged_state = {}
    # new_model_weight <- model_number / ((model_number + 1) * model_number / 2)
    new_model_weight = 2 / (model_number + 1)
    for k in old_model:
        merged_state[k] = (
            old_model[k].to(new_model[k].device) * (1 - new_model_weight)
            + new_model[k] * new_model_weight
        )
    return merged_state


class ValuePolicyCkptSyncer:
    """A holder for 2 separate syncers for policy and value models."""

    def __init__(
        self,
        prefix: str,
        models_to_keep: int = 10,
        create_dir=False,
        ckpt_extra=None,
        linear_average_policy: bool = False,
    ):
        prefix = prefix.strip(".")
        kwargs = dict(models_to_keep=models_to_keep, create_dir=create_dir, ckpt_extra=ckpt_extra)
        self.value = CkptSyncer(f"{prefix}.value", **kwargs)
        self.policy = CkptSyncer(
            f"{prefix}.policy", linear_avaraging=linear_average_policy, **kwargs
        )

    def items(self):
        return dict(value=self.value, policy=self.policy).items()


def build_searchbot_like_agent_with_syncs(
    agent_cfg: agents_cfgs.Agent,
    *,
    ckpt_sync_path: Optional[str],
    use_trained_policy: bool,
    use_trained_value: bool,
    device_id: Optional[int] = None,
    disable_exploit: bool = False,
) -> Tuple[Union[SearchBotAgent, BQRE1PAgent, TheBestAgent], Callable[[], Dict[str, Dict]]]:
    """Builds a SearchBot/BQRE using some of ckpts from the syncer and a reload function.

    Performs the following modifications on the CFR agent config before loading it:
        * model_path: use one from the syncer if use_trained_policy is set
        * value_model_path: use one from the syncer if use_trained_value is set
        * device: set to device_id if provided.
        * disable_exploit: Disables searchbot logic to exploit other agent (to allow
          train / eval to differ in whether they assume opponent policy is known).

    If ckpt_sync_path is None, loads the agent with default params. In this
    case use_trained_policy and use_trained_value must be False.

    If ckpt_sync_path is present, then any possible combination of use_trained_policy
    or use_trained_value is allowed. If use_trained_policy and use_trained_value
    are both false, then syncing will still happen for the Meta data about the
    checkpoint, but the value and policy will not be updated during training.

    Returns tuple of 2 elements:
        searchbot_agent: the agent.
        do_sync_fn: on call loads new weights from checkpoints into the agent
            and returns a dict: syncer -> meta.

    Meta is a dict with meta information about the ckeckpoint as provided by
    trainer during syncer.save_state_dict call.
    """
    # agent_proto refers to the editable version of the agent.
    # searchbot_proto is editable SearchBot message withing the proto.
    agent_proto: agents_pb2.Agent = agent_cfg.to_editable()
    searchbot_proto: Optional[agents_pb2.SearchBotAgent]
    best_proto: Optional[agents_pb2.TheBestAgent]
    the_proto: Union[agents_pb2.SearchBotAgent, agents_pb2.TheBestAgent]
    if agent_cfg.searchbot is not None:
        searchbot_proto = agent_proto.searchbot
        best_proto = None
        the_proto = searchbot_proto
    elif agent_cfg.bqre1p is not None:
        searchbot_proto = agent_proto.bqre1p.base_searchbot_cfg
        best_proto = None
        the_proto = searchbot_proto
    elif agent_cfg.best_agent is not None:
        searchbot_proto = None
        best_proto = agent_proto.best_agent
        the_proto = best_proto
    else:
        raise ValueError("Not supported agent type: %s" % agent_cfg.which_agent)
    del agent_cfg  # Now working with proto only.
    joined_ckpt_syncer = None
    if ckpt_sync_path is not None:
        logging.info("build_searchbot_agent_with_syncs: Waiting for ckpt syncers")
        joined_ckpt_syncer = ValuePolicyCkptSyncer(ckpt_sync_path)
        # If using trained policy and/or value, need to get their paths so that we
        # can construct a proper model.
        _, last_policy_path = joined_ckpt_syncer.policy.get_last_version()
        _, last_value_path = joined_ckpt_syncer.value.get_last_version()
        logging.info("build_searchbot_agent_with_syncs: Original agent_one cfg:\n%s", agent_proto)
        if searchbot_proto is not None:
            default_model_path = searchbot_proto.model_path
        else:
            assert best_proto is not None
            default_model_path = best_proto.conditional_policy_model_path
        default_value_model_path = the_proto.value_model_path or default_model_path
        assert (
            default_model_path or use_trained_policy
        ), "No default model and use_trained_policy=false"
        assert (
            default_value_model_path or use_trained_value
        ), "No default model and use_trained_value=false"
        if searchbot_proto is not None:
            searchbot_proto.model_path = (
                last_policy_path if use_trained_policy else default_model_path
            )
        else:
            assert best_proto is not None
            best_proto.conditional_policy_model_path = (
                last_policy_path if use_trained_policy else default_model_path
            )
        the_proto.value_model_path = (
            last_value_path if use_trained_value else default_value_model_path
        )
    else:
        logging.info("build_searchbot_agent_with_syncs: dummy call with no syncers")
        assert not use_trained_policy
        assert not use_trained_value
    if device_id is not None:
        the_proto.device = device_id
    if disable_exploit:
        assert (
            searchbot_proto is not None
        ), "disable_exploit only supported for searchbot-like agents"
        searchbot_proto.exploited_agent_power = ""
    del searchbot_proto  # searchbot_proto is just a pointer within agent_proto that we don't need anymore.
    del best_proto  # ditto
    del the_proto  # ditto
    agent_cfg = agent_proto.to_frozen()  # type: ignore
    del agent_proto

    logging.info(
        "build_searchbot_agent_with_syncs: The following agent_one cfg will be used:\n%s",
        agent_cfg,
    )
    skip_base_strategy_model_cache = True
    if agent_cfg.searchbot is not None:
        agent = SearchBotAgent(
            agent_cfg.searchbot, skip_base_strategy_model_cache=skip_base_strategy_model_cache
        )
    elif agent_cfg.bqre1p is not None:
        agent = BQRE1PAgent(
            agent_cfg.bqre1p, skip_base_strategy_model_cache=skip_base_strategy_model_cache
        )
    elif agent_cfg.best_agent is not None:
        agent = TheBestAgent(
            agent_cfg.best_agent, skip_base_strategy_model_cache=skip_base_strategy_model_cache
        )
    else:
        assert False

    sync_tuples = []
    versions = {}
    if use_trained_value:
        assert joined_ckpt_syncer
        sync_tuples.append(
            ("value", joined_ckpt_syncer.value, agent.base_strategy_model.value_model)
        )
        versions["value"] = None
    if use_trained_policy:
        assert joined_ckpt_syncer
        sync_tuples.append(("policy", joined_ckpt_syncer.policy, agent.base_strategy_model.model))
        versions["policy"] = None
    if joined_ckpt_syncer and not use_trained_policy and not use_trained_value:
        sync_tuples.append(("meta_only", joined_ckpt_syncer.policy, None))
        versions["meta_only"] = None

    def do_sync():
        metas = {}
        for name, syncer, model in sync_tuples:
            if model is None:
                versions[name] = syncer.maybe_load_meta_only(versions[name])
            else:
                versions[name] = syncer.maybe_load_state_dict(model, versions[name])
            metas[name] = syncer.get_meta()
        return metas

    return (agent, do_sync)
