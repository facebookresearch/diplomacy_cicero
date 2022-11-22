#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import heyhi
import conf.agents_cfgs


def build_agent_from_cfg(agent_stanza: "conf.agents_cfgs.Agent", **redefines) -> "BaseAgent":
    from .base_agent import BaseAgent
    from .br_search_agent import BRSearchAgent
    from .searchbot_agent import SearchBotAgent
    from .base_strategy_model_agent import BaseStrategyModelAgent
    from .parlai_full_press_agent import ParlaiFullPressAgent
    from .parlai_no_press_agent import ParlaiNoPressAgent
    from .random_agent import RandomAgent
    from .repro_agent import ReproAgent
    from .bqre1p_agent import BQRE1PAgent
    from .the_best_agent import TheBestAgent

    AGENT_CLASSES = {
        "br_search": BRSearchAgent,
        "base_strategy_model": BaseStrategyModelAgent,
        "searchbot": SearchBotAgent,
        "bqre1p": BQRE1PAgent,
        "repro": ReproAgent,
        "random": RandomAgent,
        "parlai": ParlaiNoPressAgent,
        "parlai_full_press": ParlaiFullPressAgent,
        "best_agent": TheBestAgent,
    }

    which_agent = agent_stanza.which_agent
    assert which_agent is not None, f"Config must define an agent type: {agent_stanza}"
    agent_cfg = agent_stanza.agent
    assert agent_cfg is not None

    if redefines:
        agent_cfg = agent_cfg.to_editable()
        # handle redefines
        for k, v in redefines.items():
            heyhi.conf_set(agent_cfg, k, v)
        agent_cfg = agent_cfg.to_frozen()

    return AGENT_CLASSES[which_agent](agent_cfg)
