#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Utils for loading agents and tasks, etc.
"""


def register_all_agents():
    # list all agents here
    import parlai_diplomacy.agents.integration_tests.agents  # noqa: F401
    import parlai_diplomacy.agents.generator_custom_inference.agents  # noqa: F401
    import parlai_diplomacy.agents.marginal_likelihood.agents  # noqa: F401
    import parlai_diplomacy.agents.bart_classifier.agents  # noqa: F401
    import parlai_diplomacy.agents.prefix_generation.agents  # noqa: F401


def register_all_tasks():
    # list all tasks here
    import parlai_diplomacy.tasks.dialogue.regular_order_agents  # noqa: F401
    import parlai_diplomacy.tasks.dialogue.pseudo_order_agents  # noqa: F401
    import parlai_diplomacy.tasks.sleep_classifier.agents  # noqa: F401
    import parlai_diplomacy.tasks.order.single_order_agents  # noqa: F401
    import parlai_diplomacy.tasks.order.single_order_rollout_agents  # noqa: F401
    import parlai_diplomacy.tasks.order.all_orders_agents  # noqa: F401
    import parlai_diplomacy.tasks.order.all_orders_rollout_agents  # noqa: F401
    import parlai_diplomacy.tasks.order.all_orders_independent_agents  # noqa: F401
    import parlai_diplomacy.tasks.order.plausible_pseudo_orders_agents  # noqa: F401
    import parlai_diplomacy.tasks.base_diplomacy_agent  # noqa: F401
    import parlai_diplomacy.tasks.discriminator.agents  # noqa: F401
    import parlai_diplomacy.tasks.lie_detector.agents  # noqa: F401
    import parlai_diplomacy.tasks.recipient_classifier.agents  # noqa: F401
    import parlai_diplomacy.tasks.draw_classifier.agents  # noqa: F401
    import parlai_diplomacy.tasks.denoising.agents  # noqa: F401
