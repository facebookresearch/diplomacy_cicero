#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
import logging
from typing import Callable, Optional

import heyhi


class ShouldStopException(Exception):
    pass


InterruptCheck = Callable[[bool], bool]
_INTERRUPTION_CONDITION: Optional[InterruptCheck] = None


@contextlib.contextmanager
def set_interruption_condition(handler: InterruptCheck):
    """Set a condition to check if the agent should stop early."""
    global _INTERRUPTION_CONDITION
    if heyhi.is_aws():
        # We don't have redis there.
        logging.warning("Ignoring set_interruption_condition as running on AWS")
        yield
    else:
        _INTERRUPTION_CONDITION = handler
        try:
            yield
        finally:
            _INTERRUPTION_CONDITION = None


def raise_if_should_stop(post_pseudoorders: bool):
    """Raises ShouldStopException if the agent should stop now. Othewise noop.

    Arguments:
    post_pseudoorders: True if message generation has finished computing pseudo-orders.
    """
    global _INTERRUPTION_CONDITION
    if _INTERRUPTION_CONDITION is not None and _INTERRUPTION_CONDITION(post_pseudoorders):
        logging.warning("Interruption condition was triggered!")
        raise ShouldStopException
