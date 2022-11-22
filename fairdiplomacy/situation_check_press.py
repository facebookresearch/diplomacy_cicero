#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.models.consts import POWER2IDX
from fairdiplomacy.models.state_space import get_order_vocabulary
from fairdiplomacy.timestamp import Timestamp
from fairdiplomacy.models.state_space import get_order_vocabulary
from fairdiplomacy.typedefs import Action, Order, PowerPolicies, RolloutJointAction
from fairdiplomacy import pydipcc
from typing import Any, Dict, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
from fairdiplomacy.typedefs import Power
import regex as re


"""
TEST ELIGIBLE FOR USE IN TESTS
"""


def has_orders(orders, *expected_orders) -> bool:
    assert len(expected_orders) > 0, "Did you forget the r argument to has_orders?"
    failed_orders = [order for order in expected_orders if order not in orders]
    return not failed_orders


VALID_TEST_FUNCTIONS = {"has_orders": has_orders}

"""
UTILITY FUNCTIONS TO HELP WITH RUNNING TESTS
"""


class PseudoOrderTestValidatorError(Exception):
    pass


class PseudoOrderTestRuntimeError(Exception):
    pass


def validate_pseudoorder_annotation_test(
    test: str, game: pydipcc.Game, sender: Power, recipient: Optional[Power],
) -> Callable[[Dict[str, Any]], bool]:
    def _has_orders(orders, *expected_orders):
        assert len(expected_orders) > 0, "Did you forget the r argument to has_orders?"
        phase = game.current_short_phase
        if phase.endswith("M"):
            all_poss_orders_map = game.get_all_possible_orders()
            orderable_locs = game.get_orderable_locations()
            powers_locs = orderable_locs[sender]

            if recipient:
                powers_locs += orderable_locs[recipient]

            all_poss_orders = [
                orders for loc, orders in all_poss_orders_map.items() if loc in powers_locs
            ]
            all_poss_orders = [order for orders in all_poss_orders for order in orders]
        else:
            # In R/A phases we have future-phase pseudos, so it's hard to know what future
            # orders are possible
            all_poss_orders = get_order_vocabulary()

        for pseudoorder in expected_orders:
            assert pseudoorder in all_poss_orders, f"Found invalid order: {pseudoorder}"

        return has_orders(orders, *expected_orders)

    try:
        # check for syntactical correctness
        func = eval(
            "lambda r: " + test,
            {**VALID_TEST_FUNCTIONS, "has_orders": _has_orders, "__builtins__": {}},
        )
        func([])

        # to make sure we catch everything despite short circuiting
        for substr in re.findall(
            r"\L<VALID_TEST_FUNCTIONS>\([\w, \'\)\"\-]*?\)",
            test,
            VALID_TEST_FUNCTIONS=VALID_TEST_FUNCTIONS,
        ):
            subfunc = eval(
                "lambda r: " + substr,
                {**VALID_TEST_FUNCTIONS, "has_orders": _has_orders, "__builtins__": {}},
            )
            subfunc([])

        return func
    except Exception as e:
        raise PseudoOrderTestValidatorError(
            f"Unsuccessful validation error: {e} (for test: {test})"
        )


def run_pseudoorder_annotation_test(test: str, pseudoorders: List[Order]) -> bool:
    try:
        func = eval("lambda r: " + test, {**VALID_TEST_FUNCTIONS, "__builtins__": {}})
        return func(pseudoorders)
    except Exception as e:
        raise PseudoOrderTestRuntimeError(
            f"Unsuccessful execution error: {e} (for test {test} on orders {pseudoorders}"
        )
