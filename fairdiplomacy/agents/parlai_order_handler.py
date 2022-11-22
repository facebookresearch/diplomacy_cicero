#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import List, Sequence, Tuple, Dict

from fairdiplomacy.agents.order_handler import OrderHandler
from fairdiplomacy.typedefs import Action, JointAction, Order, Power
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY_TO_IDX, strip_coasts
from parlai_diplomacy.wrappers.orders import BaseOrderWrapper


def get_all_possible_orders_by_power(game) -> Dict[Power, List[Order]]:
    all_possible_orders = game.get_all_possible_orders()
    all_orderable_locs = game.get_orderable_locations()
    return {
        power: [
            order
            for loc in all_orderable_locs.get(power, [])
            for order in all_possible_orders.get(loc, [])
        ]
        for power in POWERS
    }


def filter_orders(
    predicted_orders: Sequence[Order], allowed_orders: Sequence[Order]
) -> Tuple[List[Order], List[Order]]:
    assert not isinstance(predicted_orders, dict)
    allowed_orders_set = frozenset(allowed_orders)
    good_orders, bad_orders = [], []
    for order in predicted_orders:
        # In some cases, parlai will predict an equivalent order with different coasts than
        # base_strategy_model, for example F MAO S F SPA versus F MAO S F SPA/NC (when there is a fleet
        # on SPA/NC). Then when base_strategy_model produces the other format, we'll end up with a
        # policy with both of them. Next, when parlai rescores the policy, because
        # both orders get canonicalized to the format that parlai expects, parlai will
        # report an equally high score for *both* orders, which means that overall that
        # order will get twice as much mass as it deserves because each formatted copy
        # of the order gets a full copy of the probability mass that it should have.

        # This isn't great, so we attempt here to make sure any orders that parlai
        # outputs get canonicalized to the ORDER_VOCABULARY that base_strategy_model uses.
        if order in allowed_orders_set:
            if order not in ORDER_VOCABULARY_TO_IDX:
                stripped = strip_coasts(order)
                other_matches = []
                for other_allowed_order in allowed_orders_set:
                    if (
                        other_allowed_order in ORDER_VOCABULARY_TO_IDX
                        and strip_coasts(other_allowed_order) == stripped
                    ):
                        other_matches.append(other_allowed_order)
                # There is a unique alternative order that is allowed and that is recognized by base_strategy_model
                # and whose coasts are similar the predicted order - so use that one instead.
                if len(other_matches) == 1:
                    order = other_matches[0]
            good_orders.append(order)
        else:
            bad_orders.append(order)

    return good_orders, bad_orders


def filter_orders_many_powers(
    predicted_orders_by_powers: JointAction,
    allowed_orders_by_powers: Dict[Power, List[Order]],
    subset_ok: bool = False,
) -> Tuple[Dict[Power, List[Order]], Dict[Power, List[Order]]]:
    """
    Check that the prediction joint action contains legal orders.

    - predicted_orders_by_powers: predicted joint action
    - allowed_orders_by_powers: legal orders for each power
    - subset_ok: whether we should expect orders for all powers or a subset
        (E.g. for pseudo orders, we don't always need actions for all powers)
    """
    good_orders, bad_orders = {}, {}
    if not subset_ok:
        assert set(predicted_orders_by_powers.keys()) == set(allowed_orders_by_powers.keys())
    else:
        assert set(predicted_orders_by_powers.keys()).issubset(
            set(allowed_orders_by_powers.keys())
        )
    for power in good_orders:
        good_orders[power], bad_orders[power] = filter_orders(
            predicted_orders_by_powers[power], allowed_orders_by_powers[power]
        )
    return good_orders, bad_orders


class ParlaiOrderHandler(OrderHandler):
    MAX_ATTEMPTS_GEN_VALID = 16  # Max # of attempts to generate a valid order

    def __init__(self, model: BaseOrderWrapper):
        self._model = model

    def _get_possible_orders(self, game) -> Dict[Power, List[Order]]:
        return get_all_possible_orders_by_power(game)

    def _get_orders_single_power(
        self, game, power: Power, possible_orders: Sequence[str]
    ) -> Action:
        """
        Produce orders for a single power, provided the game object.

        We attempt to produce a valid order up to `self.MAX_ATTEMPTS_GEN_VALID`
        times, before returning the last generated partially valid order.
        """
        if not possible_orders:
            return tuple()

        good_orders = []
        for i in range(self.MAX_ATTEMPTS_GEN_VALID):
            # attempt to generate a valid order
            orders = self._model.produce_action(game, power)
            # determine if the orders are valid
            good_orders, bad_orders = filter_orders(orders, possible_orders)
            if not bad_orders:
                if i > 0:
                    # log that we needed multiple tries to produce a valid order
                    logging.warning(
                        f"ParlAI orders model took {i + 1} attempts to produce a valid order"
                    )
                # found a good set of orders, return
                return orders
            else:
                # log the invalid orders produced:
                logging.warning(f"Bad orders produced: {bad_orders}, trying again...")

        # if we reached this point, we did not generate a valid set of orders within
        # the range of MAX_ATTEMPTS_GEN_VALID
        #
        # we fall back to returning the last partially valid set of orders
        logging.warning(
            f"ParlAI model did not produce a valid order in {self.MAX_ATTEMPTS_GEN_VALID} attempts. "
            "Returning a partially valid order."
        )
        return tuple(good_orders)

    def get_orders(self, game, power) -> Action:
        """
        Produce orders for a power.
        """
        possible_orders = self._get_possible_orders(game)[power]
        return self._get_orders_single_power(game, power, possible_orders)

    def get_orders_many_powers(self, game, powers: Sequence[Power]) -> JointAction:
        """
        Generate a valid action (order set) for each power.

        For each p in powers:

        First, compute a batch of N actions (with scores) for p, remove all
        the ones with invalid orders, and sample from the implied probability
        distribution.

        If the batch of N actions doesn't find any that are fully valid,
        then generate 1 more action, simply remove the invalid orders in it,
        and return what remains.
        """
        power_orders = {}
        all_possible_orders = self._get_possible_orders(game)

        for power in powers:
            power_orders[power] = self._get_orders_single_power(
                game, power, all_possible_orders[power]
            )

        return power_orders
