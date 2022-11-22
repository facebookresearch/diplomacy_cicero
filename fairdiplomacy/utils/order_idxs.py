#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Helpful functions for converting:
- order idxs to/from order strings
- global idxs to/from local idxs
"""
import logging
from fairdiplomacy.typedefs import Action, JointAction, Order, Power, PowerPolicies, Policy
import torch
from typing import List, Sequence, Optional

from fairdiplomacy.models.consts import LOCS
from fairdiplomacy.models.state_space import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
    EOS_IDX,
)
from fairdiplomacy.pydipcc import Game

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}
MAX_VALID_LEN = get_order_vocabulary_idxs_len()


class OrderIdxConversionException(Exception):
    """May be raised by functions to indicate that no valid conversion can be made"""

    pass


_LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCS)}

LOC_IDX_OF_ORDER_IDX = [_LOC_TO_IDX[order.split()[1]] for order in ORDER_VOCABULARY]


def canonicalize_action(action: Action) -> Action:
    """
    Convert an Action to its canonical ordering.

    We encode actions as a tuple of order strings in a canonical ordering.
    This ordering follows the ordering of locations in LOCS.
    However, there are a couple places this convention is not followed:

    (1) In old game.json files before this convention was adopted
    (2) In parlai, where orders are sorted lexicographically

    This function converts it to the canonical ordering.

    If the order strings are not valid, this function only guarantees that
    the return value will be a deterministic function of the *set* of
    input orders (i.e. not dependent on the ordering)
    """
    try:
        return tuple(sorted(action, key=lambda o: _LOC_TO_IDX[o.split()[1]]))
    except (IndexError, KeyError):
        # If orders are not valid, fall back to lexicographic sort
        return tuple(sorted(action))


##########################
##  STRING CONVERSIONS  ##
##########################


def global_order_idxs_to_str(order_idxs) -> List[str]:
    """Convert a sequence of global order idxs to a list of order strings

    N.B. returns normal uncombined build orders that are readable by a Game
    object.
    """
    orders = []
    for idx in order_idxs:
        if idx == EOS_IDX:
            continue
        orders.extend(ORDER_VOCABULARY[idx].split(";"))
    orders.sort(key=lambda order: _LOC_TO_IDX[order.split()[1]])
    return orders


def is_single_build_order(order: Optional[str]) -> bool:
    if order is None:
        return False
    pieces = order.split()
    if len(pieces) <= 2:
        return False
    if pieces[2] == "B" and ";" not in order:
        return True
    return False


def action_strs_to_global_idxs(
    orders: Sequence[Optional[str]],
    try_strip_coasts: bool = False,
    try_vias: bool = True,
    ignore_missing: bool = False,
    return_none_for_missing: bool = False,
    return_hold_for_invalid: bool = False,
    sort_by_loc: bool = False,
    sort_by_idx: bool = False,
    match_to_possible_orders: Optional[Sequence[Order]] = None,
) -> List[int]:
    """Convert a list of order strings to a list of global idxs in ORDER_VOCABULARY

    Args:
    - orders: a list of combined or uncombined order strings
    - try_strip_coasts: if True and order is not in vocabulary, strip coasts from all
        locs and try again
    - try_vias: if True and order is not in vocabulary but it could be an army
        attempting a convoy, add a "VIA" to it and try again.
    - ignore_missing: if True, return idxs only for orders found in vocab. If False and
        and an order is not found, raise OrderIdxConversionException
    - return_none_for_missing: if True and an order conversion is not found, return None
        for that order and do not raise OrderIdxConversionException
    - return_hold_for_invalid: if True and an order conversion is not found, but the
       unit specified is still well-formed, treat it as a hold instead.
    - sort_by_loc: if True, return order idxs sorted by the actor's location.
        This matches dipcc's encoding convention, so as long as every unit is given
        an action (no orders are missing or omitted), this produces the correct ordering
        ordering of indices for base_strategy_model training/inference.
    - sort_by_idx: if True, return sorted order idxs
    - match_to_possible_orders: if set and an order is not found, look for a
        compatible coastal variant in the provided dict. e.g. "F MAR S F SPA" is not
        in the vocab, but the more specific "F MAR S F SPA/SC" or "F MAR S F SPA/NC"
        may be possible.

    N.B. accepts combined or uncombined order strings
    """
    if type(orders) == str:
        raise ValueError(f"orders must be a sequence of strings, got {orders} of type str")
    if sort_by_loc and sort_by_idx:
        raise ValueError("can't set both sort_by_loc and sort_by_idx")

    # combine build orders if necessary
    if any(is_single_build_order(x) for x in orders):
        assert all(
            is_single_build_order(x) for x in orders
        ), "Cannot combine a mix of build and non-build orders"
        orders = [";".join(sorted(orders))]  # type:ignore

    order_idxs = []
    order_loc_idxs = []
    for order in orders:
        if order is None:
            order_idx = None
        else:
            order_idx = ORDER_VOCABULARY_TO_IDX.get(order, None)
            if order_idx is None and try_strip_coasts:
                order_idx = ORDER_VOCABULARY_TO_IDX.get(strip_coasts(order), None)

            if order_idx is None and try_vias:
                order_with_via = add_via(order)
                order_idx = ORDER_VOCABULARY_TO_IDX.get(order_with_via, None)
                if order_idx is None and try_strip_coasts:
                    order_idx = ORDER_VOCABULARY_TO_IDX.get(strip_coasts(order_with_via), None)

            if order_idx is None and return_hold_for_invalid:
                holdified_order = convert_to_hold(order)
                order_idx = ORDER_VOCABULARY_TO_IDX.get(holdified_order, None)
                if order_idx is None and try_strip_coasts:
                    order_idx = ORDER_VOCABULARY_TO_IDX.get(strip_coasts(holdified_order), None)

        if order_idx is None:
            if return_none_for_missing:
                pass
            elif ignore_missing:
                continue
            elif match_to_possible_orders:
                assert order is not None, "match_to_possible_orders not supported with None orders"
                # look for an order that is possible, in the vocab, and is a
                # variant of the input order
                stripped = strip_coasts(order)
                for possible_match in match_to_possible_orders:
                    if (
                        possible_match in ORDER_VOCABULARY_TO_IDX
                        and strip_coasts(possible_match) == stripped
                    ):
                        order_idx = ORDER_VOCABULARY_TO_IDX[possible_match]
                        break
                else:
                    raise OrderIdxConversionException(order)
            else:
                raise OrderIdxConversionException(order)
        order_idxs.append(order_idx)
        if sort_by_loc:
            # This is illegal because we have no way of determining what the proper ordering of the
            # orders should be - where the Nones "should" be sorted.
            # We can't simply sort all the Nones to the beginning or end because there are some
            # places in our code where we rely on action_strs_to_global_idxs to give us a fully
            # ordered list of orders compatible with dipcc's action sequencing (i.e. sorted by coast-qualified location)
            # so putting the Nones in an arbitrary place would break things.
            assert (
                order is not None
            ), "action_strs_to_global_idxs: Nones provided in the input orders when sort_by_loc=True"
            pieces = order.split()
            assert (
                len(pieces) > 1 and pieces[1] in _LOC_TO_IDX
            ), "action_strs_to_global_idxs: Invalid orders and/or locations when sort_by_loc=True"
            order_loc_idxs.append(_LOC_TO_IDX[pieces[1]])

    if sort_by_loc:
        order_idxs = [idx for loc, idx in sorted(zip(order_loc_idxs, order_idxs))]
    if sort_by_idx:
        order_idxs.sort()

    return order_idxs


################################
##  GLOBAL/LOCAL CONVERSIONS  ##
################################


def local_order_idxs_to_global(
    local_idxs: torch.Tensor, x_possible_actions: torch.Tensor, clamp_and_mask: bool = True
) -> torch.Tensor:
    """Convert local order indices to global order indices.

    Args:
        local_indices: Long tensor [*, 7, S] of indices [0,469) of x_possible_actions
        x_possible_actions: Long tensor [*, 7, S, 469]. Containing indices in
            ORDER_VOCABULARY.
        clamp_and_mask: if True, handle EOS_IDX inputs and propagate them to outputs. Set
            False to skip as a speed optimization if not necessary.

    Returns:
        global_indices: Long tensor of the same shape as local_indices such that
            x_possible_actions[b, p, s, local_indices[b, p, s]] = global_indices[b, p, s]
    """
    assert (
        EOS_IDX == -1
    ), "the clamp_and_mask path is necessary because EOS_IDX is negative. Is that still true?"

    mask = None
    if clamp_and_mask:
        mask = local_idxs == EOS_IDX
        local_idxs = local_idxs.clamp(0)

    global_idxs = torch.gather(
        x_possible_actions, local_idxs.ndim, local_idxs.unsqueeze(-1)
    ).view_as(local_idxs)

    if clamp_and_mask:
        global_idxs[mask] = EOS_IDX

    return global_idxs


def global_order_idxs_to_local(
    global_indices: torch.Tensor, x_possible_actions: torch.Tensor, *, ignore_missing=False
) -> torch.Tensor:
    """Convert global order indices to local order indices.

    Args:
        global_indices: Long tensor [B, 7, S] of indices in ORDER_VOCABULARY
        x_possible_actions: Long tensor [B, 7, S, 469]. Containing indices in
            ORDER_VOCABULARY.
        ignore_missing: if False and an element of global_indices is not in x_possible_actions,
            raise OrderIdxConversionException. If True, set return to -1.

    Returns:
        local_indices: Long tensor of the same shape as global_indices such that
            x_possible_actions[b, p, s, local_indices[b, p, s]] = global_indices[b, p, s]
    """
    onehots = x_possible_actions == global_indices.unsqueeze(-1)
    local_indices = onehots.max(-1).indices
    local_indices[global_indices == EOS_IDX] = EOS_IDX
    missing = ~onehots.any(dim=-1)
    if missing.any():
        if ignore_missing:
            local_indices[missing] = EOS_IDX
        else:
            raise OrderIdxConversionException()
    return local_indices


############
##  MISC  ##
############


def convert_to_hold(order: Order) -> Order:
    """Attempt to convert order to hold, leaving it unchanged if not possible"""
    pieces = order.split(" ")
    # Too-short to be a valid order, return it unchanged
    if len(pieces) <= 2:
        return order
    return " ".join(pieces[:2]) + " H"


def strip_coasts(order: Order) -> Order:
    """Return order with all locations stripped of their coast suffixes"""
    for suffix in ["/NC", "/EC", "/SC", "/WC"]:
        order = order.replace(suffix, "")
    return order


def add_via(order: Order) -> Order:
    """Attempt to add VIA to army movement, leaving it unchanged if not applicable"""
    if not order.startswith("A "):
        return order
    pieces = order.split(" ")
    if len(pieces) != 4 or pieces[2] != "-":
        return order
    return order + " VIA"


def loc_idx_of_order_idx(order_idx: int) -> int:
    """Return the location index of the source location of the order.
    For combined builds, the location index of the first build in the list."""
    return LOC_IDX_OF_ORDER_IDX[order_idx]


def filter_out_of_vocab_orders(power_policies: PowerPolicies):
    """Remove (parlai generated) actions containing orders not in the vocab of base_strategy_model."""
    filtered_power_policies: PowerPolicies = {}

    for power, policy in power_policies.items():
        filtered_policy: Policy = {}
        for action, prob in policy.items():
            keep = True
            for order in action:
                if order not in ORDER_VOCABULARY_TO_IDX:
                    keep = False
                    break
            if keep:
                filtered_policy[action] = prob

        filtered_power_policies[power] = filtered_policy
    return filtered_power_policies


def num_build_orders(a: Action):
    return len([o for o in a if o.endswith(" B")])


def num_destroy_orders(a: Action):
    return len([o for o in a if o.endswith(" D")])


def is_valid_build_or_destroy(a: Action, num_builds: int):
    if num_builds == 0:
        return num_build_orders(a) == 0 and num_destroy_orders(a) == 0
    elif num_builds > 0:
        return num_build_orders(a) <= num_builds and num_destroy_orders(a) == 0
    else:
        return num_build_orders(a) == 0 and num_destroy_orders(a) == -num_builds


def is_action_valid(game: Game, power: Power, action: Action):
    """Check if all actions in the joint action is possible

    It checks:
      - if action contains the right amount of orders
      - if each order is possible given the curent game state
      - if each order is in the ORDER_VOCABULARY
      - if action contains duplications
    """
    game_state = game.get_state()
    orderable_locations = game.get_orderable_locations()
    all_possible_orders = game.get_all_possible_orders()

    possible_orders = []
    for loc in orderable_locations[power]:
        for order in all_possible_orders[loc]:
            possible_orders.append(order)

    if game.phase.endswith("RETREATS") or game.phase.endswith("MOVEMENT"):
        if len(action) != len(orderable_locations[power]):
            logging.info(
                f"Invalid action {action}: wrong number of orders {len(action)} vs {len(orderable_locations[power])}(required)"
            )
            return False
    else:
        # adjustment phase
        num_builds = game_state["builds"][power]["count"]
        if not is_valid_build_or_destroy(action, num_builds):
            logging.info(
                f"Invalid action {action}: wrong number of build or destroy, num_build={num_builds}"
            )
            return False

    for order in action:
        if order not in ORDER_VOCABULARY_TO_IDX:
            logging.info(f"Invalid action {action}: {order} is not in order vocab")
            return False
        if order not in possible_orders:
            logging.info(f"Invalid action {action}: {order} is not in possible orders")
            return False

    # check if it contains duplications
    if len(set(action)) != len(action):
        logging.info(f"Invalid action {action}: duplication")
        return False

    return True
