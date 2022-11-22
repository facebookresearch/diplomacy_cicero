#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Different ways to generate actions."""
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple
import collections
import logging
import itertools
import math
import random

import numpy as np
import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_strategy_model_wrapper import (
    BaseStrategyModelWrapper,
    forward_model_with_output_transform,
)
from fairdiplomacy.agents.plausible_order_sampling import are_supports_coordinated
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import shuffle_locations
from fairdiplomacy.models.consts import LOCS
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.models.base_strategy_model import load_model
from fairdiplomacy.models.base_strategy_model.base_strategy_model import NO_ORDER_ID, EOS_IDX
from fairdiplomacy.utils.batching import batched_forward
from fairdiplomacy.utils.order_idxs import ORDER_VOCABULARY, ORDER_VOCABULARY_TO_IDX
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.typedefs import Power, Action, PowerPolicies
import nest

GEN_BATCH_SIZE = 1024


def _make_adjacencies():
    adjacencies = collections.defaultdict(list)
    for loc in LOCS:
        for loc2 in LOCS:
            if f"A {loc} - {loc2}" in ORDER_VOCABULARY or f"F {loc} - {loc2}" in ORDER_VOCABULARY:
                adjacencies[loc].append(loc2)
    return adjacencies


ADJ_MAPPING = {
    k.upper(): sorted(set(x.split("/")[0].upper() for x in v))
    for k, v in _make_adjacencies().items()
    if "/" not in k
}


@torch.no_grad()
def generate_orders_from_base_strategy_model(
    model,
    game,
    *,
    selected_power: Power,
    agent_power: Optional[Power],
    max_actions,
    num_threads=10,
    temperature=1.0,
    top_p=1.0,
    location_order,
) -> Sequence[Action]:
    del num_threads  # Not used.

    feature_encoder = FeatureEncoder()
    power_index = POWERS.index(selected_power)
    observations = feature_encoder.encode_inputs([game], input_version=model.get_input_version())

    half_precision = next(iter(model.parameters())).dtype is torch.float16

    def fwd(batch_indices):
        x_power = None
        x = DataFields(
            nest.map(
                lambda x: x.expand(len(batch_indices), *([-1] * len(x.shape[1:]))).to(device),
                observations,
            )
        )
        x = BaseStrategyModelWrapper.add_stuff_to_datafields(
            x, has_press=False, agent_power=agent_power
        )
        if half_precision:
            x = x.to_half_precision()
        if location_order == "default":
            pass
        elif location_order == "shuffle":
            x = shuffle_locations(x)
        else:
            raise ValueError(f"Unknown location_order: {location_order}")
        order_indices, action_logprobs, _ = forward_model_with_output_transform(
            model, x, temperature=temperature, top_p=top_p, x_power=x_power, need_value=False
        )
        return order_indices, action_logprobs

    device = next(iter(model.parameters())).device
    order_indices, action_logprobs = batched_forward(
        fwd, torch.ones(max_actions), batch_size=GEN_BATCH_SIZE, device=device
    )

    all_power_actions = feature_encoder.decode_order_idxs(order_indices.cpu())
    actions = [tuple(a[power_index]) for a in all_power_actions]
    scored_actions = collections.Counter(
        dict(zip(actions, action_logprobs[:, power_index].tolist()))
    )
    actions, _ = zip(*scored_actions.most_common())
    return actions


@torch.no_grad()
def generate_order_by_column_from_base_strategy_model(
    model,
    game,
    *,
    selected_power: Power,
    agent_power: Optional[Power],
    num_threads=10,
    temperature=1.0,
    top_p=1.0,
    max_actions=None,
) -> List[Action]:
    """Produce action set where each unit tries all possible orders."""
    power_index = POWERS.index(selected_power)
    feature_encoder = FeatureEncoder(num_threads=num_threads)
    observations = feature_encoder.encode_inputs([game], input_version=model.get_input_version())
    MAX_ORDERS_PER_POSITION = 469
    observations = nest.map(
        lambda x: x.expand(MAX_ORDERS_PER_POSITION, *([-1] * len(x.shape[1:]))).clone(),
        observations,
    )

    assert game.get_current_phase().endswith("M"), "Don't use it for non-move phases"
    collected_actions: List[Action] = []
    # Shape: [MAX_SEQ_LEN, MAX_ORDERS_PER_POSITION].
    power_possible_actions = observations["x_possible_actions"][0, power_index]

    if (power_possible_actions == EOS_IDX).all():
        return []

    expected_num_actions = int_prod(
        [max((mask != EOS_IDX).sum(), 1) for mask in power_possible_actions]
    )

    if max_actions is None:
        num_rounds = 1
    else:
        num_rounds = math.ceil(max_actions / expected_num_actions)

    device = next(iter(model.parameters())).device
    for location in range(len(power_possible_actions)):
        num_possible_orders = (power_possible_actions[location] != EOS_IDX).sum()
        if not num_possible_orders:
            continue

        loc_teacher_orders = torch.full(
            (num_possible_orders, len(POWERS), len(power_possible_actions)),
            NO_ORDER_ID,
            dtype=torch.long,
        )
        loc_teacher_orders[:, power_index, location] = power_possible_actions[
            location, :num_possible_orders
        ]
        assert (power_possible_actions[location, :num_possible_orders] >= 0).all()
        loc_obserbations = nest.map(lambda x: x[:num_possible_orders], observations)
        loc_obserbations["teacher_force_orders"] = loc_teacher_orders
        loc_obserbations = BaseStrategyModelWrapper.add_stuff_to_datafields(
            loc_obserbations, has_press=False, agent_power=agent_power
        )

        for _ in range(num_rounds):
            order_indices, _, _ = batched_forward(
                lambda x: forward_model_with_output_transform(
                    model, x, temperature=temperature, top_p=top_p
                ),
                loc_obserbations,
                batch_size=GEN_BATCH_SIZE,
                device=device,
            )

            order_indices[:, power_index, location] = power_possible_actions[
                location, :num_possible_orders
            ]

            all_power_actions = feature_encoder.decode_order_idxs(order_indices.cpu())
            actions = [tuple(a[power_index]) for a in all_power_actions]
            collected_actions.extend(actions)
    non_unique_size = len(collected_actions)
    collected_actions = list({x: 1 for x in collected_actions})
    logging.info(
        "Finished column generation: num_conditions=%s num_rounds=%s row_action_count=%s unique_action_count=%s",
        expected_num_actions,
        num_rounds,
        non_unique_size,
        len(collected_actions),
    )
    return collected_actions


def yield_samples_from_product(*items):
    while True:
        yield tuple(random.choice(i) for i in items)


def int_prod(seq: Sequence[int]) -> int:
    return 1 if not seq else seq[0] * int_prod(seq[1:])


def get_all_possible_orders(game, power, max_actions=0):
    per_loc = get_power_per_loc_orders(game, power)
    prod_size = int_prod([len(x) for x in per_loc.values()])
    search_size = 10 * max_actions
    logging.info(
        "About to generate order product for %s: %s",
        power,
        " ".join(
            [
                *["%s=%s" % (loc, len(orders)) for loc, orders in per_loc.items()],
                "total_prod=%s" % prod_size,
                "max_actions=%s" % max_actions,
                "search_size=%s" % search_size,
            ]
        ),
    )
    if not max_actions or prod_size <= search_size:
        logging.info("Generating full product")
        prod = itertools.product(*per_loc.values())
    else:
        logging.info("Generating sampled product")
        prod = itertools.islice(yield_samples_from_product(*per_loc.values()), search_size)
    prod = list(prod)
    search_prod_size = len(prod)
    prod = sorted(frozenset(prod))
    random.shuffle(prod)
    coord_prod = (x for x in prod if are_supports_coordinated(x))
    if max_actions:
        coord_prod = itertools.islice(coord_prod, max_actions)
    coord_prod = sorted(coord_prod)
    logging.info(
        "Generated all orders for %s: %s",
        power,
        " ".join(
            [
                "total_prod=%s" % prod_size,
                "search_prod=%s" % search_prod_size,
                "search_prod_uniq=%s" % len(prod),
                "coord_prod=%s" % len(coord_prod),
            ]
        ),
    )
    return coord_prod


def numpy_product(first: np.ndarray, *other: np.ndarray) -> np.ndarray:
    if not other:
        return np.array(first).reshape((-1, 1))
    subblock = numpy_product(*other)
    blocks = []
    for i in first:
        blocks.append(np.concatenate([np.full((len(subblock), 1), i), subblock], 1))
    return np.concatenate(blocks, 0)


def make_supports_coordinated(orders: List[str]) -> List[str]:
    """Replace all obviously bad supports/convoys with holds."""
    # This is the most srupud implmentation ever.
    orders = list(orders)
    while not are_supports_coordinated(orders):
        for i, order in enumerate(orders):
            split = order.split()
            if split[2] in ("S", "C"):
                orders[i] = "%s %s H" % (split[0], split[1])
                break
        else:
            raise RuntimeError("No supports/convoy and still not coordinated!")
    return orders


def filter_uncoordinated_int_loc_dicts(per_loc: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Given dict of orders per location remove ones that cannot coordinate."""
    per_loc = per_loc.copy()
    good_orders = frozenset(itertools.chain(*per_loc.values()))

    need_another_pass = True
    while need_another_pass:
        need_another_pass = False
        for loc in list(per_loc):
            if len(per_loc[loc]) == 1:
                [the_order] = per_loc[loc]
                old_len = len(good_orders)
                good_orders = frozenset(
                    x for x in good_orders if are_supports_coordinated([x, the_order])
                )
                new_len = len(good_orders)
                if old_len != new_len:
                    # logging.debug(
                    #     "Reduced allowed orders from %s to %s using %s",
                    #     old_len,
                    #     new_len,
                    #     the_order,
                    # )
                    need_another_pass = True
            else:
                old_len = len(per_loc[loc])
                filtered = [x for x in per_loc[loc] if x in good_orders]
                if not filtered:
                    # logging.error("Got none available order for location %s", loc)
                    # Silienty skip filtering for this location.
                    continue
                per_loc[loc] = filtered
                if old_len != len(per_loc[loc]):
                    # logging.debug(
                    #     "Reduced allowed orders for %s from %s to %s",
                    #     loc,
                    #     old_len,
                    #     len(per_loc[loc]),
                    # )
                    need_another_pass = True
    return per_loc


def get_power_per_loc_orders(game, power: Power) -> Dict[str, List[str]]:
    """Get a list of possible orders for each unit of a power.

    The locations in the returned dict follow order in LOCS. Only orders in
    ORDER_VOCABULARY_TO_IDX are returned.

    The keys in returned dict are root locations but sorted by coastal locations.
    """
    # For coastal locations this dict will contain both the root location
    # and the coastal location.
    loc2orders = game.get_all_possible_orders()

    # Iterate over location in get_orderable_locations will order consistent
    # with x_possible_actions.
    result = {}
    for loc in game.get_orderable_locations()[power]:
        # get_all_possible_orders returns some things that are not in the vocab
        # e.g., it returns both "A POR S F SPA/SC" and "A POR S F SPA",
        # but only the former is in ORDER_VOCABULARY_TO_IDX.
        result[loc] = [x for x in loc2orders[loc] if x in ORDER_VOCABULARY_TO_IDX]
    return result


def produce_local_loc_dicts(
    game,
    power: str,
    base_action: List[str],
    close_locations: FrozenSet[str],
    *,
    fix_uncoordinated_base: bool,
) -> Dict[str, List[str]]:
    """Given an action for a power and a location build a loc->possible_orders dict."""
    # logging.debug("Close locations: %s", close_locations)
    assert game.current_short_phase.endswith("M"), game.current_short_phase
    assert all(i in LOCS for i in close_locations), (close_locations, LOCS)

    per_loc = get_power_per_loc_orders(game, power)
    if fix_uncoordinated_base:
        base_action = make_supports_coordinated(base_action)
    base_action_dict = {x.split()[1].split("/")[0]: x for x in base_action}
    for x in base_action_dict:
        assert x in per_loc, x
    for loc in list(per_loc):
        if loc not in close_locations:
            assert loc in base_action_dict, (loc, base_action)
            per_loc[loc] = [base_action_dict[loc]]
    return filter_uncoordinated_int_loc_dicts(per_loc)


def yield_round_robin(iterables):
    iterators = [iter(x) for x in iterables]
    while iterators:
        good = []
        for iterator in iterators:
            try:
                yield next(iterator)
            except StopIteration:
                continue
            good.append(iterator)
        iterators = good


def produce_cliques(
    game, power: Power, selected_location=None, with_holes=False
) -> List[FrozenSet[str]]:
    power_locs = frozenset(game.get_orderable_locations()[power])
    if selected_location is not None:
        assert selected_location in ADJ_MAPPING, (selected_location, list(ADJ_MAPPING))
        if not with_holes:
            assert selected_location in power_locs, (selected_location, power_locs)

    cliques = []
    for target_location in ADJ_MAPPING:
        if not with_holes and target_location not in power_locs:
            continue
        if selected_location is not None and target_location != selected_location:
            continue
        close_locations = set([target_location] + ADJ_MAPPING[target_location]) & power_locs
        if selected_location is not None:
            assert len(close_locations), f"No units around {selected_location}"
        if not close_locations:
            continue
        if len(close_locations) == 1 and close_locations != {target_location}:
            # If cliques contains a single element, enumerate it only when it's
            # centered in the element.
            continue
        cliques.append(frozenset(close_locations))

    return cliques


def generate_coordinated_local_modifications(
    game,
    power: Power,
    *,
    actions: List[Action],
    selected_location=None,
    max_actions=0,
    search_size=0,
    fix_uncoordinated_base=False,
    with_holes=False,
) -> List[Action]:
    good_locs = sorted(game.get_orderable_locations()[power])
    if not good_locs:
        return [tuple()]
    streams = []
    search_size = search_size or 10 * max_actions
    per_loc_search_size = search_size // len(good_locs)
    cliques = produce_cliques(
        game, power, selected_location=selected_location, with_holes=with_holes
    )
    for orders in actions:
        logging.debug("Building local modifications of %s", orders)
        for close_locations in cliques:
            per_loc = produce_local_loc_dicts(
                game, power, orders, close_locations, fix_uncoordinated_base=fix_uncoordinated_base
            )
            prod_size = int_prod([len(x) for x in per_loc.values()])
            logging.debug(
                "Building local prod generator for %s relative to %s: %s",
                power,
                close_locations,
                " ".join(
                    [
                        *["%s=%s" % (loc, len(orders)) for loc, orders in per_loc.items()],
                        "total_prod=%s" % prod_size,
                        "per_loc_search_size=%s" % per_loc_search_size,
                    ]
                ),
            )
            if not max_actions or prod_size <= per_loc_search_size:
                logging.debug("Generating full product")
                prod = itertools.product(*per_loc.values())
            else:
                logging.debug("Generating sampled product")
                prod = itertools.islice(yield_samples_from_product(*per_loc.values()), search_size)
            streams.append(prod)

    logging.info("Building set of unique orders for %s", power)
    unique_orders = set()
    n_seen = 0
    for order in yield_round_robin(streams):
        n_seen += 1
        if order not in unique_orders:
            unique_orders.add(order)
            if search_size > 0 and len(unique_orders) >= search_size:
                break

    logging.info(
        "About to start coordination filtering %s: unique=%d/%d", power, len(unique_orders), n_seen
    )
    prod = sorted(unique_orders)
    random.shuffle(prod)
    coord_prod = (x for x in prod if are_supports_coordinated(x))
    if max_actions:
        coord_prod = itertools.islice(coord_prod, max_actions)
    coord_prod = sorted(coord_prod)
    logging.info(
        "Generated all orders for %s: %s",
        power,
        " ".join(
            [
                "search_prod=%s" % n_seen,
                "search_prod_uniq=%s" % len(unique_orders),
                "coord_prod=%s" % len(coord_prod),
            ]
        ),
    )
    return coord_prod


def get_all_possible_orders_fast(
    game, power: Power, max_actions=0, search_size=0, allow_foreign_supports=True
):
    good_locs = game.get_orderable_locations()[power]
    all_per_loc = get_power_per_loc_orders(game, power)

    def is_not_foreign_support(order):
        fields = order.split()
        if fields[2] in ("-", "H"):
            return True
        return fields[4] in good_locs

    if not allow_foreign_supports:
        all_per_loc = {
            loc: [order for order in orders if is_not_foreign_support(order)]
            for loc, orders in all_per_loc.items()
        }

    def simplify(order):
        fields = [x.split("/")[0] for x in order.split()]
        if len(fields) > 2 and fields[2] == "-":
            return " ".join(fields[:4])
        else:
            return " ".join(fields[:2])

    id2order = tuple(itertools.chain(*all_per_loc.values()))
    order2id = {order: idx for idx, order in enumerate(id2order)}

    simple_orders = tuple(sorted(set(map(simplify, order2id))))
    simple_order2id = {order: idx for idx, order in enumerate(simple_orders)}

    order_id_to_simple_id = np.array([simple_order2id[simplify(order)] for order in id2order])

    requires_full = np.full(len(id2order), -1, dtype="int32")
    for idx, order in enumerate(id2order):
        fields = [x.split("/")[0] for x in order.split()]
        if fields[2] in ("C", "S") and fields[4] in good_locs:
            # internal support
            requires_full[idx] = simple_order2id[simplify(" ".join(fields[3:]))]

    per_loc = [np.array([order2id[order] for order in orders]) for orders in all_per_loc.values()]

    search_size = search_size or 10 * max_actions
    prod_size = int_prod([len(x) for x in per_loc])

    logging.info(
        "About to generate order product for %s: %s",
        power,
        " ".join(
            [
                *["%s=%s" % (loc, len(all_per_loc[loc])) for loc in good_locs],
                "total_prod=%s" % prod_size,
                "max_actions=%s" % max_actions,
                "search_size=%s" % search_size,
            ]
        ),
    )
    if not max_actions or prod_size <= search_size:
        logging.info("Generating full product")
        ids = numpy_product(*per_loc)
    else:
        logging.info("Generating sampled product")
        ids = np.concatenate(
            [
                action_set[np.random.randint(0, len(action_set), size=(search_size, 1))]
                for action_set in per_loc
            ],
            1,
        )

    is_good_row = _coordination_check_helper(ids, order_id_to_simple_id[ids], requires_full)
    actions = set()
    good_ids = ids[is_good_row == 1]
    for row in good_ids:
        actions.add(tuple(id2order[i] for i in row))

    actions = sorted(actions)

    if max_actions and len(actions) > max_actions:
        random.shuffle(actions)
        actions = sorted(actions[:max_actions])

    logging.info(
        "Generated order product for %s: %s",
        power,
        " ".join(
            [
                *["%s=%s" % (loc, len(all_per_loc[loc])) for loc in good_locs],
                "total_prod=%s" % prod_size,
                "max_actions=%s" % max_actions,
                "search_size=%s" % search_size,
                "found_size=%s" % len(actions),
            ]
        ),
    )
    return actions


def _coordination_check_helper(ids, simplified_ids, requires_full):
    good = np.ones(len(ids), dtype=np.int32)
    for idx, (row, row_simple) in enumerate(zip(ids, simplified_ids)):
        for order_id in row:
            required_order_id = requires_full[order_id]
            if required_order_id != -1 and required_order_id not in row_simple:
                good[idx] = 0
                break
    return good


def _show(name, x):
    # print("====", name)
    # print(*x, sep="\n")
    return x


def generate_double_oracle_actions(
    generation_cfg,
    game: pydipcc.Game,
    agent,
    *,
    agent_power: Optional[Power],
    allowed_powers: List[Power],
    initial_plausible_orders_policy: PowerPolicies,
    last_policies: Dict[Power, Dict[Action, float]],
) -> Dict[Power, Tuple[Action]]:
    device = next(iter(agent.base_strategy_model.model.parameters())).device
    all_actions = {power: set() for power in allowed_powers}
    if generation_cfg.uniform is not None:
        for power in allowed_powers:
            all_actions[power].update(
                _show(
                    f"{power} all",
                    get_all_possible_orders_fast(
                        game,
                        power,
                        max_actions=generation_cfg.max_actions,
                        allow_foreign_supports=generation_cfg.uniform.allow_foreign_supports,
                    ),
                )
            )
    if generation_cfg.column is not None:
        if generation_cfg.column.model_path:
            base_strategy_model_model = load_model.load_base_strategy_model_model_cached(
                checkpoint_path=generation_cfg.column.model_path, map_location=device
            )
        else:
            base_strategy_model_model = agent.base_strategy_model.model
        for power in allowed_powers:
            all_actions[power].update(
                _show(
                    f"{power} per_pos",
                    generate_order_by_column_from_base_strategy_model(
                        model=base_strategy_model_model,
                        game=game,
                        selected_power=power,
                        agent_power=agent_power,
                        max_actions=generation_cfg.max_actions,
                        temperature=generation_cfg.column.temperature,
                    ),
                )
            )
    if generation_cfg.base_strategy_model is not None:
        if generation_cfg.base_strategy_model.model_path:
            base_strategy_model_model = load_model.load_base_strategy_model_model_cached(
                checkpoint_path=generation_cfg.base_strategy_model.model_path, map_location=device
            )
        else:
            base_strategy_model_model = agent.proposal_base_strategy_model.model
        for power in allowed_powers:
            all_actions[power].update(
                _show(
                    f"{power} base_strategy_model",
                    generate_orders_from_base_strategy_model(
                        model=base_strategy_model_model,
                        game=game,
                        selected_power=power,
                        agent_power=agent_power,
                        max_actions=generation_cfg.max_actions,
                        temperature=generation_cfg.base_strategy_model.temperature,
                        location_order=generation_cfg.base_strategy_model.location_order,
                    ),
                )
            )
    if generation_cfg.local_uniform is not None:
        for power in allowed_powers:
            if not initial_plausible_orders_policy[power]:
                # When initialzed from scratch, policy may produce complete
                # garbage that will be filtered out.
                logging.error("WEIRD. No plausible actions for %s.", power)
                continue
            if generation_cfg.local_uniform.use_search_policy:
                power_plausible_actions, probs = zip(
                    *sorted(last_policies[power].items(), key=lambda x: -x[1])
                )
            else:
                power_plausible_actions, _ = zip(
                    *sorted(initial_plausible_orders_policy[power].items(), key=lambda x: -x[1])
                )
            assert generation_cfg.local_uniform.num_base_actions
            n_actions = generation_cfg.local_uniform.num_base_actions
            if not generation_cfg.local_uniform.use_sampling:
                actions = power_plausible_actions[:n_actions]
            else:
                assert (
                    generation_cfg.local_uniform.use_search_policy
                ), "Cannot use sampling with plausible actions"
                # Droping low-prob actions to make sampling with replacement well defined.
                # This implicitly fixes a problem when n_actions >
                # len(power_plausible_actions) too.
                max_samples = min(n_actions, (probs > 1e-3).long().sum().item())
                actions = [
                    power_plausible_actions[i]
                    for i in np.random.choice(
                        len(power_plausible_actions),
                        replace=False,
                        size=max_samples,
                        p=probs.numpy(),
                    )
                ]
            all_actions[power].update(
                _show(
                    f"{power} local uniform",
                    generate_coordinated_local_modifications(
                        game,
                        power,
                        actions=actions,
                        max_actions=generation_cfg.max_actions,
                        fix_uncoordinated_base=generation_cfg.local_uniform.fix_uncoordinated_base,
                        with_holes=generation_cfg.local_uniform.with_holes,
                    ),
                )
            )
    all_actions = {k: tuple(v) for k, v in all_actions.items()}
    return all_actions
