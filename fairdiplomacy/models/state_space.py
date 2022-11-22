#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Code here is modified from:
https://github.com/diplomacy/research/blob/master/diplomacy_research/models/state_space.py

License of that code:
# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
"""
import json
import logging
from typing import Dict, Set
import importlib.resources
import numpy as np
from fairdiplomacy import pydipcc

EOS_IDX = -1
_ORDER_VOCABULARY = None
_ORDER_VOCABULARY_BY_UNIT = None
_ORDER_VOCABULARY_IDXS_BY_UNIT = None
_ORDER_VOCABULARY_IDXS_LEN = None

# Constants
LOGGER = logging.getLogger(__name__)
ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# Constants
NB_LOCS = 81
NB_NODES = NB_LOCS


def get_order_vocabulary():
    global _ORDER_VOCABULARY, _ORDER_VOCABULARY_BY_UNIT, _ORDER_VOCABULARY_IDXS_BY_UNIT, _ORDER_VOCABULARY_IDXS_LEN

    if _ORDER_VOCABULARY is not None:
        return _ORDER_VOCABULARY

    _ORDER_VOCABULARY, _ORDER_VOCABULARY_BY_UNIT = _get_order_vocabulary()
    order_vocabulary_idxs = {order: i for i, order in enumerate(_ORDER_VOCABULARY)}

    _ORDER_VOCABULARY_IDXS_BY_UNIT = {
        unit: [order_vocabulary_idxs[order] for order in orders]
        for unit, orders in _ORDER_VOCABULARY_BY_UNIT.items()
    }

    _ORDER_VOCABULARY_IDXS_LEN = max(len(o) for o in _ORDER_VOCABULARY_IDXS_BY_UNIT.values())

    return _ORDER_VOCABULARY


def get_order_vocabulary_by_unit():
    get_order_vocabulary()
    return _ORDER_VOCABULARY_BY_UNIT


def get_order_vocabulary_idxs_len() -> int:
    get_order_vocabulary()
    assert _ORDER_VOCABULARY_IDXS_LEN is not None
    return _ORDER_VOCABULARY_IDXS_LEN


def get_order_vocabulary_idxs_by_unit():
    get_order_vocabulary()

    return _ORDER_VOCABULARY_IDXS_BY_UNIT


def _get_order_vocabulary():
    """ Computes the list of all valid orders on the standard map
        :return: A sorted list of all valid orders on the standard map
    """
    orders_by_unit = json.loads(
        importlib.resources.read_text("fairdiplomacy.models", "order_vocab_by_unit.json")
    )

    orders_by_unit = {k: sorted(list(v)) for k, v in orders_by_unit.items()}
    sorted_unit_keys = sorted(orders_by_unit)
    final_orders = []
    for unit in sorted_unit_keys:
        final_orders += orders_by_unit[unit]

    return final_orders, orders_by_unit


def get_board_alignments(locs, in_adjustment_phase, tokens_per_loc, decoder_length):
    """ Returns a n list of (NB_NODES vector) representing the alignments (probs) for the locs on the board state
        :param locs: The list of locs being outputted by the model
        :param in_adjustment_phase: Indicates if we are in A phase (all locs possible at every position) or not.
        :param tokens_per_loc: The number of tokens per loc (TOKENS_PER_ORDER for token_based, 1 for order_based).
        :param decoder_length: The length of the decoder.
        :return: A list of [NB_NODES] vector of probabilities (alignments for each location)
    """
    alignments = []

    # Regular phase
    if not in_adjustment_phase:
        for loc in locs:
            alignment = np.zeros([NB_NODES], dtype=np.uint8)
            alignment_index = ALIGNMENTS_INDEX.get(loc[:3], [])
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning("Location %s is not in the alignments index.", loc)
            if alignment_index:
                for index in alignment_index:
                    alignment[index] = 1
            alignments += [alignment] * tokens_per_loc
        if decoder_length != len(locs) * tokens_per_loc:
            LOGGER.warning(
                "Got %d tokens, but decoder length is %d",
                len(locs) * tokens_per_loc,
                decoder_length,
            )
        if decoder_length > len(alignments):
            LOGGER.warning("Got %d locs, but the decoder length is %d", len(locs), decoder_length)
            alignments += [np.zeros([NB_NODES], dtype=np.uint8)] * (
                decoder_length - len(alignments)
            )

    # Adjustment phase (All locs at all positions)
    else:
        alignment = np.zeros([NB_NODES], dtype=np.uint8)
        alignment_index = set()
        for loc in locs:
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning("Location %s is not in the alignments index.", loc)
            for index in ALIGNMENTS_INDEX.get(loc[:3], []):
                alignment_index.add(index)
        if alignment_index:
            for index in alignment_index:
                alignment[index] = 1
        alignments = [alignment] * decoder_length

    # Validating size
    if decoder_length != len(alignments):
        LOGGER.warning(
            "Got %d alignments, but decoder length is %d", len(alignments), decoder_length
        )

    # Returning
    return alignments


def get_alignments_index(map_name="standard"):
    """ Computes a list of nodes index for each possible location
        e.g. if the sorted list of locs is ['BRE', 'MAR', 'PAR'] would return {'BRE': [0], 'MAR': [1], 'PAR': [2]}
    """
    sorted_locs = pydipcc.Game.LOC_STRS[:]
    alignments_index = {}

    # Computing the index of each loc
    for loc in sorted_locs:
        if loc[:3] in alignments_index:
            continue
        alignments_index[loc[:3]] = [
            index for index, sorted_loc in enumerate(sorted_locs) if loc[:3] == sorted_loc[:3]
        ]
    return alignments_index


def get_adjacency_matrix():
    """ Computes the adjacency matrix for map
        :param map_name: The name of the map
        :return: A (nb_nodes) x (nb_nodes) matrix
    """
    order_vocab = get_order_vocabulary()
    # Finding list of all locations
    locs = pydipcc.Game.LOC_STRS[:]
    adjacencies = np.zeros((len(locs), len(locs)), dtype=np.bool)  # type: ignore

    # Building adjacencies between locs
    # Coasts are adjacent to their parent location (without coasts)
    for i, loc_1 in enumerate(locs):
        for j, loc_2 in enumerate(locs):
            if (
                " ".join(["A", loc_1, "-", loc_2]) in order_vocab
                or " ".join(["F", loc_1, "-", loc_2]) in order_vocab
            ):
                adjacencies[i, j] = 1
            if loc_1 != loc_2 and (loc_1[:3] == loc_2 or loc_1 == loc_2[:3]):
                adjacencies[i, j] = 1

    return adjacencies


# Caching alignments
ALIGNMENTS_INDEX = get_alignments_index()
