#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from fairdiplomacy.models.consts import LOCS, POWERS, LOC_NAMES
import string
import itertools
import random
from pygtrie import CharTrie
from typing import Dict, List, Tuple
from parlai_diplomacy.tasks.discriminator import noisy_locations

SYMBOLS_LOWER = set([l.split("/")[0].lower() for l in LOCS])
POWERS_LOWER = set([p.lower() for p in POWERS])
POWER_ADJS = {"austrian", "english", "french", "german", "italian", "russian", "turkish"}
LOCS_LOWER = set([i.lower() for i in LOC_NAMES])
BOUNDARY = [" "] + list(string.punctuation)
NOISY_LOCS_LOWER = set(noisy_locations.NOISY_LOCATIONS) - LOCS_LOWER - SYMBOLS_LOWER
PLURAL_CARDINALS = {
    "zero",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "0",
    *[str(i) for i in range(2, 19)],
}
SINGULAR_CARDINALS = {"one", "1", "a", "an", "my"}
CARDINALS = {*SINGULAR_CARDINALS, *PLURAL_CARDINALS}
DIPLOMACY_SINGULAR_NOUNS = {
    "unit",
    "build",
    "phase",
    "year",
    "turn",
    "center",
    "centre",
    "sc",
    "dot",
    "army",
    "fleet",
    "power",
    "player",
}
DIPLOMACY_PLURAL_NOUNS = {
    "units",
    "builds",
    "phases",
    "years",
    "turns",
    "centers",
    "centres",
    "scs",
    "dots",
    "armies",
    "fleets",
    "powers",
    "players",
}
DIPLOMACY_NOUNS = {*DIPLOMACY_SINGULAR_NOUNS, *DIPLOMACY_PLURAL_NOUNS}

# Negationy words and words that are near to that in meaning.
# We capture all words ending in n't, so much of the below is just for capturing
# chatty spellings of them that omit the apostrophe.
NEGATIONY_WORDS = {
    "aint",
    "any",
    "arent",
    "barely",
    "cant",
    "cannot",
    "couldnt",
    "didnt",
    "doesnt",
    "dont",
    "hadnt",
    "hardly",
    "hasnt",
    "havent",
    "instead",
    "isnt",
    "neither",
    "never",
    "no",
    "not",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "only",
    "shouldnt",
    "wasnt",
    "werent",
    "without",
    "wont",
    "wouldnt",
}
CONJUNCTIONS_AND_PUNCTUATION = {
    " and ",
    " or ",
    " but ",
    " so ",
    " because ",
    " if ",
    " then ",
    " before ",
    " after ",
    " since ",
    " in case ",
    " in order to ",
    " however ",
    " which ",
    " that ",
}.union({",", ";", "?", "??", "???", "!", "!!", "!!!", ".", "..", "..."})

JUSTIFICATION_KEY_PHRASES = [
    "so that ",
    "because ",
    "so i ",
    "so he ",
    "so she ",
    "so you ",
    "so russia ",
    "so austria ",
    "so italy ",
    "so england ",
    "so france ",
    "so germany ",
    "so turkey ",
    "but i ",
    "but you ",
    "but he ",
    "but she ",
    "but i'm ",
    "but russia ",
    "but austria ",
    "but italy ",
    "but england ",
    "but france ",
    "but germany ",
    "but turkey ",
    "but if ",
    "but it ",
    "if i ",
    "if you ",
    "if you're ",
    "if he ",
    "if she ",
    "if russia ",
    "if austria ",
    "if italy ",
    "if england ",
    "if france ",
    "if germany ",
    "if turkey ",
    "in order to ",
    "in case ",
    "since i ",
    "since he ",
    "since you ",
    "since russia ",
    "since austria ",
    "since italy ",
    "since england ",
    "since france ",
    "since germany ",
    "since turkey ",
    "however i ",
    "which will ",
    "which would ",
    "want to risk ",
    "you will ",
    "you would ",
    "that will ",
    "that would ",
]


def build_entity_trie():
    """
    Builds a search trie for each entity
    Search trie tries to match entities which have a boundary char before and after it.
    This allows us to do tokenization-free searching.

    Returns:
        trie: A pygtrie based search trie.
    """

    def get_entity_with_boundary(entity):
        for sb, eb in itertools.product(BOUNDARY, BOUNDARY):
            yield sb + entity + eb

    trie = CharTrie()
    for i in NOISY_LOCS_LOWER:
        for _i in get_entity_with_boundary(i):
            trie[_i] = "noisy_locations"
    for i in LOCS_LOWER:
        for _i in get_entity_with_boundary(i):
            trie[_i] = "locations"
    for i in SYMBOLS_LOWER:
        for _i in get_entity_with_boundary(i):
            trie[_i] = "symbols"
    for i in POWERS_LOWER:
        for _i in get_entity_with_boundary(i):
            trie[_i] = "powers"
    for i in POWER_ADJS:
        for _i in get_entity_with_boundary(i):
            trie[_i] = "power_adjs"
    return trie


def search_entities(
    message: str, trie: Dict[str, str]
) -> Dict[Tuple[str, str], List[Tuple[int, int, str]]]:
    """
    Searches for all possible matches in a message with a trie.
    Args:
        message: A message string
        trie: A Char trie
    Returns:
        matches: dict(tuple(str, str), List(int, int, str))
        A dictionary with keys of the form (entity, entity_type) and values are a list of (s_idx, e_idx, matched_span)
    """
    assert (
        message[0] == message[-1] == " "
    ), "Tokenization free searching requires a boundary symbol in the start and end of message"
    message_l = (
        message.lower()
    )  # we still keep the original (unlowered message to match case after finding matches)
    st = 0
    matches = {}
    while st < len(message):
        m = trie.longest_prefix(message_l[st:])
        if m:
            end = st + len(m.key)
            match_key = (m.key[1:-1], m.value)
            repls = matches.get(match_key, [])
            original_span = message[st:end]
            repls.append((st, end, original_span))
            matches[match_key] = repls
            st += len(m.key) - 1
        else:
            st += 1
    return matches


def match_case(new_text: str, original_text: str) -> str:
    """
    matches the case of `new_text` to that of `original_text`
    """
    if original_text.islower():
        return new_text.lower()
    elif original_text.isupper():
        return new_text.upper()
    else:
        return new_text.capitalize()


def get_possible_replacements(typ: str, original: str) -> str:
    """
    Finds an appropriate set of possible replacements for power,location or symbol for the original
    """
    if typ == "powers":
        return POWERS_LOWER - {original.lower()}
    elif typ == "power_adjs":
        return POWER_ADJS - {original.lower()}
    elif typ == "locations":
        return LOCS_LOWER - {original.lower()}
    elif typ == "symbols":
        return SYMBOLS_LOWER - {original.lower()}
    elif typ == "noisy_locations":
        return NOISY_LOCS_LOWER - {original.lower()}
    else:
        raise ("Unknown entity type for corruption!")


def get_consistant_replacements(
    entity: str, typ: str, instances: List[Tuple[int, int, str, str]]
) -> List[Tuple[int, int, str, str]]:
    flattened_replacements = []
    possible_replacements = get_possible_replacements(typ, entity)
    repl = random.sample(possible_replacements, 1)[0]
    for s_idx, e_idx, original_span in instances:
        orig = original_span[1:-1]  # removing the boundary characters
        repl_span = (
            original_span[0] + match_case(repl, orig) + original_span[-1]
        )  # insert boundary chars around replacement
        flattened_replacements.append((s_idx, e_idx, original_span, repl_span))
    return flattened_replacements


def splice_replace(message: str, st: int, end: int, replacement: str) -> str:
    """
    swap in new replacement between st and end positions
    """
    assert (
        message[0] == message[-1] == " "
    ), "Tokenization free searching requires a boundary symbol in the start and end of message"
    return message[:st] + replacement + message[end:]
