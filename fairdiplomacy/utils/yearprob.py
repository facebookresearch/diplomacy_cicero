#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Optional


def _parse_yearprob_config_str(
    config_value: str, default_key_values: Dict[int, float]
) -> Dict[int, float]:
    """Parses a string like "1901,0.3;1904,0.5" into a dict.
    Dict will include all keys and values in default_key_values unless the user overrides those
    specific years.
    """
    yearprob_mapping = {key: default_key_values[key] for key in default_key_values}
    for piece in config_value.split(";"):
        piece = piece.strip()
        if len(piece) <= 0:
            continue
        yearprob = piece.split(",")
        assert len(yearprob) == 2, f"Could not parse {piece} in {config_value}"
        try:
            year = int(yearprob[0])
            prob = float(yearprob[1])
        except ValueError:
            assert False, f"Could not parse {piece} in {config_value}"
        yearprob_mapping[year] = prob
    return yearprob_mapping


def get_prob_of_latest_year_leq(yearprob: Dict[int, float], year: int) -> float:
    """Return the probability associated with the greatest year <= the specified year."""
    largest_year_leq = max(y for y in yearprob if y <= year)
    return yearprob[largest_year_leq]


def parse_year_spring_prob_of_ending(config_value: Optional[str]) -> Optional[Dict[int, float]]:
    """Parses a yearprob dict intended to represent the probability of ending the game at
    the start of spring on each year."""
    year_spring_prob_of_ending = None
    if config_value is not None:
        # Default should be {1901: 0.0} so that even if the user doesn't specify,
        # we have get_prob_of_latest_year_leq returning 0.0 as a base case.
        year_spring_prob_of_ending = _parse_yearprob_config_str(
            config_value, default_key_values={1901: 0.0}
        )
    return year_spring_prob_of_ending
