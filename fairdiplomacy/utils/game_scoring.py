#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Sequence, Tuple, List
import collections
import json
import re

from fairdiplomacy.models.consts import POWERS, N_SCS

GameScores = collections.namedtuple(
    "GameScores",
    [
        "center_ratio",  # Ratio of the player's SCs to N_SCS.
        "draw_score",  # 1. / num alive players, but 1/0 if clear win/loss.
        "square_ratio",  # Ratio of the squure of the SCs to sum of squares.
        "square_score",  # Same as square_ratio, but 1 if is_clear_win.
        "is_complete_unroll",  # 0/1 whether last phase is complete.
        "is_clear_win",  # 0/1 whether the player has more than half SC.
        "is_clear_loss",  # 0/1 whether another player has more than half SC.
        "is_eliminated",  # 0/1 whether has 0 SC.
        "is_leader",  # 0/1 whether the player has at least as many SCs as anyone else.
        "can_draw",  # 0/1 whether the player is alive and nobody wins solo.
        "num_games",  # Number of games being averaged
    ],
)


def compute_game_sos_from_state(game_state: Dict) -> List[float]:
    center_counts = [len(game_state["centers"].get(p, [])) for p in POWERS]
    clear_wins = [c > (N_SCS / 2) for c in center_counts]
    if any(clear_wins):
        return [float(w) for w in clear_wins]
    center_squares = [x ** 2 for x in center_counts]
    sum_sq = sum(center_squares)
    return [c / sum_sq for c in center_squares]


def compute_game_dss_from_state(game_state: Dict) -> List[float]:
    center_counts = [len(game_state["centers"].get(p, [])) for p in POWERS]
    clear_wins = [c > (N_SCS / 2) for c in center_counts]
    if any(clear_wins):
        return [float(w) for w in clear_wins]
    alive = [c > 0 for c in center_counts]
    n_alive = sum(alive)
    return [a / n_alive for a in alive]


def compute_phase_scores(power_id: int, phase_json: Dict) -> GameScores:
    return compute_game_scores_from_state(power_id, phase_json["state"])


def compute_game_scores(power_id: int, game_json: Dict) -> GameScores:
    return compute_phase_scores(power_id, game_json["phases"][-1])


def add_offset_to_square_score(game_scores: GameScores, offset: float) -> GameScores:
    return game_scores._replace(square_score=game_scores.square_score + offset)


def compute_game_scores_from_state(power_id: int, game_state: Dict) -> GameScores:
    center_counts = [len(game_state["centers"].get(p, [])) for p in POWERS]
    center_squares = [x ** 2 for x in center_counts]
    complete_unroll = game_state["name"] == "COMPLETED"
    is_clear_win = center_counts[power_id] > N_SCS / 2
    someone_wins = any(c > N_SCS / 2 for c in center_counts)
    is_eliminated = center_counts[power_id] == 0
    is_clear_loss = is_eliminated or (not is_clear_win and someone_wins)
    metrics = dict(
        center_ratio=center_counts[power_id] / N_SCS,
        square_ratio=center_squares[power_id] / sum(center_squares, 1e-5),
        is_complete_unroll=float(complete_unroll),
        is_clear_win=float(is_clear_win),
        is_clear_loss=float(is_clear_loss),
        is_eliminated=float(is_eliminated),
        is_leader=float(center_counts[power_id] == max(center_counts)),
        can_draw=float(not someone_wins and not is_eliminated),
    )
    metrics["square_score"] = (
        1.0 if is_clear_win else (0 if is_clear_loss else metrics["square_ratio"])
    )
    is_alive = not is_eliminated
    num_alive = sum(int(x > 0) for x in center_counts)
    metrics["draw_score"] = float(is_clear_win) if someone_wins else float(is_alive) / num_alive
    return GameScores(**metrics, num_games=1)


def average_game_scores(many_games_scores: Sequence[GameScores]) -> Tuple[GameScores, GameScores]:
    assert many_games_scores, "Must be non_empty"
    avgs, stderrs = {}, {}
    tot_n_games = sum(scores.num_games for scores in many_games_scores)
    # In theory, we could get much better stderrs by taking into account that the means for each
    # different powers vary, and when we have enough data, computing per-power variances and combining them.
    # We don't do this since it's messy and much tricker, statistically.
    for key in GameScores._fields:
        if key == "num_games":
            continue
        avgs[key] = (
            sum(getattr(scores, key) * scores.num_games for scores in many_games_scores)
            / tot_n_games
        )
        # Divide by N-1 for unbiased estimate, with hack to not crash on divide by 0.
        # We could do better things here given that we also know that most of our values are bounded in [0,1]
        # and that our null hypothesis for many is 1/7, but again, that's messier and harder and not worth much.
        sample_variance = sum(
            (getattr(scores, key) - avgs[key]) ** 2 * scores.num_games
            for scores in many_games_scores
        ) / max(0.5, tot_n_games - 1)
        stderrs[key] = (sample_variance / tot_n_games) ** 0.5

    return GameScores(**avgs, num_games=tot_n_games), GameScores(**stderrs, num_games=tot_n_games)


def get_power_one(game_json_path):
    """This function is depreccated. Use fairdiplomacy.compare_agents_array."""
    name = re.findall("game.*\\.json", game_json_path)[0]
    for power in POWERS:
        if power[:3] in name:
            return power

    raise ValueError(f"Couldn't parse power name from {name}")


def get_game_result_from_json(game_json_path):
    """This function is depreccated. Use fairdiplomacy.compare_agents_array."""
    power_one = get_power_one(game_json_path)

    try:
        with open(game_json_path) as f:
            j = json.load(f)
    except Exception as e:
        print(e)
        return None

    rl_rewards = compute_game_scores(POWERS.index(power_one), j)

    counts = {k: len(v) for k, v in j["phases"][-1]["state"]["centers"].items()}
    for p in POWERS:
        if p not in counts:
            counts[p] = 0
    powers_won = {p for p, v in counts.items() if v == max(counts.values())}
    power_won = power_one if power_one in powers_won else powers_won.pop()

    if counts[power_one] == 0:
        return "six", power_one, power_won, rl_rewards

    winner_count, winner = max([(c, p) for p, c in counts.items()])
    if winner_count < 18:
        return "draw", power_one, power_won, rl_rewards

    if winner == power_one:
        return "one", power_one, power_won, rl_rewards
    else:
        return "six", power_one, power_won, rl_rewards
