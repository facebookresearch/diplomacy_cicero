#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""Tools to compute rating of players given a set of games."""
from typing import Any, Dict, List, Optional, Tuple
import collections
import itertools

import torch
import torch.nn
import torch.optim
import torch.utils.data


# 400 log10(e)
ELO_SCALER = 173.717792761
ELO_BIAS = 1000

WinStats = collections.namedtuple("WinRate", "num_wins,num_losses")
Agent = Any
TwoPlayerWinStats = Dict[Tuple[Agent, Agent], WinStats]


def compute_2p_scores(
    win_stats: TwoPlayerWinStats,
    known_ratings: Optional[Dict[Agent, float]] = None,
    **optimizer_kwargs,
) -> Dict[Agent, float]:
    """Computes ELO rating given a dict of winrates.

    Args:
        win_stats: a dictionary of much results between two agents.
        known_ratins: an optional dict of predifined ratings for some agents,
            e.g., from a previous run.
        **optimizer_kwargs: a dict of kwargs to pass to compute_win_mean_rating.

    Returns:
        A dict with ratings for each agent.
    """

    agent2id = {}
    for (agent1, agent2) in win_stats:
        agent2id[agent1] = agent2id.get(agent1, len(agent2id))
        agent2id[agent2] = agent2id.get(agent2, len(agent2id))

    win_loss_pairs: List[Tuple[int, int]] = []
    for (a1_name, a2_name), stats in win_stats.items():
        a1_id = agent2id[a1_name]
        a2_id = agent2id[a2_name]
        win_loss_pairs.extend(((a1_id, a2_id) for _ in range(stats.num_wins)))
        win_loss_pairs.extend(((a2_id, a1_id) for _ in range(stats.num_losses)))
    agent_pairs = torch.LongTensor(win_loss_pairs)

    known_ratings = known_ratings or {}
    known_ratings_tensor = torch.zeros(len(agent2id), device=agent_pairs.device)
    known_ratings_mask = torch.zeros(len(agent2id), device=agent_pairs.device)
    for agent, rating in known_ratings.items():
        agent_id = agent2id[agent]
        known_ratings_tensor[agent_id] = rating
        known_ratings_mask[agent_id] = 1.0

    known_logits_tensor = (known_ratings_tensor - ELO_BIAS) / ELO_SCALER
    logit_tensor = compute_win_mean_ratings(
        agent_pairs, known_logits_tensor, known_ratings_mask, **optimizer_kwargs
    )
    rating_tensor = logit_tensor * ELO_SCALER + ELO_BIAS
    agent_id_to_name = {v: k for k, v in agent2id.items()}
    return {agent_id_to_name[agent_id]: v for agent_id, v in enumerate(rating_tensor.tolist())}


def compute_win_mean_ratings(
    win_loss_pairs: torch.Tensor,
    known_ratings_tensor: torch.Tensor,
    known_ratings_mask: torch.Tensor,
    *,
    seed: Optional[int] = None,
    batch_size: int = 2048,
    lr: float = 0.5,
    l2_weight: float = 1e-3,
    max_updates: int = 2000,
    verbose=False,
) -> torch.Tensor:
    """Compute ratings assuming Prob(a1 beats a2) = sigmoid(r_a1 - r_a2).

    Ratings for some agents may be fixed. In this case known_ratings_tensor[i]
    must contain the rating for agent i and known_ratings_mask must contain 1.0.
    Otherwise, known_ratings_mask[i] must be 0.0.
    """

    def data_producer():
        dataset = torch.utils.data.TensorDataset(win_loss_pairs)
        data_loader = torch.utils.data.DataLoader(  # type:ignore
            dataset, shuffle=True, batch_size=min(batch_size, len(dataset)), drop_last=True
        )
        while True:
            yield from data_loader

    if seed is not None:
        torch.manual_seed(seed)
    assert (win_loss_pairs >= 0).all()
    ratings_raw = torch.nn.Parameter(  # type:ignore
        torch.zeros(win_loss_pairs.max().item() + 1, device=win_loss_pairs.device)
    )

    def get_ratins_with_known():
        return ratings_raw * (1 - known_ratings_mask) + known_ratings_tensor * known_ratings_mask

    optimizer = torch.optim.Adagrad([ratings_raw], lr)
    ema_loss = None
    ema_alpha = 0.01
    for i, (pairs,) in enumerate(itertools.islice(data_producer(), max_updates)):
        winner_ids, looser_ids = pairs.T
        ratings = get_ratins_with_known()
        rank_loss = torch.sigmoid(ratings[looser_ids] - ratings[winner_ids]).mean()
        l2_loss = ((ratings[looser_ids] ** 2).mean() + (ratings[winner_ids] ** 2).mean()) * 0.5
        loss = rank_loss + l2_loss * l2_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_loss = (
            rank_loss if ema_loss is None else ema_loss + ema_alpha * (rank_loss.item() - ema_loss)
        )
        if verbose and i % 100 == 0:
            print(f"EMA Loss at {i} is {ema_loss}")

    return get_ratins_with_known().cpu()
