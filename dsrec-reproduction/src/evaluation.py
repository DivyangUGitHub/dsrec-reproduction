"""
Phase 10 — DSRec evaluation / ranking metrics.

Implements:
    - Hit@K
    - NDCG@K
    - Top-K recommendation ranking

Evaluation is performed on one target item per example.
"""

from __future__ import annotations

from typing import Iterable

import torch


def topk_predictions(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Return top-k predicted item IDs.

    Args:
        logits: [B, n_items + 1]
        k: number of recommendations

    Returns:
        Tensor [B, k]
    """

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [B, N], got {tuple(logits.shape)}"
        )

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, logits.size(1))

    return torch.topk(
        logits,
        k=k,
        dim=1,
    ).indices


def hit_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Hit@K.

    A hit occurs when the target item appears anywhere
    in the model's top-k predictions.
    """

    predictions = topk_predictions(
        logits,
        k,
    )

    targets = targets.view(-1, 1)

    hits = (
        predictions == targets
    ).any(dim=1)

    return hits.float().mean().item()


def ndcg_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """
    Compute NDCG@K for one relevant target item per example.

    If target is ranked at position r (1-indexed):
        DCG = 1 / log2(r + 1)

    Otherwise:
        DCG = 0

    With one relevant item, IDCG = 1.
    """

    predictions = topk_predictions(
        logits,
        k,
    )

    targets = targets.view(-1, 1)

    matches = predictions == targets

    batch_size = predictions.size(0)

    scores = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=predictions.device,
    )

    rows, cols = torch.where(matches)

    if rows.numel() > 0:
        ranks = cols + 1

        scores[rows] = 1.0 / torch.log2(
            ranks.float() + 1.0
        )

    return scores.mean().item()


def evaluate_ranking(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """
    Compute multiple ranking metrics.
    """

    results: dict[str, float] = {}

    for k in ks:
        results[f"hit@{k}"] = hit_at_k(
            logits,
            targets,
            k,
        )

        results[f"ndcg@{k}"] = ndcg_at_k(
            logits,
            targets,
            k,
        )

    return results