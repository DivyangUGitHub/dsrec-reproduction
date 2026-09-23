"""Evaluation package for DSRec ranking metrics."""

from __future__ import annotations

from typing import Iterable

import torch

from .metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from .evaluator import evaluate_batch


def topk_predictions(logits: torch.Tensor, k: int) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, N], got {tuple(logits.shape)}")
    if k <= 0:
        raise ValueError("k must be positive")
    return torch.topk(logits, min(k, logits.size(1)), dim=1).indices


def evaluate_ranking(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Compute Hit/HR and NDCG for one relevant target per example."""
    results: dict[str, float] = {}
    for k in ks:
        results[f"hit@{k}"] = hit_rate_at_k(logits, targets, k)
        results[f"ndcg@{k}"] = ndcg_at_k(logits, targets, k)
    return results


__all__ = [
    "evaluate_batch",
    "evaluate_ranking",
    "hit_rate_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "topk_predictions",
]
