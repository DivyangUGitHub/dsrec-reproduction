from __future__ import annotations

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k


def evaluate_batch(logits, targets, ks=(5, 10, 20)):
    result = {}
    for k in ks:
        result[f"HR@{k}"] = hit_rate_at_k(logits, targets, k)
        result[f"NDCG@{k}"] = ndcg_at_k(logits, targets, k)
    return result
