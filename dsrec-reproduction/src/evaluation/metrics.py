from __future__ import annotations

import torch


def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    topk = logits.topk(k, dim=-1).indices
    return float((topk == targets.unsqueeze(-1)).any(dim=-1).float().mean())


def hit_rate_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    return recall_at_k(logits, targets, k)


def ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    topk = logits.topk(k, dim=-1).indices
    hits = topk == targets.unsqueeze(-1)
    ranks = hits.float().argmax(dim=-1) + 1
    values = torch.where(hits.any(dim=-1), 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks, dtype=torch.float))
    return float(values.mean())
