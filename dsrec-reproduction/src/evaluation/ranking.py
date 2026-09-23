from __future__ import annotations

import torch


def top_k(logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    return torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices


def rank_of_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits, dim=-1, descending=True)
    matches = order == targets.unsqueeze(-1)
    return matches.float().argmax(dim=-1) + 1
