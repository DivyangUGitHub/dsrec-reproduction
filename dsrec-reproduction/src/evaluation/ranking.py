from __future__ import annotations

import torch


def top_k(logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    return torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices


def rank_of_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits, dim=-1, descending=True)
    matches = order == targets.unsqueeze(-1)
    return matches.float().argmax(dim=-1) + 1


def mask_seen_items(
    logits: torch.Tensor,
    item_ids: torch.Tensor,
    sequence_mask: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mask context items for full-sort ranking, preserving the held-out target.

    Candidate filtering is not specified explicitly in the DSRec paper; this is
    the project's documented evaluation assumption for recommendation ranking.
    """
    if logits.ndim != 2 or item_ids.ndim != 2:
        raise ValueError("logits and item_ids must have shapes [B, N] and [B, L]")
    if item_ids.shape != sequence_mask.shape:
        raise ValueError("item_ids and sequence_mask must have the same shape")
    if item_ids.size(0) != logits.size(0) or targets.size(0) != logits.size(0):
        raise ValueError("batch dimensions must match")

    seen = torch.zeros_like(logits, dtype=torch.bool)
    safe_ids = item_ids.clamp(min=0, max=logits.size(1) - 1)
    seen.scatter_(1, safe_ids, sequence_mask.to(torch.bool))
    # Repeat interactions are possible; never hide the current ground-truth item.
    seen.scatter_(1, targets.unsqueeze(1), False)
    return logits.masked_fill(seen, torch.finfo(logits.dtype).min)
