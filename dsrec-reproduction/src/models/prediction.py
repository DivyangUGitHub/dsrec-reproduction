from __future__ import annotations

import torch


def item_logits(user_representation: torch.Tensor, item_embedding: torch.Tensor, mask_padding: bool = True) -> torch.Tensor:
    """Project a user representation against the item embedding matrix."""
    logits = user_representation @ item_embedding.transpose(0, 1)
    if mask_padding and logits.size(-1) > 0:
        logits[..., 0] = torch.finfo(logits.dtype).min
    return logits
