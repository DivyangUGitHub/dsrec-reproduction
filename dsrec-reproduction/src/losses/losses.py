from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()
