from __future__ import annotations

import torch
from torch import nn

from src.models.mamba_block import MambaBlock


class LongTermInterest(nn.Module):
    """Historical-mean representation followed by the long-term SSM block."""
    def __init__(self, d_model: int = 64, d_state: int = 32, conv_width: int = 4, expansion: int = 2, dropout: float = 0.2):
        super().__init__()
        self.ssm = MambaBlock(d_model, d_state, conv_width, expansion, dropout)

    def forward(self, item_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = torch.cumsum(mask.long(), 1) - mask.long()
        previous_sum = torch.cumsum(item_embeddings, 1) - item_embeddings
        previous_mean = previous_sum / counts.clamp_min(1).unsqueeze(-1)
        x = torch.where((counts == 0).unsqueeze(-1), item_embeddings, item_embeddings + previous_mean)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.ssm(x, mask)
