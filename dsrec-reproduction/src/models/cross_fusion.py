from __future__ import annotations

import torch
from torch import nn


class CrossFusion(nn.Module):
    """Residual bidirectional fusion used between long/short branches."""
    def __init__(self, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.long_gate = nn.Linear(2 * d_model, d_model)
        self.short_gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, long_x: torch.Tensor, short_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.cat([long_x, short_x], dim=-1)
        lg = torch.sigmoid(self.long_gate(pair))
        sg = torch.sigmoid(self.short_gate(pair))
        return long_x + self.dropout(lg * short_x), short_x + self.dropout(sg * long_x)
