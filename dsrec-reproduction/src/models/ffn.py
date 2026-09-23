from __future__ import annotations

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.2, expansion: int = 4):
        super().__init__()
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
