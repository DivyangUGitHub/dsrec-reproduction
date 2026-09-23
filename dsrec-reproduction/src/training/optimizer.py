from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
