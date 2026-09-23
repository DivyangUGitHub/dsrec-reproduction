from __future__ import annotations

import torch
from torch import nn


class ItemEmbedding(nn.Module):
    def __init__(self, n_items: int, d_model: int = 64, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_items + 1, d_model, padding_idx=padding_idx)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(item_ids)


class TimeEmbedding(nn.Module):
    def __init__(self, n_time_buckets: int = 10, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(n_time_buckets, d_model)

    def forward(self, bucket_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(bucket_ids)
