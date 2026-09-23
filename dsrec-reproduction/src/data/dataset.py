from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.time_features import TimeBucketizer

PAD_ID = 0


@dataclass
class Example:
    """One (context, target) pair, before padding — shared by train/val/test."""

    context_items: np.ndarray  # [t], t <= L
    context_gaps: np.ndarray   # [t]
    target: int


class DSRecDataset(Dataset):
    """
    Wraps a list of Example and a fitted TimeBucketizer.

    __getitem__ returns a dict of:
        item_ids:        LongTensor [L]
        time_bucket_ids: LongTensor [L] (value at padded positions is
                          arbitrary — always 0 — since the model must rely
                          on `mask`, not on this value, to ignore padding;
                          see tests/test_causality.py, added in a later phase)
        mask:             BoolTensor [L], True at real (non-pad) positions
        target:           LongTensor scalar
    """

    def __init__(self, examples: list[Example], bucketizer: TimeBucketizer, max_len: int):
        self.examples = examples
        self.bucketizer = bucketizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        items = ex.context_items[-self.max_len :]
        gaps = ex.context_gaps[-self.max_len :]
        t = len(items)

        item_ids = np.full(self.max_len, PAD_ID, dtype=np.int64)
        time_bucket_ids = np.full(self.max_len, PAD_ID, dtype=np.int64)
        mask = np.zeros(self.max_len, dtype=bool)

        item_ids[:t] = items
        time_bucket_ids[:t] = self.bucketizer.transform(gaps)
        mask[:t] = True

        return {
            "item_ids": torch.from_numpy(item_ids),
            "time_bucket_ids": torch.from_numpy(time_bucket_ids),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(ex.target, dtype=torch.long),
        }
