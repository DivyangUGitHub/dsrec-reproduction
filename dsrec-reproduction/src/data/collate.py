
from __future__ import annotations

import torch


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([example[key] for example in batch]) for key in batch[0]}
