"""
Build per-user chronological sequences from processed interactions (Phase 4).

[PAPER-SPECIFIED] Sec. III-A: S_u = [v_1, ..., v_T].

Duplicate items within a user's sequence are retained, not deduplicated
[ASSUMPTION — see docs/paper_audit.md, "other numerical details"].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.time_features import compute_time_gaps


@dataclass
class UserSequence:
    user_id: int
    items: np.ndarray       # [T], internal item IDs, chronological
    timestamps: np.ndarray  # [T], raw timestamps, chronological
    time_gaps: np.ndarray   # [T], Eq. 9: d_1 = 0, d_i = t_i - t_{i-1}


def build_sequences(df: pd.DataFrame) -> dict[int, UserSequence]:
    """
    df must already be sorted by (user_id, timestamp) — see
    src.data.preprocessing.sort_chronologically. This function does not
    re-sort, so an unsorted df silently produces wrong sequences; the
    caller owns the sort.
    """
    sequences: dict[int, UserSequence] = {}
    for user_id, group in df.groupby("user_id", sort=False):
        items = group["item_id"].to_numpy()
        timestamps = group["timestamp"].to_numpy()
        sequences[user_id] = UserSequence(
            user_id=user_id,
            items=items,
            timestamps=timestamps,
            time_gaps=compute_time_gaps(timestamps),
        )
    return sequences
