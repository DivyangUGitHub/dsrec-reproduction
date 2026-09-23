from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ("user_id", "item_id", "rating", "timestamp")


def load_ratings(path: Path) -> pd.DataFrame:
    """
    Load ratings.dat into a DataFrame with columns
    (user_id, item_id, rating, timestamp), all int64.

    Shapes: one row per raw interaction. No sorting, filtering, or ID
    remapping happens here — see src.data.preprocessing for that.
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=COLUMNS,
        header=None,
        encoding="latin-1",
    )
    return df.astype(
        {"user_id": "int64", "item_id": "int64", "rating": "int64", "timestamp": "int64"}
    )
