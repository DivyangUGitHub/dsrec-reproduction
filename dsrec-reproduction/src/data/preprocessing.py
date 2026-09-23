
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PAD_ID = 0


@dataclass
class IdMapping:
    """Maps raw MovieLens IDs to contiguous internal IDs starting at 1 (0 is PAD_ID)."""

    raw_to_internal: dict[int, int]
    internal_to_raw: dict[int, int]

    @classmethod
    def build(cls, raw_ids: pd.Series) -> "IdMapping":
        unique_ids = sorted(raw_ids.unique())
        raw_to_internal = {raw: i + 1 for i, raw in enumerate(unique_ids)}  # +1: reserve 0 for PAD_ID
        internal_to_raw = {v: k for k, v in raw_to_internal.items()}
        return cls(raw_to_internal, internal_to_raw)

    def apply(self, raw_ids: pd.Series) -> pd.Series:
        return raw_ids.map(self.raw_to_internal)

    def __len__(self) -> int:
        return len(self.raw_to_internal)


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort interactions by (user_id, timestamp), with a stable sort so
    same-timestamp ties keep their original file order instead of being
    silently reshuffled.
    """
    return df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, IdMapping, IdMapping]:
    """
    Replace raw user_id/item_id with contiguous internal IDs (1..N, 0
    reserved for padding). Returns the remapped DataFrame and both mappings.
    """
    user_map = IdMapping.build(df["user_id"])
    item_map = IdMapping.build(df["item_id"])
    out = df.copy()
    out["user_id"] = user_map.apply(df["user_id"])
    out["item_id"] = item_map.apply(df["item_id"])
    return out, user_map, item_map


def compute_statistics(df: pd.DataFrame, n_users: int, n_items: int) -> dict:
    """
    Dataset statistics matching Table I's columns: users, items,
    interactions, avg actions/user, avg actions/item, sparsity.
    """
    n_interactions = len(df)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_actions_per_user": round(n_interactions / n_users, 1),
        "avg_actions_per_item": round(n_interactions / n_items, 1),
        "sparsity_pct": round((1 - n_interactions / (n_users * n_items)) * 100, 3),
    }


def run_preprocessing(raw_ratings_path: Path, out_dir: Path) -> dict:
    """
    Full Phase 3 pipeline: load -> sort -> remap -> save -> return statistics.
    Refuses to run if out_dir already has processed .pkl output, per the
    "never silently overwrite processed data" rule.
    """
    from src.data.parser import load_ratings  # local import: avoids a hard import-time coupling

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.pkl"))
    if existing:
        raise FileExistsError(
            f"{out_dir} already has processed output ({[p.name for p in existing]}); "
            "delete it manually if you want to regenerate."
        )

    df = load_ratings(raw_ratings_path)
    df = sort_chronologically(df)
    df, user_map, item_map = remap_ids(df)

    df.to_pickle(out_dir / "interactions.pkl")
    with open(out_dir / "user_mapping.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(out_dir / "item_mapping.pkl", "wb") as f:
        pickle.dump(item_map, f)

    stats = compute_statistics(df, len(user_map), len(item_map))
    with open(out_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return stats
