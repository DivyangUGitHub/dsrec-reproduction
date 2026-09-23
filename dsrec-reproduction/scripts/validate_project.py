from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"MISSING: {path.relative_to(ROOT)}")


def main() -> None:
    required = [
        ROOT / "data/processed/interactions.pkl",
        ROOT / "data/processed/time_bucketizer.pkl",
        ROOT / "data/processed/user_mapping.pkl",
        ROOT / "data/processed/item_mapping.pkl",
        ROOT / "data/checkpoints/best.pt",
        ROOT / "src/models/dsrec.py",
        ROOT / "src/api/app.py",
    ]
    for path in required:
        require(path)

    df = pd.read_pickle(ROOT / "data/processed/interactions.pkl")
    if df.empty or not {"user_id", "item_id", "timestamp"}.issubset(df.columns):
        raise SystemExit("INVALID: interactions.pkl")

    with open(ROOT / "data/processed/user_mapping.pkl", "rb") as f:
        user_mapping = pickle.load(f)
    with open(ROOT / "data/processed/item_mapping.pkl", "rb") as f:
        item_mapping = pickle.load(f)
    if len(user_mapping) == 0 or len(item_mapping) == 0:
        raise SystemExit("INVALID: empty ID mapping")

    checkpoint = torch.load(ROOT / "data/checkpoints/best.pt", map_location="cpu", weights_only=False)
    for key in ("epoch", "model_state_dict", "optimizer_state_dict", "train_loss", "val_loss"):
        if key not in checkpoint:
            raise SystemExit(f"INVALID: checkpoint missing {key}")
    if not torch.isfinite(torch.tensor(float(checkpoint["train_loss"]))) or not torch.isfinite(torch.tensor(float(checkpoint["val_loss"]))):
        raise SystemExit("INVALID: non-finite checkpoint loss")

    print("DSRec project validation: PASS")
    print(f"interactions={len(df):,}")
    print(f"users={len(user_mapping):,}")
    print(f"items={len(item_mapping):,}")
    print(f"checkpoint_epoch={checkpoint['epoch']}")
    print(f"checkpoint_val_loss={checkpoint['val_loss']:.6f}")


if __name__ == "__main__":
    main()
