from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Inspect processed MovieLens data")
    p.add_argument("--processed-dir", default="data/processed")
    args = p.parse_args()
    root = Path(args.processed_dir)
    stats = root / "statistics.json"
    if stats.exists():
        print(json.dumps(json.loads(stats.read_text()), indent=2))
    interactions = root / "interactions.pkl"
    if interactions.exists():
        df = pd.read_pickle(interactions)
        print(f"rows={len(df):,}")
        print(df.head())


if __name__ == "__main__":
    main()
