from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Inspect a user's interaction sequence")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--user-id", type=int, default=None)
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    df = pd.read_pickle(Path(args.processed_dir) / "interactions.pkl")
    if args.user_id is not None:
        df = df[df["user_id"] == args.user_id]
    print(df.sort_values(["user_id", "timestamp"]).head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
