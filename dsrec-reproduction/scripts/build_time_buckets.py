from pathlib import Path

import pandas as pd
import numpy as np

from src.data.sequences import build_sequences
from src.data.split import leave_one_out_split
from src.data.time_features import fit_time_buckets, save_time_bucketizer


def main():
    processed_dir = Path("data/processed")
    output_path = processed_dir / "time_bucketizer.pkl"

    interactions_path = processed_dir / "interactions.pkl"

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Processed interactions not found: {interactions_path}"
        )

    print(f"Loading interactions from: {interactions_path}")
    df = pd.read_pickle(interactions_path)

    print("Building user sequences...")
    sequences = build_sequences(df)

    # Leave-one-out split:
    # train gaps are everything except validation and test interactions.
    train_gaps = []

    for seq in sequences.values():
        split = leave_one_out_split(seq)

        if split is None:
            continue

        train_gaps.extend(split.train_context_gaps.tolist())

    train_gaps = np.asarray(train_gaps, dtype=np.int64)

    print(f"Training gaps: {len(train_gaps)}")
    print(f"Min gap: {train_gaps.min()}")
    print(f"Max gap: {train_gaps.max()}")
    print(f"Zero gaps: {(train_gaps == 0).sum()}")

    print("Fitting 10 quantile time buckets...")
    bucketizer = fit_time_buckets(train_gaps, n_buckets=10)

    print("Bucket boundaries:")
    print(bucketizer.boundaries)

    save_time_bucketizer(bucketizer, output_path)

    print(f"Saved time bucketizer to: {output_path}")


if __name__ == "__main__":
    main()