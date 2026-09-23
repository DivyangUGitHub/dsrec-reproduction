from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocessing import run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-1M for DSRec."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing ratings.dat",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for processed outputs",
    )

    args = parser.parse_args()

    ratings_path = args.raw_dir / "ratings.dat"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"ratings.dat not found at: {ratings_path}"
        )

    print(f"Loading ratings from: {ratings_path}")
    print(f"Writing processed data to: {args.out_dir}")

    stats = run_preprocessing(
        raw_ratings_path=ratings_path,
        out_dir=args.out_dir,
    )

    print("\nPreprocessing complete.")
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()