
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_N_BUCKETS = 10  # [ASSUMPTION] not specified in the paper; docs/paper_audit.md item 12


def compute_time_gaps(timestamps: np.ndarray) -> np.ndarray:
    """
    Eq. 9: d_1 = 0, d_i = t_i - t_{i-1} for i > 1.
    timestamps: [T] -> returns [T], same dtype family (int64).

    Negative gaps (out-of-order timestamps — a data anomaly, not a modeling
    choice) are clipped to 0 [ASSUMPTION]. NaN timestamps raise instead of
    being silently dropped, since a NaN here means a bug upstream in
    parsing, not a real gap to model.
    """
    if np.isnan(timestamps.astype(np.float64)).any():
        raise ValueError("NaN timestamp encountered; fix upstream parsing, don't paper over it here.")
    gaps = np.empty_like(timestamps, dtype=np.int64)
    gaps[0] = 0
    gaps[1:] = timestamps[1:] - timestamps[:-1]
    negative = gaps < 0
    if negative.any():
        gaps[negative] = 0
    return gaps


def log_scale(gaps: np.ndarray) -> np.ndarray:
    """log1p, so a zero gap maps to 0.0 instead of -inf."""
    return np.log1p(gaps.astype(np.float64))


@dataclass
class TimeBucketizer:
    """Quantile bin boundaries fit on training-split log-scaled gaps."""

    boundaries: np.ndarray  # [n_buckets - 1], interior quantile cut points
    n_buckets: int

    def transform(self, gaps: np.ndarray) -> np.ndarray:
        """Map raw gaps to bucket IDs in [0, n_buckets - 1], deterministically."""
        log_gaps = log_scale(gaps)
        return np.searchsorted(self.boundaries, log_gaps, side="right")

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "TimeBucketizer":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_time_buckets(train_gaps: np.ndarray, n_buckets: int = DEFAULT_N_BUCKETS) -> TimeBucketizer:
    """
    Fit quantile boundaries on `train_gaps` ONLY — see module docstring.
    Caller is responsible for making sure this array excludes validation
    and test gaps.
    """
    log_gaps = log_scale(train_gaps)
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]  # interior cut points only
    boundaries = np.quantile(log_gaps, quantiles)
    return TimeBucketizer(boundaries=boundaries, n_buckets=n_buckets)


# Thin aliases matching the exact function names named in the project plan.
def save_time_bucketizer(bucketizer: TimeBucketizer, path: Path) -> None:
    bucketizer.save(path)


def load_time_bucketizer(path: Path) -> TimeBucketizer:
    return TimeBucketizer.load(path)


def transform_time_gaps(bucketizer: TimeBucketizer, gaps: np.ndarray) -> np.ndarray:
    return bucketizer.transform(gaps)
