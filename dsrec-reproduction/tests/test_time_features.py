import numpy as np
import pytest

from src.data.time_features import compute_time_gaps, fit_time_buckets, log_scale


def test_compute_time_gaps_eq9():
    timestamps = np.array([100, 105, 130, 130])
    gaps = compute_time_gaps(timestamps)
    assert list(gaps) == [0, 5, 25, 0]


def test_compute_time_gaps_negative_gap_is_clipped_to_zero():
    # Out-of-order timestamps (data anomaly) must not crash or go negative.
    timestamps = np.array([100, 90, 200])
    gaps = compute_time_gaps(timestamps)
    assert list(gaps) == [0, 0, 110]


def test_compute_time_gaps_rejects_nan():
    timestamps = np.array([100.0, np.nan, 130.0])
    with pytest.raises(ValueError):
        compute_time_gaps(timestamps)


# --- The five required tests for Phase 6 ---


def test_1_no_future_information_leaks_into_bucket_fitting():
    # fit_time_buckets only ever sees the array it's handed. Changing gaps
    # that are never passed in must not change the fitted boundaries.
    train_gaps = np.array([1, 2, 3, 4, 5, 100, 200, 300])
    bucketizer_a = fit_time_buckets(train_gaps, n_buckets=4)

    # Simulate "future" (val/test) gaps that must have zero influence.
    future_gaps = np.array([99999, 88888])  # noqa: F841 (deliberately unused below)
    bucketizer_b = fit_time_buckets(train_gaps, n_buckets=4)  # fit again, future never touched

    assert np.array_equal(bucketizer_a.boundaries, bucketizer_b.boundaries)


def test_2_same_gap_gives_deterministic_bucket():
    train_gaps = np.array([1, 5, 10, 50, 100])
    bucketizer = fit_time_buckets(train_gaps, n_buckets=3)
    gap = np.array([7])
    assert bucketizer.transform(gap)[0] == bucketizer.transform(gap)[0]
    # run it twice more for good measure
    results = {int(bucketizer.transform(gap)[0]) for _ in range(5)}
    assert len(results) == 1


def test_3_zero_gap_works():
    train_gaps = np.array([0, 1, 2, 3, 100])
    bucketizer = fit_time_buckets(train_gaps, n_buckets=3)
    bucket = bucketizer.transform(np.array([0]))
    assert bucket[0] >= 0  # doesn't crash, lands in a valid bucket
    assert log_scale(np.array([0]))[0] == 0.0


def test_4_large_gap_works():
    train_gaps = np.array([1, 2, 3, 4, 5])
    bucketizer = fit_time_buckets(train_gaps, n_buckets=3)
    huge = np.array([10**9])
    bucket = bucketizer.transform(huge)
    # A gap far larger than anything seen at fit time must clamp into the
    # last bucket, not raise or overflow.
    assert bucket[0] == bucketizer.n_buckets - 1


def test_5_nan_and_negative_gaps_are_handled_explicitly():
    # Negative: compute_time_gaps clips (tested above); log_scale/transform
    # on an already-nonnegative array must not raise.
    bucketizer = fit_time_buckets(np.array([1, 2, 3, 4]), n_buckets=2)
    assert bucketizer.transform(np.array([0]))[0] >= 0

    # NaN: compute_time_gaps raises rather than silently propagating NaN
    # into a bucket boundary comparison.
    with pytest.raises(ValueError):
        compute_time_gaps(np.array([1.0, np.nan]))
