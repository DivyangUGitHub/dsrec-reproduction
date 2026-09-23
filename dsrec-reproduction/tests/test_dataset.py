import numpy as np
import pytest

torch = pytest.importorskip("torch")  # this whole file is skipped, not failed, if torch isn't installed

from src.data.collate import collate_batch  # noqa: E402
from src.data.dataset import PAD_ID, DSRecDataset, Example  # noqa: E402
from src.data.time_features import fit_time_buckets  # noqa: E402


def _bucketizer():
    return fit_time_buckets(np.array([1, 2, 3, 10, 20, 30]), n_buckets=4)


def test_short_sequence_is_right_padded_with_pad_id():
    ex = Example(context_items=np.array([5, 6, 7]), context_gaps=np.array([0, 1, 2]), target=8)
    ds = DSRecDataset([ex], bucketizer=_bucketizer(), max_len=6)
    item = ds[0]

    assert item["item_ids"].tolist() == [5, 6, 7, PAD_ID, PAD_ID, PAD_ID]
    assert item["mask"].tolist() == [True, True, True, False, False, False]
    assert item["target"].item() == 8


def test_long_context_is_truncated_to_most_recent_positions():
    ex = Example(
        context_items=np.array([1, 2, 3, 4, 5, 6, 7]),
        context_gaps=np.array([0, 1, 1, 1, 1, 1, 1]),
        target=8,
    )
    ds = DSRecDataset([ex], bucketizer=_bucketizer(), max_len=3)
    item = ds[0]
    # only the last 3 items should survive: 5, 6, 7
    assert item["item_ids"].tolist() == [5, 6, 7]
    assert item["mask"].tolist() == [True, True, True]


def test_pad_id_never_collides_with_a_real_item_id():
    # PAD_ID must be 0, and real item IDs (per src.data.preprocessing.IdMapping)
    # start at 1, so this is really a contract test between the two modules.
    assert PAD_ID == 0


def test_collate_batch_stacks_fields_to_B_L():
    ex1 = Example(context_items=np.array([1, 2]), context_gaps=np.array([0, 1]), target=3)
    ex2 = Example(context_items=np.array([4]), context_gaps=np.array([0]), target=5)
    ds = DSRecDataset([ex1, ex2], bucketizer=_bucketizer(), max_len=4)

    batch = collate_batch([ds[0], ds[1]])
    assert batch["item_ids"].shape == (2, 4)
    assert batch["mask"].shape == (2, 4)
    assert batch["target"].shape == (2,)
