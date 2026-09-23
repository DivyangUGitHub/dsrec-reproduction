import numpy as np
import pandas as pd

from src.data.sequences import UserSequence, build_sequences
from src.data.split import (
    MIN_SEQUENCE_LENGTH,
    generate_training_examples,
    leave_one_out_split,
)


def test_build_sequences_orders_items_and_computes_gaps():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 40, 50],
            "timestamp": [100, 105, 130, 5, 8],
        }
    )
    sequences = build_sequences(df)

    assert set(sequences.keys()) == {1, 2}
    seq1 = sequences[1]
    assert list(seq1.items) == [10, 20, 30]
    assert list(seq1.time_gaps) == [0, 5, 25]  # Eq. 9: d_1=0, then successive diffs

    seq2 = sequences[2]
    assert list(seq2.items) == [40, 50]
    assert list(seq2.time_gaps) == [0, 3]


def _seq_from_items(items: list[int]) -> UserSequence:
    items_arr = np.array(items)
    timestamps = np.arange(len(items)) * 10  # evenly spaced, gap = 10 each
    gaps = np.zeros_like(timestamps)
    gaps[1:] = timestamps[1:] - timestamps[:-1]
    return UserSequence(user_id=1, items=items_arr, timestamps=timestamps, time_gaps=gaps)


def test_leave_one_out_split_matches_paper_ABCDE_example():
    # A=1, B=2, C=3, D=4, E=5 — the exact example from the project's Phase 5 spec.
    seq = _seq_from_items([1, 2, 3, 4, 5])
    split = leave_one_out_split(seq)

    assert list(split.train_context_items) == [1, 2, 3]  # A B C
    assert split.val_target == 4  # D
    assert split.test_target == 5  # E
    assert list(split.test_context_items) == [1, 2, 3, 4]  # A B C D


def test_leave_one_out_split_drops_too_short_sequences():
    for length in range(MIN_SEQUENCE_LENGTH):
        seq = _seq_from_items(list(range(length)))
        assert leave_one_out_split(seq) is None
    # exactly at the threshold it must succeed
    seq = _seq_from_items(list(range(MIN_SEQUENCE_LENGTH)))
    assert leave_one_out_split(seq) is not None


def test_generate_training_examples_final_only():
    seq = _seq_from_items([1, 2, 3, 4, 5])
    split = leave_one_out_split(seq)  # train_context = [1, 2, 3]
    examples = generate_training_examples(split, mode="final_only")
    assert len(examples) == 1
    assert list(examples[0].context_items) == [1, 2]
    assert examples[0].target == 3


def test_generate_training_examples_sliding_window():
    seq = _seq_from_items([1, 2, 3, 4, 5])
    split = leave_one_out_split(seq)  # train_context = [1, 2, 3]
    examples = generate_training_examples(split, mode="sliding_window")
    assert len(examples) == 2
    assert list(examples[0].context_items) == [1]
    assert examples[0].target == 2
    assert list(examples[1].context_items) == [1, 2]
    assert examples[1].target == 3


def test_generate_training_examples_empty_context_yields_nothing():
    # train_context has fewer than 2 items -> nothing to predict from.
    seq = _seq_from_items([1, 2, 3])  # train_context = [1], val=2, test=3
    split = leave_one_out_split(seq)
    assert generate_training_examples(split, mode="sliding_window") == []
