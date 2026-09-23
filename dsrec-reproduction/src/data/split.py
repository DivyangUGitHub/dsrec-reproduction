from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.data.sequences import UserSequence

MIN_SEQUENCE_LENGTH = 3  # need >=1 training-context item + val target + test target


@dataclass
class LeaveOneOutSplit:
    train_context_items: np.ndarray
    train_context_gaps: np.ndarray
    val_context_items: np.ndarray
    val_context_gaps: np.ndarray
    val_target: int
    test_context_items: np.ndarray
    test_context_gaps: np.ndarray
    test_target: int


def leave_one_out_split(seq: UserSequence) -> LeaveOneOutSplit | None:
    """
    Split one user's sequence per Sec. IV-A.4.

    Example matching the paper's own phrasing, for items [A, B, C, D, E]:
        train_context = [A, B, C]
        val_target    = D   (val_context = [A, B, C])
        test_target   = E   (test_context = [A, B, C, D])

    Returns None if the sequence is too short to yield a non-empty training
    context (< MIN_SEQUENCE_LENGTH interactions) — such users are dropped
    outright, never padded with fake history to force them to fit.
    """
    T = len(seq.items)
    if T < MIN_SEQUENCE_LENGTH:
        return None

    return LeaveOneOutSplit(
        train_context_items=seq.items[:-2],
        train_context_gaps=seq.time_gaps[:-2],
        val_context_items=seq.items[:-2],
        val_context_gaps=seq.time_gaps[:-2],
        val_target=int(seq.items[-2]),
        test_context_items=seq.items[:-1],
        test_context_gaps=seq.time_gaps[:-1],
        test_target=int(seq.items[-1]),
    )


@dataclass
class TrainingExample:
    context_items: np.ndarray
    context_gaps: np.ndarray
    target: int


def generate_training_examples(
    split: LeaveOneOutSplit,
    mode: Literal["sliding_window", "final_only"] = "sliding_window",
) -> list[TrainingExample]:
    """
    Turn a user's training context into (context, target) pairs.

    "final_only": one example — predict the last context item from
    everything before it. The most literal reading of the Phase 5 example,
    which only labels a training *context* and never an explicit target.

    "sliding_window": one example per position — predict context[i] from
    context[:i], for every i >= 1. Standard practice in SASRec-style
    training. [ASSUMPTION], not stated in the paper — default, per
    docs/paper_audit.md item 5.
    """
    items = split.train_context_items
    gaps = split.train_context_gaps
    n = len(items)
    if n < 2:
        return []  # nothing to predict from an empty or single-item context

    if mode == "final_only":
        return [TrainingExample(items[:-1], gaps[:-1], int(items[-1]))]

    if mode == "sliding_window":
        return [TrainingExample(items[:i], gaps[:i], int(items[i])) for i in range(1, n)]

    raise ValueError(f"Unknown mode: {mode}")
