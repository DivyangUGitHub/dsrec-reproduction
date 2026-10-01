import torch

from src.evaluation.metrics import mrr_at_k


def test_mrr_at_k_uses_reciprocal_rank():
    # Target 2 is ranked second for the first example and first for the second.
    logits = torch.tensor([
        [0.9, 0.8, 0.7, 0.1],
        [0.1, 0.2, 0.9, 0.3],
    ])
    targets = torch.tensor([1, 2])

    assert abs(mrr_at_k(logits, targets, k=3) - 0.75) < 1e-6


def test_mrr_at_k_returns_zero_for_miss():
    logits = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    targets = torch.tensor([3])

    assert mrr_at_k(logits, targets, k=2) == 0.0
