import torch

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k


def test_ranking_metrics_hit():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    target = torch.tensor([1])
    assert hit_rate_at_k(logits, target, 1) == 1.0
    assert ndcg_at_k(logits, target, 1) == 1.0
