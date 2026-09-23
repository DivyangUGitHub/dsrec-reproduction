import torch

from src.losses.losses import bpr_loss, cross_entropy_loss


def test_cross_entropy_finite():
    assert torch.isfinite(cross_entropy_loss(torch.randn(3, 5), torch.tensor([1, 2, 3])))


def test_bpr_finite():
    assert torch.isfinite(bpr_loss(torch.tensor([2.0]), torch.tensor([1.0])))
