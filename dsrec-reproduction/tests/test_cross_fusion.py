import torch

from src.models.cross_fusion import CrossFusion


def test_cross_fusion_shape():
    module = CrossFusion(8)
    a, b = module(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    assert a.shape == (2, 4, 8)
    assert b.shape == (2, 4, 8)
