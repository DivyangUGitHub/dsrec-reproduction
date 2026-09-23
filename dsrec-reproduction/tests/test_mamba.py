import torch

from src.models.mamba_block import MambaBlock


def test_mamba_shape_and_mask():
    model = MambaBlock(d_model=8, d_state=4, conv_width=3, expansion=2)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    y = model(x, mask)
    assert y.shape == x.shape
    assert torch.allclose(y[0, 3:], torch.zeros_like(y[0, 3:]))
