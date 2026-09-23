import torch

from src.models.mamba_block import MambaBlock


def test_prefix_causality():
    torch.manual_seed(0)
    model = MambaBlock(d_model=8, d_state=4, conv_width=3, expansion=2).eval()
    prefix = torch.randn(1, 3, 8)
    suffix = torch.randn(1, 2, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        y1 = model(torch.cat([prefix, suffix], 1), mask)
        altered = suffix.clone(); altered[:, :, :] += 10
        y2 = model(torch.cat([prefix, altered], 1), mask)
    assert torch.allclose(y1[:, :2], y2[:, :2], atol=1e-5)
