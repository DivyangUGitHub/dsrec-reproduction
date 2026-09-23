import pytest
import torch

from src.models.dsrec import DSRec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cross_fusion": False},
        {"dual_interest": False},
        {"short_ssm": False},
        {"long_branch": "mamba", "short_branch": "mamba"},
    ],
    ids=[
        "no_cross_fusion",
        "no_dual_interest",
        "no_short_ssm",
        "dual_mamba",
    ],
)
def test_ablation_forward_shape(kwargs):
    model = DSRec(
        n_items=20,
        d_model=8,
        n_blocks=1,
        d_state=4,
        conv_width=3,
        **kwargs,
    )

    items = torch.randint(1, 21, (2, 5))
    times = torch.randint(0, 10, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)

    logits = model(items, times, mask)

    assert logits.shape == (2, 21)
    assert torch.isneginf(logits[:, 0]).all() or (logits[:, 0] < -1e20).all()
