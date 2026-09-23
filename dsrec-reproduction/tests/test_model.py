import torch

from src.models.dsrec import DSRec


def test_dsrec_forward_shape():
    model = DSRec(n_items=20, d_model=8, n_blocks=1, d_state=4, conv_width=3)
    items = torch.randint(1, 21, (2, 5))
    times = torch.randint(0, 10, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = model(items, times, mask)
    assert logits.shape == (2, 21)
    assert torch.isneginf(logits[:, 0]).all() or (logits[:, 0] < -1e20).all()
