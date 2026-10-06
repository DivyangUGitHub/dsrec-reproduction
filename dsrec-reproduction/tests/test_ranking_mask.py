import torch

from src.evaluation.ranking import mask_seen_items


def test_seen_items_are_masked_but_target_is_preserved():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    item_ids = torch.tensor([[1, 3, 0]])
    mask = torch.tensor([[True, True, False]])
    targets = torch.tensor([3])

    ranked = mask_seen_items(logits, item_ids, mask, targets)

    assert ranked[0, 1] < -1e20
    # The held-out target stays eligible even if it appears in context.
    assert ranked[0, 3] == logits[0, 3]
    assert ranked[0, 4] == logits[0, 4]
