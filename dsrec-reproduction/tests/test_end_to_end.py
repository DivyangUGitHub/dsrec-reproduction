import torch

from src.models.dsrec import DSRec
from src.evaluation.evaluator import evaluate_batch


def test_end_to_end_forward_and_metrics():
    model = DSRec(n_items=12, d_model=8, n_blocks=1, d_state=4, conv_width=3).eval()
    items = torch.randint(1, 13, (2, 4))
    times = torch.randint(0, 10, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        logits = model(items, times, mask)
    result = evaluate_batch(logits, torch.tensor([1, 2]), ks=(1, 3))
    assert set(result) == {"HR@1", "NDCG@1", "HR@3", "NDCG@3"}
