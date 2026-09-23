from __future__ import annotations

import argparse
import time

import torch

from src.models.dsrec import DSRec


def main():
    p = argparse.ArgumentParser(description="Benchmark DSRec forward-pass throughput")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--items", type=int, default=3706)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    model = DSRec(n_items=args.items).eval()
    item_ids = torch.randint(1, args.items + 1, (args.batch_size, args.length))
    times = torch.randint(0, 10, (args.batch_size, args.length))
    mask = torch.ones_like(item_ids, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(item_ids, times, mask)
        start = time.perf_counter()
        for _ in range(args.steps):
            model(item_ids, times, mask)
        elapsed = time.perf_counter() - start
    print(f"device=cpu | steps={args.steps} | elapsed={elapsed:.3f}s | batches/s={args.steps/elapsed:.2f}")


if __name__ == "__main__":
    main()
