from __future__ import annotations

import torch

from src.losses.losses import cross_entropy_loss


class Trainer:
    def __init__(self, model, optimizer, device="cpu", grad_clip_norm: float | None = 1.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip_norm = grad_clip_norm

    def train_epoch(self, loader, max_batches: int | None = None) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            item_ids = batch["item_ids"].to(self.device)
            time_ids = batch["time_bucket_ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["target"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(item_ids, time_ids, mask)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def validate(self, loader, max_batches: int | None = None) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            logits = self.model(batch["item_ids"].to(self.device), batch["time_bucket_ids"].to(self.device), batch["mask"].to(self.device))
            loss = cross_entropy_loss(logits, batch["target"].to(self.device))
            total += float(loss)
            count += 1
        return total / max(count, 1)
