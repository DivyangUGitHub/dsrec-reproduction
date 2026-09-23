from __future__ import annotations

import torch
from torch import nn

from src.models.mamba_block import MambaBlock


class ShortTermInterest(nn.Module):
    """
    DSRec short-term interest branch.

    Combines item representations with learned time-interval embeddings,
    then applies a time-dependent gate around the SSM/Mamba representation.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_time_buckets: int = 10,
        d_state: int = 32,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.time_embedding = nn.Embedding(
            n_time_buckets,
            d_model,
        )

        self.input_projection = nn.Linear(
            d_model * 2,
            d_model,
        )

        self.ssm = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            conv_width=conv_width,
            expansion=expansion,
            dropout=dropout,
        )

        self.time_gate = nn.Linear(
            d_model,
            d_model,
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_embeddings:
                [B, L, D]

            time_bucket_ids:
                [B, L]

            mask:
                [B, L], True for real positions.

        Returns:
            short-term representations [B, L, D]
        """
        if item_embeddings.ndim != 3:
            raise ValueError(
                "item_embeddings must have shape [B, L, D]"
            )

        if time_bucket_ids.shape != item_embeddings.shape[:2]:
            raise ValueError(
                "time_bucket_ids must have shape [B, L]"
            )

        if mask.shape != item_embeddings.shape[:2]:
            raise ValueError(
                "mask must have shape [B, L]"
            )

        # Learned time-interval representation.
        time_emb = self.time_embedding(time_bucket_ids)

        # Combine item + temporal representation.
        x = torch.cat(
            [item_embeddings, time_emb],
            dim=-1,
        )

        x = self.input_projection(x)
        x = self.layer_norm(x)

        # SSM representation.
        ssm_output = self.ssm(x, mask)

        # Time-dependent gate.
        gate = torch.sigmoid(self.time_gate(time_emb))

        # Causal recurrence:
        # h_t = gate_t * SSM(x_t) + (1-gate_t) * h_{t-1}
        outputs = []

        previous = torch.zeros(
            item_embeddings.size(0),
            self.d_model,
            device=item_embeddings.device,
            dtype=item_embeddings.dtype,
        )

        for t in range(item_embeddings.size(1)):
            current = (
                gate[:, t, :] * ssm_output[:, t, :]
                + (1.0 - gate[:, t, :]) * previous
            )

            current_mask = mask[:, t].unsqueeze(-1)

            current = torch.where(
                current_mask,
                current,
                torch.zeros_like(current),
            )

            outputs.append(current)
            previous = current

        return torch.stack(outputs, dim=1)
