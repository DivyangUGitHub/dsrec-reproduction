"""
DSRec main dual-interest model.

Phase 8:
- Long-term interest: historical aggregation + Mamba
- Short-term interest: item/time embeddings + time-aware SSM
- Residual cross-SSM fusion
- FFN + LayerNorm + dropout
- Last valid position -> MLP -> item embedding dot product
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mamba_block import MambaBlock
from src.models.short_term import ShortTermInterest


class FeedForward(nn.Module):
    """Paper Eq. 13/14: D -> 4D -> D with GELU."""

    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSRecBlock(nn.Module):
    """
    One residual-coupled dual-SSM block.

    The paper specifies that each branch receives the detached output
    of the other branch as an auxiliary residual connection.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_time_buckets: int = 10,
        d_state: int = 32,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.long_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            conv_width=conv_width,
            expansion=expansion,
            dropout=dropout,
        )

        self.short_ssm = ShortTermInterest(
            d_model=d_model,
            n_time_buckets=n_time_buckets,
            d_state=d_state,
            conv_width=conv_width,
            expansion=expansion,
            dropout=dropout,
        )

        self.long_norm = nn.LayerNorm(d_model)
        self.short_norm = nn.LayerNorm(d_model)

        self.long_ffn = FeedForward(d_model, dropout)
        self.short_ffn = FeedForward(d_model, dropout)

    def forward(
        self,
        long_x: torch.Tensor,
        short_x: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        long_out = self.long_mamba(long_x, mask)

        short_out = self.short_ssm(
            short_x,
            time_bucket_ids,
            mask,
        )

        # Residual cross-connection.
        # Detach prevents cross-branch gradient entanglement.
        long_out = long_out + short_out.detach()
        short_out = short_out + long_out.detach()

        long_out = self.long_norm(long_out)
        short_out = self.short_norm(short_out)

        long_out = long_out + self.long_ffn(long_out)
        short_out = short_out + self.short_ffn(short_out)

        return long_out, short_out


class DSRec(nn.Module):
    """
    Dual SSM Recommendation model.

    Input:
        item_ids:        [B, L]
        time_bucket_ids: [B, L]
        mask:            [B, L]

    Output:
        logits:          [B, n_items + 1]

    ID 0 is PAD_ID and is masked from prediction.
    """

    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_time_buckets: int = 10,
        n_blocks: int = 2,
        d_state: int = 32,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_items = n_items
        self.d_model = d_model

        # +1 because internal item ID 0 is PAD_ID.
        self.item_embedding = nn.Embedding(
            n_items + 1,
            d_model,
            padding_idx=0,
        )

        # Separate short-term projection.
        self.short_item_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.time_embedding = nn.Embedding(
            n_time_buckets,
            d_model,
        )

        self.short_input_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DSRecBlock(
                    d_model=d_model,
                    n_time_buckets=n_time_buckets,
                    d_state=d_state,
                    conv_width=conv_width,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Eq. 15: concatenate final long/short representations.
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _long_term_embedding(
        self,
        item_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. 5:

            x_i^l = x_i + mean(x_1 ... x_{i-1})

        For the first valid position, x_1^l = x_1.
        """

        x = self.item_embedding(item_ids)

        # Cumulative sum of previous embeddings.
        cumulative = torch.cumsum(x, dim=1)

        # Number of previous valid positions.
        counts = torch.cumsum(
            mask.long(),
            dim=1,
        ) - mask.long()

        previous_sum = cumulative - x

        denominator = counts.clamp_min(1).unsqueeze(-1)

        previous_mean = previous_sum / denominator

        long_x = x + previous_mean

        # First position has no previous history.
        first_position = counts == 0
        long_x = torch.where(
            first_position.unsqueeze(-1),
            x,
            long_x,
        )

        # Padding must not carry meaningful representation.
        long_x = long_x * mask.unsqueeze(-1)

        return self.input_dropout(long_x)

    def _short_term_embedding(
        self,
        item_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        item_x = self.item_embedding(item_ids)
        item_x = self.short_item_projection(item_x)

        time_x = self.time_embedding(time_bucket_ids)

        short_x = torch.cat(
            [item_x, time_x],
            dim=-1,
        )

        short_x = self.short_input_projection(short_x)

        short_x = short_x * mask.unsqueeze(-1)

        return short_x

    def forward(
        self,
        item_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        long_x = self._long_term_embedding(
            item_ids,
            mask,
        )

        short_x = self._short_term_embedding(
            item_ids,
            time_bucket_ids,
            mask,
        )

        for block in self.blocks:
            long_x, short_x = block(
                long_x,
                short_x,
                time_bucket_ids,
                mask,
            )

            long_x = long_x * mask.unsqueeze(-1)
            short_x = short_x * mask.unsqueeze(-1)

        # Find final valid position for every user.
        lengths = mask.long().sum(dim=1)
        last_idx = (lengths - 1).clamp_min(0)

        batch_idx = torch.arange(
            item_ids.size(0),
            device=item_ids.device,
        )

        long_last = long_x[
            batch_idx,
            last_idx,
        ]

        short_last = short_x[
            batch_idx,
            last_idx,
        ]

        # Eq. 15.
        output = self.output_mlp(
            torch.cat(
                [long_last, short_last],
                dim=-1,
            )
        )

        # Eq. 16: output embedding dot item embedding matrix.
        logits = output @ self.item_embedding.weight.T

        # PAD_ID=0 must never be predicted.
        logits[:, 0] = torch.finfo(logits.dtype).min

        return logits
