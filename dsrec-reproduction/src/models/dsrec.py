"""
DSRec main dual-interest model.

Supports the baseline dual-interest architecture plus the four configured
ablation modes used by the reproduction experiments:
- no_cross_fusion
- no_dual_interest
- no_short_ssm
- dual_mamba
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
    """One configurable dual-interest SSM block."""

    def __init__(
        self,
        d_model: int = 64,
        n_time_buckets: int = 10,
        d_state: int = 32,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.2,
        cross_fusion: bool = True,
        dual_interest: bool = True,
        short_ssm: bool = True,
        long_branch: str = "mamba",
        short_branch: str = "time_aware_ssm",
    ):
        super().__init__()

        if long_branch != "mamba":
            raise ValueError(f"Unsupported long_branch: {long_branch}")
        if short_branch not in {"time_aware_ssm", "mamba"}:
            raise ValueError(f"Unsupported short_branch: {short_branch}")

        self.cross_fusion = cross_fusion
        self.dual_interest = dual_interest
        self.use_short_ssm = short_ssm
        self.short_branch = short_branch

        # Keep the baseline attribute name `short_ssm` so existing baseline
        # checkpoints remain loadable. The new boolean is `use_short_ssm`.
        self.long_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            conv_width=conv_width,
            expansion=expansion,
            dropout=dropout,
        )

        if dual_interest:
            if short_ssm and short_branch == "time_aware_ssm":
                self.short_ssm = ShortTermInterest(
                    d_model=d_model,
                    n_time_buckets=n_time_buckets,
                    d_state=d_state,
                    conv_width=conv_width,
                    expansion=expansion,
                    dropout=dropout,
                )
            elif short_ssm and short_branch == "mamba":
                self.short_mamba = MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    conv_width=conv_width,
                    expansion=expansion,
                    dropout=dropout,
                )
            else:
                self.short_identity = nn.Identity()

        self.long_norm = nn.LayerNorm(d_model)
        self.long_ffn = FeedForward(d_model, dropout)

        if dual_interest:
            self.short_norm = nn.LayerNorm(d_model)
            self.short_ffn = FeedForward(d_model, dropout)

    def forward(
        self,
        long_x: torch.Tensor,
        short_x: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        long_out = self.long_mamba(long_x, mask)

        if not self.dual_interest:
            long_out = self.long_norm(long_out)
            long_out = long_out + self.long_ffn(long_out)
            return long_out, short_x

        if not self.use_short_ssm:
            short_out = self.short_identity(short_x)
        elif self.short_branch == "mamba":
            short_out = self.short_mamba(short_x, mask)
        else:
            short_out = self.short_ssm(
                short_x,
                time_bucket_ids,
                mask,
            )

        if self.cross_fusion:
            # Residual cross-connection. Detach prevents cross-branch
            # gradient entanglement, matching the baseline implementation.
            long_out = long_out + short_out.detach()
            short_out = short_out + long_out.detach()

        long_out = self.long_norm(long_out)
        short_out = self.short_norm(short_out)

        long_out = long_out + self.long_ffn(long_out)
        short_out = short_out + self.short_ffn(short_out)

        return long_out, short_out


class DSRec(nn.Module):
    """Dual SSM Recommendation model with configurable ablations."""

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
        cross_fusion: bool = True,
        dual_interest: bool = True,
        short_ssm: bool = True,
        long_branch: str = "mamba",
        short_branch: str = "time_aware_ssm",
    ):
        super().__init__()

        self.n_items = n_items
        self.d_model = d_model
        self.cross_fusion = cross_fusion
        self.dual_interest = dual_interest
        self.short_ssm = short_ssm
        self.long_branch = long_branch
        self.short_branch = short_branch

        self.item_embedding = nn.Embedding(
            n_items + 1,
            d_model,
            padding_idx=0,
        )

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
                    cross_fusion=cross_fusion,
                    dual_interest=dual_interest,
                    short_ssm=short_ssm,
                    long_branch=long_branch,
                    short_branch=short_branch,
                )
                for _ in range(n_blocks)
            ]
        )

        output_input_dim = 2 * d_model if dual_interest else d_model
        self.output_mlp = nn.Sequential(
            nn.Linear(output_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _long_term_embedding(
        self,
        item_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        cumulative = torch.cumsum(x, dim=1)
        counts = torch.cumsum(mask.long(), dim=1) - mask.long()
        previous_sum = cumulative - x
        denominator = counts.clamp_min(1).unsqueeze(-1)
        previous_mean = previous_sum / denominator
        long_x = x + previous_mean

        first_position = counts == 0
        long_x = torch.where(
            first_position.unsqueeze(-1),
            x,
            long_x,
        )
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
        short_x = torch.cat([item_x, time_x], dim=-1)
        short_x = self.short_input_projection(short_x)
        return short_x * mask.unsqueeze(-1)

    def forward(
        self,
        item_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        long_x = self._long_term_embedding(item_ids, mask)
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

        lengths = mask.long().sum(dim=1)
        last_idx = (lengths - 1).clamp_min(0)
        batch_idx = torch.arange(
            item_ids.size(0),
            device=item_ids.device,
        )

        long_last = long_x[batch_idx, last_idx]

        if self.dual_interest:
            short_last = short_x[batch_idx, last_idx]
            output_input = torch.cat([long_last, short_last], dim=-1)
        else:
            output_input = long_last

        output = self.output_mlp(output_input)
        logits = output @ self.item_embedding.weight.T
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits
