from __future__ import annotations

import torch
from torch import nn


class MambaBlock(nn.Module):
    """
    DSRec Mamba-style sequence block.

    Paper-described flow:
        input
          -> linear projection
          -> 1D convolution
          -> SiLU
          -> state-space recurrence
          -> SiLU
          -> linear projection
          -> residual connection

    Input:
        x: [B, L, D]

    Output:
        [B, L, D]

    This implementation provides a PyTorch-only SSM-style recurrence because
    mamba_ssm is not installed in the current environment.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 32,
        conv_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.conv_width = conv_width
        self.expansion = expansion

        inner_dim = d_model * expansion
        self.inner_dim = inner_dim

        # W1 projection
        self.in_proj = nn.Linear(d_model, inner_dim)

        # Local temporal processing
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_width,
            padding=conv_width - 1,
            groups=inner_dim,
        )

        # Input-dependent SSM parameters.
        self.delta_proj = nn.Linear(inner_dim, inner_dim)
        self.b_proj = nn.Linear(inner_dim, d_state)
        self.c_proj = nn.Linear(inner_dim, d_state)

        # Learnable state dynamics.
        self.a_log = nn.Parameter(torch.zeros(inner_dim, d_state))

        # W2 projection
        self.out_proj = nn.Linear(inner_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight selective state-space recurrence.

        x:
            [B, L, inner_dim]

        returns:
            [B, L, inner_dim]
        """
        batch_size, seq_len, inner_dim = x.shape

        # Positive decay rate.
        a = -torch.exp(self.a_log)

        # Initial hidden state.
        state = torch.zeros(
            batch_size,
            inner_dim,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]

            # Input-dependent step size.
            delta = torch.sigmoid(self.delta_proj(xt))

            # Input-dependent B and C.
            b_t = self.b_proj(xt)
            c_t = self.c_proj(xt)

            # Discretized state update.
            decay = torch.exp(delta.unsqueeze(-1) * a)

            state = decay * state + (
                delta.unsqueeze(-1) * xt.unsqueeze(-1) * b_t.unsqueeze(1)
            )

            # Read state through C.
            yt = (state * c_t.unsqueeze(1)).sum(dim=-1)

            outputs.append(yt)

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, L, D]

            mask:
                [B, L], True for real sequence positions.

        Returns:
            [B, L, D]
        """
        if x.ndim != 3:
            raise ValueError(
                f"MambaBlock expected [B, L, D], got {tuple(x.shape)}"
            )

        residual = x

        # W1 projection + SiLU.
        h = self.in_proj(x)
        h = torch.nn.functional.silu(h)

        # Depthwise local convolution.
        # Conv1d expects [B, C, L].
        h = h.transpose(1, 2)
        h = self.conv1d(h)

        # Remove the extra right-padding position(s).
        h = h[..., : x.size(1)]
        h = h.transpose(1, 2)

        h = torch.nn.functional.silu(h)

        # Selective SSM.
        h = self._ssm(h)

        h = torch.nn.functional.silu(h)

        # W2 projection.
        h = self.out_proj(h)
        h = self.dropout(h)

        # Residual connection.
        output = residual + h

        # Padding positions must not carry sequence information.
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask must have shape {tuple(x.shape[:2])}, "
                    f"got {tuple(mask.shape)}"
                )

            output = output * mask.unsqueeze(-1).to(output.dtype)

        return output
