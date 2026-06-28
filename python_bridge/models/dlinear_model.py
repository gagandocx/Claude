"""
=============================================================
  Python ML Bridge - DLinear
  "Are Transformers Effective for Time Series Forecasting?"
  Zeng et al., AAAI 2023 — https://arxiv.org/abs/2205.13504

  Why DLinear belongs in a 12-model ensemble:
  ────────────────────────────────────────────
  DLinear is deceptively simple — and that's exactly the point.
  It decomposes the input into TREND and RESIDUAL components,
  then applies a plain linear layer to each.

  In large ensembles, simple models add DIVERSITY:
    • All 11 other models learn complex non-linear patterns
    • DLinear captures the raw linear trend of each indicator
    • When complex models overfit noise, DLinear stays right
    • The meta-learner learns when to trust DLinear vs others

  Decomposition:
    trend    = moving_average(x, kernel=25)   — 25-bar smoothed
    residual = x - trend                      — mean-reverting component

  Both components are processed with channel-independent linear
  layers (one linear per feature, shared weights across batch):
    trend_out    : Linear(seq_len → 1) per feature  → (batch, 46)
    residual_out : Linear(seq_len → 1) per feature  → (batch, 46)

  Combined: (batch, 92) → MLP classifier → (batch, 3)

  "Channel-independent" means each feature is projected separately —
  no cross-feature mixing in the linear stage (iTransformer handles that).
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DLinearConfig


# ─────────────────────────────────────────────
#  MOVING AVERAGE DECOMPOSITION
# ─────────────────────────────────────────────
class SeriesDecomposition(nn.Module):
    """
    Decomposes a time series into trend (moving average) + residual.
    Uses a symmetric 1D average pooling kernel over the time dimension.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # Padding to keep output length == input length
        padding = kernel_size // 2
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=padding
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_features)
        returns: trend (batch, seq_len, n_features),
                 residual (batch, seq_len, n_features)
        """
        # AvgPool1d expects (batch, channels, length)
        x_t     = x.transpose(1, 2)                     # (B, F, L)
        trend_t = self.avg(x_t)[:, :, :x.shape[1]]     # trim to original L
        trend   = trend_t.transpose(1, 2)               # (B, L, F)
        return trend, x - trend


# ─────────────────────────────────────────────
#  MARKET DLINEAR
# ─────────────────────────────────────────────
class MarketDLinear(nn.Module):
    """
    DLinear for market direction classification.

    Channel-independent linear projections on trend and residual
    components, then a shared MLP classifier on the combined output.
    """

    def __init__(self, config: Optional[DLinearConfig] = None):
        super().__init__()
        self.config = config or DLinearConfig()
        F_in = self.config.input_features
        L    = self.config.seq_length

        # Decomposition
        self.decomp = SeriesDecomposition(kernel_size=self.config.kernel_size)

        # Channel-independent linear: same Linear(L→1) applied to each feature
        # Implemented as Linear(L, F_in) applied on transposed input
        self.trend_linear    = nn.Linear(L, F_in)   # (B, F, L) → (B, F, F) then take diagonal
        self.residual_linear = nn.Linear(L, F_in)

        # Simpler and equivalent: one linear per component applied along seq dimension
        # We use weight_norm for stability
        from torch.nn.utils import weight_norm
        self.trend_proj    = weight_norm(nn.Linear(L, 1))
        self.residual_proj = weight_norm(nn.Linear(L, 1))

        # MLP classifier: (batch, F_in * 2) → (batch, 3)
        hidden = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(F_in * 2, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden // 2, self.config.num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(F_in * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        returns: (batch, features*2)  — [trend_repr | residual_repr]
        """
        trend, residual = self.decomp(x)   # each (B, L, F)

        # Apply channel-independent linear: project seq_len → scalar per feature
        # trend:    (B, L, F) → transpose → (B, F, L) → linear(L,1) → (B, F, 1) → squeeze
        trend_out    = self.trend_proj(
            trend.transpose(1, 2)           # (B, F, L)
        ).squeeze(-1)                       # (B, F)

        residual_out = self.residual_proj(
            residual.transpose(1, 2)
        ).squeeze(-1)                       # (B, F)

        return torch.cat([trend_out, residual_out], dim=-1)   # (B, F*2)

    # ── public API ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._encode(x)
        return F.softmax(self.classifier(enc), dim=-1), self.confidence_head(enc)
