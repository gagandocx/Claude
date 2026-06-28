"""
=============================================================
  Python ML Bridge - TimeMixer
  "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
  Wang et al., ICLR 2024 — https://arxiv.org/abs/2405.14616

  Core idea — Past Decomposable Mixing (PDM):
  ─────────────────────────────────────────────────────────────
  Financial time series have overlapping rhythms — scalping M1
  gold has micro-moves (1-4 bars), short swings (8-16 bars), and
  session-level trends (32-64 bars) all active simultaneously.

  TimeMixer explicitly handles this by:
    1. Generating representations at multiple temporal scales:
         Scale₀ = x                         (64 bars — micro)
         Scale₁ = AvgPool(x, kernel=2)      (32 bars — short)
         Scale₂ = AvgPool(x, kernel=4)      (16 bars — mid)
         Scale₃ = AvgPool(x, kernel=8)      ( 8 bars — macro)

    2. Applying a MixerBlock at each scale:
         Temporal mixer  — MLP over the TIME dimension
         Channel mixer   — MLP over the FEATURE dimension
         This captures both "what happened over these bars"
         and "how do these 46 indicators interact at this scale"

    3. Bottom-up mixing — fine scale informs coarse scale:
         coarse_final = coarse + upsample(fine_output)
         Both directions learn; neither dominates.

    4. Aggregate all scale outputs → classifier

  What TimeMixer gives that no other model does:
    N-HiTS: separate pooled windows, no cross-scale mixing
    PatchTST: patches at ONE scale only
    TimeMixer: MIXED representations where scales inform EACH OTHER
=============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TimeMixerConfig


# ─────────────────────────────────────────────
#  DECOMPOSITION
# ─────────────────────────────────────────────
class MovingAvgDecomp(nn.Module):
    """Decompose input into trend (moving avg) and seasonal (residual)."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1,
                                padding=kernel // 2)

    def forward(self, x: torch.Tensor):
        """x: (B, L, F) → trend (B, L, F), seasonal (B, L, F)"""
        x_t = x.transpose(1, 2)                    # (B, F, L)
        trend_t = self.avg(x_t)[:, :, :x.shape[1]] # trim to L
        trend = trend_t.transpose(1, 2)
        return trend, x - trend


# ─────────────────────────────────────────────
#  MIXER BLOCK
# ─────────────────────────────────────────────
class MixerBlock(nn.Module):
    """
    Applies temporal mixing (across time steps) then channel mixing
    (across feature dimensions) with residual connections.
    Operates on a fixed sequence length L.
    """

    def __init__(self, seq_len: int, n_features: int,
                 hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(n_features)

        # MLP over TIME dimension (applied to transposed input)
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
            nn.Dropout(dropout),
        )
        # MLP over FEATURE dimension
        self.channel = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) → (B, L, F)"""
        # Temporal mixing: treat (B, F, L), apply MLP on L dim
        x = x + self.temporal(
            self.norm1(x.transpose(1, 2))
        ).transpose(1, 2)
        # Channel mixing: apply MLP on F dim
        x = x + self.channel(self.norm2(x))
        return x


# ─────────────────────────────────────────────
#  MARKET TIMEMIXER
# ─────────────────────────────────────────────
class MarketTimeMixer(nn.Module):
    """
    TimeMixer for market direction classification.
    Uses Past Decomposable Mixing (PDM) to process the price sequence
    simultaneously at 4 temporal scales and mix between them.
    """

    def __init__(self, config: Optional[TimeMixerConfig] = None):
        super().__init__()
        self.config = config or TimeMixerConfig()
        L = self.config.seq_length
        F_in = self.config.input_features
        H = self.config.hidden_size

        # Decomposition module
        self.decomp = MovingAvgDecomp(kernel=self.config.decomp_kernel)

        # Down-sampling to get multi-scale inputs
        # pool_sizes[i]: downsample factor for scale i
        pool_sizes = self.config.pool_sizes  # e.g. [1, 2, 4, 8]
        self.scales = pool_sizes
        self.n_scales = len(pool_sizes)

        # Compute sequence lengths at each scale
        self.scale_lens = [max(1, L // ps) for ps in pool_sizes]

        # MixerBlock for trend and seasonal at each scale
        self.trend_mixers = nn.ModuleList([
            MixerBlock(sl, F_in, H, self.config.dropout)
            for sl in self.scale_lens
        ])
        self.seasonal_mixers = nn.ModuleList([
            MixerBlock(sl, F_in, H, self.config.dropout)
            for sl in self.scale_lens
        ])

        # Bottom-up mixing: project coarser scale up to finer scale length
        self.up_projs = nn.ModuleList([
            nn.Linear(self.scale_lens[i + 1], self.scale_lens[i])
            for i in range(self.n_scales - 1)
        ])

        # Final aggregation: mean-pool each scale output, concat, classify
        self.norm = nn.LayerNorm(F_in)
        agg_dim = F_in * self.n_scales

        self.classifier = nn.Sequential(
            nn.Linear(agg_dim, H),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(H // 2, self.config.num_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(agg_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _downsample(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """Average pool x from (B, L, F) to (B, L//pool, F)."""
        if pool_size == 1:
            return x
        x_t = x.transpose(1, 2)             # (B, F, L)
        pooled = F.avg_pool1d(x_t, kernel_size=pool_size, stride=pool_size)
        return pooled.transpose(1, 2)       # (B, L//pool, F)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) → (B, F * n_scales)"""
        # 1. Generate multi-scale inputs
        scales_x = [self._downsample(x, ps) for ps in self.scales]

        # 2. Decompose and mix at each scale
        trend_outs    = []
        seasonal_outs = []
        for i, (xs, tmixer, smixer) in enumerate(
            zip(scales_x, self.trend_mixers, self.seasonal_mixers)
        ):
            trend, seasonal = self.decomp(xs)
            trend_outs.append(tmixer(trend))
            seasonal_outs.append(smixer(seasonal))

        # 3. Bottom-up mixing (finest → coarsest direction)
        for i in range(self.n_scales - 2, -1, -1):
            # Upsample coarser (i+1) to match finer (i) length
            coarser_t = trend_outs[i + 1].transpose(1, 2)       # (B, F, L_coarse)
            up_t = self.up_projs[i](coarser_t).transpose(1, 2)  # (B, L_fine, F)
            trend_outs[i] = trend_outs[i] + up_t

            coarser_s = seasonal_outs[i + 1].transpose(1, 2)
            up_s = self.up_projs[i](coarser_s).transpose(1, 2)
            seasonal_outs[i] = seasonal_outs[i] + up_s

        # 4. Aggregate: mean-pool each scale, concat all scales
        aggs = []
        for i in range(self.n_scales):
            combined = trend_outs[i] + seasonal_outs[i]  # (B, L_i, F)
            aggs.append(self.norm(combined).mean(dim=1))  # (B, F)

        return torch.cat(aggs, dim=-1)  # (B, F * n_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._encode(x)
        return F.softmax(self.classifier(enc), dim=-1), self.confidence_head(enc)
