"""
=============================================================
  Python ML Bridge - TimesNet
  "TimesNet: Temporal 2D-Variation Modeling for General Time Series"
  Wu et al., ICLR 2023 — https://arxiv.org/abs/2210.02186

  Why TimesNet is unique in the 14-model stack:
  ──────────────────────────────────────────────
  Every other model (Transformer, LSTM, TCN, Mamba, xLSTM…) treats
  price data as a 1D sequence:  [bar₁, bar₂, …, bar₆₄]

  TimesNet discovers that time series have MULTI-PERIODICITY — multiple
  overlapping cycles at different frequencies (e.g. 4-bar, 8-bar,
  16-bar rhythms in gold M1). It:

    1. Finds dominant periods via FFT amplitude spectrum
    2. Reshapes the 1D sequence into a 2D "image" for each period T:
         [ bar₁  bar₂  … bar_T  ]   ← one cycle (intraperiod variation)
         [ bar_T+1   …  bar_2T  ]   ← next cycle (interperiod variation)
         [ …                    ]
    3. Applies 2D Inception-style convolutions (vertical = same phase
       across cycles, horizontal = within one cycle)
    4. Aggregates across all discovered periods

  The 2D convolutions capture what no 1D model can:
    • HORIZONTAL: "At bar 4 of each 8-bar cycle, gold tends to reverse"
    • VERTICAL:   "The pattern in the last cycle repeats the one before"
    • DIAGONAL:   "Momentum from mid-cycle carries to next cycle start"

  Architecture:
    Input (batch, 64, 46)
    → Linear embedding → (batch, 64, d_model)
    → FFT period discovery → top-k periods [T₁, T₂, T₃]
    → For each Tᵢ: 1D→2D reshape → InceptionBlock2D → 2D→1D reshape
    → Sum across k periods + residual
    → LayerNorm → global avg pool → Classifier → (batch, 3)
=============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TimesNetConfig


# ─────────────────────────────────────────────
#  2D INCEPTION BLOCK
# ─────────────────────────────────────────────
class InceptionBlock2D(nn.Module):
    """
    Inception-style 2D convolution block with three parallel branches:
      Branch A — (1,3): captures within-cycle patterns (horizontal)
      Branch B — (3,1): captures same-phase cross-cycle patterns (vertical)
      Branch C — (3,3): captures diagonal / mixed patterns

    All branches concatenated → 1×1 conv back to d_model channels.
    BatchNorm + GELU throughout for stable training.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        branch_ch = max(d_model // 3, 8)   # channels per branch

        self.branch_h = nn.Sequential(
            nn.Conv2d(d_model, branch_ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(d_model, branch_ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )
        self.branch_d = nn.Sequential(
            nn.Conv2d(d_model, branch_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(branch_ch * 3, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, rows, cols) → (B, d_model, rows, cols)"""
        h = self.branch_h(x)
        v = self.branch_v(x)
        d = self.branch_d(x)
        return self.merge(torch.cat([h, v, d], dim=1))


# ─────────────────────────────────────────────
#  TIMES BLOCK (one period)
# ─────────────────────────────────────────────
class TimesBlock(nn.Module):
    """
    Processes one dominant period T:
      1D sequence → 2D grid (rows × cols = ceil(L/T) × T) → 2D conv → 1D
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.inception = InceptionBlock2D(d_model, dropout)

    def forward(
        self, x: torch.Tensor, period: int
    ) -> torch.Tensor:
        """
        x:      (B, L, d_model)
        period: int T — reshape period
        returns (B, L, d_model)
        """
        B, L, d = x.shape
        rows = math.ceil(L / period)        # number of "cycles"
        L_pad = rows * period               # padded length

        # Pad sequence if needed
        if L_pad > L:
            pad = x[:, :L_pad - L].detach() * 0   # zero-pad
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x

        # Reshape: (B, rows, period, d) → (B, d, rows, period) for Conv2d
        x_2d = x_pad.reshape(B, rows, period, d).permute(0, 3, 1, 2)

        # 2D Inception block
        out_2d = self.inception(x_2d)

        # Reshape back: (B, d, rows, period) → (B, rows*period, d) → trim
        out = out_2d.permute(0, 2, 3, 1).reshape(B, L_pad, d)
        return out[:, :L, :]    # trim padding


# ─────────────────────────────────────────────
#  MARKET TIMESNET
# ─────────────────────────────────────────────
class MarketTimesNet(nn.Module):
    """
    TimesNet for BUY / SELL / HOLD classification.

    Discovers the dominant temporal periods in the input via FFT,
    then processes the sequence as a 2D image for each period.
    No other model in the stack does this — it is the only model
    with true 2D inductive bias over the price sequence.
    """

    def __init__(self, config: Optional[TimesNetConfig] = None):
        super().__init__()
        self.config = config or TimesNetConfig()
        d = self.config.d_model

        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(self.config.input_features, d),
            nn.LayerNorm(d),
        )

        # One TimesBlock per period (shared architecture, different periods at runtime)
        self.blocks = nn.ModuleList([
            TimesBlock(d, self.config.dropout)
            for _ in range(self.config.top_k)
        ])

        self.norm = nn.LayerNorm(d)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d // 2, self.config.num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    # ── period discovery ──────────────────────────────────────────────────

    @torch.no_grad()
    def _discover_periods(self, x: torch.Tensor) -> List[int]:
        """
        Find top-k dominant periods via FFT of the mean signal.

        Args:
            x: (B, L, d_model)  — after embedding
        Returns:
            List of k integer periods, e.g. [8, 16, 4]
        """
        L = x.shape[1]

        # Average across batch and feature dims → 1D signal (L,)
        signal = x.detach().mean(dim=(0, 2))

        # Real FFT amplitude spectrum (L//2+1 frequencies)
        freqs = torch.fft.rfft(signal, n=L)
        amp   = torch.abs(freqs)

        # Exclude DC (index 0) and find top-k frequencies
        amp_valid = amp[1: L // 2]
        k = min(self.config.top_k, amp_valid.shape[0])
        _, top_idx = torch.topk(amp_valid, k)
        top_idx = top_idx + 1   # offset for skipped DC

        # Convert frequency index → period = L / freq_idx
        periods = []
        seen    = set()
        for idx in top_idx:
            p = max(2, int(round(L / idx.item())))
            p = min(p, L // 2)   # period can't exceed half the window
            if p not in seen:
                seen.add(p)
                periods.append(p)
        # Pad to top_k with a fallback period if needed
        fallbacks = [4, 8, 16, 2, 32]
        for fb in fallbacks:
            if len(periods) >= self.config.top_k:
                break
            if fb not in seen:
                seen.add(fb)
                periods.append(fb)
        return periods[: self.config.top_k]

    # ── encoding ──────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, features) → (B, d_model)"""
        h = self.embed(x)                        # (B, L, d)
        periods = self._discover_periods(h)

        # Apply each TimesBlock for its discovered period, accumulate
        out = h
        for block, period in zip(self.blocks, periods):
            out = out + block(h, period)         # residual accumulation

        return self.norm(out).mean(dim=1)        # global avg pool → (B, d)

    # ── public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(x)
        return F.softmax(self.classifier(pooled), dim=-1), self.confidence_head(pooled)
