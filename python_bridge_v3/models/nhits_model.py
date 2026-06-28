"""
=============================================================
  Python ML Bridge - N-HiTS
  Neural Hierarchical Interpolation for Time Series
  Challu et al., Mila / Element AI, 2022
  https://arxiv.org/abs/2201.12886

  Why N-HiTS works for M1 gold scalping:
    Traditional models see all 64 bars at the same scale.
    N-HiTS uses a STACK of blocks where each block deliberately
    downsamples the input to a different temporal resolution:

      Block 0  pool=8  → sees 8 "compressed" bars  → captures MACRO trend
      Block 1  pool=4  → sees 16 bars              → captures MID-TERM flow
      Block 2  pool=2  → sees 32 bars              → captures SHORT pattern
      Block 3  pool=1  → sees all 64 bars          → captures MICRO details

    Each block outputs a residual. The final residuals are combined
    for the classification decision. This means the model ALWAYS has
    both a high-level trend view AND a micro price action view.

    Key advantage over standard Transformer/LSTM:
      Explicitly hierarchical — long-term trend cannot "drown out"
      short-term signals because they're processed in separate blocks.

  Architecture:
    Input (batch, 64, 46)
    → Block 0: MaxPool(8) → flatten → MLP → head   (macro scale)
    → Block 1: MaxPool(4) → flatten → MLP → head   (mid scale)
    → Block 2: MaxPool(2) → flatten → MLP → head   (short scale)
    → Block 3: no pool   → flatten → MLP → head    (micro scale)
    → Concat all heads → final classifier → (batch, 3)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import NHiTSConfig


# ─────────────────────────────────────────────
#  N-HiTS BLOCK
# ─────────────────────────────────────────────
class NHiTSBlock(nn.Module):
    """
    Single N-HiTS block operating at one temporal resolution.

    Steps:
      1. MaxPool along time dimension (downsamples to pool-sized windows)
      2. Flatten to 1D
      3. 3-layer MLP with residual
      4. Output a block-level embedding for the classifier
    """

    def __init__(
        self,
        pool_size: int,
        seq_len: int,
        n_features: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool_size = pool_size

        # After pooling: ceil(seq_len / pool_size) timesteps
        pooled_len = (seq_len + pool_size - 1) // pool_size
        flat_size = pooled_len * n_features

        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)

        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.GELU(),
        )

        # Residual projection if sizes differ
        self.residual = (
            nn.Linear(flat_size, output_size, bias=False)
            if flat_size != output_size else None
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, output_size)
        """
        B, T, F = x.shape

        # Pool along time: (batch, n_features, seq_len) → (batch, n_features, pooled_len)
        x_t = x.transpose(1, 2)                     # (B, F, T)
        pooled = self.pool(x_t)                      # (B, F, pooled_len)
        pooled = pooled.transpose(1, 2)              # (B, pooled_len, F)

        # Flatten: (B, pooled_len * F)
        flat = pooled.reshape(B, -1)

        # MLP with residual
        out = self.mlp(flat)
        res = flat if self.residual is None else self.residual(flat)
        return self.norm(out + res)


# ─────────────────────────────────────────────
#  MARKET N-HiTS
# ─────────────────────────────────────────────
class MarketNHiTS(nn.Module):
    """
    N-HiTS for market direction classification.

    Four hierarchical stacks process the 64-bar window at pool scales
    [8, 4, 2, 1], producing embeddings from macro to micro perspective.
    All embeddings are concatenated and fed to the final classifier.

    This explicit multi-scale decomposition is especially valuable for
    gold scalping: a trade may be counter-trend at the 64-bar scale
    but perfectly aligned at the 8-bar micro scale — or vice versa.
    """

    def __init__(self, config: Optional[NHiTSConfig] = None):
        super().__init__()
        self.config = config or NHiTSConfig()

        # ── hierarchical blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                pool_size=ps,
                seq_len=self.config.seq_length,
                n_features=self.config.input_features,
                hidden_size=self.config.hidden_size,
                output_size=self.config.block_output_size,
                dropout=self.config.dropout,
            )
            for ps in self.config.pool_sizes
        ])

        # ── fusion layer: concat all block outputs ─────────────────────────
        fusion_input = self.config.block_output_size * len(self.config.pool_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm(self.config.hidden_size),
        )

        # ── classifier ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.num_classes),
        )

        # ── confidence head ────────────────────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features) → (batch, hidden_size)
        """
        # Collect output from each hierarchical block
        block_outs = [block(x) for block in self.blocks]   # list of (B, block_output_size)
        combined = torch.cat(block_outs, dim=-1)            # (B, block_output_size * n_blocks)
        return self.fusion(combined)                         # (B, hidden_size)

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
