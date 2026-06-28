"""
=============================================================
  Python ML Bridge - SOFTS (Star Of Time-Series)
  Han et al., NeurIPS 2024 — https://arxiv.org/abs/2408.05718

  Core idea — Star Aggregate for O(N) cross-series interaction:
  ─────────────────────────────────────────────────────────────
  Full cross-attention between N=46 feature series costs O(N²).
  iTransformer already does this. SOFTS achieves the same global
  context sharing at O(N) cost using a STAR topology:

    Instead of every feature attending to every other feature:
    ┌─────┐     ┌─────┐
    │ RSI │────▶│ ATR │  ← O(N²): N×N pairs
    └─────┘     └─────┘

    SOFTS routes everything through one central STAR node:
    ┌─────┐        ┌──────┐        ┌─────┐
    │ RSI │──────▶ │ STAR │◀────── │ ATR │  ← O(N): N×1 pairs
    └─────┘        └──────┘        └─────┘
            ↑ aggregates all N series
            ↓ broadcasts global context back

  The STAR node becomes a learned "market state" summary:
    • It absorbs the current state of all 46 indicators
    • It broadcasts global context back to each indicator
    • Each indicator gates how much to use local vs global info
    • Result: every feature is aware of the broader market without
      paying the quadratic cost of full cross-attention

  Architecture:
    Input (batch, 64, 46)
    → Embed each feature series independently → (batch, 64, d_model)
    → N × STARBlock:
          star  = mean(all feature embeddings)         → (B, 64, d_model)
          For each feature: gate(local, star)          → (B, 64, d_model)
    → LayerNorm → Global avg pool → (batch, d_model)
    → Classifier → (batch, 3)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SOFTSConfig


# ─────────────────────────────────────────────
#  STAR AGGREGATE BLOCK
# ─────────────────────────────────────────────
class STARBlock(nn.Module):
    """
    Star aggregate block: O(N) cross-series mixing via a central node.

    For N feature series of length L:
      1. Compute star = weighted mean across all N series (global context)
      2. For each series i: fused_i = gate * local_i + (1-gate) * star
      3. Optional FFN applied to each fused representation
    """

    def __init__(self, d_model: int, n_features: int, dropout: float):
        super().__init__()
        self.n_features = n_features

        # Learned importance weights for computing the star
        self.star_weights = nn.Linear(n_features, n_features)

        # Series-level FFN (applied independently to each feature)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        # Gate: decides how much each series uses global star context
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, N, d_model)  — N feature embeddings of length L
        returns: (B, L, N, d_model)
        """
        B, L, N, d = h.shape

        # ── 1. Build star node (global context) ───────────────────────────
        # Attention weights across N series: (B, L, N) → softmax
        # Use a learned projection on the N dimension
        h_flat = h.mean(dim=-1)                      # (B, L, N) — reduce to scalar per series
        w = F.softmax(self.star_weights(h_flat), dim=-1)  # (B, L, N)
        star = (w.unsqueeze(-1) * h).sum(dim=2)      # (B, L, d_model) — weighted sum

        # ── 2. Broadcast star to all N series and gate ─────────────────────
        star_exp = star.unsqueeze(2).expand(-1, -1, N, -1)  # (B, L, N, d_model)
        combined = torch.cat([h, star_exp], dim=-1)          # (B, L, N, 2*d_model)
        gate     = torch.sigmoid(self.gate_proj(combined))   # (B, L, N, d_model)
        fused    = self.norm1(gate * h + (1.0 - gate) * star_exp)

        # ── 3. Per-series FFN ──────────────────────────────────────────────
        fused_flat = fused.reshape(B * L * N, d)
        out = self.ffn(fused_flat).reshape(B, L, N, d)
        return self.norm2(fused + out)


# ─────────────────────────────────────────────
#  MARKET SOFTS
# ─────────────────────────────────────────────
class MarketSOFTS(nn.Module):
    """
    SOFTS for market direction classification.

    Embeds each of the 46 feature series independently, then stacks
    N STARBlocks that mix information through a central "market state"
    node. More efficient than iTransformer (O(N) vs O(N²)) and captures
    a complementary global-context signal.
    """

    def __init__(self, config: Optional[SOFTSConfig] = None):
        super().__init__()
        self.config = config or SOFTSConfig()
        d = self.config.d_model
        N = self.config.input_features

        # Embed each feature's full time history independently
        self.feature_embed = nn.Sequential(
            nn.Linear(self.config.seq_length, d),
            nn.LayerNorm(d),
            nn.Dropout(self.config.dropout),
        )

        # Stack of STAR blocks
        self.blocks = nn.ModuleList([
            STARBlock(d, N, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])

        self.norm = nn.LayerNorm(d)

        # Classifier: aggregate across N features and L time steps
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d // 2, self.config.num_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, N) → (B, d_model)"""
        B, L, N = x.shape

        # Embed each feature series: (B, N, L) → (B, N, d)
        x_t = x.transpose(1, 2)                     # (B, N, L)
        h   = self.feature_embed(x_t)               # (B, N, d)

        # Reshape to (B, L=1, N, d) for STARBlock — we treat N as "series"
        # and use L=1 (aggregate representation per series, not per timestep)
        # This is more memory efficient for N=46
        h = h.unsqueeze(1)                           # (B, 1, N, d)

        for block in self.blocks:
            h = block(h)                             # (B, 1, N, d)

        # Aggregate across N series
        out = self.norm(h.squeeze(1).mean(dim=1))    # (B, d)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._encode(x)
        return F.softmax(self.classifier(enc), dim=-1), self.confidence_head(enc)
