"""
=============================================================
  Python ML Bridge - iTransformer
  "iTransformer: Inverted Transformers Are Effective for Time Series"
  Liu et al., ICLR 2024 — https://arxiv.org/abs/2310.06625

  THE KEY IDEA — what makes iTransformer unique in your 12-model stack:
  ─────────────────────────────────────────────────────────────────────
  Every other model in the ensemble (Transformer, LSTM, TCN, PatchTST,
  TFT, N-HiTS, Mamba) treats the input as:
      token₁=bar₁, token₂=bar₂, ..., token₆₄=bar₆₄
  They capture TEMPORAL dependencies — how bars relate over time.

  iTransformer flips this completely:
      token₁=RSI, token₂=MACD, token₃=ATR, ..., token₄₆=EMA200
  It captures CROSS-FEATURE dependencies — how your 46 indicators
  relate TO EACH OTHER at this moment in the market.

  Examples of what iTransformer learns that others miss:
    • "When RSI is OB AND ATR is expanding AND MACD diverges → high prob SELL"
    • "When EMA9 > EMA21 > EMA50 AND momentum > threshold → strong BUY"
    • "When BB width AND VIX AND ATR all spike → volatility regime, no trade"

  Architecture:
    Input  (batch, seq_len=64, features=46)
    → Transpose → (batch, 46, 64)
    → Feature Embedding: Linear(64, d_model) per feature
    → LayerNorm
    → 3-layer Transformer Encoder over 46 feature tokens
    → Mean pool over feature dim → (batch, d_model)
    → Classifier → (batch, 3)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ITransformerConfig


class MarketITransformer(nn.Module):
    """
    iTransformer for market direction classification.

    Each of your 46 features (RSI, MACD, ATR, BB, EMA9...) becomes
    a single token with a d_model-dimensional embedding derived from
    its 64-bar history. Attention runs ACROSS features — the model
    learns which feature combinations signal BUY / SELL / HOLD.

    This is the only model in the stack that explicitly models
    cross-indicator relationships rather than temporal sequences.
    """

    def __init__(self, config: Optional[ITransformerConfig] = None):
        super().__init__()
        self.config = config or ITransformerConfig()

        # ── feature embedding ──────────────────────────────────────────────
        # Each feature's 64-bar history → d_model representation
        # Applied independently per feature (channel-independent projection)
        self.feature_embed = nn.Sequential(
            nn.Linear(self.config.seq_length, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.Dropout(self.config.dropout),
        )

        # ── transformer encoder over feature tokens ────────────────────────
        # n_features (46) tokens, each = one indicator's d_model embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            norm=nn.LayerNorm(self.config.d_model),
        )

        # ── classifier ──────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(self.config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_classes),
        )

        # ── confidence head ─────────────────────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Invert input, embed each feature, run attention over features.
        x: (batch, seq_len=64, features=46) → returns (batch, d_model)
        """
        # Invert: treat features as tokens, each with seq_len as its dimension
        # (batch, features=46, seq_len=64)
        x_inv = x.transpose(1, 2)

        # Embed each feature's 64-bar history independently
        # (batch, 46, d_model)
        h = self.feature_embed(x_inv)

        # Transformer over the 46 feature tokens
        h = self.encoder(h)          # (batch, 46, d_model)

        # Mean pool over feature dimension → single vector
        return self.norm(h.mean(dim=1))   # (batch, d_model)

    # ── public API ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(x)
        return F.softmax(self.classifier(pooled), dim=-1), self.confidence_head(pooled)
