"""
=============================================================
  Python ML Bridge - PatchTST
  "A Time Series is Worth 64 Words" (Nie et al., MIT/IBM 2023)
  https://arxiv.org/abs/2211.14730

  Key innovation over vanilla Transformer:
    Instead of feeding 64 individual bar tokens to the Transformer,
    we divide the sequence into PATCHES (e.g. 8 bars per patch → 8 tokens).
    Each patch captures a local window of price action, then attention
    runs across patches — not individual bars.

    Benefits for XAUUSD M1 scalping:
      • 64-bar window → 8 patch tokens → attention is 64x more efficient
      • Patches capture local micro-patterns (2-bar reversals, 8-bar consolidations)
      • Learnable CLS token aggregates the global trade decision
      • Pre-norm architecture → stable training without warmup

  Architecture:
    Input (batch, 64, 46)
    → Patchify → (batch, 8, 8×46=368)
    → Patch Embedding → (batch, 8, d_model=128)
    → CLS token prepend → (batch, 9, 128)
    → Positional Encoding
    → 3-layer Transformer Encoder (Pre-Norm, 4 heads)
    → CLS token → classifier → (batch, 3)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PatchTSTConfig


# ─────────────────────────────────────────────
#  PATCH EMBEDDING
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Projects flattened patches to model dimension.
    Each patch: (patch_size × n_features) → d_model
    """

    def __init__(self, patch_size: int, n_features: int, d_model: int, dropout: float):
        super().__init__()
        patch_dim = patch_size * n_features
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_patches, patch_dim) → (batch, n_patches, d_model)"""
        return self.dropout(self.proj(x))


# ─────────────────────────────────────────────
#  MARKET PATCH TST
# ─────────────────────────────────────────────
class MarketPatchTST(nn.Module):
    """
    PatchTST adapted for 3-class market direction classification.

    The original paper targets forecasting; here we use the encoder
    representations (via a CLS token) for BUY / SELL / HOLD classification.
    Pre-norm (norm_first=True) provides stable gradients without LR warmup.
    """

    def __init__(self, config: Optional[PatchTSTConfig] = None):
        super().__init__()
        self.config = config or PatchTSTConfig()

        assert self.config.seq_length % self.config.patch_size == 0, (
            f"seq_length ({self.config.seq_length}) must be "
            f"divisible by patch_size ({self.config.patch_size})"
        )
        self.n_patches = self.config.seq_length // self.config.patch_size

        # ── patch embedding ────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            self.config.patch_size,
            self.config.input_features,
            self.config.d_model,
            self.config.dropout,
        )

        # ── learnable CLS token (like BERT) ───────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── learnable positional embeddings (n_patches + 1 for CLS) ───────
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, self.config.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(self.config.dropout)

        # ── transformer encoder (pre-norm for stability) ──────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,           # Pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            norm=nn.LayerNorm(self.config.d_model),
        )

        # ── classification head ────────────────────────────────────────────
        self.norm = nn.LayerNorm(self.config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_classes),
        )

        # ── confidence head ────────────────────────────────────────────────
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

    # ── internal encoder ──────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patchify → embed → CLS → positional → transformer → CLS out.
        x: (batch, seq_len, features) → returns (batch, d_model)
        """
        B, L, F = x.shape

        # Patchify: (batch, n_patches, patch_size × features)
        patches = x.reshape(B, self.n_patches, self.config.patch_size * F)

        # Embed patches: (batch, n_patches, d_model)
        h = self.patch_embed(patches)

        # Prepend CLS token: (batch, n_patches+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        # Add positional encoding and dropout
        h = self.pos_drop(h + self.pos_embed)

        # Transformer encoder
        h = self.encoder(h)

        # Return normalised CLS token representation
        return self.norm(h[:, 0])  # (batch, d_model)

    # ── public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits. x: (batch, seq_len, features) → (batch, 3)"""
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities. → (batch, 3)"""
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probs: (batch,3), confidence: (batch,1))"""
        pooled = self._encode(x)
        return F.softmax(self.classifier(pooled), dim=-1), self.confidence_head(pooled)
