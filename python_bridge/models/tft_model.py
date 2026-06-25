"""
=============================================================
  Python ML Bridge - Temporal Fusion Transformer (TFT)
  Bryan Lim et al., Google, 2021
  https://arxiv.org/abs/1912.09363

  Why TFT is exceptional for XAUUSD trading:
    • Variable Selection Networks (VSN) learn WHICH of your 46 features
      matter most at each bar — e.g. RSI matters more during ranging,
      ATR matters more during trending markets
    • Gated Residual Networks (GRN) suppress irrelevant information
      paths entirely — acts like a learned attention mask on features
    • LSTM encoder captures LOCAL sequential context (last 8 bars)
    • Multi-head self-attention captures GLOBAL dependencies (full 64 bars)
    • All gating is learned end-to-end from trade outcomes

  Architecture:
    Input (batch, 64, 46)
    → Variable Selection Network  →  (batch, 64, d_model=64)
    → LSTM Encoder                →  (batch, 64, d_model)
    → Post-LSTM GRN + Gate
    → Multi-Head Self-Attention   →  (batch, 64, d_model)
    → Post-Attention GRN + Gate
    → Global Mean Pool            →  (batch, d_model)
    → Classifier                  →  (batch, 3)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TFTConfig


# ─────────────────────────────────────────────
#  GATED LINEAR UNIT
# ─────────────────────────────────────────────
class GLU(nn.Module):
    """
    Gated Linear Unit: splits the projection into value and gate halves.
    out = value_half × sigmoid(gate_half)
    Acts as a learned on/off switch for each information channel.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        half = out.shape[-1] // 2
        return out[..., :half] * torch.sigmoid(out[..., half:])


# ─────────────────────────────────────────────
#  GATED RESIDUAL NETWORK
# ─────────────────────────────────────────────
class GRN(nn.Module):
    """
    Gated Residual Network — the fundamental building block of TFT.

    GRN(x) = LayerNorm(x + GLU(ELU(W1·x) + W2·context))

    The gating (GLU) allows the network to completely suppress an
    information path when it is not relevant, which is key for handling
    the highly variable nature of financial time series.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size else None
        )
        self.glu = GLU(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size else None
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = F.elu(self.fc1(x))
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = self.dropout(self.fc2(h))
        return self.layer_norm(residual + self.glu(h))


# ─────────────────────────────────────────────
#  VARIABLE SELECTION NETWORK
# ─────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network: learns soft feature importance per timestep.

    Produces a weighted mixture of per-feature processed embeddings.
    The weights (softmax over features) tell us: at this bar, which
    of the 46 input features is the model relying on?

    Simplified from the original paper for CPU inference efficiency:
    instead of per-feature GRNs (expensive), we use a shared projection
    + attention-style selection weights.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features

        # Project each feature to d_model individually
        self.feature_proj = nn.Linear(n_features, d_model * n_features)
        # Selection weights over features
        self.selection = nn.Linear(n_features, n_features)
        # Post-selection GRN
        self.grn = GRN(d_model, d_model, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_features)
        returns: selected (batch, seq_len, d_model), weights (batch, seq_len, n_features)
        """
        B, T, n_feat = x.shape

        # Selection weights — which features matter this timestep
        weights  = torch.softmax(self.selection(x), dim=-1)  # (B, T, n_feat)
        weighted = weights * x                                # (B, T, n_feat)

        # Project weighted features to d_model
        projected = self.feature_proj(weighted)               # (B, T, d_model*n_feat)
        # Reshape and sum across feature dimension
        projected = projected.reshape(B, T, n_feat, self.d_model)
        combined = projected.sum(dim=2)                     # (B, T, d_model)

        # GRN post-processing
        out = self.grn(combined)
        return self.dropout(out), weights


# ─────────────────────────────────────────────
#  MARKET TFT
# ─────────────────────────────────────────────
class MarketTFT(nn.Module):
    """
    Temporal Fusion Transformer for market direction classification.

    The TFT's gating mechanisms are particularly powerful for trading
    because financial features have highly regime-dependent importance:
    — In trending markets: momentum and EMA features dominate
    — In ranging markets: RSI and Bollinger Band features dominate
    — During news events: ATR and volatility features dominate
    The VSN and GRN layers learn to route information accordingly.
    """

    def __init__(self, config: Optional[TFTConfig] = None):
        super().__init__()
        self.config = config or TFTConfig()
        d = self.config.d_model

        # ── 1. Variable Selection Network ──────────────────────────────────
        self.vsn = VariableSelectionNetwork(
            self.config.input_features, d, self.config.dropout
        )

        # ── 2. LSTM encoder (local sequential context) ─────────────────────
        self.lstm_encoder = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )
        self.lstm_gate = GRN(d, d, d, dropout=self.config.dropout)
        self.lstm_norm = nn.LayerNorm(d)

        # ── 3. Multi-head self-attention (global dependencies) ─────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attn_gate = GRN(d, d, d, dropout=self.config.dropout)
        self.attn_norm = nn.LayerNorm(d)

        # ── 4. Final position-wise processing ──────────────────────────────
        self.poswise_grn = GRN(d, d * 2, d, dropout=self.config.dropout)
        self.poswise_norm = nn.LayerNorm(d)

        # ── 5. Classifier ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d // 2, self.config.num_classes),
        )

        # ── 6. Confidence head ─────────────────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm_encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)   # forget-gate bias = 1

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full TFT encoding pass.
        x: (batch, seq_len, input_features) → (batch, d_model)
        """
        # 1. Variable selection
        vsn_out, _ = self.vsn(x)                          # (B, T, d)

        # 2. LSTM encoder
        lstm_out, _ = self.lstm_encoder(vsn_out)          # (B, T, d)
        lstm_out = self.lstm_norm(vsn_out + self.lstm_gate(lstm_out))

        # 3. Multi-head self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + self.attn_gate(attn_out))

        # 4. Position-wise GRN
        out = self.poswise_norm(attn_out + self.poswise_grn(attn_out))

        # 5. Global mean pool → single vector
        return out.mean(dim=1)                             # (B, d)

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
