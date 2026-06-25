"""
=============================================================
  Python ML Bridge - Chronos (Foundation Model)
  Amazon, 2024 — https://arxiv.org/abs/2403.07815

  Why Chronos is unlike every other model in the stack:
  ──────────────────────────────────────────────────────
  Every other model (Transformer, LSTM, Mamba, xLSTM…) is trained
  FROM SCRATCH on your data. Their weights start random.

  Chronos was pre-trained on BILLIONS of time series data points
  from finance, energy, weather, retail, IoT — before it ever
  sees your XAUUSD data. It already understands:
    • Mean reversion patterns
    • Trend persistence
    • Volatility clustering
    • Seasonal rhythms at multiple scales

  Integration:
    Input (batch, 64, 46)
    → Extract close-price proxy (mean over features, normalised)
    → Chronos T5 encoder  [FROZEN — pre-trained weights]
    → Mean-pool encoder hidden states → (batch, chronos_dim)
    → Parallel: learned feature projection (batch, 46×64 → proj_dim)
    → Concatenate → classification head [TRAINED on your data]
    → (batch, 3)

  Install:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

  Falls back to a lightweight learned encoder if chronos is not installed.
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ChronosConfig

logger = logging.getLogger(__name__)

# ── optional chronos import ───────────────────────────────────────────────────
try:
    from chronos import ChronosPipeline
    _CHRONOS_AVAILABLE = True
    logger.info("Chronos library found — using Amazon pre-trained encoder")
except ImportError:
    _CHRONOS_AVAILABLE = False
    logger.warning(
        "chronos-forecasting not installed. Using lightweight fallback encoder.\n"
        "Install for full pre-trained benefit:\n"
        "  pip install git+https://github.com/amazon-science/chronos-forecasting.git"
    )


# ─────────────────────────────────────────────
#  FALLBACK ENCODER (when chronos not installed)
# ─────────────────────────────────────────────
class _FallbackEncoder(nn.Module):
    """Lightweight 2-layer Transformer encoder used when chronos is absent."""

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj  = nn.Linear(input_dim, d_model)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm  = nn.LayerNorm(d_model)
        self.d_out = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, d_model)"""
        h = self.proj(x.unsqueeze(-1))      # (B, L, 1) → (B, L, d_model)
        h = self.enc(h)
        return self.norm(h.mean(dim=1))      # global avg pool → (B, d_model)


# ─────────────────────────────────────────────
#  MARKET CHRONOS
# ─────────────────────────────────────────────
class MarketChronos(nn.Module):
    """
    Chronos-based classification model.

    Path A (Chronos available):
        Close-proxy signal → frozen T5 encoder → mean-pool → (B, chronos_dim)

    Path B (fallback):
        Close-proxy signal → lightweight learned encoder → (B, d_model)

    Both paths concatenated with a learned projection of all 46 features,
    then fed to a classification head.
    """

    def __init__(self, config: Optional[ChronosConfig] = None):
        super().__init__()
        self.config = config or ChronosConfig()

        # ── Path A or B: scalar-signal encoder ────────────────────────────
        if _CHRONOS_AVAILABLE:
            self._setup_chronos_encoder()
            enc_dim = self.config.chronos_d_model
        else:
            self._chronos_enc = _FallbackEncoder(
                input_dim=1,
                d_model=self.config.d_model,
                dropout=self.config.dropout,
            )
            enc_dim = self.config.d_model

        # ── Parallel feature projection (all 46 indicators → compact vec) ─
        feat_flat = self.config.seq_length * self.config.input_features
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_flat, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm(self.config.d_model),
        )

        # ── Classification head ────────────────────────────────────────────
        combined_dim = enc_dim + self.config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(combined_dim // 2, self.config.num_classes),
        )

        # ── Confidence head ────────────────────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _setup_chronos_encoder(self):
        """Load and freeze the Chronos T5 encoder."""
        try:
            pipeline = ChronosPipeline.from_pretrained(
                self.config.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            # Extract just the T5 encoder (no decoder needed for classification)
            self._chronos_enc = pipeline.model.model.encoder
            self._chronos_tok = pipeline.tokenizer

            # Freeze all encoder weights — we only train the head
            for p in self._chronos_enc.parameters():
                p.requires_grad = False
            logger.info(
                f"Chronos encoder loaded: {self.config.model_name} "
                f"(frozen, {sum(p.numel() for p in self._chronos_enc.parameters()):,} params)"
            )
        except Exception as e:
            logger.warning(f"Chronos load failed ({e}), using fallback encoder.")
            self._chronos_enc = _FallbackEncoder(
                input_dim=1,
                d_model=self.config.d_model,
                dropout=self.config.dropout,
            )
            self.config.chronos_d_model = self.config.d_model

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode a scalar 1-D signal (B, L) through the encoder.
        Returns (B, enc_dim).
        """
        if _CHRONOS_AVAILABLE and hasattr(self, '_chronos_tok'):
            try:
                # Chronos tokenises the signal into discrete bins
                # signal: (B, L) as a list of 1-D tensors
                context = [signal[i] for i in range(signal.shape[0])]
                token_ids, attn_mask, _ = self._chronos_tok.context_input_transform(context)
                token_ids = token_ids.to(signal.device)
                attn_mask = attn_mask.to(signal.device)
                enc_out   = self._chronos_enc(
                    input_ids=token_ids,
                    attention_mask=attn_mask,
                ).last_hidden_state                         # (B, L_tok, d_enc)
                return enc_out.mean(dim=1)                  # (B, d_enc)
            except Exception:
                # If tokenisation fails, fall through to fallback path
                pass
        # Fallback: pass through learned encoder
        return self._chronos_enc(signal)                    # (B, d_model)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features) → (batch, combined_dim)
        """
        B = x.shape[0]

        # Scalar signal: use normalised mean across all features as price proxy
        signal = x.mean(dim=-1)                             # (B, L)
        signal = (signal - signal.mean(dim=1, keepdim=True)) / (
            signal.std(dim=1, keepdim=True) + 1e-8
        )
        enc_vec = self._encode_signal(signal)               # (B, enc_dim)

        # Full feature projection
        feat_vec = self.feat_proj(x.reshape(B, -1))        # (B, d_model)

        return torch.cat([enc_vec, feat_vec], dim=-1)       # (B, combined_dim)

    # ── public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._encode(x)
        return F.softmax(self.classifier(enc), dim=-1), self.confidence_head(enc)
