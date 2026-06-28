"""
=============================================================
  Python ML Bridge - Chronos Model (SimpleChronos)

  Uses a lightweight learned encoder — no T5 tokenization.
  The checkpoint (chronos.pth) was trained with this architecture.
  The Amazon T5 pre-trained path is disabled to avoid
  tokenization dtype errors on Windows Python 3.14.

  Architecture:
    Path A: scalar mean-signal encoder (temporal patterns)
    Path B: full 46-feature projection (cross-indicator)
    Combined → classification head
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ChronosConfig

logger = logging.getLogger(__name__)


class MarketChronos(nn.Module):
    """
    Lightweight Chronos replacement — two-path encoder.
    Matches the architecture used when chronos.pth was saved.
    """

    def __init__(self, config: Optional[ChronosConfig] = None):
        super().__init__()
        self.config = config or ChronosConfig()
        d = self.config.d_model
        flat = self.config.seq_length * self.config.input_features

        self.signal_enc = nn.Sequential(
            nn.Linear(self.config.seq_length, d * 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(flat, d),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.LayerNorm(d),
        )
        combined = d * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined, combined // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(combined // 2, self.config.num_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(combined, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        logger.info("MarketChronos: using SimpleChronos architecture (no T5)")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        sig = x.mean(dim=-1)
        sig = (sig - sig.mean(dim=1, keepdim=True)) / (
            sig.std(dim=1, keepdim=True) + 1e-8
        )
        return torch.cat([
            self.signal_enc(sig),
            self.feat_proj(x.reshape(B, -1)),
        ], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._encode(x)
        return F.softmax(self.classifier(enc), dim=-1), self.confidence_head(enc)
