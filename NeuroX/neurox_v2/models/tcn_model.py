"""
=============================================================
  Python ML Bridge - Temporal Convolutional Network (TCN)
  Dilated causal 1D-CNN for multi-scale time-series pattern recognition.

  Architecture:
    - Input projection: features → n_filters channels
    - 6 TemporalBlocks with exponential dilation (1, 2, 4, 8, 16, 32)
    - Receptive field covers full 64-bar window
    - Global average pooling → classifier + confidence head

  Why TCN complements Transformer + LSTM:
    - Transformer: captures GLOBAL dependencies via self-attention
    - LSTM: captures SEQUENTIAL order and gating
    - TCN: captures MULTI-SCALE local patterns in parallel
      (e.g. 2-bar reversals AND 32-bar trends simultaneously)
    - Fully parallelisable → fast CPU inference
=============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TCNConfig


# ─────────────────────────────────────────────
#  TEMPORAL BLOCK
# ─────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """
    Single TCN block: two dilated conv layers with residual connection.

    Uses weight normalisation (instead of batch norm) for stability with
    small batches during live inference. Same-padding ensures output
    sequence length == input sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Same-padding: output length = input length
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=padding)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=padding)
        )
        self.net = nn.Sequential(
            self.conv1, nn.GELU(), nn.Dropout(dropout),
            self.conv2, nn.GELU(), nn.Dropout(dropout),
        )

        # 1×1 conv for residual projection when channel dims differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out",
                                nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out",
                                nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


# ─────────────────────────────────────────────
#  MARKET TCN
# ─────────────────────────────────────────────
class MarketTCN(nn.Module):
    """
    Temporal Convolutional Network for market direction prediction.

    Input:  (batch, seq_len, input_features)  — same shape as Transformer/LSTM
    Output: (batch, num_classes)              — BUY / SELL / HOLD logits

    Dilation schedule for n_layers=6, kernel_size=3:
        Layer 0  dilation=1   receptive field =  3 bars
        Layer 1  dilation=2   receptive field =  7 bars
        Layer 2  dilation=4   receptive field = 15 bars
        Layer 3  dilation=8   receptive field = 31 bars
        Layer 4  dilation=16  receptive field = 63 bars  (covers seq_len=64)
        Layer 5  dilation=32  receptive field = 127 bars (full context + overlap)
    """

    def __init__(self, config: Optional[TCNConfig] = None):
        super().__init__()
        self.config = config or TCNConfig()

        # Input projection: (batch, features, seq_len) → (batch, n_filters, seq_len)
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.config.input_features, self.config.n_filters, 1),
            nn.GELU(),
        )

        # Stack of temporal blocks with exponential dilation
        blocks = []
        for i in range(self.config.n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32
            blocks.append(
                TemporalBlock(
                    self.config.n_filters,
                    self.config.n_filters,
                    self.config.kernel_size,
                    dilation,
                    self.config.dropout,
                )
            )
        self.network = nn.Sequential(*blocks)

        # Classification head (after global avg pool)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_filters, self.config.n_filters // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_filters // 2, self.config.num_classes),
        )

        # Separate confidence head (aligned with Transformer + LSTM API)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.n_filters, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    # ── internal helpers ──────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run input through projection + TCN blocks + global avg pool.

        Args:
            x: (batch, seq_len, features)
        Returns:
            pooled: (batch, n_filters)
        """
        # Conv1d expects (batch, channels, seq_len)
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = self.network(h)
        # Global average pooling over time dimension
        return h.mean(dim=2)

    # ── public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw logits for cross-entropy loss.

        Args:
            x: (batch, seq_len, input_features)
        Returns:
            logits: (batch, num_classes)
        """
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Class probabilities (softmax applied).

        Args:
            x: (batch, seq_len, input_features)
        Returns:
            probs: (batch, num_classes)
        """
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with separate confidence score.

        Returns:
            (probs: (batch, num_classes), confidence: (batch, 1))
        """
        pooled = self._encode(x)
        probs = F.softmax(self.classifier(pooled), dim=-1)
        confidence = self.confidence_head(pooled)
        return probs, confidence
