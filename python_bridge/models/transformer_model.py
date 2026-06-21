"""
=============================================================
  Python ML Bridge - Transformer Model
  Multi-head self-attention transformer for market prediction.
  8 attention heads, 4 encoder layers, 256-dim hidden.
  Input: (batch, seq_len, features)
  Output: (batch, 3) buy/sell/hold probabilities
=============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TransformerConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MarketTransformer(nn.Module):
    """
    Transformer model for market prediction.

    Architecture:
        - Input projection to d_model dimensions
        - Positional encoding
        - N encoder layers with multi-head self-attention
        - Global average pooling
        - Classification head with buy/sell/hold output
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()

        # Input projection
        self.input_projection = nn.Linear(
            self.config.input_features, self.config.d_model
        )
        self.input_norm = nn.LayerNorm(self.config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.d_model,
            max_len=self.config.seq_length,
            dropout=self.config.dropout
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            norm=nn.LayerNorm(self.config.d_model)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.d_model // 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout / 2),
            nn.Linear(self.config.d_model // 4, self.config.num_classes)
        )

        # Confidence head (separate from classification)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)
            mask: Optional attention mask

        Returns:
            Tensor of shape (batch, num_classes) with class probabilities
        """
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.input_norm(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)

    def predict_with_confidence(self, x: torch.Tensor) -> tuple:
        """
        Predict with separate confidence score.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)

        Returns:
            Tuple of (probabilities, confidence) where
            probabilities is (batch, 3) and confidence is (batch, 1)
        """
        # Project and encode
        h = self.input_projection(x)
        h = self.input_norm(h)
        h = self.pos_encoding(h)
        h = self.transformer_encoder(h)

        # Pool
        pooled = h.mean(dim=1)

        # Classification and confidence
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_head(pooled)

        return probs, confidence

    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights for interpretability.

        Args:
            x: Input tensor

        Returns:
            List of attention weight tensors per layer
        """
        # This is a simplified version - full attention extraction
        # would require hooks on the encoder layers
        h = self.input_projection(x)
        h = self.input_norm(h)
        h = self.pos_encoding(h)

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # Get self-attention output
            h_out = layer(h)
            attention_weights.append(h_out.detach())
            h = h_out

        return attention_weights
