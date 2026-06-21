"""
=============================================================
  Python ML Bridge - Bidirectional LSTM with Attention
  3-layer BiLSTM with attention mechanism for sequential
  price pattern recognition.
  Input: (batch, seq_len, features)
  Output: (batch, 3) buy/sell/hold probabilities
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LSTMConfig


class Attention(nn.Module):
    """Scaled dot-product attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention over LSTM outputs.

        Args:
            lstm_output: Tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tuple of (context_vector, attention_weights)
            context_vector: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention_weights(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_size)

        context = self.layer_norm(context)
        return context, weights


class MarketLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for market prediction.

    Architecture:
        - Input projection
        - 3-layer Bidirectional LSTM
        - Attention mechanism over hidden states
        - Classification head with buy/sell/hold output
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        self.config = config or LSTMConfig()

        # Effective hidden size (doubled for bidirectional)
        self.effective_hidden = (
            self.config.hidden_size * 2
            if self.config.bidirectional
            else self.config.hidden_size
        )

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_features, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout / 2),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )

        # Attention mechanism
        self.attention = Attention(self.effective_hidden)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.effective_hidden, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout / 2),
            nn.Linear(self.config.hidden_size // 2, self.config.num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.effective_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)

        Returns:
            Tensor of shape (batch, num_classes) with raw logits.
            Use predict() or predict_with_confidence() for probabilities.
        """
        # Project input
        x = self.input_projection(x)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, effective_hidden)

        # Apply attention
        if self.config.attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            # Use last hidden state if no attention
            context = lstm_out[:, -1, :]

        # Classification (raw logits)
        logits = self.classifier(context)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (applies softmax to logits).

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)

        Returns:
            Tensor of shape (batch, num_classes) with class probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence score.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)

        Returns:
            Tuple of (probabilities, confidence)
        """
        # Project input
        h = self.input_projection(x)

        # LSTM
        lstm_out, _ = self.lstm(h)

        # Attention
        if self.config.attention:
            context, _ = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]

        # Outputs
        logits = self.classifier(context)
        probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_head(context)

        return probs, confidence

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get all hidden states for analysis.

        Args:
            x: Input tensor

        Returns:
            LSTM output tensor (batch, seq_len, effective_hidden)
        """
        h = self.input_projection(x)
        lstm_out, _ = self.lstm(h)
        return lstm_out
