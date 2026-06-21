"""
=============================================================
  Python ML Bridge - Model Unit Tests
  Tests for Transformer and LSTM models:
    - Forward pass shape verification
    - Gradient flow check
    - Output probability ranges
    - Confidence head output
=============================================================
"""

import pytest
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TransformerConfig, LSTMConfig
from models.transformer_model import MarketTransformer, PositionalEncoding
from models.lstm_model import MarketLSTM, Attention


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def transformer_config():
    return TransformerConfig(
        input_features=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        seq_length=16,
        num_classes=3,
    )


@pytest.fixture
def lstm_config():
    return LSTMConfig(
        input_features=32,
        hidden_size=64,
        num_layers=2,
        seq_length=16,
        num_classes=3,
        bidirectional=True,
        attention=True,
    )


@pytest.fixture
def sample_input():
    """Sample input tensor: (batch=4, seq_len=16, features=32)."""
    return torch.randn(4, 16, 32)


# ─────────────────────────────────────────────
#  TRANSFORMER TESTS
# ─────────────────────────────────────────────
class TestTransformer:
    """Tests for the MarketTransformer model."""

    def test_forward_pass_shape(self, transformer_config, sample_input):
        """Test that forward pass produces correct output shape."""
        model = MarketTransformer(transformer_config)
        output = model(sample_input)
        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"

    def test_output_probabilities(self, transformer_config, sample_input):
        """Test that output sums to 1 (valid probabilities)."""
        model = MarketTransformer(transformer_config)
        output = model(sample_input)
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"Probabilities should sum to 1, got {sums}"

    def test_output_range(self, transformer_config, sample_input):
        """Test that all outputs are in [0, 1] range."""
        model = MarketTransformer(transformer_config)
        output = model(sample_input)
        assert (output >= 0).all(), "Output contains negative values"
        assert (output <= 1).all(), "Output contains values > 1"

    def test_gradient_flow(self, transformer_config, sample_input):
        """Test that gradients flow through the model."""
        model = MarketTransformer(transformer_config)
        model.train()
        output = model(sample_input)
        # Use cross-entropy loss with target labels to get meaningful gradients
        target = torch.tensor([0, 1, 2, 0])
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"

    def test_predict_with_confidence(self, transformer_config, sample_input):
        """Test predict_with_confidence returns both probs and confidence."""
        model = MarketTransformer(transformer_config)
        probs, confidence = model.predict_with_confidence(sample_input)
        assert probs.shape == (4, 3), f"Expected probs (4, 3), got {probs.shape}"
        assert confidence.shape == (4, 1), f"Expected conf (4, 1), got {confidence.shape}"
        assert (confidence >= 0).all() and (confidence <= 1).all(), \
            "Confidence should be in [0, 1]"

    def test_different_batch_sizes(self, transformer_config):
        """Test model works with different batch sizes."""
        model = MarketTransformer(transformer_config)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 16, 32)
            output = model(x)
            assert output.shape == (batch_size, 3)

    def test_deterministic_eval(self, transformer_config, sample_input):
        """Test that eval mode gives deterministic outputs."""
        model = MarketTransformer(transformer_config)
        model.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ─────────────────────────────────────────────
#  LSTM TESTS
# ─────────────────────────────────────────────
class TestLSTM:
    """Tests for the MarketLSTM model."""

    def test_forward_pass_shape(self, lstm_config, sample_input):
        """Test that forward pass produces correct output shape."""
        model = MarketLSTM(lstm_config)
        output = model(sample_input)
        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"

    def test_output_probabilities(self, lstm_config, sample_input):
        """Test that output sums to 1 (valid probabilities)."""
        model = MarketLSTM(lstm_config)
        output = model(sample_input)
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"Probabilities should sum to 1, got {sums}"

    def test_output_range(self, lstm_config, sample_input):
        """Test that all outputs are in [0, 1] range."""
        model = MarketLSTM(lstm_config)
        output = model(sample_input)
        assert (output >= 0).all(), "Output contains negative values"
        assert (output <= 1).all(), "Output contains values > 1"

    def test_gradient_flow(self, lstm_config, sample_input):
        """Test that gradients flow through the LSTM model."""
        model = MarketLSTM(lstm_config)
        model.train()
        output = model(sample_input)
        # Use cross-entropy loss with target labels to get meaningful gradients
        target = torch.tensor([0, 1, 2, 0])
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in LSTM parameters"

    def test_predict_with_confidence(self, lstm_config, sample_input):
        """Test predict_with_confidence for LSTM."""
        model = MarketLSTM(lstm_config)
        probs, confidence = model.predict_with_confidence(sample_input)
        assert probs.shape == (4, 3)
        assert confidence.shape == (4, 1)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_bidirectional(self, lstm_config, sample_input):
        """Test bidirectional LSTM works correctly."""
        model = MarketLSTM(lstm_config)
        output = model(sample_input)
        # Bidirectional should give 2x hidden_size
        assert model.effective_hidden == lstm_config.hidden_size * 2

    def test_attention_mechanism(self, lstm_config, sample_input):
        """Test that attention mechanism is applied."""
        model = MarketLSTM(lstm_config)
        hidden_states = model.get_hidden_states(sample_input)
        # hidden_states should have shape (batch, seq_len, effective_hidden)
        assert hidden_states.shape == (4, 16, lstm_config.hidden_size * 2)

    def test_different_batch_sizes(self, lstm_config):
        """Test LSTM with different batch sizes."""
        model = MarketLSTM(lstm_config)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 16, 32)
            output = model(x)
            assert output.shape == (batch_size, 3)


# ─────────────────────────────────────────────
#  POSITIONAL ENCODING TESTS
# ─────────────────────────────────────────────
class TestPositionalEncoding:
    """Tests for positional encoding."""

    def test_shape_preserved(self):
        """Test that positional encoding preserves input shape."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 16, 64)
        output = pe(x)
        assert output.shape == x.shape

    def test_adds_positional_info(self):
        """Test that PE modifies the input."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        pe.eval()
        x = torch.zeros(1, 16, 64)
        output = pe(x)
        # Output should not be all zeros (PE was added)
        assert output.abs().sum() > 0


# ─────────────────────────────────────────────
#  ATTENTION TESTS
# ─────────────────────────────────────────────
class TestAttention:
    """Tests for the attention mechanism."""

    def test_attention_output_shape(self):
        """Test attention produces correct shapes."""
        attn = Attention(hidden_size=128)
        lstm_output = torch.randn(4, 16, 128)
        context, weights = attn(lstm_output)
        assert context.shape == (4, 128)
        assert weights.shape == (4, 16)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights are valid distribution."""
        attn = Attention(hidden_size=128)
        lstm_output = torch.randn(4, 16, 128)
        _, weights = attn(lstm_output)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)
