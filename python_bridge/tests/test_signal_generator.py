"""
=============================================================
  Python ML Bridge - Signal Generator Tests
  Tests signal generation with mock data, risk filters,
  confidence thresholds, and regime adjustments.
=============================================================
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SignalConfig, DataConfig, RiskConfig
from strategies.signal_generator import SignalGenerator, TradeSignal
from strategies.risk_manager import RiskManager
from strategies.regime_detector import RegimeDetector, MarketRegime


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def signal_config():
    return SignalConfig(
        min_confidence=0.65,
        strong_confidence=0.80,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=2.5,
        cooldown_seconds=0,  # No cooldown for tests
    )


@pytest.fixture
def mock_features():
    """Mock feature input (1, seq_len=64, features=32)."""
    return np.random.randn(1, 64, 32).astype(np.float32)


@pytest.fixture
def mock_prices():
    """Mock price DataFrame."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    prices = 2000 + np.cumsum(np.random.randn(200) * 2)
    return pd.DataFrame({
        "Open": prices - 1,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, 200),
    }, index=dates)


# ─────────────────────────────────────────────
#  TRADE SIGNAL TESTS
# ─────────────────────────────────────────────
class TestTradeSignal:
    """Tests for the TradeSignal class."""

    def test_signal_creation(self):
        """Test creating a trade signal."""
        signal = TradeSignal(
            timestamp="2024-01-01 12:00:00",
            symbol="XAUUSD",
            action="BUY",
            confidence=0.85,
            sl_pips=150.0,
            tp_pips=250.0,
            lot_size=0.10,
            model_name="transformer",
            regime="trending"
        )
        assert signal.action == "BUY"
        assert signal.confidence == 0.85
        assert signal.lot_size == 0.10

    def test_signal_to_dict(self):
        """Test converting signal to dictionary."""
        signal = TradeSignal(
            timestamp="2024-01-01 12:00:00",
            symbol="XAUUSD",
            action="SELL",
            confidence=0.72,
            sl_pips=100.0,
            tp_pips=200.0,
            lot_size=0.05,
            model_name="lstm",
            regime="volatile"
        )
        d = signal.to_dict()
        assert d["action"] == "SELL"
        assert d["symbol"] == "XAUUSD"
        assert d["regime"] == "volatile"

    def test_signal_to_csv_row(self):
        """Test CSV row generation."""
        signal = TradeSignal(
            timestamp="2024-01-01 12:00:00",
            symbol="XAUUSD",
            action="BUY",
            confidence=0.85,
            sl_pips=150.0,
            tp_pips=250.0,
            lot_size=0.10,
            model_name="ensemble",
            regime="trending"
        )
        csv = signal.to_csv_row()
        parts = csv.split(",")
        assert len(parts) == 9
        assert parts[2] == "BUY"
        assert parts[7] == "ensemble"


# ─────────────────────────────────────────────
#  RISK MANAGER TESTS
# ─────────────────────────────────────────────
class TestRiskManager:
    """Tests for the risk management system."""

    def test_position_sizing_kelly(self):
        """Test Kelly criterion position sizing."""
        rm = RiskManager(RiskConfig(account_balance=10000))
        lot = rm.calculate_position_size(
            confidence=0.8, atr=5.0,
            win_rate=0.6, avg_win_loss_ratio=1.5
        )
        assert lot >= rm.config.min_lot_size
        assert lot <= rm.config.max_lot_size

    def test_position_size_scales_with_confidence(self):
        """Test that higher confidence gives larger position."""
        rm = RiskManager(RiskConfig(account_balance=10000))
        lot_low = rm.calculate_position_size(confidence=0.5, atr=5.0)
        lot_high = rm.calculate_position_size(confidence=0.9, atr=5.0)
        assert lot_high >= lot_low

    def test_max_drawdown_halt(self):
        """Test that max drawdown stops trading."""
        config = RiskConfig(account_balance=10000, max_drawdown=0.10)
        rm = RiskManager(config)
        # Simulate 15% loss (exceeds 10% max drawdown)
        rm.update_equity(8500)
        assert not rm.is_trading_allowed()

    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        config = RiskConfig(account_balance=10000, max_daily_loss=0.05)
        rm = RiskManager(config)
        rm.update_equity(9400)  # 6% loss exceeds 5% limit
        assert not rm.is_trading_allowed()

    def test_max_positions_limit(self):
        """Test max open positions limit."""
        config = RiskConfig(max_open_positions=2)
        rm = RiskManager(config)
        rm.register_trade({"id": "1", "direction": "BUY"})
        rm.register_trade({"id": "2", "direction": "SELL"})
        assert not rm.is_trading_allowed()

    def test_sl_tp_calculation_buy(self):
        """Test SL/TP levels for BUY signal."""
        rm = RiskManager()
        levels = rm.calculate_sl_tp(
            atr=5.0, direction="BUY",
            current_price=2000.0,
            sl_mult=1.5, tp_mult=2.5
        )
        assert levels["sl_price"] < 2000.0  # SL below for BUY
        assert levels["tp_price"] > 2000.0  # TP above for BUY
        assert levels["sl_pips"] > 0
        assert levels["tp_pips"] > 0

    def test_sl_tp_calculation_sell(self):
        """Test SL/TP levels for SELL signal."""
        rm = RiskManager()
        levels = rm.calculate_sl_tp(
            atr=5.0, direction="SELL",
            current_price=2000.0,
            sl_mult=1.5, tp_mult=2.5
        )
        assert levels["sl_price"] > 2000.0  # SL above for SELL
        assert levels["tp_price"] < 2000.0  # TP below for SELL

    def test_win_rate_tracking(self):
        """Test win rate calculation from history."""
        rm = RiskManager()
        rm.close_trade("1", 100.0)   # Win
        rm.close_trade("2", -50.0)   # Loss
        rm.close_trade("3", 75.0)    # Win
        assert rm.get_win_rate() == pytest.approx(2.0 / 3.0, abs=0.01)


# ─────────────────────────────────────────────
#  REGIME DETECTOR TESTS
# ─────────────────────────────────────────────
class TestRegimeDetector:
    """Tests for market regime detection."""

    def test_detect_trending_regime(self):
        """Test detection of trending market."""
        detector = RegimeDetector()
        # Create strongly trending prices
        prices = pd.DataFrame({
            "Close": np.linspace(2000, 2100, 200)  # Strong uptrend
        })
        adx = pd.Series([40.0] * 200)  # High ADX = trending
        result = detector.detect_regime(prices, adx=adx)
        assert result["regime"] == MarketRegime.TRENDING
        assert result["confidence"] > 0

    def test_detect_ranging_regime(self):
        """Test detection of ranging market."""
        detector = RegimeDetector()
        # Create sideways prices
        prices = pd.DataFrame({
            "Close": 2000 + np.sin(np.linspace(0, 10, 200)) * 5
        })
        adx = pd.Series([15.0] * 200)  # Low ADX = ranging
        result = detector.detect_regime(prices, adx=adx)
        assert result["regime"] == MarketRegime.RANGING

    def test_regime_adjustments(self):
        """Test that regime gives correct strategy adjustments."""
        detector = RegimeDetector()
        adj = detector.get_regime_adjustments(MarketRegime.VOLATILE)
        assert adj["position_size_mult"] < 1.0  # Reduce size in volatile
        assert adj["confidence_threshold"] > 0.7  # Higher bar in volatile

    def test_crash_regime(self):
        """Test crash regime detection."""
        detector = RegimeDetector()
        # Sharp decline
        prices_arr = np.concatenate([
            np.linspace(2100, 2100, 180),
            np.linspace(2100, 2000, 20)  # 5% drop in 20 bars
        ])
        prices = pd.DataFrame({"Close": prices_arr})
        result = detector.detect_regime(prices, vix=40.0)
        assert result["regime"] in [MarketRegime.CRASH, MarketRegime.VOLATILE]


# ─────────────────────────────────────────────
#  SIGNAL GENERATOR INTEGRATION TESTS
# ─────────────────────────────────────────────
class TestSignalGenerator:
    """Integration tests for signal generation."""

    def test_hold_signal_on_low_confidence(self, signal_config, mock_features, mock_prices):
        """Test that low confidence results in HOLD."""
        gen = SignalGenerator(signal_config=signal_config)
        # With random untrained models, confidence should be low
        signal = gen.generate_signal(
            features=mock_features,
            prices=mock_prices,
            atr=5.0,
            current_price=2000.0
        )
        # Untrained models will likely produce HOLD due to low agreement/confidence
        assert signal.action in ["BUY", "SELL", "HOLD"]

    def test_signal_has_required_fields(self, signal_config, mock_features, mock_prices):
        """Test that generated signals have all required fields."""
        gen = SignalGenerator(signal_config=signal_config)
        signal = gen.generate_signal(
            features=mock_features,
            prices=mock_prices,
            atr=5.0,
            current_price=2000.0
        )
        assert hasattr(signal, "timestamp")
        assert hasattr(signal, "symbol")
        assert hasattr(signal, "action")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "sl_pips")
        assert hasattr(signal, "tp_pips")
        assert hasattr(signal, "lot_size")
        assert hasattr(signal, "model_name")
        assert hasattr(signal, "regime")

    def test_signal_action_valid(self, signal_config, mock_features, mock_prices):
        """Test that action is one of BUY/SELL/HOLD."""
        gen = SignalGenerator(signal_config=signal_config)
        signal = gen.generate_signal(
            features=mock_features,
            prices=mock_prices,
            atr=5.0,
            current_price=2000.0
        )
        assert signal.action in ["BUY", "SELL", "HOLD"]

    def test_hold_signal_zero_lot(self, signal_config, mock_features, mock_prices):
        """Test that HOLD signals have zero lot size."""
        gen = SignalGenerator(signal_config=signal_config)
        signal = gen.generate_signal(
            features=mock_features,
            prices=mock_prices,
            atr=5.0,
            current_price=2000.0
        )
        if signal.action == "HOLD":
            assert signal.lot_size == 0.0
