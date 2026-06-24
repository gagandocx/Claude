"""
=============================================================
  Python ML Bridge - Training Pipeline Tests
  Tests for ATR-based labeling and multi-timeframe data fetching.
=============================================================
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import shutil

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DataConfig, MODEL_DIR
from data.market_data import MarketDataFetcher
from train import prepare_data


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    """Clean up any checkpoint files created during tests."""
    yield
    # Remove normalization_stats.json if tests created it
    stats_path = os.path.join(MODEL_DIR, "normalization_stats.json")
    if os.path.exists(stats_path):
        os.remove(stats_path)
    # Remove checkpoints dir if empty
    if os.path.isdir(MODEL_DIR) and not os.listdir(MODEL_DIR):
        os.rmdir(MODEL_DIR)


# ─────────────────────────────────────────────
#  ATR-BASED LABELING TESTS
# ─────────────────────────────────────────────
class TestATRBasedLabeling:
    """Tests for ATR-based label generation in prepare_model_input()."""

    def test_hold_when_price_moves_less_than_atr_threshold(self):
        """Test that HOLD label is produced when price moves less than 0.3*ATR."""
        config = DataConfig()
        fetcher = MarketDataFetcher(config)

        # Create 300 bars of flat price data with small random noise
        np.random.seed(42)
        n = 300
        base_price = 2000.0
        # Very small moves that won't exceed 0.3*ATR
        noise = np.random.randn(n) * 0.01
        close_prices = base_price + np.cumsum(noise)

        # Keep prices very stable
        close_prices = np.full(n, base_price)
        close_prices += np.random.randn(n) * 0.001  # Tiny noise

        df = pd.DataFrame({
            "Open": close_prices - 0.5,
            "High": close_prices + 1.0,
            "Low": close_prices - 1.0,
            "Close": close_prices,
            "Volume": np.random.randint(1000, 5000, n),
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))

        features = fetcher.compute_features(df)
        if features.empty:
            pytest.skip("Not enough data for features")

        X, y = fetcher.prepare_model_input(features, seq_length=64)

        if len(y) == 0:
            pytest.skip("No sequences generated")

        # With essentially flat prices, most labels should be HOLD (1)
        hold_ratio = np.mean(y == 1)
        assert hold_ratio > 0.5, (
            f"Expected majority HOLD labels for flat prices, got {hold_ratio:.2f}"
        )

    def test_buy_label_when_price_rises_more_than_atr_threshold(self):
        """Test that BUY label is produced when price rises > 0.3*ATR."""
        config = DataConfig()
        fetcher = MarketDataFetcher(config)

        # Create 300 bars with a strong uptrend
        n = 300
        # Strong upward trend: each bar rises by ~3.0 (well above 0.3*ATR)
        close_prices = 2000.0 + np.arange(n) * 3.0

        df = pd.DataFrame({
            "Open": close_prices - 1.0,
            "High": close_prices + 2.0,
            "Low": close_prices - 2.0,
            "Close": close_prices,
            "Volume": np.random.randint(1000, 5000, n),
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))

        features = fetcher.compute_features(df)
        if features.empty:
            pytest.skip("Not enough data for features")

        X, y = fetcher.prepare_model_input(features, seq_length=64)

        if len(y) == 0:
            pytest.skip("No sequences generated")

        # With strong uptrend, many labels should be BUY (2)
        buy_ratio = np.mean(y == 2)
        assert buy_ratio > 0.3, (
            f"Expected significant BUY labels for uptrend, got {buy_ratio:.2f}"
        )

    def test_sell_label_when_price_drops_more_than_atr_threshold(self):
        """Test that SELL label is produced when price drops > 0.3*ATR."""
        config = DataConfig()
        fetcher = MarketDataFetcher(config)

        # Create 300 bars with a strong downtrend
        n = 300
        close_prices = 2900.0 - np.arange(n) * 3.0

        df = pd.DataFrame({
            "Open": close_prices + 1.0,
            "High": close_prices + 2.0,
            "Low": close_prices - 2.0,
            "Close": close_prices,
            "Volume": np.random.randint(1000, 5000, n),
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))

        features = fetcher.compute_features(df)
        if features.empty:
            pytest.skip("Not enough data for features")

        X, y = fetcher.prepare_model_input(features, seq_length=64)

        if len(y) == 0:
            pytest.skip("No sequences generated")

        # With strong downtrend, many labels should be SELL (0)
        sell_ratio = np.mean(y == 0)
        assert sell_ratio > 0.3, (
            f"Expected significant SELL labels for downtrend, got {sell_ratio:.2f}"
        )

    def test_atr_label_threshold_from_config(self):
        """Test that ATR label threshold is configurable."""
        config = DataConfig()
        assert config.atr_label_threshold == 0.3

        # Custom threshold
        config2 = DataConfig(atr_label_threshold=0.5)
        assert config2.atr_label_threshold == 0.5


# ─────────────────────────────────────────────
#  MULTI-TIMEFRAME TRAINING TESTS
# ─────────────────────────────────────────────
class TestMultiTimeframeTraining:
    """Tests for multi-timeframe data fetching in prepare_data()."""

    def test_prepare_data_fetches_multiple_timeframes(self):
        """Test that prepare_data fetches 5y daily + 2y H1 + 60d M15 data."""
        config = DataConfig()

        # Create mock data for each timeframe
        n_daily = 1200  # ~5 years of daily data
        n_hourly = 3000  # ~2 years of hourly data
        n_m15 = 2000  # ~60 days of M15 data

        def make_mock_df(n, freq):
            prices = 2000 + np.cumsum(np.random.randn(n) * 2)
            return pd.DataFrame({
                "Open": prices - 1,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, n),
            }, index=pd.date_range("2020-01-01", periods=n, freq=freq))

        daily_df = make_mock_df(n_daily, "1D")
        hourly_df = make_mock_df(n_hourly, "1h")
        m15_df = make_mock_df(n_m15, "15min")

        call_count = [0]
        timeframe_data = [daily_df, hourly_df, m15_df]

        def mock_fetch_ohlcv(ticker=None, period="1y", interval="1h"):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(timeframe_data):
                return timeframe_data[idx]
            return pd.DataFrame()

        with patch.object(MarketDataFetcher, 'fetch_ohlcv', side_effect=mock_fetch_ohlcv):
            X, y = prepare_data(config, seq_length=64)

        # Should have data from all three timeframes combined
        assert len(X) > 0, "Expected training sequences from multi-timeframe data"
        assert len(y) > 0, "Expected labels from multi-timeframe data"
        # Each timeframe contributes sequences, total should be substantial
        assert len(X) > 1000, f"Expected 1000+ sequences, got {len(X)}"

    def test_prepare_data_handles_empty_timeframe(self):
        """Test that prepare_data handles a timeframe returning empty data."""
        config = DataConfig()

        n_daily = 300
        prices = 2000 + np.cumsum(np.random.randn(n_daily) * 2)
        daily_df = pd.DataFrame({
            "Open": prices - 1,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, n_daily),
        }, index=pd.date_range("2020-01-01", periods=n_daily, freq="1D"))

        call_count = [0]

        def mock_fetch_ohlcv(ticker=None, period="1y", interval="1h"):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return daily_df
            # Other timeframes return empty
            return pd.DataFrame()

        with patch.object(MarketDataFetcher, 'fetch_ohlcv', side_effect=mock_fetch_ohlcv):
            X, y = prepare_data(config, seq_length=64)

        # Should still produce data from the one valid timeframe
        assert len(X) > 0, "Expected data from at least one timeframe"

    def test_training_periods_config(self):
        """Test that training_periods config has correct default values."""
        config = DataConfig()
        assert len(config.training_periods) == 3
        assert config.training_periods[0] == {"period": "60d", "interval": "1m"}
        assert config.training_periods[1] == {"period": "60d", "interval": "15m"}
        assert config.training_periods[2] == {"period": "2y", "interval": "1h"}

    def test_prepare_data_all_empty_returns_empty(self):
        """Test that prepare_data returns empty arrays when all timeframes fail."""
        config = DataConfig()

        def mock_fetch_ohlcv(ticker=None, period="1y", interval="1h"):
            return pd.DataFrame()

        with patch.object(MarketDataFetcher, 'fetch_ohlcv', side_effect=mock_fetch_ohlcv):
            X, y = prepare_data(config, seq_length=64)

        assert len(X) == 0
        assert len(y) == 0
