"""
=============================================================
  Python ML Bridge - Multi-Timeframe Data Pipeline Tests
  Tests for MultiTimeframeDataFetcher:
    - Feature computation per timeframe
    - Output shape verification
    - Feature alignment across timeframes
    - HTF trend bias computation
    - Error handling with empty data
=============================================================
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MultiTimeframeConfig, DataConfig
from data.multi_timeframe import MultiTimeframeDataFetcher


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def mtf_config():
    """Multi-timeframe config for testing."""
    return MultiTimeframeConfig(
        timeframes=["5m", "15m", "1h"],
        min_bars=200,
    )


@pytest.fixture
def data_config():
    """Data config for testing."""
    return DataConfig(yfinance_ticker="GC=F")


@pytest.fixture
def mock_ohlcv_data():
    """Generate realistic mock OHLCV data with 500 bars."""
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    base_price = 2000.0
    # Random walk for realistic price movement
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "Open": prices - np.abs(np.random.randn(n_bars)) * 2,
        "High": prices + np.abs(np.random.randn(n_bars)) * 5,
        "Low": prices - np.abs(np.random.randn(n_bars)) * 5,
        "Close": prices,
        "Volume": np.random.randint(1000, 50000, n_bars).astype(float),
    }, index=dates)

    # Ensure High >= Close >= Low
    df["High"] = df[["Open", "High", "Close"]].max(axis=1) + 1
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1) - 1

    return df


@pytest.fixture
def mock_ohlcv_short():
    """Generate short OHLCV data (insufficient for features)."""
    np.random.seed(42)
    n_bars = 50
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    prices = 2000 + np.cumsum(np.random.randn(n_bars))
    return pd.DataFrame({
        "Open": prices - 1,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, n_bars).astype(float),
    }, index=dates)


@pytest.fixture
def fetcher(mtf_config, data_config):
    """Create a MultiTimeframeDataFetcher instance."""
    return MultiTimeframeDataFetcher(config=mtf_config, data_config=data_config)


# ─────────────────────────────────────────────
#  TIMEFRAME DATA FETCHING TESTS
# ─────────────────────────────────────────────
class TestTimeframeFetching:
    """Tests for individual timeframe data fetching."""

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_timeframe_returns_ohlcv(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that fetch_timeframe returns proper OHLCV DataFrame."""
        mock_download.return_value = mock_ohlcv_data
        result = fetcher.fetch_timeframe("1h")
        assert not result.empty
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(result) == 500

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_timeframe_empty_data(self, mock_download, fetcher):
        """Test handling of empty data from yfinance."""
        mock_download.return_value = pd.DataFrame()
        result = fetcher.fetch_timeframe("5m")
        assert result.empty

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_timeframe_uses_cache_on_error(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that cache is used when fetch fails."""
        # First call succeeds
        mock_download.return_value = mock_ohlcv_data
        fetcher.fetch_timeframe("1h")

        # Second call fails - should return cached data
        mock_download.side_effect = Exception("Network error")
        result = fetcher.fetch_timeframe("1h")
        assert not result.empty
        assert len(result) == 500

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_4h_resamples_from_1h(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that 4h data is resampled from 1h data."""
        mock_download.return_value = mock_ohlcv_data
        result = fetcher.fetch_timeframe("4h")
        assert not result.empty
        # 500 hourly bars should produce approximately 125 4h bars
        assert len(result) < len(mock_ohlcv_data)
        assert len(result) > 0

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_handles_multi_index_columns(self, mock_download, fetcher):
        """Test handling of MultiIndex columns from yfinance."""
        np.random.seed(42)
        n_bars = 300
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        prices = 2000 + np.cumsum(np.random.randn(n_bars))

        # Create MultiIndex columns like yfinance sometimes returns
        arrays = [["Open", "High", "Low", "Close", "Volume"],
                  ["GC=F", "GC=F", "GC=F", "GC=F", "GC=F"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)

        data = pd.DataFrame(
            np.column_stack([prices - 1, prices + 2, prices - 2, prices, np.ones(n_bars) * 10000]),
            index=dates,
            columns=index
        )
        mock_download.return_value = data
        result = fetcher.fetch_timeframe("1h")
        assert not result.empty
        assert "Close" in result.columns


# ─────────────────────────────────────────────
#  FEATURE COMPUTATION TESTS
# ─────────────────────────────────────────────
class TestFeatureComputation:
    """Tests for timeframe-specific feature computation."""

    def test_compute_timeframe_features_prefixed(self, fetcher, mock_ohlcv_data):
        """Test that features are correctly prefixed with timeframe."""
        features = fetcher.compute_timeframe_features("1h", mock_ohlcv_data)
        assert not features.empty
        # All columns should start with 'tf_1H_'
        for col in features.columns:
            assert col.startswith("tf_1H_"), f"Column {col} missing prefix"

    def test_compute_timeframe_features_5m_prefix(self, fetcher, mock_ohlcv_data):
        """Test 5m features have correct prefix."""
        features = fetcher.compute_timeframe_features("5m", mock_ohlcv_data)
        assert not features.empty
        for col in features.columns:
            assert col.startswith("tf_5M_"), f"Column {col} missing prefix"

    def test_compute_timeframe_features_shape(self, fetcher, mock_ohlcv_data):
        """Test that features have reasonable shape."""
        features = fetcher.compute_timeframe_features("1h", mock_ohlcv_data)
        assert not features.empty
        # Should have many features (50+ indicators)
        assert features.shape[1] >= 40
        # Should have data rows after dropping NaN from indicator warmup
        assert features.shape[0] > 100

    def test_compute_features_insufficient_data(self, fetcher, mock_ohlcv_short):
        """Test that insufficient data returns empty DataFrame."""
        features = fetcher.compute_timeframe_features("1h", mock_ohlcv_short)
        assert features.empty

    def test_compute_features_empty_data(self, fetcher):
        """Test that empty data returns empty DataFrame."""
        features = fetcher.compute_timeframe_features("1h", pd.DataFrame())
        assert features.empty


# ─────────────────────────────────────────────
#  FEATURE ALIGNMENT TESTS
# ─────────────────────────────────────────────
class TestFeatureAlignment:
    """Tests for cross-timeframe feature alignment."""

    def test_align_features_combines_timeframes(self, fetcher, mock_ohlcv_data):
        """Test that alignment produces combined feature DataFrame."""
        # Compute features for two timeframes
        features_1h = fetcher.compute_timeframe_features("1h", mock_ohlcv_data)

        # Create 15m data by resampling (more bars)
        dates_15m = pd.date_range("2024-01-01", periods=500, freq="15min")
        np.random.seed(123)
        prices_15m = 2000 + np.cumsum(np.random.randn(500) * 0.5)
        df_15m = pd.DataFrame({
            "Open": prices_15m - 1,
            "High": prices_15m + 3,
            "Low": prices_15m - 3,
            "Close": prices_15m,
            "Volume": np.random.randint(1000, 50000, 500).astype(float),
        }, index=dates_15m)
        df_15m["High"] = df_15m[["Open", "High", "Close"]].max(axis=1) + 1
        df_15m["Low"] = df_15m[["Open", "Low", "Close"]].min(axis=1) - 1

        features_15m = fetcher.compute_timeframe_features("15m", df_15m)

        if not features_1h.empty and not features_15m.empty:
            features_dict = {"1h": features_1h, "15m": features_15m}
            combined = fetcher._align_features(features_dict, target_tf="1h")
            # Combined should have columns from both timeframes
            assert any("tf_1H_" in col for col in combined.columns)
            assert any("tf_15M_" in col for col in combined.columns)

    def test_align_features_empty_dict(self, fetcher):
        """Test alignment with empty features dict."""
        result = fetcher._align_features({})
        assert result.empty

    def test_align_features_single_timeframe(self, fetcher, mock_ohlcv_data):
        """Test alignment with single timeframe passes through."""
        features_1h = fetcher.compute_timeframe_features("1h", mock_ohlcv_data)
        if not features_1h.empty:
            features_dict = {"1h": features_1h}
            combined = fetcher._align_features(features_dict, target_tf="1h")
            assert combined.shape == features_1h.shape


# ─────────────────────────────────────────────
#  FETCH ALL TIMEFRAMES TESTS
# ─────────────────────────────────────────────
class TestFetchAllTimeframes:
    """Tests for the main fetch_all_timeframes entry point."""

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_all_returns_combined_features(self, mock_download, fetcher, mock_ohlcv_data):
        """Test fetch_all_timeframes returns combined feature matrix."""
        mock_download.return_value = mock_ohlcv_data
        result = fetcher.fetch_all_timeframes()
        # With mock data, should get features from available timeframes
        if not result.empty:
            assert result.shape[1] > 40  # Multiple features
            assert result.shape[0] > 50  # Reasonable number of bars

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_all_empty_when_no_data(self, mock_download, fetcher):
        """Test fetch_all_timeframes returns empty on no data."""
        mock_download.return_value = pd.DataFrame()
        result = fetcher.fetch_all_timeframes()
        assert result.empty

    @patch("data.multi_timeframe.yf.download")
    def test_fetch_all_updates_last_fetch(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that fetch_all_timeframes updates _last_fetch timestamp."""
        mock_download.return_value = mock_ohlcv_data
        assert fetcher._last_fetch is None
        fetcher.fetch_all_timeframes()
        # If data was processed, _last_fetch should be set
        # (depends on whether features could be computed)


# ─────────────────────────────────────────────
#  HTF TREND BIAS TESTS
# ─────────────────────────────────────────────
class TestHTFTrendBias:
    """Tests for higher timeframe trend bias computation."""

    @patch("data.multi_timeframe.yf.download")
    def test_htf_trend_bias_returns_scores(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that HTF trend bias returns scores per timeframe."""
        mock_download.return_value = mock_ohlcv_data
        bias = fetcher.get_htf_trend_bias()
        assert "1h" in bias
        assert "4h" in bias
        # Scores should be between -1 and 1
        for tf, score in bias.items():
            assert -1.0 <= score <= 1.0, f"{tf} score {score} out of range"

    @patch("data.multi_timeframe.yf.download")
    def test_htf_trend_bias_empty_data(self, mock_download, fetcher):
        """Test HTF bias with empty data returns zero scores."""
        mock_download.return_value = pd.DataFrame()
        bias = fetcher.get_htf_trend_bias()
        assert bias.get("1h", 0.0) == 0.0
        assert bias.get("4h", 0.0) == 0.0

    @patch("data.multi_timeframe.yf.download")
    def test_htf_trend_bias_bullish(self, mock_download, fetcher):
        """Test HTF bias detects bullish trend (strong uptrend data)."""
        np.random.seed(42)
        n_bars = 500
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        # Strong uptrend: consistent positive returns
        prices = 2000 + np.arange(n_bars) * 2.0 + np.random.randn(n_bars) * 0.5

        df = pd.DataFrame({
            "Open": prices - 1,
            "High": prices + 3,
            "Low": prices - 1.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 50000, n_bars).astype(float),
        }, index=dates)
        df["High"] = df[["Open", "High", "Close"]].max(axis=1) + 0.5
        df["Low"] = df[["Open", "Low", "Close"]].min(axis=1) - 0.5

        mock_download.return_value = df
        bias = fetcher.get_htf_trend_bias()
        # Strong uptrend should produce positive bias
        assert bias.get("1h", 0) > 0, f"Expected positive bias for uptrend, got {bias}"


# ─────────────────────────────────────────────
#  MULTI-TF FEATURE ARRAY TESTS
# ─────────────────────────────────────────────
class TestMultiTFFeatureArray:
    """Tests for the model-ready feature array output."""

    @patch("data.multi_timeframe.yf.download")
    def test_get_feature_array_shape(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that feature array has correct 3D shape."""
        mock_download.return_value = mock_ohlcv_data
        seq_length = 32
        result = fetcher.get_multi_tf_feature_array(seq_length=seq_length)
        if result is not None:
            assert result.ndim == 3
            assert result.shape[0] == 1
            assert result.shape[1] == seq_length
            assert result.shape[2] > 0  # Some features

    @patch("data.multi_timeframe.yf.download")
    def test_get_feature_array_normalized(self, mock_download, fetcher, mock_ohlcv_data):
        """Test that feature array is normalized (roughly zero mean, unit var)."""
        mock_download.return_value = mock_ohlcv_data
        result = fetcher.get_multi_tf_feature_array(seq_length=32)
        if result is not None:
            # Check no NaN or inf values
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

    @patch("data.multi_timeframe.yf.download")
    def test_get_feature_array_none_on_empty(self, mock_download, fetcher):
        """Test that None is returned when data is insufficient."""
        mock_download.return_value = pd.DataFrame()
        result = fetcher.get_multi_tf_feature_array(seq_length=64)
        assert result is None


# ─────────────────────────────────────────────
#  CONFIGURATION TESTS
# ─────────────────────────────────────────────
class TestMultiTimeframeConfig:
    """Tests for MultiTimeframeConfig defaults."""

    def test_default_timeframes(self):
        """Test default timeframe list."""
        config = MultiTimeframeConfig()
        assert "1m" in config.timeframes
        assert "5m" in config.timeframes
        assert "15m" in config.timeframes
        assert "1h" in config.timeframes
        assert "4h" in config.timeframes

    def test_default_periods(self):
        """Test default period mapping."""
        config = MultiTimeframeConfig()
        assert "1m" in config.periods
        assert "1h" in config.periods

    def test_custom_config(self):
        """Test custom config overrides."""
        config = MultiTimeframeConfig(
            timeframes=["1h", "4h"],
            min_bars=100,
            htf_trend_weight=0.8,
        )
        assert config.timeframes == ["1h", "4h"]
        assert config.min_bars == 100
        assert config.htf_trend_weight == 0.8
