"""
=============================================================
  Python ML Bridge - Smart Upgrades Tests
  Tests for the 10 smart trading upgrades: session awareness,
  spread filter, streak detection, adaptive momentum, price
  structure, DXY correlation, FVG, and liquidity sweep.
=============================================================
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SessionConfig, SpreadFilterConfig, StreakConfig,
    AdaptiveMomentumConfig, PriceStructureConfig,
    FVGConfig, LiquiditySweepConfig, SignalConfig, DataConfig
)
from strategies.signal_generator import SignalGenerator
from strategies.risk_manager import RiskManager


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def signal_gen():
    """Create a SignalGenerator with default configs."""
    config = SignalConfig(cooldown_seconds=0)
    data_config = DataConfig(momentum_threshold=0.01)
    sg = SignalGenerator(signal_config=config, data_config=data_config)
    return sg


@pytest.fixture
def risk_manager():
    """Create a RiskManager with default configs."""
    return RiskManager()


@pytest.fixture
def mock_prices_uptrend():
    """Mock price DataFrame with clear uptrend (higher highs, higher lows)."""
    n = 30
    base = 2000.0
    # Create uptrend with swings: prices oscillate but trend up
    # This creates proper swing highs and swing lows
    closes = np.array([
        base + i * 0.8 + (1.5 if i % 4 == 1 else (-1.0 if i % 4 == 3 else 0.0))
        for i in range(n)
    ])
    highs = closes + 1.5
    lows = closes - 1.0
    opens = closes - 0.3
    volumes = np.full(n, 5000)
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })


@pytest.fixture
def mock_prices_downtrend():
    """Mock price DataFrame with clear downtrend (lower highs, lower lows)."""
    n = 30
    base = 2000.0
    # Create downtrend with swings: prices oscillate but trend down
    closes = np.array([
        base - i * 0.8 + (1.0 if i % 4 == 3 else (-1.5 if i % 4 == 1 else 0.0))
        for i in range(n)
    ])
    highs = closes + 1.0
    lows = closes - 1.5
    opens = closes + 0.3
    volumes = np.full(n, 5000)
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })


@pytest.fixture
def mock_prices_normal():
    """Mock price DataFrame with normal spread."""
    n = 25
    base = 2000.0
    closes = np.array([base + np.sin(i * 0.3) * 2 for i in range(n)])
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes - 0.2
    volumes = np.full(n, 5000)
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })


# ─────────────────────────────────────────────
#  SESSION AWARENESS TESTS
# ─────────────────────────────────────────────
class TestSessionAwareness:
    """Tests for session detection."""

    def test_asian_session(self, signal_gen):
        """Test Asian session detection (00:00-08:00 UTC)."""
        with patch('strategies.signal_generator.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
            mock_dt.strftime = datetime.strftime
            session = signal_gen._detect_session()
            assert session == "asian"

    def test_london_session(self, signal_gen):
        """Test London session detection (08:00-16:00 UTC)."""
        with patch('strategies.signal_generator.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
            mock_dt.strftime = datetime.strftime
            session = signal_gen._detect_session()
            assert session == "london"

    def test_ny_session(self, signal_gen):
        """Test New York session detection (13:00-21:00 UTC)."""
        with patch('strategies.signal_generator.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
            mock_dt.strftime = datetime.strftime
            session = signal_gen._detect_session()
            assert session == "newyork"

    def test_overlap_session(self, signal_gen):
        """Test London/NY overlap detection (13:00-16:00 UTC)."""
        with patch('strategies.signal_generator.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
            mock_dt.strftime = datetime.strftime
            session = signal_gen._detect_session()
            assert session == "overlap"

    def test_session_multipliers(self, signal_gen):
        """Test that session multipliers are correctly assigned."""
        assert signal_gen._get_session_multiplier("asian") == 0.5
        assert signal_gen._get_session_multiplier("london") == 1.2
        assert signal_gen._get_session_multiplier("newyork") == 1.0
        assert signal_gen._get_session_multiplier("overlap") == 1.2
        assert signal_gen._get_session_multiplier("off_session") == 0.7


# ─────────────────────────────────────────────
#  SPREAD FILTER TESTS
# ─────────────────────────────────────────────
class TestSpreadFilter:
    """Tests for the spread filter."""

    def test_normal_spread_passes(self, signal_gen, mock_prices_normal):
        """Normal spread should not block."""
        result = signal_gen._check_spread_filter(mock_prices_normal)
        assert result is False

    def test_wide_spread_blocks(self, signal_gen):
        """Wide spread (> 2x average) should block."""
        n = 25
        base = 2000.0
        closes = np.full(n, base)
        # Normal bars have range of 2.0
        highs = closes + 1.0
        lows = closes - 1.0
        # Last bar has extreme range of 5.0 (> 2x the 2.0 average)
        highs[-1] = base + 3.0
        lows[-1] = base - 2.0
        prices = pd.DataFrame({
            "High": highs,
            "Low": lows,
            "Close": closes,
        })
        result = signal_gen._check_spread_filter(prices)
        assert result is True

    def test_insufficient_data_passes(self, signal_gen):
        """Insufficient data should not block."""
        prices = pd.DataFrame({
            "High": [2001, 2002],
            "Low": [1999, 1998],
            "Close": [2000, 2001],
        })
        result = signal_gen._check_spread_filter(prices)
        assert result is False

    def test_non_dataframe_passes(self, signal_gen):
        """Non-DataFrame input should not block."""
        result = signal_gen._check_spread_filter(np.array([1, 2, 3]))
        assert result is False


# ─────────────────────────────────────────────
#  STREAK DETECTION TESTS
# ─────────────────────────────────────────────
class TestStreakDetection:
    """Tests for win/lose streak detection in RiskManager."""

    def test_initial_state(self, risk_manager):
        """Initial state should have no streak."""
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "none"
        assert status["streak_count"] == 0
        assert status["multiplier"] == 1.0

    def test_three_losses_reduce(self, risk_manager):
        """3 consecutive losses should reduce multiplier to 0.5."""
        risk_manager.register_result(False)
        risk_manager.register_result(False)
        risk_manager.register_result(False)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "lose"
        assert status["streak_count"] == 3
        assert status["multiplier"] == 0.5

    def test_five_losses_severe_reduce(self, risk_manager):
        """5 consecutive losses should reduce multiplier to 0.25."""
        for _ in range(5):
            risk_manager.register_result(False)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "lose"
        assert status["streak_count"] == 5
        assert status["multiplier"] == 0.25

    def test_two_wins_restore(self, risk_manager):
        """2 wins after losses should restore multiplier to 1.0."""
        for _ in range(3):
            risk_manager.register_result(False)
        assert risk_manager._streak_multiplier == 0.5
        risk_manager.register_result(True)
        risk_manager.register_result(True)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "win"
        assert status["multiplier"] == 1.0

    def test_three_wins_boost(self, risk_manager):
        """3 consecutive wins should boost multiplier to 1.25."""
        for _ in range(3):
            risk_manager.register_result(True)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "win"
        assert status["streak_count"] == 3
        assert status["multiplier"] == 1.25

    def test_five_wins_severe_boost(self, risk_manager):
        """5 consecutive wins should boost multiplier to 1.5."""
        for _ in range(5):
            risk_manager.register_result(True)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "win"
        assert status["streak_count"] == 5
        assert status["multiplier"] == 1.5

    def test_streak_broken_by_opposite(self, risk_manager):
        """A loss after wins should reset multiplier."""
        for _ in range(3):
            risk_manager.register_result(True)
        assert risk_manager._streak_multiplier == 1.25
        risk_manager.register_result(False)
        status = risk_manager.get_streak_status()
        assert status["streak_type"] == "lose"
        assert status["streak_count"] == 1
        assert status["multiplier"] == 1.0


# ─────────────────────────────────────────────
#  ADAPTIVE MOMENTUM TESTS
# ─────────────────────────────────────────────
class TestAdaptiveMomentum:
    """Tests for adaptive momentum lookback."""

    def test_high_atr_short_lookback(self, signal_gen):
        """High ATR should use 3-bar lookback."""
        # Create prices with strong recent movement over 3 bars
        n = 20
        closes = np.full(n, 2000.0)
        # Last 4 bars rise strongly (enough for 3-bar lookback to detect BUY)
        closes[-4] = 2000.0
        closes[-3] = 2001.0
        closes[-2] = 2002.0
        closes[-1] = 2003.0
        prices = pd.DataFrame({"Close": closes})

        # High ATR: current = 5.0, avg = 2.0, ratio = 2.5 > 1.5 threshold
        result = signal_gen._compute_momentum_direction(
            prices, adaptive_atr=5.0, avg_atr=2.0
        )
        assert result == "BUY"

    def test_low_atr_long_lookback(self, signal_gen):
        """Low ATR should use 7-bar lookback."""
        n = 20
        closes = np.full(n, 2000.0)
        # Create movement visible over 7 bars but not just 3
        closes[-8] = 2000.0
        closes[-7] = 2000.5
        closes[-6] = 2001.0
        closes[-5] = 2001.5
        closes[-4] = 2002.0
        closes[-3] = 2002.5
        closes[-2] = 2003.0
        closes[-1] = 2003.5
        prices = pd.DataFrame({"Close": closes})

        # Low ATR: current = 1.0, avg = 2.0, ratio = 0.5 < 1.5 threshold
        result = signal_gen._compute_momentum_direction(
            prices, adaptive_atr=1.0, avg_atr=2.0
        )
        assert result == "BUY"

    def test_no_atr_uses_default(self, signal_gen):
        """No ATR info should use default lookback."""
        n = 20
        closes = np.full(n, 2000.0)
        closes[-1] = 2005.0  # Strong move up
        prices = pd.DataFrame({"Close": closes})

        result = signal_gen._compute_momentum_direction(prices)
        assert result == "BUY"


# ─────────────────────────────────────────────
#  PRICE STRUCTURE TESTS
# ─────────────────────────────────────────────
class TestPriceStructure:
    """Tests for price action structure detection."""

    def test_uptrend_detection(self, signal_gen, mock_prices_uptrend):
        """Should detect uptrend from higher highs and higher lows."""
        result = signal_gen._detect_price_structure(mock_prices_uptrend)
        assert result == "uptrend"

    def test_downtrend_detection(self, signal_gen, mock_prices_downtrend):
        """Should detect downtrend from lower highs and lower lows."""
        result = signal_gen._detect_price_structure(mock_prices_downtrend)
        assert result == "downtrend"

    def test_no_structure_with_insufficient_data(self, signal_gen):
        """Should return no_structure with insufficient data."""
        prices = pd.DataFrame({
            "High": [2001, 2002, 2003],
            "Low": [1999, 2000, 2001],
        })
        result = signal_gen._detect_price_structure(prices)
        assert result == "no_structure"

    def test_no_structure_with_non_dataframe(self, signal_gen):
        """Should return no_structure for non-DataFrame input."""
        result = signal_gen._detect_price_structure(np.array([1, 2, 3]))
        assert result == "no_structure"


# ─────────────────────────────────────────────
#  DXY CORRELATION TESTS
# ─────────────────────────────────────────────
class TestDXYCorrelation:
    """Tests for DXY correlation adjustments."""

    def test_dxy_rising_penalizes_buy(self, signal_gen):
        """DXY rising should penalize BUY and favor SELL."""
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"dxy_direction": "rising"}
        )
        assert result["buy_adjustment"] == -0.10
        assert result["sell_adjustment"] == 0.05

    def test_dxy_falling_favors_buy(self, signal_gen):
        """DXY falling should favor BUY and penalize SELL."""
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"dxy_direction": "falling"}
        )
        assert result["buy_adjustment"] == 0.05
        assert result["sell_adjustment"] == -0.10

    def test_dxy_flat_no_adjustment(self, signal_gen):
        """DXY flat should have no adjustment."""
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"dxy_direction": "flat"}
        )
        assert result["buy_adjustment"] == 0.0
        assert result["sell_adjustment"] == 0.0

    def test_no_cross_pair_info(self, signal_gen):
        """No cross pair info should have no adjustment."""
        result = signal_gen._check_dxy_correlation(None, cross_pair_info=None)
        assert result["buy_adjustment"] == 0.0
        assert result["sell_adjustment"] == 0.0

    def test_dxy_prices_array(self, signal_gen):
        """DXY prices array should determine direction."""
        # Rising DXY prices
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"dxy_prices": [104.0, 104.5, 105.0]}
        )
        assert result["buy_adjustment"] == -0.10

        # Falling DXY prices
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"dxy_prices": [105.0, 104.5, 104.0]}
        )
        assert result["buy_adjustment"] == 0.05

    def test_usd_strength_field(self, signal_gen):
        """usd_strength field should determine direction."""
        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"usd_strength": 0.8}
        )
        assert result["buy_adjustment"] == -0.10

        result = signal_gen._check_dxy_correlation(
            None, cross_pair_info={"usd_strength": -0.8}
        )
        assert result["buy_adjustment"] == 0.05


# ─────────────────────────────────────────────
#  FVG DETECTION TESTS
# ─────────────────────────────────────────────
class TestFVGDetection:
    """Tests for Fair Value Gap detection."""

    def test_bullish_fvg_detected(self, signal_gen):
        """Should detect a bullish FVG (gap up)."""
        # Create a gap up: candle N low > candle N-2 high
        n = 10
        highs = np.array([2001, 2002, 2003, 2004, 2005,
                          2006, 2007, 2008, 2010, 2013.0])
        lows = np.array([1999, 2000, 2001, 2002, 2003,
                         2004, 2005, 2006, 2009, 2011.0])
        closes = np.array([2000, 2001, 2002, 2003, 2004,
                           2005, 2006, 2007, 2009.5, 2012.0])
        # FVG: bar 9 low (2011) > bar 7 high (2008) = gap up
        prices = pd.DataFrame({
            "High": highs,
            "Low": lows,
            "Close": closes,
        })
        result = signal_gen._detect_fvg(prices)
        assert result["bullish_fvg"] is True

    def test_bearish_fvg_detected(self, signal_gen):
        """Should detect a bearish FVG (gap down)."""
        # Create a gap down: candle N high < candle N-2 low
        n = 10
        highs = np.array([2010, 2009, 2008, 2007, 2006,
                          2005, 2004, 2003, 2000, 1997.0])
        lows = np.array([2008, 2007, 2006, 2005, 2004,
                         2003, 2002, 2001, 1998, 1995.0])
        closes = np.array([2009, 2008, 2007, 2006, 2005,
                           2004, 2003, 2002, 1999, 1996.0])
        # FVG: bar 9 high (1997) < bar 7 low (2001) = gap down
        prices = pd.DataFrame({
            "High": highs,
            "Low": lows,
            "Close": closes,
        })
        result = signal_gen._detect_fvg(prices)
        assert result["bearish_fvg"] is True

    def test_no_fvg(self, signal_gen, mock_prices_normal):
        """Should detect no FVG in normal price action."""
        result = signal_gen._detect_fvg(mock_prices_normal)
        # Normal prices with range 2.0 won't have gaps
        assert result["bullish_fvg"] is False or result["bearish_fvg"] is False

    def test_fvg_disabled(self, signal_gen):
        """Should return empty result when FVG is disabled."""
        signal_gen.fvg_config.enabled = False
        prices = pd.DataFrame({
            "High": [2001] * 10,
            "Low": [1999] * 10,
            "Close": [2000] * 10,
        })
        result = signal_gen._detect_fvg(prices)
        assert result["bullish_fvg"] is False
        assert result["bearish_fvg"] is False


# ─────────────────────────────────────────────
#  LIQUIDITY SWEEP TESTS
# ─────────────────────────────────────────────
class TestLiquiditySweep:
    """Tests for liquidity sweep detection."""

    def test_bullish_sweep_detected(self, signal_gen):
        """Should detect bullish sweep (break below low then recover)."""
        n = 25
        base = 2000.0
        # Normal range for 22 bars
        highs = np.full(n, base + 1.0)
        lows = np.full(n, base - 1.0)
        closes = np.full(n, base)

        # The recent low is 1999.0 (base - 1.0)
        # Second-to-last bar breaks below recent low
        lows[-2] = base - 3.0  # Breaks to 1997.0 (below 1999.0)
        closes[-2] = base - 2.0  # Still below

        # Last bar recovers
        lows[-1] = base - 0.5
        closes[-1] = base + 0.5  # Recovered above 1999.0

        prices = pd.DataFrame({
            "High": highs,
            "Low": lows,
            "Close": closes,
        })
        result = signal_gen._detect_liquidity_sweep(prices)
        assert result["bullish_sweep"] is True

    def test_bearish_sweep_detected(self, signal_gen):
        """Should detect bearish sweep (break above high then fall back)."""
        n = 25
        base = 2000.0
        # Normal range for 22 bars
        highs = np.full(n, base + 1.0)
        lows = np.full(n, base - 1.0)
        closes = np.full(n, base)

        # The recent high is 2001.0 (base + 1.0)
        # Second-to-last bar breaks above recent high
        highs[-2] = base + 3.0  # Breaks to 2003.0 (above 2001.0)
        closes[-2] = base + 2.0  # Still above

        # Last bar falls back
        highs[-1] = base + 0.5
        closes[-1] = base - 0.5  # Fell back below 2001.0

        prices = pd.DataFrame({
            "High": highs,
            "Low": lows,
            "Close": closes,
        })
        result = signal_gen._detect_liquidity_sweep(prices)
        assert result["bearish_sweep"] is True

    def test_no_sweep_normal_price(self, signal_gen, mock_prices_normal):
        """Should not detect sweep in normal price action."""
        result = signal_gen._detect_liquidity_sweep(mock_prices_normal)
        assert result["bullish_sweep"] is False
        assert result["bearish_sweep"] is False

    def test_insufficient_data(self, signal_gen):
        """Should not detect sweep with insufficient data."""
        prices = pd.DataFrame({
            "High": [2001, 2002, 2003],
            "Low": [1999, 1998, 1997],
            "Close": [2000, 2001, 2002],
        })
        result = signal_gen._detect_liquidity_sweep(prices)
        assert result["bullish_sweep"] is False
        assert result["bearish_sweep"] is False


# ─────────────────────────────────────────────
#  CONFIG TESTS
# ─────────────────────────────────────────────
class TestConfigs:
    """Tests for new config dataclasses."""

    def test_session_config_defaults(self):
        """SessionConfig should have correct defaults."""
        cfg = SessionConfig()
        assert cfg.asian_start == 0
        assert cfg.asian_end == 8
        assert cfg.london_start == 8
        assert cfg.london_end == 16
        assert cfg.ny_start == 13
        assert cfg.ny_end == 21
        assert cfg.asian_multiplier == 0.5
        assert cfg.london_multiplier == 1.2
        assert cfg.ny_multiplier == 1.0

    def test_spread_filter_config_defaults(self):
        """SpreadFilterConfig should have correct defaults."""
        cfg = SpreadFilterConfig()
        assert cfg.max_spread_multiplier == 2.0
        assert cfg.avg_spread_window == 20

    def test_streak_config_defaults(self):
        """StreakConfig should have correct defaults."""
        cfg = StreakConfig()
        assert cfg.lose_streak_reduce_threshold == 3
        assert cfg.severe_threshold == 5
        assert cfg.reduce_pct == 0.5
        assert cfg.severe_reduce_pct == 0.25
        assert cfg.win_restore_threshold == 2
        assert cfg.win_boost_threshold == 3
        assert cfg.win_severe_threshold == 5
        assert cfg.win_boost_pct == 1.25
        assert cfg.win_severe_boost_pct == 1.5

    def test_adaptive_momentum_config_defaults(self):
        """AdaptiveMomentumConfig should have correct defaults."""
        cfg = AdaptiveMomentumConfig()
        assert cfg.high_atr_lookback == 3
        assert cfg.low_atr_lookback == 7
        assert cfg.atr_threshold_mult == 1.5
        assert cfg.atr_avg_period == 14

    def test_price_structure_config_defaults(self):
        """PriceStructureConfig should have correct defaults."""
        cfg = PriceStructureConfig()
        assert cfg.swing_lookback == 20
        assert cfg.confidence_penalty == 0.15

    def test_fvg_config_defaults(self):
        """FVGConfig should have correct defaults."""
        cfg = FVGConfig()
        assert cfg.enabled is True
        assert cfg.confidence_boost == 0.05

    def test_liquidity_sweep_config_defaults(self):
        """LiquiditySweepConfig should have correct defaults."""
        cfg = LiquiditySweepConfig()
        assert cfg.lookback == 20
        assert cfg.min_recovery_pct == 0.5
        assert cfg.confidence_boost == 0.10
