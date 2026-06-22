"""
=============================================================
  Python ML Bridge - Market Regime Detector
  Classifies market regimes using volatility clustering and
  trend strength: trending, ranging, volatile, crash.
  Uses simplified HMM-like approach for regime detection.
=============================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from enum import IntEnum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RegimeConfig


class MarketRegime(IntEnum):
    """Market regime classifications."""
    TRENDING = 0
    RANGING = 1
    VOLATILE = 2
    CRASH = 3


class RegimeDetector:
    """
    Detects current market regime using volatility and trend metrics.

    Regimes:
        - TRENDING: Strong directional movement (high ADX, low volatility ratio)
        - RANGING: Sideways movement (low ADX, mean-reverting)
        - VOLATILE: High volatility expansion (VIX spike, wide ranges)
        - CRASH: Extreme downside volatility (rapid drops, fear indicators)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._regime_history: list = []
        self._current_regime = MarketRegime.RANGING

    def detect_regime(self, prices: pd.DataFrame,
                      adx: Optional[pd.Series] = None,
                      atr: Optional[pd.Series] = None,
                      vix: Optional[float] = None) -> Dict:
        """
        Detect current market regime.

        Args:
            prices: DataFrame with at least 'Close' column
            adx: ADX indicator series (optional)
            atr: ATR indicator series (optional)
            vix: Current VIX level (optional)

        Returns:
            Dict with 'regime', 'confidence', 'details'
        """
        if prices.empty or len(prices) < self.config.lookback_bars:
            return {
                "regime": MarketRegime.RANGING,
                "regime_name": "ranging",
                "confidence": 0.5,
                "details": {}
            }

        close = prices["Close"].values[-self.config.lookback_bars:]
        returns = np.diff(close) / close[:-1]

        # Compute regime indicators
        volatility = np.std(returns[-20:])
        avg_volatility = np.std(returns)
        vol_ratio = volatility / (avg_volatility + 1e-10)

        # Trend strength (using price momentum)
        trend_strength = abs(close[-1] - close[-20]) / (np.std(close[-20:]) + 1e-10)

        # Directional bias
        up_moves = np.sum(returns[-20:] > 0)
        down_moves = np.sum(returns[-20:] < 0)
        directional_bias = abs(up_moves - down_moves) / 20.0

        # ADX-based trend detection
        adx_value = float(adx.iloc[-1]) if adx is not None and not adx.empty else 20.0

        # Crash detection
        max_drawdown = self._compute_max_drawdown(close[-20:])
        consecutive_down = self._count_consecutive_direction(returns[-10:], negative=True)

        # Regime scoring
        scores = {
            MarketRegime.TRENDING: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CRASH: 0.0,
        }

        # Trending signals
        if adx_value > self.config.trend_strength_threshold:
            scores[MarketRegime.TRENDING] += 0.4
        if trend_strength > 1.5:
            scores[MarketRegime.TRENDING] += 0.3
        if directional_bias > 0.3:
            scores[MarketRegime.TRENDING] += 0.3

        # Ranging signals
        if adx_value < 20:
            scores[MarketRegime.RANGING] += 0.4
        if vol_ratio < 1.0:
            scores[MarketRegime.RANGING] += 0.3
        if trend_strength < 0.5:
            scores[MarketRegime.RANGING] += 0.3

        # Volatile signals
        if vol_ratio > self.config.volatility_threshold:
            scores[MarketRegime.VOLATILE] += 0.5
        if vix is not None and vix > 25:
            scores[MarketRegime.VOLATILE] += 0.3
        if np.max(np.abs(returns[-5:])) > 2 * avg_volatility:
            scores[MarketRegime.VOLATILE] += 0.2

        # Crash signals
        if max_drawdown > 0.03:  # 3% drawdown in 20 bars
            scores[MarketRegime.CRASH] += 0.4
        if consecutive_down >= 5:
            scores[MarketRegime.CRASH] += 0.3
        if vix is not None and vix > 35:
            scores[MarketRegime.CRASH] += 0.3

        # Determine regime
        regime = max(scores, key=scores.get)
        confidence = scores[regime] / (sum(scores.values()) + 1e-10)

        self._current_regime = regime
        self._regime_history.append(regime)

        return {
            "regime": regime,
            "regime_name": self.config.regime_names[int(regime)],
            "confidence": float(confidence),
            "details": {
                "volatility_ratio": float(vol_ratio),
                "trend_strength": float(trend_strength),
                "adx": float(adx_value),
                "max_drawdown": float(max_drawdown),
                "scores": {k.name: float(v) for k, v in scores.items()},
            }
        }

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get strategy parameter adjustments for the current regime.

        Args:
            regime: Current market regime

        Returns:
            Dict with multipliers for various strategy parameters
        """
        adjustments = {
            MarketRegime.TRENDING: {
                "position_size_mult": 1.2,
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "confidence_threshold": 0.15,
                "prefer_trend_following": True,
            },
            MarketRegime.RANGING: {
                "position_size_mult": 0.8,
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "confidence_threshold": 0.15,
                "prefer_trend_following": False,
            },
            MarketRegime.VOLATILE: {
                "position_size_mult": 0.5,
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "confidence_threshold": 0.20,
                "prefer_trend_following": False,
            },
            MarketRegime.CRASH: {
                "position_size_mult": 0.3,
                "sl_mult": 1.0,
                "tp_mult": 1.0,
                "confidence_threshold": 0.25,
                "prefer_trend_following": True,
            },
        }
        return adjustments.get(regime, adjustments[MarketRegime.RANGING])

    def _compute_max_drawdown(self, prices: np.ndarray) -> float:
        """Compute maximum drawdown from a price array."""
        if len(prices) < 2:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for price in prices[1:]:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    def _count_consecutive_direction(self, returns: np.ndarray,
                                     negative: bool = True) -> int:
        """Count maximum consecutive moves in one direction."""
        max_count = 0
        count = 0
        for r in returns:
            if (negative and r < 0) or (not negative and r > 0):
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    @property
    def current_regime(self) -> MarketRegime:
        """Get the last detected regime."""
        return self._current_regime

    @property
    def regime_name(self) -> str:
        """Get the name of the current regime."""
        return self.config.regime_names[int(self._current_regime)]
