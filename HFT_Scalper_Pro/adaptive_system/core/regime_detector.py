"""
Market Regime Detection Engine
===============================
Classifies market conditions into discrete regimes using multiple indicators.
Uses a rolling window approach with exponential weighting on recent bars.

Regimes:
    TRENDING_UP      - Strong upward trend (ADX high, DI+ > DI-, positive slope)
    TRENDING_DOWN    - Strong downward trend (ADX high, DI- > DI+, negative slope)
    RANGING_NARROW   - Low volatility sideways (low ADX, low volatility ratio)
    RANGING_WIDE     - Higher volatility sideways (low ADX, moderate volatility)
    VOLATILE_BREAKOUT - Volatility expansion from compression (spike in ATR ratio)
    MEAN_REVERTING   - Strong mean-reversion signal (Hurst < 0.4, low ADX)

Architecture:
    - Combines ADX (trend strength), volatility ratio, EMA slope, and Hurst exponent
    - Each indicator votes for regimes with weighted confidence
    - Final regime is the highest-confidence classification
    - Maintains transition history for pattern detection
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np

from .indicators import (
    compute_adx,
    compute_atr,
    compute_ema,
    compute_hurst_exponent,
    compute_volatility_ratio,
    compute_ema_slope,
)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING_NARROW = auto()
    RANGING_WIDE = auto()
    VOLATILE_BREAKOUT = auto()
    MEAN_REVERTING = auto()


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds."""
    # ADX thresholds
    adx_trend_threshold: float = 25.0   # ADX above this = trending
    adx_strong_trend: float = 40.0      # ADX above this = strong trend
    adx_weak_threshold: float = 18.0    # ADX below this = ranging/mean-reverting

    # Volatility ratio thresholds
    vol_ratio_high: float = 1.4         # Above = volatile/breakout
    vol_ratio_low: float = 0.7          # Below = compressed/narrow range
    vol_ratio_very_high: float = 2.0    # Above = definite breakout

    # EMA slope (ATR-normalized) thresholds
    slope_strong: float = 0.15          # Normalized slope above = strong trend direction
    slope_weak: float = 0.05            # Below = no directional bias

    # Hurst exponent thresholds
    hurst_mean_revert: float = 0.4      # Below = mean-reverting
    hurst_trending: float = 0.6         # Above = trending/persistent

    # Detection parameters
    ema_period: int = 20
    atr_period: int = 14
    adx_period: int = 14
    vol_lookback: int = 50
    slope_lookback: int = 5
    hurst_window: int = 100

    # Confidence decay for regime persistence
    regime_persistence_alpha: float = 0.85  # How much weight on current vs previous


@dataclass
class RegimeState:
    """Internal state of the regime detector."""
    current_regime: MarketRegime = MarketRegime.RANGING_NARROW
    confidence: float = 0.5
    regime_duration: int = 0  # Bars in current regime
    adx_value: float = 0.0
    volatility_ratio: float = 1.0
    ema_slope_norm: float = 0.0
    hurst_value: float = 0.5


@dataclass
class RegimeHistory:
    """Tracks regime transitions for pattern detection."""
    transitions: List[Tuple[MarketRegime, float, int]] = field(default_factory=list)
    max_history: int = 200

    def add_transition(self, regime: MarketRegime, confidence: float, bar_index: int):
        """Record a regime transition."""
        self.transitions.append((regime, confidence, bar_index))
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history:]

    def get_last_n(self, n: int = 10) -> List[Tuple[MarketRegime, float, int]]:
        """Get the last N regime transitions."""
        return self.transitions[-n:]

    def regime_frequency(self, lookback: int = 50) -> dict:
        """Count regime occurrences in recent history."""
        recent = self.transitions[-lookback:]
        counts = {}
        for regime, _, _ in recent:
            counts[regime] = counts.get(regime, 0) + 1
        return counts

    def avg_duration(self, regime: MarketRegime, lookback: int = 20) -> float:
        """Compute average duration of a specific regime in recent transitions."""
        recent = self.transitions[-lookback:]
        durations = []
        for i in range(1, len(recent)):
            if recent[i - 1][0] == regime:
                duration = recent[i][2] - recent[i - 1][2]
                if duration > 0:
                    durations.append(duration)
        return np.mean(durations) if durations else 0.0


class RegimeDetector:
    """
    Market regime classification engine.

    Combines multiple technical indicators to classify the current market
    into one of six regimes with a confidence score.

    Usage:
        detector = RegimeDetector()
        regime, confidence = detector.detect(high, low, close, volume)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.state = RegimeState()
        self.history = RegimeHistory()
        self._bar_count = 0

    def detect(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
               volume: Optional[np.ndarray] = None) -> Tuple[MarketRegime, float]:
        """
        Detect the current market regime from OHLCV data.

        Parameters
        ----------
        high : np.ndarray
            High prices (full history or sliding window).
        low : np.ndarray
            Low prices.
        close : np.ndarray
            Close prices.
        volume : np.ndarray, optional
            Volume data (not required for regime detection).

        Returns
        -------
        Tuple[MarketRegime, float]
            (detected_regime, confidence_score) where confidence is 0-1.
        """
        n = len(close)
        if n < self.config.hurst_window:
            # Not enough data - return default with low confidence
            return MarketRegime.RANGING_NARROW, 0.3

        cfg = self.config

        # Compute indicators
        adx, di_plus, di_minus = compute_adx(high, low, close, cfg.adx_period)
        atr = compute_atr(high, low, close, cfg.atr_period)
        ema = compute_ema(close, cfg.ema_period)
        vol_ratio = compute_volatility_ratio(atr, cfg.vol_lookback)
        slope = compute_ema_slope(ema, atr, cfg.slope_lookback)

        # Hurst exponent on the last hurst_window bars
        hurst_segment = close[-cfg.hurst_window:]
        hurst = compute_hurst_exponent(hurst_segment, max_lag=40)

        # Get latest values
        idx = n - 1
        cur_adx = adx[idx]
        cur_di_plus = di_plus[idx]
        cur_di_minus = di_minus[idx]
        cur_vol_ratio = vol_ratio[idx]
        cur_slope = slope[idx]
        cur_hurst = hurst

        # Store in state
        self.state.adx_value = cur_adx
        self.state.volatility_ratio = cur_vol_ratio
        self.state.ema_slope_norm = cur_slope
        self.state.hurst_value = cur_hurst

        # Score each regime
        scores = self._compute_regime_scores(
            cur_adx, cur_di_plus, cur_di_minus,
            cur_vol_ratio, cur_slope, cur_hurst
        )

        # Select regime with highest score
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # Normalize confidence to 0-1
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.3

        # Apply persistence: blend with previous regime confidence
        if best_regime == self.state.current_regime:
            # Same regime - boost confidence slightly
            confidence = min(1.0, confidence * 1.05)
            self.state.regime_duration += 1
        else:
            # Regime change - apply hysteresis (need higher confidence to switch)
            switch_threshold = 0.35
            if confidence < switch_threshold:
                # Not confident enough to switch - keep current
                best_regime = self.state.current_regime
                confidence = max(0.3, confidence * 0.8)
            else:
                # Regime transition
                self._bar_count += 1
                self.history.add_transition(best_regime, confidence, self._bar_count)
                self.state.regime_duration = 0

        # Exponential smoothing of confidence
        alpha = cfg.regime_persistence_alpha
        confidence = alpha * confidence + (1.0 - alpha) * self.state.confidence

        self.state.current_regime = best_regime
        self.state.confidence = confidence

        return best_regime, confidence

    def _compute_regime_scores(self, adx: float, di_plus: float, di_minus: float,
                               vol_ratio: float, slope: float,
                               hurst: float) -> dict:
        """
        Compute score for each regime based on indicator values.
        Higher score means more likely to be in that regime.
        """
        cfg = self.config
        scores = {regime: 0.0 for regime in MarketRegime}

        # --- TRENDING_UP ---
        score = 0.0
        if adx > cfg.adx_trend_threshold:
            score += 0.3 * min(adx / cfg.adx_strong_trend, 1.0)
        if di_plus > di_minus:
            score += 0.2 * min((di_plus - di_minus) / 20.0, 1.0)
        if slope > cfg.slope_strong:
            score += 0.3 * min(slope / (cfg.slope_strong * 2), 1.0)
        if hurst > cfg.hurst_trending:
            score += 0.2 * min((hurst - 0.5) * 4, 1.0)
        scores[MarketRegime.TRENDING_UP] = max(0.0, score)

        # --- TRENDING_DOWN ---
        score = 0.0
        if adx > cfg.adx_trend_threshold:
            score += 0.3 * min(adx / cfg.adx_strong_trend, 1.0)
        if di_minus > di_plus:
            score += 0.2 * min((di_minus - di_plus) / 20.0, 1.0)
        if slope < -cfg.slope_strong:
            score += 0.3 * min(abs(slope) / (cfg.slope_strong * 2), 1.0)
        if hurst > cfg.hurst_trending:
            score += 0.2 * min((hurst - 0.5) * 4, 1.0)
        scores[MarketRegime.TRENDING_DOWN] = max(0.0, score)

        # --- RANGING_NARROW ---
        score = 0.0
        if adx < cfg.adx_weak_threshold:
            score += 0.3 * (1.0 - adx / cfg.adx_weak_threshold)
        if vol_ratio < cfg.vol_ratio_low:
            score += 0.35 * (1.0 - vol_ratio / cfg.vol_ratio_low)
        if abs(slope) < cfg.slope_weak:
            score += 0.2
        if 0.4 <= hurst <= 0.6:
            score += 0.15
        scores[MarketRegime.RANGING_NARROW] = max(0.0, score)

        # --- RANGING_WIDE ---
        score = 0.0
        if adx < cfg.adx_trend_threshold:
            score += 0.25 * (1.0 - adx / cfg.adx_trend_threshold)
        if cfg.vol_ratio_low <= vol_ratio <= cfg.vol_ratio_high:
            score += 0.3
        if abs(slope) < cfg.slope_strong:
            score += 0.2
        if 0.4 <= hurst <= 0.6:
            score += 0.15
        # Distinguish from narrow: needs moderate volatility
        if vol_ratio > cfg.vol_ratio_low:
            score += 0.1
        scores[MarketRegime.RANGING_WIDE] = max(0.0, score)

        # --- VOLATILE_BREAKOUT ---
        score = 0.0
        if vol_ratio > cfg.vol_ratio_high:
            score += 0.4 * min(vol_ratio / cfg.vol_ratio_very_high, 1.0)
        if adx > cfg.adx_trend_threshold:
            score += 0.2
        # Recent volatility expansion (ratio increasing)
        if vol_ratio > 1.3:
            score += 0.2
        if hurst > 0.55:
            score += 0.1
        # Extreme moves signal breakout
        if abs(slope) > cfg.slope_strong * 1.5:
            score += 0.1
        scores[MarketRegime.VOLATILE_BREAKOUT] = max(0.0, score)

        # --- MEAN_REVERTING ---
        score = 0.0
        if hurst < cfg.hurst_mean_revert:
            score += 0.4 * (1.0 - hurst / cfg.hurst_mean_revert)
        if adx < cfg.adx_weak_threshold:
            score += 0.25
        if vol_ratio < cfg.vol_ratio_high:
            score += 0.15
        if abs(slope) < cfg.slope_strong:
            score += 0.2
        scores[MarketRegime.MEAN_REVERTING] = max(0.0, score)

        return scores

    def get_state(self) -> RegimeState:
        """Get current internal state for debugging/logging."""
        return self.state

    def get_history(self) -> RegimeHistory:
        """Get regime transition history."""
        return self.history

    def reset(self):
        """Reset detector state (for new trading session or symbol switch)."""
        self.state = RegimeState()
        self.history = RegimeHistory()
        self._bar_count = 0
