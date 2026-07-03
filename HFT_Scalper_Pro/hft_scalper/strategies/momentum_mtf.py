"""
Multi-Timeframe Momentum Strategy.

Uses the 60-bar momentum signal (+0.103 correlation) for direction,
but enters on 5-bar mean-reversion pullbacks (negative correlation at 5 bars).

Logic:
- Compute 60-bar EMA trend direction (slow timeframe)
- Compute 5-bar RSI for pullback detection (fast timeframe)
- Enter LONG when 60-bar trend is UP and 5-bar RSI is oversold (pullback in uptrend)
- Enter SHORT when 60-bar trend is DOWN and 5-bar RSI is overbought (pullback in downtrend)
- This combines trend following (60-bar) with mean-reversion entry (5-bar)
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class MomentumMTFStrategy(BaseStrategy):
    """Multi-timeframe momentum + pullback strategy."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "MomentumMTF",
            "slow_period": 60,  # Trend direction timeframe
            "fast_period": 5,   # Pullback detection timeframe
            "fast_rsi_period": 5,
            "fast_rsi_ob": 75,
            "fast_rsi_os": 25,
            "trend_strength_threshold": 0.2,  # Minimum trend slope
            "atr_period": 14,
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.0,
            "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate multi-timeframe momentum signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values

        slow_period = self.config["slow_period"]
        fast_rsi_period = self.config["fast_rsi_period"]
        fast_rsi_ob = self.config["fast_rsi_ob"]
        fast_rsi_os = self.config["fast_rsi_os"]
        trend_thresh = self.config["trend_strength_threshold"]
        atr_period = self.config["atr_period"]
        sl_mult = self.config["sl_atr_mult"]
        tp_mult = self.config["tp_atr_mult"]
        active_hours = self.config["active_hours"]

        # Compute slow EMA for trend direction
        slow_ema = _compute_ema(closes, slow_period)

        # Compute trend slope (normalized by ATR)
        atr = _compute_atr(highs, lows, closes, atr_period)
        trend_slope = np.zeros(n)
        for i in range(slow_period, n):
            if atr[i] > 0:
                trend_slope[i] = (slow_ema[i] - slow_ema[i - slow_period // 4]) / atr[i]

        # Compute fast RSI for pullback detection
        fast_rsi = _compute_rsi(closes, fast_rsi_period)

        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        warmup = max(slow_period, atr_period, fast_rsi_period) + 5

        for i in range(warmup, n):
            if atr[i] < 0.01:
                continue

            if active_hours and hours[i] not in active_hours:
                continue

            sl_dist = atr[i] * sl_mult
            tp_dist = atr[i] * tp_mult

            # Trend is UP and fast RSI shows pullback (oversold)
            if trend_slope[i] > trend_thresh and fast_rsi[i] < fast_rsi_os:
                signals[i] = [1, sl_dist, tp_dist]

            # Trend is DOWN and fast RSI shows pullback (overbought)
            elif trend_slope[i] < -trend_thresh and fast_rsi[i] > fast_rsi_ob:
                signals[i] = [-1, sl_dist, tp_dist]

        return signals

    def get_param_grid(self) -> dict:
        return {
            "slow_period": [40, 60, 80],
            "fast_rsi_period": [3, 5, 7],
            "fast_rsi_ob": [70, 75, 80],
            "fast_rsi_os": [20, 25, 30],
            "trend_strength_threshold": [0.1, 0.2, 0.3],
            "sl_atr_mult": [0.8, 1.2, 1.8],
            "tp_atr_mult": [1.5, 2.0, 3.0],
        }


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    result = np.zeros_like(arr)
    multiplier = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


def _compute_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI indicator."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    deltas = np.diff(closes)

    if len(deltas) < period:
        return rsi

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - 100 / (1 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - 100 / (1 + rs)

    return rsi


def _compute_atr(highs, lows, closes, period):
    """Compute ATR."""
    n = len(highs)
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr
