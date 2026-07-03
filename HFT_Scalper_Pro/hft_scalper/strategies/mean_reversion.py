"""
Adaptive Mean Reversion Strategy.

Uses dynamic Bollinger Bands with RSI confirmation.
The data shows strong negative bar autocorrelation - this is the primary strategy.

Logic:
- Compute Bollinger Bands (adaptive period based on volatility)
- Enter SHORT when price touches upper band AND RSI > overbought threshold
- Enter LONG when price touches lower band AND RSI < oversold threshold
- Exit at middle band (mean) or take profit
- Use ATR-based stops
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands + RSI."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "MeanReversion",
            "bb_period": 14,
            "bb_std": 1.5,
            "rsi_period": 7,
            "rsi_overbought": 60,
            "rsi_oversold": 40,
            "atr_period": 14,
            "sl_atr_mult": 1.0,
            "tp_atr_mult": 1.0,  # Tighter TP for mean reversion (revert to mean)
            "active_hours": [1, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "use_keltner_squeeze": False,  # Disable for more signals
            "keltner_mult": 1.5,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate mean reversion signals."""
        n = len(bars)
        signals = np.zeros((n, 3))  # direction, sl_dist, tp_dist

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values

        bb_period = self.config["bb_period"]
        bb_std = self.config["bb_std"]
        rsi_period = self.config["rsi_period"]
        rsi_ob = self.config["rsi_overbought"]
        rsi_os = self.config["rsi_oversold"]
        atr_period = self.config["atr_period"]
        sl_mult = self.config["sl_atr_mult"]
        tp_mult = self.config["tp_atr_mult"]
        active_hours = self.config["active_hours"]
        use_squeeze = self.config.get("use_keltner_squeeze", True)
        keltner_mult = self.config.get("keltner_mult", 1.5)

        # Compute indicators
        sma = _rolling_mean(closes, bb_period)
        std = _rolling_std(closes, bb_period)
        upper_band = sma + bb_std * std
        lower_band = sma - bb_std * std

        rsi = _compute_rsi(closes, rsi_period)
        atr = _compute_atr(highs, lows, closes, atr_period)

        # Keltner Channel for squeeze detection
        keltner_upper = sma + keltner_mult * atr
        keltner_lower = sma - keltner_mult * atr

        # Get hour of each bar for time filtering
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        warmup = max(bb_period, rsi_period, atr_period) + 1

        for i in range(warmup, n):
            if atr[i] < 0.01:  # Skip if ATR is too small
                continue

            # Time filter
            if active_hours and hours[i] not in active_hours:
                continue

            # Squeeze detection: BB inside Keltner = squeeze (low vol, skip)
            if use_squeeze:
                in_squeeze = (upper_band[i] < keltner_upper[i]) and (lower_band[i] > keltner_lower[i])
                if in_squeeze:
                    continue

            sl_dist = atr[i] * sl_mult
            tp_dist = atr[i] * tp_mult

            # LONG signal: price at/below lower band, RSI oversold
            if closes[i] <= lower_band[i] and rsi[i] < rsi_os:
                signals[i] = [1, sl_dist, tp_dist]

            # SHORT signal: price at/above upper band, RSI overbought
            elif closes[i] >= upper_band[i] and rsi[i] > rsi_ob:
                signals[i] = [-1, sl_dist, tp_dist]

        return signals

    def get_param_grid(self) -> dict:
        """Return parameter grid for optimization."""
        return {
            "bb_period": [10, 14, 20, 30],
            "bb_std": [1.2, 1.5, 2.0, 2.5],
            "rsi_period": [5, 7, 10, 14],
            "rsi_overbought": [55, 60, 65, 70],
            "rsi_oversold": [30, 35, 40, 45],
            "sl_atr_mult": [0.7, 1.0, 1.5],
            "tp_atr_mult": [0.8, 1.0, 1.5, 2.0],
        }


def _rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute rolling mean."""
    result = np.full_like(arr, np.nan)
    cumsum = np.cumsum(arr)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return result


def _rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute rolling standard deviation."""
    result = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.std(arr[i - period + 1:i + 1], ddof=1)
    return result


def _compute_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI indicator."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    deltas = np.diff(closes)

    if len(deltas) < period:
        return rsi

    # Initial average gain/loss
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - 100 / (1 + rs)

    # Exponential smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - 100 / (1 + rs)

    return rsi


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Compute Average True Range."""
    n = len(highs)
    atr = np.zeros(n)

    # True range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # EMA of true range
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr
