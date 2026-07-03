"""
Momentum burst strategy for capturing fast directional moves.

Detects sudden momentum bursts (rate of change > 2 std deviations)
and enters in that direction with time-based exits.
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class MomentumBurstStrategy(BaseStrategy):
    """Short-term momentum burst strategy."""

    def __init__(self, config=None):
        default_config = {
            "name": "MomentumBurst",
            "roc_period": 7,  # Rate of change period
            "roc_std_period": 50,  # Period for calculating std of ROC
            "roc_threshold": 1.8,  # Std devs for burst detection
            "atr_period": 14,
            "trend_ema": 60,  # Only trade bursts aligned with trend
            "sl_mult": 1.2,  # SL = 1.2x ATR
            "tp_mult": 3.0,  # TP = 3x ATR
            "active_hours": list(range(7, 21)),
            "warmup": 80,
            "min_burst_size": 1.0,  # Minimum burst size in points
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate momentum burst signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values

        # Rate of change
        roc_period = self.config["roc_period"]
        roc = np.zeros(n)
        for i in range(roc_period, n):
            roc[i] = close[i] - close[i - roc_period]

        # Rolling std of ROC
        std_period = self.config["roc_std_period"]
        roc_std = np.zeros(n)
        roc_mean = np.zeros(n)
        for i in range(std_period, n):
            window = roc[i - std_period:i]
            roc_std[i] = np.std(window)
            roc_mean[i] = np.mean(window)

        # ATR
        atr = self._atr(high, low, close, self.config["atr_period"])

        # Trend EMA
        trend_ema = self._ema(close, self.config["trend_ema"])

        # Session filter
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.full(n, 12)

        warmup = self.config["warmup"]
        threshold = self.config["roc_threshold"]

        for i in range(warmup, n):
            if hours[i] not in self.config["active_hours"]:
                continue

            if atr[i] < 0.3 or roc_std[i] < 0.01:
                continue

            # Detect burst: ROC exceeds threshold std deviations
            z_score = (roc[i] - roc_mean[i]) / roc_std[i] if roc_std[i] > 0 else 0

            # Must exceed minimum size in absolute points
            if abs(roc[i]) < self.config["min_burst_size"]:
                continue

            if z_score > threshold:
                # Bullish burst - only if aligned with trend
                if close[i] > trend_ema[i]:
                    sl_dist = atr[i] * self.config["sl_mult"]
                    tp_dist = atr[i] * self.config["tp_mult"]
                    signals[i] = [1, sl_dist, tp_dist]

            elif z_score < -threshold:
                # Bearish burst - only if aligned with trend
                if close[i] < trend_ema[i]:
                    sl_dist = atr[i] * self.config["sl_mult"]
                    tp_dist = atr[i] * self.config["tp_mult"]
                    signals[i] = [-1, sl_dist, tp_dist]

        return signals

    @staticmethod
    def _atr(high, low, close, period):
        """Average True Range."""
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr_arr = np.zeros(n)
        if n >= period:
            atr_arr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
            atr_arr[:period - 1] = atr_arr[period - 1]
        return atr_arr

    @staticmethod
    def _ema(data, period):
        """Exponential moving average."""
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        result[:period - 1] = result[period - 1]
        return result
