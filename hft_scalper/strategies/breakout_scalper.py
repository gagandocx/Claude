"""
Volatility breakout strategy optimized for XAUUSD's large intraday moves.

Identifies consolidation periods (low ATR) then enters on breakouts
with aggressive trailing and pyramiding support.
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class BreakoutScalperStrategy(BaseStrategy):
    """Volatility breakout strategy."""

    def __init__(self, config=None):
        default_config = {
            "name": "BreakoutScalper",
            "lookback": 15,  # Bars to look back for range
            "atr_period": 14,
            "atr_squeeze_ratio": 0.7,  # Current ATR < 70% of avg ATR = squeeze
            "atr_avg_period": 50,
            "breakout_buffer": 0.3,  # Points above/below range for breakout
            "sl_mult": 1.0,  # SL = 1x ATR
            "tp_mult": 4.0,  # TP = 4x ATR (high R:R)
            "min_range_bars": 8,  # Min bars of consolidation
            "volume_confirm": False,  # Require volume spike on breakout
            "active_hours": list(range(7, 21)),
            "warmup": 60,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate breakout signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        open_ = bars["open"].values

        # ATR
        atr = self._atr(high, low, close, self.config["atr_period"])

        # Average ATR for squeeze detection
        atr_avg = np.zeros(n)
        avg_period = self.config["atr_avg_period"]
        for i in range(avg_period, n):
            atr_avg[i] = np.mean(atr[i - avg_period:i])
        atr_avg[:avg_period] = atr_avg[avg_period] if avg_period < n else 1.0

        # Session hours
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.full(n, 12)

        warmup = self.config["warmup"]
        lookback = self.config["lookback"]

        for i in range(warmup, n):
            if hours[i] not in self.config["active_hours"]:
                continue

            if atr[i] < 0.3:
                continue

            # Check for consolidation (squeeze)
            is_squeeze = atr[i] < atr_avg[i] * self.config["atr_squeeze_ratio"]

            if not is_squeeze:
                # Also check if we just broke out of a squeeze
                # Look back a few bars to see if squeeze was recent
                recent_squeeze = False
                for j in range(max(0, i - 5), i):
                    if atr[j] < atr_avg[j] * self.config["atr_squeeze_ratio"]:
                        recent_squeeze = True
                        break
                if not recent_squeeze:
                    continue

            # Calculate consolidation range
            range_high = np.max(high[i - lookback:i])
            range_low = np.min(low[i - lookback:i])
            range_size = range_high - range_low

            # Skip if range is too large (not really consolidation)
            if range_size > atr_avg[i] * 3:
                continue

            # Breakout detection
            buffer = self.config["breakout_buffer"]

            if close[i] > range_high + buffer and close[i] > open_[i]:
                # Bullish breakout
                sl_dist = atr[i] * self.config["sl_mult"]
                tp_dist = atr[i] * self.config["tp_mult"]
                signals[i] = [1, sl_dist, tp_dist]

            elif close[i] < range_low - buffer and close[i] < open_[i]:
                # Bearish breakout
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
        atr = np.zeros(n)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            atr[:period - 1] = atr[period - 1]
        return atr
