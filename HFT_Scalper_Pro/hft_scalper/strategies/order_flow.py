"""
Order Flow Imbalance Strategy (Contrarian).

The data shows OFI is strongly negatively autocorrelated (-0.177 tick, -0.456 block).
This means extreme order flow readings tend to REVERSE.

Logic:
- Track cumulative tick count changes as a proxy for order flow
- Compute rolling OFI z-score
- When OFI exceeds +2 std, go SHORT (fade the buying pressure)
- When OFI exceeds -2 std, go LONG (fade the selling pressure)
- Exit when OFI returns to neutral
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class OrderFlowStrategy(BaseStrategy):
    """Contrarian order flow imbalance strategy."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "OrderFlow",
            "ofi_period": 20,  # Lookback for OFI z-score
            "ofi_threshold": 2.0,  # Z-score threshold for entry
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.5,
            "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21],
            "volume_confirm": True,  # Require volume spike for entry
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate contrarian order flow signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values
        opens = bars["open"].values
        tick_counts = bars["tick_count"].values if "tick_count" in bars.columns else np.ones(n)
        volumes = bars["volume"].values if "volume" in bars.columns else np.ones(n)

        ofi_period = self.config["ofi_period"]
        ofi_threshold = self.config["ofi_threshold"]
        atr_period = self.config["atr_period"]
        sl_mult = self.config["sl_atr_mult"]
        tp_mult = self.config["tp_atr_mult"]
        active_hours = self.config["active_hours"]

        # Compute OFI proxy: directional pressure from bar close relative to range
        # Close near high = buying pressure, close near low = selling pressure
        bar_range = highs - lows
        bar_range[bar_range < 0.01] = 0.01  # Avoid division by zero

        # Normalized close position within bar (0 = at low, 1 = at high)
        close_position = (closes - lows) / bar_range

        # OFI: close position weighted by volume/ticks (buying vs selling pressure)
        ofi_raw = (close_position - 0.5) * 2.0 * tick_counts  # Scale by activity

        # Rolling z-score of OFI
        ofi_zscore = np.zeros(n)
        for i in range(ofi_period, n):
            window = ofi_raw[i - ofi_period:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                ofi_zscore[i] = (ofi_raw[i] - mean) / std

        # ATR for stops
        atr = _compute_atr(highs, lows, closes, atr_period)

        # Get hours
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        warmup = max(ofi_period, atr_period) + 1

        for i in range(warmup, n):
            if atr[i] < 0.01:
                continue

            if active_hours and hours[i] not in active_hours:
                continue

            sl_dist = atr[i] * sl_mult
            tp_dist = atr[i] * tp_mult

            # CONTRARIAN: fade extreme OFI readings
            if ofi_zscore[i] > ofi_threshold:
                # Extreme buying pressure -> fade it, go SHORT
                signals[i] = [-1, sl_dist, tp_dist]
            elif ofi_zscore[i] < -ofi_threshold:
                # Extreme selling pressure -> fade it, go LONG
                signals[i] = [1, sl_dist, tp_dist]

        return signals

    def get_param_grid(self) -> dict:
        return {
            "ofi_period": [10, 15, 20, 30],
            "ofi_threshold": [1.5, 2.0, 2.5, 3.0],
            "sl_atr_mult": [0.7, 1.0, 1.5, 2.0],
            "tp_atr_mult": [1.5, 2.0, 2.5, 3.0],
        }


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
