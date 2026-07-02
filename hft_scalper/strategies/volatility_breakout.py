"""
Volatility Breakout Strategy.

Detects ATR compression (low volatility consolidation) and trades the breakout.
When current ATR drops below 0.5x of the 20-period average ATR, the market is
consolidating. Enter on breakout with volume/tick confirmation.

Logic:
- Compute ATR and its 20-period average
- Identify compression: current ATR < compression_ratio * avg ATR
- Set breakout levels at recent high/low of consolidation period
- Enter LONG on breakout above consolidation high
- Enter SHORT on breakout below consolidation low
- Use consolidation range for stop placement
- Target minimum 2:1 reward/risk
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility compression breakout strategy."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "VolBreakout",
            "atr_period": 14,
            "atr_avg_period": 20,
            "compression_ratio": 0.6,  # ATR must be below this * avg ATR
            "consolidation_lookback": 10,  # Bars to define consolidation range
            "breakout_buffer": 0.1,  # Points above/below consolidation for breakout
            "sl_mult": 1.0,  # SL at consolidation range * this mult
            "rr_ratio": 2.0,  # Reward/risk minimum
            "tick_confirm": False,  # Disable tick confirmation to get more signals
            "tick_spike_mult": 1.3,  # Tick count must be > this * avg for breakout bar
            "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21],
            "min_compression_bars": 2,  # Minimum bars in compression
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate volatility breakout signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values
        tick_counts = bars["tick_count"].values if "tick_count" in bars.columns else np.ones(n)

        atr_period = self.config["atr_period"]
        atr_avg_period = self.config["atr_avg_period"]
        compression_ratio = self.config["compression_ratio"]
        consol_lookback = self.config["consolidation_lookback"]
        breakout_buffer = self.config["breakout_buffer"]
        sl_mult = self.config["sl_mult"]
        rr_ratio = self.config["rr_ratio"]
        tick_confirm = self.config["tick_confirm"]
        tick_spike_mult = self.config["tick_spike_mult"]
        active_hours = self.config["active_hours"]

        # Compute ATR
        atr = _compute_atr(highs, lows, closes, atr_period)

        # Compute rolling average of ATR
        avg_atr = np.zeros(n)
        for i in range(atr_avg_period - 1, n):
            avg_atr[i] = np.mean(atr[i - atr_avg_period + 1:i + 1])

        # Compute rolling average tick count
        avg_ticks = np.zeros(n)
        for i in range(20, n):
            avg_ticks[i] = np.mean(tick_counts[i - 20:i])

        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        warmup = max(atr_period, atr_avg_period, consol_lookback) + 5
        in_compression = False
        compression_start = 0
        compression_bars = 0

        for i in range(warmup, n):
            if avg_atr[i] < 0.01:
                continue

            if active_hours and hours[i] not in active_hours:
                continue

            atr_ratio = atr[i] / avg_atr[i] if avg_atr[i] > 0 else 1.0

            # Detect compression
            if atr_ratio < compression_ratio:
                if not in_compression:
                    in_compression = True
                    compression_start = i
                compression_bars += 1
            elif in_compression and compression_bars >= self.config.get("min_compression_bars", 2):
                # Breakout from compression!
                # Get consolidation range
                start_idx = max(compression_start, i - consol_lookback)
                consol_high = np.max(highs[start_idx:i])
                consol_low = np.min(lows[start_idx:i])
                consol_range = consol_high - consol_low

                if consol_range < 0.5:
                    in_compression = False
                    compression_bars = 0
                    continue

                # Check for breakout direction
                sl_dist = consol_range * sl_mult
                tp_dist = sl_dist * rr_ratio

                # Volume/tick confirmation
                tick_confirmed = True
                if tick_confirm and avg_ticks[i] > 0:
                    tick_confirmed = tick_counts[i] > avg_ticks[i] * tick_spike_mult

                if closes[i] > consol_high + breakout_buffer and tick_confirmed:
                    # Upside breakout
                    signals[i] = [1, sl_dist, tp_dist]
                elif closes[i] < consol_low - breakout_buffer and tick_confirmed:
                    # Downside breakout
                    signals[i] = [-1, sl_dist, tp_dist]

                in_compression = False
                compression_bars = 0
            elif in_compression and atr_ratio >= compression_ratio:
                # ATR expanded but no clear breakout yet
                # Check if price broke consolidation range
                start_idx = max(compression_start, i - consol_lookback)
                consol_high = np.max(highs[start_idx:i])
                consol_low = np.min(lows[start_idx:i])
                consol_range = consol_high - consol_low

                if consol_range >= 0.5:
                    sl_dist = consol_range * sl_mult
                    tp_dist = sl_dist * rr_ratio

                    tick_confirmed = True
                    if tick_confirm and avg_ticks[i] > 0:
                        tick_confirmed = tick_counts[i] > avg_ticks[i] * tick_spike_mult

                    if closes[i] > consol_high + breakout_buffer and tick_confirmed:
                        signals[i] = [1, sl_dist, tp_dist]
                    elif closes[i] < consol_low - breakout_buffer and tick_confirmed:
                        signals[i] = [-1, sl_dist, tp_dist]

                in_compression = False
                compression_bars = 0

        return signals

    def get_param_grid(self) -> dict:
        return {
            "atr_period": [10, 14, 20],
            "compression_ratio": [0.5, 0.6, 0.7],
            "consolidation_lookback": [5, 8, 10, 15],
            "sl_mult": [0.8, 1.0, 1.5],
            "rr_ratio": [2.0, 2.5, 3.0],
            "breakout_buffer": [0.05, 0.1, 0.2],
            "min_compression_bars": [2, 3, 4],
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
