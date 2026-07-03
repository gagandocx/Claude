"""
Spread Contraction Fade Strategy.

When spread widens significantly (>2x median), it indicates uncertainty/volatility.
When spread contracts back to normal, it signals resolution.
Trade in the direction of the price move during contraction.

Logic:
- Track rolling spread statistics
- Detect when spread exceeds 2x median (wide spread event)
- Wait for spread to contract back below 1.2x median
- Enter in direction of price movement during the contraction phase
- Use ATR-based stops
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class SpreadFadeStrategy(BaseStrategy):
    """Spread contraction fade strategy."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "SpreadFade",
            "spread_lookback": 50,  # Bars to compute median spread
            "wide_threshold": 2.0,  # Multiple of median for "wide" spread
            "contract_threshold": 1.2,  # Spread must contract below this * median
            "price_lookback": 5,  # Bars to measure price direction during contraction
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.5,
            "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "cooldown_bars": 5,  # Minimum bars between trades
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate spread fade signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values
        spreads = bars["avg_spread"].values if "avg_spread" in bars.columns else np.full(n, 0.1)

        spread_lookback = self.config["spread_lookback"]
        wide_thresh = self.config["wide_threshold"]
        contract_thresh = self.config["contract_threshold"]
        price_lookback = self.config["price_lookback"]
        atr_period = self.config["atr_period"]
        sl_mult = self.config["sl_atr_mult"]
        tp_mult = self.config["tp_atr_mult"]
        active_hours = self.config["active_hours"]
        cooldown = self.config["cooldown_bars"]

        atr = _compute_atr(highs, lows, closes, atr_period)

        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        warmup = max(spread_lookback, atr_period, price_lookback) + 1
        was_wide = False
        wide_start_price = 0.0
        last_signal_bar = -cooldown - 1

        for i in range(warmup, n):
            if atr[i] < 0.01:
                continue

            if active_hours and hours[i] not in active_hours:
                continue

            # Compute rolling median spread
            window_spreads = spreads[i - spread_lookback:i]
            median_spread = np.median(window_spreads)

            if median_spread < 0.01:
                continue

            current_spread = spreads[i]
            spread_ratio = current_spread / median_spread

            # Detect spread widening
            if spread_ratio >= wide_thresh and not was_wide:
                was_wide = True
                wide_start_price = closes[i]

            # Detect contraction after widening
            if was_wide and spread_ratio <= contract_thresh:
                was_wide = False

                if (i - last_signal_bar) < cooldown:
                    continue

                # Determine price direction during the contraction
                price_change = closes[i] - wide_start_price

                sl_dist = atr[i] * sl_mult
                tp_dist = atr[i] * tp_mult

                if price_change > 0:
                    # Price moved up during contraction - go LONG
                    signals[i] = [1, sl_dist, tp_dist]
                    last_signal_bar = i
                elif price_change < 0:
                    # Price moved down during contraction - go SHORT
                    signals[i] = [-1, sl_dist, tp_dist]
                    last_signal_bar = i

            # Reset wide state if spread normalizes without proper contraction
            if was_wide and spread_ratio < wide_thresh * 0.7:
                was_wide = False

        return signals

    def get_param_grid(self) -> dict:
        return {
            "spread_lookback": [30, 50, 80],
            "wide_threshold": [1.5, 2.0, 2.5],
            "contract_threshold": [1.0, 1.2, 1.5],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "tp_atr_mult": [2.0, 2.5, 3.0],
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
