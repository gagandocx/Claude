"""
Ensemble Strategy: Combining OrderFlow + MomentumMTF + SpreadFade.

Combines the three best strategies into a single consensus-driven approach:
1. OrderFlow (contrarian) - primary entry signal when OFI z-score exceeds threshold
2. MomentumMTF - trend direction filter (40-bar momentum confirms direction)
3. SpreadFade - additional confidence boost (spread dynamics confirm entry)

Signal scoring:
- Score 1-3 based on how many sub-strategies agree on direction
- Only trade when score >= 2 (consensus required)
- Tighter stops (1.5x ATR) when score=3 (high confidence)
- Wider stops (2.5x ATR) when score=2 (moderate confidence)

Session filter: only trade hours 4, 8-21 UTC (avoid quiet Asian session hours 0-3, 5-7)
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class EnsembleStrategy(BaseStrategy):
    """Ensemble strategy combining OrderFlow, MomentumMTF, and SpreadFade."""

    def __init__(self, config: dict = None):
        default_config = {
            "name": "Ensemble",
            # OrderFlow parameters (best from optimization)
            "ofi_period": 30,
            "ofi_threshold": 2.0,
            # MomentumMTF parameters (best from optimization)
            "slow_period": 40,
            "fast_rsi_period": 7,
            "fast_rsi_ob": 75,
            "fast_rsi_os": 20,
            "trend_strength_threshold": 0.1,
            # SpreadFade parameters (best from optimization)
            "spread_lookback": 30,
            "wide_threshold": 2.5,
            "contract_threshold": 1.5,
            # Ensemble parameters
            "min_score": 2,  # Minimum agreement score to trade
            "atr_period": 14,
            "sl_atr_mult_high": 1.5,  # SL when score=3 (high confidence)
            "sl_atr_mult_low": 2.5,   # SL when score=2 (moderate confidence)
            "tp_atr_mult_high": 3.0,  # TP when score=3
            "tp_atr_mult_low": 2.0,   # TP when score=2
            # Session filter: trade only hours 4, 8-21 UTC
            "active_hours": [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "cooldown_bars": 3,  # Minimum bars between signals
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate ensemble signals from combined sub-strategy logic."""
        n = len(bars)
        signals = np.zeros((n, 3))

        closes = bars["close"].values
        highs = bars["high"].values
        lows = bars["low"].values
        opens = bars["open"].values
        tick_counts = bars["tick_count"].values if "tick_count" in bars.columns else np.ones(n)
        spreads = bars["avg_spread"].values if "avg_spread" in bars.columns else np.full(n, 0.1)

        # Extract config
        ofi_period = self.config["ofi_period"]
        ofi_threshold = self.config["ofi_threshold"]
        slow_period = self.config["slow_period"]
        fast_rsi_period = self.config["fast_rsi_period"]
        fast_rsi_ob = self.config["fast_rsi_ob"]
        fast_rsi_os = self.config["fast_rsi_os"]
        trend_thresh = self.config["trend_strength_threshold"]
        spread_lookback = self.config["spread_lookback"]
        wide_thresh = self.config["wide_threshold"]
        contract_thresh = self.config["contract_threshold"]
        min_score = self.config["min_score"]
        atr_period = self.config["atr_period"]
        sl_mult_high = self.config["sl_atr_mult_high"]
        sl_mult_low = self.config["sl_atr_mult_low"]
        tp_mult_high = self.config["tp_atr_mult_high"]
        tp_mult_low = self.config["tp_atr_mult_low"]
        active_hours = self.config["active_hours"]
        cooldown = self.config["cooldown_bars"]

        # ===== Compute OrderFlow component =====
        bar_range = highs - lows
        bar_range[bar_range < 0.01] = 0.01
        close_position = (closes - lows) / bar_range
        ofi_raw = (close_position - 0.5) * 2.0 * tick_counts

        ofi_zscore = np.zeros(n)
        for i in range(ofi_period, n):
            window = ofi_raw[i - ofi_period:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                ofi_zscore[i] = (ofi_raw[i] - mean) / std

        # ===== Compute MomentumMTF component =====
        slow_ema = _compute_ema(closes, slow_period)
        atr = _compute_atr(highs, lows, closes, atr_period)

        trend_slope = np.zeros(n)
        for i in range(slow_period, n):
            if atr[i] > 0:
                trend_slope[i] = (slow_ema[i] - slow_ema[i - slow_period // 4]) / atr[i]

        fast_rsi = _compute_rsi(closes, fast_rsi_period)

        # ===== Compute SpreadFade component =====
        spread_signal = np.zeros(n)  # 1=buy, -1=sell, 0=neutral
        was_wide = False
        wide_start_price = 0.0

        for i in range(spread_lookback, n):
            window_spreads = spreads[i - spread_lookback:i]
            median_spread = np.median(window_spreads)
            if median_spread < 0.01:
                continue

            spread_ratio = spreads[i] / median_spread

            if spread_ratio >= wide_thresh and not was_wide:
                was_wide = True
                wide_start_price = closes[i]

            if was_wide and spread_ratio <= contract_thresh:
                was_wide = False
                price_change = closes[i] - wide_start_price
                if price_change > 0:
                    spread_signal[i] = 1
                elif price_change < 0:
                    spread_signal[i] = -1

            if was_wide and spread_ratio < wide_thresh * 0.7:
                was_wide = False

        # ===== Get hours =====
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.zeros(n, dtype=int)

        # ===== Combine signals =====
        warmup = max(slow_period, ofi_period, atr_period, spread_lookback, fast_rsi_period) + 5
        last_signal_bar = -cooldown - 1

        for i in range(warmup, n):
            if atr[i] < 0.01:
                continue

            # Session filter
            if active_hours and hours[i] not in active_hours:
                continue

            # Cooldown check
            if (i - last_signal_bar) < cooldown:
                continue

            # Compute direction votes from each sub-strategy
            buy_score = 0
            sell_score = 0

            # Sub-strategy 1: OrderFlow (contrarian)
            if ofi_zscore[i] > ofi_threshold:
                sell_score += 1  # Extreme buying -> fade it -> sell
            elif ofi_zscore[i] < -ofi_threshold:
                buy_score += 1  # Extreme selling -> fade it -> buy

            # Sub-strategy 2: MomentumMTF (trend + pullback)
            if trend_slope[i] > trend_thresh and fast_rsi[i] < fast_rsi_os:
                buy_score += 1  # Uptrend with pullback -> buy
            elif trend_slope[i] < -trend_thresh and fast_rsi[i] > fast_rsi_ob:
                sell_score += 1  # Downtrend with pullback -> sell

            # Sub-strategy 3: SpreadFade
            if spread_signal[i] == 1:
                buy_score += 1
            elif spread_signal[i] == -1:
                sell_score += 1

            # Determine final signal based on consensus
            total_score = max(buy_score, sell_score)
            if total_score < min_score:
                continue

            # Determine direction
            if buy_score >= min_score:
                direction = 1
                score = buy_score
            elif sell_score >= min_score:
                direction = -1
                score = sell_score
            else:
                continue

            # Set stops based on confidence (score)
            if score >= 3:
                sl_dist = atr[i] * sl_mult_high
                tp_dist = atr[i] * tp_mult_high
            else:
                sl_dist = atr[i] * sl_mult_low
                tp_dist = atr[i] * tp_mult_low

            signals[i] = [direction, sl_dist, tp_dist]
            last_signal_bar = i

        return signals

    def get_param_grid(self) -> dict:
        """Return parameter grid for ensemble optimization."""
        return {
            "ofi_period": [20, 30, 40],
            "ofi_threshold": [1.5, 2.0, 2.5],
            "slow_period": [30, 40, 60],
            "trend_strength_threshold": [0.1, 0.15, 0.2],
            "min_score": [2],
            "sl_atr_mult_high": [1.2, 1.5, 2.0],
            "tp_atr_mult_high": [2.5, 3.0, 3.5],
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
