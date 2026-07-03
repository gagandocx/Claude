"""
Adaptive Strategy Implementations
===================================
Five distinct strategies, each specialized for different market regimes.
All strategies output signal arrays of shape (n, 3): [direction, sl_dist, tp_dist].

Strategies:
    1. TrendFollower   - EMA crossover + ADX filter, enters on pullback
    2. MeanReversion   - Bollinger Band + RSI + Keltner squeeze filter
    3. BreakoutTrader  - ATR compression detection + breakout confirmation
    4. ScalpMomentum   - Fast RSI + VWAP deviation + momentum
    5. FadeStrategy    - Order flow imbalance contrarian fading
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .indicators import (
    compute_rsi,
    compute_atr,
    compute_ema,
    compute_sma,
    compute_bollinger_bands,
    compute_adx,
    compute_stochastic,
    compute_vwap,
    compute_keltner_channels,
    compute_volatility_ratio,
)


class AdaptiveBaseStrategy(ABC):
    """
    Abstract base class for adaptive strategies.

    All strategies must implement generate_signals() returning (n, 3) arrays.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals from bar data.

        Parameters
        ----------
        bars : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume
            (volume can be tick_count).

        Returns
        -------
        np.ndarray
            Shape (n, 3): [direction, sl_dist, tp_dist]
            direction: 1=buy, -1=sell, 0=no signal
            sl_dist: stop-loss distance from entry price
            tp_dist: take-profit distance from entry price
        """
        pass

    def get_name(self) -> str:
        """Return strategy name."""
        return self.name


class TrendFollower(AdaptiveBaseStrategy):
    """
    Trend-Following Strategy.

    Logic:
        - Uses fast/slow EMA crossover to identify trend direction
        - ADX > threshold as trend strength filter
        - Enters on pullback to fast EMA in trend direction
        - Tighter stop, wider target (rides the trend)

    Best in: TRENDING_UP, TRENDING_DOWN regimes.
    """

    def __init__(self, config: Optional[Dict] = None):
        default = {
            "name": "TrendFollower",
            "fast_ema": 8,
            "slow_ema": 21,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "pullback_tolerance": 0.3,  # How close to fast EMA for pullback (ATR units)
            "trend_bars_confirm": 3,     # Bars EMA must be aligned before entry
        }
        if config:
            default.update(config)
        super().__init__(default)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate trend-following signals with pullback entry."""
        n = len(bars)
        signals = np.zeros((n, 3), dtype=np.float64)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)

        cfg = self.config
        fast_ema = compute_ema(close, cfg["fast_ema"])
        slow_ema = compute_ema(close, cfg["slow_ema"])
        adx, di_plus, di_minus = compute_adx(high, low, close, cfg["adx_period"])
        atr = compute_atr(high, low, close, cfg["atr_period"])

        warmup = max(cfg["slow_ema"], cfg["adx_period"] * 2) + cfg["trend_bars_confirm"]

        for i in range(warmup, n):
            if atr[i] < 1e-10:
                continue

            # ADX filter: only trade when trend is strong enough
            if adx[i] < cfg["adx_threshold"]:
                continue

            sl_dist = atr[i] * cfg["sl_atr_mult"]
            tp_dist = atr[i] * cfg["tp_atr_mult"]

            # Check EMA alignment for trend_bars_confirm consecutive bars
            bullish_aligned = all(
                fast_ema[i - j] > slow_ema[i - j]
                for j in range(cfg["trend_bars_confirm"])
            )
            bearish_aligned = all(
                fast_ema[i - j] < slow_ema[i - j]
                for j in range(cfg["trend_bars_confirm"])
            )

            # Pullback detection: price near fast EMA
            pullback_dist = abs(close[i] - fast_ema[i]) / atr[i]

            if bullish_aligned and di_plus[i] > di_minus[i]:
                # Uptrend: look for pullback to fast EMA from above
                if pullback_dist < cfg["pullback_tolerance"] and close[i] >= fast_ema[i] - atr[i] * 0.5:
                    signals[i] = [1.0, sl_dist, tp_dist]

            elif bearish_aligned and di_minus[i] > di_plus[i]:
                # Downtrend: look for pullback to fast EMA from below
                if pullback_dist < cfg["pullback_tolerance"] and close[i] <= fast_ema[i] + atr[i] * 0.5:
                    signals[i] = [-1.0, sl_dist, tp_dist]

        return signals


class MeanReversion(AdaptiveBaseStrategy):
    """
    Mean Reversion Strategy.

    Logic:
        - Price touches Bollinger Band + RSI confirmation
        - Keltner Channel squeeze filter (skip during squeeze)
        - Target is the Bollinger middle band (mean)
        - Works best in ranging/mean-reverting markets

    Best in: RANGING_WIDE, MEAN_REVERTING regimes.
    """

    def __init__(self, config: Optional[Dict] = None):
        default = {
            "name": "MeanReversion",
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "sl_atr_mult": 1.2,
            "tp_atr_mult": 1.5,
            "keltner_period": 20,
            "keltner_mult": 1.5,
            "use_keltner_filter": True,
        }
        if config:
            default.update(config)
        super().__init__(default)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate mean-reversion signals on Bollinger Band touches."""
        n = len(bars)
        signals = np.zeros((n, 3), dtype=np.float64)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)

        cfg = self.config
        upper_bb, middle_bb, lower_bb = compute_bollinger_bands(
            close, cfg["bb_period"], cfg["bb_std"]
        )
        rsi = compute_rsi(close, cfg["rsi_period"])
        atr = compute_atr(high, low, close, cfg["atr_period"])

        # Keltner Channels for squeeze detection
        kc_upper, kc_middle, kc_lower = compute_keltner_channels(
            high, low, close, cfg["keltner_period"], cfg["atr_period"], cfg["keltner_mult"]
        )

        warmup = max(cfg["bb_period"], cfg["rsi_period"], cfg["atr_period"]) + 1

        for i in range(warmup, n):
            if atr[i] < 1e-10 or np.isnan(upper_bb[i]):
                continue

            # Keltner squeeze filter: if BB inside KC, volatility is compressed
            # In squeeze, mean reversion is risky (breakout likely)
            if cfg["use_keltner_filter"]:
                in_squeeze = (upper_bb[i] < kc_upper[i]) and (lower_bb[i] > kc_lower[i])
                if in_squeeze:
                    continue

            sl_dist = atr[i] * cfg["sl_atr_mult"]
            # TP targets the mean (middle BB distance)
            tp_to_mean = abs(close[i] - middle_bb[i])
            tp_dist = max(tp_to_mean, atr[i] * cfg["tp_atr_mult"])

            # LONG: price at/below lower band + RSI oversold
            if close[i] <= lower_bb[i] and rsi[i] < cfg["rsi_oversold"]:
                signals[i] = [1.0, sl_dist, tp_dist]

            # SHORT: price at/above upper band + RSI overbought
            elif close[i] >= upper_bb[i] and rsi[i] > cfg["rsi_overbought"]:
                signals[i] = [-1.0, sl_dist, tp_dist]

        return signals


class BreakoutTrader(AdaptiveBaseStrategy):
    """
    Breakout Strategy.

    Logic:
        - Detects ATR compression (current ATR < 0.6 * average ATR for N bars)
        - Waits for breakout above/below the consolidation range
        - Volume confirmation (tick_count spike)
        - Wide stops, wider targets (breakouts can run)

    Best in: VOLATILE_BREAKOUT regime, transitioning from RANGING_NARROW.
    """

    def __init__(self, config: Optional[Dict] = None):
        default = {
            "name": "BreakoutTrader",
            "atr_period": 14,
            "compression_ratio": 0.6,   # ATR < this * avg_ATR = compressed
            "compression_bars": 5,       # Minimum bars of compression before valid
            "lookback_range": 20,        # Bars to compute consolidation range
            "volume_spike_mult": 1.5,    # Volume must be > this * avg volume
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 4.0,          # Wide target for breakout moves
            "avg_atr_lookback": 50,
        }
        if config:
            default.update(config)
        super().__init__(default)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate breakout signals after ATR compression."""
        n = len(bars)
        signals = np.zeros((n, 3), dtype=np.float64)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)

        # Volume: use tick_count or volume column
        if "tick_count" in bars.columns:
            volume = bars["tick_count"].values.astype(np.float64)
        elif "volume" in bars.columns:
            volume = bars["volume"].values.astype(np.float64)
        else:
            volume = np.ones(n, dtype=np.float64)

        cfg = self.config
        atr = compute_atr(high, low, close, cfg["atr_period"])
        vol_ratio = compute_volatility_ratio(atr, cfg["avg_atr_lookback"])

        warmup = max(cfg["avg_atr_lookback"], cfg["lookback_range"]) + cfg["compression_bars"]

        for i in range(warmup, n):
            if atr[i] < 1e-10:
                continue

            # Check for compression over last N bars
            compressed_count = 0
            for j in range(cfg["compression_bars"]):
                idx = i - 1 - j
                if idx >= 0 and vol_ratio[idx] < cfg["compression_ratio"]:
                    compressed_count += 1

            if compressed_count < cfg["compression_bars"]:
                continue  # Not enough compression

            # Current bar must show volatility expansion (breakout bar)
            if vol_ratio[i] < 0.9:
                continue  # Still compressed, no breakout yet

            # Volume confirmation
            avg_vol = np.mean(volume[max(0, i - cfg["avg_atr_lookback"]):i])
            if avg_vol > 0 and volume[i] < cfg["volume_spike_mult"] * avg_vol:
                continue  # No volume spike

            # Consolidation range (high/low of lookback period)
            range_high = np.max(high[i - cfg["lookback_range"]:i])
            range_low = np.min(low[i - cfg["lookback_range"]:i])

            sl_dist = atr[i] * cfg["sl_atr_mult"]
            tp_dist = atr[i] * cfg["tp_atr_mult"]

            # Breakout direction
            if close[i] > range_high:
                signals[i] = [1.0, sl_dist, tp_dist]
            elif close[i] < range_low:
                signals[i] = [-1.0, sl_dist, tp_dist]

        return signals


class ScalpMomentum(AdaptiveBaseStrategy):
    """
    Scalp Momentum Strategy.

    Logic:
        - Fast RSI (period 5) at extremes signals short-term momentum exhaustion
        - VWAP deviation confirms momentum direction
        - Strong momentum: N consecutive bars in same direction
        - Quick scalp with tight risk-reward (1:1.5 or 1:2)

    Best in: TRENDING_UP, TRENDING_DOWN (fast scalps with the trend).
    """

    def __init__(self, config: Optional[Dict] = None):
        default = {
            "name": "ScalpMomentum",
            "rsi_period": 5,
            "rsi_extreme_high": 80,
            "rsi_extreme_low": 20,
            "vwap_period": 20,
            "vwap_deviation_threshold": 0.5,  # ATR units deviation from VWAP
            "momentum_bars": 3,               # Consecutive same-direction bars
            "atr_period": 14,
            "sl_atr_mult": 0.8,               # Tight stop for scalps
            "tp_atr_mult": 1.6,               # Quick target
        }
        if config:
            default.update(config)
        super().__init__(default)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate momentum scalp signals."""
        n = len(bars)
        signals = np.zeros((n, 3), dtype=np.float64)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        open_price = bars["open"].values.astype(np.float64)

        if "tick_count" in bars.columns:
            volume = bars["tick_count"].values.astype(np.float64)
        elif "volume" in bars.columns:
            volume = bars["volume"].values.astype(np.float64)
        else:
            volume = np.ones(n, dtype=np.float64)

        cfg = self.config
        rsi = compute_rsi(close, cfg["rsi_period"])
        atr = compute_atr(high, low, close, cfg["atr_period"])
        vwap = compute_vwap(high, low, close, volume, cfg["vwap_period"])

        warmup = max(cfg["vwap_period"], cfg["rsi_period"], cfg["atr_period"]) + cfg["momentum_bars"]

        for i in range(warmup, n):
            if atr[i] < 1e-10:
                continue

            sl_dist = atr[i] * cfg["sl_atr_mult"]
            tp_dist = atr[i] * cfg["tp_atr_mult"]

            # VWAP deviation (normalized by ATR)
            vwap_dev = (close[i] - vwap[i]) / atr[i]

            # Momentum: consecutive bullish/bearish bars
            bullish_momentum = all(
                close[i - j] > open_price[i - j]
                for j in range(cfg["momentum_bars"])
            )
            bearish_momentum = all(
                close[i - j] < open_price[i - j]
                for j in range(cfg["momentum_bars"])
            )

            # BUY scalp: RSI not extreme high (still has room), 
            # price above VWAP (momentum), consecutive bullish bars
            if (rsi[i] > 50 and rsi[i] < cfg["rsi_extreme_high"]
                    and vwap_dev > cfg["vwap_deviation_threshold"]
                    and bullish_momentum):
                signals[i] = [1.0, sl_dist, tp_dist]

            # SELL scalp: RSI not extreme low, price below VWAP, bearish momentum
            elif (rsi[i] < 50 and rsi[i] > cfg["rsi_extreme_low"]
                  and vwap_dev < -cfg["vwap_deviation_threshold"]
                  and bearish_momentum):
                signals[i] = [-1.0, sl_dist, tp_dist]

        return signals


class FadeStrategy(AdaptiveBaseStrategy):
    """
    Fade (Contrarian) Strategy.

    Logic:
        - Detects order flow imbalance using bar close position and tick count
        - Computes z-score of the imbalance metric
        - Fades (trades against) extreme one-sided moves
        - Logic: when everyone is on one side, the market reverses

    Best in: VOLATILE_BREAKOUT (false breakouts), RANGING_WIDE (extreme touches).
    """

    def __init__(self, config: Optional[Dict] = None):
        default = {
            "name": "FadeStrategy",
            "imbalance_lookback": 20,
            "z_score_threshold": 2.0,     # Z-score to trigger fade
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.0,
            "min_tick_count": 5,          # Minimum ticks for valid imbalance
            "confirmation_bars": 2,        # Bars of extreme before fading
        }
        if config:
            default.update(config)
        super().__init__(default)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate contrarian fade signals on extreme order flow imbalance."""
        n = len(bars)
        signals = np.zeros((n, 3), dtype=np.float64)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        open_price = bars["open"].values.astype(np.float64)

        if "tick_count" in bars.columns:
            tick_count = bars["tick_count"].values.astype(np.float64)
        elif "volume" in bars.columns:
            tick_count = bars["volume"].values.astype(np.float64)
        else:
            tick_count = np.ones(n, dtype=np.float64)

        cfg = self.config
        atr = compute_atr(high, low, close, cfg["atr_period"])

        # Compute bar close position: where close is within the bar range
        # +1 = closed at high (bullish), -1 = closed at low (bearish)
        bar_range = high - low
        close_position = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if bar_range[i] > 1e-10:
                close_position[i] = 2.0 * (close[i] - low[i]) / bar_range[i] - 1.0

        # Order flow imbalance: close_position * log(tick_count)
        # Extreme values indicate one-sided pressure
        log_ticks = np.log(np.maximum(tick_count, 1.0))
        imbalance = close_position * log_ticks

        warmup = cfg["imbalance_lookback"] + cfg["confirmation_bars"]

        for i in range(warmup, n):
            if atr[i] < 1e-10:
                continue
            if tick_count[i] < cfg["min_tick_count"]:
                continue

            # Rolling z-score of imbalance
            window = imbalance[i - cfg["imbalance_lookback"]:i]
            mean_imb = np.mean(window)
            std_imb = np.std(window, ddof=1)

            if std_imb < 1e-10:
                continue

            z_score = (imbalance[i] - mean_imb) / std_imb

            sl_dist = atr[i] * cfg["sl_atr_mult"]
            tp_dist = atr[i] * cfg["tp_atr_mult"]

            # Check confirmation: extreme for multiple bars
            extreme_bullish_count = 0
            extreme_bearish_count = 0
            for j in range(cfg["confirmation_bars"]):
                idx = i - j
                if idx >= cfg["imbalance_lookback"]:
                    w = imbalance[idx - cfg["imbalance_lookback"]:idx]
                    m = np.mean(w)
                    s = np.std(w, ddof=1)
                    if s > 1e-10:
                        z = (imbalance[idx] - m) / s
                        if z > cfg["z_score_threshold"] * 0.7:
                            extreme_bullish_count += 1
                        elif z < -cfg["z_score_threshold"] * 0.7:
                            extreme_bearish_count += 1

            # FADE: if extreme bullish imbalance -> sell (fade the buying)
            if z_score > cfg["z_score_threshold"] and extreme_bullish_count >= cfg["confirmation_bars"]:
                signals[i] = [-1.0, sl_dist, tp_dist]

            # FADE: if extreme bearish imbalance -> buy (fade the selling)
            elif z_score < -cfg["z_score_threshold"] and extreme_bearish_count >= cfg["confirmation_bars"]:
                signals[i] = [1.0, sl_dist, tp_dist]

        return signals


# Strategy registry for easy lookup
STRATEGY_REGISTRY: Dict[str, type] = {
    "TrendFollower": TrendFollower,
    "MeanReversion": MeanReversion,
    "BreakoutTrader": BreakoutTrader,
    "ScalpMomentum": ScalpMomentum,
    "FadeStrategy": FadeStrategy,
}


def create_strategy(name: str, config: Optional[Dict] = None) -> AdaptiveBaseStrategy:
    """
    Factory function to create a strategy by name.

    Parameters
    ----------
    name : str
        Strategy class name (must be in STRATEGY_REGISTRY).
    config : dict, optional
        Configuration overrides.

    Returns
    -------
    AdaptiveBaseStrategy
        Instantiated strategy.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](config)
