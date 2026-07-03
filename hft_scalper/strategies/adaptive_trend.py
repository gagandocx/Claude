"""
Adaptive trend-following strategy for aggressive scalping.

Detects regime (trending vs ranging) and adapts:
- Trending: enter on pullbacks to EMA with wide targets
- Ranging: mean-reversion at Bollinger extremes
- Session filter for London/NY
"""

import numpy as np
import pandas as pd
from .base import BaseStrategy


class AdaptiveTrendStrategy(BaseStrategy):
    """Adaptive trend-following strategy with regime detection."""

    def __init__(self, config=None):
        default_config = {
            "name": "AdaptiveTrend",
            "fast_ema": 12,
            "slow_ema": 50,
            "trend_ema": 100,
            "atr_period": 14,
            "adx_period": 14,
            "adx_trend_threshold": 20,
            "rsi_period": 14,
            "rsi_ob": 75,
            "rsi_os": 25,
            "bb_period": 20,
            "bb_std": 2.0,
            "trend_sl_mult": 1.2,
            "trend_tp_mult": 3.5,
            "range_sl_mult": 1.8,
            "range_tp_mult": 2.2,
            "active_hours": list(range(7, 21)),  # London + NY sessions UTC
            "warmup": 100,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def generate_signals(self, bars: pd.DataFrame) -> np.ndarray:
        """Generate adaptive trend signals."""
        n = len(bars)
        signals = np.zeros((n, 3))

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        open_ = bars["open"].values

        # Calculate indicators
        fast_ema = self._ema(close, self.config["fast_ema"])
        slow_ema = self._ema(close, self.config["slow_ema"])
        trend_ema = self._ema(close, self.config["trend_ema"])

        atr = self._atr(high, low, close, self.config["atr_period"])
        adx = self._adx(high, low, close, self.config["adx_period"])
        rsi = self._rsi(close, self.config["rsi_period"])

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = self._bollinger(
            close, self.config["bb_period"], self.config["bb_std"]
        )

        # Session filter
        if hasattr(bars.index, 'hour'):
            hours = bars.index.hour
        else:
            hours = np.full(n, 12)  # Default to active

        warmup = self.config["warmup"]

        for i in range(warmup, n):
            # Session filter
            if hours[i] not in self.config["active_hours"]:
                continue

            # Skip if ATR is too small (no volatility)
            if atr[i] < 0.5:
                continue

            # Determine regime
            is_trending = adx[i] > self.config["adx_trend_threshold"]

            if is_trending:
                # TRENDING REGIME: enter on pullbacks
                trend_up = fast_ema[i] > slow_ema[i] and close[i] > trend_ema[i]
                trend_down = fast_ema[i] < slow_ema[i] and close[i] < trend_ema[i]

                if trend_up:
                    # Buy on pullback to fast EMA
                    pullback = close[i] <= fast_ema[i] * 1.001 and close[i] > fast_ema[i] * 0.998
                    # Additional: RSI not overbought
                    rsi_ok = rsi[i] < self.config["rsi_ob"]
                    # Price bouncing (close > open)
                    bullish_bar = close[i] > open_[i]

                    if pullback and rsi_ok and bullish_bar:
                        sl_dist = atr[i] * self.config["trend_sl_mult"]
                        tp_dist = atr[i] * self.config["trend_tp_mult"]
                        signals[i] = [1, sl_dist, tp_dist]

                elif trend_down:
                    # Sell on pullback to fast EMA
                    pullback = close[i] >= fast_ema[i] * 0.999 and close[i] < fast_ema[i] * 1.002
                    rsi_ok = rsi[i] > self.config["rsi_os"]
                    bearish_bar = close[i] < open_[i]

                    if pullback and rsi_ok and bearish_bar:
                        sl_dist = atr[i] * self.config["trend_sl_mult"]
                        tp_dist = atr[i] * self.config["trend_tp_mult"]
                        signals[i] = [-1, sl_dist, tp_dist]

            else:
                # RANGING REGIME: mean-reversion at BB extremes
                if close[i] <= bb_lower[i] and rsi[i] < self.config["rsi_os"]:
                    # Buy at lower BB
                    sl_dist = atr[i] * self.config["range_sl_mult"]
                    tp_dist = atr[i] * self.config["range_tp_mult"]
                    signals[i] = [1, sl_dist, tp_dist]

                elif close[i] >= bb_upper[i] and rsi[i] > self.config["rsi_ob"]:
                    # Sell at upper BB
                    sl_dist = atr[i] * self.config["range_sl_mult"]
                    tp_dist = atr[i] * self.config["range_tp_mult"]
                    signals[i] = [-1, sl_dist, tp_dist]

        return signals

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        # Fill initial values
        result[:period - 1] = result[period - 1]
        return result

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
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

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Simplified ADX calculation."""
        n = len(high)
        adx = np.zeros(n)

        if n < period * 2:
            return adx

        # Calculate +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        # Smooth with EMA
        smooth_tr = np.zeros(n)
        smooth_plus = np.zeros(n)
        smooth_minus = np.zeros(n)

        smooth_tr[period] = np.sum(tr[1:period + 1])
        smooth_plus[period] = np.sum(plus_dm[1:period + 1])
        smooth_minus[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
            smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
            smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

        # DI+ and DI-
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if smooth_tr[i] > 0:
                plus_di[i] = 100 * smooth_plus[i] / smooth_tr[i]
                minus_di[i] = 100 * smooth_minus[i] / smooth_tr[i]
            denom = plus_di[i] + minus_di[i]
            if denom > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

        # Smooth DX to get ADX
        start = period * 2
        if start < n:
            adx[start] = np.mean(dx[period:start + 1])
            for i in range(start + 1, n):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index."""
        n = len(close)
        rsi = np.full(n, 50.0)

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if n < period + 1:
            return rsi

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def _bollinger(close: np.ndarray, period: int, num_std: float):
        """Bollinger Bands."""
        n = len(close)
        mid = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)

        for i in range(period - 1, n):
            window = close[i - period + 1:i + 1]
            m = np.mean(window)
            s = np.std(window)
            mid[i] = m
            upper[i] = m + num_std * s
            lower[i] = m - num_std * s

        # Fill initial
        mid[:period - 1] = mid[period - 1]
        upper[:period - 1] = upper[period - 1]
        lower[:period - 1] = lower[period - 1]

        return mid, upper, lower
