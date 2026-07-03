"""
Online Learning Module - Lightweight Adaptive Learning
========================================================
Implements online statistical learning WITHOUT external ML libraries.
All computation is pure Python + NumPy.

Components:
    1. ExponentialStats       - Tracks mean, variance, count with exponential decay
    2. StrategyPerformanceTracker - One ExponentialStats per (strategy, regime) pair
    3. RegimeTransitionMatrix - Counts transitions, normalizes to probabilities
    4. ParameterAdapter       - Tracks parameter-outcome relationships
    5. MarketProfiler         - Rolling statistics on market microstructure
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from .regime_detector import MarketRegime


@dataclass
class ExponentialStats:
    """
    Tracks running statistics with exponential decay.

    Maintains exponentially-weighted mean, variance, min, max, and count.
    Recent observations have more weight (controlled by alpha).

    Usage:
        stats = ExponentialStats(alpha=0.05)
        stats.update(1.5)
        stats.update(-0.3)
        print(f"Mean: {stats.mean}, Std: {stats.std()}")
    """
    alpha: float = 0.05          # Decay rate (higher = faster adaptation)
    mean: float = 0.0
    variance: float = 0.0
    count: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    _initialized: bool = False

    def update(self, value: float):
        """Add a new observation with exponential weighting."""
        if not self._initialized:
            self.mean = value
            self.variance = 0.0
            self.count = 1.0
            self.min_val = value
            self.max_val = value
            self._initialized = True
            return

        # Exponential decay of count
        self.count = self.count * (1.0 - self.alpha) + 1.0

        # Update mean with exponential smoothing
        delta = value - self.mean
        self.mean = self.mean + self.alpha * delta

        # Update variance (Welford-like with exponential weighting)
        delta2 = value - self.mean
        self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * delta * delta2)

        # Track extremes
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def std(self) -> float:
        """Return exponentially-weighted standard deviation."""
        return np.sqrt(max(0.0, self.variance))

    def z_score(self, value: float) -> float:
        """Compute z-score of a value relative to tracked distribution."""
        s = self.std()
        if s < 1e-10:
            return 0.0
        return (value - self.mean) / s

    def confidence_interval(self, sigma: float = 1.96) -> Tuple[float, float]:
        """Return confidence interval around the mean."""
        s = self.std()
        return (self.mean - sigma * s, self.mean + sigma * s)

    def reset(self):
        """Reset all tracked statistics."""
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self._initialized = False


class StrategyPerformanceTracker:
    """
    Tracks performance statistics per (strategy, regime) pair.

    Each combination gets its own ExponentialStats for:
    - PnL distribution
    - Win/loss tracking
    - Duration (how long trades last)
    - Risk-reward realized ratio

    Usage:
        tracker = StrategyPerformanceTracker()
        tracker.record_trade("TrendFollower", MarketRegime.TRENDING_UP, pnl=15.0, duration=12)
        stats = tracker.get_stats("TrendFollower", MarketRegime.TRENDING_UP)
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._pnl_stats: Dict[Tuple[str, MarketRegime], ExponentialStats] = {}
        self._win_stats: Dict[Tuple[str, MarketRegime], ExponentialStats] = {}
        self._duration_stats: Dict[Tuple[str, MarketRegime], ExponentialStats] = {}
        self._trade_counts: Dict[Tuple[str, MarketRegime], int] = defaultdict(int)

    def record_trade(self, strategy_name: str, regime: MarketRegime,
                     pnl: float, duration: int = 0):
        """
        Record a completed trade for tracking.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy.
        regime : MarketRegime
            Market regime when trade was entered.
        pnl : float
            Trade profit/loss.
        duration : int
            Trade duration in bars.
        """
        key = (strategy_name, regime)
        self._trade_counts[key] += 1

        # PnL stats
        if key not in self._pnl_stats:
            self._pnl_stats[key] = ExponentialStats(alpha=self.alpha)
        self._pnl_stats[key].update(pnl)

        # Win/loss tracking (1 for win, 0 for loss)
        if key not in self._win_stats:
            self._win_stats[key] = ExponentialStats(alpha=self.alpha)
        self._win_stats[key].update(1.0 if pnl > 0 else 0.0)

        # Duration tracking
        if duration > 0:
            if key not in self._duration_stats:
                self._duration_stats[key] = ExponentialStats(alpha=self.alpha)
            self._duration_stats[key].update(float(duration))

    def get_pnl_stats(self, strategy_name: str, regime: MarketRegime) -> ExponentialStats:
        """Get PnL statistics for a strategy-regime pair."""
        key = (strategy_name, regime)
        if key not in self._pnl_stats:
            self._pnl_stats[key] = ExponentialStats(alpha=self.alpha)
        return self._pnl_stats[key]

    def get_win_rate(self, strategy_name: str, regime: MarketRegime) -> float:
        """Get exponentially-weighted win rate for a strategy-regime pair."""
        key = (strategy_name, regime)
        if key not in self._win_stats:
            return 0.5  # No data, assume neutral
        return self._win_stats[key].mean

    def get_trade_count(self, strategy_name: str, regime: MarketRegime) -> int:
        """Get total trade count for a strategy-regime pair."""
        return self._trade_counts.get((strategy_name, regime), 0)

    def get_sharpe_estimate(self, strategy_name: str, regime: MarketRegime) -> float:
        """
        Estimate Sharpe ratio for a strategy-regime pair.
        Returns mean_pnl / std_pnl.
        """
        stats = self.get_pnl_stats(strategy_name, regime)
        if stats.count < 3 or stats.std() < 1e-10:
            return 0.0
        return stats.mean / stats.std()

    def get_best_strategy(self, regime: MarketRegime,
                          strategy_names: List[str]) -> str:
        """Get the best-performing strategy for a given regime."""
        best_name = strategy_names[0] if strategy_names else ""
        best_sharpe = -999.0

        for name in strategy_names:
            sharpe = self.get_sharpe_estimate(name, regime)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = name

        return best_name

    def summary(self) -> Dict[str, Dict]:
        """Get a summary of all tracked performance data."""
        result = {}
        for key, stats in self._pnl_stats.items():
            strategy_name, regime = key
            label = f"{strategy_name}|{regime.name}"
            result[label] = {
                "mean_pnl": round(stats.mean, 4),
                "std_pnl": round(stats.std(), 4),
                "win_rate": round(self.get_win_rate(strategy_name, regime), 4),
                "trade_count": self._trade_counts.get(key, 0),
                "sharpe": round(self.get_sharpe_estimate(strategy_name, regime), 4),
            }
        return result


class RegimeTransitionMatrix:
    """
    Tracks regime-to-regime transition probabilities.
    Updated online as new regime classifications arrive.

    Usage:
        matrix = RegimeTransitionMatrix()
        matrix.update(MarketRegime.RANGING_NARROW, MarketRegime.VOLATILE_BREAKOUT)
        probs = matrix.get_transition_probs(MarketRegime.RANGING_NARROW)
        # probs = {VOLATILE_BREAKOUT: 0.4, TRENDING_UP: 0.2, ...}
    """

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self._regimes = list(MarketRegime)
        n = len(self._regimes)
        self._regime_idx = {r: i for i, r in enumerate(self._regimes)}

        # Transition count matrix (exponentially decayed)
        self._counts = np.ones((n, n), dtype=np.float64)  # Start with uniform prior
        self._total_transitions = n * n  # Prior count

        self._last_regime: Optional[MarketRegime] = None

    def update(self, from_regime: MarketRegime, to_regime: MarketRegime):
        """
        Record a regime transition.

        Parameters
        ----------
        from_regime : MarketRegime
            Previous regime.
        to_regime : MarketRegime
            New regime.
        """
        i = self._regime_idx[from_regime]
        j = self._regime_idx[to_regime]

        # Decay all counts slightly (exponential forgetting)
        self._counts *= (1.0 - self.alpha)
        # Add new observation
        self._counts[i, j] += 1.0
        self._total_transitions = np.sum(self._counts)
        self._last_regime = to_regime

    def observe(self, current_regime: MarketRegime):
        """
        Observe the current regime. If it differs from last observed, record transition.
        """
        if self._last_regime is not None and current_regime != self._last_regime:
            self.update(self._last_regime, current_regime)
        self._last_regime = current_regime

    def get_transition_probs(self, from_regime: MarketRegime) -> Dict[MarketRegime, float]:
        """
        Get probability distribution over next regime given current regime.

        Returns
        -------
        Dict[MarketRegime, float]
            Probability of transitioning to each regime.
        """
        i = self._regime_idx[from_regime]
        row = self._counts[i]
        row_sum = np.sum(row)

        if row_sum < 1e-10:
            # Uniform if no data
            n = len(self._regimes)
            return {r: 1.0 / n for r in self._regimes}

        probs = row / row_sum
        return {self._regimes[j]: float(probs[j]) for j in range(len(self._regimes))}

    def most_likely_next(self, from_regime: MarketRegime) -> Tuple[MarketRegime, float]:
        """Get the most likely next regime and its probability."""
        probs = self.get_transition_probs(from_regime)
        best_regime = max(probs, key=probs.get)
        return best_regime, probs[best_regime]

    def get_matrix(self) -> np.ndarray:
        """Get the full transition probability matrix."""
        row_sums = self._counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
        return self._counts / row_sums


class ParameterAdapter:
    """
    Tracks which parameter values lead to better outcomes.
    Uses gradient-free optimization by tracking success rates for parameter ranges.

    For each parameter, maintains a distribution of recent outcomes bucketed
    by parameter value, then suggests adjustments toward better-performing values.

    Usage:
        adapter = ParameterAdapter()
        adapter.record("sl_mult", 1.5, pnl=10.0)
        adapter.record("sl_mult", 2.0, pnl=-5.0)
        adapter.record("sl_mult", 1.5, pnl=8.0)
        suggestion = adapter.suggest("sl_mult", current_value=2.0)
        # suggestion might be 1.5 (better performing value)
    """

    def __init__(self, alpha: float = 0.05, num_buckets: int = 10):
        self.alpha = alpha
        self.num_buckets = num_buckets
        # param_name -> {bucket_center: ExponentialStats}
        self._param_stats: Dict[str, Dict[float, ExponentialStats]] = defaultdict(dict)
        # Track min/max for each parameter to define buckets
        self._param_ranges: Dict[str, Tuple[float, float]] = {}

    def record(self, param_name: str, param_value: float, pnl: float):
        """
        Record an outcome for a specific parameter value.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        param_value : float
            Value the parameter was set to.
        pnl : float
            Resulting trade PnL.
        """
        # Update range
        if param_name in self._param_ranges:
            old_min, old_max = self._param_ranges[param_name]
            self._param_ranges[param_name] = (
                min(old_min, param_value),
                max(old_max, param_value)
            )
        else:
            self._param_ranges[param_name] = (param_value, param_value)

        # Bucket the value
        bucket = self._get_bucket(param_name, param_value)

        if bucket not in self._param_stats[param_name]:
            self._param_stats[param_name][bucket] = ExponentialStats(alpha=self.alpha)
        self._param_stats[param_name][bucket].update(pnl)

    def suggest(self, param_name: str, current_value: float,
                step_size: float = 0.1) -> float:
        """
        Suggest an adjusted parameter value based on learned outcomes.

        Moves toward the bucket with highest average PnL.

        Parameters
        ----------
        param_name : str
            Parameter to adjust.
        current_value : float
            Current parameter value.
        step_size : float
            Maximum fraction to adjust by (relative to range).

        Returns
        -------
        float
            Suggested new parameter value.
        """
        if param_name not in self._param_stats:
            return current_value

        stats = self._param_stats[param_name]
        if not stats:
            return current_value

        # Find best-performing bucket
        best_bucket = current_value
        best_mean = -float('inf')

        for bucket, s in stats.items():
            if s.count >= 3 and s.mean > best_mean:
                best_mean = s.mean
                best_bucket = bucket

        # Move toward best bucket, limited by step_size
        if param_name in self._param_ranges:
            min_val, max_val = self._param_ranges[param_name]
            param_range = max_val - min_val
            if param_range > 1e-10:
                max_step = param_range * step_size
                delta = best_bucket - current_value
                delta = np.clip(delta, -max_step, max_step)
                return current_value + delta

        return current_value

    def _get_bucket(self, param_name: str, value: float) -> float:
        """Bucket a parameter value for tracking."""
        if param_name not in self._param_ranges:
            return value

        min_val, max_val = self._param_ranges[param_name]
        if max_val - min_val < 1e-10:
            return value

        # Quantize to bucket centers
        bucket_width = (max_val - min_val) / self.num_buckets
        bucket_idx = int((value - min_val) / bucket_width)
        bucket_idx = max(0, min(self.num_buckets - 1, bucket_idx))
        return min_val + (bucket_idx + 0.5) * bucket_width

    def get_summary(self, param_name: str) -> Dict[float, float]:
        """Get mean PnL for each bucket of a parameter."""
        if param_name not in self._param_stats:
            return {}
        return {
            bucket: s.mean
            for bucket, s in self._param_stats[param_name].items()
            if s.count >= 2
        }


class MarketProfiler:
    """
    Rolling market microstructure statistics.

    Tracks:
    - Spread patterns (average, std, time-of-day variation)
    - Volatility cycles (session-based, day-of-week)
    - Volume patterns (time-of-day, session averages)
    - Price movement characteristics (autocorrelation, mean bar size)

    Usage:
        profiler = MarketProfiler()
        profiler.update(spread=0.15, volatility=2.5, volume=150, hour=14, bar_return=0.001)
        session_vol = profiler.get_session_volatility(14)
    """

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha

        # Per-hour statistics (0-23)
        self._hourly_spread: Dict[int, ExponentialStats] = {
            h: ExponentialStats(alpha=alpha) for h in range(24)
        }
        self._hourly_volatility: Dict[int, ExponentialStats] = {
            h: ExponentialStats(alpha=alpha) for h in range(24)
        }
        self._hourly_volume: Dict[int, ExponentialStats] = {
            h: ExponentialStats(alpha=alpha) for h in range(24)
        }

        # Overall statistics
        self._spread_stats = ExponentialStats(alpha=alpha)
        self._volatility_stats = ExponentialStats(alpha=alpha)
        self._volume_stats = ExponentialStats(alpha=alpha)
        self._return_stats = ExponentialStats(alpha=alpha)

        # Autocorrelation tracking (recent returns buffer)
        self._recent_returns: List[float] = []
        self._max_return_buffer: int = 100

    def update(self, spread: float = 0.0, volatility: float = 0.0,
               volume: float = 0.0, hour: int = 0, bar_return: float = 0.0):
        """
        Update market profile with new bar data.

        Parameters
        ----------
        spread : float
            Current spread.
        volatility : float
            Current volatility (ATR or range).
        volume : float
            Bar volume or tick count.
        hour : int
            Hour of day (0-23).
        bar_return : float
            Bar return (close/open - 1 or similar).
        """
        hour = hour % 24

        # Update per-hour stats
        if spread > 0:
            self._hourly_spread[hour].update(spread)
            self._spread_stats.update(spread)

        if volatility > 0:
            self._hourly_volatility[hour].update(volatility)
            self._volatility_stats.update(volatility)

        if volume > 0:
            self._hourly_volume[hour].update(volume)
            self._volume_stats.update(volume)

        # Return tracking
        self._return_stats.update(bar_return)
        self._recent_returns.append(bar_return)
        if len(self._recent_returns) > self._max_return_buffer:
            self._recent_returns = self._recent_returns[-self._max_return_buffer:]

    def get_session_volatility(self, hour: int) -> float:
        """Get expected volatility for a given hour."""
        hour = hour % 24
        stats = self._hourly_volatility[hour]
        if stats.count > 0:
            return stats.mean
        return self._volatility_stats.mean if self._volatility_stats.count > 0 else 0.0

    def get_session_spread(self, hour: int) -> float:
        """Get expected spread for a given hour."""
        hour = hour % 24
        stats = self._hourly_spread[hour]
        if stats.count > 0:
            return stats.mean
        return self._spread_stats.mean if self._spread_stats.count > 0 else 0.0

    def get_session_volume(self, hour: int) -> float:
        """Get expected volume for a given hour."""
        hour = hour % 24
        stats = self._hourly_volume[hour]
        if stats.count > 0:
            return stats.mean
        return self._volume_stats.mean if self._volume_stats.count > 0 else 0.0

    def get_autocorrelation(self, lag: int = 1) -> float:
        """
        Compute autocorrelation of recent returns at given lag.

        Returns value in [-1, 1]. Negative = mean-reverting, Positive = trending.
        """
        returns = np.array(self._recent_returns)
        n = len(returns)
        if n < lag + 10:
            return 0.0

        x = returns[:-lag]
        y = returns[lag:]
        if len(x) < 2:
            return 0.0

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)

        if std_x < 1e-10 or std_y < 1e-10:
            return 0.0

        correlation = np.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
        return float(np.clip(correlation, -1.0, 1.0))

    def is_high_spread_session(self, hour: int, threshold_mult: float = 1.5) -> bool:
        """Check if a given hour typically has high spreads."""
        session_spread = self.get_session_spread(hour)
        avg_spread = self._spread_stats.mean if self._spread_stats.count > 0 else session_spread
        return session_spread > threshold_mult * avg_spread if avg_spread > 0 else False

    def get_profile_summary(self) -> Dict:
        """Get a summary of the market profile."""
        return {
            "avg_spread": round(self._spread_stats.mean, 4) if self._spread_stats.count > 0 else 0,
            "avg_volatility": round(self._volatility_stats.mean, 4) if self._volatility_stats.count > 0 else 0,
            "avg_volume": round(self._volume_stats.mean, 2) if self._volume_stats.count > 0 else 0,
            "return_autocorr_lag1": round(self.get_autocorrelation(1), 4),
            "return_mean": round(self._return_stats.mean, 6) if self._return_stats.count > 0 else 0,
            "return_std": round(self._return_stats.std(), 6) if self._return_stats.count > 0 else 0,
        }
