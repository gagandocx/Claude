"""
Strategy Selector - Regime-Aware Strategy Router
=================================================
Routes detected market regimes to optimal strategies using performance tracking
and Bayesian confidence updating.

Key Features:
    - Configurable regime-to-strategy mapping
    - Performance scorecard per (strategy, regime) pair
    - Bayesian confidence: tracks wins, losses, avg_pnl per combo
    - Selects highest expected-Sharpe strategy for current regime
    - Confidence gate: low-confidence regime halves position size
    - Ensemble mode: can blend signals from top-2 strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .regime_detector import MarketRegime
from .strategies import (
    AdaptiveBaseStrategy,
    TrendFollower,
    MeanReversion,
    BreakoutTrader,
    ScalpMomentum,
    FadeStrategy,
    STRATEGY_REGISTRY,
    create_strategy,
)


@dataclass
class PerformanceRecord:
    """
    Exponentially-weighted performance statistics for a strategy-regime pair.

    Uses an EMA of the binary win/loss signal (1.0 for win, 0.0 for loss)
    so that the most recent 10-15 trades dominate the win_rate estimate.
    This ensures fast adaptation when market conditions change.
    """
    # EMA of win/loss binary signal (1=win, 0=loss) - this IS the win rate
    ema_win_rate: float = 0.5  # Prior: assume 50/50 before any data
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_estimate: float = 0.0
    trade_count: float = 0.0
    # Exponential decay factor - alpha=0.12 means ~8 recent trades carry 63% of weight
    alpha: float = 0.12
    _has_data: bool = False

    def update(self, pnl: float):
        """Update statistics with a new trade result."""
        self.trade_count = self.trade_count * (1.0 - self.alpha) + 1.0
        self.total_pnl = self.total_pnl * (1.0 - self.alpha) + pnl

        # Binary win/loss signal fed into EMA
        win_signal = 1.0 if pnl > 0 else 0.0

        if not self._has_data:
            # First observation: initialize directly
            self.ema_win_rate = win_signal
            self._has_data = True
        else:
            # Exponential moving average of binary win/loss
            self.ema_win_rate = self.ema_win_rate * (1.0 - self.alpha) + self.alpha * win_signal

        if pnl > 0:
            self.avg_win = self.avg_win * (1.0 - self.alpha) + self.alpha * pnl
        else:
            self.avg_loss = self.avg_loss * (1.0 - self.alpha) + self.alpha * abs(pnl)

        # Update average PnL with exponential smoothing
        self.avg_pnl = self.avg_pnl * (1.0 - self.alpha) + self.alpha * pnl

        # Estimate Sharpe-like metric: mean_pnl / std_pnl (approximation)
        wr = self.win_rate()
        if wr > 0 and wr < 1.0 and self.avg_loss > 1e-10:
            expectancy = wr * self.avg_win - (1.0 - wr) * self.avg_loss
            risk = max(self.avg_loss, 1e-6)
            self.sharpe_estimate = expectancy / risk
        elif wr >= 0.95:
            self.sharpe_estimate = 2.0  # Cap at excellent
        else:
            self.sharpe_estimate = -1.0

    def win_rate(self) -> float:
        """
        Calculate recency-weighted win rate using EMA of binary signal.

        Returns a value between 0 and 1 that responds quickly to regime
        changes. After ~10-12 trades in a new regime, the win_rate fully
        reflects recent performance.
        """
        if not self._has_data:
            return 0.5  # Prior: assume 50/50
        return max(0.0, min(1.0, self.ema_win_rate))

    def expected_pnl(self) -> float:
        """Calculate expected PnL per trade."""
        wr = self.win_rate()
        return wr * self.avg_win - (1.0 - wr) * self.avg_loss

    def confidence_level(self) -> float:
        """
        How confident we are in this performance estimate.
        More trades = higher confidence (saturates at 1.0).
        """
        return min(1.0, self.trade_count / 20.0)


@dataclass
class StrategyScorecard:
    """
    Complete scorecard tracking performance across all regime-strategy combinations.
    """
    records: Dict[Tuple[str, MarketRegime], PerformanceRecord] = field(default_factory=dict)
    alpha: float = 0.05

    def get_record(self, strategy_name: str, regime: MarketRegime) -> PerformanceRecord:
        """Get or create performance record for a strategy-regime pair."""
        key = (strategy_name, regime)
        if key not in self.records:
            self.records[key] = PerformanceRecord(alpha=self.alpha)
        return self.records[key]

    def update(self, strategy_name: str, regime: MarketRegime, pnl: float):
        """Record a trade result for the given strategy-regime pair."""
        record = self.get_record(strategy_name, regime)
        record.update(pnl)

    def best_strategy_for_regime(self, regime: MarketRegime,
                                 available_strategies: List[str]) -> Tuple[str, float]:
        """
        Select the best strategy for the given regime based on expected Sharpe.

        Returns
        -------
        Tuple[str, float]
            (strategy_name, expected_sharpe)
        """
        best_name = available_strategies[0] if available_strategies else "TrendFollower"
        best_score = -999.0

        for name in available_strategies:
            record = self.get_record(name, regime)
            score = record.sharpe_estimate

            # Bayesian prior: blend with prior based on confidence
            confidence = record.confidence_level()
            prior_score = 0.0  # Neutral prior
            blended_score = confidence * score + (1.0 - confidence) * prior_score

            if blended_score > best_score:
                best_score = blended_score
                best_name = name

        return best_name, best_score

    def top_n_strategies(self, regime: MarketRegime,
                         available_strategies: List[str],
                         n: int = 2) -> List[Tuple[str, float]]:
        """Get top N strategies sorted by expected performance for a regime."""
        scores = []
        for name in available_strategies:
            record = self.get_record(name, regime)
            confidence = record.confidence_level()
            blended = confidence * record.sharpe_estimate + (1.0 - confidence) * 0.0
            scores.append((name, blended))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


@dataclass
class SelectorConfig:
    """Configuration for the strategy selector."""
    # Default regime-to-strategy mapping (used before learning kicks in)
    regime_mapping: Dict[MarketRegime, List[str]] = field(default_factory=lambda: {
        MarketRegime.TRENDING_UP: ["ScalpMomentum", "TrendFollower"],
        MarketRegime.TRENDING_DOWN: ["ScalpMomentum", "TrendFollower"],
        MarketRegime.RANGING_NARROW: ["MeanReversion", "FadeStrategy"],
        MarketRegime.RANGING_WIDE: ["MeanReversion", "FadeStrategy"],
        MarketRegime.VOLATILE_BREAKOUT: ["BreakoutTrader", "ScalpMomentum"],
        MarketRegime.MEAN_REVERTING: ["MeanReversion", "FadeStrategy"],
    })

    # Confidence gate threshold
    confidence_gate: float = 0.5
    # Position size reduction factor when below confidence gate
    low_confidence_size_mult: float = 0.5
    # Enable ensemble mode (blend top-2 strategies)
    ensemble_mode: bool = False
    # Minimum trade count before trusting learned performance
    min_trades_for_learning: float = 3.0
    # Scorecard exponential decay alpha
    scorecard_alpha: float = 0.05


class StrategySelector:
    """
    Regime-aware strategy router that selects the optimal strategy
    for the current market conditions.

    Usage:
        selector = StrategySelector()
        strategy, size_mult = selector.select(regime, confidence)
        # After trade closes:
        selector.record_result("TrendFollower", regime, pnl=12.5)
    """

    def __init__(self, config: Optional[SelectorConfig] = None):
        self.config = config or SelectorConfig()
        self.scorecard = StrategyScorecard(alpha=self.config.scorecard_alpha)
        self._strategies: Dict[str, AdaptiveBaseStrategy] = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Create all strategy instances."""
        for name in STRATEGY_REGISTRY:
            self._strategies[name] = create_strategy(name)

    def select(self, regime: MarketRegime,
               regime_confidence: float) -> Tuple[AdaptiveBaseStrategy, float]:
        """
        Select the best strategy for the current regime.

        Parameters
        ----------
        regime : MarketRegime
            Current detected regime.
        regime_confidence : float
            Confidence in the regime classification (0-1).

        Returns
        -------
        Tuple[AdaptiveBaseStrategy, float]
            (selected_strategy, position_size_multiplier)
            size_multiplier is 1.0 normally, reduced if confidence is low.
        """
        cfg = self.config

        # Get candidate strategies for this regime
        candidates = cfg.regime_mapping.get(
            regime, list(STRATEGY_REGISTRY.keys())
        )

        # Select best strategy based on scorecard
        best_name, best_score = self.scorecard.best_strategy_for_regime(regime, candidates)

        # If we don't have enough data, fall back to the default primary for this regime
        record = self.scorecard.get_record(best_name, regime)
        if record.trade_count < cfg.min_trades_for_learning:
            # Use the first strategy in the mapping (default pick)
            best_name = candidates[0]

        # Confidence gate: reduce position size if regime confidence is low
        size_mult = 1.0
        if regime_confidence < cfg.confidence_gate:
            size_mult = cfg.low_confidence_size_mult

        strategy = self._strategies.get(best_name)
        if strategy is None:
            strategy = self._strategies[list(self._strategies.keys())[0]]

        return strategy, size_mult

    def select_ensemble(self, regime: MarketRegime,
                        regime_confidence: float,
                        bars: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Ensemble mode: blend signals from top-2 strategies weighted by confidence.

        Parameters
        ----------
        regime : MarketRegime
            Current regime.
        regime_confidence : float
            Regime classification confidence.
        bars : pd.DataFrame
            Bar data for signal generation.

        Returns
        -------
        Tuple[np.ndarray, float]
            (blended_signals, position_size_multiplier)
        """
        cfg = self.config
        candidates = cfg.regime_mapping.get(regime, list(STRATEGY_REGISTRY.keys()))

        top_strategies = self.scorecard.top_n_strategies(regime, candidates, n=2)

        if len(top_strategies) < 2:
            # Fallback to single strategy
            strategy, size_mult = self.select(regime, regime_confidence)
            return strategy.generate_signals(bars), size_mult

        # Generate signals from top-2
        name1, score1 = top_strategies[0]
        name2, score2 = top_strategies[1]

        strategy1 = self._strategies[name1]
        strategy2 = self._strategies[name2]

        signals1 = strategy1.generate_signals(bars)
        signals2 = strategy2.generate_signals(bars)

        # Weight by relative scores (softmax-style)
        total_score = abs(score1) + abs(score2)
        if total_score > 1e-10:
            w1 = max(0.0, score1) / total_score if score1 > 0 else 0.3
            w2 = max(0.0, score2) / total_score if score2 > 0 else 0.3
        else:
            w1, w2 = 0.5, 0.5

        # Normalize weights
        w_sum = w1 + w2
        if w_sum > 0:
            w1 /= w_sum
            w2 /= w_sum
        else:
            w1, w2 = 0.5, 0.5

        # Blend: use primary strategy signal direction, average SL/TP
        n = len(bars)
        blended = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            dir1, dir2 = signals1[i, 0], signals2[i, 0]

            if dir1 != 0 and dir2 != 0:
                # Both have signals
                if dir1 == dir2:
                    # Agreement: use signal with weighted SL/TP
                    blended[i, 0] = dir1
                    blended[i, 1] = w1 * signals1[i, 1] + w2 * signals2[i, 1]
                    blended[i, 2] = w1 * signals1[i, 2] + w2 * signals2[i, 2]
                else:
                    # Conflict: use higher-weighted strategy
                    if w1 >= w2:
                        blended[i] = signals1[i]
                    else:
                        blended[i] = signals2[i]
            elif dir1 != 0:
                # Only strategy 1 has signal
                blended[i] = signals1[i]
                blended[i, 1] *= (1.0 + 0.2 * (1.0 - w1))  # Slightly wider SL if less confident
            elif dir2 != 0:
                # Only strategy 2 has signal
                blended[i] = signals2[i]
                blended[i, 1] *= (1.0 + 0.2 * (1.0 - w2))

        # Confidence gate
        size_mult = 1.0
        if regime_confidence < cfg.confidence_gate:
            size_mult = cfg.low_confidence_size_mult

        return blended, size_mult

    def record_result(self, strategy_name: str, regime: MarketRegime, pnl: float):
        """
        Record a trade result for learning.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy that produced the trade.
        regime : MarketRegime
            Regime when the trade was entered.
        pnl : float
            Profit/loss of the trade.
        """
        self.scorecard.update(strategy_name, regime, pnl)

    def get_strategy(self, name: str) -> Optional[AdaptiveBaseStrategy]:
        """Get a specific strategy instance by name."""
        return self._strategies.get(name)

    def get_all_strategies(self) -> Dict[str, AdaptiveBaseStrategy]:
        """Get all strategy instances."""
        return self._strategies.copy()

    def get_scorecard(self) -> StrategyScorecard:
        """Get the performance scorecard for inspection."""
        return self.scorecard
