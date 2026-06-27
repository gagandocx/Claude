"""
=============================================================
  Python ML Bridge v3 - Adaptive Confidence Threshold (Tier 2)

  Dynamically adjusts the minimum confidence threshold based on
  recent trade accuracy. Replaces the fixed 0.60 threshold with
  a self-tuning value that:
    - Lowers when accuracy is high (more trades, maximize opportunity)
    - Raises when accuracy is low (fewer trades, protect capital)

  Bounded between min_threshold (0.15) and max_threshold (0.55).
  Change rate limited to prevent whiplash (2% per trade).
=============================================================
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import AdaptiveThresholdConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade for threshold computation."""
    confidence_at_entry: float
    won: bool
    timestamp: float = 0.0


class AdaptiveConfidenceThreshold:
    """
    Self-tuning confidence threshold for trade entry gating.

    Instead of a fixed min_confidence value, this class tracks recent
    trade outcomes and dynamically finds the optimal threshold: the
    confidence level above which the win rate exceeds the target.

    Logic:
        1. Record each trade with its entry confidence and outcome (won/lost)
        2. After each trade, recalculate the optimal threshold
        3. Find the confidence level where win_rate > target (e.g., 55%)
        4. Move current threshold toward optimal at limited rate (2% per trade)
        5. Bound the threshold between min (0.15) and max (0.55)

    The effect:
        - If recent trades at conf=0.30 are winning 60%+ -> lower threshold
          to capture more winning trades at lower confidence
        - If recent trades at conf=0.40 are losing 55%+ -> raise threshold
          to avoid bad entries
        - Rate limiting prevents oscillation

    Usage:
        threshold = AdaptiveConfidenceThreshold()
        threshold.record_trade(0.42, won=True)
        threshold.record_trade(0.35, won=False)
        current = threshold.get_current_threshold()
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None):
        self.config = config or AdaptiveThresholdConfig()

        # Trade history (rolling window)
        self._trade_history: deque = deque(maxlen=self.config.lookback_trades)

        # Current adaptive threshold
        self._current_threshold: float = (
            self.config.min_threshold + self.config.max_threshold
        ) / 2.0  # Start at midpoint (0.35)

        # Statistics
        self._total_trades: int = 0
        self._total_wins: int = 0

        logger.info("[AdaptiveThreshold] Initialized. lookback=%d, "
                    "adjustment_rate=%.3f, bounds=[%.2f, %.2f], "
                    "target_win_rate=%.2f, initial_threshold=%.3f",
                    self.config.lookback_trades,
                    self.config.adjustment_rate,
                    self.config.min_threshold,
                    self.config.max_threshold,
                    self.config.target_win_rate,
                    self._current_threshold)

    def record_trade(self, confidence_at_entry: float, won: bool) -> None:
        """
        Record a completed trade for threshold adaptation.

        After recording, the threshold is recalculated. The adjustment
        is rate-limited to prevent rapid oscillation.

        Args:
            confidence_at_entry: The timing confidence when the trade was entered.
            won: Whether the trade was profitable (True) or not (False).
        """
        import time as _time

        record = TradeRecord(
            confidence_at_entry=confidence_at_entry,
            won=won,
            timestamp=_time.time(),
        )
        self._trade_history.append(record)
        self._total_trades += 1
        if won:
            self._total_wins += 1

        # Recalculate threshold after each trade
        self._recalculate()

        logger.debug("[AdaptiveThreshold] Trade recorded: conf=%.3f, won=%s | "
                     "new_threshold=%.3f (trades=%d)",
                     confidence_at_entry, won, self._current_threshold,
                     len(self._trade_history))

    def get_current_threshold(self) -> float:
        """
        Get the current adaptive confidence threshold.

        Returns:
            Float: the dynamic minimum confidence required for trade entry.
            Bounded by [min_threshold, max_threshold].
        """
        return self._current_threshold

    def _recalculate(self) -> None:
        """
        Recalculate the optimal threshold based on recent trade history.

        Algorithm:
            1. Sort recent trades by confidence level
            2. For each possible threshold (in 0.01 steps from min to max):
               - Count trades above that threshold
               - Compute win rate for those trades
               - Find the lowest threshold where win_rate >= target
            3. Move current threshold toward optimal at limited rate
        """
        trades = list(self._trade_history)
        if len(trades) < 5:
            # Not enough data to adapt - keep current threshold
            return

        target_wr = self.config.target_win_rate
        adjustment_rate = self.config.adjustment_rate

        # Compute overall recent win rate
        recent_wins = sum(1 for t in trades if t.won)
        recent_wr = recent_wins / len(trades)

        # Find optimal threshold: lowest confidence where win_rate >= target
        # Test thresholds from low to high in 0.01 steps
        optimal_threshold = self.config.max_threshold  # Default to max if no good threshold found
        found_optimal = False

        step = 0.01
        test_thresholds = np.arange(
            self.config.min_threshold,
            self.config.max_threshold + step,
            step
        )

        for test_thresh in test_thresholds:
            # Trades that would have been taken at this threshold
            trades_above = [t for t in trades if t.confidence_at_entry >= test_thresh]
            if len(trades_above) < 3:
                continue  # Need minimum sample size

            wr_above = sum(1 for t in trades_above if t.won) / len(trades_above)

            if wr_above >= target_wr:
                optimal_threshold = test_thresh
                found_optimal = True
                break  # Found the lowest threshold meeting target

        # If no threshold meets target, use a heuristic:
        # If overall win rate is very low, raise threshold
        # If overall win rate is high, lower threshold
        if not found_optimal:
            if recent_wr < 0.45:
                # Accuracy is low - raise threshold (be more selective)
                optimal_threshold = min(
                    self._current_threshold + 0.05,
                    self.config.max_threshold
                )
            elif recent_wr > 0.60:
                # Accuracy is high - lower threshold (trade more)
                optimal_threshold = max(
                    self._current_threshold - 0.03,
                    self.config.min_threshold
                )
            else:
                # Moderate accuracy - keep current
                optimal_threshold = self._current_threshold

        # Rate-limited movement toward optimal
        diff = optimal_threshold - self._current_threshold

        if abs(diff) <= adjustment_rate:
            # Small enough to apply directly
            self._current_threshold = optimal_threshold
        else:
            # Limit movement to adjustment_rate per trade
            direction = 1.0 if diff > 0 else -1.0
            self._current_threshold += direction * adjustment_rate

        # Enforce bounds
        self._current_threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, self._current_threshold)
        )

    def get_stats(self) -> Dict:
        """
        Get statistics about the adaptive threshold state.

        Returns:
            Dict with threshold info, trade counts, win rates.
        """
        trades = list(self._trade_history)
        recent_wins = sum(1 for t in trades if t.won) if trades else 0
        recent_wr = recent_wins / len(trades) if trades else 0.0

        return {
            'current_threshold': self._current_threshold,
            'min_threshold': self.config.min_threshold,
            'max_threshold': self.config.max_threshold,
            'target_win_rate': self.config.target_win_rate,
            'recent_trades': len(trades),
            'recent_wins': recent_wins,
            'recent_win_rate': recent_wr,
            'total_trades': self._total_trades,
            'total_wins': self._total_wins,
            'overall_win_rate': self._total_wins / max(self._total_trades, 1),
        }

    def reset(self) -> None:
        """Reset the threshold to initial state (for testing or restart)."""
        self._trade_history.clear()
        self._current_threshold = (
            self.config.min_threshold + self.config.max_threshold
        ) / 2.0
        self._total_trades = 0
        self._total_wins = 0
        logger.info("[AdaptiveThreshold] Reset to initial state (threshold=%.3f)",
                    self._current_threshold)
