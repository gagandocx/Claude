"""
=============================================================
  NeuroX AI Trading EA - A/B Testing Framework

  Allows rigorous comparison of parameter changes by randomly
  assigning trades to variant A (current) or variant B (candidate),
  tracking performance of each, and computing statistical
  significance after N trades per variant.

  Usage:
    framework = ABTestFramework(config)
    variant = framework.assign_variant()
    # ... execute trade with variant params ...
    framework.record_result(variant, won=True, pnl=1.50)
    result = framework.get_test_result()
=============================================================
"""

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ABTestConfig

logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    """Tracks performance metrics for a single variant."""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    pnl_history: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trades if self.trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(p for p in self.pnl_history if p > 0)
        losses = abs(sum(p for p in self.pnl_history if p < 0))
        return gains / losses if losses > 0 else 3.0


class ABTestFramework:
    """
    A/B testing framework for rigorous parameter comparison.

    Randomly assigns each trade to variant A (control/current) or
    variant B (candidate/new parameters). After accumulating enough
    trades per variant, computes statistical significance using a
    z-test on win rates.

    Features:
      - Random 50/50 assignment (unbiased)
      - Tracks win rate, PnL, profit factor per variant
      - Z-test for significance after min_trades_per_variant
      - Logs which variant is winning and by how much
      - Can define parameter overrides per variant
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self._variant_a = VariantResult()
        self._variant_b = VariantResult()
        self._test_start_time = time.time()
        self._parameter_overrides: Dict[str, Dict[str, Any]] = {
            'A': {},  # Control (current parameters)
            'B': {},  # Candidate (new parameters)
        }
        logger.info(
            f"[ABTest] Initialized: test='{self.config.test_name}' "
            f"A='{self.config.variant_a_label}' B='{self.config.variant_b_label}' "
            f"min_trades={self.config.min_trades_per_variant} "
            f"significance={self.config.significance_level}"
        )

    def set_variant_params(self, variant: str, params: Dict[str, Any]):
        """
        Set parameter overrides for a variant.

        Args:
            variant: 'A' or 'B'
            params: Dict of parameter names to values
        """
        if variant in ('A', 'B'):
            self._parameter_overrides[variant] = params
            label = self.config.variant_a_label if variant == 'A' else self.config.variant_b_label
            logger.info(f"[ABTest] Variant {variant} ({label}) params: {params}")

    def assign_variant(self) -> str:
        """
        Randomly assign the next trade to variant A or B.

        Returns:
            'A' or 'B'
        """
        variant = 'A' if random.random() < 0.5 else 'B'
        return variant

    def get_variant_params(self, variant: str) -> Dict[str, Any]:
        """
        Get the parameter overrides for the assigned variant.

        Args:
            variant: 'A' or 'B'

        Returns:
            Dict of parameter overrides (empty dict means use defaults)
        """
        return self._parameter_overrides.get(variant, {})

    def record_result(self, variant: str, won: bool, pnl: float = 0.0):
        """
        Record a trade result for the given variant.

        Args:
            variant: 'A' or 'B'
            won: Whether the trade was profitable
            pnl: Dollar P&L of the trade
        """
        result = self._variant_a if variant == 'A' else self._variant_b
        result.trades += 1
        if won:
            result.wins += 1
        else:
            result.losses += 1
        result.total_pnl += pnl
        result.pnl_history.append(pnl)

        # Log progress every 10 trades
        total_trades = self._variant_a.trades + self._variant_b.trades
        if total_trades % 10 == 0:
            self._log_progress()

    def get_test_result(self) -> Dict[str, Any]:
        """
        Get current A/B test results with statistical analysis.

        Returns:
            Dict with keys:
                test_name: Name of the test
                variant_a: VariantResult metrics
                variant_b: VariantResult metrics
                significant: Whether the result is statistically significant
                p_value: P-value from z-test
                winner: 'A', 'B', or 'inconclusive'
                ready: Whether enough trades have been collected
        """
        a = self._variant_a
        b = self._variant_b
        min_trades = self.config.min_trades_per_variant

        ready = a.trades >= min_trades and b.trades >= min_trades

        # Compute z-test for difference in win rates
        p_value = 1.0
        significant = False
        winner = 'inconclusive'

        if ready:
            p_value = self._z_test_proportions(
                a.wins, a.trades, b.wins, b.trades
            )
            significant = p_value < self.config.significance_level

            if significant:
                winner = 'A' if a.win_rate > b.win_rate else 'B'

        return {
            'test_name': self.config.test_name,
            'variant_a': {
                'label': self.config.variant_a_label,
                'trades': a.trades,
                'win_rate': round(a.win_rate, 4),
                'avg_pnl': round(a.avg_pnl, 4),
                'total_pnl': round(a.total_pnl, 2),
                'profit_factor': round(a.profit_factor, 2),
            },
            'variant_b': {
                'label': self.config.variant_b_label,
                'trades': b.trades,
                'win_rate': round(b.win_rate, 4),
                'avg_pnl': round(b.avg_pnl, 4),
                'total_pnl': round(b.total_pnl, 2),
                'profit_factor': round(b.profit_factor, 2),
            },
            'ready': ready,
            'significant': significant,
            'p_value': round(p_value, 6),
            'winner': winner,
            'elapsed_hours': round((time.time() - self._test_start_time) / 3600, 1),
        }

    @staticmethod
    def _z_test_proportions(wins_a: int, n_a: int, wins_b: int, n_b: int) -> float:
        """
        Two-proportion z-test for comparing win rates.

        Tests H0: p_a = p_b vs H1: p_a != p_b (two-sided).

        Args:
            wins_a: Number of wins in variant A
            n_a: Total trades in variant A
            wins_b: Number of wins in variant B
            n_b: Total trades in variant B

        Returns:
            p-value (two-sided)
        """
        if n_a == 0 or n_b == 0:
            return 1.0

        p_a = wins_a / n_a
        p_b = wins_b / n_b

        # Pooled proportion
        p_pool = (wins_a + wins_b) / (n_a + n_b)
        if p_pool == 0.0 or p_pool == 1.0:
            return 1.0

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        if se == 0:
            return 1.0

        # Z-statistic
        z = abs(p_a - p_b) / se

        # Approximate p-value using normal CDF (two-sided)
        # Using the complementary error function for accuracy
        p_value = 2.0 * (1.0 - _normal_cdf(z))
        return max(0.0, min(1.0, p_value))

    def _log_progress(self):
        """Log current A/B test progress."""
        a = self._variant_a
        b = self._variant_b
        logger.info(
            f"[ABTest] '{self.config.test_name}' progress: "
            f"A({self.config.variant_a_label}): {a.trades}t wr={a.win_rate:.1%} pnl=${a.total_pnl:+.2f} | "
            f"B({self.config.variant_b_label}): {b.trades}t wr={b.win_rate:.1%} pnl=${b.total_pnl:+.2f}"
        )

    def reset(self):
        """Reset all test data for a new test."""
        self._variant_a = VariantResult()
        self._variant_b = VariantResult()
        self._test_start_time = time.time()
        logger.info(f"[ABTest] Test '{self.config.test_name}' reset")


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
