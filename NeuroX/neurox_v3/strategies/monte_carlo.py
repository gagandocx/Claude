"""
=============================================================
  Python ML Bridge v8.0 - Monte Carlo Risk Simulation
  Tier 3: Institutional-Grade Feature

  Before each trade, simulates N scenarios given current drawdown,
  win/loss streak, and regime. Gates trades that have >X% probability
  of hitting the daily loss limit.

  This is how institutional risk desks operate: they simulate
  thousands of potential outcomes before allowing a position.
=============================================================
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MonteCarloConfig, BrainConfig

logger = logging.getLogger(__name__)


class MonteCarloRiskSimulator:
    """
    Monte Carlo simulation for trade-level risk gating.

    Before allowing a trade, simulates N future scenarios
    (default 1000) to estimate:
      - Probability of hitting the daily loss limit
      - Expected drawdown from this point forward
      - 95th percentile worst-case drawdown
      - Whether it is safe to continue trading

    Trade is gated (skipped) if ruin probability exceeds the
    configured threshold (default 5%).

    The simulation uses:
      - Current drawdown state
      - Win rate and average win/loss ratio
      - Current losing streak length
      - Market regime (volatile regimes reduce expected win rate)
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None,
                 brain_config: Optional[BrainConfig] = None):
        self.config = config or MonteCarloConfig()
        self.brain_config = brain_config or BrainConfig()

        # Cache last simulation results
        self._last_result: Optional[Dict] = None
        self._last_sim_time: float = 0.0
        self._sim_cache_ttl: float = 30.0  # Cache for 30 seconds

        # Track state for should_skip_trade
        self._current_drawdown: float = 0.0
        self._current_win_rate: float = 0.55
        self._current_avg_win: float = 2.0
        self._current_avg_loss: float = 1.0
        self._current_losing_streak: int = 0
        self._current_regime: str = 'neutral'

        logger.info("[MonteCarloRisk] Initialized: sims=%d, max_loss_prob=%.2f, "
                    "dd_threshold=%.2f, confidence=%.2f",
                    self.config.num_simulations,
                    self.config.max_daily_loss_prob,
                    self.config.drawdown_threshold,
                    self.config.confidence_level)

    def simulate_scenarios(self, current_drawdown: float, win_rate: float,
                           avg_win: float, avg_loss: float,
                           losing_streak: int, regime: str,
                           num_sims: int = None) -> Dict:
        """
        Run Monte Carlo simulation of future trade outcomes.

        Simulates num_sims paths of N future trades (default 20 trades
        forward) and computes statistics about potential drawdown.

        Args:
            current_drawdown: Current drawdown as fraction of account (e.g. 0.02 = 2%)
            win_rate: Current win rate (0.0 to 1.0)
            avg_win: Average winning trade in dollars
            avg_loss: Average losing trade in dollars (positive value)
            losing_streak: Current consecutive losing trades
            regime: Market regime name (affects simulation parameters)
            num_sims: Number of simulations (default from config)

        Returns:
            Dict with keys:
                ruin_probability: Probability of hitting daily loss limit
                expected_drawdown: Median expected drawdown
                max_drawdown_95pct: 95th percentile worst-case drawdown
                safe_to_trade: True if ruin_probability < threshold
                simulations_run: Number of simulations performed
        """
        if num_sims is None:
            num_sims = self.config.num_simulations

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1.0:
            win_rate = 0.5
        if avg_win <= 0:
            avg_win = 1.0
        if avg_loss <= 0:
            avg_loss = 1.0

        # Regime adjustments: reduce expected win rate in adverse regimes
        regime_wr_adjustment = self._get_regime_adjustment(regime)
        adjusted_win_rate = win_rate * regime_wr_adjustment

        # Streak penalty: consecutive losses reduce effective win rate
        # (psychological/mechanical correlation of losses)
        streak_penalty = min(losing_streak * 0.02, 0.10)  # Max 10% penalty
        adjusted_win_rate = max(0.30, adjusted_win_rate - streak_penalty)

        # Daily loss limit from brain config
        daily_loss_limit = self.brain_config.daily_loss_limit
        account_balance = self.brain_config.account_balance

        # Convert daily loss limit to fraction if it is in dollars
        if daily_loss_limit > 1.0:
            # It is in dollars, convert to fraction
            loss_limit_fraction = daily_loss_limit / account_balance
        else:
            loss_limit_fraction = daily_loss_limit

        # Number of future trades to simulate per path
        trades_forward = 20

        # Run simulations
        ruin_count = 0
        max_drawdowns = np.zeros(num_sims)

        # Vectorized simulation for performance
        # Generate all random outcomes at once
        random_outcomes = np.random.random((num_sims, trades_forward))
        wins_mask = random_outcomes < adjusted_win_rate

        # Add noise to win/loss amounts (realistic variance)
        win_noise = 1.0 + np.random.normal(0, 0.2, (num_sims, trades_forward))
        loss_noise = 1.0 + np.random.normal(0, 0.15, (num_sims, trades_forward))
        win_noise = np.clip(win_noise, 0.5, 2.0)
        loss_noise = np.clip(loss_noise, 0.5, 1.8)

        # Compute PnL for each simulation
        pnl_matrix = np.where(
            wins_mask,
            avg_win * win_noise,
            -avg_loss * loss_noise
        )

        # Cumulative PnL from current state
        cumulative_pnl = np.cumsum(pnl_matrix, axis=1)

        # Current drawdown in dollar terms
        dd_dollars = current_drawdown * account_balance

        # Total equity curve (starting from current drawdown)
        equity_from_here = cumulative_pnl - dd_dollars

        # Track minimum equity (worst point) in each simulation
        min_equity = np.min(equity_from_here, axis=1)

        # Ruin = hitting the daily loss limit
        ruin_threshold = -daily_loss_limit  # In dollar terms relative to start of day
        ruin_count = int(np.sum(min_equity < ruin_threshold))

        # Maximum drawdown for each sim (in fraction terms)
        # Peak equity at each step
        peak_equity = np.maximum.accumulate(equity_from_here, axis=1)
        drawdown_matrix = (peak_equity - equity_from_here) / (account_balance + 1e-6)
        max_drawdowns = np.max(drawdown_matrix, axis=1)

        # Compute statistics
        ruin_probability = ruin_count / num_sims
        expected_drawdown = float(np.median(max_drawdowns))
        confidence_pct = self.config.confidence_level
        max_dd_95 = float(np.percentile(max_drawdowns, confidence_pct * 100))
        safe_to_trade = ruin_probability < self.config.max_daily_loss_prob

        result = {
            "ruin_probability": ruin_probability,
            "expected_drawdown": expected_drawdown,
            "max_drawdown_95pct": max_dd_95,
            "safe_to_trade": safe_to_trade,
            "simulations_run": num_sims,
            "adjusted_win_rate": adjusted_win_rate,
            "regime": regime,
            "losing_streak": losing_streak,
        }

        # Cache result
        self._last_result = result
        self._last_sim_time = _current_time()

        # Update internal state
        self._current_drawdown = current_drawdown
        self._current_win_rate = win_rate
        self._current_avg_win = avg_win
        self._current_avg_loss = avg_loss
        self._current_losing_streak = losing_streak
        self._current_regime = regime

        logger.info("[MonteCarloRisk] Simulation: ruin_prob=%.4f, "
                    "expected_dd=%.4f, max_dd_95=%.4f, safe=%s "
                    "(wr=%.3f, streak=%d, regime=%s)",
                    ruin_probability, expected_drawdown, max_dd_95,
                    safe_to_trade, adjusted_win_rate, losing_streak, regime)

        return result

    def should_skip_trade(self) -> Tuple[bool, str]:
        """
        Determine if the current trade should be skipped based on
        the latest Monte Carlo simulation results.

        Uses cached results if available and recent. Otherwise
        runs a new simulation with current state.

        Returns:
            Tuple of (skip: bool, reason: str)
            - skip=True means the trade should NOT be taken
            - reason explains why
        """
        import time as _time

        # Use cached result if fresh
        now = _time.time()
        if (self._last_result is not None and
                now - self._last_sim_time < self._sim_cache_ttl):
            result = self._last_result
        else:
            # Run simulation with current state
            result = self.simulate_scenarios(
                current_drawdown=self._current_drawdown,
                win_rate=self._current_win_rate,
                avg_win=self._current_avg_win,
                avg_loss=self._current_avg_loss,
                losing_streak=self._current_losing_streak,
                regime=self._current_regime,
            )

        if not result["safe_to_trade"]:
            reason = (
                f"Monte Carlo: {result['ruin_probability']:.1%} probability of "
                f"hitting daily loss limit (threshold: "
                f"{self.config.max_daily_loss_prob:.1%}). "
                f"Expected DD: {result['expected_drawdown']:.2%}, "
                f"95%% worst: {result['max_drawdown_95pct']:.2%}"
            )
            return True, reason

        return False, ""

    def update_state(self, drawdown: float = None, win_rate: float = None,
                     avg_win: float = None, avg_loss: float = None,
                     losing_streak: int = None, regime: str = None,
                     account_balance: float = None):
        """
        Update the simulator's internal state for the next should_skip_trade call.

        Args:
            drawdown: Current drawdown fraction
            win_rate: Current win rate
            avg_win: Average win in dollars
            avg_loss: Average loss in dollars
            losing_streak: Current consecutive losses
            regime: Current market regime
            account_balance: Live account balance/equity (overrides config default)
        """
        if drawdown is not None:
            self._current_drawdown = drawdown
        if win_rate is not None:
            self._current_win_rate = win_rate
        if avg_win is not None:
            self._current_avg_win = avg_win
        if avg_loss is not None:
            self._current_avg_loss = avg_loss
        if losing_streak is not None:
            self._current_losing_streak = losing_streak
        if regime is not None:
            self._current_regime = regime
        if account_balance is not None:
            self.brain_config.account_balance = account_balance

        # Invalidate cache when state changes
        self._last_result = None

    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get win rate adjustment factor for the current regime.

        Volatile and crash regimes reduce expected win rate.
        Trending regimes slightly increase it.

        Args:
            regime: Market regime name

        Returns:
            Multiplier for win rate (0.7 to 1.1)
        """
        adjustments = {
            'strong_trend_up': 1.10,
            'strong_trend_down': 1.10,
            'trending_up': 1.05,
            'trending_down': 1.05,
            'trending': 1.05,
            'neutral': 1.00,
            'ranging': 0.90,
            'volatile': 0.80,
            'crash': 0.70,
        }
        return adjustments.get(regime, 1.00)


def _current_time() -> float:
    """Get current time (separated for testability)."""
    import time as _time
    return _time.time()
