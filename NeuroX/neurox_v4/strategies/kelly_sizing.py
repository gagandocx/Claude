"""
=============================================================
  Python ML Bridge v8.0 - Kelly Criterion Position Sizing
  Tier 3: Institutional-Grade Feature

  Implements the Kelly criterion for mathematically optimal
  position sizing. Uses fractional Kelly (default 25%) for
  safety, providing geometric growth with controlled variance.

  Full Kelly formula: f* = (p * b - q) / b
    where p = win rate, q = loss rate (1-p), b = win/loss ratio

  Fractional Kelly multiplies by a safety factor (0.25) to
  reduce the risk of ruin while maintaining positive expectancy.
=============================================================
"""

import logging
from typing import Optional

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import KellyConfig, RiskConfig

logger = logging.getLogger(__name__)


class KellySizer:
    """
    Kelly criterion position sizing for optimal geometric growth.

    The Kelly criterion computes the fraction of capital to risk
    that maximizes long-term geometric growth. For trading:

    f* = (p * b - q) / b

    Where:
        p = probability of winning (win rate)
        q = probability of losing (1 - p)
        b = ratio of average win to average loss

    Full Kelly is aggressive; we use fractional Kelly (25%) for:
        - Reduced variance in returns
        - Protection against estimation errors in win rate
        - Smoother equity curve
        - Less psychological stress

    The class validates inputs before computing Kelly, requiring:
        - Minimum number of trades (default 20) for statistical significance
        - Win rate above 50% (positive edge required)
        - Positive average win and loss values
    """

    def __init__(self, config: Optional[KellyConfig] = None,
                 risk_config: Optional[RiskConfig] = None):
        self.config = config or KellyConfig()
        self.risk_config = risk_config or RiskConfig()

        # Track trade history for validation
        self._trade_count: int = 0
        self._wins: int = 0
        self._total_win_amount: float = 0.0
        self._total_loss_amount: float = 0.0

        logger.info("[KellySizer] Initialized: fraction=%.2f, min_win_rate=%.2f, "
                    "min_trades=%d, max_lot=%.2f",
                    self.config.kelly_fraction,
                    self.config.min_win_rate,
                    self.config.min_trades,
                    self.config.max_kelly_lot)

    def compute_kelly_fraction(self, win_rate: float, avg_win: float,
                                avg_loss: float) -> float:
        """
        Compute the full Kelly fraction for given trade statistics.

        Formula: f* = (p * b - q) / b
        Where: p = win_rate, q = 1 - p, b = avg_win / avg_loss

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            avg_win: Average winning trade amount (positive)
            avg_loss: Average losing trade amount (positive, absolute value)

        Returns:
            Full Kelly fraction (can be > 1.0 for extreme edges).
            Returns 0.0 if edge is negative or inputs are invalid.
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1.0:
            return 0.0
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly formula
        kelly = (p * b - q) / b

        # Negative Kelly means negative edge - don't trade
        if kelly <= 0:
            return 0.0

        return float(kelly)

    def get_optimal_lot(self, account_balance: float, risk_per_trade: float,
                        win_rate: float, avg_win: float, avg_loss: float,
                        current_atr: float) -> float:
        """
        Compute optimal lot size using fractional Kelly criterion.

        Steps:
        1. Compute full Kelly fraction
        2. Apply fractional Kelly safety (default 0.25x)
        3. Convert to dollar risk amount
        4. Convert to lot size based on ATR and risk
        5. Clamp to min/max lot limits

        Args:
            account_balance: Current account balance in dollars
            risk_per_trade: Maximum risk per trade as fraction (e.g. 0.02 = 2%)
            win_rate: Win rate (0.0 to 1.0)
            avg_win: Average win amount in dollars (positive)
            avg_loss: Average loss amount in dollars (positive, absolute)
            current_atr: Current ATR in price units (for lot calculation)

        Returns:
            Optimal lot size clamped to config limits.
        """
        if account_balance <= 0 or current_atr <= 0:
            return self.risk_config.min_lot_size

        # Step 1: Compute full Kelly
        full_kelly = self.compute_kelly_fraction(win_rate, avg_win, avg_loss)

        if full_kelly <= 0:
            logger.debug("[KellySizer] No edge (Kelly <= 0): win_rate=%.3f, "
                         "avg_win=%.2f, avg_loss=%.2f", win_rate, avg_win, avg_loss)
            return self.risk_config.min_lot_size

        # Step 2: Apply fractional Kelly
        if self.config.full_kelly:
            fraction = full_kelly
        else:
            fraction = full_kelly * self.config.kelly_fraction

        # Cap fraction at risk_per_trade maximum
        fraction = min(fraction, risk_per_trade)

        # Step 3: Dollar amount to risk
        risk_dollars = account_balance * fraction

        # Step 4: Convert to lot size
        # For XAUUSD: 1 lot = 100 oz, 1 pip = $1 per 0.01 lot
        # Risk = SL_distance * lot * contract_size
        # SL_distance approximated from ATR
        sl_distance = current_atr * 1.5  # Typical SL = 1.5 ATR
        if sl_distance <= 0:
            sl_distance = 1.0

        # lot = risk_dollars / (sl_distance * contract_size_per_lot)
        # For gold: contract_size = 100, so per-pip value for 0.01 lot = $1
        contract_value_per_lot = 100.0  # XAUUSD standard
        lot_size = risk_dollars / (sl_distance * contract_value_per_lot)

        # Step 5: Clamp to limits
        min_lot = self.risk_config.min_lot_size
        max_lot = min(self.config.max_kelly_lot, self.risk_config.max_lot_size)
        lot_size = float(np.clip(lot_size, min_lot, max_lot))

        # Round to 2 decimal places (standard lot precision)
        lot_size = round(lot_size, 2)

        logger.debug("[KellySizer] Optimal lot: %.2f (full_kelly=%.4f, "
                     "frac=%.4f, risk$=%.2f, balance=%.0f)",
                     lot_size, full_kelly, fraction, risk_dollars, account_balance)

        return lot_size

    def is_kelly_valid(self) -> bool:
        """
        Check if Kelly criterion has enough data to be reliable.

        Requirements:
        1. Minimum number of trades recorded (default 20)
        2. Win rate exceeds minimum threshold (default 50%)

        Returns:
            True if Kelly-based sizing is statistically reliable.
        """
        if self._trade_count < self.config.min_trades:
            logger.debug("[KellySizer] Not valid: %d trades < %d minimum",
                         self._trade_count, self.config.min_trades)
            return False

        win_rate = self._wins / self._trade_count if self._trade_count > 0 else 0.0
        if win_rate < self.config.min_win_rate:
            logger.debug("[KellySizer] Not valid: win_rate=%.3f < %.3f minimum",
                         win_rate, self.config.min_win_rate)
            return False

        return True

    def record_trade(self, pnl: float, won: bool):
        """
        Record a completed trade for Kelly statistics tracking.

        Args:
            pnl: Profit/loss in dollars (positive for win, negative for loss)
            won: Whether the trade was profitable
        """
        self._trade_count += 1
        if won:
            self._wins += 1
            self._total_win_amount += abs(pnl)
        else:
            self._total_loss_amount += abs(pnl)

    def get_current_stats(self) -> dict:
        """Get current Kelly sizing statistics."""
        win_rate = self._wins / self._trade_count if self._trade_count > 0 else 0.0
        losses = self._trade_count - self._wins
        avg_win = self._total_win_amount / self._wins if self._wins > 0 else 0.0
        avg_loss = self._total_loss_amount / losses if losses > 0 else 0.0

        kelly_f = self.compute_kelly_fraction(win_rate, avg_win, avg_loss) if avg_loss > 0 else 0.0

        return {
            "trade_count": self._trade_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "full_kelly": kelly_f,
            "fractional_kelly": kelly_f * self.config.kelly_fraction,
            "is_valid": self.is_kelly_valid(),
        }
