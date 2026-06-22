"""
=============================================================
  Python ML Bridge - Risk Manager
  Portfolio-level risk management with Kelly criterion sizing,
  max drawdown monitoring, correlation limits, and time filters.
=============================================================
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RiskConfig, StreakConfig


class RiskManager:
    """
    Portfolio-level risk management system.

    Features:
        - Kelly criterion position sizing with volatility scaling
        - Maximum drawdown monitoring and halt
        - Daily loss limit enforcement
        - Correlation-based exposure limits
        - Time-of-day trading filters
        - Position size calculation with proper lot sizing
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

        # Streak detection config
        self.streak_config = StreakConfig()

        # State tracking
        self._equity_high = self.config.account_balance
        self._current_equity = self.config.account_balance
        self._daily_pnl = 0.0
        self._daily_start_equity = self.config.account_balance
        self._open_positions: List[Dict] = []
        self._trade_history: deque = deque(maxlen=200)
        self._last_reset_day = datetime.now().date()

        # Streak tracking
        self._consecutive_results: deque = deque(maxlen=50)
        self._streak_multiplier: float = 1.0

    def calculate_position_size(self, confidence: float,
                                atr: float,
                                win_rate: Optional[float] = None,
                                avg_win_loss_ratio: Optional[float] = None,
                                regime_mult: float = 1.0) -> float:
        """
        Calculate position size using Kelly criterion with safety adjustments.

        Args:
            confidence: Model confidence (0-1)
            atr: Current ATR value for the instrument
            win_rate: Historical win rate (optional, defaults to 0.55)
            avg_win_loss_ratio: Average win/loss ratio (optional, defaults to 1.5)
            regime_mult: Regime-based position size multiplier

        Returns:
            Lot size (0.01 to max_lot_size)
        """
        # Check if trading is allowed
        if not self.is_trading_allowed():
            return 0.0

        # If insufficient trade history, return minimum lot size
        # to avoid using assumed edge from default parameters
        if len(self._trade_history) < 20:
            return self.config.min_lot_size

        # Default parameters
        win_rate = win_rate or 0.55
        avg_win_loss_ratio = avg_win_loss_ratio or 1.5

        # Kelly criterion: f* = (bp - q) / b
        # where b = avg_win_loss_ratio, p = win_rate, q = 1 - win_rate
        b = avg_win_loss_ratio
        p = win_rate
        q = 1.0 - p
        kelly = (b * p - q) / b

        # Apply fraction (quarter-Kelly for safety)
        kelly_adjusted = max(0, kelly * self.config.kelly_fraction)

        # Scale by confidence
        risk_fraction = kelly_adjusted * confidence

        # Apply regime multiplier
        risk_fraction *= regime_mult

        # Cap at max risk per trade
        risk_fraction = min(risk_fraction, self.config.max_risk_per_trade)

        # Convert to lot size
        # Risk amount in account currency
        risk_amount = self._current_equity * risk_fraction

        # Position size based on ATR (SL distance)
        if atr <= 0:
            return self.config.min_lot_size

        sl_distance = atr * 1.5  # 1.5 ATR stop loss
        # For gold: 1 lot = 100 oz, so $1 move = $100 per lot
        point_value = 100.0  # Value per lot per $1 move for gold
        lot_size = risk_amount / (sl_distance * point_value)

        # Clamp to allowed range
        lot_size = max(self.config.min_lot_size,
                       min(lot_size, self.config.max_lot_size))

        # Apply streak multiplier
        lot_size *= self._streak_multiplier

        # Re-clamp after streak adjustment
        lot_size = max(self.config.min_lot_size,
                       min(lot_size, self.config.max_lot_size))

        # Round to 2 decimal places
        lot_size = round(lot_size, 2)

        return lot_size

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed based on risk limits.

        Returns:
            True if trading is permitted
        """
        # Reset daily P&L if new day
        today = datetime.now().date()
        if today != self._last_reset_day:
            self._daily_pnl = 0.0
            self._daily_start_equity = self._current_equity
            self._last_reset_day = today

        # Check max drawdown
        if self._equity_high > 0:
            current_drawdown = (
                (self._equity_high - self._current_equity) / self._equity_high
            )
            if current_drawdown >= self.config.max_drawdown:
                return False

        # Check daily loss limit (percentage-based)
        if self._daily_start_equity > 0:
            daily_loss = (
                (self._daily_start_equity - self._current_equity) /
                self._daily_start_equity
            )
            if daily_loss >= self.config.max_daily_loss:
                return False

        # Check daily loss limit (absolute dollar amount)
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= self.config.max_daily_loss_dollars:
            return False

        # Check max open positions
        if len(self._open_positions) >= self.config.max_open_positions:
            return False

        # Check time-of-day filter
        current_hour = datetime.now().hour
        if current_hour in self.config.no_trade_hours:
            return False

        return True

    def check_correlation_limit(self, new_signal_direction: str) -> bool:
        """
        Check if a new position would exceed correlation limits.

        Args:
            new_signal_direction: "BUY" or "SELL"

        Returns:
            True if position is allowed (correlation OK)
        """
        if not self._open_positions:
            return True

        # Count positions in same direction
        same_direction = sum(
            1 for p in self._open_positions
            if p.get("direction") == new_signal_direction
        )

        # If too many in same direction, reject
        max_same_direction = self.config.max_open_positions - 1
        return same_direction < max_same_direction

    def update_equity(self, new_equity: float):
        """Update current equity and track high watermark."""
        self._current_equity = new_equity
        self._equity_high = max(self._equity_high, new_equity)
        self._daily_pnl = new_equity - self._daily_start_equity

    def register_trade(self, trade: Dict):
        """Register a new open position."""
        self._open_positions.append(trade)

    def close_trade(self, trade_id: str, pnl: float):
        """Close a position and record P&L."""
        self._open_positions = [
            p for p in self._open_positions if p.get("id") != trade_id
        ]
        self._trade_history.append({"id": trade_id, "pnl": pnl})
        self.update_equity(self._current_equity + pnl)

        # Register streak result automatically on trade close
        self.register_result(pnl > 0)

    def register_result(self, won: bool):
        """
        Register a trade result for streak tracking.

        Updates the consecutive results deque and recalculates
        the streak multiplier based on current win/loss streak.

        Args:
            won: True if the trade was a winner, False if loser
        """
        self._consecutive_results.append('W' if won else 'L')
        self._update_streak_multiplier()

    def _update_streak_multiplier(self):
        """Recalculate streak multiplier based on consecutive results."""
        if not self._consecutive_results:
            self._streak_multiplier = 1.0
            return

        # Count consecutive results from the end
        last_result = self._consecutive_results[-1]
        streak_count = 0
        for result in reversed(self._consecutive_results):
            if result == last_result:
                streak_count += 1
            else:
                break

        cfg = self.streak_config

        if last_result == 'L':
            # Losing streak
            if streak_count >= cfg.severe_threshold:
                self._streak_multiplier = cfg.severe_reduce_pct
            elif streak_count >= cfg.lose_streak_reduce_threshold:
                self._streak_multiplier = cfg.reduce_pct
            else:
                self._streak_multiplier = 1.0
        else:
            # Winning streak
            if streak_count >= cfg.win_severe_threshold:
                self._streak_multiplier = cfg.win_severe_boost_pct
            elif streak_count >= cfg.win_boost_threshold:
                self._streak_multiplier = cfg.win_boost_pct
            elif streak_count >= cfg.win_restore_threshold:
                self._streak_multiplier = 1.0
            else:
                self._streak_multiplier = 1.0

    def get_streak_status(self) -> Dict:
        """
        Get current streak status for logging and monitoring.

        Returns:
            Dict with streak type, count, and current multiplier.
        """
        if not self._consecutive_results:
            return {
                "streak_type": "none",
                "streak_count": 0,
                "multiplier": 1.0,
            }

        last_result = self._consecutive_results[-1]
        streak_count = 0
        for result in reversed(self._consecutive_results):
            if result == last_result:
                streak_count += 1
            else:
                break

        return {
            "streak_type": "win" if last_result == 'W' else "lose",
            "streak_count": streak_count,
            "multiplier": self._streak_multiplier,
        }

    def get_win_rate(self) -> float:
        """Calculate win rate from recent trade history."""
        if not self._trade_history:
            return 0.55  # Default assumption
        wins = sum(1 for t in self._trade_history if t["pnl"] > 0)
        return wins / len(self._trade_history)

    def get_avg_win_loss_ratio(self) -> float:
        """Calculate average win/loss ratio from history."""
        if not self._trade_history:
            return 1.5  # Default assumption

        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self._trade_history if t["pnl"] < 0]

        if not wins or not losses:
            return 1.5

        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        return avg_win / (avg_loss + 1e-10)

    def get_current_drawdown(self) -> float:
        """Get current drawdown as a percentage."""
        if self._equity_high <= 0:
            return 0.0
        return (self._equity_high - self._current_equity) / self._equity_high

    def get_risk_summary(self) -> Dict:
        """Get a summary of current risk state."""
        return {
            "equity": self._current_equity,
            "equity_high": self._equity_high,
            "drawdown": self.get_current_drawdown(),
            "daily_pnl": self._daily_pnl,
            "open_positions": len(self._open_positions),
            "win_rate": self.get_win_rate(),
            "win_loss_ratio": self.get_avg_win_loss_ratio(),
            "trading_allowed": self.is_trading_allowed(),
        }

    def calculate_sl_tp(self, atr: float, direction: str,
                        current_price: float,
                        sl_mult: float = 1.5,
                        tp_mult: float = 2.5) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels based on ATR.

        Args:
            atr: Current ATR value
            direction: "BUY" or "SELL"
            current_price: Current market price
            sl_mult: ATR multiplier for stop loss
            tp_mult: ATR multiplier for take profit

        Returns:
            Dict with 'sl_price', 'tp_price', 'sl_pips', 'tp_pips'
        """
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        if direction == "BUY":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:  # SELL
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        # Convert to pips (for gold, 1 pip = 0.1)
        pip_value = 0.1
        sl_pips = sl_distance / pip_value
        tp_pips = tp_distance / pip_value

        return {
            "sl_price": round(sl_price, 2),
            "tp_price": round(tp_price, 2),
            "sl_pips": round(sl_pips, 1),
            "tp_pips": round(tp_pips, 1),
        }
