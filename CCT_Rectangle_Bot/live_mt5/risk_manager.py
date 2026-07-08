"""
Risk Manager for CCT Rectangle Bot - Live MT5 Trading.

Implements safety controls:
- Maximum daily loss limit
- Maximum trades per day
- Maximum drawdown from equity peak
- Trading hours filter (session-based)
- Position sizing based on account equity and risk percentage
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional

import mt5_config


class RiskManager:
    """
    Manages risk controls for live trading.

    Tracks daily P&L, trade count, and drawdown. All safety checks
    must pass before a new trade is allowed.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize RiskManager with safety limits from config.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("risk_manager")

        # Safety limits from config
        self.max_daily_loss_pct = mt5_config.MAX_DAILY_LOSS_PCT
        self.max_trades_per_day = mt5_config.MAX_TRADES_PER_DAY
        self.max_drawdown_pct = mt5_config.MAX_DRAWDOWN_PCT
        self.max_concurrent_trades = mt5_config.MAX_CONCURRENT_TRADES

        # Daily tracking
        self._current_date: Optional[date] = None
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._daily_wins: int = 0
        self._daily_losses: int = 0

        # Equity tracking for drawdown
        self._equity_peak: float = 0.0
        self._starting_balance: float = 0.0

        # State
        self._is_halted: bool = False
        self._halt_reason: str = ""

    def initialize(self, account_balance: float, account_equity: float):
        """
        Initialize risk manager with current account state.

        Args:
            account_balance: Current account balance.
            account_equity: Current account equity.
        """
        self._starting_balance = account_balance
        self._equity_peak = max(account_equity, account_balance)
        self._current_date = datetime.now(timezone.utc).date()
        self.logger.info(
            f"Risk Manager initialized: Balance={account_balance:.2f}, "
            f"Equity Peak={self._equity_peak:.2f}"
        )

    def can_trade(self, current_equity: float, open_positions_count: int = 0) -> tuple:
        """
        Check if a new trade is allowed based on all safety limits.

        Args:
            current_equity: Current account equity.
            open_positions_count: Number of currently open positions.

        Returns:
            Tuple of (allowed: bool, reason: str).
            If allowed is False, reason explains why.
        """
        # Check for date rollover
        self._check_date_rollover()

        # Update equity peak
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity

        # Check if manually halted
        if self._is_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # Check 1: Maximum daily loss
        if self._starting_balance > 0:
            daily_loss_pct = abs(min(self._daily_pnl, 0)) / self._starting_balance
            if daily_loss_pct >= self.max_daily_loss_pct:
                self._halt_trading(
                    f"Daily loss limit reached: {daily_loss_pct*100:.2f}% "
                    f"(max: {self.max_daily_loss_pct*100:.1f}%)"
                )
                return False, self._halt_reason

        # Check 2: Maximum trades per day
        if self._daily_trade_count >= self.max_trades_per_day:
            return False, (
                f"Max trades per day reached: {self._daily_trade_count} "
                f"(max: {self.max_trades_per_day})"
            )

        # Check 3: Maximum drawdown from peak
        if self._equity_peak > 0:
            drawdown_pct = (self._equity_peak - current_equity) / self._equity_peak
            if drawdown_pct >= self.max_drawdown_pct:
                self._halt_trading(
                    f"Max drawdown reached: {drawdown_pct*100:.2f}% "
                    f"(max: {self.max_drawdown_pct*100:.1f}%)"
                )
                return False, self._halt_reason

        # Check 4: Trading hours
        if not self.is_within_trading_hours():
            return False, "Outside trading hours"

        # Check 5: Concurrent positions
        if open_positions_count >= self.max_concurrent_trades:
            return False, (
                f"Max concurrent trades reached: {open_positions_count} "
                f"(max: {self.max_concurrent_trades})"
            )

        return True, "All checks passed"

    def is_within_trading_hours(self) -> bool:
        """
        Check if current UTC time is within configured trading sessions.

        Returns:
            True if within allowed trading hours.
        """
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour

        # Check main sessions
        for session_name, hours in mt5_config.TRADING_SESSIONS.items():
            start = hours["start"]
            end = hours["end"]
            if start <= current_hour < end:
                return True

        # Check Asia session if allowed
        if mt5_config.ALLOW_ASIA_SESSION:
            asia = mt5_config.ASIA_SESSION
            if asia["start"] <= current_hour < asia["end"]:
                return True

        return False

    def calculate_lot_size(
        self,
        account_equity: float,
        stop_loss_distance: float,
        symbol_point: float = 0.01,
        symbol_trade_tick_value: float = 1.0,
    ) -> float:
        """
        Calculate position size based on risk percentage and stop loss distance.

        Args:
            account_equity: Current account equity.
            stop_loss_distance: Distance from entry to stop loss in price units.
            symbol_point: Symbol's point value (e.g., 0.01 for XAUUSD).
            symbol_trade_tick_value: Value per tick movement per lot.

        Returns:
            Calculated lot size, capped to configured LOT_SIZE.
        """
        if stop_loss_distance <= 0 or account_equity <= 0:
            return mt5_config.LOT_SIZE

        # Risk amount in account currency
        risk_amount = account_equity * mt5_config.RISK_PER_TRADE

        # Calculate stop loss in points
        sl_points = stop_loss_distance / symbol_point

        # Value per point per lot
        if symbol_trade_tick_value > 0 and sl_points > 0:
            calculated_lots = risk_amount / (sl_points * symbol_trade_tick_value)
        else:
            calculated_lots = mt5_config.LOT_SIZE

        # Round to 2 decimal places (standard lot step)
        calculated_lots = round(calculated_lots, 2)

        # Apply minimum and maximum constraints
        calculated_lots = max(0.01, calculated_lots)  # Minimum micro lot
        calculated_lots = min(calculated_lots, mt5_config.LOT_SIZE * 10)  # Safety cap

        self.logger.debug(
            f"Lot size calculation: equity={account_equity:.2f}, "
            f"risk={risk_amount:.2f}, SL dist={stop_loss_distance:.5f}, "
            f"lots={calculated_lots:.2f}"
        )

        return calculated_lots

    def record_trade(self, pnl: float, is_win: bool):
        """
        Record a completed trade for daily tracking.

        Args:
            pnl: Profit/loss amount for the trade.
            is_win: Whether the trade was profitable.
        """
        self._check_date_rollover()
        self._daily_pnl += pnl
        self._daily_trade_count += 1

        if is_win:
            self._daily_wins += 1
        else:
            self._daily_losses += 1

        self.logger.info(
            f"Trade recorded: P&L={pnl:+.2f}, "
            f"Daily total: {self._daily_pnl:+.2f}, "
            f"Trades today: {self._daily_trade_count} "
            f"(W:{self._daily_wins}/L:{self._daily_losses})"
        )

    def reset_daily_stats(self):
        """Reset daily statistics. Called at midnight UTC or on new day."""
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._daily_wins = 0
        self._daily_losses = 0
        self._is_halted = False
        self._halt_reason = ""
        self._current_date = datetime.now(timezone.utc).date()
        self.logger.info("Daily statistics reset")

    def get_daily_stats(self) -> dict:
        """
        Get current daily trading statistics.

        Returns:
            Dictionary with daily stats.
        """
        self._check_date_rollover()
        return {
            "date": str(self._current_date),
            "pnl": self._daily_pnl,
            "trade_count": self._daily_trade_count,
            "wins": self._daily_wins,
            "losses": self._daily_losses,
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_loss_remaining": (
                self._starting_balance * self.max_daily_loss_pct - abs(min(self._daily_pnl, 0))
                if self._starting_balance > 0 else 0
            ),
            "trades_remaining": self.max_trades_per_day - self._daily_trade_count,
        }

    def _check_date_rollover(self):
        """Check if date has changed and reset stats if needed."""
        today = datetime.now(timezone.utc).date()
        if self._current_date is not None and today != self._current_date:
            self.logger.info(f"New trading day: {today}")
            self.reset_daily_stats()

    def _halt_trading(self, reason: str):
        """Halt trading for the rest of the day."""
        self._is_halted = True
        self._halt_reason = reason
        self.logger.warning(f"TRADING HALTED: {reason}")
