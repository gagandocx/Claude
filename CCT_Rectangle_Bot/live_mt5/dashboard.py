"""
Console Dashboard for CCT Rectangle Bot - Live MT5 Trading.

Displays formatted status information including:
- Account balance/equity/margin
- Open positions with P&L
- Today's trading stats
- Signal status
- Safety status
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class Dashboard:
    """
    Console dashboard that displays live trading status.

    Prints a formatted status table to the terminal every polling cycle,
    showing account info, positions, daily stats, and signal status.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Dashboard.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("dashboard")
        self._signal_status: str = "Initializing..."
        self._last_signal_time: Optional[str] = None
        self._cycle_count: int = 0

    def update(
        self,
        account_info: Optional[Dict[str, Any]] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        daily_stats: Optional[Dict[str, Any]] = None,
        signal_status: Optional[str] = None,
    ):
        """
        Update and display the dashboard.

        Args:
            account_info: Account details (balance, equity, margin, etc.)
            positions: List of open position dictionaries
            daily_stats: Daily statistics from RiskManager
            signal_status: Current signal detection status
        """
        self._cycle_count += 1

        if signal_status:
            self._signal_status = signal_status

        # Clear screen for clean display
        self._clear_screen()

        # Build and print dashboard
        output = []
        output.append(self._header())
        output.append(self._account_section(account_info))
        output.append(self._positions_section(positions))
        output.append(self._daily_stats_section(daily_stats))
        output.append(self._signal_section())
        output.append(self._safety_section(daily_stats))
        output.append(self._footer())

        print("\n".join(output))

    def set_signal_status(self, status: str):
        """Update the signal status display."""
        self._signal_status = status
        self._last_signal_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def _header(self) -> str:
        """Generate dashboard header."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "+==============================================================+",
            "|        CCT RECTANGLE BOT - LIVE MT5 TRADER                   |",
            "+==============================================================+",
            f"| Last Update: {now:<45}|",
            f"| Cycle: #{self._cycle_count:<51}|",
            "+--------------------------------------------------------------+",
        ]
        return "\n".join(lines)

    def _account_section(self, account_info: Optional[Dict[str, Any]]) -> str:
        """Generate account information section."""
        lines = ["| ACCOUNT INFO                                                 |"]
        lines.append("+--------------------------------------------------------------+")

        if account_info is None:
            lines.append("| (No account data available)                                  |")
        else:
            balance = account_info.get("balance", 0)
            equity = account_info.get("equity", 0)
            margin = account_info.get("margin", 0)
            free_margin = account_info.get("free_margin", 0)
            profit = account_info.get("profit", 0)
            currency = account_info.get("currency", "USD")

            lines.append(f"| Balance:     {balance:>12.2f} {currency:<30}|")
            lines.append(f"| Equity:      {equity:>12.2f} {currency:<30}|")
            lines.append(f"| Margin Used: {margin:>12.2f} {currency:<30}|")
            lines.append(f"| Free Margin: {free_margin:>12.2f} {currency:<30}|")
            lines.append(f"| Float P&L:   {profit:>+12.2f} {currency:<30}|")

        lines.append("+--------------------------------------------------------------+")
        return "\n".join(lines)

    def _positions_section(self, positions: Optional[List[Dict[str, Any]]]) -> str:
        """Generate open positions section."""
        lines = ["| OPEN POSITIONS                                               |"]
        lines.append("+--------------------------------------------------------------+")

        if not positions:
            lines.append("| (No open positions)                                          |")
        else:
            for pos in positions:
                direction = pos.get("type", "?").upper()
                volume = pos.get("volume", 0)
                open_price = pos.get("open_price", 0)
                profit = pos.get("profit", 0)
                sl = pos.get("sl", 0)
                tp = pos.get("tp", 0)
                ticket = pos.get("ticket", 0)

                line1 = f"| #{ticket} {direction} {volume} lots @ {open_price:.5f}"
                line1 = f"{line1:<61}|"
                lines.append(line1)

                line2 = f"|   SL: {sl:.5f}  TP: {tp:.5f}  P&L: {profit:+.2f}"
                line2 = f"{line2:<61}|"
                lines.append(line2)

        lines.append("+--------------------------------------------------------------+")
        return "\n".join(lines)

    def _daily_stats_section(self, daily_stats: Optional[Dict[str, Any]]) -> str:
        """Generate daily statistics section."""
        lines = ["| TODAY'S STATS                                                 |"]
        lines.append("+--------------------------------------------------------------+")

        if daily_stats is None:
            lines.append("| (No stats available)                                         |")
        else:
            pnl = daily_stats.get("pnl", 0)
            trades = daily_stats.get("trade_count", 0)
            wins = daily_stats.get("wins", 0)
            losses = daily_stats.get("losses", 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0

            lines.append(f"| Trades Today:  {trades:<48}|")
            lines.append(f"| Wins/Losses:   {wins}W / {losses}L ({win_rate:.0f}% win rate){' ' * max(0, 34 - len(f'{wins}W / {losses}L ({win_rate:.0f}% win rate)'))}|")
            lines.append(f"| Daily P&L:     {pnl:+.2f}{' ' * max(0, 47 - len(f'{pnl:+.2f}'))}|")

        lines.append("+--------------------------------------------------------------+")
        return "\n".join(lines)

    def _signal_section(self) -> str:
        """Generate signal status section."""
        lines = ["| SIGNAL STATUS                                                |"]
        lines.append("+--------------------------------------------------------------+")

        status_line = f"| Status: {self._signal_status}"
        status_line = f"{status_line:<61}|"
        lines.append(status_line)

        if self._last_signal_time:
            time_line = f"| Last check: {self._last_signal_time}"
            time_line = f"{time_line:<61}|"
            lines.append(time_line)

        lines.append("+--------------------------------------------------------------+")
        return "\n".join(lines)

    def _safety_section(self, daily_stats: Optional[Dict[str, Any]]) -> str:
        """Generate safety status section."""
        lines = ["| SAFETY STATUS                                                |"]
        lines.append("+--------------------------------------------------------------+")

        if daily_stats is None:
            lines.append("| (Initializing...)                                            |")
        else:
            is_halted = daily_stats.get("is_halted", False)
            halt_reason = daily_stats.get("halt_reason", "")
            loss_remaining = daily_stats.get("daily_loss_remaining", 0)
            trades_remaining = daily_stats.get("trades_remaining", 0)

            if is_halted:
                halt_line = f"| *** HALTED: {halt_reason}"
                halt_line = f"{halt_line:<61}|"
                lines.append(halt_line)
            else:
                lines.append(f"| Status: ACTIVE{' ' * 47}|")

            lines.append(f"| Daily loss budget remaining: {loss_remaining:+.2f}{' ' * max(0, 31 - len(f'{loss_remaining:+.2f}'))}|")
            lines.append(f"| Trades remaining today:      {trades_remaining}{' ' * max(0, 31 - len(str(trades_remaining)))}|")

        lines.append("+--------------------------------------------------------------+")
        return "\n".join(lines)

    def _footer(self) -> str:
        """Generate dashboard footer."""
        lines = [
            "| Press Ctrl+C to stop the bot gracefully                      |",
            "+==============================================================+",
        ]
        return "\n".join(lines)
