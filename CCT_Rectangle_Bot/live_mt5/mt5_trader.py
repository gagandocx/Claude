"""
CCT Rectangle Bot - Live MT5 Trader (Main Loop).

This is the main orchestration module that:
1. Connects to MT5 terminal
2. Fetches live candle data every 60 seconds
3. Runs the CCT Rectangle Strategy
4. Executes trades via MT5 when signals are found
5. Manages open positions (trailing stop logic)
6. Displays status via console dashboard
7. Handles graceful shutdown on SIGINT
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# Add parent directory to path so we can import strategy classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mt5_config
from logger_setup import setup_logger
from mt5_connector import MT5Connection
from mt5_data_feed import MT5DataFeed
from risk_manager import RiskManager
from dashboard import Dashboard

# Import strategy classes from parent CCT_Rectangle_Bot
try:
    from strategy import CCTRectangleStrategy, TradeSetup
except ImportError:
    # Fallback: try relative import
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from strategy import CCTRectangleStrategy, TradeSetup


class LiveTrader:
    """
    Main live trading orchestrator for CCT Rectangle Bot.

    Runs a 60-second polling loop that:
    - Fetches fresh OHLCV data from MT5
    - Runs the CCT Rectangle Strategy
    - Executes new signals (if risk manager approves)
    - Manages existing positions (trailing stop)
    - Updates the console dashboard
    """

    def __init__(self):
        """Initialize LiveTrader components."""
        self.logger = setup_logger("cct_live_trader")
        self.connection = MT5Connection(logger=self.logger)
        self.data_feed = MT5DataFeed(logger=self.logger)
        self.risk_manager = RiskManager(logger=self.logger)
        self.dashboard = Dashboard(logger=self.logger)

        # State tracking
        self._running: bool = False
        self._last_signal_time: Optional[datetime] = None
        self._executed_signals: List[str] = []  # Track signal IDs to avoid duplicates

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def start(self):
        """
        Start the live trading bot.

        Performs startup validation, then enters the main polling loop.
        """
        self.logger.info("=" * 60)
        self.logger.info("CCT Rectangle Bot - Live Trader Starting")
        self.logger.info(f"Symbol: {mt5_config.SYMBOL}")
        self.logger.info(f"Demo Mode: {mt5_config.DEMO_MODE}")
        self.logger.info(f"Risk Per Trade: {mt5_config.RISK_PER_TRADE * 100:.1f}%")
        self.logger.info(f"Max Daily Loss: {mt5_config.MAX_DAILY_LOSS_PCT * 100:.1f}%")
        self.logger.info(f"Max Trades/Day: {mt5_config.MAX_TRADES_PER_DAY}")
        self.logger.info("=" * 60)

        # Startup validation
        if not self._validate_startup():
            self.logger.error("Startup validation failed. Exiting.")
            return

        self._running = True
        self.logger.info("Startup validation passed. Entering main loop.")

        # Main polling loop
        self._main_loop()

    def _validate_startup(self) -> bool:
        """
        Perform startup checks before entering the main loop.

        Returns:
            True if all checks pass.
        """
        # Check 1: Connect to MT5
        self.logger.info("Connecting to MT5...")
        if not self.connection.connect():
            self.logger.error("Failed to connect to MT5 terminal")
            return False

        # Check 2: Verify symbol is available
        self.logger.info(f"Checking symbol {mt5_config.SYMBOL}...")
        symbol_info = self.connection.get_symbol_info()
        if symbol_info is None:
            self.logger.error(f"Symbol {mt5_config.SYMBOL} not available")
            self.connection.disconnect()
            return False
        self.logger.info(f"Symbol OK: {symbol_info['name']}, Spread: {symbol_info['spread']}")

        # Check 3: Verify account
        account_info = self.connection.get_account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            self.connection.disconnect()
            return False

        # Check 4: Demo mode safety check
        if mt5_config.DEMO_MODE:
            self.logger.info("DEMO MODE - Using demo account for testing")

        # Initialize risk manager with account state
        self.risk_manager.initialize(
            account_balance=account_info["balance"],
            account_equity=account_info["equity"],
        )

        # Check 5: Verify data feed works
        self.logger.info("Testing data feed...")
        test_data = self.data_feed.get_all_timeframes()
        if test_data is None:
            self.logger.error("Failed to fetch initial data")
            self.connection.disconnect()
            return False
        self.logger.info("Data feed OK")

        return True

    def _main_loop(self):
        """
        Main trading loop. Runs every POLL_INTERVAL_SECONDS.

        Each cycle:
        1. Check connection
        2. Fetch fresh data
        3. Run strategy
        4. Execute new signals
        5. Manage positions
        6. Update dashboard
        """
        self.logger.info(
            f"Main loop started. Polling every {mt5_config.POLL_INTERVAL_SECONDS} seconds."
        )

        while self._running:
            try:
                cycle_start = time.time()

                # Step 1: Check connection
                if not self.connection.is_connected():
                    self.logger.warning("Connection lost. Attempting reconnect...")
                    self.dashboard.set_signal_status("Reconnecting...")
                    if not self.connection.reconnect():
                        self.logger.error("Reconnection failed. Stopping.")
                        break

                # Step 2: Fetch fresh data
                self.dashboard.set_signal_status("Fetching data...")
                data = self.data_feed.get_all_timeframes()
                if data is None:
                    self.logger.warning("Failed to fetch data this cycle. Skipping.")
                    self._wait_for_next_cycle(cycle_start)
                    continue

                # Step 3: Run strategy
                self.dashboard.set_signal_status("Analyzing signals...")
                signals = self._run_strategy(data)

                # Step 4: Execute new signals
                if signals:
                    self._process_signals(signals)
                else:
                    self.dashboard.set_signal_status("Scanning (no signal)")

                # Step 5: Manage existing positions (trailing stop)
                self._manage_positions(data)

                # Step 6: Update dashboard
                self._update_dashboard()

                # Wait for next cycle
                self._wait_for_next_cycle(cycle_start)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                self.dashboard.set_signal_status(f"Error: {str(e)[:30]}")
                time.sleep(5)  # Brief pause before retrying

        # Cleanup on exit
        self._shutdown()

    def _run_strategy(self, data: Dict[str, Any]) -> List[TradeSetup]:
        """
        Run the CCT Rectangle Strategy on fresh data.

        Args:
            data: Dictionary with df_4h, df_15m, df_1m DataFrames.

        Returns:
            List of TradeSetup signals.
        """
        try:
            strategy = CCTRectangleStrategy(
                df_4h=data["df_4h"],
                df_15m=data["df_15m"],
                df_1m=data["df_1m"],
            )
            signals = strategy.generate_signals()
            return signals
        except Exception as e:
            self.logger.error(f"Strategy error: {e}", exc_info=True)
            return []

    def _process_signals(self, signals: List[TradeSetup]):
        """
        Process new trade signals. Execute if risk manager approves.

        Args:
            signals: List of TradeSetup objects from the strategy.
        """
        # Get most recent signal (closest to current time)
        if not signals:
            return

        latest_signal = signals[-1]

        # Create a unique signal ID to avoid duplicate execution
        signal_id = (
            f"{latest_signal.entry_time}_{latest_signal.direction}_"
            f"{latest_signal.entry_price:.5f}"
        )

        if signal_id in self._executed_signals:
            self.logger.debug(f"Signal already executed: {signal_id}")
            return

        self.logger.info(
            f"New signal detected: {latest_signal.direction.upper()} "
            f"@ {latest_signal.entry_price:.5f}, "
            f"SL={latest_signal.stop_loss:.5f}, TP={latest_signal.take_profit:.5f}, "
            f"RR={latest_signal.rr_ratio:.1f}"
        )

        self.dashboard.set_signal_status(
            f"Signal: {latest_signal.direction.upper()} "
            f"@ {latest_signal.entry_price:.5f} (RR: {latest_signal.rr_ratio:.1f})"
        )

        # Check with risk manager
        account_info = self.connection.get_account_info()
        if account_info is None:
            self.logger.warning("Cannot get account info for risk check")
            return

        positions = self.connection.get_open_positions()
        can_trade, reason = self.risk_manager.can_trade(
            current_equity=account_info["equity"],
            open_positions_count=len(positions),
        )

        if not can_trade:
            self.logger.info(f"Trade not allowed: {reason}")
            self.dashboard.set_signal_status(f"Blocked: {reason}")
            return

        # Calculate lot size based on risk
        symbol_info = self.connection.get_symbol_info()
        if symbol_info is None:
            self.logger.warning("Cannot get symbol info for lot size calc")
            return

        sl_distance = abs(latest_signal.entry_price - latest_signal.stop_loss)

        lot_size = self.risk_manager.calculate_lot_size(
            account_equity=account_info["equity"],
            stop_loss_distance=sl_distance,
            symbol_point=symbol_info.get("point", 0.01),
        )

        # Execute the trade
        direction = "buy" if latest_signal.direction == "bullish" else "sell"
        result = self.connection.send_order(
            direction=direction,
            volume=lot_size,
            stop_loss=latest_signal.stop_loss,
            take_profit=latest_signal.take_profit,
            comment=f"CCT_{latest_signal.direction[:4]}_{latest_signal.rr_ratio:.1f}R",
        )

        if result.success:
            self._executed_signals.append(signal_id)
            # Keep only last 100 signal IDs to prevent memory growth
            if len(self._executed_signals) > 100:
                self._executed_signals = self._executed_signals[-50:]

            self.dashboard.set_signal_status(
                f"EXECUTED: {direction.upper()} {lot_size} lots, "
                f"Ticket #{result.ticket}"
            )
            self.logger.info(
                f"Trade executed: {direction.upper()} {lot_size} lots @ {result.price:.5f}, "
                f"Ticket #{result.ticket}"
            )
        else:
            self.logger.error(f"Order failed: {result.error_message}")
            self.dashboard.set_signal_status(f"Order failed: {result.error_message[:30]}")

    def _manage_positions(self, data: Dict[str, Any]):
        """
        Manage existing positions - implement trailing stop logic.

        When price moves in favor by TRAILING_STOP_ACTIVATION_RR times risk,
        move stop loss to trail at TRAILING_STOP_DISTANCE_RR times risk.

        Args:
            data: Current market data (for reference).
        """
        if not mt5_config.USE_TRAILING_STOP:
            return

        positions = self.connection.get_open_positions()
        if not positions:
            return

        for pos in positions:
            try:
                self._apply_trailing_stop(pos)
            except Exception as e:
                self.logger.error(
                    f"Error managing position {pos.get('ticket')}: {e}",
                    exc_info=True,
                )

    def _apply_trailing_stop(self, position: Dict[str, Any]):
        """
        Apply trailing stop logic to a single position.

        Args:
            position: Position dictionary from get_open_positions().
        """
        ticket = position["ticket"]
        pos_type = position["type"]
        open_price = position["open_price"]
        current_sl = position["sl"]
        current_price = position["current_price"]

        # Calculate risk (distance from entry to original SL)
        if current_sl <= 0:
            return  # No SL set, skip

        risk = abs(open_price - current_sl)
        if risk <= 0:
            return

        # Calculate current profit in R multiples
        if pos_type == "buy":
            current_profit_r = (current_price - open_price) / risk
            activation_price = open_price + (risk * mt5_config.TRAILING_STOP_ACTIVATION_RR)
            trail_distance = risk * mt5_config.TRAILING_STOP_DISTANCE_RR

            # Check if trailing stop should activate
            if current_price >= activation_price:
                new_sl = current_price - trail_distance
                # Only move SL up, never down
                if new_sl > current_sl:
                    self.logger.info(
                        f"Trailing stop: Ticket #{ticket}, "
                        f"Moving SL from {current_sl:.5f} to {new_sl:.5f} "
                        f"(profit: {current_profit_r:.1f}R)"
                    )
                    self.connection.modify_position(ticket, stop_loss=new_sl)

        elif pos_type == "sell":
            current_profit_r = (open_price - current_price) / risk
            activation_price = open_price - (risk * mt5_config.TRAILING_STOP_ACTIVATION_RR)
            trail_distance = risk * mt5_config.TRAILING_STOP_DISTANCE_RR

            # Check if trailing stop should activate
            if current_price <= activation_price:
                new_sl = current_price + trail_distance
                # Only move SL down, never up
                if new_sl < current_sl:
                    self.logger.info(
                        f"Trailing stop: Ticket #{ticket}, "
                        f"Moving SL from {current_sl:.5f} to {new_sl:.5f} "
                        f"(profit: {current_profit_r:.1f}R)"
                    )
                    self.connection.modify_position(ticket, stop_loss=new_sl)

    def _update_dashboard(self):
        """Update the console dashboard with current state."""
        account_info = self.connection.get_account_info()
        positions = self.connection.get_open_positions()
        daily_stats = self.risk_manager.get_daily_stats()

        self.dashboard.update(
            account_info=account_info,
            positions=positions,
            daily_stats=daily_stats,
        )

    def _wait_for_next_cycle(self, cycle_start: float):
        """
        Wait until the next polling interval.

        Args:
            cycle_start: Timestamp when the current cycle started.
        """
        elapsed = time.time() - cycle_start
        remaining = mt5_config.POLL_INTERVAL_SECONDS - elapsed

        if remaining > 0:
            time.sleep(remaining)

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        self.logger.info(f"Shutdown signal received (signal {signum})")
        self._running = False

    def _shutdown(self):
        """Clean up resources on shutdown."""
        self.logger.info("Shutting down Live Trader...")
        self.connection.disconnect()
        self.logger.info("Live Trader stopped.")
        print("\n\nCCT Rectangle Bot stopped. Goodbye.")


def main():
    """Entry point for the live trader."""
    print("=" * 60)
    print("  CCT Rectangle Bot - Live MT5 Trader")
    print("=" * 60)
    print()

    # Safety confirmation
    if not mt5_config.DEMO_MODE:
        print("WARNING: DEMO_MODE is OFF. This will trade with REAL money!")
        print()
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            print("Aborted. Set DEMO_MODE = True in mt5_config.py for testing.")
            return

    # Verify we are on Windows (MT5 requirement)
    if sys.platform != "win32":
        print("ERROR: MetaTrader 5 requires Windows.")
        print("This bot can only run on Windows with MT5 terminal installed.")
        print("See README_LIVE_TRADING.md for setup instructions.")
        return

    trader = LiveTrader()
    trader.start()


if __name__ == "__main__":
    main()
