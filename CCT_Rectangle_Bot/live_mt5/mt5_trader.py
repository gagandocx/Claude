"""
CCT Rectangle Bot - Live MT5 Trader (Main Loop) - ZERO-DELAY MODE.

This is the main orchestration module that:
1. Connects to MT5 terminal
2. Monitors tick stream continuously in a tight spin loop (10ms sleep max)
3. Detects exact 1M candle close moments via tick timestamp comparison
4. Pre-computes direction (4H) and weakness (15M) signals between candle closes
5. Pre-stages orders when ARMED for instant fire on trigger
6. Executes trades instantly on candle close (target: under 50ms)
7. Manages open positions (trailing stop logic)
8. Reconciles closed positions to update risk manager stats
9. Displays GIANT PnL via ASCII art terminal display (updated every 100ms)
10. Handles graceful shutdown on SIGINT

Signal flow:
- Continuous spin loop: check ticks every 10ms (NO polling delay)
- Between candle closes: pre-compute direction + weakness (cached)
- When ARMED: pre-stage order with calculated lot size, SL, TP
- On 1M candle close: fire pre-staged order instantly (no computation)
- Target: tick-to-execution under 50ms
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# Add parent directory to path so we can import strategy classes.
# This makes CCT_Rectangle_Bot/ importable (for strategy, config, utils).
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import mt5_config
from logger_setup import setup_logger
from mt5_connector import MT5Connection
from mt5_data_feed import MT5DataFeed
from mt5_tick_monitor import TickMonitor, SignalState
from risk_manager import RiskManager
from dashboard import Dashboard
from pnl_display import GiantPnLDisplay

# Import strategy classes from parent CCT_Rectangle_Bot
from strategy import CCTRectangleStrategy, TradeSetup


class LiveTrader:
    """
    Main live trading orchestrator for CCT Rectangle Bot - Zero-Delay Mode.

    Runs a continuous spin loop (10ms sleep max) that:
    - Checks for new ticks by comparing timestamps (no polling delay)
    - Detects 1M candle close by tick timestamp minute boundary crossing
    - Pre-computes direction and weakness signals between candle closes
    - Pre-stages orders when ARMED (order dict ready to fire)
    - On candle close: fires pre-staged order instantly (under 50ms)
    - Displays GIANT ASCII art PnL that fills the terminal
    - Manages existing positions (trailing stop)
    - Updates display every 100ms for smooth real-time updates

    Performance target: tick-to-execution under 50ms.
    """

    def __init__(self):
        """Initialize LiveTrader components."""
        self.logger = setup_logger("cct_live_trader")
        self.connection = MT5Connection(logger=self.logger)
        self.data_feed = MT5DataFeed(logger=self.logger)
        self.tick_monitor = TickMonitor(data_feed=self.data_feed, logger=self.logger)
        self.risk_manager = RiskManager(logger=self.logger)
        self.dashboard = Dashboard(logger=self.logger)
        self.pnl_display = GiantPnLDisplay()

        # State tracking
        self._running: bool = False
        self._last_signal_time: Optional[datetime] = None
        self._executed_signals: List[str] = []  # Track signal IDs to avoid duplicates
        self._reconciled_deals: set = set()  # Track deals already reconciled

        # Store the original entry risk for each position (ticket -> risk distance)
        self._original_risk: Dict[int, float] = {}

        # Performance tracking
        self._loop_count: int = 0
        self._last_precompute_time: float = 0.0
        self._precompute_interval: float = 30.0  # Re-check pre-computation every 30s
        self._last_display_time: float = 0.0
        self._last_periodic_time: float = 0.0

        # Display interval in seconds (from config: 100ms)
        self._display_interval: float = mt5_config.DISPLAY_UPDATE_MS / 1000.0

        # Spin loop sleep in seconds (from config: 10ms)
        self._loop_sleep: float = mt5_config.TICK_LOOP_SLEEP_MS / 1000.0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def start(self):
        """
        Start the live trading bot in zero-delay mode.

        Performs startup validation, then enters the continuous spin loop.
        """
        self.logger.info("=" * 60)
        self.logger.info("CCT Rectangle Bot - Live Trader Starting (ZERO-DELAY MODE)")
        self.logger.info(f"Symbol: {mt5_config.SYMBOL}")
        self.logger.info(f"Demo Mode: {mt5_config.DEMO_MODE}")
        self.logger.info(f"Tick Loop Sleep: {mt5_config.TICK_LOOP_SLEEP_MS}ms")
        self.logger.info(f"Display Update: {mt5_config.DISPLAY_UPDATE_MS}ms")
        self.logger.info(f"Tick Monitoring: {mt5_config.TICK_MONITORING_ENABLED}")
        self.logger.info(f"Pre-compute Signals: {mt5_config.PRE_COMPUTE_SIGNALS}")
        self.logger.info(f"Pre-stage Orders: {mt5_config.PRE_STAGE_ORDERS}")
        self.logger.info(f"Risk Per Trade: {mt5_config.RISK_PER_TRADE * 100:.1f}%")
        self.logger.info(f"Max Daily Loss: {mt5_config.MAX_DAILY_LOSS_PCT * 100:.1f}%")
        self.logger.info(f"Max Trades/Day: {mt5_config.MAX_TRADES_PER_DAY}")
        self.logger.info(f"Target Execution: <{mt5_config.MAX_EXECUTION_DELAY_MS}ms")
        self.logger.info("=" * 60)

        # Startup validation
        if not self._validate_startup():
            self.logger.error("Startup validation failed. Exiting.")
            return

        self._running = True
        self.logger.info("Startup validation passed. Entering zero-delay spin loop.")

        # Initialize tick monitor
        self.tick_monitor.start()

        # Main zero-delay spin loop
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

        # Check 6: Verify tick data is available
        self.logger.info("Testing tick data...")
        tick = self.data_feed.get_latest_tick()
        if tick is None:
            self.logger.warning("Tick data unavailable - will retry in main loop")
        else:
            self.logger.info(f"Tick OK: Bid={tick['bid']}, Ask={tick['ask']}")

        return True

    def _main_loop(self):
        """
        Zero-delay continuous spin loop. Sleeps only 10ms between iterations.

        Each spin cycle (~10ms):
        1. Check for new tick (via timestamp comparison)
        2. Check for 1M candle close (minute boundary detection)
        3. If candle close: fire pre-staged order or run fast signal check
        4. If ARMED: monitor tick for rectangle breakout, keep order pre-staged
        5. Every 100ms: update giant PnL display
        6. Every 30s: pre-compute direction/weakness signals
        7. Every 10s: manage positions, reconcile, periodic tasks

        NO time.sleep(1) or time.sleep(60) anywhere in this path.
        """
        self.logger.info(
            f"Zero-delay spin loop started. "
            f"Sleep: {mt5_config.TICK_LOOP_SLEEP_MS}ms, "
            f"Display: {mt5_config.DISPLAY_UPDATE_MS}ms, "
            f"Target: <{mt5_config.MAX_EXECUTION_DELAY_MS}ms execution."
        )

        while self._running:
            try:
                cycle_start = time.time()
                self._loop_count += 1

                # Step 1: Check connection (every 1000 cycles ~ every 10s)
                if self._loop_count % 1000 == 0:
                    if not self.connection.is_connected():
                        self.logger.warning("Connection lost. Attempting reconnect...")
                        if not self.connection.reconnect():
                            self.logger.error("Reconnection failed. Stopping.")
                            break
                        self.tick_monitor.start()

                # Step 2: Check for new 1M candle close (tick timestamp)
                new_candle = self.tick_monitor.check_candle_close()

                if new_candle:
                    # HOT PATH: New candle closed - execute immediately
                    self._on_candle_close()

                # Step 3: Between candles - pre-compute and monitor
                now = time.time()

                # Pre-compute signals every 30 seconds
                if now - self._last_precompute_time > self._precompute_interval:
                    self._between_candles()
                    self._last_precompute_time = now

                # Step 4: When ARMED, monitor rectangle zone on every tick
                if self.tick_monitor.is_armed:
                    self._monitor_rectangle_zone()

                # Step 5: Update giant PnL display every DISPLAY_UPDATE_MS
                if now - self._last_display_time >= self._display_interval:
                    self._update_giant_pnl_display()
                    self._last_display_time = now

                # Step 6: Periodic tasks every 10 seconds
                if now - self._last_periodic_time >= 10.0:
                    self._periodic_tasks()
                    self._last_periodic_time = now

                # Spin loop sleep: 10ms max (configurable via TICK_LOOP_SLEEP_MS)
                time.sleep(self._loop_sleep)

            except Exception as e:
                self.logger.error(f"Error in spin loop: {e}", exc_info=True)
                # Brief 100ms pause on error, then continue (not 1 second!)
                time.sleep(0.1)

        # Cleanup on exit
        self._shutdown()

    def _on_candle_close(self):
        """
        HOT PATH: Called immediately when a new 1M candle close is detected.

        This is the performance-critical path. If a pre-staged order exists,
        fires it INSTANTLY (no computation needed). Otherwise falls back to
        running the strategy check with pre-computed signals.

        Target: complete execution in under 50ms from candle close detection.
        """
        execution_start = time.time()

        # FASTEST PATH: Fire pre-staged order if available
        prestaged = self.tick_monitor.get_prestaged_order()
        if prestaged is not None and prestaged.is_valid:
            self._fire_prestaged_order(prestaged, execution_start)
            return

        # FAST PATH: Use pre-computed 4H and 15M data, only fetch 1M
        cached = self.tick_monitor.get_cached_dataframes()

        if cached is not None:
            df_1m = self.data_feed.get_1m_data()
            if df_1m is None:
                self.logger.warning("Failed to fetch 1M data on candle close")
                return

            data = {
                "df_4h": cached["df_4h"],
                "df_15m": cached["df_15m"],
                "df_1m": df_1m,
            }
        else:
            # Fallback: fetch all timeframes (slower but reliable)
            data = self.data_feed.get_all_timeframes()
            if data is None:
                self.logger.warning("Failed to fetch data on candle close")
                return

        # Run strategy
        signals = self._run_strategy(data)

        # Execute signals immediately (no delay)
        if signals:
            self._execute_signals_instant(signals)
        else:
            state = self.tick_monitor.signal_state
            if state == SignalState.ARMED:
                self.dashboard.set_signal_status("ARMED - No entry (waiting)")
            else:
                self.dashboard.set_signal_status(f"Scanning ({state.value})")

        # Log execution time for performance monitoring
        execution_time_ms = (time.time() - execution_start) * 1000
        self.logger.info(
            f"Candle close processed in {execution_time_ms:.1f}ms "
            f"(target: <{mt5_config.MAX_EXECUTION_DELAY_MS}ms)"
        )

        if execution_time_ms > mt5_config.MAX_EXECUTION_DELAY_MS:
            self.logger.warning(
                f"Execution time {execution_time_ms:.1f}ms exceeded target "
                f"{mt5_config.MAX_EXECUTION_DELAY_MS}ms"
            )

    def _fire_prestaged_order(self, prestaged, execution_start: float):
        """
        Fire a pre-staged order with zero computation delay.

        The order parameters were already calculated when the system
        became ARMED. This just sends the order to MT5.

        Args:
            prestaged: PreStagedOrder object with order parameters.
            execution_start: Time when execution started (for latency tracking).
        """
        self.logger.info(
            f"FIRING PRE-STAGED ORDER: {prestaged.direction.upper()} "
            f"{prestaged.volume} lots"
        )

        # Send order immediately
        result = self.connection.send_order(
            direction=prestaged.direction,
            volume=prestaged.volume,
            stop_loss=prestaged.stop_loss,
            take_profit=prestaged.take_profit,
            comment=prestaged.comment,
        )

        total_execution_ms = (time.time() - execution_start) * 1000

        if result.success:
            # Store original risk for trailing stop
            sl_distance = abs(prestaged.entry_price_reference - prestaged.stop_loss)
            if result.ticket is not None and sl_distance > 0:
                self._original_risk[result.ticket] = sl_distance

            self.dashboard.set_signal_status(
                f"FILLED: {prestaged.direction.upper()} {prestaged.volume} lots, "
                f"#{result.ticket} ({total_execution_ms:.0f}ms)"
            )
            self.logger.info(
                f"Pre-staged order executed in {total_execution_ms:.0f}ms: "
                f"{prestaged.direction.upper()} {prestaged.volume} lots @ {result.price:.5f}, "
                f"Ticket #{result.ticket}"
            )
        else:
            self.logger.error(
                f"Pre-staged order failed after {total_execution_ms:.0f}ms: "
                f"{result.error_message}"
            )
            self.dashboard.set_signal_status(f"Order failed: {result.error_message[:30]}")

        # Invalidate the pre-staged order after attempt
        self.tick_monitor.invalidate_prestaged_order()

    def _between_candles(self):
        """
        COLD PATH: Called periodically (~every 30s) when no new candle has closed.

        Uses idle time to pre-compute signals and pre-stage orders
        so the hot path is as fast as possible.
        """
        # Pre-compute direction and weakness signals
        self.tick_monitor.precompute_signals()

        # Update dashboard with signal state
        state = self.tick_monitor.signal_state
        if state == SignalState.ARMED:
            precomputed = self.tick_monitor.get_precomputed_signals()
            weakness = precomputed.weakness_signal
            if weakness is not None:
                self.dashboard.set_signal_status(
                    f"ARMED [{weakness.rectangle_bottom:.2f}-"
                    f"{weakness.rectangle_top:.2f}]"
                )
            else:
                self.dashboard.set_signal_status("ARMED - Ready for entry")

            # Pre-stage order when ARMED (so hot path just fires it)
            if mt5_config.PRE_STAGE_ORDERS:
                self._prestage_order()
        elif state == SignalState.SCANNING:
            self.dashboard.set_signal_status("Scanning 15M weakness...")
        else:
            self.dashboard.set_signal_status("Idle - Waiting for 4H direction")

    def _prestage_order(self):
        """
        Pre-stage an order when ARMED so it can fire instantly on trigger.

        Calculates lot size, SL, TP based on current account state and
        pre-computed weakness signal. The order dict is stored in the
        tick monitor, ready to fire with zero computation on candle close.
        """
        # Skip if already have a valid pre-staged order
        existing = self.tick_monitor.get_prestaged_order()
        if existing is not None:
            return

        precomputed = self.tick_monitor.get_precomputed_signals()
        if precomputed.weakness_signal is None or precomputed.direction_signal is None:
            return

        # Get account info for lot size calculation
        account_info = self.connection.get_account_info()
        if account_info is None:
            return

        # Check if we can trade
        positions = self.connection.get_open_positions()
        can_trade, reason = self.risk_manager.can_trade(
            current_equity=account_info["equity"],
            open_positions_count=len(positions),
        )
        if not can_trade:
            return

        # Get symbol info for lot size calculation
        symbol_info = self.connection.get_symbol_info()
        if symbol_info is None:
            return

        # Determine direction and entry parameters from pre-computed signals
        direction_signal = precomputed.direction_signal
        weakness_signal = precomputed.weakness_signal

        direction = "buy" if direction_signal.direction == "bullish" else "sell"

        # Use rectangle boundaries for SL/TP estimation
        rect_top = weakness_signal.rectangle_top
        rect_bottom = weakness_signal.rectangle_bottom
        rect_height = rect_top - rect_bottom

        if direction == "buy":
            entry_ref = rect_top  # Breakout above
            sl = rect_bottom  # SL below rectangle
            tp = rect_top + (rect_height * mt5_config.TARGET_RR_RATIO)
        else:
            entry_ref = rect_bottom  # Breakout below
            sl = rect_top  # SL above rectangle
            tp = rect_bottom - (rect_height * mt5_config.TARGET_RR_RATIO)

        sl_distance = abs(entry_ref - sl)
        if sl_distance <= 0:
            return

        # Calculate lot size
        lot_size = self.risk_manager.calculate_lot_size(
            account_equity=account_info["equity"],
            stop_loss_distance=sl_distance,
            symbol_point=symbol_info.get("point", 0.01),
            symbol_trade_tick_value=symbol_info.get("trade_tick_value", 1.0),
        )

        # Stage the order
        self.tick_monitor.stage_order(
            direction=direction,
            volume=lot_size,
            stop_loss=sl,
            take_profit=tp,
            comment=f"CCT_{direction_signal.direction[:4]}_prestaged",
            entry_price_reference=entry_ref,
        )

    def _monitor_rectangle_zone(self):
        """
        Monitor tick price relative to rectangle zone when ARMED.

        When price is near the rectangle boundary, log the proximity.
        This provides real-time awareness of potential breakout conditions
        before the candle close confirms it.
        """
        precomputed = self.tick_monitor.get_precomputed_signals()
        if precomputed.weakness_signal is None:
            return

        rect_top = precomputed.weakness_signal.rectangle_top
        rect_bottom = precomputed.weakness_signal.rectangle_bottom

        position = self.tick_monitor.is_price_in_rectangle_zone(rect_top, rect_bottom)

        if position == "above":
            self.logger.debug(
                f"Price ABOVE rectangle top {rect_top:.5f} - potential bullish breakout"
            )
        elif position == "below":
            self.logger.debug(
                f"Price BELOW rectangle bottom {rect_bottom:.5f} - potential bearish breakout"
            )

    def _periodic_tasks(self):
        """
        Tasks that run periodically (every ~10 seconds), not on every tick.

        Includes position management, reconciliation, and logging.
        Note: Dashboard display is handled separately by _update_giant_pnl_display().
        """
        # Manage existing positions (trailing stop)
        data = self.tick_monitor.get_cached_dataframes()
        if data:
            self._manage_positions(data)

        # Reconcile closed positions
        self._reconcile_closed_positions()

    def _update_giant_pnl_display(self):
        """
        Update the giant ASCII art PnL display in the terminal.

        Called every DISPLAY_UPDATE_MS (100ms) for smooth real-time updates.
        The giant PnL number is the primary visual - fills the terminal.
        """
        try:
            # Get account info for display
            account_info = self.connection.get_account_info()
            if account_info is None:
                return

            # Get positions for PnL
            positions = self.connection.get_open_positions()
            daily_stats = self.risk_manager.get_daily_stats()

            # Get current spread
            tick = self.data_feed.get_latest_tick()
            spread = 0
            if tick:
                spread = int((tick["ask"] - tick["bid"]) / 0.01)  # Convert to points

            # Get signal state
            signal_state = self.tick_monitor.signal_state.value.upper()

            # Floating PnL (unrealized)
            float_pnl = account_info.get("profit", 0.0)

            # Render the giant display
            self.pnl_display.render(
                pnl=float_pnl,
                balance=account_info.get("balance", 0.0),
                equity=account_info.get("equity", 0.0),
                spread=spread,
                signal_state=signal_state,
                positions_count=len(positions) if positions else 0,
                daily_pnl=daily_stats.get("pnl", 0.0) if daily_stats else 0.0,
                daily_trades=daily_stats.get("trade_count", 0) if daily_stats else 0,
            )
        except Exception as e:
            # Display errors should never crash the trading loop
            self.logger.debug(f"Display update error: {e}")

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

    def _execute_signals_instant(self, signals: List[TradeSetup]):
        """
        Execute trade signals with zero delay (instant execution).

        This is the performance-critical execution path. No sleep or delay
        between signal detection and order placement.

        Only acts on signals whose entry_time is within the last few bars
        of the entry timeframe (1M) to avoid executing stale historical setups.

        Args:
            signals: List of TradeSetup objects from the strategy.
        """
        if not signals:
            return

        execution_start = time.time()

        # Filter to recent signals only: entry_time must be within the last
        # 2 minutes (2 bars of the 1M entry timeframe) for real-time mode.
        now_utc = datetime.now(timezone.utc)
        max_signal_age_seconds = 2 * 60  # 2 minutes (tighter than polling mode)

        recent_signals = []
        for sig in signals:
            try:
                sig_time = sig.entry_time
                if isinstance(sig_time, str):
                    sig_time = datetime.fromisoformat(sig_time)
                if sig_time.tzinfo is None:
                    sig_time = sig_time.replace(tzinfo=timezone.utc)
                age = (now_utc - sig_time).total_seconds()
                if age <= max_signal_age_seconds:
                    recent_signals.append(sig)
            except (ValueError, TypeError, AttributeError):
                continue

        if not recent_signals:
            self.logger.debug("No recent signals (all older than 2 minutes)")
            self.dashboard.set_signal_status("ARMED - signals too old")
            return

        latest_signal = recent_signals[-1]

        # Create a unique signal ID to avoid duplicate execution
        signal_id = (
            f"{latest_signal.entry_time}_{latest_signal.direction}_"
            f"{latest_signal.entry_price:.5f}"
        )

        if signal_id in self._executed_signals:
            self.logger.debug(f"Signal already executed: {signal_id}")
            return

        self.logger.info(
            f"INSTANT EXECUTION: {latest_signal.direction.upper()} "
            f"@ {latest_signal.entry_price:.5f}, "
            f"SL={latest_signal.stop_loss:.5f}, TP={latest_signal.take_profit:.5f}, "
            f"RR={latest_signal.rr_ratio:.1f}"
        )

        self.dashboard.set_signal_status(
            f"EXECUTING: {latest_signal.direction.upper()} "
            f"@ {latest_signal.entry_price:.5f} (RR: {latest_signal.rr_ratio:.1f})"
        )

        # Check with risk manager (fast check - no network calls)
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

        # Calculate lot size based on risk, using broker tick value
        symbol_info = self.connection.get_symbol_info()
        if symbol_info is None:
            self.logger.warning("Cannot get symbol info for lot size calc")
            return

        sl_distance = abs(latest_signal.entry_price - latest_signal.stop_loss)

        lot_size = self.risk_manager.calculate_lot_size(
            account_equity=account_info["equity"],
            stop_loss_distance=sl_distance,
            symbol_point=symbol_info.get("point", 0.01),
            symbol_trade_tick_value=symbol_info.get("trade_tick_value", 1.0),
        )

        # INSTANT ORDER: Send immediately with no delay
        direction = "buy" if latest_signal.direction == "bullish" else "sell"
        result = self.connection.send_order(
            direction=direction,
            volume=lot_size,
            stop_loss=latest_signal.stop_loss,
            take_profit=latest_signal.take_profit,
            comment=f"CCT_{latest_signal.direction[:4]}_{latest_signal.rr_ratio:.1f}R",
        )

        # Log total execution time
        total_execution_ms = (time.time() - execution_start) * 1000

        if result.success:
            self._executed_signals.append(signal_id)
            # Keep only last 100 signal IDs to prevent memory growth
            if len(self._executed_signals) > 100:
                self._executed_signals = self._executed_signals[-50:]

            # Store the original risk for trailing stop calculation
            if result.ticket is not None:
                self._original_risk[result.ticket] = sl_distance

            self.dashboard.set_signal_status(
                f"FILLED: {direction.upper()} {lot_size} lots, "
                f"#{result.ticket} ({total_execution_ms:.0f}ms)"
            )
            self.logger.info(
                f"Trade executed in {total_execution_ms:.0f}ms: "
                f"{direction.upper()} {lot_size} lots @ {result.price:.5f}, "
                f"Ticket #{result.ticket}"
            )
        else:
            self.logger.error(
                f"Order failed after {total_execution_ms:.0f}ms: {result.error_message}"
            )
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

        Uses the originally stored risk distance (from entry to initial SL)
        rather than the current SL, which may have been modified by prior
        trailing stop adjustments. This prevents accelerating stop drift.

        Args:
            position: Position dictionary from get_open_positions().
        """
        ticket = position["ticket"]
        pos_type = position["type"]
        open_price = position["open_price"]
        current_sl = position["sl"]
        current_price = position["current_price"]

        # Calculate original risk: prefer stored value, fall back to current SL
        if ticket in self._original_risk:
            risk = self._original_risk[ticket]
        else:
            # Position was opened before this session or risk was not recorded.
            # Fall back to current SL distance but store it so it doesn't drift.
            if current_sl <= 0:
                return  # No SL set, skip
            risk = abs(open_price - current_sl)
            self._original_risk[ticket] = risk

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

    def _reconcile_closed_positions(self):
        """
        Detect positions closed since the last cycle (by SL, TP, or manual close)
        and record them with the risk manager so daily P&L and trade-count
        limits remain accurate.
        """
        # Use a wider lookback window for the 1-second loop
        lookback_seconds = max(mt5_config.POLL_INTERVAL_SECONDS * 15, 30)
        closed_deals = self.connection.get_closed_positions(
            since_seconds=lookback_seconds
        )
        if not closed_deals:
            return

        for deal in closed_deals:
            deal_ticket = deal.get("deal_ticket")
            if deal_ticket in self._reconciled_deals:
                continue  # Already processed

            pnl = deal.get("profit", 0.0) + deal.get("swap", 0.0) + deal.get("commission", 0.0)
            is_win = pnl > 0

            self.risk_manager.record_trade(pnl=pnl, is_win=is_win)
            self._reconciled_deals.add(deal_ticket)

            # Clean up the original risk tracker for this position
            pos_ticket = deal.get("ticket")
            self._original_risk.pop(pos_ticket, None)

            self.logger.info(
                f"Closed position reconciled: Ticket #{pos_ticket}, "
                f"P&L={pnl:+.2f}, Win={is_win}"
            )

        # Prevent unbounded growth of reconciled set
        if len(self._reconciled_deals) > 500:
            # Keep only the most recent 250 entries
            self._reconciled_deals = set(list(self._reconciled_deals)[-250:])

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        self.logger.info(f"Shutdown signal received (signal {signum})")
        self._running = False

    def _shutdown(self):
        """Clean up resources on shutdown."""
        self.logger.info("Shutting down Live Trader...")
        self.pnl_display.cleanup()
        self.connection.disconnect()
        self.logger.info("Live Trader stopped.")
        print("\n\nCCT Rectangle Bot stopped. Goodbye.")


def main():
    """Entry point for the live trader."""
    print("=" * 60)
    print("  CCT Rectangle Bot - Live MT5 Trader (ZERO-DELAY)")
    print("=" * 60)
    print()
    print(f"  Mode: Zero-delay continuous spin loop")
    print(f"  Tick loop: {mt5_config.TICK_LOOP_SLEEP_MS}ms sleep")
    print(f"  Display update: {mt5_config.DISPLAY_UPDATE_MS}ms")
    print(f"  Execution target: <{mt5_config.MAX_EXECUTION_DELAY_MS}ms")
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
