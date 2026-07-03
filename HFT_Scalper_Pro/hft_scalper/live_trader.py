"""
Live Trading Bot - Zero-Discrepancy from Backtest
==================================================
This script uses the EXACT same functions (compute_rsi, compute_atr),
signal generation logic, and position sizing from run_aggressive_backtest.py.

Strategy: Two-Mode Adaptive Compounding with Dual RSI (8/14)
- GROW mode (near peak equity): risk_grow=0.17
- PROTECT mode (in drawdown): risk_protect=0.025
- Transition: exponential scaling via (equity/peak)^dd_power

Requirements:
- Windows OS with MetaTrader 5 terminal installed and running
- pip install MetaTrader5 numpy
- MT5 terminal logged into a broker account

Usage:
    python live_trader.py --symbol XAUUSD --magic 202401
    python live_trader.py --config my_params.json
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import csv
import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not found.")
    print("Install with: pip install MetaTrader5")
    print("Note: MetaTrader5 only works on Windows with MT5 terminal installed.")
    sys.exit(1)


# =============================================================================
# EXACT COPIES FROM run_aggressive_backtest.py - DO NOT MODIFY
# =============================================================================

def compute_rsi(close, period):
    """RSI with Wilder smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return rsi


def compute_atr(high, low, close, period):
    """Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr[:period - 1] = atr[period - 1]
    return atr


# =============================================================================
# CONSTANTS (matching backtest)
# =============================================================================

SLIPPAGE = 0.15
COMMISSION_PER_LOT = 7.0
CONTRACT_SIZE = 100.0
INITIAL_EQUITY = 1000.0
WARMUP_BARS = 50

# Best parameters from aggressive_results.json
DEFAULT_PARAMS = {
    "rsi_entry": 25,
    "sl_mult": 2.0,
    "tp_mult": 3.0,
    "risk_grow": 0.17,
    "risk_protect": 0.025,
    "dd_power": 13,
    "at_high_thresh": 0.01,
    "loss_boost": 2.0,
    "win_reduce": 0.4,
    "cooldown": 3,
    "max_positions": 2,
    "use_4bar": True,
    "session_start": 7,
    "session_end": 20,
    "dd_halt": 0.149,
    "streak_n": 3,
    "streak_mult": 1.3,
    "max_risk_cap": 0.25,
}


# =============================================================================
# LIVE TRADING ENGINE
# =============================================================================

class LiveTrader:
    """
    Live trading engine that replicates the backtest logic exactly.

    Uses the SAME signal generation and position sizing as run_aggressive_backtest.py.
    """

    def __init__(self, symbol: str, magic: int, params: Dict, log_dir: str = None):
        self.symbol = symbol
        self.magic = magic
        self.params = params
        self.running = False

        # State tracking (mirrors backtest state)
        self.equity = 0.0
        self.peak_equity = 0.0
        self.consec_wins = 0
        self.consec_losses = 0
        self.last_entry_bars = [-params["cooldown"] - 1] * params["max_positions"]
        self.bar_count = 0
        self.open_positions_count = 0

        # Deal history tracking for consecutive wins
        self._last_deal_time = 0  # timestamp of last processed deal
        self._known_position_tickets = set()  # tickets of positions we are tracking

        # Reconnection settings
        self.max_retries = 10
        self.base_retry_delay = 5  # seconds
        self.max_retry_delay = 300  # 5 minutes

        # Logging setup
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent
        self._setup_logging()
        self._setup_csv_log()

        # Symbol info cache
        self.symbol_info = None
        self.min_lot = 0.01
        self.lot_step = 0.01
        self.max_lot = 200.0
        self.fill_type = None  # determined at connect time

        # State persistence file
        self.state_file = self.log_dir / "live_trader_state.json"

    def _setup_logging(self):
        """Configure Python logging with file and console handlers."""
        self.logger = logging.getLogger("LiveTrader")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.log_dir / "live_trader.log"
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        self.logger.addHandler(ch)

    def _setup_csv_log(self):
        """Initialize CSV trade log."""
        self.csv_path = self.log_dir / "live_trades.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "event", "direction", "lot_size",
                    "entry_price", "sl", "tp", "magic", "signal_type",
                    "rsi_fast", "rsi_slow", "atr", "equity", "peak_equity",
                    "dd_scale", "risk_pct", "bar_count", "profit", "ticket"
                ])

    def _save_state(self):
        """Persist critical state to JSON file for restart recovery."""
        state = {
            "peak_equity": self.peak_equity,
            "consec_wins": self.consec_wins,
            "consec_losses": self.consec_losses,
            "bar_count": self.bar_count,
            "last_entry_bars": self.last_entry_bars,
            "last_deal_time": self._last_deal_time,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            tmp_path = self.state_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.replace(self.state_file)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load persisted state from JSON file if it exists."""
        if not self.state_file.exists():
            self.logger.info("No state file found, starting fresh")
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.peak_equity = state.get("peak_equity", self.equity)
            self.consec_wins = state.get("consec_wins", 0)
            self.consec_losses = state.get("consec_losses", 0)
            self.bar_count = state.get("bar_count", 0)
            saved_entry_bars = state.get("last_entry_bars", None)
            if saved_entry_bars and len(saved_entry_bars) == self.params["max_positions"]:
                self.last_entry_bars = saved_entry_bars
            self._last_deal_time = state.get("last_deal_time", 0)
            self.logger.info(
                f"Restored state: peak_equity={self.peak_equity:.2f}, "
                f"consec_wins={self.consec_wins}, consec_losses={self.consec_losses}, "
                f"bar_count={self.bar_count}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load state file: {e}. Starting fresh.")

    def _log_trade_csv(self, direction: str, lot: float, entry_price: float,
                       sl: float, tp: float, signal_type: str,
                       rsi_fast: float, rsi_slow: float, atr_val: float,
                       dd_scale: float, risk_pct: float):
        """Append a trade entry to the CSV log."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                self.symbol,
                "ENTRY",
                direction,
                lot,
                entry_price,
                sl,
                tp,
                self.magic,
                signal_type,
                round(rsi_fast, 2),
                round(rsi_slow, 2),
                round(atr_val, 4),
                round(self.equity, 2),
                round(self.peak_equity, 2),
                round(dd_scale, 6),
                round(risk_pct, 6),
                self.bar_count,
                "",
                "",
            ])

    def _log_exit_csv(self, direction: str, lot: float, exit_price: float,
                      profit: float, ticket: int):
        """Append a trade exit to the CSV log."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                self.symbol,
                "EXIT",
                direction,
                lot,
                exit_price,
                "",
                "",
                self.magic,
                "",
                "",
                "",
                "",
                round(self.equity, 2),
                round(self.peak_equity, 2),
                "",
                "",
                self.bar_count,
                round(profit, 2),
                ticket,
            ])

    def connect(self) -> bool:
        """Connect to MT5 terminal with retry logic."""
        retry_count = 0
        while retry_count < self.max_retries:
            if mt5.initialize():
                self.logger.info("Connected to MT5 terminal")
                # Verify symbol is available
                info = mt5.symbol_info(self.symbol)
                if info is None:
                    self.logger.error(f"Symbol {self.symbol} not found. "
                                     f"Enabling symbol...")
                    if not mt5.symbol_select(self.symbol, True):
                        self.logger.error(f"Failed to select {self.symbol}")
                        mt5.shutdown()
                        return False
                    info = mt5.symbol_info(self.symbol)

                self.symbol_info = info
                self.min_lot = info.volume_min
                self.lot_step = info.volume_step
                self.max_lot = info.volume_max

                # Determine supported fill mode from symbol info
                self.fill_type = self._detect_fill_mode(info)

                self.logger.info(
                    f"Symbol {self.symbol}: spread={info.spread}, "
                    f"min_lot={self.min_lot}, lot_step={self.lot_step}, "
                    f"max_lot={self.max_lot}, fill_mode={self.fill_type}"
                )
                return True
            else:
                error = mt5.last_error()
                retry_count += 1
                delay = min(
                    self.base_retry_delay * (2 ** (retry_count - 1)),
                    self.max_retry_delay
                )
                self.logger.warning(
                    f"MT5 connection failed (attempt {retry_count}/"
                    f"{self.max_retries}): {error}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        self.logger.error("Failed to connect to MT5 after all retries")
        return False

    def disconnect(self):
        """Disconnect from MT5."""
        mt5.shutdown()
        self.logger.info("Disconnected from MT5")

    def _ensure_connected(self) -> bool:
        """Check connection and reconnect if necessary."""
        info = mt5.terminal_info()
        if info is None:
            self.logger.warning("MT5 connection lost. Reconnecting...")
            mt5.shutdown()
            return self.connect()
        return True

    def _detect_fill_mode(self, info) -> int:
        """
        Query symbol_info.filling_mode bitmask to select a supported fill mode.
        Falls back through FOK -> IOC -> RETURN.
        """
        filling = info.filling_mode
        # Bit 1: ORDER_FILLING_FOK
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        # Bit 2: ORDER_FILLING_IOC
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        # Fallback to RETURN (always supported for exchange-execution symbols)
        return mt5.ORDER_FILLING_RETURN

    def _get_account_equity(self) -> float:
        """Get current account equity from MT5."""
        account = mt5.account_info()
        if account is None:
            self.logger.error("Failed to get account info")
            return self.equity
        return account.equity

    def _get_open_position_count(self) -> int:
        """Count positions open by this EA (matching magic number)."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return 0
        count = 0
        for pos in positions:
            if pos.magic == self.magic:
                count += 1
        return count

    def _normalize_lot(self, lot: float) -> float:
        """Normalize lot size to broker requirements."""
        lot = max(self.min_lot, lot)
        lot = min(self.max_lot, lot)
        # Round to lot_step
        if self.lot_step > 0:
            steps = round(lot / self.lot_step)
            lot = steps * self.lot_step
        lot = round(lot, 2)
        return max(self.min_lot, lot)

    def _fetch_bars(self, count: int = 500) -> Optional[np.ndarray]:
        """Fetch recent 1-minute OHLC bars from MT5.
        
        Uses position 1 to skip the current forming bar (position 0),
        ensuring we only evaluate completed bars - matching backtest semantics.
        """
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 1, count)
        if rates is None or len(rates) == 0:
            self.logger.error("Failed to fetch bars from MT5")
            return None
        return rates

    def _compute_indicators(self, rates) -> Optional[Dict]:
        """Compute indicators on fetched bars - same as backtest."""
        close = np.array([r[4] for r in rates], dtype=np.float64)  # close
        high = np.array([r[2] for r in rates], dtype=np.float64)   # high
        low = np.array([r[3] for r in rates], dtype=np.float64)    # low

        if len(close) < WARMUP_BARS:
            return None

        # Dual RSI: periods 8 (fast) and 14 (slow) - same as backtest
        rsi_fast = compute_rsi(close, 8)
        rsi_slow = compute_rsi(close, 14)
        atr = compute_atr(high, low, close, 14)

        return {
            "close": close,
            "high": high,
            "low": low,
            "rsi_fast": rsi_fast,
            "rsi_slow": rsi_slow,
            "atr": atr,
        }

    def _generate_signal(self, indicators: Dict, bar_idx: int) -> Tuple[int, str]:
        """
        Generate trading signal - EXACT same logic as run_backtest().

        Returns:
            (signal, signal_type) where signal is 1 (buy), -1 (sell), or 0 (none)
        """
        rsi_fast = indicators["rsi_fast"]
        rsi_slow = indicators["rsi_slow"]
        close = indicators["close"]
        rsi_entry = self.params["rsi_entry"]
        use_4bar = self.params["use_4bar"]

        signal = 0
        signal_type = ""

        # Primary: RSI(8) fast mean-reversion
        if rsi_fast[bar_idx] < rsi_entry:
            signal = 1
            signal_type = "rsi8_oversold"
        elif rsi_fast[bar_idx] > (100 - rsi_entry):
            signal = -1
            signal_type = "rsi8_overbought"

        # Secondary: RSI(14) confirmation (boost if both agree)
        if signal == 0:
            if rsi_slow[bar_idx] < rsi_entry + 5:
                signal = 1
                signal_type = "rsi14_oversold"
            elif rsi_slow[bar_idx] > (95 - rsi_entry):
                signal = -1
                signal_type = "rsi14_overbought"

        # Tertiary: 4-bar reversal pattern
        if signal == 0 and use_4bar and bar_idx >= 4:
            all_down = all(
                close[bar_idx - j] < close[bar_idx - j - 1] for j in range(4)
            )
            all_up = all(
                close[bar_idx - j] > close[bar_idx - j - 1] for j in range(4)
            )
            if all_down:
                signal = 1
                signal_type = "4bar_reversal_buy"
            elif all_up:
                signal = -1
                signal_type = "4bar_reversal_sell"

        return signal, signal_type

    def _compute_position_size(self, sl_dist: float, current_dd: float) -> Tuple[float, float]:
        """
        Compute position size - EXACT same logic as run_backtest().

        Two-mode position sizing matching the original 1911% backtest:
        - Mode 1 (at equity high): if drawdown < at_high_thresh, use full risk_grow
        - Mode 2 (in drawdown): risk = risk_protect * (equity/peak)^dd_power
        - loss_boost: after consecutive losses (>=2), multiply risk by loss_boost (Martingale-lite)
        - win_reduce: after consecutive wins (>=2), multiply risk by win_reduce (lock profits)

        Returns (lot_size, risk_pct)
        """
        risk_grow = self.params["risk_grow"]
        risk_protect = self.params["risk_protect"]
        dd_power = self.params["dd_power"]
        at_high_thresh = self.params["at_high_thresh"]
        loss_boost = self.params["loss_boost"]
        win_reduce = self.params["win_reduce"]
        max_risk_cap = self.params["max_risk_cap"]

        # Two-mode position sizing (matches original backtest exactly)
        if current_dd <= at_high_thresh:
            # Mode 1: At or near equity high - be aggressive
            risk = risk_grow
        else:
            # Mode 2: In drawdown - exponential decay
            eq_ratio = self.equity / self.peak_equity if self.peak_equity > 0 else 1.0
            risk = risk_protect * (eq_ratio ** dd_power)

        # Streak adjustment (matches original backtest)
        if self.consec_losses >= 2:
            risk *= loss_boost  # Martingale-lite: double risk after losses in GROW mode
        elif self.consec_wins >= 2:
            risk *= win_reduce  # Lock profits: reduce risk after wins

        risk = max(0.003, min(max_risk_cap, risk))

        # Lot sizing based on risk
        lot = (self.equity * risk) / (sl_dist * CONTRACT_SIZE)
        lot = max(0.01, min(200.0, round(lot, 2)))

        # Normalize to broker constraints
        lot = self._normalize_lot(lot)

        return lot, risk

    def _send_order(self, signal: int, lot: float, sl: float, tp: float,
                    signal_type: str) -> bool:
        """Send order to MT5."""
        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
            direction_str = "BUY"
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
            direction_str = "SELL"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"LiveTrader_{signal_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.fill_type,
        }

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(f"Order send returned None: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(
                f"Order failed: retcode={result.retcode}, "
                f"comment={result.comment}"
            )
            return False

        self.logger.info(
            f"ORDER FILLED: {direction_str} {lot} lots @ {result.price}, "
            f"SL={sl:.2f}, TP={tp:.2f}, ticket={result.order}, "
            f"signal={signal_type}"
        )
        return True

    def _check_closed_trades(self):
        """
        Check if any positions were closed since last check.
        Query deal history to detect exits, update consec_wins and log exits.
        """
        self.equity = self._get_account_equity()
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Count current positions for this magic
        self.open_positions_count = self._get_open_position_count()

        # Query deal history since last known deal to detect closed trades
        # Use a time window from last_deal_time to now
        from_time = datetime.fromtimestamp(self._last_deal_time, tz=timezone.utc)
        to_time = datetime.now(timezone.utc)

        deals = mt5.history_deals_get(from_time, to_time, group=self.symbol)
        if deals is None or len(deals) == 0:
            return

        for deal in deals:
            # Skip deals we already processed (by time)
            if deal.time <= self._last_deal_time:
                continue
            # Only process deals with our magic number
            if deal.magic != self.magic:
                continue
            # Only process exit deals (DEAL_ENTRY_OUT = 1)
            if deal.entry != 1:
                continue

            # This is an exit deal - update streak tracking
            profit = deal.profit + deal.swap + deal.commission
            if profit > 0:
                self.consec_wins += 1
                self.consec_losses = 0
            else:
                self.consec_losses += 1
                self.consec_wins = 0

            # Determine direction of the closed position
            # DEAL_TYPE_BUY=0 means closing a sell, DEAL_TYPE_SELL=1 means closing a buy
            direction = "SELL_EXIT" if deal.type == 0 else "BUY_EXIT"

            self.logger.info(
                f"TRADE EXIT: {direction} {deal.volume} lots @ {deal.price:.2f}, "
                f"profit={profit:.2f}, ticket={deal.position_id}, "
                f"consec_wins={self.consec_wins}"
            )

            # Log exit to CSV
            self._log_exit_csv(
                direction=direction,
                lot=deal.volume,
                exit_price=deal.price,
                profit=profit,
                ticket=deal.position_id,
            )

            # Update the last processed deal time
            if deal.time > self._last_deal_time:
                self._last_deal_time = deal.time

        # Save state after processing deals
        self._save_state()

    def run(self):
        """
        Main trading loop.

        Checks for new 1-minute bars and evaluates signals.
        Mirrors the bar-by-bar logic of run_backtest().
        """
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("LIVE TRADER STARTING")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Magic: {self.magic}")
        self.logger.info(f"Parameters: {json.dumps(self.params, indent=2)}")
        self.logger.info("=" * 60)

        # Initialize equity tracking
        self.equity = self._get_account_equity()
        self.peak_equity = self.equity
        self.logger.info(f"Starting equity: ${self.equity:.2f}")

        # Load persisted state (peak_equity, consec_wins, bar_count, last_entry_bars)
        self._load_state()
        # Ensure peak_equity is at least current equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.logger.info(
            f"After state load: peak_equity=${self.peak_equity:.2f}, "
            f"consec_wins={self.consec_wins}, bar_count={self.bar_count}"
        )

        # Initialize deal time if not loaded from state
        if self._last_deal_time == 0:
            self._last_deal_time = int(calendar.timegm(
                datetime.now(timezone.utc).timetuple()
            ))

        last_bar_time = 0

        while self.running:
            try:
                # Ensure MT5 connection is alive
                if not self._ensure_connected():
                    self.logger.error("Cannot reconnect. Waiting 60s...")
                    time.sleep(60)
                    continue

                # Fetch latest bars
                rates = self._fetch_bars(count=500)
                if rates is None:
                    time.sleep(5)
                    continue

                # Check if we have a new bar
                current_bar_time = int(rates[-1][0])  # time field
                if current_bar_time == last_bar_time:
                    # No new bar yet, sleep briefly
                    time.sleep(1)
                    continue

                last_bar_time = current_bar_time
                self.bar_count += 1

                # Update equity and position tracking
                self._check_closed_trades()

                # Compute indicators on full bar history
                indicators = self._compute_indicators(rates)
                if indicators is None:
                    continue

                bar_idx = len(indicators["close"]) - 1  # latest completed bar (forming bar excluded by fetch)

                # === ENTRY LOGIC (mirrors run_backtest exactly) ===

                # Check max positions
                if self.open_positions_count >= self.params["max_positions"]:
                    continue

                # Session filter
                bar_time = datetime.fromtimestamp(
                    current_bar_time, tz=timezone.utc
                )
                hour = bar_time.hour
                if hour < self.params["session_start"] or hour > self.params["session_end"]:
                    continue

                # ATR minimum check
                atr_val = indicators["atr"][bar_idx]
                if atr_val < 0.5:
                    continue

                # DD halt check
                current_dd = 0.0
                if self.peak_equity > 0:
                    current_dd = (self.peak_equity - self.equity) / self.peak_equity
                if current_dd >= self.params["dd_halt"]:
                    self.logger.warning(
                        f"DD HALT: current_dd={current_dd:.4f} >= "
                        f"dd_halt={self.params['dd_halt']}"
                    )
                    continue

                # Cooldown check
                slot_available = False
                for s in range(self.params["max_positions"]):
                    if s >= self.open_positions_count:
                        if (self.bar_count - self.last_entry_bars[s]) >= self.params["cooldown"]:
                            slot_available = True
                            break
                if not slot_available:
                    continue

                # Generate signal - EXACT same logic as backtest
                signal, signal_type = self._generate_signal(indicators, bar_idx)
                if signal == 0:
                    continue

                # === TWO-MODE POSITION SIZING (exact copy from backtest) ===
                current_dd = 0.0
                if self.peak_equity > 0:
                    current_dd = (self.peak_equity - self.equity) / self.peak_equity

                # SL/TP distances
                sl_dist = atr_val * self.params["sl_mult"]
                tp_dist = atr_val * self.params["tp_mult"]
                if sl_dist < 0.5:
                    sl_dist = 0.5
                if tp_dist < 0.3:
                    tp_dist = 0.3

                # Position size calculation
                lot, risk_pct = self._compute_position_size(sl_dist, current_dd)

                # Calculate SL/TP prices
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    self.logger.error("Failed to get tick data")
                    continue

                if signal == 1:
                    entry_price = tick.ask
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    entry_price = tick.bid
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                # Send order
                self.logger.info(
                    f"SIGNAL: {signal_type} | dir={'BUY' if signal == 1 else 'SELL'} | "
                    f"lot={lot} | sl_dist={sl_dist:.2f} | tp_dist={tp_dist:.2f} | "
                    f"risk={risk_pct:.4f} | current_dd={current_dd:.6f} | "
                    f"rsi8={indicators['rsi_fast'][bar_idx]:.1f} | "
                    f"rsi14={indicators['rsi_slow'][bar_idx]:.1f} | "
                    f"atr={atr_val:.4f}"
                )

                success = self._send_order(signal, lot, sl_price, tp_price, signal_type)

                if success:
                    # Log trade to CSV
                    self._log_trade_csv(
                        direction="BUY" if signal == 1 else "SELL",
                        lot=lot,
                        entry_price=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        signal_type=signal_type,
                        rsi_fast=indicators["rsi_fast"][bar_idx],
                        rsi_slow=indicators["rsi_slow"][bar_idx],
                        atr_val=atr_val,
                        dd_scale=current_dd,
                        risk_pct=risk_pct,
                    )
                    # Update cooldown tracking
                    for s in range(self.params["max_positions"]):
                        if s >= self.open_positions_count:
                            self.last_entry_bars[s] = self.bar_count
                            break
                    self.open_positions_count += 1
                    # Persist state after entry
                    self._save_state()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)

        self.logger.info("Trading loop ended")

    def stop(self):
        """Signal the trading loop to stop."""
        self.running = False
        self._save_state()
        self.logger.info("Stop signal received - state saved, will exit after current iteration")


# =============================================================================
# SIGNAL HANDLERS AND MAIN
# =============================================================================

_trader_instance: Optional[LiveTrader] = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _trader_instance
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[{sig_name}] Shutting down gracefully...")
    if _trader_instance is not None:
        _trader_instance.stop()


def load_config(config_path: str) -> Dict:
    """Load parameters from a JSON config file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    # Merge with defaults (config overrides defaults)
    params = DEFAULT_PARAMS.copy()
    if "best_params" in config:
        params.update(config["best_params"])
    else:
        params.update(config)
    return params


def main():
    """Entry point with argparse CLI."""
    global _trader_instance

    parser = argparse.ArgumentParser(
        description="Live Trading Bot - Zero-Discrepancy from Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python live_trader.py --symbol XAUUSD --magic 202401
    python live_trader.py --config aggressive_results.json
    python live_trader.py --symbol XAUUSD --magic 100 --log-dir /path/to/logs
        """,
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD",
        help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--magic", type=int, default=202401,
        help="Magic number for order identification (default: 202401)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file with parameters (overrides defaults)"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for log files (default: same as script)"
    )

    args = parser.parse_args()

    # Load parameters
    if args.config:
        params = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        params = DEFAULT_PARAMS.copy()
        print("Using default parameters from aggressive_results.json")

    print(f"Symbol: {args.symbol}")
    print(f"Magic: {args.magic}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Create trader instance
    trader = LiveTrader(
        symbol=args.symbol,
        magic=args.magic,
        params=params,
        log_dir=args.log_dir,
    )
    _trader_instance = trader

    # Connect to MT5
    if not trader.connect():
        print("ERROR: Could not connect to MT5. Exiting.")
        sys.exit(1)

    try:
        # Run the trading loop
        trader.run()
    finally:
        trader.disconnect()
        print("\nTrader shut down cleanly.")


if __name__ == "__main__":
    main()
