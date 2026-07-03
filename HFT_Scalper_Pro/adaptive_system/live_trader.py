"""
Multi-Symbol Adaptive Live Trader
====================================
Connects to MetaTrader 5 and runs the adaptive multi-currency system in real-time.

Features:
    - Multiple symbol monitoring in a single loop
    - Per-symbol regime detection on each new bar
    - Strategy selection and signal generation per-symbol
    - Portfolio-level risk management before order placement
    - State persistence (regime state, online learner, portfolio)
    - Graceful shutdown with state save (SIGINT/SIGTERM)
    - Auto-reconnection with exponential backoff
    - CSV trade log and Python logging
    - CLI configuration

Usage:
    python live_trader.py --symbols XAUUSD,EURUSD,GBPJPY --magic 202501
    python live_trader.py --config config.json --log-dir ./logs
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Ensure core is importable
sys.path.insert(0, str(Path(__file__).parent))

from core.regime_detector import RegimeDetector, MarketRegime
from core.strategies import STRATEGY_REGISTRY
from core.strategy_selector import StrategySelector
from core.position_sizer import PositionSizer, SizingConfig
from core.risk_manager import RiskManager, RiskConfig, TradeProposal, PortfolioPosition
from core.online_learner import StrategyPerformanceTracker, RegimeTransitionMatrix, MarketProfiler
from core.portfolio_manager import PortfolioManager, SymbolConfig
from data_loader import SYMBOL_METADATA
from config import SystemConfig, load_config, get_balanced_config


# ============================================================================
# LIVE TRADER
# ============================================================================

class AdaptiveLiveTrader:
    """
    Multi-symbol adaptive live trading engine.

    Connects to MT5, monitors multiple symbols, detects regimes, selects
    strategies, and executes trades with portfolio-level risk management.
    """

    def __init__(self, symbols: List[str], magic: int = 202501,
                 config: Optional[SystemConfig] = None, log_dir: Optional[str] = None):
        """
        Initialize the adaptive live trader.

        Parameters
        ----------
        symbols : list of str
            Symbols to monitor and trade.
        magic : int
            Magic number for trade identification.
        config : SystemConfig, optional
            System configuration. Uses balanced defaults if not provided.
        log_dir : str, optional
            Directory for log and state files.
        """
        self.symbols = symbols
        self.magic = magic
        self.config = config or get_balanced_config()
        self.running = False

        # Paths
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent
        self.state_file = self.log_dir / "adaptive_state.json"

        # Logging
        self._setup_logging()
        self._setup_csv_log()

        # Per-symbol components
        self._detectors: Dict[str, RegimeDetector] = {}
        self._sizers: Dict[str, PositionSizer] = {}
        self._profilers: Dict[str, MarketProfiler] = {}
        self._last_bar_times: Dict[str, int] = {}
        self._bar_counts: Dict[str, int] = {}

        for symbol in symbols:
            self._detectors[symbol] = RegimeDetector()
            meta = SYMBOL_METADATA.get(symbol, SYMBOL_METADATA.get("XAUUSD", {}))
            self._sizers[symbol] = PositionSizer(SizingConfig(
                contract_size=meta.get("contract_size", 100.0),
                initial_equity=self.config.position_sizing.initial_equity,
                leverage=self.config.position_sizing.leverage,
                risk_grow=self.config.position_sizing.risk_grow,
                risk_protect=self.config.position_sizing.risk_protect,
                dd_power=self.config.position_sizing.dd_power,
                dd_halt=self.config.position_sizing.dd_halt,
            ))
            self._profilers[symbol] = MarketProfiler()
            self._last_bar_times[symbol] = 0
            self._bar_counts[symbol] = 0

        # Shared components
        self._selector = StrategySelector()
        self._risk_mgr = RiskManager(RiskConfig(
            max_concurrent_positions=self.config.risk_management.max_concurrent_positions,
            max_total_risk=self.config.risk_management.max_total_risk,
            max_risk_per_symbol=self.config.risk_management.max_risk_per_symbol,
            dd_halt_threshold=self.config.risk_management.dd_halt_threshold,
            daily_loss_limit=self.config.risk_management.daily_loss_limit,
        ))
        self._perf_tracker = StrategyPerformanceTracker()
        self._transition_matrix = RegimeTransitionMatrix()

        # Portfolio state
        self.equity = 0.0
        self.peak_equity = 0.0
        self.open_positions: List[Dict] = []
        self.consec_wins = 0
        self.consec_losses = 0
        self._daily_pnl = 0.0
        self._last_deal_time = 0

        # MT5 connection
        self.max_retries = self.config.live_trading.max_retries
        self.base_retry_delay = self.config.live_trading.base_retry_delay
        self.max_retry_delay = self.config.live_trading.max_retry_delay
        self._symbol_info: Dict[str, object] = {}
        self._fill_types: Dict[str, int] = {}

    # ========================================================================
    # SETUP
    # ========================================================================

    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger("AdaptiveLiveTrader")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.log_dir / "adaptive_trader.log"
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self.config.live_trading.log_level, logging.INFO))
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        self.logger.addHandler(ch)

    def _setup_csv_log(self):
        """Initialize CSV trade log."""
        self.csv_path = self.log_dir / "adaptive_trades.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "event", "direction", "lot_size",
                    "entry_price", "sl", "tp", "magic", "regime", "strategy",
                    "confidence", "equity", "peak_equity", "risk_pct", "profit", "ticket"
                ])

    # ========================================================================
    # MT5 CONNECTION
    # ========================================================================

    def connect(self) -> bool:
        """Connect to MT5 terminal with retry logic."""
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not available.")
            self.logger.error("Install with: pip install MetaTrader5")
            self.logger.error("Note: Only works on Windows with MT5 terminal installed.")
            return False

        retry_count = 0
        while retry_count < self.max_retries:
            if mt5.initialize():
                self.logger.info("Connected to MT5 terminal")

                # Verify and setup all symbols
                for symbol in self.symbols:
                    info = mt5.symbol_info(symbol)
                    if info is None:
                        self.logger.warning(f"Symbol {symbol} not found, enabling...")
                        if not mt5.symbol_select(symbol, True):
                            self.logger.error(f"Failed to select {symbol}")
                            continue
                        info = mt5.symbol_info(symbol)

                    if info is not None:
                        self._symbol_info[symbol] = info
                        self._fill_types[symbol] = self._detect_fill_mode(info)
                        self.logger.info(
                            f"  {symbol}: spread={info.spread}, "
                            f"min_lot={info.volume_min}, "
                            f"lot_step={info.volume_step}"
                        )

                return True
            else:
                error = mt5.last_error() if hasattr(mt5, 'last_error') else "Unknown"
                retry_count += 1
                delay = min(
                    self.base_retry_delay * (2 ** (retry_count - 1)),
                    self.max_retry_delay
                )
                self.logger.warning(
                    f"MT5 connection failed (attempt {retry_count}/{self.max_retries}): "
                    f"{error}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        self.logger.error("Failed to connect to MT5 after all retries")
        return False

    def disconnect(self):
        """Disconnect from MT5."""
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.logger.info("Disconnected from MT5")

    def _ensure_connected(self) -> bool:
        """Check connection and reconnect if necessary."""
        if not MT5_AVAILABLE:
            return False
        info = mt5.terminal_info()
        if info is None:
            self.logger.warning("MT5 connection lost. Reconnecting...")
            mt5.shutdown()
            return self.connect()
        return True

    def _detect_fill_mode(self, info) -> int:
        """Detect supported fill mode for a symbol."""
        filling = info.filling_mode
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================

    def _save_state(self):
        """Save system state to JSON for crash recovery."""
        state = {
            "peak_equity": self.peak_equity,
            "consec_wins": self.consec_wins,
            "consec_losses": self.consec_losses,
            "daily_pnl": self._daily_pnl,
            "last_deal_time": self._last_deal_time,
            "bar_counts": self._bar_counts.copy(),
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
        """Load persisted state from JSON file."""
        if not self.state_file.exists():
            self.logger.info("No state file found, starting fresh")
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.peak_equity = state.get("peak_equity", self.equity)
            self.consec_wins = state.get("consec_wins", 0)
            self.consec_losses = state.get("consec_losses", 0)
            self._daily_pnl = state.get("daily_pnl", 0.0)
            self._last_deal_time = state.get("last_deal_time", 0)
            saved_bar_counts = state.get("bar_counts", {})
            for symbol in self.symbols:
                if symbol in saved_bar_counts:
                    self._bar_counts[symbol] = saved_bar_counts[symbol]
            self.logger.info(
                f"Restored state: peak_equity={self.peak_equity:.2f}, "
                f"consec_wins={self.consec_wins}, consec_losses={self.consec_losses}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}. Starting fresh.")

    # ========================================================================
    # TRADING LOGIC
    # ========================================================================

    def _get_account_equity(self) -> float:
        """Get current account equity from MT5."""
        account = mt5.account_info()
        if account is None:
            return self.equity
        return account.equity

    def _get_open_positions(self) -> List[Dict]:
        """Get all positions opened by this trader (matching magic)."""
        positions = []
        for symbol in self.symbols:
            symbol_positions = mt5.positions_get(symbol=symbol)
            if symbol_positions is None:
                continue
            for pos in symbol_positions:
                if pos.magic == self.magic:
                    positions.append({
                        "symbol": pos.symbol,
                        "direction": 1 if pos.type == 0 else -1,
                        "lot_size": pos.volume,
                        "entry_price": pos.price_open,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "ticket": pos.ticket,
                        "profit": pos.profit,
                    })
        return positions

    def _fetch_bars(self, symbol: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch recent bars from MT5 for a symbol."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df.columns = ["time", "open", "high", "low", "close", "tick_volume",
                      "spread", "real_volume"]
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)
        df["volume"] = df["tick_volume"]
        df["tick_count"] = df["tick_volume"]

        return df[["open", "high", "low", "close", "volume", "tick_count"]]

    def _send_order(self, symbol: str, direction: int, lot: float,
                    sl: float, tp: float, regime: str, strategy: str) -> bool:
        """Send order to MT5."""
        if direction == 1:
            order_type = mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if tick else 0
        else:
            order_type = mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else 0

        if price == 0:
            self.logger.error(f"Failed to get price for {symbol}")
            return False

        fill_type = self._fill_types.get(symbol, mt5.ORDER_FILLING_FOK)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"Adaptive_{strategy}_{regime[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(f"Order send returned None: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(
                f"Order failed for {symbol}: retcode={result.retcode}, "
                f"comment={result.comment}"
            )
            return False

        dir_str = "BUY" if direction == 1 else "SELL"
        self.logger.info(
            f"ORDER FILLED: {dir_str} {lot} {symbol} @ {result.price:.5f}, "
            f"SL={sl:.5f}, TP={tp:.5f}, regime={regime}, strategy={strategy}"
        )
        return True

    def _log_trade_csv(self, symbol: str, event: str, direction: int,
                       lot: float, price: float, sl: float, tp: float,
                       regime: str, strategy: str, confidence: float,
                       risk_pct: float, profit: float = 0.0, ticket: int = 0):
        """Log trade event to CSV."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                symbol, event,
                "BUY" if direction == 1 else "SELL",
                lot, round(price, 5), round(sl, 5), round(tp, 5),
                self.magic, regime, strategy,
                round(confidence, 3),
                round(self.equity, 2), round(self.peak_equity, 2),
                round(risk_pct, 6), round(profit, 2), ticket,
            ])

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        """
        Main trading loop.

        Monitors all configured symbols for new bars, runs the adaptive
        pipeline, and executes approved trades.
        """
        if not MT5_AVAILABLE:
            self.logger.error("Cannot run: MetaTrader5 package not installed.")
            self.logger.error("This trader requires Windows with MT5 terminal.")
            return

        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("ADAPTIVE MULTI-SYMBOL LIVE TRADER STARTING")
        self.logger.info(f"Symbols: {', '.join(self.symbols)}")
        self.logger.info(f"Magic: {self.magic}")
        self.logger.info(f"Risk Profile: {self.config.risk_profile}")
        self.logger.info("=" * 60)

        # Initialize equity
        self.equity = self._get_account_equity()
        self.peak_equity = self.equity
        self.logger.info(f"Starting equity: ${self.equity:.2f}")

        # Load persisted state
        self._load_state()
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Initialize deal time
        if self._last_deal_time == 0:
            import calendar
            self._last_deal_time = int(calendar.timegm(
                datetime.now(timezone.utc).timetuple()
            ))

        state_save_counter = 0

        while self.running:
            try:
                if not self._ensure_connected():
                    self.logger.error("Cannot reconnect. Waiting 60s...")
                    time.sleep(60)
                    continue

                # Update equity
                self.equity = self._get_account_equity()
                if self.equity > self.peak_equity:
                    self.peak_equity = self.equity

                # Get current positions
                self.open_positions = self._get_open_positions()

                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)

                # Periodic state save
                state_save_counter += 1
                if state_save_counter >= self.config.live_trading.state_save_interval:
                    self._save_state()
                    state_save_counter = 0

                # Sleep between iterations
                time.sleep(self.config.live_trading.loop_interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)

        # Graceful shutdown
        self._save_state()
        self.logger.info("Trading loop ended. State saved.")

    def _process_symbol(self, symbol: str):
        """Process a single symbol: fetch bars, detect regime, generate signal."""
        # Fetch bars
        bars = self._fetch_bars(symbol, count=self.config.live_trading.bar_fetch_count)
        if bars is None or len(bars) < 150:
            return

        # Check for new bar
        current_bar_time = int(bars.index[-1].timestamp())
        if current_bar_time == self._last_bar_times.get(symbol, 0):
            return  # No new bar

        self._last_bar_times[symbol] = current_bar_time
        self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

        # Get bar data
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        close = bars["close"].values.astype(np.float64)

        meta = SYMBOL_METADATA.get(symbol, SYMBOL_METADATA.get("XAUUSD", {}))
        pip_size = meta.get("pip_size", 0.01)
        contract_size = meta.get("contract_size", 100.0)
        typical_spread = meta.get("typical_spread", 0.30)

        # Step 1: Detect regime
        regime, confidence = self._detectors[symbol].detect(high, low, close)
        self._transition_matrix.observe(regime)

        # Step 2: Select strategy
        strategy, size_mult = self._selector.select(regime, confidence)
        strategy_name = strategy.get_name()

        # Step 3: Generate signal
        window_size = min(300, len(bars))
        signal_bars = bars.iloc[-window_size:]
        signals = strategy.generate_signals(signal_bars)

        latest_signal = signals[-1]
        direction = int(latest_signal[0])
        sl_dist = latest_signal[1]
        tp_dist = latest_signal[2]

        if direction == 0 or sl_dist <= 0:
            return

        # Check if we already have same-direction position on this symbol
        has_position = any(
            p["symbol"] == symbol and p["direction"] == direction
            for p in self.open_positions
        )
        if has_position:
            return

        # Step 4: Position sizing
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        win_rate = self._perf_tracker.get_win_rate(strategy_name, regime)

        sizing_result = self._sizers[symbol].compute_size(
            equity=self.equity,
            peak_equity=self.peak_equity,
            sl_distance=sl_dist,
            atr=sl_dist / 1.5,
            win_rate=win_rate,
            regime_size_mult=size_mult,
            consec_wins=self.consec_wins,
            consec_losses=self.consec_losses,
        )

        if not sizing_result.approved:
            return

        lot_size = sizing_result.lot_size

        # Step 5: Risk manager check
        portfolio_positions = [
            PortfolioPosition(
                symbol=p["symbol"],
                direction=p["direction"],
                lot_size=p["lot_size"],
                entry_price=p["entry_price"],
                sl_distance=abs(p["entry_price"] - p["sl"]) if p["sl"] > 0 else sl_dist,
                contract_size=SYMBOL_METADATA.get(p["symbol"], {}).get("contract_size", 100.0),
            ) for p in self.open_positions
        ]

        proposal = TradeProposal(
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            contract_size=contract_size,
            strategy_name=strategy_name,
            hour=datetime.now(timezone.utc).hour,
        )

        risk_decision = self._risk_mgr.approve_trade(
            proposal=proposal,
            current_positions=portfolio_positions,
            equity=self.equity,
            peak_equity=self.peak_equity,
            daily_pnl=self._daily_pnl,
        )

        if not risk_decision.approved:
            return

        final_lot = risk_decision.adjusted_lot_size

        # Step 6: Execute trade
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return

        if direction == 1:
            entry_price = tick.ask
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            entry_price = tick.bid
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        success = self._send_order(
            symbol, direction, final_lot, sl_price, tp_price,
            regime.name, strategy_name
        )

        if success:
            self._log_trade_csv(
                symbol=symbol, event="ENTRY", direction=direction,
                lot=final_lot, price=entry_price, sl=sl_price, tp=tp_price,
                regime=regime.name, strategy=strategy_name,
                confidence=confidence, risk_pct=sizing_result.risk_pct,
            )
            self.logger.info(
                f"  Regime: {regime.name} (conf={confidence:.2f}), "
                f"Strategy: {strategy_name}, Size: {final_lot} lots"
            )

    def stop(self):
        """Signal the trading loop to stop."""
        self.running = False
        self._save_state()
        self.logger.info("Stop signal received - state saved")


# ============================================================================
# SIGNAL HANDLERS AND MAIN
# ============================================================================

_trader_instance: Optional[AdaptiveLiveTrader] = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _trader_instance
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[{sig_name}] Shutting down gracefully...")
    if _trader_instance is not None:
        _trader_instance.stop()


def main():
    """Entry point with argparse CLI."""
    global _trader_instance

    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Symbol Live Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python live_trader.py --symbols XAUUSD,EURUSD,GBPJPY --magic 202501
    python live_trader.py --symbols XAUUSD --config config.json
    python live_trader.py --symbols XAUUSD,EURUSD --magic 100 --log-dir ./logs
        """,
    )
    parser.add_argument(
        "--symbols", type=str, default="XAUUSD,EURUSD,GBPJPY",
        help="Comma-separated list of symbols to trade (default: XAUUSD,EURUSD,GBPJPY)"
    )
    parser.add_argument(
        "--magic", type=int, default=202501,
        help="Magic number for order identification (default: 202501)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for log files (default: same as script)"
    )
    parser.add_argument(
        "--risk-profile", type=str, default=None,
        choices=["conservative", "balanced", "aggressive"],
        help="Risk profile preset (overridden by --config if both specified)"
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Load config
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    elif args.risk_profile:
        from config import RISK_PROFILES
        config = RISK_PROFILES[args.risk_profile]()
        print(f"Using risk profile: {args.risk_profile}")
    else:
        config = get_balanced_config()
        print("Using balanced risk profile (default)")

    print(f"Symbols: {symbols}")
    print(f"Magic: {args.magic}")
    print()

    # Check MT5 availability
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 package not installed.")
        print("Install with: pip install MetaTrader5")
        print("Note: MetaTrader5 only works on Windows with MT5 terminal installed.")
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Create trader
    trader = AdaptiveLiveTrader(
        symbols=symbols,
        magic=args.magic,
        config=config,
        log_dir=args.log_dir,
    )
    _trader_instance = trader

    # Connect to MT5
    if not trader.connect():
        print("ERROR: Could not connect to MT5. Exiting.")
        sys.exit(1)

    try:
        trader.run()
    finally:
        trader.disconnect()
        print("\nAdaptive trader shut down cleanly.")


if __name__ == "__main__":
    main()
