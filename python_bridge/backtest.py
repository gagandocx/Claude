"""
=============================================================
  Python ML Bridge - Fast-Forward Backtester
  Simulates the exact trading strategy bar-by-bar on historical
  data downloaded from yfinance, records trades to the
  AutoOptimizer, and outputs a detailed summary plus JSON results.
=============================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import SessionConfig, DataConfig, AutoOptimizerConfig
from strategies.auto_optimizer import AutoOptimizer
from data.market_data import MarketDataFetcher


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_SL_DISTANCE = 3.0       # $3 stop loss
MAX_POSITIONS = 5               # Max concurrent positions
MAX_HOLD_BARS = 50              # Max bars to hold a trade
MOMENTUM_LOOKBACK = 5           # 5 bars for momentum
MOMENTUM_THRESHOLD = 0.50       # $0.50 for momentum direction
MOMENTUM_EXIT_THRESHOLD = 0.30  # $0.30 reversal triggers exit
MIN_BARS_BETWEEN_ENTRIES = 3    # Minimum bars between entries (cooldown)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Progressive trailing stop thresholds
TRAIL_BE_THRESHOLD = 0.50       # Move SL to break-even at $0.50 profit
TRAIL_TIER1_THRESHOLD = 1.0     # Trail $0.50 behind at $1.0 profit
TRAIL_TIER1_DISTANCE = 0.50
TRAIL_TIER2_THRESHOLD = 2.0     # Trail $0.30 behind at $2.0 profit
TRAIL_TIER2_DISTANCE = 0.30
TRAIL_TIER3_THRESHOLD = 3.0     # Trail $0.20 behind at $3.0 profit
TRAIL_TIER3_DISTANCE = 0.20


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────
class Position:
    """Represents an open simulated trade position."""

    def __init__(self, direction: str, entry_price: float, sl_price: float,
                 entry_bar_index: int, entry_time: str, rsi_at_entry: float,
                 session: str, confidence: float = 0.5):
        self.direction = direction
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.entry_bar_index = entry_bar_index
        self.entry_time = entry_time
        self.rsi_at_entry = rsi_at_entry
        self.session = session
        self.confidence = confidence
        self.max_profit = 0.0
        self.trail_tier = "none"

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P/L at current price."""
        if self.direction == "BUY":
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price


# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────
def compute_momentum_direction(closes: pd.Series, index: int) -> str:
    """
    Compute momentum direction from last MOMENTUM_LOOKBACK bars.

    Compares close[-1] to close[-6] (5 bars back).
    Returns "BUY", "SELL", or "FLAT".
    """
    if index < MOMENTUM_LOOKBACK:
        return "FLAT"

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    diff = current_close - lookback_close

    if diff > MOMENTUM_THRESHOLD:
        return "BUY"
    elif diff < -MOMENTUM_THRESHOLD:
        return "SELL"
    else:
        return "FLAT"


def compute_momentum_magnitude(closes: pd.Series, index: int) -> float:
    """
    Compute the absolute momentum magnitude for synthetic confidence.

    Returns a value between 0.0 and 1.0 based on momentum strength:
    - Momentum at threshold ($0.50) maps to ~0.3 confidence
    - Momentum at $1.50 maps to ~0.7 confidence
    - Momentum at $3.0+ maps to ~0.9 confidence

    Uses a sigmoid-like scaling: confidence = 0.2 + 0.7 * (1 - exp(-abs_diff / scale))
    """
    if index < MOMENTUM_LOOKBACK:
        return 0.0

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    abs_diff = abs(current_close - lookback_close)

    # Scale factor: momentum of $1.50 yields ~0.7 confidence
    scale = 1.5
    confidence = 0.2 + 0.7 * (1.0 - np.exp(-abs_diff / scale))
    return round(min(confidence, 0.95), 4)


def detect_session(timestamp) -> str:
    """
    Detect trading session from bar timestamp UTC hour.

    Uses SessionConfig ranges:
    - Asian: 0-8 UTC
    - London: 8-16 UTC
    - New York: 13-21 UTC
    - Overlap (London/NY): 13-16 UTC
    """
    session_config = SessionConfig()

    if hasattr(timestamp, 'hour'):
        hour = timestamp.hour
    else:
        hour = pd.Timestamp(timestamp).hour

    # Check overlap first (London/NY overlap)
    if session_config.ny_start <= hour < session_config.london_end:
        return "overlap"
    elif session_config.asian_start <= hour < session_config.asian_end:
        return "asian"
    elif session_config.london_start <= hour < session_config.london_end:
        return "london"
    elif session_config.ny_start <= hour < session_config.ny_end:
        return "newyork"
    else:
        return "off_session"


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI using the ta library."""
    rsi = ta.momentum.RSIIndicator(close=closes, window=period).rsi()
    return rsi


def apply_progressive_trailing(position: Position, current_price: float) -> float:
    """
    Apply progressive trailing stop logic.

    - At $0.50 unrealized profit: move SL to break-even
    - At $1.0 profit: trail $0.50 behind current price
    - At $2.0 profit: trail $0.30 behind current price
    - At $3.0 profit: trail $0.20 behind current price

    Returns the updated SL price.
    """
    pnl = position.unrealized_pnl(current_price)
    position.max_profit = max(position.max_profit, pnl)

    sl_price = position.sl_price

    if pnl >= TRAIL_TIER3_THRESHOLD:
        # Trail $0.20 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER3_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER3_DISTANCE
        position.trail_tier = "tight"
    elif pnl >= TRAIL_TIER2_THRESHOLD:
        # Trail $0.30 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER2_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER2_DISTANCE
        position.trail_tier = "medium"
    elif pnl >= TRAIL_TIER1_THRESHOLD:
        # Trail $0.50 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER1_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER1_DISTANCE
        position.trail_tier = "wide"
    elif pnl >= TRAIL_BE_THRESHOLD:
        # Move SL to break-even
        new_sl = position.entry_price
        if position.trail_tier == "none":
            position.trail_tier = "breakeven"
    else:
        return sl_price

    # Only move SL in favorable direction (never move it further from price)
    if position.direction == "BUY":
        sl_price = max(sl_price, new_sl)
    else:
        sl_price = min(sl_price, new_sl)

    position.sl_price = sl_price
    return sl_price


def check_sl_hit(position: Position, bar_low: float, bar_high: float) -> Tuple[bool, float]:
    """
    Check if stop loss was hit during this bar.

    For BUY: if bar.Low <= sl_price, exit at sl_price
    For SELL: if bar.High >= sl_price, exit at sl_price

    Returns (hit, exit_price).
    """
    if position.direction == "BUY":
        if bar_low <= position.sl_price:
            return True, position.sl_price
    else:
        if bar_high >= position.sl_price:
            return True, position.sl_price
    return False, 0.0


def check_momentum_exit(position: Position, closes: pd.Series, index: int) -> bool:
    """
    Check if momentum has reversed against the position.

    If momentum reverses > $0.30 against position, signal exit.
    """
    if index < MOMENTUM_LOOKBACK:
        return False

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    diff = current_close - lookback_close

    if position.direction == "BUY" and diff < -MOMENTUM_EXIT_THRESHOLD:
        return True
    elif position.direction == "SELL" and diff > MOMENTUM_EXIT_THRESHOLD:
        return True

    return False


# ─────────────────────────────────────────────
#  BACKTESTER ENGINE
# ─────────────────────────────────────────────
class Backtester:
    """
    Fast-forward backtester that simulates the trading strategy
    bar-by-bar on historical data.
    """

    def __init__(self, verbose: bool = False, min_bars_between_entries: int = MIN_BARS_BETWEEN_ENTRIES):
        self.verbose = verbose
        self.min_bars_between_entries = min_bars_between_entries
        self.open_positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.last_entry_bar: int = -min_bars_between_entries  # Allow entry on first qualifying bar
        self.auto_optimizer = AutoOptimizer(
            config=AutoOptimizerConfig(
                optimize_frequency=50,
                min_trades_before_tuning=10,
            ),
            state_dir=os.path.dirname(os.path.abspath(__file__))
        )

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run the backtest on the provided OHLCV DataFrame.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
                and DatetimeIndex (UTC).

        Returns:
            Dict with trade_log and summary statistics.
        """
        if df.empty or len(df) < MOMENTUM_LOOKBACK + RSI_PERIOD + 1:
            return {"trade_log": [], "summary": self._empty_summary()}

        # Compute RSI for all bars
        rsi_series = compute_rsi(df["Close"])

        # Bar-by-bar simulation
        for i in range(MOMENTUM_LOOKBACK + RSI_PERIOD, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]
            current_price = bar["Close"]

            # --- Trade management for open positions ---
            positions_to_close = []
            for pos in self.open_positions:
                bars_held = i - pos.entry_bar_index

                # Apply progressive trailing stop
                apply_progressive_trailing(pos, current_price)

                # Check SL hit
                sl_hit, exit_price = check_sl_hit(pos, bar["Low"], bar["High"])
                if sl_hit:
                    pnl = pos.unrealized_pnl(exit_price)
                    positions_to_close.append((pos, exit_price, pnl, "sl_hit"))
                    continue

                # Check time exit
                if bars_held >= MAX_HOLD_BARS:
                    pnl = pos.unrealized_pnl(current_price)
                    positions_to_close.append((pos, current_price, pnl, "time_exit"))
                    continue

                # Check momentum exit
                if check_momentum_exit(pos, df["Close"], i):
                    pnl = pos.unrealized_pnl(current_price)
                    positions_to_close.append((pos, current_price, pnl, "momentum_exit"))
                    continue

            # Close positions that triggered exit
            for pos, exit_price, pnl, exit_reason in positions_to_close:
                self._close_position(pos, exit_price, pnl, bar_time, exit_reason)
                self.open_positions.remove(pos)

            # --- Entry logic ---
            if len(self.open_positions) < MAX_POSITIONS:
                # Check cooldown: min bars since last entry
                bars_since_last_entry = i - self.last_entry_bar
                if bars_since_last_entry >= self.min_bars_between_entries:
                    momentum = compute_momentum_direction(df["Close"], i)
                    if momentum != "FLAT":
                        rsi_value = rsi_series.iloc[i]
                        if not np.isnan(rsi_value):
                            # RSI filter: don't buy overbought, don't sell oversold
                            rsi_ok = True
                            if momentum == "BUY" and rsi_value > RSI_OVERBOUGHT:
                                rsi_ok = False
                            elif momentum == "SELL" and rsi_value < RSI_OVERSOLD:
                                rsi_ok = False

                            if rsi_ok:
                                session = detect_session(bar_time)
                                entry_price = current_price

                                if momentum == "BUY":
                                    sl_price = entry_price - DEFAULT_SL_DISTANCE
                                else:
                                    sl_price = entry_price + DEFAULT_SL_DISTANCE

                                # Compute synthetic confidence from momentum magnitude
                                confidence = compute_momentum_magnitude(df["Close"], i)

                                entry_time_str = str(bar_time)
                                pos = Position(
                                    direction=momentum,
                                    entry_price=entry_price,
                                    sl_price=sl_price,
                                    entry_bar_index=i,
                                    entry_time=entry_time_str,
                                    rsi_at_entry=rsi_value,
                                    session=session,
                                    confidence=confidence,
                                )
                                self.open_positions.append(pos)
                                self.last_entry_bar = i

                                if self.verbose:
                                    print(f"  [ENTRY] {momentum} at {entry_price:.2f} "
                                          f"SL={sl_price:.2f} RSI={rsi_value:.1f} "
                                          f"conf={confidence:.3f} session={session} "
                                          f"time={entry_time_str}")

        # Close any remaining open positions at last bar's close
        if self.open_positions:
            last_bar = df.iloc[-1]
            last_time = df.index[-1]
            last_close = last_bar["Close"]
            for pos in list(self.open_positions):
                pnl = pos.unrealized_pnl(last_close)
                self._close_position(pos, last_close, pnl, last_time, "end_of_data")
            self.open_positions.clear()

        # Save auto_optimizer state
        self.auto_optimizer.save_state()

        # Build results
        summary = self._compute_summary()
        return {
            "trade_log": self.closed_trades,
            "summary": summary,
        }

    def _close_position(self, pos: Position, exit_price: float, pnl: float,
                        exit_time, exit_reason: str) -> None:
        """Record a closed trade and feed to AutoOptimizer."""
        exit_time_str = str(exit_time)

        # Determine trail tier for optimizer
        trail_tier = pos.trail_tier if pos.trail_tier != "none" else "wide"

        trade_record = {
            "direction": pos.direction,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "entry_time": pos.entry_time,
            "exit_time": exit_time_str,
            "session": pos.session,
            "rsi_at_entry": round(pos.rsi_at_entry, 2),
            "exit_reason": exit_reason,
            "trail_tier": trail_tier,
            "max_profit": round(pos.max_profit, 2),
            "confidence": pos.confidence,
        }
        self.closed_trades.append(trade_record)

        # Record to AutoOptimizer
        trade_context = {
            "session": pos.session,
            "confidence": pos.confidence,
            "momentum_lookback": MOMENTUM_LOOKBACK,
            "sl_distance": DEFAULT_SL_DISTANCE,
            "result_pnl": round(pnl, 2),
            "direction": pos.direction,
            "rsi_at_entry": round(pos.rsi_at_entry, 2),
            "trail_tier": trail_tier,
            "cooldown_used": self.min_bars_between_entries,
            "max_positions_at_entry": MAX_POSITIONS,
            "entry_time": pos.entry_time,
            "exit_time": exit_time_str,
        }
        self.auto_optimizer.record_trade(trade_context)

        if self.verbose:
            print(f"  [EXIT] {pos.direction} PnL={pnl:+.2f} reason={exit_reason} "
                  f"exit_price={exit_price:.2f} time={exit_time_str}")

    def _compute_summary(self) -> Dict:
        """Compute backtest summary statistics."""
        if not self.closed_trades:
            return self._empty_summary()

        pnls = [t["pnl"] for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_count = len(wins)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(pnls)

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_note": "closed-trade drawdown (not mark-to-market equity drawdown)",
            "wins": win_count,
            "losses": len(losses),
        }

    def _compute_max_drawdown(self, pnls: List[float]) -> float:
        """Compute maximum drawdown from a series of P/L values."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return max_dd

    def _empty_summary(self) -> Dict:
        """Return an empty summary when no trades are generated."""
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "wins": 0,
            "losses": 0,
        }


# ─────────────────────────────────────────────
#  CLI AND MAIN
# ─────────────────────────────────────────────
def download_data(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """
    Download historical data from yfinance.

    Args:
        symbol: Trading symbol (XAUUSD maps to GC=F)
        days: Number of days of data
        interval: Bar interval (1m, 5m, 15m)

    Returns:
        DataFrame with OHLCV data
    """
    # Map symbol to yfinance ticker
    ticker_map = {
        "XAUUSD": "GC=F",
        "GC=F": "GC=F",
    }
    ticker = ticker_map.get(symbol.upper(), symbol)

    # Determine period based on interval limits
    if interval == "1m":
        max_days = 7
        actual_days = min(days, max_days)
    else:
        max_days = 60
        actual_days = min(days, max_days)

    period = f"{actual_days}d"

    print(f"[Backtest] Downloading {ticker} data: period={period}, interval={interval}")

    fetcher = MarketDataFetcher()
    df = fetcher.fetch_ohlcv(ticker=ticker, period=period, interval=interval)

    if df.empty:
        print("[Backtest] WARNING: No data received from yfinance")
    else:
        print(f"[Backtest] Downloaded {len(df)} bars")

    return df


def print_summary(summary: Dict) -> None:
    """Print backtest summary to console."""
    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:    {summary['total_trades']}")
    print(f"  Win Rate:        {summary['win_rate'] * 100:.1f}%")
    print(f"  Total P/L:       ${summary['total_pnl']:.2f}")
    print(f"  Avg Win:         ${summary['avg_win']:.2f}")
    print(f"  Avg Loss:        ${summary['avg_loss']:.2f}")
    print(f"  Profit Factor:   {summary['profit_factor']}")
    print(f"  Max Drawdown:    ${summary['max_drawdown']:.2f} (closed-trade)")
    print(f"  Wins/Losses:     {summary['wins']}/{summary['losses']}")
    print("=" * 60)


def save_results(results: Dict, output_dir: str) -> None:
    """Save backtest results to JSON file."""
    output_path = os.path.join(output_dir, "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Backtest] Results saved to {output_path}")


def main():
    """Main entry point for the backtester CLI."""
    parser = argparse.ArgumentParser(
        description="Fast-Forward Backtester - Simulates trading strategy on historical data"
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of days of historical data (default: 5)"
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD",
        help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--interval", type=str, default="1m",
        choices=["1m", "5m", "15m"],
        help="Bar interval (default: 1m)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each trade entry/exit"
    )

    args = parser.parse_args()

    print(f"[Backtest] Starting backtest: symbol={args.symbol}, "
          f"days={args.days}, interval={args.interval}")

    # Download data
    df = download_data(args.symbol, args.days, args.interval)
    if df.empty:
        print("[Backtest] ERROR: No data available. Exiting.")
        sys.exit(1)

    # Run backtest
    backtester = Backtester(verbose=args.verbose)
    results = backtester.run(df)

    # Print summary
    print_summary(results["summary"])

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_results(results, output_dir)

    # Save auto_optimizer state (already saved in run(), but ensure it's there)
    print(f"[Backtest] Auto-optimizer state saved "
          f"({backtester.auto_optimizer.trade_count} trades recorded)")

    return results


if __name__ == "__main__":
    main()
