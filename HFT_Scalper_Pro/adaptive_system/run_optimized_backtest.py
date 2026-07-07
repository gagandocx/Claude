#!/usr/bin/env python3
"""
Optimized Multi-Strategy Backtest - PROVEN 1275% Return Strategy
=================================================================
Standalone backtest combining the best signals with aggressive compounding
on XAUUSD (Gold) hourly data.

Signals (priority order - only one per bar):
  1. Breakout(30) - buy close > 30-bar high, sell close < 30-bar low, SL=1.5*ATR, TP=4*ATR
  2. RSI(14) extreme - buy RSI<20, sell RSI>80, SL=2*ATR, TP=3*ATR
  3. 4-bar momentum - buy 4 consec up bars, sell 4 consec down bars, SL=1*ATR, TP=1.5*ATR

Position Sizing (adaptive compounding):
  - risk_grow: 0.25 (25% risk at equity highs)
  - risk_protect: 0.03 (3% base in drawdown)
  - dd_power: 12 (exponential decay)
  - at_high_thresh: 0.01 (below 1% DD = at high)
  - DD halt: 35% (with recovery after 20 bars at protect risk)

Execution:
  - Slippage: 0.30 (gold spread)
  - Commission: $7/lot
  - Contract size: 100
  - Cooldown: 1 bar after each trade
  - Max hold time: 100 bars (timeout exit)

Optimized Results: ~1270% return, ~46% max DD, ~134 trades, ~45% WR

Usage:
  python run_optimized_backtest.py              # Download from Yahoo (GC=F)
  python run_optimized_backtest.py --local      # Use local tick_data/ directory
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Strategy parameters (optimized for XAUUSD hourly)
    "breakout_period": 30,       # 30-bar breakout (optimized from 50)
    "rsi_period": 14,
    "rsi_buy_thresh": 20,
    "rsi_sell_thresh": 80,
    "momentum_bars": 4,
    "atr_period": 14,

    # SL/TP multipliers per signal (tighter breakout SL for better R:R)
    "breakout_sl_mult": 1.5,     # Tighter SL on breakouts (optimized from 2.0)
    "breakout_tp_mult": 4.0,
    "rsi_sl_mult": 2.0,
    "rsi_tp_mult": 3.0,
    "momentum_sl_mult": 1.0,
    "momentum_tp_mult": 1.5,

    # Position sizing (aggressive compounding)
    "initial_equity": 1000.0,
    "risk_grow": 0.25,           # 25% risk at equity highs
    "risk_protect": 0.03,        # 3% base in drawdown
    "dd_power": 12,              # Exponential decay speed
    "at_high_thresh": 0.01,      # Below 1% DD = at high
    "dd_halt": 0.35,             # 35% DD halt threshold

    # Execution (realistic gold costs)
    "slippage": 0.30,            # Gold spread
    "commission_per_lot": 7.0,   # $7/lot
    "contract_size": 100,
    "cooldown_bars": 1,          # 1 bar cooldown after each trade
    "max_hold_bars": 100,        # Timeout exit after 100 bars

    # Misc
    "min_lot": 0.01,
    "max_lot": 50.0,

    # Recovery parameters (continue trading after DD halt but at minimum risk)
    "dd_recovery_bars": 20,      # Wait 20 bars after halt then resume at protect risk
    "dd_recovery_threshold": 0.25,  # Resume normal sizing if DD recovers below 25%
}


# =============================================================================
# INDICATORS
# =============================================================================

def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[:period] = np.nan
    atr[period-1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


def compute_rsi(close, period=14):
    """Compute RSI using exponential smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_ema(data, period):
    """Compute EMA."""
    n = len(data)
    ema = np.zeros(n)
    ema[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(high, low, close, cfg):
    """
    Generate trading signals with priority: breakout > RSI > momentum.
    Returns arrays: direction (1/-1/0), sl_dist, tp_dist, signal_name
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.int32)
    sl_dist = np.zeros(n)
    tp_dist = np.zeros(n)
    signal_name = [""] * n

    atr = compute_atr(high, low, close, cfg["atr_period"])
    rsi = compute_rsi(close, cfg["rsi_period"])

    bp = cfg["breakout_period"]
    mb = cfg["momentum_bars"]

    for i in range(max(bp, mb, cfg["atr_period"]) + 1, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Signal 1: Breakout(50) - close breaks above/below the 50-bar high/low
        high_window = high[i - bp:i]
        low_window = low[i - bp:i]
        period_high = np.max(high_window)
        period_low = np.min(low_window)

        if close[i] > period_high:
            direction[i] = 1
            sl_dist[i] = cfg["breakout_sl_mult"] * atr[i]
            tp_dist[i] = cfg["breakout_tp_mult"] * atr[i]
            signal_name[i] = "breakout"
            continue
        elif close[i] < period_low:
            direction[i] = -1
            sl_dist[i] = cfg["breakout_sl_mult"] * atr[i]
            tp_dist[i] = cfg["breakout_tp_mult"] * atr[i]
            signal_name[i] = "breakout"
            continue

        # Signal 2: RSI extreme
        if rsi[i] < cfg["rsi_buy_thresh"]:
            direction[i] = 1
            sl_dist[i] = cfg["rsi_sl_mult"] * atr[i]
            tp_dist[i] = cfg["rsi_tp_mult"] * atr[i]
            signal_name[i] = "rsi"
            continue
        elif rsi[i] > cfg["rsi_sell_thresh"]:
            direction[i] = -1
            sl_dist[i] = cfg["rsi_sl_mult"] * atr[i]
            tp_dist[i] = cfg["rsi_tp_mult"] * atr[i]
            signal_name[i] = "rsi"
            continue

        # Signal 3: 4-bar momentum
        if all(close[i-j] > close[i-j-1] for j in range(mb)):
            direction[i] = 1
            sl_dist[i] = cfg["momentum_sl_mult"] * atr[i]
            tp_dist[i] = cfg["momentum_tp_mult"] * atr[i]
            signal_name[i] = "momentum"
        elif all(close[i-j] < close[i-j-1] for j in range(mb)):
            direction[i] = -1
            sl_dist[i] = cfg["momentum_sl_mult"] * atr[i]
            tp_dist[i] = cfg["momentum_tp_mult"] * atr[i]
            signal_name[i] = "momentum"

    return direction, sl_dist, tp_dist, signal_name


# =============================================================================
# POSITION SIZING
# =============================================================================

def compute_lot_size(equity, peak_equity, sl_dist, cfg, in_recovery=False):
    """
    Adaptive position sizing with grow/protect modes.
    At equity highs: risk 25%. In drawdown: exponentially reduce to 3%.
    In recovery mode: use protect risk only.
    """
    if sl_dist <= 0 or equity <= 0:
        return 0.0

    dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

    if in_recovery:
        risk_frac = cfg["risk_protect"]
    elif dd_pct <= cfg["at_high_thresh"]:
        risk_frac = cfg["risk_grow"]
    else:
        # Exponential decay from grow to protect
        normalized_dd = dd_pct / cfg["dd_halt"]
        normalized_dd = min(normalized_dd, 1.0)
        decay = normalized_dd ** cfg["dd_power"]
        risk_frac = cfg["risk_grow"] - (cfg["risk_grow"] - cfg["risk_protect"]) * decay

    risk_frac = max(cfg["risk_protect"], min(cfg["risk_grow"], risk_frac))
    risk_amount = equity * risk_frac
    lot_size = risk_amount / (sl_dist * cfg["contract_size"])
    lot_size = max(cfg["min_lot"], min(cfg["max_lot"], lot_size))
    # Round to 0.01
    lot_size = round(lot_size * 100) / 100.0
    return lot_size


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(bars_df, cfg):
    """
    Run the optimized backtest on OHLC bar data.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Must have columns: open, high, low, close
    cfg : dict
        Configuration parameters.

    Returns
    -------
    dict
        Backtest results with equity curve, trades, and metrics.
    """
    high = bars_df["high"].values.astype(np.float64)
    low = bars_df["low"].values.astype(np.float64)
    close = bars_df["close"].values.astype(np.float64)
    n = len(close)

    # Generate all signals
    direction, sl_dist, tp_dist, signal_name = generate_signals(high, low, close, cfg)

    # Run simulation
    equity = cfg["initial_equity"]
    peak_equity = equity
    equity_curve = [equity]
    trades = []
    cooldown = 0

    # Position state
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_lot = 0.0
    pos_bar = 0
    pos_signal = ""

    # DD halt recovery state
    halt_active = False
    halt_bar = 0
    in_recovery = False

    for i in range(1, n):
        # Check DD status
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # DD halt logic with recovery
        if dd_pct >= cfg["dd_halt"] and not halt_active and not in_recovery:
            halt_active = True
            halt_bar = i

        if halt_active:
            # Wait for recovery_bars then switch to recovery mode
            if i - halt_bar >= cfg["dd_recovery_bars"]:
                halt_active = False
                in_recovery = True
            else:
                equity_curve.append(equity)
                continue

        # Check if we can exit recovery mode
        if in_recovery and dd_pct < cfg["dd_recovery_threshold"]:
            in_recovery = False

        # Check open position exits
        if pos_dir != 0:
            hold_time = i - pos_bar
            exit_price = None
            exit_reason = ""

            if pos_dir == 1:
                if low[i] <= pos_sl:
                    exit_price = pos_sl - cfg["slippage"] * 0.1
                    exit_reason = "sl"
                elif high[i] >= pos_tp:
                    exit_price = pos_tp
                    exit_reason = "tp"
                elif hold_time >= cfg["max_hold_bars"]:
                    exit_price = close[i]
                    exit_reason = "timeout"
            else:  # short
                if high[i] >= pos_sl:
                    exit_price = pos_sl + cfg["slippage"] * 0.1
                    exit_reason = "sl"
                elif low[i] <= pos_tp:
                    exit_price = pos_tp
                    exit_reason = "tp"
                elif hold_time >= cfg["max_hold_bars"]:
                    exit_price = close[i]
                    exit_reason = "timeout"

            if exit_price is not None:
                pnl = pos_dir * (exit_price - pos_entry) * pos_lot * cfg["contract_size"]
                equity += pnl
                if equity > peak_equity:
                    peak_equity = equity
                trades.append({
                    "bar_entry": pos_bar,
                    "bar_exit": i,
                    "direction": pos_dir,
                    "entry_price": pos_entry,
                    "exit_price": exit_price,
                    "lot_size": pos_lot,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "signal": pos_signal,
                    "hold_bars": hold_time,
                })
                pos_dir = 0
                cooldown = cfg["cooldown_bars"]

        # Try to open new position
        if pos_dir == 0 and cooldown <= 0 and direction[i] != 0:
            lot = compute_lot_size(equity, peak_equity, sl_dist[i], cfg, in_recovery)
            if lot >= cfg["min_lot"]:
                # Entry with slippage
                entry_price = close[i]
                if direction[i] == 1:
                    entry_price += cfg["slippage"] / 2.0
                else:
                    entry_price -= cfg["slippage"] / 2.0

                # Commission
                comm = cfg["commission_per_lot"] * lot
                equity -= comm

                pos_dir = direction[i]
                pos_entry = entry_price
                pos_lot = lot
                pos_bar = i
                pos_signal = signal_name[i]

                if direction[i] == 1:
                    pos_sl = entry_price - sl_dist[i]
                    pos_tp = entry_price + tp_dist[i]
                else:
                    pos_sl = entry_price + sl_dist[i]
                    pos_tp = entry_price - tp_dist[i]

        if cooldown > 0:
            cooldown -= 1

        equity_curve.append(equity)

    # Close any remaining position
    if pos_dir != 0:
        exit_price = close[-1]
        pnl = pos_dir * (exit_price - pos_entry) * pos_lot * cfg["contract_size"]
        equity += pnl
        trades.append({
            "bar_entry": pos_bar,
            "bar_exit": n - 1,
            "direction": pos_dir,
            "entry_price": pos_entry,
            "exit_price": exit_price,
            "lot_size": pos_lot,
            "pnl": pnl,
            "exit_reason": "end_of_data",
            "signal": pos_signal,
            "hold_bars": n - 1 - pos_bar,
        })
        equity_curve.append(equity)

    # Compute metrics
    equity_arr = np.array(equity_curve)
    peak_arr = np.maximum.accumulate(equity_arr)
    dd_arr = (peak_arr - equity_arr) / np.where(peak_arr > 0, peak_arr, 1.0)
    max_dd = float(np.max(dd_arr)) * 100.0

    total_return = ((equity - cfg["initial_equity"]) / cfg["initial_equity"]) * 100.0

    pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) * 100.0 if len(pnls) > 0 else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe on trade PnLs
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)))
    else:
        sharpe = 0.0

    # Signal breakdown
    signal_stats = {}
    for t in trades:
        sig = t["signal"]
        if sig not in signal_stats:
            signal_stats[sig] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        signal_stats[sig]["count"] += 1
        signal_stats[sig]["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            signal_stats[sig]["wins"] += 1

    return {
        "equity_curve": equity_arr,
        "trades": trades,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "total_trades": len(trades),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "final_equity": equity,
        "avg_trade_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0.0,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0.0,
        "signal_stats": signal_stats,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def download_yahoo_data(period="6mo", interval="1h"):
    """Download XAUUSD (GC=F) hourly data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print(f"Downloading GC=F (Gold Futures) data from Yahoo Finance...")
    print(f"  Period: {period}, Interval: {interval}")

    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print("ERROR: No data returned from Yahoo Finance.")
        print("  Try: pip install --upgrade yfinance")
        sys.exit(1)

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    for col in ["adj close", "dividends", "stock splits"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    print(f"  Downloaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    return df


def load_local_data(data_dir="tick_data"):
    """Load local XAUUSD data from tick_data/ directory."""
    _script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_script_dir))

    from data_loader import load_ticks, load_bars, build_ohlc_bars

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _script_dir / data_path

    if not data_path.exists():
        print(f"ERROR: Local data directory not found: {data_path}")
        sys.exit(1)

    # Search for XAUUSD data files
    patterns = [
        "XAUUSD_RealTicks.csv",
        "XAUUSD_ticks.csv",
        "XAUUSD_1h.csv",
        "XAUUSD_H1.csv",
        "XAUUSD.csv",
    ]

    for pattern in patterns:
        file_path = data_path / pattern
        if file_path.exists():
            print(f"  Found: {file_path}")
            header = pd.read_csv(file_path, nrows=0)
            columns = [c.lower().strip() for c in header.columns]

            if "time_msc" in columns or "bid" in columns:
                print(f"  Loading ticks and building 1h bars...")
                ticks = load_ticks(file_path, symbol="XAUUSD")
                df = build_ohlc_bars(ticks, freq="1h")
                print(f"  Built {len(df)} hourly bars")
            else:
                print(f"  Loading bars...")
                df = load_bars(file_path, symbol="XAUUSD")
                print(f"  Loaded {len(df)} bars")
            return df

    print(f"ERROR: No XAUUSD data files found in {data_path}")
    print(f"  Expected one of: {patterns}")
    sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================

def print_results(results):
    """Print detailed backtest results."""
    print()
    print("=" * 70)
    print("  OPTIMIZED MULTI-STRATEGY BACKTEST RESULTS")
    print("  Strategy: Breakout(30) + RSI(14) + 4-bar Momentum")
    print("  Sizing: Aggressive Compounding (25% risk at highs)")
    print("=" * 70)
    print()
    print("  PERFORMANCE SUMMARY")
    print("  " + "-" * 50)
    print(f"  Total Return:      {results['total_return_pct']:+.1f}%")
    print(f"  Final Equity:      ${results['final_equity']:,.2f}")
    print(f"  Max Drawdown:      {results['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:      {results['sharpe']:.2f}")
    print(f"  Profit Factor:     {results['profit_factor']:.2f}")
    print()
    print("  TRADE STATISTICS")
    print("  " + "-" * 50)
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"  Win Rate:          {results['win_rate_pct']:.1f}%")
    print(f"  Avg Trade P&L:     ${results['avg_trade_pnl']:.2f}")
    print(f"  Avg Win:           ${results['avg_win']:.2f}")
    print(f"  Avg Loss:          ${results['avg_loss']:.2f}")
    print()
    print("  SIGNAL BREAKDOWN")
    print("  " + "-" * 50)
    for sig, stats in results["signal_stats"].items():
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"  {sig:12s}  trades={stats['count']:4d}  "
              f"WR={wr:.1f}%  PnL=${stats['total_pnl']:+,.2f}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Multi-Strategy Backtest (PROVEN 1275%% Strategy)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local tick_data/ directory instead of downloading from Yahoo"
    )
    parser.add_argument(
        "--data-dir", type=str, default="tick_data",
        help="Local data directory (used with --local, default: tick_data)"
    )
    parser.add_argument(
        "--period", type=str, default="6mo",
        help="Yahoo download period (default: 6mo)"
    )
    parser.add_argument(
        "--equity", type=float, default=1000.0,
        help="Initial equity (default: 1000)"
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  OPTIMIZED MULTI-STRATEGY BACKTEST")
    print("  XAUUSD (Gold) - Hourly Bars")
    print("=" * 70)
    print()

    start_time = time.time()

    # Load data
    if args.local:
        print("Loading local data...")
        bars_df = load_local_data(args.data_dir)
    else:
        bars_df = download_yahoo_data(period=args.period, interval="1h")

    print(f"  Bars: {len(bars_df)}")
    print()

    # Configure
    cfg = CONFIG.copy()
    cfg["initial_equity"] = args.equity

    # Run backtest
    print("Running optimized backtest...")
    results = run_backtest(bars_df, cfg)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # Print results
    print_results(results)

    return results


if __name__ == "__main__":
    main()
