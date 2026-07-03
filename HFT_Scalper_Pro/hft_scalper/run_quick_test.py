"""
Quick Test: Run proven winning parameters on tick data
======================================================
Skips parameter sweep entirely. Runs the best-known params on whatever
tick data is in tick_data/XAUUSD_RealTicks.csv and displays:
- Trade-by-trade results table
- Summary statistics
- Signal analysis (signal type breakdown)

Usage:
    python run_quick_test.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_ticks, build_ohlc_bars
from run_aggressive_backtest import compute_rsi, compute_atr, run_backtest, full_metrics

# Proven winning parameters
PARAMS = {
    "risk_grow": 0.17,
    "risk_protect": 0.025,
    "dd_power": 13,
    "sl_mult": 2.0,
    "tp_mult": 3.0,
    "at_high_thresh": 0.01,
    "loss_boost": 2.0,
    "win_reduce": 0.4,
}


def analyze_signals(close, high, low, hours, n, rsi_fast, rsi_slow, atr):
    """Analyze how many signals fired per type and direction."""
    WARMUP = 50
    signals = {
        "4bar_reversal": {"long": 0, "short": 0},
        "rsi_fast": {"long": 0, "short": 0},
        "rsi_slow": {"long": 0, "short": 0},
    }

    for i in range(WARMUP, n):
        if hours[i] < 7 or hours[i] > 20:
            continue
        if atr[i] < 1.0:
            continue

        # Slot 0: 4-bar reversal + RSI(8) extremes
        if i >= 4:
            all_down = all(close[i - j] < close[i - j - 1] for j in range(4))
            all_up = all(close[i - j] > close[i - j - 1] for j in range(4))
            if all_down:
                signals["4bar_reversal"]["long"] += 1
            elif all_up:
                signals["4bar_reversal"]["short"] += 1
            else:
                # Check RSI fast only if no 4-bar reversal
                if rsi_fast[i] < 25:
                    signals["rsi_fast"]["long"] += 1
                elif rsi_fast[i] > 75:
                    signals["rsi_fast"]["short"] += 1
        else:
            if rsi_fast[i] < 25:
                signals["rsi_fast"]["long"] += 1
            elif rsi_fast[i] > 75:
                signals["rsi_fast"]["short"] += 1

        # Slot 1: RSI(14) extremes
        if rsi_slow[i] < 30:
            signals["rsi_slow"]["long"] += 1
        elif rsi_slow[i] > 70:
            signals["rsi_slow"]["short"] += 1

    return signals


def format_trade_table(trades, bars_index):
    """Format trades into a nicely aligned table."""
    header = (
        f"{'#':>4}  {'Entry Time':<20} {'Exit Time':<20} {'Dir':<6} "
        f"{'Entry':>10} {'Exit':>10} {'Lots':>6} {'P&L':>10} {'Reason':<6}"
    )
    separator = "-" * len(header)

    rows = [header, separator]
    for idx, t in enumerate(trades, 1):
        entry_time = str(bars_index[t.entry_bar])[:19]
        exit_time = str(bars_index[t.exit_bar])[:19]
        direction = "LONG" if t.direction == 1 else "SHORT"
        pnl_str = f"${t.pnl:+.2f}"
        rows.append(
            f"{idx:>4}  {entry_time:<20} {exit_time:<20} {direction:<6} "
            f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.lot_size:>6.2f} "
            f"{pnl_str:>10} {t.reason:<6}"
        )

    return "\n".join(rows)


def main():
    print("=" * 70)
    print("QUICK TEST - Proven Winning Parameters")
    print("=" * 70)
    print("No parameter sweep. Running best-known params directly.")
    print()
    print("Parameters:")
    for k, v in PARAMS.items():
        print(f"  {k:<16} = {v}")
    print()

    # Load data
    tick_path = Path(__file__).parent / "tick_data" / "XAUUSD_RealTicks.csv"
    if not tick_path.exists():
        print(f"ERROR: Tick data not found at {tick_path}")
        print("Please place XAUUSD_RealTicks.csv in the tick_data/ directory.")
        sys.exit(1)

    ticks = load_ticks(tick_path)
    bars = build_ohlc_bars(ticks, freq="5min")
    print(f"\n5-min bars: {len(bars):,}")
    print(f"Period: {bars.index[0]} to {bars.index[-1]}")

    close = bars["close"].values.astype(np.float64)
    high = bars["high"].values.astype(np.float64)
    low = bars["low"].values.astype(np.float64)
    hours = bars.index.hour.values
    n = len(bars)

    # Precompute indicators
    print("\nPrecomputing indicators...")
    rsi_fast = compute_rsi(close, 8)
    rsi_slow = compute_rsi(close, 14)
    atr = compute_atr(high, low, close, 14)
    print("Done.")

    # Run backtest
    print("\nRunning backtest...")
    ret_pct, dd_pct, trades, final_equity = run_backtest(
        close, high, low, hours, n,
        rsi_fast, rsi_slow, atr, PARAMS
    )

    # ===== TRADE-BY-TRADE RESULTS =====
    print("\n" + "=" * 70)
    print("TRADE LOG")
    print("=" * 70)
    if trades:
        print(format_trade_table(trades, bars.index))
    else:
        print("No trades generated.")

    # ===== SUMMARY STATISTICS =====
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    metrics = full_metrics(trades)

    print(f"  Total Trades:      {metrics['total_trades']}")
    print(f"  Winning Trades:    {metrics.get('winning_trades', 0)}")
    print(f"  Losing Trades:     {metrics.get('losing_trades', 0)}")
    print(f"  Win Rate:          {metrics.get('win_rate', 0):.1%}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
    print(f"  Total Return:      {metrics.get('total_return_pct', 0):.1f}%")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Final Equity:      ${metrics.get('final_equity', 1000):.2f}")
    print(f"  Avg Trade P&L:     ${metrics.get('avg_trade_pnl', 0):.2f}")
    print(f"  Avg Winner:        ${metrics.get('avg_winner', 0):.2f}")
    print(f"  Avg Loser:         ${metrics.get('avg_loser', 0):.2f}")
    print(f"  Avg Lot Size:      {metrics.get('avg_lot_size', 0):.3f}")
    print(f"  Max Lot Size:      {metrics.get('max_lot_size', 0):.3f}")

    # ===== SIGNAL ANALYSIS =====
    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS")
    print("=" * 70)
    signals = analyze_signals(close, high, low, hours, n, rsi_fast, rsi_slow, atr)

    total_signals = 0
    print(f"  {'Signal Type':<20} {'Long':>8} {'Short':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for sig_type, counts in signals.items():
        sig_total = counts["long"] + counts["short"]
        total_signals += sig_total
        print(f"  {sig_type:<20} {counts['long']:>8} {counts['short']:>8} {sig_total:>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    total_long = sum(c["long"] for c in signals.values())
    total_short = sum(c["short"] for c in signals.values())
    print(f"  {'TOTAL':<20} {total_long:>8} {total_short:>8} {total_signals:>8}")

    # ===== SAVE RESULTS =====
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "quick_test_results.json"

    # Build trade log for JSON
    trade_log = []
    for t in trades:
        trade_log.append({
            "entry_time": str(bars.index[t.entry_bar]),
            "exit_time": str(bars.index[t.exit_bar]),
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "lot_size": t.lot_size,
            "pnl": round(t.pnl, 2),
            "reason": t.reason,
        })

    results = {
        "params": PARAMS,
        "summary": metrics,
        "signal_analysis": signals,
        "trade_log": trade_log,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
