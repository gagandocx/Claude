#!/usr/bin/env python3
"""
Production Backtest Script
============================
Loads real tick/bar data from user-specified paths, runs the multi-symbol
adaptive backtest, and saves results to JSON.

Usage:
    python run_backtest.py --data-dir ./tick_data --symbols XAUUSD,EURUSD --timeframe 5min
    python run_backtest.py --data-dir ./tick_data --symbols XAUUSD --output results.json
    python run_backtest.py --config config.json --data-dir ./data
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Handle imports regardless of where the script is run from
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent))

from data_loader import load_ticks, load_bars, build_ohlc_bars, SYMBOL_METADATA
from backtest_engine import MultiSymbolBacktester, BacktestResult
from config import SystemConfig, load_config, get_balanced_config


def find_data_files(data_dir: Path, symbols: list, timeframe: str) -> Dict[str, pd.DataFrame]:
    """
    Find and load data files for the specified symbols.

    Searches for files matching common naming patterns:
        - {SYMBOL}_ticks.csv (tick data)
        - {SYMBOL}_RealTicks.csv (MT5 tick export)
        - {SYMBOL}_{timeframe}.csv (bar data)
        - {SYMBOL}.csv (generic)

    Parameters
    ----------
    data_dir : Path
        Directory containing data files.
    symbols : list
        Symbols to load.
    timeframe : str
        Target timeframe for bar construction.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Loaded bar data per symbol.
    """
    data_dict = {}

    for symbol in symbols:
        df = None

        # Try various file patterns
        patterns = [
            f"{symbol}_RealTicks.csv",
            f"{symbol}_ticks.csv",
            f"{symbol}_Ticks.csv",
            f"{symbol}_{timeframe}.csv",
            f"{symbol}.csv",
            f"{symbol.lower()}_ticks.csv",
            f"{symbol.lower()}.csv",
        ]

        for pattern in patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                print(f"  Found: {file_path}")

                # Detect if tick or bar data
                header = pd.read_csv(file_path, nrows=0)
                columns = [c.lower().strip() for c in header.columns]

                if "time_msc" in columns or "bid" in columns:
                    # Tick data - load and build bars
                    print(f"    Loading ticks and building {timeframe} bars...")
                    ticks = load_ticks(file_path, symbol=symbol)
                    df = build_ohlc_bars(ticks, freq=timeframe)
                    print(f"    Built {len(df)} bars")
                else:
                    # Bar data
                    print(f"    Loading bars...")
                    df = load_bars(file_path, symbol=symbol)
                    print(f"    Loaded {len(df)} bars")

                break

        if df is not None and len(df) > 0:
            data_dict[symbol] = df
        else:
            print(f"  WARNING: No data found for {symbol} in {data_dir}")

    return data_dict


def save_results(result: BacktestResult, output_path: Path):
    """Save backtest results to JSON."""
    output = {
        "summary": {
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "profit_factor": result.profit_factor,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "avg_trade_pnl": result.avg_trade_pnl,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "max_consec_wins": result.max_consec_wins,
            "max_consec_losses": result.max_consec_losses,
        },
        "regime_breakdown": result.regime_breakdown,
        "strategy_breakdown": result.strategy_breakdown,
        "symbol_breakdown": result.symbol_breakdown,
        "regime_summary": result.regime_summary,
        "strategy_usage": result.strategy_usage,
        "trades": [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(t.exit_price, 5),
                "lot_size": t.lot_size,
                "pnl": round(t.pnl, 2),
                "pnl_pips": round(t.pnl_pips, 1),
                "regime": t.regime,
                "strategy": t.strategy,
                "exit_reason": t.exit_reason,
                "duration_bars": t.duration_bars,
            }
            for t in result.trades
        ],
        "equity_curve": [round(e, 2) for e in result.portfolio_equity.tolist()]
        if len(result.portfolio_equity) > 0 else [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Currency System - Production Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py --data-dir ./tick_data --symbols XAUUSD,EURUSD
    python run_backtest.py --data-dir ./data --symbols XAUUSD --timeframe 5min
    python run_backtest.py --data-dir ./data --symbols XAUUSD,EURUSD,GBPJPY --output results.json
    python run_backtest.py --config config.json --data-dir ./data --symbols XAUUSD
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing tick/bar CSV files"
    )
    parser.add_argument(
        "--symbols", type=str, default="XAUUSD",
        help="Comma-separated list of symbols (default: XAUUSD)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1min",
        help="Target bar timeframe if building from ticks (default: 1min)"
    )
    parser.add_argument(
        "--output", type=str, default="backtest_results.json",
        help="Output JSON file path (default: backtest_results.json)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to system config JSON file"
    )
    parser.add_argument(
        "--initial-equity", type=float, default=1000.0,
        help="Starting equity (default: 1000)"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Enable walk-forward mode"
    )
    parser.add_argument(
        "--wf-windows", type=int, default=4,
        help="Number of walk-forward windows (default: 4)"
    )

    args = parser.parse_args()

    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(",")]
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_balanced_config()

    print("=" * 60)
    print("  ADAPTIVE MULTI-CURRENCY BACKTEST")
    print("=" * 60)
    print(f"  Data dir:    {data_dir}")
    print(f"  Symbols:     {', '.join(symbols)}")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Equity:      ${args.initial_equity:,.2f}")
    print(f"  Walk-fwd:    {'Yes' if args.walk_forward else 'No'}")
    print()

    # Load data
    print("Loading data...")
    start_time = time.time()
    data_dict = find_data_files(data_dir, symbols, args.timeframe)

    if not data_dict:
        print("\nERROR: No data files found. Check --data-dir and --symbols.")
        print(f"  Searched in: {data_dir}")
        print(f"  For symbols: {symbols}")
        sys.exit(1)

    load_time = time.time() - start_time
    print(f"\nData loaded in {load_time:.1f}s")
    for sym, df in data_dict.items():
        print(f"  {sym}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    print()

    # Run backtest
    print("Running backtest...")
    bt_start = time.time()

    backtester = MultiSymbolBacktester()
    bt_config = {
        "initial_equity": args.initial_equity,
        "commission_per_lot": 7.0,
        "slippage_pips": 1.0,
        "min_bars": 150,
    }

    if args.walk_forward:
        result = backtester.run_walk_forward(
            data_dict, n_windows=args.wf_windows, config=bt_config
        )
    else:
        result = backtester.run(data_dict, bt_config)

    bt_time = time.time() - bt_start
    print(f"Backtest completed in {bt_time:.1f}s")
    print()

    # Print summary
    print("RESULTS:")
    print("-" * 40)
    print(f"  Total Return:    {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Win Rate:        {result.win_rate*100:.1f}%")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Avg Trade P&L:   ${result.avg_trade_pnl:.2f}")

    # Save results
    save_results(result, output_path)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
