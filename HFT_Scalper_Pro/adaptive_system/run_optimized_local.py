#!/usr/bin/env python3
"""
Optimized Multi-Strategy Backtest - LOCAL MT5 Tick Data Version
================================================================
Uses data_loader.py to load local MT5 tick data and build 1h bars,
then runs the proven 1275% return strategy.

This is for users who want to test on their own MT5 data without Yahoo.
Place tick data files in the tick_data/ directory with names like:
  - XAUUSD_RealTicks.csv
  - XAUUSD_ticks.csv

Usage:
  python run_optimized_local.py
  python run_optimized_local.py --data-dir /path/to/ticks
  python run_optimized_local.py --timeframe 1h
  python run_optimized_local.py --file XAUUSD_RealTicks.csv
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure imports work from script directory
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from data_loader import load_ticks, load_bars, build_ohlc_bars, SYMBOL_METADATA
from run_optimized_backtest import CONFIG, run_backtest, print_results


def find_and_load_data(data_dir, timeframe="1h", specific_file=None):
    """
    Find and load XAUUSD tick/bar data from the specified directory.
    Builds bars at the specified timeframe from tick data.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing tick/bar CSV files.
    timeframe : str
        Target timeframe for bar construction (e.g., '1h', '30min', '15min').
    specific_file : str, optional
        Specific filename to load.

    Returns
    -------
    pd.DataFrame
        OHLC bars ready for backtesting.
    """
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _script_dir / data_path

    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        print(f"  Please create {data_path} and add your MT5 tick export files.")
        print(f"  Expected files: XAUUSD_RealTicks.csv or XAUUSD_ticks.csv")
        sys.exit(1)

    # If specific file given, use it directly
    if specific_file:
        file_path = data_path / specific_file
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        return _load_single_file(file_path, timeframe)

    # Search for data files in priority order
    search_patterns = [
        "XAUUSD_RealTicks.csv",
        "XAUUSD_ticks.csv",
        "XAUUSD_Ticks.csv",
        "xauusd_ticks.csv",
        f"XAUUSD_{timeframe}.csv",
        "XAUUSD_H1.csv",
        "XAUUSD_1h.csv",
        "XAUUSD.csv",
    ]

    for pattern in search_patterns:
        file_path = data_path / pattern
        if file_path.exists():
            return _load_single_file(file_path, timeframe)

    # Try any CSV file in the directory
    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        print(f"  No XAUUSD-named files found. Trying: {csv_files[0].name}")
        return _load_single_file(csv_files[0], timeframe)

    print(f"ERROR: No data files found in {data_path}")
    print(f"  Please add MT5 tick export CSV files to this directory.")
    print(f"  Expected format: time_msc, bid, ask, volume_real")
    sys.exit(1)


def _load_single_file(file_path, timeframe):
    """Load a single data file, detecting format automatically."""
    print(f"  Loading: {file_path.name}")

    # Detect format by reading header
    header = pd.read_csv(file_path, nrows=0)
    columns = [c.lower().strip() for c in header.columns]

    if "time_msc" in columns or "bid" in columns:
        # Tick data - load and build bars
        print(f"  Format: MT5 tick data")
        print(f"  Building {timeframe} bars from ticks...")
        ticks = load_ticks(file_path, symbol="XAUUSD")
        print(f"  Loaded {len(ticks)} ticks")
        bars = build_ohlc_bars(ticks, freq=timeframe)
        print(f"  Built {len(bars)} bars at {timeframe} timeframe")
        return bars
    else:
        # Bar data
        print(f"  Format: OHLC bars")
        bars = load_bars(file_path, symbol="XAUUSD")
        print(f"  Loaded {len(bars)} bars")
        return bars


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Backtest using LOCAL MT5 tick data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimized_local.py
  python run_optimized_local.py --data-dir /path/to/tick_data
  python run_optimized_local.py --timeframe 30min
  python run_optimized_local.py --file XAUUSD_RealTicks.csv
  python run_optimized_local.py --equity 5000
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, default="tick_data",
        help="Directory containing tick/bar data (default: tick_data/)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Specific data file to load (within data-dir)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        help="Target bar timeframe (default: 1h)"
    )
    parser.add_argument(
        "--equity", type=float, default=1000.0,
        help="Initial equity (default: 1000)"
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  OPTIMIZED MULTI-STRATEGY BACKTEST (LOCAL DATA)")
    print("  XAUUSD (Gold) - Using MT5 Tick Data")
    print("  Strategy: Breakout(30) + RSI(14) + 4-bar Momentum")
    print("=" * 70)
    print()

    start_time = time.time()

    # Load data
    print("Loading local MT5 data...")
    bars_df = find_and_load_data(
        args.data_dir,
        timeframe=args.timeframe,
        specific_file=args.file,
    )

    if len(bars_df) < 100:
        print(f"WARNING: Only {len(bars_df)} bars loaded. Need at least 100 for reliable results.")

    print(f"  Total bars: {len(bars_df)}")
    print(f"  Date range: {bars_df.index[0]} to {bars_df.index[-1]}")
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
