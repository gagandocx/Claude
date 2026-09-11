"""
CCT Rectangle Bot v1 - Main Entry Point (SUPERSEDED)

    The returns this script prints are not real. Do not trade them.

This backtester can see up to eight hours into the future, enters setups that had
already been stopped out, fabricates outcomes when exit data is missing, applies
compounding out of chronological order, and charges no transaction costs. Full
accounting in cct2/AUDIT.md.

Use the corrected engine instead:

    python -m cct2.run selftest
    python -m cct2.run audit
    python -m cct2.run backtest --csv your_m1_data.csv

Retained so `cct2/audit.py` has something to reproduce, and for git history.

Usage:
    python main.py
"""

import sys
import warnings
from datetime import datetime

# Suppress pandas/yfinance warnings for cleaner output
warnings.filterwarnings("ignore")

import config
from data_loader import load_multi_timeframe_data, get_overlapping_period
from backtester import BacktestEngine


def print_banner():
    """Print the bot banner."""
    print("\n" + "!" * 60)
    print("  WARNING: v1 backtester. Its results are a measurement artefact")
    print("  (look-ahead, no setup invalidation, fabricated fills, zero costs).")
    print("  See cct2/AUDIT.md. Use `python -m cct2.run backtest` instead.")
    print("!" * 60)
    print("\n" + "=" * 60)
    print("    CCT RECTANGLE BOT - AGGRESSIVE MODE (SUPERSEDED)")
    print("    Maximum Frequency & Profitability Configuration")
    print("=" * 60)
    print(f"\n  Symbol:          {config.SYMBOL}")
    print(f"  Direction TF:    {config.TF_DIRECTION}")
    print(f"  Weakness TF:     {config.TF_WEAKNESS}")
    print(f"  Entry TF:        {config.TF_ENTRY}")
    print(f"  EMA Filter:      {'ON' if config.USE_EMA_FILTER else 'OFF (more signals)'}")
    print(f"  Min RR Ratio:    {config.MIN_RR_RATIO}:1")
    print(f"  Risk Per Trade:  {config.RISK_PER_TRADE * 100:.0f}%")
    print(f"  Leverage:        {config.LEVERAGE}x")
    print(f"  Compounding:     {'ON' if config.COMPOUNDING else 'OFF'}")
    print(f"  Max Concurrent:  {config.MAX_CONCURRENT_TRADES}")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Run Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def run_backtest():
    """Run the CCT Rectangle strategy backtest in aggressive mode."""
    print_banner()

    # Step 1: Load data
    print("Loading market data...")
    data = load_multi_timeframe_data(config.SYMBOL)

    # Validate data
    df_4h = data.get("4h")
    df_15m = data.get("15m")
    df_1m = data.get("1m")

    if df_4h is None or df_4h.empty:
        print("ERROR: No 4H data available. Cannot run backtest.")
        sys.exit(1)

    if df_15m is None or df_15m.empty:
        print("ERROR: No 15M data available. Cannot run backtest.")
        sys.exit(1)

    if df_1m is None or df_1m.empty:
        print("WARNING: No 1M data available. Using 15M for entry simulation.")
        df_1m = df_15m.copy()

    # Find overlapping period
    overlap_start, overlap_end = get_overlapping_period(data)
    if overlap_start is None:
        print("WARNING: No overlapping period across all timeframes.")
        print("  Strategy will use 15M as fallback for entry when 1M unavailable.")
    else:
        print(f"\n1M data available from: {overlap_start} to {overlap_end}")

    print(f"\nData loaded successfully:")
    print(f"  4H candles:  {len(df_4h)}")
    print(f"  15M candles: {len(df_15m)}")
    print(f"  1M candles:  {len(df_1m)}")

    # Step 2: Run backtest
    engine = BacktestEngine(
        df_4h=df_4h,
        df_15m=df_15m,
        df_1m=df_1m,
        initial_capital=config.INITIAL_CAPITAL,
        risk_per_trade=config.RISK_PER_TRADE,
    )

    stats = engine.run()

    # Step 3: Print results
    engine.print_results()

    # Strategy configuration summary
    print("\n" + "-" * 60)
    print("AGGRESSIVE MODE SETTINGS:")
    print("-" * 60)
    print(f"  [x] 4H Direction ({'strict engulfing' if config.REQUIRE_FULL_ENGULF else 'partial sweeps allowed'})")
    print(f"  [x] EMA Filter: {'ON (trend alignment)' if config.USE_EMA_FILTER else 'OFF (both directions)'}")
    print(f"  [x] Continuation Filter: {'ON' if config.CONTINUATION_ONLY else 'OFF'}")
    print(f"  [x] 15M Weakness (up to {config.MAX_WEAKNESS_PER_DIRECTION} per direction, {config.WEAKNESS_WINDOW_HOURS}h window)")
    print(f"  [x] Sweep minimum: {config.SWEEP_MIN_PIPS}")
    print(f"  [x] Rectangle Entry ({config.MAX_CANDLES_FOR_ENTRY}min window)")
    print(f"  [x] Min RR: {config.MIN_RR_RATIO}:1")
    print(f"  [x] Trailing Stop: {'ON' if config.USE_TRAILING_STOP else 'OFF'} (activate at {config.TRAILING_STOP_ACTIVATION_RR}R, trail {config.TRAILING_STOP_DISTANCE_RR}R)")
    print(f"  [x] Risk: {config.RISK_PER_TRADE*100:.0f}% per trade")
    print(f"  [x] Compounding: {'ENABLED' if config.COMPOUNDING else 'DISABLED'}")
    print(f"  [x] Max {config.MAX_CONCURRENT_TRADES} concurrent positions")
    print("-" * 60)

    # Monthly performance highlight
    if stats.monthly_return_pct > 0:
        print(f"\n  MONTHLY RETURN: {stats.monthly_return_pct:+.1f}%")
        print(f"  TOTAL RETURN:   {stats.total_return_pct:+.1f}% over {stats.total_days} days")
        if stats.monthly_return_pct >= 500:
            print(f"\n  ** TARGET OF 500%/month ACHIEVED! **")
        elif stats.total_return_pct >= 500:
            print(f"\n  ** 500%+ TOTAL RETURN ACHIEVED! **")
    print("-" * 60)

    return stats


if __name__ == "__main__":
    try:
        stats = run_backtest()
        print("\nBacktest completed successfully.")
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
