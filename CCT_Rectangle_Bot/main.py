"""
CCT Rectangle Bot - Main Entry Point

Runs the full CCT Rectangle strategy backtest:
1. Fetches multi-timeframe data (4H, 15M, 1M) via yfinance
2. Applies the CCT Rectangle strategy (direction -> weakness -> entry)
3. Simulates trades with proper SL/TP management
4. Outputs formatted trade results and performance statistics

Strategy: CCT (Candle Continuity Theory) Rectangle Setup
- 4H: Direction candle detection with EMA filter
- 15M: Weakness detection (wick rejection at key levels)
- 1M: Rectangle entry (close outside rectangle zone)

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
    print("\n" + "=" * 60)
    print("    CCT RECTANGLE BOT")
    print("    Candle Continuity Theory - Rectangle Scalping Strategy")
    print("=" * 60)
    print(f"\n  Symbol:          {config.SYMBOL}")
    print(f"  Direction TF:    {config.TF_DIRECTION}")
    print(f"  Weakness TF:     {config.TF_WEAKNESS}")
    print(f"  Entry TF:        {config.TF_ENTRY}")
    print(f"  EMA Fast:        {config.EMA_FAST}")
    print(f"  EMA Slow:        {config.EMA_SLOW}")
    print(f"  Min RR Ratio:    {config.MIN_RR_RATIO}:1")
    print(f"  Risk Per Trade:  {config.RISK_PER_TRADE * 100:.1f}%")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Run Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def run_backtest():
    """Run the CCT Rectangle strategy backtest."""
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
    
    # Find overlapping period for information
    overlap_start, overlap_end = get_overlapping_period(data)
    if overlap_start is None:
        print("WARNING: No overlapping period across all timeframes.")
        print("  Strategy will use 15M as fallback for entry when 1M unavailable.")
    else:
        print(f"\n1M data available from: {overlap_start} to {overlap_end}")
        print("  For earlier signals, 15M will be used for entry detection.")
    
    # Use full 15M range for the backtest (not restricted to 1M overlap)
    # The strategy handles fallback to 15M when 1M is not available
    
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
    
    # Final summary
    print("\n" + "-" * 60)
    print("STRATEGY RULES APPLIED:")
    print("-" * 60)
    print("  [x] 4H Direction Candle (CCT engulfing with sweep)")
    print("  [x] EMA Directional Filter (50/200 EMA alignment)")
    print("  [x] 15M Weakness Detection (wick rejection)")
    print("  [x] Fair Value Gap Filter (imbalance check)")
    print("  [x] Session Extreme Filter (Asia/London/NY)")
    print("  [x] 1M Rectangle Entry (close outside rectangle)")
    print("  [x] Stop Loss beyond rectangle extreme")
    print("  [x] Take Profit at 3:1+ RR ratio")
    print("  [x] Risk management (1% per trade)")
    print("  [x] Continuation only (trade with trend)")
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
