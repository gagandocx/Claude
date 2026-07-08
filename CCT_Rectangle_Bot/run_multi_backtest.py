"""
CCT Rectangle Bot - Multi-Symbol Backtest Comparison

Tests the CCT Rectangle strategy across ALL major currency pairs and NAS100,
then outputs a comparison table sorted by total return.

Symbols tested:
- EURUSD=X (EUR/USD)
- GBPUSD=X (GBP/USD)
- USDJPY=X (USD/JPY)
- AUDUSD=X (AUD/USD)
- USDCAD=X (USD/CAD)
- NZDUSD=X (NZD/USD)
- USDCHF=X (USD/CHF)
- GC=F (Gold/XAUUSD)
- NQ=F (Nasdaq 100 / NAS100)

Usage:
    python run_multi_backtest.py
"""

import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import config
from data_loader import load_multi_timeframe_data
from backtester import BacktestEngine


# Symbol configurations with appropriate parameters
SYMBOL_CONFIGS = {
    "EURUSD=X": {
        "name": "EUR/USD",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "GBPUSD=X": {
        "name": "GBP/USD",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "USDJPY=X": {
        "name": "USD/JPY",
        "PIP_MULTIPLIER": 100,
        "SWEEP_MIN_PIPS": 0.010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.005,
        "FVG_MIN_SIZE_PIPS": 0.020,
        "SPREAD": 0.015,
    },
    "AUDUSD=X": {
        "name": "AUD/USD",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "USDCAD=X": {
        "name": "USD/CAD",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "NZDUSD=X": {
        "name": "NZD/USD",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "USDCHF=X": {
        "name": "USD/CHF",
        "PIP_MULTIPLIER": 10000,
        "SWEEP_MIN_PIPS": 0.00010,
        "MIN_RECTANGLE_SIZE_PIPS": 0.00005,
        "FVG_MIN_SIZE_PIPS": 0.00020,
        "SPREAD": 0.00015,
    },
    "GC=F": {
        "name": "Gold/XAUUSD",
        "PIP_MULTIPLIER": 10,
        "SWEEP_MIN_PIPS": 0.10,
        "MIN_RECTANGLE_SIZE_PIPS": 0.15,
        "FVG_MIN_SIZE_PIPS": 0.50,
        "SPREAD": 0.20,
    },
    "NQ=F": {
        "name": "NAS100",
        "PIP_MULTIPLIER": 1,
        "SWEEP_MIN_PIPS": 5.0,
        "MIN_RECTANGLE_SIZE_PIPS": 3.0,
        "FVG_MIN_SIZE_PIPS": 10.0,
        "SPREAD": 2.0,
    },
}


def set_config_for_symbol(symbol: str, cfg: dict):
    """Override config module attributes for a specific symbol."""
    config.SYMBOL = symbol
    config.PIP_MULTIPLIER = cfg["PIP_MULTIPLIER"]
    config.SWEEP_MIN_PIPS = cfg["SWEEP_MIN_PIPS"]
    config.MIN_RECTANGLE_SIZE_PIPS = cfg["MIN_RECTANGLE_SIZE_PIPS"]
    config.FVG_MIN_SIZE_PIPS = cfg["FVG_MIN_SIZE_PIPS"]
    config.SPREAD = cfg["SPREAD"]

    # Aggressive settings for all symbols
    config.RISK_PER_TRADE = 0.25
    config.COMPOUNDING = True
    config.USE_TRAILING_STOP = True
    config.LEVERAGE = 1
    config.INITIAL_CAPITAL = 10000.0
    config.MIN_RR_RATIO = 3.0
    config.USE_EMA_FILTER = True
    config.REQUIRE_FULL_ENGULF = True
    config.CONTINUATION_ONLY = True
    config.MAX_CONCURRENT_TRADES = 2
    config.TRAILING_STOP_ACTIVATION_RR = 3.0
    config.TRAILING_STOP_DISTANCE_RR = 2.5


def run_single_backtest(symbol: str, cfg: dict) -> dict:
    """
    Run backtest for a single symbol and return results dictionary.

    Returns dict with keys: symbol, name, total_trades, win_rate,
    total_return_pct, monthly_return_pct, profit_factor, max_drawdown_pct,
    best_trade_pips, error
    """
    result = {
        "symbol": symbol,
        "name": cfg["name"],
        "total_trades": 0,
        "win_rate": 0.0,
        "total_return_pct": 0.0,
        "monthly_return_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "best_trade_pips": 0.0,
        "error": None,
    }

    try:
        # Set config for this symbol
        set_config_for_symbol(symbol, cfg)

        # Load data
        data = load_multi_timeframe_data(symbol)

        df_4h = data.get("4h")
        df_15m = data.get("15m")
        df_1m = data.get("1m")

        if df_4h is None or df_4h.empty:
            result["error"] = "No 4H data available"
            return result

        if df_15m is None or df_15m.empty:
            result["error"] = "No 15M data available"
            return result

        if df_1m is None or df_1m.empty:
            # Use 15M as fallback for entry simulation
            df_1m = df_15m.copy()

        # Run backtest
        engine = BacktestEngine(
            df_4h=df_4h,
            df_15m=df_15m,
            df_1m=df_1m,
            initial_capital=config.INITIAL_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE,
        )

        stats = engine.run()

        # Extract results
        result["total_trades"] = stats.total_trades
        result["win_rate"] = stats.win_rate
        result["total_return_pct"] = stats.total_return_pct
        result["monthly_return_pct"] = stats.monthly_return_pct
        result["profit_factor"] = stats.profit_factor
        result["max_drawdown_pct"] = stats.max_drawdown_pct
        result["best_trade_pips"] = stats.best_trade_pips

    except Exception as e:
        result["error"] = str(e)

    return result


def print_comparison_table(results: list):
    """Print formatted comparison table sorted by total return %."""
    print("\n")
    print("=" * 110)
    print("               CCT RECTANGLE BOT - MULTI-SYMBOL BACKTEST COMPARISON")
    print("               Aggressive Mode: 25% Risk | Compounding ON | Trailing Stops ON")
    print("=" * 110)
    print(f"  Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Sort by total return descending
    valid_results = [r for r in results if r["error"] is None and r["total_trades"] > 0]
    failed_results = [r for r in results if r["error"] is not None or r["total_trades"] == 0]

    valid_results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    # Table header
    header = (
        f"{'#':<3} {'Symbol':<14} {'Name':<12} {'Trades':<8} "
        f"{'Win %':<8} {'Return %':<11} {'Monthly %':<11} "
        f"{'PF':<7} {'Max DD %':<10} {'Best (pips)':<12}"
    )
    print("-" * 110)
    print(header)
    print("-" * 110)

    # Table rows
    for i, r in enumerate(valid_results, 1):
        row = (
            f"{i:<3} {r['symbol']:<14} {r['name']:<12} {r['total_trades']:<8} "
            f"{r['win_rate']:<8.1f} {r['total_return_pct']:<+11.1f} "
            f"{r['monthly_return_pct']:<+11.1f} "
            f"{r['profit_factor']:<7.2f} {r['max_drawdown_pct']:<10.1f} "
            f"{r['best_trade_pips']:<+12.1f}"
        )
        print(row)

    # Show failed/no-trade symbols
    if failed_results:
        print("-" * 110)
        for r in failed_results:
            reason = r["error"] if r["error"] else "No trades generated"
            print(f"    {r['symbol']:<14} {r['name']:<12} -- {reason}")

    print("-" * 110)

    # Summary
    print()
    if valid_results:
        best = valid_results[0]
        total_symbols_traded = len(valid_results)
        avg_return = sum(r["total_return_pct"] for r in valid_results) / total_symbols_traded
        avg_win_rate = sum(r["win_rate"] for r in valid_results) / total_symbols_traded
        total_trades_all = sum(r["total_trades"] for r in valid_results)

        print("=" * 110)
        print("  SUMMARY")
        print("=" * 110)
        print(f"  Symbols Tested:       {len(results)}")
        print(f"  Symbols with Trades:  {total_symbols_traded}")
        print(f"  Total Trades (all):   {total_trades_all}")
        print(f"  Average Win Rate:     {avg_win_rate:.1f}%")
        print(f"  Average Return:       {avg_return:+.1f}%")
        print()
        print(f"  *** BEST PERFORMER: {best['symbol']} ({best['name']}) ***")
        print(f"      Total Return:   {best['total_return_pct']:+.1f}%")
        print(f"      Monthly Return: {best['monthly_return_pct']:+.1f}%")
        print(f"      Win Rate:       {best['win_rate']:.1f}%")
        print(f"      Profit Factor:  {best['profit_factor']:.2f}")
        print(f"      Total Trades:   {best['total_trades']}")
        print("=" * 110)
    else:
        print("  No valid results to compare.")
        print("=" * 110)


def main():
    """Run multi-symbol backtest and print comparison."""
    print("\n" + "=" * 70)
    print("  CCT RECTANGLE BOT - MULTI-SYMBOL BACKTEST")
    print("  Testing ALL major pairs + Gold + NAS100")
    print("=" * 70)
    print(f"\n  Symbols: {len(SYMBOL_CONFIGS)}")
    print(f"  Mode: AGGRESSIVE (25% risk, compounding, trailing stops)")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []

    for i, (symbol, cfg) in enumerate(SYMBOL_CONFIGS.items(), 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(SYMBOL_CONFIGS)}] Running backtest for {symbol} ({cfg['name']})")
        print(f"{'='*70}")

        result = run_single_backtest(symbol, cfg)
        results.append(result)

        # Print quick summary for this symbol
        if result["error"]:
            print(f"\n  RESULT: ERROR - {result['error']}")
        elif result["total_trades"] == 0:
            print(f"\n  RESULT: No trades generated")
        else:
            print(f"\n  RESULT: {result['total_trades']} trades | "
                  f"Win Rate: {result['win_rate']:.1f}% | "
                  f"Return: {result['total_return_pct']:+.1f}% | "
                  f"PF: {result['profit_factor']:.2f}")

    # Print final comparison table
    print_comparison_table(results)

    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
