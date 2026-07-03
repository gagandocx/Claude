"""
Master Backtest Runner.

Runs all strategies with walk-forward optimization,
compares results, and identifies the winning strategy.
Saves comprehensive results to hft_scalper/results/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_scalper.data_loader import load_ticks, build_ohlc_bars
from hft_scalper.backtest_engine import run_backtest, BacktestConfig, BacktestResult
from hft_scalper.optimizer import quick_optimize
from hft_scalper.strategies.mean_reversion import MeanReversionStrategy
from hft_scalper.strategies.order_flow import OrderFlowStrategy
from hft_scalper.strategies.spread_fade import SpreadFadeStrategy
from hft_scalper.strategies.momentum_mtf import MomentumMTFStrategy
from hft_scalper.strategies.volatility_breakout import VolatilityBreakoutStrategy


RESULTS_DIR = Path(__file__).parent / "results"


def result_to_dict(result: BacktestResult) -> dict:
    """Convert BacktestResult to JSON-serializable dict."""
    return {
        "total_pnl": round(result.total_pnl, 2),
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": round(result.win_rate * 100, 2),
        "profit_factor": round(result.profit_factor, 3),
        "max_drawdown": round(result.max_drawdown, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 3),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "sortino_ratio": round(result.sortino_ratio, 3),
        "avg_trade_pnl": round(result.avg_trade_pnl, 2),
        "avg_winner": round(result.avg_winner, 2),
        "avg_loser": round(result.avg_loser, 2),
        "max_consecutive_losses": result.max_consecutive_losses,
        "avg_trade_duration_bars": round(result.avg_trade_duration, 1),
        "calmar_ratio": round(result.calmar_ratio, 3),
        "expectancy": round(result.expectancy, 2),
    }


def run_all_strategies():
    """Run all strategies with optimization and compare."""
    start_time = time.time()
    print("=" * 70)
    print("HFT SCALPING EA - BACKTEST RUNNER")
    print("XAUUSD 1-Minute Bars | Walk-Forward Optimization")
    print("Account: $1,000 | Leverage: 1:500 | Lot Size: 0.1")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading tick data and building bars...")
    ticks = load_ticks()
    bars = build_ohlc_bars(ticks, freq="1min")
    del ticks  # Free memory

    print(f"Total bars: {len(bars)}")
    print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
    print(f"Price range: {bars['close'].min():.2f} - {bars['close'].max():.2f}")

    # Backtest config
    # User specs: $1000 initial deposit, 1:500 leverage, 0.1 lot size
    config = BacktestConfig(
        slippage_points=0.3,
        commission_per_lot_rt=0.7,  # $0.7 for 0.1 lot (scaled from $7/lot)
        lot_size=0.1,  # 0.1 standard lot
        contract_size=100.0,  # 100 oz per standard lot (so 10 oz for 0.1 lot)
        initial_equity=1000.0,  # $1000 initial deposit
        leverage=500,  # 1:500
    )

    # Strategy classes to test
    strategies = [
        ("MeanReversion", MeanReversionStrategy),
        ("OrderFlow", OrderFlowStrategy),
        ("SpreadFade", SpreadFadeStrategy),
        ("MomentumMTF", MomentumMTFStrategy),
        ("VolBreakout", VolatilityBreakoutStrategy),
    ]

    # Run optimization and backtest for each strategy
    print("\n[2/4] Optimizing and backtesting strategies...")
    print("-" * 70)

    all_results = {}

    for name, strat_class in strategies:
        print(f"\n>>> Strategy: {name}")
        try:
            best_params, train_result, val_result = quick_optimize(
                strat_class, bars, config=config, metric="total_pnl"
            )

            # Run on full dataset with best params
            full_strategy = strat_class(best_params)
            full_signals = full_strategy.generate_signals(bars)
            full_result = run_backtest(bars, full_signals, config)

            all_results[name] = {
                "params": best_params,
                "train": result_to_dict(train_result),
                "validation": result_to_dict(val_result),
                "full": result_to_dict(full_result),
                "equity_curve": full_result.equity_curve.tolist(),
                "trade_count_full": full_result.total_trades,
            }

            print(f"    Full PnL: ${full_result.total_pnl:,.2f} | "
                  f"Trades: {full_result.total_trades} | "
                  f"Win Rate: {full_result.win_rate*100:.1f}% | "
                  f"Sharpe: {full_result.sharpe_ratio:.2f} | "
                  f"Max DD: {full_result.max_drawdown_pct:.2f}%")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[name] = {"error": str(e)}

    # Compare and find winner
    print("\n[3/4] Comparing strategies...")
    print("=" * 70)

    valid_strategies = {k: v for k, v in all_results.items() if "error" not in v}

    if not valid_strategies:
        print("ERROR: No strategies produced valid results!")
        return

    # Rank by multiple criteria with weighted score
    ranking = []
    for name, data in valid_strategies.items():
        full = data["full"]
        val = data["validation"]

        # Skip strategies with 0 trades
        if full["total_trades"] == 0:
            continue

        # Composite score: user wants MAXIMUM PnL with less drawdown
        # Primary: PnL, Secondary: risk-adjusted (Sharpe), Penalty: extreme drawdown
        pnl_score = full["total_pnl"] / 100.0  # Primary weight on PnL
        sharpe_score = full["sharpe_ratio"] * 2.0
        pf_score = max(full["profit_factor"] - 1.0, 0) * 5.0  # Reward PF > 1
        val_score = max(val["total_pnl"], 0) / 200.0  # Bonus for positive OOS
        dd_penalty = max(-full["max_drawdown_pct"] - 30, 0) * 2.0  # Penalize DD > 30%

        score = pnl_score + sharpe_score + pf_score + val_score - dd_penalty

        ranking.append((name, score, full, val))

    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Strategy':<20} {'PnL ($)':<15} {'Sharpe':<10} {'Win%':<10} {'MaxDD%':<10} {'PF':<10} {'Trades':<10}")
    print("-" * 85)
    for name, score, full, val in ranking:
        print(f"{name:<20} {full['total_pnl']:>12,.2f} {full['sharpe_ratio']:>8.2f} "
              f"{full['win_rate']:>8.1f} {full['max_drawdown_pct']:>8.2f} "
              f"{full['profit_factor']:>8.3f} {full['total_trades']:>8d}")

    winner_name = ranking[0][0]
    winner_full = ranking[0][2]
    winner_val = ranking[0][3]
    winner_params = valid_strategies[winner_name]["params"]

    print(f"\n{'='*70}")
    print(f"WINNER: {winner_name}")
    print(f"{'='*70}")
    print(f"  Total PnL:        ${winner_full['total_pnl']:,.2f}")
    print(f"  Max Drawdown:     {winner_full['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:     {winner_full['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:    {winner_full['sortino_ratio']:.3f}")
    print(f"  Win Rate:         {winner_full['win_rate']:.1f}%")
    print(f"  Profit Factor:    {winner_full['profit_factor']:.3f}")
    print(f"  Total Trades:     {winner_full['total_trades']}")
    print(f"  Avg Trade:        ${winner_full['avg_trade_pnl']:.2f}")
    print(f"  Avg Winner:       ${winner_full['avg_winner']:.2f}")
    print(f"  Avg Loser:        ${winner_full['avg_loser']:.2f}")
    print(f"  Max Consec Losses: {winner_full['max_consecutive_losses']}")
    print(f"  OOS Validation PnL: ${winner_val['total_pnl']:,.2f}")
    print(f"  Parameters: {winner_params}")

    # Save results
    print("\n[4/4] Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy comparison
    comparison = {
        "winner": winner_name,
        "winner_params": winner_params,
        "winner_metrics": winner_full,
        "winner_validation": winner_val,
        "all_strategies": {k: {kk: vv for kk, vv in v.items() if kk != "equity_curve"}
                          for k, v in all_results.items()},
        "backtest_config": {
            "slippage_points": config.slippage_points,
            "commission_per_lot_rt": config.commission_per_lot_rt,
            "lot_size": config.lot_size,
            "contract_size": config.contract_size,
            "initial_equity": config.initial_equity,
            "leverage": config.leverage,
        },
        "data_info": {
            "total_bars": len(bars),
            "date_start": str(bars.index[0]),
            "date_end": str(bars.index[-1]),
            "train_bars": int(len(bars) * 0.7),
            "validation_bars": int(len(bars) * 0.3),
        },
        "ranking": [
            {"strategy": name, "score": round(score, 3), "pnl": full["total_pnl"],
             "sharpe": full["sharpe_ratio"], "max_dd_pct": full["max_drawdown_pct"]}
            for name, score, full, val in ranking
        ],
    }

    with open(RESULTS_DIR / "strategy_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Save equity curves
    equity_data = {}
    for name, data in valid_strategies.items():
        if "equity_curve" in data:
            # Downsample for storage (every 100 bars)
            curve = data["equity_curve"]
            equity_data[name] = curve[::100]

    with open(RESULTS_DIR / "equity_curves.json", "w") as f:
        json.dump(equity_data, f)

    # Save trade log for winner
    winner_strategy = strategies[[s[0] for s in strategies].index(winner_name)][1]
    winner_strat_instance = winner_strategy(winner_params)
    winner_signals = winner_strat_instance.generate_signals(bars)
    winner_result = run_backtest(bars, winner_signals, config)

    trade_log = []
    for t in winner_result.trades[:500]:  # Save first 500 trades
        trade_log.append({
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_price": round(t.entry_price, 3),
            "exit_price": round(t.exit_price, 3),
            "pnl": round(t.pnl, 2),
            "pnl_points": round(t.pnl_points, 3),
            "exit_reason": t.exit_reason,
            "duration_bars": t.exit_bar - t.entry_bar,
        })

    with open(RESULTS_DIR / "winner_trade_log.json", "w") as f:
        json.dump({
            "strategy": winner_name,
            "total_trades": winner_result.total_trades,
            "trades": trade_log,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Total execution time: {elapsed:.1f}s")
    print(f"\nFiles saved:")
    print(f"  - strategy_comparison.json")
    print(f"  - equity_curves.json")
    print(f"  - winner_trade_log.json")

    return comparison


if __name__ == "__main__":
    run_all_strategies()
