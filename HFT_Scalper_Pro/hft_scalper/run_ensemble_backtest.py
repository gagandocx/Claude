"""
Ensemble Strategy Backtest Runner.

Runs the ensemble strategy with walk-forward optimization,
compares it against individual strategies (OrderFlow, MomentumMTF, SpreadFade),
and determines the final winning approach for the production EA.

Saves comprehensive results to hft_scalper/results/ensemble_results.json.
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
from hft_scalper.strategies.ensemble import EnsembleStrategy
from hft_scalper.strategies.order_flow import OrderFlowStrategy
from hft_scalper.strategies.momentum_mtf import MomentumMTFStrategy
from hft_scalper.strategies.spread_fade import SpreadFadeStrategy


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


def run_ensemble_comparison():
    """Run ensemble vs individual strategies comparison."""
    start_time = time.time()
    print("=" * 70)
    print("ENSEMBLE STRATEGY BACKTEST")
    print("OrderFlow + MomentumMTF + SpreadFade Consensus")
    print("Account: $1,000 | Leverage: 1:500 | Lot Size: 0.1")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading tick data and building 1-minute bars...")
    ticks = load_ticks()
    bars = build_ohlc_bars(ticks, freq="1min")
    del ticks  # Free memory

    n_bars = len(bars)
    print(f"Total bars: {n_bars}")
    print(f"Date range: {bars.index[0]} to {bars.index[-1]}")

    # Backtest config matching user requirements
    config = BacktestConfig(
        slippage_points=0.3,
        commission_per_lot_rt=0.7,
        lot_size=0.1,
        contract_size=100.0,
        initial_equity=1000.0,
        leverage=500,
    )

    # Train/validation split
    split_idx = int(n_bars * 0.7)
    train_bars = bars.iloc[:split_idx].copy()
    val_bars = bars.iloc[split_idx:].copy()
    print(f"Train: {len(train_bars)} bars | Validation: {len(val_bars)} bars")

    # ===== Run Ensemble with optimization =====
    print("\n[2/5] Optimizing Ensemble strategy...")
    print("-" * 70)

    ensemble_params, ensemble_train, ensemble_val = quick_optimize(
        EnsembleStrategy, bars, config=config, metric="total_pnl"
    )

    # Run ensemble on full dataset
    ensemble_strat = EnsembleStrategy(ensemble_params)
    ensemble_signals = ensemble_strat.generate_signals(bars)
    ensemble_full = run_backtest(bars, ensemble_signals, config)

    print(f"\n  Ensemble Results (Full Dataset):")
    print(f"    PnL: ${ensemble_full.total_pnl:,.2f}")
    print(f"    Trades: {ensemble_full.total_trades}")
    print(f"    Win Rate: {ensemble_full.win_rate*100:.1f}%")
    print(f"    Sharpe: {ensemble_full.sharpe_ratio:.2f}")
    print(f"    Max DD: {ensemble_full.max_drawdown_pct:.2f}%")
    print(f"    Profit Factor: {ensemble_full.profit_factor:.3f}")
    print(f"    Parameters: {ensemble_params}")

    # ===== Run Individual Strategies (best params from previous optimization) =====
    print("\n[3/5] Running individual strategies for comparison...")
    print("-" * 70)

    # OrderFlow with best params
    of_params = {
        "name": "OrderFlow",
        "ofi_period": 30,
        "ofi_threshold": 2.0,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21],
        "volume_confirm": True,
    }
    of_strat = OrderFlowStrategy(of_params)
    of_signals = of_strat.generate_signals(bars)
    of_full = run_backtest(bars, of_signals, config)

    # Validation
    of_val_signals = of_strat.generate_signals(val_bars)
    of_val = run_backtest(val_bars, of_val_signals, config)

    print(f"  OrderFlow:    PnL=${of_full.total_pnl:,.2f}  Sharpe={of_full.sharpe_ratio:.2f}  "
          f"DD={of_full.max_drawdown_pct:.1f}%  Trades={of_full.total_trades}  "
          f"Val=${of_val.total_pnl:,.2f}")

    # MomentumMTF with best params
    mtf_params = {
        "name": "MomentumMTF",
        "slow_period": 40,
        "fast_rsi_period": 7,
        "fast_rsi_ob": 75,
        "fast_rsi_os": 20,
        "trend_strength_threshold": 0.1,
        "atr_period": 14,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 2.0,
        "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    }
    mtf_strat = MomentumMTFStrategy(mtf_params)
    mtf_signals = mtf_strat.generate_signals(bars)
    mtf_full = run_backtest(bars, mtf_signals, config)

    # Validation
    mtf_val_signals = mtf_strat.generate_signals(val_bars)
    mtf_val = run_backtest(val_bars, mtf_val_signals, config)

    print(f"  MomentumMTF:  PnL=${mtf_full.total_pnl:,.2f}  Sharpe={mtf_full.sharpe_ratio:.2f}  "
          f"DD={mtf_full.max_drawdown_pct:.1f}%  Trades={mtf_full.total_trades}  "
          f"Val=${mtf_val.total_pnl:,.2f}")

    # SpreadFade with best params
    sf_params = {
        "name": "SpreadFade",
        "spread_lookback": 30,
        "wide_threshold": 2.5,
        "contract_threshold": 1.5,
        "price_lookback": 5,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "active_hours": [1, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "cooldown_bars": 5,
    }
    sf_strat = SpreadFadeStrategy(sf_params)
    sf_signals = sf_strat.generate_signals(bars)
    sf_full = run_backtest(bars, sf_signals, config)

    # Validation
    sf_val_signals = sf_strat.generate_signals(val_bars)
    sf_val = run_backtest(val_bars, sf_val_signals, config)

    print(f"  SpreadFade:   PnL=${sf_full.total_pnl:,.2f}  Sharpe={sf_full.sharpe_ratio:.2f}  "
          f"DD={sf_full.max_drawdown_pct:.1f}%  Trades={sf_full.total_trades}  "
          f"Val=${sf_val.total_pnl:,.2f}")

    # Ensemble validation
    ensemble_val_signals = ensemble_strat.generate_signals(val_bars)
    ensemble_val_result = run_backtest(val_bars, ensemble_val_signals, config)

    print(f"  Ensemble:     PnL=${ensemble_full.total_pnl:,.2f}  Sharpe={ensemble_full.sharpe_ratio:.2f}  "
          f"DD={ensemble_full.max_drawdown_pct:.1f}%  Trades={ensemble_full.total_trades}  "
          f"Val=${ensemble_val_result.total_pnl:,.2f}")

    # ===== Comparison Table =====
    print("\n[4/5] Strategy Comparison")
    print("=" * 90)
    print(f"{'Strategy':<15} {'PnL ($)':<12} {'Sharpe':<8} {'Win%':<8} {'MaxDD%':<9} "
          f"{'PF':<8} {'Trades':<8} {'Val PnL':<10} {'Avg Trade':<10}")
    print("-" * 90)

    strategies_data = [
        ("Ensemble", ensemble_full, ensemble_val_result, ensemble_params),
        ("OrderFlow", of_full, of_val, of_params),
        ("MomentumMTF", mtf_full, mtf_val, mtf_params),
        ("SpreadFade", sf_full, sf_val, sf_params),
    ]

    for name, full_r, val_r, _ in strategies_data:
        print(f"{name:<15} {full_r.total_pnl:>10,.2f} {full_r.sharpe_ratio:>6.2f} "
              f"{full_r.win_rate*100:>6.1f} {full_r.max_drawdown_pct:>7.2f} "
              f"{full_r.profit_factor:>6.3f} {full_r.total_trades:>6d} "
              f"{val_r.total_pnl:>8,.2f} {full_r.avg_trade_pnl:>8.2f}")

    # ===== Determine Winner =====
    # Score: Weighted combination of PnL, Sharpe, and validation performance
    scores = {}
    for name, full_r, val_r, params in strategies_data:
        if full_r.total_trades == 0:
            scores[name] = -999
            continue
        pnl_score = full_r.total_pnl / 100.0
        sharpe_score = full_r.sharpe_ratio * 2.0
        pf_score = max(full_r.profit_factor - 1.0, 0) * 5.0
        val_score = max(val_r.total_pnl, 0) / 200.0
        dd_penalty = max(-full_r.max_drawdown_pct - 30, 0) * 2.0
        scores[name] = pnl_score + sharpe_score + pf_score + val_score - dd_penalty

    winner_name = max(scores, key=scores.get)
    winner_data = next(d for d in strategies_data if d[0] == winner_name)

    print(f"\n{'='*70}")
    print(f"WINNER: {winner_name} (Score: {scores[winner_name]:.2f})")
    print(f"{'='*70}")
    print(f"  Total PnL:          ${winner_data[1].total_pnl:,.2f}")
    print(f"  Max Drawdown:       {winner_data[1].max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:       {winner_data[1].sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:      {winner_data[1].sortino_ratio:.3f}")
    print(f"  Win Rate:           {winner_data[1].win_rate*100:.1f}%")
    print(f"  Profit Factor:      {winner_data[1].profit_factor:.3f}")
    print(f"  Total Trades:       {winner_data[1].total_trades}")
    print(f"  Avg Trade PnL:      ${winner_data[1].avg_trade_pnl:.2f}")
    print(f"  Avg Winner:         ${winner_data[1].avg_winner:.2f}")
    print(f"  Avg Loser:          ${winner_data[1].avg_loser:.2f}")
    print(f"  Consec Losses:      {winner_data[1].max_consecutive_losses}")
    print(f"  Validation PnL:     ${winner_data[2].total_pnl:,.2f}")
    print(f"  Parameters:         {winner_data[3]}")

    # ===== Save Results =====
    print("\n[5/5] Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "winner": winner_name,
        "winner_params": winner_data[3],
        "winner_score": round(scores[winner_name], 3),
        "comparison": {},
        "backtest_config": {
            "slippage_points": config.slippage_points,
            "commission_per_lot_rt": config.commission_per_lot_rt,
            "lot_size": config.lot_size,
            "contract_size": config.contract_size,
            "initial_equity": config.initial_equity,
            "leverage": config.leverage,
        },
        "data_info": {
            "total_bars": n_bars,
            "date_start": str(bars.index[0]),
            "date_end": str(bars.index[-1]),
            "train_bars": len(train_bars),
            "validation_bars": len(val_bars),
        },
    }

    for name, full_r, val_r, params in strategies_data:
        results["comparison"][name] = {
            "full": result_to_dict(full_r),
            "validation": result_to_dict(val_r),
            "params": params,
            "composite_score": round(scores.get(name, 0), 3),
        }

    # Save trade log for winner
    winner_result = winner_data[1]
    trade_log = []
    for t in winner_result.trades[:500]:
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
    results["winner_trades"] = trade_log

    with open(RESULTS_DIR / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nResults saved to {RESULTS_DIR / 'ensemble_results.json'}")
    print(f"Total execution time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    run_ensemble_comparison()
