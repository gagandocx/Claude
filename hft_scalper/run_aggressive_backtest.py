"""
Aggressive backtest runner for XAUUSD scalping system.

Strategy: Multi-position RSI(8) mean-reversion with compounding.

Key features:
- Multiple concurrent positions (up to 3) for maximum signal utilization
- RSI(8) extreme entries at oversold/overbought levels
- ATR-based SL/TP (2.5x/2.0x) with 50-bar timeout
- Compounding position sizing proportional to current equity
- Realistic execution: 0.3pt slippage, $7/lot commission, 0.15pt spread
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from hft_scalper.data_loader import load_ticks, build_ohlc_bars


@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    lot_size: float
    pnl_points: float
    pnl_dollar: float
    commission: float
    exit_reason: str
    equity_after: float


def compute_rsi(close, period):
    """Compute RSI indicator."""
    n = len(close)
    rsi = np.full(n, 50.0)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if n < period + 1:
        return rsi
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return rsi


def compute_atr(high, low, close, period):
    """Compute Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        atr[:period-1] = atr[period-1]
    return atr


def run_backtest(bars: pd.DataFrame) -> Tuple[List[TradeRecord], np.ndarray, dict]:
    """
    Run multi-position backtest with compounding.
    
    Allows up to MAX_POSITIONS concurrent positions to maximize
    signal utilization from RSI extremes. Each position is independently
    managed with its own SL/TP levels.
    """
    n = len(bars)
    close = bars["close"].values.astype(np.float64)
    high = bars["high"].values.astype(np.float64)
    low = bars["low"].values.astype(np.float64)
    
    # Indicators
    rsi = compute_rsi(close, 8)
    atr = compute_atr(high, low, close, 14)
    
    # Session hours
    hours = bars.index.hour if hasattr(bars.index, 'hour') else np.full(n, 12)
    
    # Constants
    INITIAL_EQUITY = 1000.0
    SLIPPAGE = 0.3
    SPREAD_HALF = 0.15
    COMMISSION_PER_LOT = 7.0
    CONTRACT_SIZE = 100.0
    MIN_LOT = 0.01
    MAX_LOT = 10.0
    
    # Strategy parameters
    RSI_THRESH = 25.0
    SL_MULT = 2.5
    TP_MULT = 2.0
    MAX_POSITIONS = 3
    TOTAL_RISK = 0.03  # Total risk across all positions
    MAX_HOLD = 50
    WARMUP = 50
    
    # State
    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    equity_curve = np.zeros(n)
    trades: List[TradeRecord] = []
    
    # Positions: list of dicts
    positions = []
    
    for i in range(n):
        # Check all open positions for exits
        closed_indices = []
        for pos_idx in range(len(positions)):
            pos = positions[pos_idx]
            exit_p = 0.0
            reason = ""
            
            if pos["direction"] == 1:
                if low[i] <= pos["stop_loss"]:
                    exit_p = pos["stop_loss"] - SLIPPAGE
                    reason = "sl"
                elif high[i] >= pos["take_profit"]:
                    exit_p = pos["take_profit"]
                    reason = "tp"
            else:
                if high[i] >= pos["stop_loss"]:
                    exit_p = pos["stop_loss"] + SLIPPAGE
                    reason = "sl"
                elif low[i] <= pos["take_profit"]:
                    exit_p = pos["take_profit"]
                    reason = "tp"
            
            # Timeout
            if not reason and (i - pos["entry_bar"]) >= MAX_HOLD:
                if pos["direction"] == 1:
                    exit_p = close[i] - SPREAD_HALF - SLIPPAGE
                else:
                    exit_p = close[i] + SPREAD_HALF + SLIPPAGE
                reason = "timeout"
            
            if reason:
                if pos["direction"] == 1:
                    pnl_pts = exit_p - pos["entry_price"]
                else:
                    pnl_pts = pos["entry_price"] - exit_p
                
                comm = COMMISSION_PER_LOT * pos["lot_size"]
                pnl_d = pnl_pts * CONTRACT_SIZE * pos["lot_size"] - comm
                equity += pnl_d
                
                if equity > peak_equity:
                    peak_equity = equity
                
                trades.append(TradeRecord(
                    entry_bar=pos["entry_bar"], exit_bar=i,
                    direction=pos["direction"],
                    entry_price=pos["entry_price"], exit_price=exit_p,
                    lot_size=pos["lot_size"], pnl_points=pnl_pts,
                    pnl_dollar=pnl_d, commission=comm,
                    exit_reason=reason, equity_after=equity,
                ))
                closed_indices.append(pos_idx)
        
        # Remove closed positions
        for idx in sorted(closed_indices, reverse=True):
            positions.pop(idx)
        
        # Entry: new position if slots available
        if len(positions) < MAX_POSITIONS and i >= WARMUP:
            if hours[i] >= 7 and hours[i] <= 20 and atr[i] >= 0.5:
                sig = 0
                if rsi[i] < RSI_THRESH:
                    sig = 1
                elif rsi[i] > (100.0 - RSI_THRESH):
                    sig = -1
                
                if sig != 0:
                    # Position sizing: divide total risk among max positions
                    risk_per_pos = TOTAL_RISK / MAX_POSITIONS
                    sl_d = atr[i] * SL_MULT
                    tp_d = atr[i] * TP_MULT
                    
                    lot_size = (equity * risk_per_pos) / (sl_d * CONTRACT_SIZE)
                    lot_size = max(MIN_LOT, min(MAX_LOT, round(lot_size, 2)))
                    
                    if sig == 1:
                        entry_price = close[i] + SPREAD_HALF + SLIPPAGE
                        stop_loss = entry_price - sl_d
                        take_profit = entry_price + tp_d
                    else:
                        entry_price = close[i] - SPREAD_HALF - SLIPPAGE
                        stop_loss = entry_price + sl_d
                        take_profit = entry_price - tp_d
                    
                    positions.append({
                        "entry_bar": i,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "direction": sig,
                        "lot_size": lot_size,
                    })
        
        equity_curve[i] = equity
    
    # Close remaining positions
    for pos in positions:
        if pos["direction"] == 1:
            exit_p = close[-1] - SPREAD_HALF - SLIPPAGE
            pnl_pts = exit_p - pos["entry_price"]
        else:
            exit_p = close[-1] + SPREAD_HALF + SLIPPAGE
            pnl_pts = pos["entry_price"] - exit_p
        comm = COMMISSION_PER_LOT * pos["lot_size"]
        pnl_d = pnl_pts * CONTRACT_SIZE * pos["lot_size"] - comm
        equity += pnl_d
        equity_curve[-1] = equity
        trades.append(TradeRecord(
            entry_bar=pos["entry_bar"], exit_bar=n-1,
            direction=pos["direction"],
            entry_price=pos["entry_price"], exit_price=exit_p,
            lot_size=pos["lot_size"], pnl_points=pnl_pts,
            pnl_dollar=pnl_d, commission=comm,
            exit_reason="end", equity_after=equity,
        ))
    
    metrics = _compute_metrics(trades, equity_curve, INITIAL_EQUITY)
    return trades, equity_curve, metrics


def _compute_metrics(trades, equity_curve, initial_equity):
    """Compute performance metrics."""
    if not trades:
        return {"total_trades": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "initial_equity": initial_equity, "final_equity": initial_equity,
                "winning_trades": 0, "losing_trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_trade_pnl": 0.0, "avg_winner": 0.0,
                "avg_loser": 0.0, "max_consecutive_losses": 0, "net_points": 0.0,
                "avg_lot_size": 0.0, "max_lot_size": 0.0, "total_commission": 0.0,
                "sharpe_ratio": 0.0}
    
    pnls = np.array([t.pnl_dollar for t in trades])
    valid_eq = equity_curve[equity_curve > 0]
    final_equity = valid_eq[-1] if len(valid_eq) > 0 else initial_equity
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100
    
    if len(valid_eq) > 0:
        peak = np.maximum.accumulate(valid_eq)
        dd_pct = (valid_eq - peak) / peak * 100
        max_dd = float(np.min(dd_pct))
    else:
        max_dd = 0.0
    
    winning = int(np.sum(pnls > 0))
    losing = int(np.sum(pnls <= 0))
    total = len(pnls)
    wr = winning / total if total > 0 else 0
    
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = abs(float(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    pf = gross_profit / gross_loss
    
    avg_pnl = float(np.mean(pnls))
    avg_winner = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    avg_loser = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0
    
    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    
    lot_sizes = [t.lot_size for t in trades]
    total_comm = sum(t.commission for t in trades)
    net_pts = float(np.sum([t.pnl_points for t in trades]))
    
    sharpe = 0.0
    if len(valid_eq) > 1:
        eq_ret = np.diff(valid_eq) / valid_eq[:-1]
        eq_ret = eq_ret[np.isfinite(eq_ret)]
        if len(eq_ret) > 1 and np.std(eq_ret) > 0:
            sharpe = float(np.mean(eq_ret) / np.std(eq_ret) * np.sqrt(252 * 78))
    
    return {
        "initial_equity": initial_equity,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 1),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 2),
        "avg_trade_pnl": round(avg_pnl, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "max_consecutive_losses": max_streak,
        "net_points": round(net_pts, 1),
        "avg_lot_size": round(float(np.mean(lot_sizes)), 3),
        "max_lot_size": round(float(np.max(lot_sizes)), 3),
        "total_commission": round(total_comm, 2),
        "sharpe_ratio": round(sharpe, 2),
    }


def main():
    """Run the aggressive backtest."""
    start_time = time.time()
    
    print("=" * 70)
    print("AGGRESSIVE XAUUSD SCALPING BACKTEST")
    print("=" * 70)
    print("Strategy: Multi-Position RSI(8) Mean-Reversion with Compounding")
    print("Initial Equity: $1,000 | Leverage: 1:500")
    print("Target: >500% return, <15% max drawdown")
    print()
    
    # Load data
    tick_path = Path(__file__).parent.parent / "tick_data" / "XAUUSD_RealTicks.csv"
    ticks = load_ticks(tick_path)
    bars = build_ohlc_bars(ticks, freq="5min")
    print(f"\n5-min bars: {len(bars):,}")
    print(f"Period: {bars.index[0]} to {bars.index[-1]}")
    print()
    
    # Run backtest
    print("Running multi-position backtest...")
    trades, equity_curve, metrics = run_backtest(bars)
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Initial Equity:     ${metrics['initial_equity']:,.2f}")
    print(f"Final Equity:       ${metrics['final_equity']:,.2f}")
    print(f"Return:             {metrics['total_return_pct']:.1f}%")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:       {metrics['total_trades']}")
    print(f"Win Rate:           {metrics['win_rate']:.1%}")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Avg Trade PnL:      ${metrics['avg_trade_pnl']:.2f}")
    print(f"Avg Winner:         ${metrics['avg_winner']:.2f}")
    print(f"Avg Loser:          ${metrics['avg_loser']:.2f}")
    print(f"Max Consec. Losses: {metrics['max_consecutive_losses']}")
    print(f"Net Points:         {metrics['net_points']:.1f}")
    print(f"Avg Lot Size:       {metrics['avg_lot_size']:.3f}")
    print(f"Max Lot Size:       {metrics['max_lot_size']:.3f}")
    print(f"Total Commission:   ${metrics['total_commission']:.2f}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"Execution Time:     {elapsed:.1f}s")
    print("=" * 70)
    
    target_return = metrics["total_return_pct"] > 500
    target_dd = metrics["max_drawdown_pct"] > -15
    print(f"\nTarget >500% Return: {'PASS' if target_return else 'FAIL'} ({metrics['total_return_pct']:.1f}%)")
    print(f"Target <15% Max DD:  {'PASS' if target_dd else 'FAIL'} ({abs(metrics['max_drawdown_pct']):.2f}%)")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_json = {
        "strategy": "MultiPosition_RSI8_MeanReversion_Compounding",
        "description": (
            "Multi-position RSI(8) mean-reversion with compounding. "
            "Up to 3 concurrent positions with RSI<25 buy and RSI>75 sell entries. "
            "SL=2.5x ATR, TP=2.0x ATR, position sizing proportional to equity."
        ),
        **metrics,
        "execution_config": {
            "rsi_period": 8,
            "rsi_threshold": 25,
            "sl_atr_mult": 2.5,
            "tp_atr_mult": 2.0,
            "max_positions": 3,
            "total_risk_pct": 0.03,
            "slippage_points": 0.3,
            "spread_half": 0.15,
            "commission_per_lot_rt": 7.0,
            "contract_size": 100,
            "leverage": 500,
            "timeframe": "5min",
            "active_hours": "07:00-20:00 UTC",
            "max_hold_bars": 50,
            "compounding": True,
            "dynamic_position_sizing": True,
        },
        "execution_time_seconds": round(elapsed, 1),
    }
    
    results_path = results_dir / "aggressive_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return metrics


if __name__ == "__main__":
    main()
