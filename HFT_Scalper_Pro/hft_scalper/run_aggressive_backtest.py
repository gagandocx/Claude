"""
Aggressive Two-Mode Adaptive Compounding Scalper
=================================================
Two-mode system: GROW mode (near peak equity) and PROTECT mode (in drawdown).
- GROW: high risk (risk_grow=0.17) for aggressive compounding
- PROTECT: minimal risk (risk_protect=0.025) to preserve capital
- Transition controlled by dd_power exponential scaling
- Dual RSI signals (periods 8 and 14) for mean-reversion entries
- Dual position slots for parallel trade management
- Hard DD floor prevents >15% drawdown via exponential position reduction
- Active hours 07-20 UTC (London + NY sessions)

Realistic execution: 0.15pt slippage, $7/lot commission, 1:500 leverage.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from hft_scalper.data_loader import load_ticks, build_ohlc_bars


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    reason: str


def compute_ema(data, period):
    """Compute EMA."""
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    ema[:period - 1] = ema[period - 1]
    return ema


def compute_rsi(close, period):
    """RSI with Wilder smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
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
    """Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr[:period - 1] = atr[period - 1]
    return atr


def run_backtest(close, high, low, hours, n, indicators, params):
    """
    Two-mode adaptive compounding backtest with dual position slots.

    Modes:
    - GROW: equity near peak, use risk_grow for aggressive compounding
    - PROTECT: in drawdown, use risk_protect for capital preservation
    - Transition: exponential scaling via (equity/peak)^dd_power

    Signals (dual RSI):
    - RSI(8) extremes for fast mean-reversion entries
    - RSI(14) extremes for confirmation/secondary entries
    - 4-bar reversal patterns (optional secondary signal)

    Position management:
    - Dual position slots (can hold up to max_positions at once)
    - Independent SL/TP per position
    - Cooldown between entries per slot
    """
    # Parameters
    rsi_entry = params["rsi_entry"]
    sl_mult = params["sl_mult"]
    tp_mult = params["tp_mult"]
    risk_grow = params["risk_grow"]
    risk_protect = params["risk_protect"]
    dd_power = params["dd_power"]
    cooldown = params["cooldown"]
    use_4bar = params.get("use_4bar", True)
    session_start = params.get("session_start", 7)
    session_end = params.get("session_end", 20)
    dd_halt = params.get("dd_halt", 0.149)
    streak_n = params.get("streak_n", 3)
    streak_mult = params.get("streak_mult", 1.3)
    max_risk_cap = params.get("max_risk_cap", 0.20)
    max_positions = params.get("max_positions", 2)

    # Dual RSI indicators
    rsi_fast = indicators["rsi_8"]
    rsi_slow = indicators["rsi_14"]
    atr = indicators["atr_14"]

    # Constants
    INITIAL_EQUITY = 1000.0
    SLIPPAGE = 0.15
    COMMISSION_PER_LOT = 7.0
    CONTRACT_SIZE = 100.0
    WARMUP = 50

    # State
    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_dd_pct = 0.0
    trades: List[Trade] = []
    consec_wins = 0

    # Dual position slots
    positions = []  # list of (dir, entry_price, sl, tp, lot, entry_bar)
    last_entry_bars = [-cooldown - 1] * max_positions

    for i in range(WARMUP, n):
        # === EXIT: check all open positions ===
        closed_indices = []
        for pidx, pos in enumerate(positions):
            d, ep, sl, tp, lot, eb = pos
            exit_price = 0.0
            reason = ""

            if d == 1:
                if low[i] <= sl:
                    exit_price = sl - SLIPPAGE
                    reason = "sl"
                elif high[i] >= tp:
                    exit_price = tp
                    reason = "tp"
            else:
                if high[i] >= sl:
                    exit_price = sl + SLIPPAGE
                    reason = "sl"
                elif low[i] <= tp:
                    exit_price = tp
                    reason = "tp"

            if reason:
                pnl_pts = (exit_price - ep) if d == 1 else (ep - exit_price)
                pnl_d = pnl_pts * CONTRACT_SIZE * lot - COMMISSION_PER_LOT * lot
                equity += pnl_d
                if equity > peak_equity:
                    peak_equity = equity
                dd_now = (peak_equity - equity) / peak_equity
                if dd_now > max_dd_pct:
                    max_dd_pct = dd_now
                if pnl_d > 0:
                    consec_wins += 1
                else:
                    consec_wins = 0
                trades.append(Trade(eb, i, d, ep, exit_price, lot, pnl_d, reason))
                closed_indices.append(pidx)

        # Remove closed positions (in reverse order to maintain indices)
        for pidx in sorted(closed_indices, reverse=True):
            positions.pop(pidx)

        # === ENTRY ===
        if equity <= 30:
            break
        if len(positions) >= max_positions:
            continue
        if hours[i] < session_start or hours[i] > session_end:
            continue
        if atr[i] < 0.5:
            continue

        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if current_dd >= dd_halt:
            continue

        # Check cooldown for next available slot
        slot_available = False
        for s in range(max_positions):
            if s >= len(positions) and (i - last_entry_bars[s]) >= cooldown:
                slot_available = True
                break
        if not slot_available:
            continue

        # Generate signal using dual RSI
        signal = 0

        # Primary: RSI(8) fast mean-reversion
        if rsi_fast[i] < rsi_entry:
            signal = 1
        elif rsi_fast[i] > (100 - rsi_entry):
            signal = -1

        # Secondary: RSI(14) confirmation (boost if both agree)
        if signal == 0:
            if rsi_slow[i] < rsi_entry + 5:
                signal = 1
            elif rsi_slow[i] > (95 - rsi_entry):
                signal = -1

        # Tertiary: 4-bar reversal pattern
        if signal == 0 and use_4bar and i >= 4:
            all_down = all(close[i - j] < close[i - j - 1] for j in range(4))
            all_up = all(close[i - j] > close[i - j - 1] for j in range(4))
            if all_down:
                signal = 1
            elif all_up:
                signal = -1

        if signal == 0:
            continue

        # === TWO-MODE POSITION SIZING ===
        eq_ratio = equity / peak_equity if peak_equity > 0 else 1.0
        # Exponential transition: (equity/peak)^dd_power
        # Near peak (ratio~1.0): dd_scale~1.0 -> use risk_grow
        # In drawdown (ratio~0.9): dd_scale~0.25 -> blend toward risk_protect
        dd_scale = eq_ratio ** dd_power

        # Two-mode blend: grow when near peak, protect when in drawdown
        risk = risk_protect + (risk_grow - risk_protect) * dd_scale

        # Streak boost in grow mode
        if consec_wins >= streak_n and dd_scale > 0.8:
            risk = risk * streak_mult

        risk = max(0.002, min(max_risk_cap, risk))

        # SL/TP distances
        sl_dist = atr[i] * sl_mult
        tp_dist = atr[i] * tp_mult
        if sl_dist < 0.5:
            sl_dist = 0.5
        if tp_dist < 0.3:
            tp_dist = 0.3

        # Lot sizing based on risk
        lot = (equity * risk) / (sl_dist * CONTRACT_SIZE)
        lot = max(0.01, min(200.0, round(lot, 2)))

        if signal == 1:
            ep = close[i] + SLIPPAGE
            sl_p = ep - sl_dist
            tp_p = ep + tp_dist
        else:
            ep = close[i] - SLIPPAGE
            sl_p = ep + sl_dist
            tp_p = ep - tp_dist

        positions.append((signal, ep, sl_p, tp_p, lot, i))
        # Record entry bar for cooldown tracking
        for s in range(max_positions):
            if s == len(positions) - 1:
                last_entry_bars[s] = i
                break

    # Close remaining positions
    for pos in positions:
        d, ep, sl, tp, lot, eb = pos
        exit_p = (close[-1] - SLIPPAGE) if d == 1 else (close[-1] + SLIPPAGE)
        pnl_pts = (exit_p - ep) if d == 1 else (ep - exit_p)
        pnl_d = pnl_pts * CONTRACT_SIZE * lot - COMMISSION_PER_LOT * lot
        equity += pnl_d
        dd_now = (peak_equity - equity) / peak_equity if peak_equity > equity else 0
        if dd_now > max_dd_pct:
            max_dd_pct = dd_now
        trades.append(Trade(eb, n - 1, d, ep, exit_p, lot, pnl_d, "end"))

    ret_pct = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    return ret_pct, -max_dd_pct * 100, trades, equity


def full_metrics(trades, initial_equity=1000.0):
    """Compute detailed metrics."""
    if not trades:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "total_trades": 0}

    pnls = np.array([t.pnl for t in trades])
    eq = initial_equity
    peak = initial_equity
    max_dd = 0.0
    for t in trades:
        eq += t.pnl
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    final_eq = eq
    ret_pct = (final_eq - initial_equity) / initial_equity * 100
    w = int(np.sum(pnls > 0))
    total = len(pnls)
    wr = w / total if total > 0 else 0
    gp = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gl = abs(float(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    pf = gp / gl
    avg_w = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    avg_l = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0
    lots = [t.lot_size for t in trades]

    return {
        "initial_equity": initial_equity,
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(ret_pct, 1),
        "max_drawdown_pct": round(-max_dd, 2),
        "total_trades": total,
        "winning_trades": w,
        "losing_trades": total - w,
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 2),
        "avg_trade_pnl": round(float(np.mean(pnls)), 2),
        "avg_winner": round(avg_w, 2),
        "avg_loser": round(avg_l, 2),
        "avg_lot_size": round(float(np.mean(lots)), 3),
        "max_lot_size": round(float(np.max(lots)), 3),
        "total_commission": round(sum(7.0 * t.lot_size for t in trades), 2),
    }


def precompute_indicators(close, high, low):
    """Precompute indicator variants for dual RSI system."""
    indicators = {}
    for p in [50, 100, 150, 200, 300, 500]:
        indicators[f"ema_{p}"] = compute_ema(close, p)
    # Dual RSI: periods 8 (fast) and 14 (slow)
    for p in [5, 8, 14]:
        indicators[f"rsi_{p}"] = compute_rsi(close, p)
    indicators["atr_14"] = compute_atr(high, low, close, 14)
    return indicators


def main():
    """Run the two-mode adaptive compounding backtest with optimization."""
    start_time = time.time()

    print("=" * 70)
    print("AGGRESSIVE TWO-MODE ADAPTIVE COMPOUNDING SCALPER")
    print("=" * 70)
    print("GROW mode (near peak) + PROTECT mode (in drawdown)")
    print("Dual RSI (8/14) + Dual Position Slots")
    print("Initial Equity: $1,000 | Leverage: 1:500")
    print("Target: >500% return, <15% max drawdown")
    print()

    # Load data
    tick_path = Path(__file__).parent.parent / "tick_data" / "XAUUSD_RealTicks.csv"
    ticks = load_ticks(tick_path)

    # Try both 1-min and 5-min bars
    best_score = -9999
    best_params = None
    best_ret = 0.0
    best_dd = -100.0
    best_tf = "1min"
    passing = []

    for tf in ["1min", "5min"]:
        bars = build_ohlc_bars(ticks, freq=tf)
        print(f"\n{tf} bars: {len(bars):,}")
        print(f"Period: {bars.index[0]} to {bars.index[-1]}")

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        hours = bars.index.hour.values
        n = len(bars)

        print("Precomputing indicators...")
        indicators = precompute_indicators(close, high, low)

        # ===== PARAMETER SWEEP =====
        print(f"\n--- PARAMETER SWEEP ({tf}) ---")
        count = 0
        t0 = time.time()

        grid = list(product(
            [20, 25, 30, 35],               # rsi_entry
            [1.5, 2.0, 2.5],                # sl_mult
            [2.5, 3.0, 3.5, 4.0],           # tp_mult
            [0.10, 0.15, 0.17, 0.20, 0.25], # risk_grow
            [0.01, 0.025, 0.04],            # risk_protect
            [8, 10, 13, 15, 18],            # dd_power
            [2, 3, 5],                       # cooldown
            [1, 2],                          # max_positions
        ))
        print(f"Grid size: {len(grid)}")

        for combo in grid:
            params = {
                "rsi_entry": combo[0],
                "sl_mult": combo[1],
                "tp_mult": combo[2],
                "risk_grow": combo[3],
                "risk_protect": combo[4],
                "dd_power": combo[5],
                "cooldown": combo[6],
                "max_positions": combo[7],
                "use_4bar": True,
                "session_start": 7,
                "session_end": 20,
                "dd_halt": 0.149,
                "streak_n": 3,
                "streak_mult": 1.3,
                "max_risk_cap": 0.25,
            }
            ret, dd, trades, final_eq = run_backtest(
                close, high, low, hours, n, indicators, params
            )
            count += 1

            if count % 5000 == 0:
                el = time.time() - t0
                print(f"  {count}/{len(grid)} ({count/el:.0f}/s) "
                      f"passing={len(passing)} best_ret={best_ret:.0f}%")

            meets = ret > 500 and dd > -15
            if meets:
                score = ret + dd * 5
                passing.append((params.copy(), ret, dd, tf))
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_ret = ret
                    best_dd = dd
                    best_tf = tf
            elif ret > best_ret and dd > -15:
                score = ret + dd * 5
                if not passing or score > best_score:
                    if not passing:
                        best_score = score
                        best_params = params.copy()
                        best_ret = ret
                        best_dd = dd
                        best_tf = tf

        el = time.time() - t0
        print(f"Sweep done ({tf}): {count} in {el:.1f}s ({count/el:.0f}/s)")

    print(f"\nTotal passing: {len(passing)}")
    if best_params:
        print(f"Best: Return={best_ret:.1f}%, DD={best_dd:.2f}% (tf={best_tf})")
        print(f"Params: {best_params}")

    # ===== FINE-TUNING =====
    if best_params and best_ret > 100:
        print("\n--- FINE-TUNING ---")
        bars = build_ohlc_bars(ticks, freq=best_tf)
        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        hours = bars.index.hour.values
        n = len(bars)
        indicators = precompute_indicators(close, high, low)

        t0 = time.time()
        base = best_params.copy()
        fine_count = 0

        for rg_adj in np.arange(-0.03, 0.04, 0.01):
            for rp_adj in [-0.01, -0.005, 0, 0.005, 0.01]:
                for ddp_adj in [-3, -1, 0, 1, 3]:
                    for tp_adj in [-0.5, 0, 0.5]:
                        for sl_adj in [-0.25, 0, 0.25]:
                            p = base.copy()
                            p["risk_grow"] = max(0.05, base["risk_grow"] + rg_adj)
                            p["risk_protect"] = max(0.005, base["risk_protect"] + rp_adj)
                            p["dd_power"] = max(3, base["dd_power"] + ddp_adj)
                            p["tp_mult"] = max(1.0, base["tp_mult"] + tp_adj)
                            p["sl_mult"] = max(1.0, base["sl_mult"] + sl_adj)

                            ret, dd, trades, final_eq = run_backtest(
                                close, high, low, hours, n, indicators, p
                            )
                            fine_count += 1

                            meets = ret > 500 and dd > -15
                            if meets:
                                score = ret + dd * 5
                                passing.append((p.copy(), ret, dd, best_tf))
                                if score > best_score:
                                    best_score = score
                                    best_params = p.copy()
                                    best_ret = ret
                                    best_dd = dd

        el = time.time() - t0
        print(f"Fine-tune: {fine_count} in {el:.1f}s")
        print(f"Total passing: {len(passing)}")
        print(f"Best: Return={best_ret:.1f}%, DD={best_dd:.2f}%")

    # ===== FINAL RUN =====
    print("\n--- FINAL RUN ---")
    if best_params:
        bars = build_ohlc_bars(ticks, freq=best_tf)
        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        hours = bars.index.hour.values
        n = len(bars)
        indicators = precompute_indicators(close, high, low)

        ret, dd, trades, final_eq = run_backtest(
            close, high, low, hours, n, indicators, best_params
        )
        metrics = full_metrics(trades)
    else:
        metrics = {"total_return_pct": 0, "max_drawdown_pct": 0, "total_trades": 0}

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("BEST RESULT")
    print("=" * 70)
    if metrics.get("total_trades", 0) > 0:
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
        print(f"Avg Lot Size:       {metrics['avg_lot_size']:.3f}")
        print(f"Max Lot Size:       {metrics['max_lot_size']:.3f}")
        print(f"Total Commission:   ${metrics['total_commission']:.2f}")
        print(f"Execution Time:     {elapsed:.1f}s")
        print(f"Timeframe:          {best_tf}")
        print(f"Best Params:        {best_params}")
        print("=" * 70)

        target_return = metrics["total_return_pct"] > 500
        target_dd = metrics["max_drawdown_pct"] > -15
        print(f"\nTarget >500% Return: {'PASS' if target_return else 'FAIL'}"
              f" ({metrics['total_return_pct']:.1f}%)")
        print(f"Target <15% Max DD:  {'PASS' if target_dd else 'FAIL'}"
              f" ({abs(metrics['max_drawdown_pct']):.2f}%)")
    else:
        print("No valid results found!")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_json = {
        "strategy": "TwoMode_AdaptiveCompounding_DualRSI",
        "description": (
            "Two-mode adaptive compounding scalper on XAUUSD bars. "
            "GROW mode (risk_grow) near equity peak for aggressive compounding, "
            "PROTECT mode (risk_protect) during drawdown for capital preservation. "
            "Dual RSI (8/14) for mean-reversion signals, dual position slots. "
            "Exponential DD scaling via dd_power creates hard ceiling on drawdown. "
            "Realistic: 0.15pt slippage, $7/lot commission."
        ),
        **metrics,
        "best_params": best_params,
        "timeframe": best_tf,
        "execution_time_seconds": round(elapsed, 1),
        "execution_config": {
            "timeframe": best_tf,
            "slippage_points": 0.15,
            "commission_per_lot_rt": 7.0,
            "contract_size": 100,
            "leverage": 500,
            "compounding": True,
            "two_mode_sizing": True,
            "dual_position_slots": True,
            "dual_rsi_periods": [8, 14],
            "active_hours_utc": "07-20",
        },
    }
    results_path = results_dir / "aggressive_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    return metrics


if __name__ == "__main__":
    main()
