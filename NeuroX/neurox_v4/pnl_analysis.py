#!/usr/bin/env python3
"""
NeuroX v4 - Real Trade PnL Analysis
====================================
Analyzes 26 real trades from June 26, 2026 (one day of live trading).
Calculates comprehensive performance metrics including:
- Total PnL, win rate, profit factor
- Session breakdown (asian/london/overlap)
- Direction breakdown (BUY/SELL)
- Confidence level breakdown (high >0.55 vs low <0.55)
- Equity curve, drawdown analysis
- Forward projections (weekly/monthly)
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Trade:
    id: int
    direction: str
    session: str
    confidence: float
    pnl: float


# ============================================================
# RAW TRADE DATA - 26 Real NeuroX v4 Trades (June 26, 2026)
# ============================================================
TRADES = [
    Trade(1,  "BUY",  "asian",   0.578,  2.40),
    Trade(2,  "SELL", "asian",   0.432, -6.90),
    Trade(3,  "BUY",  "asian",   0.425,  1.30),
    Trade(4,  "SELL", "asian",   0.491,  3.00),
    Trade(5,  "BUY",  "asian",   0.645,  6.00),
    Trade(6,  "SELL", "asian",   0.550,  6.20),
    Trade(7,  "SELL", "london",  0.259, -1.80),
    Trade(8,  "BUY",  "london",  0.612,  6.20),
    Trade(9,  "SELL", "london",  0.546, -0.30),
    Trade(10, "BUY",  "london",  0.615, -4.90),
    Trade(11, "SELL", "london",  0.413, -0.20),
    Trade(12, "BUY",  "london",  0.659, 15.40),
    Trade(13, "BUY",  "london",  0.522,  0.00),
    Trade(14, "BUY",  "london",  0.634,  0.50),
    Trade(15, "SELL", "london",  0.534, -5.70),
    Trade(16, "BUY",  "london",  0.626, -6.60),
    Trade(17, "SELL", "london",  0.514, -1.30),
    Trade(18, "SELL", "london",  0.610, -0.90),
    Trade(19, "BUY",  "london",  0.582,  5.10),
    Trade(20, "SELL", "london",  0.401,  2.10),
    Trade(21, "BUY",  "london",  0.590, 17.83),
    Trade(22, "BUY",  "london",  0.651, 18.10),
    Trade(23, "BUY",  "overlap", 0.361, -1.40),
    Trade(24, "SELL", "overlap", 0.297, -0.10),
    Trade(25, "BUY",  "overlap", 0.418, -8.70),
    Trade(26, "SELL", "overlap", 0.432,-13.00),
]


def calculate_metrics(trades: List[Trade]) -> dict:
    """Calculate comprehensive trading metrics for a list of trades."""
    if not trades:
        return {
            "count": 0, "total_pnl": 0, "win_rate": 0,
            "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
            "sharpe": 0, "max_drawdown": 0, "avg_pnl": 0,
        }

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    breakeven = [p for p in pnls if p == 0]
    
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0
    
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    avg_pnl = np.mean(pnls)
    
    # Sharpe ratio (annualized, assuming ~252 trading days)
    if len(pnls) > 1:
        pnl_std = np.std(pnls, ddof=1)
        sharpe = (np.mean(pnls) / pnl_std) * np.sqrt(252) if pnl_std > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown
    equity_curve = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        "count": len(pnls),
        "winners": len(winners),
        "losers": len(losers),
        "breakeven": len(breakeven),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_pnl": avg_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
        "pnl_std": np.std(pnls, ddof=1) if len(pnls) > 1 else 0,
    }


def print_metrics(title: str, metrics: dict, indent: str = ""):
    """Pretty-print a metrics dictionary."""
    print(f"\n{indent}{'='*60}")
    print(f"{indent}{title}")
    print(f"{indent}{'='*60}")
    
    if metrics["count"] == 0:
        print(f"{indent}  No trades")
        return
    
    print(f"{indent}  Trades:          {metrics['count']} ({metrics['winners']}W / {metrics['losers']}L / {metrics['breakeven']}BE)")
    print(f"{indent}  Total PnL:       ${metrics['total_pnl']:+.2f}")
    print(f"{indent}  Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"{indent}  Profit Factor:   {metrics['profit_factor']:.3f}")
    print(f"{indent}  Avg Win:         ${metrics['avg_win']:+.2f}")
    print(f"{indent}  Avg Loss:        ${metrics['avg_loss']:+.2f}")
    print(f"{indent}  Avg Trade:       ${metrics['avg_pnl']:+.2f}")
    print(f"{indent}  Best Trade:      ${metrics['best_trade']:+.2f}")
    print(f"{indent}  Worst Trade:     ${metrics['worst_trade']:+.2f}")
    print(f"{indent}  Sharpe Ratio:    {metrics['sharpe']:.2f} (annualized)")
    print(f"{indent}  Max Drawdown:    ${metrics['max_drawdown']:.2f}")
    print(f"{indent}  PnL Std Dev:     ${metrics['pnl_std']:.2f}")


def print_equity_curve(trades: List[Trade]):
    """Print the running equity curve."""
    print(f"\n{'='*60}")
    print("EQUITY CURVE (Running PnL)")
    print(f"{'='*60}")
    
    cumulative = 0.0
    peak = 0.0
    
    print(f"  {'#':<4} {'Dir':<5} {'Session':<8} {'Conf':<6} {'PnL':<8} {'Cumul':<9} {'DD':<8}")
    print(f"  {'-'*4} {'-'*5} {'-'*8} {'-'*6} {'-'*8} {'-'*9} {'-'*8}")
    
    for t in trades:
        cumulative += t.pnl
        peak = max(peak, cumulative)
        dd = cumulative - peak
        
        dd_str = f"${dd:.2f}" if dd < 0 else "---"
        print(f"  {t.id:<4} {t.direction:<5} {t.session:<8} {t.confidence:<6.3f} "
              f"${t.pnl:<+7.2f} ${cumulative:<+8.2f} {dd_str}")


def print_forward_projections(metrics: dict, trades_per_day: int = 26):
    """Project forward based on observed performance."""
    print(f"\n{'='*60}")
    print("FORWARD PROJECTIONS")
    print(f"{'='*60}")
    
    avg_per_trade = metrics["avg_pnl"]
    daily_pnl = metrics["total_pnl"]  # We have 1 day of data
    
    # Assuming 5 trading days per week, ~22 per month
    weekly_pnl = daily_pnl * 5
    monthly_pnl = daily_pnl * 22
    yearly_pnl = daily_pnl * 252
    
    print(f"  Based on: {trades_per_day} trades/day, 1 day of live data")
    print(f"  Avg PnL/trade:   ${avg_per_trade:+.2f}")
    print(f"  Daily PnL:       ${daily_pnl:+.2f}")
    print(f"  Weekly (5d):     ${weekly_pnl:+.2f}")
    print(f"  Monthly (22d):   ${monthly_pnl:+.2f}")
    print(f"  Yearly (252d):   ${yearly_pnl:+.2f}")
    
    # Confidence interval using standard error
    if metrics["pnl_std"] > 0:
        daily_std = metrics["pnl_std"] * np.sqrt(trades_per_day)
        monthly_std = daily_std * np.sqrt(22)
        
        print(f"\n  Risk-Adjusted Estimates (1 std dev range):")
        print(f"  Daily:   ${daily_pnl - daily_std:+.2f} to ${daily_pnl + daily_std:+.2f}")
        print(f"  Monthly: ${monthly_pnl - monthly_std:+.2f} to ${monthly_pnl + monthly_std:+.2f}")
    
    print(f"\n  NOTE: Projections based on a single day of trading.")
    print(f"  More data needed for reliable estimates. Use with caution.")


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 14 + "NeuroX v4 PnL ANALYSIS" + " " * 22 + "#")
    print("#" + " " * 10 + "26 Real Trades - June 26, 2026" + " " * 17 + "#")
    print("#" * 60)
    
    # ============================================================
    # 1. OVERALL METRICS
    # ============================================================
    overall = calculate_metrics(TRADES)
    print_metrics("OVERALL PERFORMANCE", overall)
    
    # ============================================================
    # 2. BREAKDOWN BY SESSION
    # ============================================================
    print(f"\n\n{'#'*60}")
    print("SESSION BREAKDOWN")
    print(f"{'#'*60}")
    
    sessions = ["asian", "london", "overlap"]
    for session in sessions:
        session_trades = [t for t in TRADES if t.session == session]
        session_metrics = calculate_metrics(session_trades)
        print_metrics(f"SESSION: {session.upper()} ({len(session_trades)} trades)", session_metrics, indent="  ")
    
    # ============================================================
    # 3. BREAKDOWN BY DIRECTION
    # ============================================================
    print(f"\n\n{'#'*60}")
    print("DIRECTION BREAKDOWN")
    print(f"{'#'*60}")
    
    for direction in ["BUY", "SELL"]:
        dir_trades = [t for t in TRADES if t.direction == direction]
        dir_metrics = calculate_metrics(dir_trades)
        print_metrics(f"DIRECTION: {direction} ({len(dir_trades)} trades)", dir_metrics, indent="  ")
    
    # ============================================================
    # 4. BREAKDOWN BY CONFIDENCE LEVEL
    # ============================================================
    print(f"\n\n{'#'*60}")
    print("CONFIDENCE LEVEL BREAKDOWN")
    print(f"{'#'*60}")
    
    high_conf = [t for t in TRADES if t.confidence > 0.55]
    low_conf = [t for t in TRADES if t.confidence <= 0.55]
    
    high_metrics = calculate_metrics(high_conf)
    low_metrics = calculate_metrics(low_conf)
    
    print_metrics(f"HIGH CONFIDENCE (>0.55) - {len(high_conf)} trades", high_metrics, indent="  ")
    print_metrics(f"LOW CONFIDENCE (<=0.55) - {len(low_conf)} trades", low_metrics, indent="  ")
    
    # Edge analysis
    print(f"\n  {'~'*56}")
    print(f"  CONFIDENCE EDGE ANALYSIS:")
    if high_metrics["count"] > 0 and low_metrics["count"] > 0:
        print(f"    High conf avg PnL: ${high_metrics['avg_pnl']:+.2f}/trade")
        print(f"    Low conf avg PnL:  ${low_metrics['avg_pnl']:+.2f}/trade")
        print(f"    Edge (high - low): ${high_metrics['avg_pnl'] - low_metrics['avg_pnl']:+.2f}/trade")
        print(f"    High conf win rate: {high_metrics['win_rate']:.1f}%")
        print(f"    Low conf win rate:  {low_metrics['win_rate']:.1f}%")
        
        if high_metrics['avg_pnl'] > low_metrics['avg_pnl']:
            print(f"\n    --> Higher confidence signals produce BETTER results")
        else:
            print(f"\n    --> Higher confidence does NOT correlate with better results")
    
    # ============================================================
    # 5. EQUITY CURVE
    # ============================================================
    print_equity_curve(TRADES)
    
    # ============================================================
    # 6. FORWARD PROJECTIONS
    # ============================================================
    print_forward_projections(overall)
    
    # ============================================================
    # 7. KEY INSIGHTS
    # ============================================================
    print(f"\n{'='*60}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Best/worst sessions
    session_pnls = {}
    for session in sessions:
        session_trades = [t for t in TRADES if t.session == session]
        session_pnls[session] = sum(t.pnl for t in session_trades)
    
    best_session = max(session_pnls, key=session_pnls.get)
    worst_session = min(session_pnls, key=session_pnls.get)
    
    print(f"\n  1. BEST SESSION:  {best_session.upper()} (${session_pnls[best_session]:+.2f})")
    print(f"     WORST SESSION: {worst_session.upper()} (${session_pnls[worst_session]:+.2f})")
    
    # Direction edge
    buy_trades = [t for t in TRADES if t.direction == "BUY"]
    sell_trades = [t for t in TRADES if t.direction == "SELL"]
    buy_pnl = sum(t.pnl for t in buy_trades)
    sell_pnl = sum(t.pnl for t in sell_trades)
    
    print(f"\n  2. BUY total:  ${buy_pnl:+.2f} ({len(buy_trades)} trades)")
    print(f"     SELL total: ${sell_pnl:+.2f} ({len(sell_trades)} trades)")
    if buy_pnl > sell_pnl:
        print(f"     --> BUY side is more profitable")
    else:
        print(f"     --> SELL side is more profitable")
    
    # Confidence filter effect
    print(f"\n  3. CONFIDENCE FILTER ANALYSIS:")
    print(f"     If we only took HIGH confidence (>0.55) trades:")
    print(f"       Trades: {high_metrics['count']}, PnL: ${high_metrics['total_pnl']:+.2f}")
    print(f"     vs ALL trades:")
    print(f"       Trades: {overall['count']}, PnL: ${overall['total_pnl']:+.2f}")
    
    filtered_improvement = high_metrics['total_pnl'] - overall['total_pnl']
    print(f"     Removing low-conf trades would change PnL by: ${filtered_improvement:+.2f}")
    
    # Risk assessment
    print(f"\n  4. RISK ASSESSMENT:")
    print(f"     Max Drawdown:    ${overall['max_drawdown']:.2f}")
    print(f"     Recovery needed: {abs(overall['max_drawdown']):.2f}")
    
    # Consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for t in TRADES:
        if t.pnl < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0
    print(f"     Max consecutive losses: {max_consec_loss}")
    
    # Win streaks
    max_consec_win = 0
    current_streak = 0
    for t in TRADES:
        if t.pnl > 0:
            current_streak += 1
            max_consec_win = max(max_consec_win, current_streak)
        else:
            current_streak = 0
    print(f"     Max consecutive wins:   {max_consec_win}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Date:              June 26, 2026")
    print(f"  System:            NeuroX v4")
    print(f"  Total Trades:      {overall['count']}")
    print(f"  Net PnL:           ${overall['total_pnl']:+.2f}")
    print(f"  Win Rate:          {overall['win_rate']:.1f}%")
    print(f"  Profit Factor:     {overall['profit_factor']:.3f}")
    print(f"  Sharpe (ann.):     {overall['sharpe']:.2f}")
    print(f"  Max Drawdown:      ${overall['max_drawdown']:.2f}")
    
    if overall['total_pnl'] > 0:
        print(f"\n  VERDICT: PROFITABLE DAY (+${overall['total_pnl']:.2f})")
    else:
        print(f"\n  VERDICT: LOSING DAY (${overall['total_pnl']:.2f})")
    
    print(f"\n{'#'*60}")
    print(f"# END OF ANALYSIS")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
