#!/usr/bin/env python3
"""
NeuroX v4 - Filtered 1-Year XAUUSD Backtest
============================================
Same strategy as backtest_1yr_fusion.py but with TWO new filters:
1. Skip overlap session entirely (13:00-16:00 UTC) - no new trades
2. Confidence threshold: BUY >= 0.55, SELL >= 0.60

Confidence is estimated from indicator alignment strength.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

# Import shared components from original backtest
from backtest_1yr_fusion import (
    FusionMarketsCosts, download_data, compute_indicators,
    TrailingStop, compute_metrics
)


# ============================================================
# CONFIDENCE SCORING (simulated from indicator alignment)
# ============================================================
def compute_confidence(row, prev_row, direction):
    """
    Estimate model confidence from indicator alignment.
    Since we don't have actual ML model inference for historical data,
    we simulate confidence based on how many technical indicators align
    with the trade direction. Score ranges from 0.0 to 1.0.
    """
    if direction == "NONE":
        return 0.0

    score = 0
    max_score = 10

    if direction == "BUY":
        # Trend alignment
        if row["ema_50"] > row["ema_200"]:
            score += 1
        if row["close"] > row["ema_200"]:
            score += 1
        if row["close"] > row["ema_50"]:
            score += 1
        # Momentum
        if row["ema_9"] > row["ema_21"]:
            score += 1
        if row["macd_hist"] > 0:
            score += 1
        if row["macd_hist"] > prev_row["macd_hist"]:
            score += 1
        # RSI not overbought but positive
        if 45 < row["rsi"] < 70:
            score += 1
        # ADX showing trend
        if row["adx"] > 25:
            score += 1
        # Momentum positive
        if row["momentum"] > 0:
            score += 1
        # Price above BB middle
        if row["close"] > row["bb_mid"]:
            score += 1

    elif direction == "SELL":
        # Trend alignment
        if row["ema_50"] < row["ema_200"]:
            score += 1
        if row["close"] < row["ema_200"]:
            score += 1
        if row["close"] < row["ema_50"]:
            score += 1
        # Momentum
        if row["ema_9"] < row["ema_21"]:
            score += 1
        if row["macd_hist"] < 0:
            score += 1
        if row["macd_hist"] < prev_row["macd_hist"]:
            score += 1
        # RSI not oversold but negative
        if 30 < row["rsi"] < 55:
            score += 1
        # ADX showing trend
        if row["adx"] > 25:
            score += 1
        # Momentum negative
        if row["momentum"] < 0:
            score += 1
        # Price below BB middle
        if row["close"] < row["bb_mid"]:
            score += 1

    confidence = score / max_score
    return confidence


# ============================================================
# SESSION FILTER (with overlap skip)
# ============================================================
def get_session_filtered(hour_utc):
    """Determine trading session from UTC hour."""
    if 13 <= hour_utc < 16:
        return "OVERLAP"
    elif 8 <= hour_utc < 16:
        return "LONDON"
    elif 13 <= hour_utc < 21:
        return "NEW_YORK"
    elif 0 <= hour_utc < 8:
        return "ASIAN"
    return "OFF_HOURS"


def session_multiplier(session):
    """Position size multiplier based on session."""
    return {
        "OVERLAP": 1.3,
        "LONDON": 1.2,
        "NEW_YORK": 1.0,
        "ASIAN": 0.7,
        "OFF_HOURS": 0.5,
    }.get(session, 0.5)


# ============================================================
# FILTERED STRATEGY
# ============================================================
class NeuroXFilteredStrategy:
    """
    NeuroX strategy with overlap skip and confidence filters.
    - NO trades during 13:00-16:00 UTC (overlap session)
    - BUY trades require confidence >= 0.55
    - SELL trades require confidence >= 0.60
    """

    # Confidence thresholds
    BUY_CONFIDENCE_MIN = 0.55
    SELL_CONFIDENCE_MIN = 0.60

    def __init__(self, account_balance=10000.0):
        self.account_balance = account_balance
        self.base_lot = 0.05
        self.risk_per_trade = 0.01
        self.max_positions = 1
        self.position = None
        self.trades = []
        self.equity_curve = [account_balance]
        self.peak_equity = account_balance
        self.daily_pnl = 0.0
        self.daily_loss_limit = -100.0
        self.last_trade_date = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        # Filter stats
        self.signals_generated = 0
        self.signals_skipped_overlap = 0
        self.signals_skipped_confidence = 0
        self.confidence_scores = []

    def calculate_lot_size(self, atr, session):
        """Dynamic lot sizing based on ATR, session, and streak."""
        if atr <= 0:
            return FusionMarketsCosts.MIN_LOT
        sl_distance = 2.0 * atr
        risk_amount = self.account_balance * self.risk_per_trade
        lot_size = risk_amount / (sl_distance * FusionMarketsCosts.LOT_SIZE)
        lot_size *= session_multiplier(session)
        if self.consecutive_losses >= 5:
            lot_size *= 0.25
        elif self.consecutive_losses >= 3:
            lot_size *= 0.5
        elif self.consecutive_wins >= 3:
            lot_size *= 1.2
        lot_size = max(FusionMarketsCosts.MIN_LOT, min(0.5, lot_size))
        return round(lot_size, 2)

    def check_entry_raw(self, row, prev_row):
        """
        Core signal logic (same as original NeuroX).
        Returns 'BUY', 'SELL', or 'NONE'.
        """
        if self.position is not None:
            return "NONE"
        if self.daily_pnl <= self.daily_loss_limit:
            return "NONE"
        required = ["ema_9", "ema_21", "ema_50", "ema_200", "rsi",
                    "atr", "macd_hist", "adx", "momentum"]
        for col in required:
            if pd.isna(row.get(col)) or pd.isna(prev_row.get(col)):
                return "NONE"
        if row["atr"] < 0.5:
            return "NONE"

        ema_50_above_200 = row["ema_50"] > row["ema_200"]
        price_above_200 = row["close"] > row["ema_200"]
        ema_50_below_200 = row["ema_50"] < row["ema_200"]
        price_below_200 = row["close"] < row["ema_200"]

        # BUY SIGNAL
        if ema_50_above_200 and price_above_200:
            ema_cross_up = (row["ema_9"] > row["ema_21"] and
                           prev_row["ema_9"] <= prev_row["ema_21"])
            bounce_off_21 = (
                prev_row["close"] <= prev_row["ema_21"] * 1.002 and
                row["close"] > row["ema_21"] and
                row["close"] > prev_row["close"]
            )
            strong_momentum = (
                row["momentum"] > row["atr"] * 0.8 and
                row["macd_hist"] > 0 and
                row["macd_hist"] > prev_row["macd_hist"]
            )
            if ema_cross_up or bounce_off_21 or strong_momentum:
                score = 0
                if row["rsi"] > 45 and row["rsi"] < 75:
                    score += 1
                if row["macd_hist"] > 0:
                    score += 1
                if row["macd_hist"] > prev_row["macd_hist"]:
                    score += 1
                if row["adx"] > 20:
                    score += 1
                if row["ema_9"] > row["ema_21"]:
                    score += 1
                if row["close"] > row["ema_9"]:
                    score += 1
                if score >= 4:
                    return "BUY"

        # SELL SIGNAL
        if ema_50_below_200 and price_below_200:
            ema_200_declining = row["ema_200"] < prev_row["ema_200"]
            if not ema_200_declining:
                return "NONE"
            ema_cross_down = (row["ema_9"] < row["ema_21"] and
                              prev_row["ema_9"] >= prev_row["ema_21"])
            rejection_21 = (
                prev_row["close"] >= prev_row["ema_21"] * 0.998 and
                row["close"] < row["ema_21"] and
                row["close"] < prev_row["close"]
            )
            strong_momentum = (
                row["momentum"] < -row["atr"] * 0.8 and
                row["macd_hist"] < 0 and
                row["macd_hist"] < prev_row["macd_hist"]
            )
            if ema_cross_down or rejection_21 or strong_momentum:
                score = 0
                if row["rsi"] < 55 and row["rsi"] > 25:
                    score += 1
                if row["macd_hist"] < 0:
                    score += 1
                if row["macd_hist"] < prev_row["macd_hist"]:
                    score += 1
                if row["adx"] > 20:
                    score += 1
                if row["ema_9"] < row["ema_21"]:
                    score += 1
                if row["close"] < row["ema_9"]:
                    score += 1
                if score >= 5:
                    return "SELL"

        return "NONE"

    def check_entry_filtered(self, row, prev_row, hour_utc):
        """
        Entry check with overlap skip and confidence filter.
        Returns (direction, confidence) or ('NONE', 0.0).
        """
        # FILTER 1: Skip overlap session (13:00-16:00 UTC)
        session = get_session_filtered(hour_utc)
        if session == "OVERLAP":
            # Still check for raw signal to count skipped
            raw_signal = self.check_entry_raw(row, prev_row)
            if raw_signal != "NONE":
                self.signals_generated += 1
                self.signals_skipped_overlap += 1
            return "NONE", 0.0

        # Get raw signal
        raw_signal = self.check_entry_raw(row, prev_row)
        if raw_signal == "NONE":
            return "NONE", 0.0

        self.signals_generated += 1

        # FILTER 2: Confidence threshold
        confidence = compute_confidence(row, prev_row, raw_signal)
        self.confidence_scores.append(confidence)

        if raw_signal == "BUY" and confidence < self.BUY_CONFIDENCE_MIN:
            self.signals_skipped_confidence += 1
            return "NONE", confidence

        if raw_signal == "SELL" and confidence < self.SELL_CONFIDENCE_MIN:
            self.signals_skipped_confidence += 1
            return "NONE", confidence

        return raw_signal, confidence

    def open_position(self, direction, price, atr, session, timestamp, confidence):
        """Open a new position."""
        lot_size = self.calculate_lot_size(atr, session)
        sl_distance = 2.0 * atr
        if direction == "BUY":
            entry_price = price + FusionMarketsCosts.SPREAD_USD / 2
            sl = entry_price - sl_distance
        else:
            entry_price = price - FusionMarketsCosts.SPREAD_USD / 2
            sl = entry_price + sl_distance
        self.position = {
            "direction": direction,
            "entry_price": entry_price,
            "lot_size": lot_size,
            "sl": sl,
            "atr": atr,
            "trailing_stop": TrailingStop(entry_price, direction, atr),
            "entry_time": timestamp,
            "bars_held": 0,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "confidence": confidence,
        }

    def check_exit(self, row):
        """Check if position should be closed. Returns True if closed."""
        if self.position is None:
            return False
        pos = self.position
        pos["bars_held"] += 1
        price = row["close"]
        direction = pos["direction"]
        if direction == "BUY":
            unrealized = price - pos["entry_price"]
        else:
            unrealized = pos["entry_price"] - price
        pos["max_favorable"] = max(pos["max_favorable"], unrealized)
        pos["max_adverse"] = min(pos["max_adverse"], unrealized)
        trail_level = pos["trailing_stop"].update(price)

        exit_reason = None
        exit_price = None

        if direction == "BUY" and row["low"] <= pos["sl"]:
            exit_price = pos["sl"]
            exit_reason = "STOP_LOSS"
        elif direction == "SELL" and row["high"] >= pos["sl"]:
            exit_price = pos["sl"]
            exit_reason = "STOP_LOSS"
        elif direction == "BUY" and row["low"] <= trail_level:
            exit_price = trail_level
            exit_reason = "TRAILING_STOP"
        elif direction == "SELL" and row["high"] >= trail_level:
            exit_price = trail_level
            exit_reason = "TRAILING_STOP"
        elif pos["bars_held"] >= 40:
            exit_price = price
            exit_reason = "MAX_HOLD"
        elif direction == "BUY" and not pd.isna(row.get("rsi")) and row["rsi"] > 80:
            exit_price = price
            exit_reason = "RSI_EXTREME"
        elif direction == "SELL" and not pd.isna(row.get("rsi")) and row["rsi"] < 20:
            exit_price = price
            exit_reason = "RSI_EXTREME"

        if exit_reason is None:
            if direction == "BUY":
                pos["sl"] = max(pos["sl"], trail_level)
            else:
                pos["sl"] = min(pos["sl"], trail_level)
            return False

        self._close_position(exit_price, exit_reason, row.name)
        return True

    def _close_position(self, exit_price, reason, timestamp):
        """Close position and record trade."""
        pos = self.position
        direction = pos["direction"]
        lot_size = pos["lot_size"]
        if direction == "BUY":
            exit_price -= FusionMarketsCosts.SLIPPAGE_USD
            raw_pnl = (exit_price - pos["entry_price"]) * lot_size * FusionMarketsCosts.LOT_SIZE
        else:
            exit_price += FusionMarketsCosts.SLIPPAGE_USD
            raw_pnl = (pos["entry_price"] - exit_price) * lot_size * FusionMarketsCosts.LOT_SIZE
        trade_cost = FusionMarketsCosts.total_cost_per_trade(lot_size)
        nights_held = max(0, pos["bars_held"] // 24)
        swap = FusionMarketsCosts.swap_cost(direction, lot_size, nights_held)
        net_pnl = raw_pnl - trade_cost + swap
        self.account_balance += net_pnl
        self.equity_curve.append(self.account_balance)
        self.peak_equity = max(self.peak_equity, self.account_balance)
        self.daily_pnl += net_pnl
        if net_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        self.trades.append({
            "direction": direction,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "lot_size": lot_size,
            "raw_pnl": raw_pnl,
            "costs": trade_cost,
            "swap": swap,
            "net_pnl": net_pnl,
            "bars_held": pos["bars_held"],
            "exit_reason": reason,
            "entry_time": str(pos["entry_time"]),
            "exit_time": str(timestamp),
            "max_favorable_excursion": pos["max_favorable"],
            "max_adverse_excursion": pos["max_adverse"],
            "atr_at_entry": pos["atr"],
            "confidence": pos.get("confidence", 0),
        })
        self.position = None

    def reset_daily(self, current_date):
        """Reset daily P&L tracking."""
        if self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.last_trade_date = current_date


# ============================================================
# COMPARISON TABLE
# ============================================================
def print_comparison(original_results, filtered_metrics, filter_stats):
    """Print side-by-side comparison of original vs filtered results."""
    print()
    print("=" * 72)
    print("  COMPARISON: Original (No Filters) vs Filtered (Overlap Skip + Conf)")
    print("=" * 72)
    print()
    print(f"  {'Metric':<30} {'Original':<20} {'Filtered':<20}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")

    o = original_results
    f = filtered_metrics

    rows = [
        ("Total Trades", o["summary"]["total_trades"], f["summary"]["total_trades"]),
        ("Win Rate %", f'{o["summary"]["win_rate_pct"]}%', f'{f["summary"]["win_rate_pct"]}%'),
        ("Total Net P&L", f'${o["summary"]["total_net_pnl_usd"]:+.2f}', f'${f["summary"]["total_net_pnl_usd"]:+.2f}'),
        ("Profit Factor", o["summary"]["profit_factor"], f["summary"]["profit_factor"]),
        ("Sharpe Ratio", o["risk_metrics"]["sharpe_ratio"], f["risk_metrics"]["sharpe_ratio"]),
        ("Max Drawdown $", f'${o["risk_metrics"]["max_drawdown_usd"]:.2f}', f'${f["risk_metrics"]["max_drawdown_usd"]:.2f}'),
        ("Max Drawdown %", f'{o["risk_metrics"]["max_drawdown_pct"]:.2f}%', f'{f["risk_metrics"]["max_drawdown_pct"]:.2f}%'),
        ("Avg Win", f'${o["trade_details"]["avg_win_usd"]:.2f}', f'${f["trade_details"]["avg_win_usd"]:.2f}'),
        ("Avg Loss", f'${o["trade_details"]["avg_loss_usd"]:.2f}', f'${f["trade_details"]["avg_loss_usd"]:.2f}'),
        ("R:R Ratio", o["summary"]["risk_reward_ratio"], f["summary"]["risk_reward_ratio"]),
        ("Avg Hold (hours)", o["trade_details"]["avg_hold_hours"], f["trade_details"]["avg_hold_hours"]),
        ("Total Costs", f'${o["costs"]["total_commission_and_spread_usd"]:.2f}', f'${f["costs"]["total_commission_and_spread_usd"]:.2f}'),
    ]

    for label, orig, filt in rows:
        print(f"  {label:<30} {str(orig):<20} {str(filt):<20}")

    print()
    print(f"  {'FILTER IMPACT':=^50}")
    print(f"  Signals generated:           {filter_stats['signals_generated']}")
    print(f"  Skipped (overlap):           {filter_stats['signals_skipped_overlap']}")
    print(f"  Skipped (low confidence):    {filter_stats['signals_skipped_confidence']}")
    total_skipped = filter_stats['signals_skipped_overlap'] + filter_stats['signals_skipped_confidence']
    pct_skipped = (total_skipped / filter_stats['signals_generated'] * 100) if filter_stats['signals_generated'] > 0 else 0
    print(f"  Total signals skipped:       {total_skipped} ({pct_skipped:.1f}%)")
    print(f"  Trades taken:                {filter_stats['trades_taken']}")
    if filter_stats['avg_confidence'] > 0:
        print(f"  Avg confidence (all sigs):   {filter_stats['avg_confidence']:.3f}")
        print(f"  Avg confidence (taken):      {filter_stats['avg_confidence_taken']:.3f}")
    print()

    # Improvement summary
    pnl_diff = f["summary"]["total_net_pnl_usd"] - o["summary"]["total_net_pnl_usd"]
    wr_diff = f["summary"]["win_rate_pct"] - o["summary"]["win_rate_pct"]
    dd_diff = f["risk_metrics"]["max_drawdown_usd"] - o["risk_metrics"]["max_drawdown_usd"]

    print(f"  {'IMPROVEMENT SUMMARY':=^50}")
    print(f"  P&L Change:              ${pnl_diff:+.2f}")
    print(f"  Win Rate Change:         {wr_diff:+.2f}%")
    print(f"  Drawdown Change:         ${dd_diff:+.2f} ({'better' if dd_diff < 0 else 'worse'})")
    trades_reduced = o["summary"]["total_trades"] - f["summary"]["total_trades"]
    print(f"  Trades Reduced:          {trades_reduced} ({trades_reduced/o['summary']['total_trades']*100:.1f}%)")
    print()


# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def run_backtest():
    """Run the filtered backtest."""
    print("=" * 72)
    print("  NeuroX v4 - FILTERED XAUUSD 1-Year Backtest")
    print("  Filters: Skip Overlap (13:00-16:00 UTC) + Min Confidence")
    print("  Broker Simulation: Fusion Markets (Raw ECN Account)")
    print("=" * 72)
    print()
    print("  Filter Settings:")
    print(f"    - Overlap Skip:     13:00-16:00 UTC (no new trades)")
    print(f"    - BUY Confidence:   >= {NeuroXFilteredStrategy.BUY_CONFIDENCE_MIN}")
    print(f"    - SELL Confidence:  >= {NeuroXFilteredStrategy.SELL_CONFIDENCE_MIN}")
    print()

    # Download data
    df = download_data()

    # Compute indicators
    print("Computing technical indicators...")
    df = compute_indicators(df)
    df = df.dropna(subset=["ema_200", "rsi", "atr", "adx", "macd_hist", "bb_mid"])
    print(f"  Usable bars after indicator warmup: {len(df)}")
    print()

    # Initialize strategy
    initial_balance = 10000.0
    strategy = NeuroXFilteredStrategy(account_balance=initial_balance)

    # Run backtest
    print("Running filtered backtest...")
    prev_row = None
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        if prev_row is None:
            prev_row = row
            continue

        current_date = timestamp.date() if hasattr(timestamp, "date") else None
        if current_date:
            strategy.reset_daily(current_date)

        hour_utc = timestamp.hour if hasattr(timestamp, "hour") else 12

        # Check exit first
        if strategy.position is not None:
            strategy.check_exit(row)

        # Check entry with filters
        signal, confidence = strategy.check_entry_filtered(row, prev_row, hour_utc)
        if signal != "NONE":
            session = get_session_filtered(hour_utc)
            strategy.open_position(
                direction=signal,
                price=row["close"],
                atr=row["atr"],
                session=session,
                timestamp=timestamp,
                confidence=confidence,
            )

        prev_row = row

    # Force close any open position
    if strategy.position is not None:
        last_row = df.iloc[-1]
        strategy._close_position(last_row["close"], "END_OF_TEST", df.index[-1])

    # Compute metrics
    print(f"\n  Total trades executed: {len(strategy.trades)}")
    metrics = compute_metrics(strategy.trades, strategy.equity_curve, initial_balance)

    # Filter statistics
    taken_confidences = [t["confidence"] for t in strategy.trades if "confidence" in t]
    filter_stats = {
        "signals_generated": strategy.signals_generated,
        "signals_skipped_overlap": strategy.signals_skipped_overlap,
        "signals_skipped_confidence": strategy.signals_skipped_confidence,
        "trades_taken": len(strategy.trades),
        "avg_confidence": np.mean(strategy.confidence_scores) if strategy.confidence_scores else 0,
        "avg_confidence_taken": np.mean(taken_confidences) if taken_confidences else 0,
    }

    # Add filter info to metrics
    metrics["filter_settings"] = {
        "overlap_skip": "13:00-16:00 UTC (no new trades)",
        "buy_confidence_min": NeuroXFilteredStrategy.BUY_CONFIDENCE_MIN,
        "sell_confidence_min": NeuroXFilteredStrategy.SELL_CONFIDENCE_MIN,
        "confidence_method": "Simulated from indicator alignment (10 factors)",
    }
    metrics["filter_stats"] = filter_stats

    # Print results
    print()
    print("=" * 72)
    print("  FILTERED BACKTEST RESULTS - Fusion Markets Conditions")
    print("=" * 72)
    print()

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return metrics

    s = metrics["summary"]
    r = metrics["risk_metrics"]
    t = metrics["trade_details"]
    c = metrics["costs"]
    d = metrics["direction_breakdown"]

    print(f"  {'PERFORMANCE SUMMARY':=^50}")
    print(f"  Total Trades:          {s['total_trades']}")
    print(f"  Win Rate:              {s['win_rate_pct']}%")
    print(f"  Profit Factor:         {s['profit_factor']}")
    print(f"  Risk/Reward:           {s['risk_reward_ratio']}")
    print(f"  Total Net P&L:         ${s['total_net_pnl_usd']:+.2f}")
    print(f"  Total Return:          {s['total_return_pct']:+.2f}%")
    print(f"  Final Balance:         ${metrics['final_balance_usd']:,.2f}")
    print()

    print(f"  {'RISK METRICS':=^50}")
    print(f"  Sharpe Ratio:          {r['sharpe_ratio']}")
    print(f"  Sortino Ratio:         {r['sortino_ratio']}")
    print(f"  Calmar Ratio:          {r['calmar_ratio']}")
    print(f"  Max Drawdown:          ${r['max_drawdown_usd']:.2f} ({r['max_drawdown_pct']:.2f}%)")
    print(f"  Max Consec Wins:       {r['max_consecutive_wins']}")
    print(f"  Max Consec Losses:     {r['max_consecutive_losses']}")
    print()

    print(f"  {'TRADE DETAILS':=^50}")
    print(f"  Avg Win:               ${t['avg_win_usd']:+.2f}")
    print(f"  Avg Loss:              ${t['avg_loss_usd']:+.2f}")
    print(f"  Largest Win:           ${t['largest_win_usd']:+.2f}")
    print(f"  Largest Loss:          ${t['largest_loss_usd']:+.2f}")
    print(f"  Avg Hold Time:         {t['avg_hold_hours']:.1f} hours")
    print()

    print(f"  {'FUSION MARKETS COSTS':=^50}")
    print(f"  Total Costs (spread+comm+slip): ${c['total_commission_and_spread_usd']:.2f}")
    print(f"  Total Swap:            ${c['total_swap_usd']:.2f}")
    print(f"  Avg Cost/Trade:        ${c['avg_cost_per_trade_usd']:.2f}")
    print(f"  Costs as % of Gross:   {c['costs_as_pct_of_gross']:.1f}%")
    print()

    print(f"  {'DIRECTION BREAKDOWN':=^50}")
    print(f"  BUY trades:  {d['buy_trades']}  (P&L: ${d['buy_pnl_usd']:+.2f})")
    print(f"  SELL trades: {d['sell_trades']}  (P&L: ${d['sell_pnl_usd']:+.2f})")
    print()

    print(f"  {'EXIT REASONS':=^50}")
    for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason:<20} {count:>4} ({count/s['total_trades']*100:.1f}%)")
    print()

    print(f"  {'MONTHLY P&L':=^50}")
    for month, pnl in sorted(metrics["monthly_pnl"].items()):
        bar = "+" * int(abs(pnl) / 10) if pnl > 0 else "-" * int(abs(pnl) / 10)
        print(f"  {month}: ${pnl:>+8.2f}  {bar}")
    print()

    # Load original results for comparison
    original_path = Path(__file__).parent / "backtest_results_1yr.json"
    if original_path.exists():
        with open(original_path) as f:
            original_results = json.load(f)
        print_comparison(original_results, metrics, filter_stats)
    else:
        print("  (Original results file not found for comparison)")
    print()

    print("=" * 72)
    print("  NOTES:")
    print("  - Overlap skip removes ALL new entries during 13:00-16:00 UTC")
    print("  - Confidence estimated from 10-factor indicator alignment score")
    print("  - BUY threshold: 0.55 (6/10 factors), SELL: 0.60 (6/10 factors)")
    print("  - Existing positions still managed (exits allowed) during overlap")
    print("  - Fusion Markets costs: Spread $0.05 + Comm $3.50/lot + Slip $0.02")
    print("=" * 72)

    # Save results
    output_path = Path(__file__).parent / "backtest_results_1yr_filtered.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    run_backtest()
