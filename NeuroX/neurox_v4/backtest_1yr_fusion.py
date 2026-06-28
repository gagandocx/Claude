#!/usr/bin/env python3
"""
NeuroX v4 - 1-Year XAUUSD Backtest with Fusion Markets Conditions
==================================================================
Uses yfinance for price data but applies Fusion Markets-realistic costs:
- Spread: $0.05 (5 cents) average on XAUUSD (Fusion raw spread ~0.0 + markup)
- Commission: $3.50 per lot round trip (Fusion Raw ECN account)
- Slippage: $0.02 average (Fusion equinix servers, very low latency)
- Swap: -$3.50/lot/night for gold longs, -$1.50 for shorts (approximate)

Strategy: NeuroX core logic (EMA crossover + RSI + ATR + Session + Trailing)
Data: 1h bars for 1 year (M1 not available for >7 days from free sources)
Note: Parameters scaled for 1h timeframe. Results shown as proxy for M1.
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


# ============================================================
# FUSION MARKETS COST MODEL
# ============================================================
class FusionMarketsCosts:
    """Realistic Fusion Markets Raw ECN account costs for XAUUSD."""
    SPREAD_USD = 0.05          # Average spread in USD (5 cents)
    COMMISSION_PER_LOT_RT = 3.50  # Round-trip commission per standard lot
    SLIPPAGE_USD = 0.02        # Average slippage per fill
    SWAP_LONG_PER_LOT = -3.50  # Daily swap for long (per lot per night)
    SWAP_SHORT_PER_LOT = -1.50 # Daily swap for short (per lot per night)
    LOT_SIZE = 100             # 1 lot = 100 oz for gold
    MIN_LOT = 0.01

    @classmethod
    def total_cost_per_trade(cls, lot_size: float) -> float:
        """Total cost to open+close a trade (spread + commission + slippage)."""
        spread_cost = cls.SPREAD_USD * lot_size * cls.LOT_SIZE
        commission = cls.COMMISSION_PER_LOT_RT * lot_size
        slippage = cls.SLIPPAGE_USD * 2 * lot_size * cls.LOT_SIZE  # entry + exit
        return spread_cost + commission + slippage

    @classmethod
    def swap_cost(cls, direction: str, lot_size: float, nights: int) -> float:
        """Swap cost for holding position overnight."""
        if direction == "BUY":
            return cls.SWAP_LONG_PER_LOT * lot_size * nights
        return cls.SWAP_SHORT_PER_LOT * lot_size * nights


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data() -> pd.DataFrame:
    """Download 1 year of XAUUSD (GC=F) 1h data from yfinance."""
    print("Downloading XAUUSD (GC=F) 1h data for ~1 year...")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="1y", interval="1h")
    if df.empty:
        # Fallback: try with explicit dates
        end = datetime.now()
        start = end - timedelta(days=365)
        df = ticker.history(start=start, end=end, interval="1h")
    if df.empty:
        print("ERROR: Could not download data from yfinance.")
        sys.exit(1)
    print(f"  Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    df.columns = [c.lower() for c in df.columns]
    return df


# ============================================================
# INDICATOR CALCULATION
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators needed for the strategy."""
    # EMAs
    df["ema_9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(df["close"], window=200).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    # MACD
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ADX
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # Momentum
    df["momentum"] = df["close"].diff(8)

    # Volume MA (for volume confirmation)
    if "volume" in df.columns:
        df["vol_ma"] = df["volume"].rolling(20).mean()
    else:
        df["vol_ma"] = 0

    return df


# ============================================================
# SESSION FILTER
# ============================================================
def get_session(hour_utc: int) -> str:
    """Determine trading session from UTC hour."""
    if 8 <= hour_utc < 16:
        if 13 <= hour_utc < 16:
            return "OVERLAP"  # London/NY overlap - best liquidity
        return "LONDON"
    elif 13 <= hour_utc < 21:
        return "NEW_YORK"
    elif 0 <= hour_utc < 8:
        return "ASIAN"
    return "OFF_HOURS"


def session_multiplier(session: str) -> float:
    """Position size multiplier based on session (Fusion Markets liquidity)."""
    return {
        "OVERLAP": 1.3,
        "LONDON": 1.2,
        "NEW_YORK": 1.0,
        "ASIAN": 0.7,
        "OFF_HOURS": 0.5,
    }.get(session, 0.5)


# ============================================================
# PROGRESSIVE TRAILING STOP (4-TIER)
# ============================================================
class TrailingStop:
    """4-tier progressive trailing stop system."""

    def __init__(self, entry_price: float, direction: str, atr: float):
        self.entry_price = entry_price
        self.direction = direction
        self.atr = atr
        self.best_price = entry_price
        self.tier = 0
        # Tier thresholds (multiples of ATR from entry)
        self.tier_thresholds = [0.8, 1.5, 2.5, 4.0]
        # Trail distance per tier (wider initially, tighter as profit grows)
        self.trail_distances = [1.5, 1.0, 0.7, 0.4]  # ATR multiples

    def update(self, current_price: float) -> float:
        """Update trailing stop, return stop level."""
        if self.direction == "BUY":
            self.best_price = max(self.best_price, current_price)
            profit_atr = (self.best_price - self.entry_price) / self.atr
        else:
            self.best_price = min(self.best_price, current_price)
            profit_atr = (self.entry_price - self.best_price) / self.atr

        # Determine tier
        for i, thresh in enumerate(self.tier_thresholds):
            if profit_atr >= thresh:
                self.tier = i

        trail_dist = self.trail_distances[self.tier] * self.atr

        if self.direction == "BUY":
            return self.best_price - trail_dist
        else:
            return self.best_price + trail_dist


# ============================================================
# STRATEGY: NEUROX CORE LOGIC
# ============================================================
class NeuroXStrategy:
    """
    Core NeuroX strategy adapted for 1h timeframe:
    - EMA crossover (9/21) with trend filter (50/200)
    - RSI confirmation (not overbought/oversold against trade)
    - MACD histogram momentum
    - ADX trend strength filter
    - Session-aware sizing
    - 4-tier progressive trailing stop
    - ATR-based stop loss
    """

    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
        self.base_lot = 0.05  # Base lot size for 10k account on 1h
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_positions = 1
        self.position = None
        self.trades = []
        self.equity_curve = [account_balance]
        self.peak_equity = account_balance
        self.daily_pnl = 0.0
        self.daily_loss_limit = -100.0  # $100 daily loss limit
        self.last_trade_date = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0

    def calculate_lot_size(self, atr: float, session: str) -> float:
        """Dynamic lot sizing based on ATR, session, and streak."""
        if atr <= 0:
            return FusionMarketsCosts.MIN_LOT

        # Risk-based: risk 1% of account, SL = 2.0*ATR
        sl_distance = 2.0 * atr
        risk_amount = self.account_balance * self.risk_per_trade
        lot_size = risk_amount / (sl_distance * FusionMarketsCosts.LOT_SIZE)

        # Session adjustment
        lot_size *= session_multiplier(session)

        # Streak adjustment
        if self.consecutive_losses >= 3:
            lot_size *= 0.5
        elif self.consecutive_losses >= 5:
            lot_size *= 0.25
        elif self.consecutive_wins >= 3:
            lot_size *= 1.2

        # Clamp
        lot_size = max(FusionMarketsCosts.MIN_LOT, min(0.5, lot_size))
        return round(lot_size, 2)

    def check_entry(self, row: pd.Series, prev_row: pd.Series) -> str:
        """
        Adaptive trend-following strategy optimized for gold.
        
        Core principle: Trade with the dominant trend direction ONLY.
        Gold has strong trending characteristics - this strategy:
        1. Identifies trend via 50/200 EMA relationship
        2. Enters on momentum resumption after consolidation
        3. Uses ATR-adaptive position sizing
        4. Only trades in direction of major trend
        
        Returns 'BUY', 'SELL', or 'NONE'.
        """
        if self.position is not None:
            return "NONE"

        # Daily loss limit check
        if self.daily_pnl <= self.daily_loss_limit:
            return "NONE"

        # Need all indicators valid
        required = ["ema_9", "ema_21", "ema_50", "ema_200", "rsi",
                    "atr", "macd_hist", "adx", "momentum"]
        for col in required:
            if pd.isna(row.get(col)) or pd.isna(prev_row.get(col)):
                return "NONE"

        # ATR sanity
        if row["atr"] < 0.5:
            return "NONE"

        # === DETERMINE MAJOR TREND ===
        ema_50_above_200 = row["ema_50"] > row["ema_200"]
        price_above_50 = row["close"] > row["ema_50"]
        price_above_200 = row["close"] > row["ema_200"]

        ema_50_below_200 = row["ema_50"] < row["ema_200"]
        price_below_50 = row["close"] < row["ema_50"]
        price_below_200 = row["close"] < row["ema_200"]

        # === BUY SIGNAL: Trend is UP ===
        if ema_50_above_200 and price_above_200:
            # Entry trigger: EMA 9 crosses above EMA 21 (momentum resumes)
            ema_cross_up = (row["ema_9"] > row["ema_21"] and
                           prev_row["ema_9"] <= prev_row["ema_21"])

            # Alternative trigger: price bounces off EMA 21 with momentum
            bounce_off_21 = (
                prev_row["close"] <= prev_row["ema_21"] * 1.002 and
                row["close"] > row["ema_21"] and
                row["close"] > prev_row["close"]
            )

            # Alternative: strong momentum bar in trend direction
            strong_momentum = (
                row["momentum"] > row["atr"] * 0.8 and
                row["macd_hist"] > 0 and
                row["macd_hist"] > prev_row["macd_hist"]
            )

            if ema_cross_up or bounce_off_21 or strong_momentum:
                # Confirmation score
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

        # === SELL SIGNAL: Trend is DOWN ===
        if ema_50_below_200 and price_below_200:
            # Additional filter: 200 EMA must be declining (true downtrend)
            ema_200_declining = row["ema_200"] < prev_row["ema_200"]
            if not ema_200_declining:
                return "NONE"
            # Entry trigger: EMA 9 crosses below EMA 21
            ema_cross_down = (row["ema_9"] < row["ema_21"] and
                              prev_row["ema_9"] >= prev_row["ema_21"])

            # Alternative: rejection from EMA 21 in downtrend
            rejection_21 = (
                prev_row["close"] >= prev_row["ema_21"] * 0.998 and
                row["close"] < row["ema_21"] and
                row["close"] < prev_row["close"]
            )

            # Strong downward momentum
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

                # Higher bar for sells (need 5 not 4)
                if score >= 5:
                    return "SELL"

        return "NONE"

    def open_position(self, direction: str, price: float, atr: float,
                      session: str, timestamp):
        """Open a new position."""
        lot_size = self.calculate_lot_size(atr, session)
        sl_distance = 2.0 * atr  # Wider stop for 1h timeframe

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
        }

    def check_exit(self, row: pd.Series) -> bool:
        """Check if position should be closed. Returns True if closed."""
        if self.position is None:
            return False

        pos = self.position
        pos["bars_held"] += 1
        price = row["close"]
        direction = pos["direction"]

        # Update max favorable/adverse excursion
        if direction == "BUY":
            unrealized = price - pos["entry_price"]
        else:
            unrealized = pos["entry_price"] - price
        pos["max_favorable"] = max(pos["max_favorable"], unrealized)
        pos["max_adverse"] = min(pos["max_adverse"], unrealized)

        # Update trailing stop
        trail_level = pos["trailing_stop"].update(price)

        # === EXIT CONDITIONS ===
        exit_reason = None

        # 1. Hard stop loss hit
        if direction == "BUY" and row["low"] <= pos["sl"]:
            exit_price = pos["sl"]
            exit_reason = "STOP_LOSS"
        elif direction == "SELL" and row["high"] >= pos["sl"]:
            exit_price = pos["sl"]
            exit_reason = "STOP_LOSS"

        # 2. Trailing stop hit
        elif direction == "BUY" and row["low"] <= trail_level:
            exit_price = trail_level
            exit_reason = "TRAILING_STOP"
        elif direction == "SELL" and row["high"] >= trail_level:
            exit_price = trail_level
            exit_reason = "TRAILING_STOP"

        # 3. Max hold time (40 bars = 40 hours on 1h)
        elif pos["bars_held"] >= 40:
            exit_price = price
            exit_reason = "MAX_HOLD"

        # 4. RSI extreme reversal
        elif direction == "BUY" and not pd.isna(row.get("rsi")) and row["rsi"] > 80:
            exit_price = price
            exit_reason = "RSI_EXTREME"
        elif direction == "SELL" and not pd.isna(row.get("rsi")) and row["rsi"] < 20:
            exit_price = price
            exit_reason = "RSI_EXTREME"

        if exit_reason is None:
            # Update SL to trailing if better
            if direction == "BUY":
                pos["sl"] = max(pos["sl"], trail_level)
            else:
                pos["sl"] = min(pos["sl"], trail_level)
            return False

        # Close position
        self._close_position(exit_price, exit_reason, row.name)
        return True

    def _close_position(self, exit_price: float, reason: str, timestamp):
        """Close position and record trade."""
        pos = self.position
        direction = pos["direction"]
        lot_size = pos["lot_size"]

        # Apply exit slippage
        if direction == "BUY":
            exit_price -= FusionMarketsCosts.SLIPPAGE_USD
            raw_pnl = (exit_price - pos["entry_price"]) * lot_size * FusionMarketsCosts.LOT_SIZE
        else:
            exit_price += FusionMarketsCosts.SLIPPAGE_USD
            raw_pnl = (pos["entry_price"] - exit_price) * lot_size * FusionMarketsCosts.LOT_SIZE

        # Costs
        trade_cost = FusionMarketsCosts.total_cost_per_trade(lot_size)

        # Swap (approximate: count bars held / 24 as nights)
        nights_held = max(0, pos["bars_held"] // 24)
        swap = FusionMarketsCosts.swap_cost(direction, lot_size, nights_held)

        net_pnl = raw_pnl - trade_cost + swap

        # Update account
        self.account_balance += net_pnl
        self.equity_curve.append(self.account_balance)
        self.peak_equity = max(self.peak_equity, self.account_balance)
        self.daily_pnl += net_pnl

        # Streak tracking
        if net_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Record trade
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
        })

        self.position = None

    def reset_daily(self, current_date):
        """Reset daily P&L tracking."""
        if self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.last_trade_date = current_date


# ============================================================
# PERFORMANCE METRICS
# ============================================================
def compute_metrics(trades: list, equity_curve: list,
                    initial_balance: float) -> dict:
    """Compute comprehensive performance metrics."""
    if not trades:
        return {"error": "No trades executed"}

    pnls = [t["net_pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")

    # Risk-reward ratio
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Sharpe ratio (annualized, assuming ~252 trading days, ~6 trades/day equiv)
    if len(pnls) > 1:
        pnl_std = np.std(pnls)
        if pnl_std > 0:
            sharpe = (np.mean(pnls) / pnl_std) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Sortino ratio
    downside = [p for p in pnls if p < 0]
    if downside:
        downside_std = np.std(downside)
        sortino = (np.mean(pnls) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = float("inf")

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    max_dd_pct = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = peak - eq
        dd_pct = dd / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    # Calmar ratio
    calmar = (total_pnl / initial_balance * 100) / max_dd_pct if max_dd_pct > 0 else 0

    # Average bars held
    avg_bars = np.mean([t["bars_held"] for t in trades])

    # Consecutive stats
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for p in pnls:
        if p > 0:
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        else:
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Total costs
    total_costs = sum(t["costs"] for t in trades)
    total_swap = sum(t["swap"] for t in trades)

    # Monthly breakdown
    monthly = {}
    for t in trades:
        month = t["entry_time"][:7]
        monthly[month] = monthly.get(month, 0) + t["net_pnl"]

    # Direction breakdown
    buy_trades = [t for t in trades if t["direction"] == "BUY"]
    sell_trades = [t for t in trades if t["direction"] == "SELL"]
    buy_pnl = sum(t["net_pnl"] for t in buy_trades)
    sell_pnl = sum(t["net_pnl"] for t in sell_trades)

    return {
        "summary": {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate_pct": round(win_rate, 2),
            "total_net_pnl_usd": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / initial_balance * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "risk_reward_ratio": round(rr_ratio, 3),
        },
        "risk_metrics": {
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_usd": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
        },
        "trade_details": {
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "largest_win_usd": round(max(pnls), 2) if pnls else 0,
            "largest_loss_usd": round(min(pnls), 2) if pnls else 0,
            "avg_bars_held": round(avg_bars, 1),
            "avg_hold_hours": round(avg_bars, 1),
        },
        "costs": {
            "total_commission_and_spread_usd": round(total_costs, 2),
            "total_swap_usd": round(total_swap, 2),
            "avg_cost_per_trade_usd": round(total_costs / len(trades), 2),
            "costs_as_pct_of_gross": round(
                total_costs / abs(sum(winners)) * 100, 2
            ) if winners else 0,
        },
        "direction_breakdown": {
            "buy_trades": len(buy_trades),
            "buy_pnl_usd": round(buy_pnl, 2),
            "sell_trades": len(sell_trades),
            "sell_pnl_usd": round(sell_pnl, 2),
        },
        "exit_reasons": exit_reasons,
        "monthly_pnl": {k: round(v, 2) for k, v in monthly.items()},
        "final_balance_usd": round(initial_balance + total_pnl, 2),
        "fusion_markets_conditions": {
            "spread_usd": FusionMarketsCosts.SPREAD_USD,
            "commission_per_lot_rt": FusionMarketsCosts.COMMISSION_PER_LOT_RT,
            "slippage_usd": FusionMarketsCosts.SLIPPAGE_USD,
            "swap_long_per_lot_night": FusionMarketsCosts.SWAP_LONG_PER_LOT,
            "swap_short_per_lot_night": FusionMarketsCosts.SWAP_SHORT_PER_LOT,
        },
        "data_info": {
            "timeframe": "1h",
            "note": "1h bars used as proxy (M1 data only available for 7 days from free sources). "
                    "Fusion Markets cost model applied for realistic execution simulation.",
        },
    }


# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def run_backtest():
    """Run the full backtest."""
    print("=" * 70)
    print("  NeuroX v4 - XAUUSD 1-Year Backtest")
    print("  Broker Simulation: Fusion Markets (Raw ECN Account)")
    print("=" * 70)
    print()

    # Download data
    df = download_data()

    # Compute indicators
    print("Computing technical indicators...")
    df = compute_indicators(df)

    # Drop rows with NaN indicators (warmup period)
    df = df.dropna(subset=["ema_200", "rsi", "atr", "adx", "macd_hist"])
    print(f"  Usable bars after indicator warmup: {len(df)}")
    print()

    # Initialize strategy
    initial_balance = 10000.0
    strategy = NeuroXStrategy(account_balance=initial_balance)

    # Run backtest
    print("Running backtest...")
    prev_row = None
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        if prev_row is None:
            prev_row = row
            continue

        # Reset daily P&L
        current_date = timestamp.date() if hasattr(timestamp, "date") else None
        if current_date:
            strategy.reset_daily(current_date)

        # Get session
        hour_utc = timestamp.hour if hasattr(timestamp, "hour") else 12
        session = get_session(hour_utc)

        # Check exit first
        if strategy.position is not None:
            strategy.check_exit(row)

        # Check entry
        signal = strategy.check_entry(row, prev_row)
        if signal != "NONE":
            strategy.open_position(
                direction=signal,
                price=row["close"],
                atr=row["atr"],
                session=session,
                timestamp=timestamp,
            )

        prev_row = row

    # Force close any open position at end
    if strategy.position is not None:
        last_row = df.iloc[-1]
        strategy._close_position(last_row["close"], "END_OF_TEST", df.index[-1])

    # Compute metrics
    print(f"\n  Total trades executed: {len(strategy.trades)}")
    metrics = compute_metrics(strategy.trades, strategy.equity_curve, initial_balance)

    # Print results
    print()
    print("=" * 70)
    print("  BACKTEST RESULTS - Fusion Markets Conditions")
    print("=" * 70)
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
    for reason, count in sorted(metrics["exit_reasons"].items(),
                                 key=lambda x: -x[1]):
        print(f"  {reason:<20} {count:>4} ({count/s['total_trades']*100:.1f}%)")
    print()

    print(f"  {'MONTHLY P&L':=^50}")
    for month, pnl in sorted(metrics["monthly_pnl"].items()):
        bar = "+" * int(abs(pnl) / 10) if pnl > 0 else "-" * int(abs(pnl) / 10)
        print(f"  {month}: ${pnl:>+8.2f}  {bar}")
    print()

    print("=" * 70)
    print("  NOTE: Uses 1h bars as proxy for M1 (free data limited to 7 days).")
    print("  Costs modeled on Fusion Markets Raw ECN (tight spreads, low comm).")
    print("  Spread $0.05 + Commission $3.50/lot RT + Slippage $0.02 per fill.")
    print()
    print("  IMPORTANT: This is a RULE-BASED strategy (no trained ML models).")
    print("  The NeuroX 17-model ensemble requires trained checkpoints for")
    print("  better signal generation. Rule-based logic alone struggles with")
    print("  gold's high volatility and whipsaw behavior.")
    print("=" * 70)

    # Save results
    output_path = Path(__file__).parent / "backtest_results_1yr.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    run_backtest()