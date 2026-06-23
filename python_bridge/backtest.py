"""
=============================================================
  Python ML Bridge - Fast-Forward Backtester
  Simulates the exact trading strategy bar-by-bar on historical
  data downloaded from yfinance or real MT5 exported data,
  records trades to the AutoOptimizer, and outputs a detailed
  summary plus JSON results.

  Supports:
  - Real broker data via --data-file (CSV exported from MT5)
  - Variable spreads per bar from broker data
  - Fixed spread simulation via --spread
  - Random slippage simulation via --slippage
  - Commission per lot via --commission and --lot-size
=============================================================
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import SessionConfig, DataConfig, AutoOptimizerConfig
from strategies.auto_optimizer import AutoOptimizer
from data.market_data import MarketDataFetcher


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_SL_DISTANCE = 5.0       # $5 stop loss (balanced for gold M1)
MAX_POSITIONS = 1               # Only 1 position at a time (maximum selectivity)
MAX_HOLD_BARS = 90              # Max bars to hold a trade (1.5 hours on M1)
MOMENTUM_LOOKBACK = 8           # 8 bars for momentum detection
MOMENTUM_THRESHOLD = 3.00       # $3.00 for momentum direction
MOMENTUM_EXIT_THRESHOLD = 2.50  # $2.50 reversal triggers exit
MIN_BARS_BETWEEN_ENTRIES = 60   # Minimum bars between entries (1 hour cooldown)
MAX_TRADES_PER_DAY = 3          # Maximum trades per calendar day
RSI_PERIOD = 14
RSI_OVERBOUGHT = 62
RSI_OVERSOLD = 38

# Progressive trailing stop thresholds
TRAIL_BE_THRESHOLD = 2.0        # Move SL to break-even at $2.0 profit
TRAIL_TIER1_THRESHOLD = 3.5     # Trail $2.0 behind at $3.5 profit
TRAIL_TIER1_DISTANCE = 2.0
TRAIL_TIER2_THRESHOLD = 5.5     # Trail $1.5 behind at $5.5 profit
TRAIL_TIER2_DISTANCE = 1.5
TRAIL_TIER3_THRESHOLD = 8.0     # Trail $1.0 behind at $8.0 profit
TRAIL_TIER3_DISTANCE = 1.0


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────
class Position:
    """Represents an open simulated trade position."""

    def __init__(self, direction: str, entry_price: float, sl_price: float,
                 entry_bar_index: int, entry_time: str, rsi_at_entry: float,
                 session: str, confidence: float = 0.5):
        self.direction = direction
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.entry_bar_index = entry_bar_index
        self.entry_time = entry_time
        self.rsi_at_entry = rsi_at_entry
        self.session = session
        self.confidence = confidence
        self.max_profit = 0.0
        self.trail_tier = "none"

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P/L at current price."""
        if self.direction == "BUY":
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price


# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────
def compute_momentum_direction(closes: pd.Series, index: int) -> str:
    """
    Compute momentum direction from last MOMENTUM_LOOKBACK bars.

    Compares close[-1] to close[-6] (5 bars back).
    Returns "BUY", "SELL", or "FLAT".
    """
    if index < MOMENTUM_LOOKBACK:
        return "FLAT"

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    diff = current_close - lookback_close

    if diff > MOMENTUM_THRESHOLD:
        return "BUY"
    elif diff < -MOMENTUM_THRESHOLD:
        return "SELL"
    else:
        return "FLAT"


def compute_momentum_magnitude(closes: pd.Series, index: int) -> float:
    """
    Compute the absolute momentum magnitude for synthetic confidence.

    Returns a value between 0.0 and 1.0 based on momentum strength:
    - Momentum at threshold ($0.50) maps to ~0.3 confidence
    - Momentum at $1.50 maps to ~0.7 confidence
    - Momentum at $3.0+ maps to ~0.9 confidence

    Uses a sigmoid-like scaling: confidence = 0.2 + 0.7 * (1 - exp(-abs_diff / scale))
    """
    if index < MOMENTUM_LOOKBACK:
        return 0.0

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    abs_diff = abs(current_close - lookback_close)

    # Scale factor: momentum of $1.50 yields ~0.7 confidence
    scale = 1.5
    confidence = 0.2 + 0.7 * (1.0 - np.exp(-abs_diff / scale))
    return round(min(confidence, 0.95), 4)


def detect_session(timestamp) -> str:
    """
    Detect trading session from bar timestamp UTC hour.

    Uses SessionConfig ranges:
    - Asian: 0-8 UTC
    - London: 8-16 UTC
    - New York: 13-21 UTC
    - Overlap (London/NY): 13-16 UTC
    """
    session_config = SessionConfig()

    if hasattr(timestamp, 'hour'):
        hour = timestamp.hour
    else:
        hour = pd.Timestamp(timestamp).hour

    # Check overlap first (London/NY overlap)
    if session_config.ny_start <= hour < session_config.london_end:
        return "overlap"
    elif session_config.asian_start <= hour < session_config.asian_end:
        return "asian"
    elif session_config.london_start <= hour < session_config.london_end:
        return "london"
    elif session_config.ny_start <= hour < session_config.ny_end:
        return "newyork"
    else:
        return "off_session"


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI using the ta library."""
    rsi = ta.momentum.RSIIndicator(close=closes, window=period).rsi()
    return rsi


def apply_progressive_trailing(position: Position, current_price: float) -> float:
    """
    Apply progressive trailing stop logic.

    - At $0.50 unrealized profit: move SL to break-even
    - At $1.0 profit: trail $0.50 behind current price
    - At $2.0 profit: trail $0.30 behind current price
    - At $3.0 profit: trail $0.20 behind current price

    Returns the updated SL price.
    """
    pnl = position.unrealized_pnl(current_price)
    position.max_profit = max(position.max_profit, pnl)

    sl_price = position.sl_price

    if pnl >= TRAIL_TIER3_THRESHOLD:
        # Trail $0.20 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER3_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER3_DISTANCE
        position.trail_tier = "tight"
    elif pnl >= TRAIL_TIER2_THRESHOLD:
        # Trail $0.30 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER2_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER2_DISTANCE
        position.trail_tier = "medium"
    elif pnl >= TRAIL_TIER1_THRESHOLD:
        # Trail $0.50 behind
        if position.direction == "BUY":
            new_sl = current_price - TRAIL_TIER1_DISTANCE
        else:
            new_sl = current_price + TRAIL_TIER1_DISTANCE
        position.trail_tier = "wide"
    elif pnl >= TRAIL_BE_THRESHOLD:
        # Move SL to break-even
        new_sl = position.entry_price
        if position.trail_tier == "none":
            position.trail_tier = "breakeven"
    else:
        return sl_price

    # Only move SL in favorable direction (never move it further from price)
    if position.direction == "BUY":
        sl_price = max(sl_price, new_sl)
    else:
        sl_price = min(sl_price, new_sl)

    position.sl_price = sl_price
    return sl_price


def check_sl_hit(position: Position, bar_low: float, bar_high: float) -> Tuple[bool, float]:
    """
    Check if stop loss was hit during this bar.

    For BUY: if bar.Low <= sl_price, exit at sl_price
    For SELL: if bar.High >= sl_price, exit at sl_price

    Returns (hit, exit_price).
    """
    if position.direction == "BUY":
        if bar_low <= position.sl_price:
            return True, position.sl_price
    else:
        if bar_high >= position.sl_price:
            return True, position.sl_price
    return False, 0.0


def check_momentum_exit(position: Position, closes: pd.Series, index: int) -> bool:
    """
    Check if momentum has reversed against the position while near stop loss.

    Only triggers exit if:
    1. Momentum has reversed significantly ($1.50+ against position)
    2. The trade is already near the stop loss (unrealized PnL < -$4.50)

    With a $5 stop loss and progressive trailing stops, all exit management
    is handled by the trailing stop system. This is a near-disabled safety
    valve that only triggers when price is almost at the stop AND momentum
    confirms we should cut before the full stop hit.
    """
    if index < MOMENTUM_LOOKBACK:
        return False

    current_close = closes.iloc[index]
    lookback_close = closes.iloc[index - MOMENTUM_LOOKBACK]
    diff = current_close - lookback_close

    # Only exit if near stop loss level (90% of SL distance reached)
    # and momentum confirms the trade direction has fully failed
    current_pnl = position.unrealized_pnl(current_close)
    if current_pnl >= -4.5:
        return False  # Let trailing stop manage everything else

    if position.direction == "BUY" and diff < -MOMENTUM_EXIT_THRESHOLD:
        return True
    elif position.direction == "SELL" and diff > MOMENTUM_EXIT_THRESHOLD:
        return True

    return False


def detect_price_structure(df: pd.DataFrame, index: int, lookback: int = 20) -> str:
    """
    Detect price action structure (trend) from swing highs and lows.

    Analyzes the last `lookback` bars for higher highs + higher lows (uptrend)
    or lower highs + lower lows (downtrend).

    Returns "uptrend", "downtrend", or "no_structure".
    """
    if index < lookback:
        return "no_structure"

    highs = df["High"].values[index - lookback:index]
    lows = df["Low"].values[index - lookback:index]

    # Find swing highs (local maxima over 3-bar window)
    swing_highs = []
    swing_lows = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append(lows[i])

    # Need at least 2 swing points to determine structure
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "no_structure"

    # Check for higher highs and higher lows (uptrend)
    higher_highs = swing_highs[-1] > swing_highs[-2]
    higher_lows = swing_lows[-1] > swing_lows[-2]

    # Check for lower highs and lower lows (downtrend)
    lower_highs = swing_highs[-1] < swing_highs[-2]
    lower_lows = swing_lows[-1] < swing_lows[-2]

    if higher_highs and higher_lows:
        return "uptrend"
    elif lower_highs and lower_lows:
        return "downtrend"
    else:
        return "no_structure"


# ─────────────────────────────────────────────
#  COST MODEL
# ─────────────────────────────────────────────
class TradingCosts:
    """
    Models real broker trading costs: spread, slippage, and commission.

    Spread can be variable (from broker data) or fixed.
    Slippage is random between 0 and max_slippage.
    Commission is per lot round-trip.
    """

    def __init__(self, fixed_spread: float = 0.0, max_slippage: float = 0.0,
                 commission: float = 0.0, lot_size: float = 0.01):
        self.fixed_spread = fixed_spread      # Fixed spread in dollars (used when no bar spread)
        self.max_slippage = max_slippage      # Max random slippage in dollars
        self.commission = commission           # Commission per lot round-trip in dollars
        self.lot_size = lot_size              # Lot size for commission calc
        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.total_commission_cost = 0.0
        self.trade_count = 0

    def get_entry_cost(self, bar_spread_dollars: Optional[float] = None) -> float:
        """
        Calculate cost at entry: half spread + random slippage.

        Args:
            bar_spread_dollars: Variable spread in dollars from broker data.
                               If None, uses fixed_spread.

        Returns:
            Total entry cost in dollars (always positive, deducted from P/L).
        """
        # Spread cost at entry = half the spread
        spread = bar_spread_dollars if bar_spread_dollars is not None else self.fixed_spread
        spread_cost = spread / 2.0

        # Random slippage
        slippage = random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0

        return spread_cost + slippage

    def get_exit_cost(self, bar_spread_dollars: Optional[float] = None) -> float:
        """
        Calculate cost at exit: half spread + random slippage.

        Args:
            bar_spread_dollars: Variable spread in dollars from broker data.
                               If None, uses fixed_spread.

        Returns:
            Total exit cost in dollars (always positive, deducted from P/L).
        """
        spread = bar_spread_dollars if bar_spread_dollars is not None else self.fixed_spread
        spread_cost = spread / 2.0

        slippage = random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0

        return spread_cost + slippage

    def get_commission_cost(self) -> float:
        """
        Calculate commission cost for one round-trip trade.

        Returns:
            Commission in dollars.
        """
        return self.commission * self.lot_size

    def get_total_trade_cost(self, entry_bar_spread: Optional[float] = None,
                             exit_bar_spread: Optional[float] = None) -> float:
        """
        Calculate total cost for one complete trade (entry + exit + commission).

        Returns:
            Total cost in dollars.
        """
        entry_cost = self.get_entry_cost(entry_bar_spread)
        exit_cost = self.get_exit_cost(exit_bar_spread)
        commission_cost = self.get_commission_cost()

        self.total_spread_cost += (entry_cost + exit_cost - 
                                   (random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0) -
                                   (random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0))
        # Simpler accounting: track each component
        total = entry_cost + exit_cost + commission_cost
        self.trade_count += 1
        return total

    def record_trade_costs(self, entry_bar_spread: Optional[float] = None,
                           exit_bar_spread: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate and record costs for one trade, returning breakdown.

        Returns:
            Dict with spread_cost, slippage_cost, commission_cost, total_cost.
        """
        spread_entry = (entry_bar_spread if entry_bar_spread is not None else self.fixed_spread) / 2.0
        spread_exit = (exit_bar_spread if exit_bar_spread is not None else self.fixed_spread) / 2.0
        spread_cost = spread_entry + spread_exit

        slippage_entry = random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0
        slippage_exit = random.uniform(0, self.max_slippage) if self.max_slippage > 0 else 0.0
        slippage_cost = slippage_entry + slippage_exit

        commission_cost = self.get_commission_cost()

        total_cost = spread_cost + slippage_cost + commission_cost

        self.total_spread_cost += spread_cost
        self.total_slippage_cost += slippage_cost
        self.total_commission_cost += commission_cost
        self.trade_count += 1

        return {
            "spread_cost": round(spread_cost, 4),
            "slippage_cost": round(slippage_cost, 4),
            "commission_cost": round(commission_cost, 4),
            "total_cost": round(total_cost, 4),
        }

    def get_cost_summary(self) -> Dict[str, float]:
        """Return summary of all accumulated costs."""
        return {
            "total_trades": self.trade_count,
            "total_spread_cost": round(self.total_spread_cost, 2),
            "total_slippage_cost": round(self.total_slippage_cost, 2),
            "total_commission_cost": round(self.total_commission_cost, 2),
            "total_all_costs": round(self.total_spread_cost + self.total_slippage_cost + self.total_commission_cost, 2),
            "avg_cost_per_trade": round(
                (self.total_spread_cost + self.total_slippage_cost + self.total_commission_cost) / self.trade_count, 4
            ) if self.trade_count > 0 else 0.0,
        }


def convert_spread_points_to_dollars(spread_points: float, avg_price: float) -> float:
    """
    Convert spread from points to dollars.

    Auto-detects point value based on average price:
    - If avg price > 1000: point = 0.01 (gold 2-digit, e.g., XAUUSD at 2000.xx)
    - If avg price > 100 and <= 1000: point = 0.001 (3-digit gold pricing)
    - Otherwise: point = 0.00001 (standard forex 5-digit)

    Args:
        spread_points: Spread in points from MT5
        avg_price: Average price of the instrument

    Returns:
        Spread in dollars
    """
    if avg_price > 1000:
        # Gold 2-digit pricing (e.g., 2350.45) - point = 0.01
        point_value = 0.01
    elif avg_price > 100:
        # Gold 3-digit pricing (e.g., 235.045) - point = 0.001
        point_value = 0.001
    else:
        # Standard forex 5-digit (e.g., 1.12345) - point = 0.00001
        point_value = 0.00001

    return spread_points * point_value


def load_broker_data(filepath: str) -> pd.DataFrame:
    """
    Load exported broker data from CSV file.

    Expected CSV format (from Export_Tick_Data.mq5):
    timestamp,open,high,low,close,volume,spread

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with Open, High, Low, Close, Volume, Spread columns
        and DatetimeIndex.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume", "spread"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Parse timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Rename to match expected format (capitalized)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "spread": "Spread",
    }, inplace=True)

    # Convert spread from points to dollars
    avg_price = df["Close"].mean()
    df["Spread_Dollars"] = df["Spread"].apply(
        lambda s: convert_spread_points_to_dollars(s, avg_price)
    )

    print(f"[Backtest] Loaded {len(df)} bars from {filepath}")
    print(f"[Backtest] Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    print(f"[Backtest] Avg spread: {df['Spread'].mean():.1f} points "
          f"(${df['Spread_Dollars'].mean():.4f})")
    print(f"[Backtest] Date range: {df.index[0]} to {df.index[-1]}")

    return df


# ─────────────────────────────────────────────
#  BACKTESTER ENGINE
# ─────────────────────────────────────────────
class Backtester:
    """
    Fast-forward backtester that simulates the trading strategy
    bar-by-bar on historical data.
    """

    def __init__(self, verbose: bool = False, min_bars_between_entries: int = MIN_BARS_BETWEEN_ENTRIES,
                 trading_costs: Optional[TradingCosts] = None):
        self.verbose = verbose
        self.min_bars_between_entries = min_bars_between_entries
        self.trading_costs = trading_costs
        self.open_positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.last_entry_bar: int = -min_bars_between_entries  # Allow entry on first qualifying bar
        self.trades_today: int = 0
        self.current_trade_day: Optional[str] = None
        self.auto_optimizer = AutoOptimizer(
            config=AutoOptimizerConfig(
                optimize_frequency=50,
                min_trades_before_tuning=10,
            ),
            state_dir=os.path.dirname(os.path.abspath(__file__))
        )

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run the backtest on the provided OHLCV DataFrame.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
                and DatetimeIndex (UTC). Optionally includes Spread_Dollars
                for variable spread from broker data.

        Returns:
            Dict with trade_log and summary statistics.
        """
        if df.empty or len(df) < MOMENTUM_LOOKBACK + RSI_PERIOD + 1:
            return {"trade_log": [], "summary": self._empty_summary()}

        # Store reference for cost model lookups
        self._current_df = df
        has_spread_data = "Spread_Dollars" in df.columns

        # Compute RSI for all bars
        rsi_series = compute_rsi(df["Close"])

        # Bar-by-bar simulation
        for i in range(MOMENTUM_LOOKBACK + RSI_PERIOD, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]
            current_price = bar["Close"]

            # --- Trade management for open positions ---
            positions_to_close = []
            for pos in self.open_positions:
                bars_held = i - pos.entry_bar_index

                # Apply progressive trailing stop
                apply_progressive_trailing(pos, current_price)

                # Check SL hit
                sl_hit, exit_price = check_sl_hit(pos, bar["Low"], bar["High"])
                if sl_hit:
                    pnl = pos.unrealized_pnl(exit_price)
                    positions_to_close.append((pos, exit_price, pnl, "sl_hit"))
                    continue

                # Check time exit
                if bars_held >= MAX_HOLD_BARS:
                    pnl = pos.unrealized_pnl(current_price)
                    positions_to_close.append((pos, current_price, pnl, "time_exit"))
                    continue

                # Check momentum exit
                if check_momentum_exit(pos, df["Close"], i):
                    pnl = pos.unrealized_pnl(current_price)
                    positions_to_close.append((pos, current_price, pnl, "momentum_exit"))
                    continue

            # Close positions that triggered exit
            for pos, exit_price, pnl, exit_reason in positions_to_close:
                self._close_position(pos, exit_price, pnl, bar_time, exit_reason, exit_bar_index=i)
                self.open_positions.remove(pos)

            # --- Entry logic ---
            if len(self.open_positions) < MAX_POSITIONS:
                # Check cooldown: min bars since last entry
                bars_since_last_entry = i - self.last_entry_bar
                if bars_since_last_entry >= self.min_bars_between_entries:
                    # Daily trade limit
                    bar_day = str(bar_time.date()) if hasattr(bar_time, 'date') else str(bar_time)[:10]
                    if self.current_trade_day != bar_day:
                        self.current_trade_day = bar_day
                        self.trades_today = 0
                    if self.trades_today >= MAX_TRADES_PER_DAY:
                        continue

                    momentum = compute_momentum_direction(df["Close"], i)

                    # Range trading: only if momentum is FLAT and range > $8
                    if momentum == "FLAT":
                        range_bars = 20
                        if i >= range_bars:
                            range_high = float(df["High"].iloc[i - range_bars:i].max())
                            range_low = float(df["Low"].iloc[i - range_bars:i].min())
                            range_size = range_high - range_low

                            if range_size > 8.0 and current_price > 0:
                                position_in_range = (current_price - range_low) / range_size

                                if position_in_range <= 0.10:
                                    # Near bottom 10% -> BUY
                                    momentum = "BUY"
                                elif position_in_range >= 0.90:
                                    # Near top 10% -> SELL
                                    momentum = "SELL"

                    if momentum != "FLAT":
                        rsi_value = rsi_series.iloc[i]
                        if not np.isnan(rsi_value):
                            # RSI filter: don't buy overbought, don't sell oversold
                            rsi_ok = True
                            if momentum == "BUY" and rsi_value > RSI_OVERBOUGHT:
                                rsi_ok = False
                            elif momentum == "SELL" and rsi_value < RSI_OVERSOLD:
                                rsi_ok = False

                            # RSI zone filter: require RSI in favorable zone
                            # BUY: RSI 35-58 (not overbought, shows upward room)
                            # SELL: RSI 42-65 (not oversold, shows downward room)
                            if rsi_ok:
                                if momentum == "BUY" and not (35 <= rsi_value <= 58):
                                    rsi_ok = False
                                elif momentum == "SELL" and not (42 <= rsi_value <= 65):
                                    rsi_ok = False

                            if rsi_ok:
                                session = detect_session(bar_time)

                                # Session filter: only trade during London or overlap
                                if session not in ("london", "overlap"):
                                    continue

                                # Price structure alignment: REQUIRE structure to confirm
                                structure = detect_price_structure(df, i)
                                structure_ok = False
                                if momentum == "BUY" and structure == "uptrend":
                                    structure_ok = True
                                elif momentum == "SELL" and structure == "downtrend":
                                    structure_ok = True

                                if structure_ok:
                                    # EMA 50 trend confirmation filter
                                    ema_ok = True
                                    if i >= 50:
                                        ema50 = df["Close"].iloc[i - 50:i].mean()
                                        if momentum == "BUY" and current_price < ema50:
                                            ema_ok = False
                                        elif momentum == "SELL" and current_price > ema50:
                                            ema_ok = False

                                    if not ema_ok:
                                        continue

                                    # Pullback filter: require price to be pulling back
                                    # from recent extreme (don't chase peaks/troughs)
                                    pullback_ok = True
                                    if i >= 5:
                                        recent_highs = df["High"].iloc[i - 5:i]
                                        recent_lows = df["Low"].iloc[i - 5:i]
                                        if momentum == "BUY":
                                            # For buying: current price should be below recent high
                                            # (means we're buying on a pullback, not at the peak)
                                            recent_high = float(recent_highs.max())
                                            if current_price >= recent_high:
                                                pullback_ok = False
                                        elif momentum == "SELL":
                                            # For selling: current price should be above recent low
                                            recent_low = float(recent_lows.min())
                                            if current_price <= recent_low:
                                                pullback_ok = False

                                    if not pullback_ok:
                                        continue

                                    # ATR volatility filter: only trade when ATR is 1.0-6.0
                                    atr_ok = True
                                    if i >= 14:
                                        highs_arr = df["High"].iloc[i - 14:i].values
                                        lows_arr = df["Low"].iloc[i - 14:i].values
                                        closes_arr = df["Close"].iloc[i - 14:i].values
                                        tr_values = []
                                        for k in range(1, len(highs_arr)):
                                            tr = max(
                                                highs_arr[k] - lows_arr[k],
                                                abs(highs_arr[k] - closes_arr[k - 1]),
                                                abs(lows_arr[k] - closes_arr[k - 1])
                                            )
                                            tr_values.append(tr)
                                        atr_val = np.mean(tr_values) if tr_values else 0.0
                                        if atr_val < 1.0 or atr_val > 6.0:
                                            atr_ok = False

                                    if not atr_ok:
                                        continue

                                    entry_price = current_price

                                    if momentum == "BUY":
                                        sl_price = entry_price - DEFAULT_SL_DISTANCE
                                    else:
                                        sl_price = entry_price + DEFAULT_SL_DISTANCE

                                    # Compute synthetic confidence from momentum magnitude
                                    confidence = compute_momentum_magnitude(df["Close"], i)

                                    # Minimum confidence filter: reject weak momentum
                                    if confidence < 0.70:
                                        continue

                                    entry_time_str = str(bar_time)
                                    pos = Position(
                                        direction=momentum,
                                        entry_price=entry_price,
                                        sl_price=sl_price,
                                        entry_bar_index=i,
                                        entry_time=entry_time_str,
                                        rsi_at_entry=rsi_value,
                                        session=session,
                                        confidence=confidence,
                                    )
                                    # Store bar spread for cost model
                                    if has_spread_data:
                                        pos.entry_bar_spread = bar["Spread_Dollars"]
                                    else:
                                        pos.entry_bar_spread = None
                                    self.open_positions.append(pos)
                                    self.last_entry_bar = i
                                    self.trades_today += 1

                                    if self.verbose:
                                        print(f"  [ENTRY] {momentum} at {entry_price:.2f} "
                                              f"SL={sl_price:.2f} RSI={rsi_value:.1f} "
                                              f"conf={confidence:.3f} session={session} "
                                              f"structure={structure} "
                                              f"time={entry_time_str}")

        # Close any remaining open positions at last bar's close
        if self.open_positions:
            last_bar = df.iloc[-1]
            last_time = df.index[-1]
            last_close = last_bar["Close"]
            last_idx = len(df) - 1
            for pos in list(self.open_positions):
                pnl = pos.unrealized_pnl(last_close)
                self._close_position(pos, last_close, pnl, last_time, "end_of_data", exit_bar_index=last_idx)
            self.open_positions.clear()

        # Save auto_optimizer state
        self.auto_optimizer.save_state()

        # Build results
        summary = self._compute_summary()
        results = {
            "trade_log": self.closed_trades,
            "summary": summary,
        }

        # Add cost summary if trading costs are enabled
        if self.trading_costs and self.trading_costs.trade_count > 0:
            results["cost_summary"] = self.trading_costs.get_cost_summary()

        return results

    def _close_position(self, pos: Position, exit_price: float, pnl: float,
                        exit_time, exit_reason: str, exit_bar_index: int = 0) -> None:
        """Record a closed trade and feed to AutoOptimizer."""
        exit_time_str = str(exit_time)

        # Determine trail tier for optimizer
        trail_tier = pos.trail_tier if pos.trail_tier != "none" else "wide"

        # Apply trading costs if configured
        cost_breakdown = None
        if self.trading_costs:
            # Get spread from entry/exit bars if available (broker data)
            entry_spread = getattr(pos, 'entry_bar_spread', None)
            exit_spread = getattr(pos, 'exit_bar_spread', None)
            # Use exit bar index to look up spread if available
            if hasattr(self, '_current_df') and 'Spread_Dollars' in self._current_df.columns:
                if exit_bar_index > 0 and exit_bar_index < len(self._current_df):
                    exit_spread = self._current_df.iloc[exit_bar_index]['Spread_Dollars']

            cost_breakdown = self.trading_costs.record_trade_costs(
                entry_bar_spread=entry_spread,
                exit_bar_spread=exit_spread,
            )
            pnl -= cost_breakdown["total_cost"]

        trade_record = {
            "direction": pos.direction,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "entry_time": pos.entry_time,
            "exit_time": exit_time_str,
            "session": pos.session,
            "rsi_at_entry": round(pos.rsi_at_entry, 2),
            "exit_reason": exit_reason,
            "trail_tier": trail_tier,
            "max_profit": round(pos.max_profit, 2),
            "confidence": pos.confidence,
        }
        if cost_breakdown:
            trade_record["costs"] = cost_breakdown
        self.closed_trades.append(trade_record)

        # Record to AutoOptimizer
        trade_context = {
            "session": pos.session,
            "confidence": pos.confidence,
            "momentum_lookback": MOMENTUM_LOOKBACK,
            "sl_distance": DEFAULT_SL_DISTANCE,
            "result_pnl": round(pnl, 2),
            "direction": pos.direction,
            "rsi_at_entry": round(pos.rsi_at_entry, 2),
            "trail_tier": trail_tier,
            "cooldown_used": self.min_bars_between_entries,
            "max_positions_at_entry": MAX_POSITIONS,
            "entry_time": pos.entry_time,
            "exit_time": exit_time_str,
        }
        self.auto_optimizer.record_trade(trade_context)

        if self.verbose:
            print(f"  [EXIT] {pos.direction} PnL={pnl:+.2f} reason={exit_reason} "
                  f"exit_price={exit_price:.2f} time={exit_time_str}")

    def _compute_summary(self) -> Dict:
        """Compute backtest summary statistics."""
        if not self.closed_trades:
            return self._empty_summary()

        pnls = [t["pnl"] for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_count = len(wins)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(pnls)

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_note": "closed-trade drawdown (not mark-to-market equity drawdown)",
            "wins": win_count,
            "losses": len(losses),
        }

    def _compute_max_drawdown(self, pnls: List[float]) -> float:
        """Compute maximum drawdown from a series of P/L values."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return max_dd

    def _empty_summary(self) -> Dict:
        """Return an empty summary when no trades are generated."""
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "wins": 0,
            "losses": 0,
        }


# ─────────────────────────────────────────────
#  CLI AND MAIN
# ─────────────────────────────────────────────
def download_data(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """
    Download historical data from yfinance.

    Args:
        symbol: Trading symbol (XAUUSD maps to GC=F)
        days: Number of days of data
        interval: Bar interval (1m, 5m, 15m)

    Returns:
        DataFrame with OHLCV data
    """
    # Map symbol to yfinance ticker
    ticker_map = {
        "XAUUSD": "GC=F",
        "GC=F": "GC=F",
    }
    ticker = ticker_map.get(symbol.upper(), symbol)

    # Determine period based on interval limits
    if interval == "1m":
        max_days = 7
        actual_days = min(days, max_days)
    else:
        max_days = 60
        actual_days = min(days, max_days)

    period = f"{actual_days}d"

    print(f"[Backtest] Downloading {ticker} data: period={period}, interval={interval}")

    fetcher = MarketDataFetcher()
    df = fetcher.fetch_ohlcv(ticker=ticker, period=period, interval=interval)

    if df.empty:
        print("[Backtest] WARNING: No data received from yfinance")
    else:
        print(f"[Backtest] Downloaded {len(df)} bars")

    return df


def print_summary(summary: Dict, cost_summary: Optional[Dict] = None) -> None:
    """Print backtest summary to console."""
    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:    {summary['total_trades']}")
    print(f"  Win Rate:        {summary['win_rate'] * 100:.1f}%")
    print(f"  Total P/L:       ${summary['total_pnl']:.2f}")
    print(f"  Avg Win:         ${summary['avg_win']:.2f}")
    print(f"  Avg Loss:        ${summary['avg_loss']:.2f}")
    print(f"  Profit Factor:   {summary['profit_factor']}")
    print(f"  Max Drawdown:    ${summary['max_drawdown']:.2f} (closed-trade)")
    print(f"  Wins/Losses:     {summary['wins']}/{summary['losses']}")

    if cost_summary:
        print("-" * 60)
        print("  COST BREAKDOWN")
        print("-" * 60)
        print(f"  Total Spread Cost:     ${cost_summary['total_spread_cost']:.2f}")
        print(f"  Total Slippage Cost:   ${cost_summary['total_slippage_cost']:.2f}")
        print(f"  Total Commission Cost: ${cost_summary['total_commission_cost']:.2f}")
        print(f"  Total All Costs:       ${cost_summary['total_all_costs']:.2f}")
        print(f"  Avg Cost Per Trade:    ${cost_summary['avg_cost_per_trade']:.4f}")

    print("=" * 60)


def save_results(results: Dict, output_dir: str) -> None:
    """Save backtest results to JSON file."""
    output_path = os.path.join(output_dir, "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Backtest] Results saved to {output_path}")


def main():
    """Main entry point for the backtester CLI."""
    parser = argparse.ArgumentParser(
        description="Fast-Forward Backtester - Simulates trading strategy on historical data"
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of days of historical data (default: 5)"
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD",
        help="Trading symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--interval", type=str, default="1m",
        choices=["1m", "5m", "15m"],
        help="Bar interval (default: 1m)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each trade entry/exit"
    )
    # Real broker data options
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Path to exported CSV from MT5 (uses real broker data instead of yfinance)"
    )
    parser.add_argument(
        "--spread", type=float, default=0.30,
        help="Fixed spread in dollars (default: 0.30, used when --data-file not provided)"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.10,
        help="Max random slippage in dollars per entry/exit (default: 0.10)"
    )
    parser.add_argument(
        "--commission", type=float, default=7.0,
        help="Commission per lot round-trip in dollars (default: 7.0)"
    )
    parser.add_argument(
        "--lot-size", type=float, default=0.01,
        help="Lot size for commission calculation (default: 0.01)"
    )

    args = parser.parse_args()

    # Determine data source and cost model
    trading_costs = None
    use_costs = False

    if args.data_file:
        # Real broker data mode: variable spreads from CSV
        print(f"[Backtest] MODE: Real broker data from {args.data_file}")
        print(f"[Backtest] Costs: variable spread + slippage={args.slippage:.2f} + "
              f"commission={args.commission:.2f}/lot * {args.lot_size} lots")

        df = load_broker_data(args.data_file)
        if df.empty:
            print("[Backtest] ERROR: No data in file. Exiting.")
            sys.exit(1)

        # Use variable spread from data (spread_dollars per bar), plus slippage and commission
        trading_costs = TradingCosts(
            fixed_spread=0.0,  # Variable spread comes from bar data
            max_slippage=args.slippage,
            commission=args.commission,
            lot_size=args.lot_size,
        )
        use_costs = True

    else:
        # yfinance mode
        print(f"[Backtest] MODE: yfinance data (symbol={args.symbol}, "
              f"days={args.days}, interval={args.interval})")

        # Apply fixed spread/slippage/commission if --spread is explicitly given or defaults
        if args.spread > 0 or args.slippage > 0 or args.commission > 0:
            print(f"[Backtest] Costs: spread={args.spread:.2f} + slippage={args.slippage:.2f} + "
                  f"commission={args.commission:.2f}/lot * {args.lot_size} lots")
            trading_costs = TradingCosts(
                fixed_spread=args.spread,
                max_slippage=args.slippage,
                commission=args.commission,
                lot_size=args.lot_size,
            )
            use_costs = True

        df = download_data(args.symbol, args.days, args.interval)
        if df.empty:
            print("[Backtest] ERROR: No data available. Exiting.")
            sys.exit(1)

    # Run backtest
    backtester = Backtester(verbose=args.verbose, trading_costs=trading_costs)
    results = backtester.run(df)

    # Print summary
    cost_summary = results.get("cost_summary") if use_costs else None
    print_summary(results["summary"], cost_summary)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_results(results, output_dir)

    # Save auto_optimizer state (already saved in run(), but ensure it's there)
    print(f"[Backtest] Auto-optimizer state saved "
          f"({backtester.auto_optimizer.trade_count} trades recorded)")

    return results


if __name__ == "__main__":
    main()
