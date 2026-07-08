"""
Utility functions for CCT Rectangle Bot.
Includes: EMA calculation, FVG detection, session level identification,
swing high/low detection, and other helpers.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

import config


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series (typically Close prices)
        period: EMA period (e.g., 50, 200)
    
    Returns:
        Series with EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVGs/Imbalances) in OHLC data.
    
    A bullish FVG: candle[i-2] high < candle[i] low (gap between)
    A bearish FVG: candle[i-2] low > candle[i] high (gap between)
    
    Args:
        df: DataFrame with Open, High, Low, Close columns
    
    Returns:
        DataFrame with FVG information (type, top, bottom, candle index)
    """
    fvgs = []
    
    if len(df) < 3:
        return pd.DataFrame(columns=["time", "type", "top", "bottom"])
    
    for i in range(2, len(df)):
        candle_prev2 = df.iloc[i - 2]
        candle_curr = df.iloc[i]
        
        # Bullish FVG: gap up (prev-2 high is below current low)
        if candle_prev2["High"] < candle_curr["Low"]:
            gap_size = candle_curr["Low"] - candle_prev2["High"]
            if gap_size >= config.FVG_MIN_SIZE_PIPS:
                fvgs.append({
                    "time": df.index[i],
                    "type": "bullish",
                    "top": candle_curr["Low"],
                    "bottom": candle_prev2["High"],
                    "midpoint": (candle_curr["Low"] + candle_prev2["High"]) / 2,
                })
        
        # Bearish FVG: gap down (prev-2 low is above current high)
        if candle_prev2["Low"] > candle_curr["High"]:
            gap_size = candle_prev2["Low"] - candle_curr["High"]
            if gap_size >= config.FVG_MIN_SIZE_PIPS:
                fvgs.append({
                    "time": df.index[i],
                    "type": "bearish",
                    "top": candle_prev2["Low"],
                    "bottom": candle_curr["High"],
                    "midpoint": (candle_prev2["Low"] + candle_curr["High"]) / 2,
                })
    
    return pd.DataFrame(fvgs) if fvgs else pd.DataFrame(
        columns=["time", "type", "top", "bottom", "midpoint"]
    )


def identify_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify session highs and lows for Asia, London, and New York sessions.
    
    Args:
        df: DataFrame with OHLC data (needs datetime index)
    
    Returns:
        DataFrame with session highs/lows per day
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "session", "high", "low"])
    
    levels = []
    
    # Group by date
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date
    df_copy["hour"] = df_copy.index.hour
    
    for date, day_data in df_copy.groupby("date"):
        for session_name, session_times in config.SESSIONS.items():
            start_hour = session_times["start"]
            end_hour = session_times["end"]
            
            # Handle sessions that might wrap around midnight
            if start_hour < end_hour:
                session_data = day_data[
                    (day_data["hour"] >= start_hour) & (day_data["hour"] < end_hour)
                ]
            else:
                session_data = day_data[
                    (day_data["hour"] >= start_hour) | (day_data["hour"] < end_hour)
                ]
            
            if not session_data.empty:
                levels.append({
                    "date": date,
                    "session": session_name,
                    "high": session_data["High"].max(),
                    "low": session_data["Low"].min(),
                    "high_time": session_data["High"].idxmax(),
                    "low_time": session_data["Low"].idxmin(),
                })
    
    return pd.DataFrame(levels) if levels else pd.DataFrame(
        columns=["date", "session", "high", "low", "high_time", "low_time"]
    )


def calculate_swing_highs(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """
    Detect swing highs in price data.
    A swing high is a candle whose High is higher than the 'lookback' candles
    on either side.
    
    Args:
        df: DataFrame with High column
        lookback: Number of candles to check on each side
    
    Returns:
        Boolean Series marking swing highs
    """
    if lookback is None:
        lookback = config.SWING_LOOKBACK
    
    highs = df["High"].values
    n = len(highs)
    swing_highs = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing = False
                break
        swing_highs[i] = is_swing
    
    return pd.Series(swing_highs, index=df.index)


def calculate_swing_lows(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """
    Detect swing lows in price data.
    A swing low is a candle whose Low is lower than the 'lookback' candles
    on either side.
    
    Args:
        df: DataFrame with Low column
        lookback: Number of candles to check on each side
    
    Returns:
        Boolean Series marking swing lows
    """
    if lookback is None:
        lookback = config.SWING_LOOKBACK
    
    lows = df["Low"].values
    n = len(lows)
    swing_lows = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        swing_lows[i] = is_swing
    
    return pd.Series(swing_lows, index=df.index)


def is_level_in_fvg(level: float, fvgs: pd.DataFrame, direction: str) -> bool:
    """
    Check if a price level sits inside a Fair Value Gap.
    
    Args:
        level: Price level to check
        fvgs: DataFrame of detected FVGs
        direction: 'bullish' or 'bearish' - the trade direction
    
    Returns:
        True if the level is within any relevant FVG
    """
    if fvgs.empty:
        return False
    
    for _, fvg in fvgs.iterrows():
        # For bullish trades, we want bullish FVGs (buying into the gap)
        # For bearish trades, we want bearish FVGs (selling into the gap)
        if fvg["type"] == direction:
            if fvg["bottom"] <= level <= fvg["top"]:
                return True
    
    return False


def is_at_session_extreme(
    level: float, 
    session_levels: pd.DataFrame, 
    timestamp: pd.Timestamp,
    tolerance: float = None
) -> bool:
    """
    Check if a price level is near a session high or low.
    
    Args:
        level: Price level to check
        session_levels: DataFrame of session highs/lows
        timestamp: Current timestamp to determine relevant sessions
        tolerance: How close the level must be to session extreme.
                   Auto-scales based on price magnitude if None.
    
    Returns:
        True if level is near a session high or low
    """
    if session_levels.empty:
        return False
    
    # Auto-scale tolerance based on price level (works for both forex and gold)
    if tolerance is None:
        tolerance = level * 0.001  # 0.1% of price
    
    current_date = timestamp.date()
    # Check today's and yesterday's session levels
    relevant = session_levels[
        (session_levels["date"] >= current_date - pd.Timedelta(days=1)) &
        (session_levels["date"] <= current_date)
    ]
    
    for _, row in relevant.iterrows():
        if abs(level - row["high"]) <= tolerance:
            return True
        if abs(level - row["low"]) <= tolerance:
            return True
    
    return False


def get_next_key_level(
    df_15m: pd.DataFrame, 
    current_idx: int, 
    direction: str,
    lookback: int = 50
) -> Optional[float]:
    """
    Find the next key M15 level for take profit targeting.
    
    Args:
        df_15m: 15M DataFrame
        current_idx: Current position in the DataFrame
        direction: 'bullish' or 'bearish'
        lookback: How far back to look for key levels
    
    Returns:
        Price of next key level, or None if not found
    """
    start_idx = max(0, current_idx - lookback)
    subset = df_15m.iloc[start_idx:current_idx]
    
    if subset.empty:
        return None
    
    current_price = df_15m.iloc[current_idx]["Close"]
    
    swing_highs = calculate_swing_highs(subset, lookback=3)
    swing_lows = calculate_swing_lows(subset, lookback=3)
    
    if direction == "bullish":
        # Look for swing highs above current price as targets
        high_levels = subset.loc[swing_highs, "High"].values
        targets = [h for h in high_levels if h > current_price]
        if targets:
            return min(targets)  # Nearest target above
    else:
        # Look for swing lows below current price as targets
        low_levels = subset.loc[swing_lows, "Low"].values
        targets = [l for l in low_levels if l < current_price]
        if targets:
            return max(targets)  # Nearest target below
    
    return None


def calculate_rr_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """
    Calculate the reward:risk ratio.
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
    
    Returns:
        RR ratio (e.g., 3.0 means 3:1)
    """
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return 0.0
    
    return reward / risk
