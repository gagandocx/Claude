"""
Data Loader for CCT Rectangle Bot.
Fetches multi-timeframe OHLC data via yfinance.
Handles yfinance limitations:
  - 1M data: last 7 days only
  - 15M data: last 60 days
  - 4H data: resampled from 1H or fetched directly for longer periods
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

import config


def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLC data from yfinance.
    
    Args:
        symbol: Ticker symbol (e.g., 'EURUSD=X', 'GC=F')
        interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
        period: Data period ('7d', '60d', '1y', etc.)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    ticker = yf.Ticker(symbol)
    
    # yfinance uses specific interval strings
    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h",  # Fetch 1h and resample to 4h
        "1d": "1d",
    }
    
    fetch_interval = interval_map.get(interval, interval)
    
    # Adjust period for resampling needs
    fetch_period = period
    if interval == "4h":
        # Need more 1h data to produce enough 4h candles
        fetch_period = period
    
    try:
        df = ticker.history(period=fetch_period, interval=fetch_interval)
    except Exception as e:
        print(f"Error fetching {symbol} at {interval}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print(f"No data returned for {symbol} at {fetch_interval} over {fetch_period}")
        return pd.DataFrame()
    
    # Remove timezone info for easier handling
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Resample to 4H if needed
    if interval == "4h":
        df = resample_to_4h(df)
    
    # Keep only OHLCV columns
    columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Drop rows with NaN
    df.dropna(inplace=True)
    
    return df


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1H data to 4H candles.
    
    Args:
        df: DataFrame with 1H OHLCV data
    
    Returns:
        DataFrame with 4H OHLCV data
    """
    resampled = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    resampled.dropna(inplace=True)
    return resampled


def load_multi_timeframe_data(symbol: str = None) -> dict:
    """
    Load data for all three timeframes needed by the CCT Rectangle strategy.
    
    Returns:
        Dictionary with keys '4h', '15m', '1m' containing DataFrames
    """
    if symbol is None:
        symbol = config.SYMBOL
    
    print(f"Loading data for {symbol}...")
    
    # Fetch 15M data (up to 60 days) - this is our primary timeframe
    print(f"  Fetching 15M data ({config.DATA_PERIOD_15M})...")
    df_15m = fetch_data(symbol, "15m", config.DATA_PERIOD_15M)
    if df_15m.empty:
        print("  WARNING: No 15M data available. Trying alternative periods...")
        for alt_period in ["30d", "14d", "7d"]:
            df_15m = fetch_data(symbol, "15m", alt_period)
            if not df_15m.empty:
                print(f"  Got 15M data with period={alt_period}")
                break
    
    # Fetch 4H data - resample from 1H over the same period as 15M
    print(f"  Fetching 4H data (resampled from 1H, {config.DATA_PERIOD_15M})...")
    df_4h = fetch_data(symbol, "4h", config.DATA_PERIOD_15M)
    if df_4h.empty:
        print("  WARNING: No 4H data. Resampling from 15M...")
        if not df_15m.empty:
            df_4h = df_15m.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()
    
    # Fetch 1M data (last 7 days only)
    print(f"  Fetching 1M data ({config.DATA_PERIOD_1M})...")
    df_1m = fetch_data(symbol, "1m", config.DATA_PERIOD_1M)
    if df_1m.empty:
        print("  WARNING: No 1M data available. Trying alternative periods...")
        for alt_period in ["5d", "3d", "2d"]:
            df_1m = fetch_data(symbol, "1m", alt_period)
            if not df_1m.empty:
                print(f"  Got 1M data with period={alt_period}")
                break
    
    # If 1M data is unavailable, we can simulate from 15M for backtesting
    if df_1m.empty and not df_15m.empty:
        print("  WARNING: Using 5M as entry timeframe (1M unavailable)")
        df_1m = fetch_data(symbol, "5m", config.DATA_PERIOD_15M)
    
    data = {
        "4h": df_4h,
        "15m": df_15m,
        "1m": df_1m,
    }
    
    # Print data summary
    for tf, df in data.items():
        if not df.empty:
            print(f"  {tf}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  {tf}: NO DATA")
    
    return data


def get_overlapping_period(data: dict) -> tuple:
    """
    Find the overlapping time period across all timeframes.
    
    Args:
        data: Dictionary of DataFrames keyed by timeframe
    
    Returns:
        Tuple of (start_time, end_time) representing the overlap
    """
    starts = []
    ends = []
    
    for tf, df in data.items():
        if not df.empty:
            starts.append(df.index[0])
            ends.append(df.index[-1])
    
    if not starts or not ends:
        return None, None
    
    overlap_start = max(starts)
    overlap_end = min(ends)
    
    if overlap_start >= overlap_end:
        return None, None
    
    return overlap_start, overlap_end
