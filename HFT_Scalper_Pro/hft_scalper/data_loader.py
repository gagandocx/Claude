"""
Tick data loader for XAUUSD broker ticks.

Efficiently loads 367MB CSV with ~6M ticks using dtype optimization
and chunked processing. Computes mid price, spread, and 1-minute OHLC bars.
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path


# Column dtypes optimized for memory efficiency
TICK_DTYPES = {
    "time_msc": np.int64,
    "bid": np.float64,
    "ask": np.float64,
    "last": np.float64,
    "volume": np.int32,
    "flags": np.int32,
    "flags_str": "category",
    "volume_real": np.float64,
}

DEFAULT_TICK_PATH = Path(__file__).parent.parent / "tick_data" / "XAUUSD_RealTicks.csv"


def load_ticks(path: Optional[Union[str, Path]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw tick data from CSV with memory-efficient dtypes.

    Parameters
    ----------
    path : str or Path, optional
        Path to the tick CSV file. Defaults to tick_data/XAUUSD_RealTicks.csv
    nrows : int, optional
        Number of rows to load (for testing). None loads all.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, bid, ask, mid, spread, volume_real
    """
    if path is None:
        path = DEFAULT_TICK_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tick data not found at {path}")

    print(f"Loading tick data from {path}...")

    df = pd.read_csv(
        path,
        dtype=TICK_DTYPES,
        usecols=["time_msc", "bid", "ask", "volume_real"],
        nrows=nrows,
    )

    # Convert millisecond timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df.drop(columns=["time_msc"], inplace=True)

    # Compute mid price and spread
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]

    # Filter out zero-bid or zero-ask ticks (invalid)
    mask = (df["bid"] > 0) & (df["ask"] > 0)
    df = df[mask].reset_index(drop=True)

    # Sort by timestamp to ensure chronological order
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df):,} ticks spanning {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def build_ohlc_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Build OHLC bars from tick data at specified frequency.

    Uses mid price for OHLC calculation.

    Parameters
    ----------
    ticks : pd.DataFrame
        Tick data with 'timestamp', 'mid', 'bid', 'ask', 'spread', 'volume_real'
    freq : str
        Resampling frequency (default '1min' for 1-minute bars)

    Returns
    -------
    pd.DataFrame
        OHLC bars with columns: open, high, low, close, tick_count, volume,
        avg_spread, max_spread, volatility (high-low range)
    """
    print(f"Building {freq} OHLC bars...")

    df = ticks.set_index("timestamp")

    bars = df["mid"].resample(freq).ohlc()
    bars.columns = ["open", "high", "low", "close"]

    # Aggregate additional metrics per bar
    bars["tick_count"] = df["mid"].resample(freq).count()
    bars["volume"] = df["volume_real"].resample(freq).sum()
    bars["avg_spread"] = df["spread"].resample(freq).mean()
    bars["max_spread"] = df["spread"].resample(freq).max()
    bars["range"] = bars["high"] - bars["low"]

    # Drop bars with no ticks
    bars = bars.dropna(subset=["open"])
    bars = bars[bars["tick_count"] > 0]

    # Compute returns
    bars["returns"] = bars["close"].pct_change()
    bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))

    print(f"Built {len(bars):,} bars from {bars.index[0]} to {bars.index[-1]}")

    return bars


def get_tick_returns(ticks: pd.DataFrame, n: int = 1) -> np.ndarray:
    """
    Compute tick-level returns (mid price changes over n ticks).

    Parameters
    ----------
    ticks : pd.DataFrame
        Tick data with 'mid' column
    n : int
        Number of ticks for return computation

    Returns
    -------
    np.ndarray
        Array of n-tick returns
    """
    mid = ticks["mid"].values
    returns = (mid[n:] - mid[:-n]) / mid[:-n]
    return returns
