"""
MT5 Data Feed Module.

Fetches OHLCV candle data from MetaTrader 5 terminal and converts
to pandas DataFrames compatible with the CCT Rectangle Strategy.

Required DataFrame format:
- Columns: Open, High, Low, Close, Volume
- Index: DatetimeIndex in UTC
"""

import logging
from typing import Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import mt5_config


# MT5 timeframe mapping
MT5_TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1 if mt5 else 1,
    "5m": mt5.TIMEFRAME_M5 if mt5 else 5,
    "15m": mt5.TIMEFRAME_M15 if mt5 else 15,
    "30m": mt5.TIMEFRAME_M30 if mt5 else 30,
    "1h": mt5.TIMEFRAME_H1 if mt5 else 16385,
    "4h": mt5.TIMEFRAME_H4 if mt5 else 16388,
    "1d": mt5.TIMEFRAME_D1 if mt5 else 16408,
}


class MT5DataFeed:
    """
    Fetches and prepares OHLCV data from MT5 terminal.

    Converts MT5 rate arrays to pandas DataFrames with the format
    expected by the CCT Rectangle Strategy classes:
    - Columns: Open, High, Low, Close, Volume
    - DatetimeIndex in UTC timezone
    """

    def __init__(self, symbol: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize MT5DataFeed.

        Args:
            symbol: Trading symbol. Defaults to config SYMBOL.
            logger: Optional logger instance.
        """
        self.symbol = symbol or mt5_config.SYMBOL
        self.logger = logger or logging.getLogger("mt5_data_feed")

    def get_4h_data(self, bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch 4-hour candle data.

        Args:
            bars: Number of bars to fetch. Defaults to config BARS_4H.

        Returns:
            DataFrame with OHLCV data or None on failure.
        """
        bars = bars or mt5_config.BARS_4H
        return self._fetch_data("4h", bars)

    def get_15m_data(self, bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch 15-minute candle data.

        Args:
            bars: Number of bars to fetch. Defaults to config BARS_15M.

        Returns:
            DataFrame with OHLCV data or None on failure.
        """
        bars = bars or mt5_config.BARS_15M
        return self._fetch_data("15m", bars)

    def get_1m_data(self, bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute candle data.

        Args:
            bars: Number of bars to fetch. Defaults to config BARS_1M.

        Returns:
            DataFrame with OHLCV data or None on failure.
        """
        bars = bars or mt5_config.BARS_1M
        return self._fetch_data("1m", bars)

    def _fetch_data(self, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from MT5 using copy_rates_from_pos.

        Args:
            timeframe: Timeframe string ('1m', '15m', '4h', etc.)
            bars: Number of bars to fetch from current position.

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume]
            and a UTC DatetimeIndex, or None on failure.
        """
        if mt5 is None:
            self.logger.error("MetaTrader5 package not available")
            return None

        mt5_tf = MT5_TIMEFRAMES.get(timeframe)
        if mt5_tf is None:
            self.logger.error(f"Unknown timeframe: {timeframe}")
            return None

        # Fetch rates from current position (0 = most recent)
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            self.logger.error(
                f"Failed to fetch {timeframe} data for {self.symbol}: {error}"
            )
            return None

        # Convert numpy structured array to DataFrame
        df = pd.DataFrame(rates)

        # Convert time column from Unix timestamp to UTC datetime
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        # Rename columns to match strategy expectations
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            },
            inplace=True,
        )

        # Keep only required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in required_cols if c in df.columns]

        # If 'real_volume' exists but 'Volume' does not, use real_volume
        if "Volume" not in df.columns and "real_volume" in df.columns:
            df["Volume"] = df["real_volume"]
            available_cols = required_cols

        df = df[available_cols]

        self.logger.debug(
            f"Fetched {len(df)} bars of {timeframe} data for {self.symbol} "
            f"({df.index[0]} to {df.index[-1]})"
        )

        return df

    def get_all_timeframes(self) -> Optional[dict]:
        """
        Fetch data for all three required timeframes.

        Returns:
            Dictionary with keys 'df_4h', 'df_15m', 'df_1m' or None if any fails.
        """
        df_4h = self.get_4h_data()
        if df_4h is None:
            self.logger.error("Failed to fetch 4H data")
            return None

        df_15m = self.get_15m_data()
        if df_15m is None:
            self.logger.error("Failed to fetch 15M data")
            return None

        df_1m = self.get_1m_data()
        if df_1m is None:
            self.logger.error("Failed to fetch 1M data")
            return None

        self.logger.info(
            f"Data fetched: 4H={len(df_4h)} bars, "
            f"15M={len(df_15m)} bars, 1M={len(df_1m)} bars"
        )

        return {
            "df_4h": df_4h,
            "df_15m": df_15m,
            "df_1m": df_1m,
        }
