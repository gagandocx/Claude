"""
Universal Multi-Symbol Data Loader
====================================
Supports loading tick and bar data from MT5 CSV exports for any symbol,
plus synthetic data generation for testing without real data.

Supported Formats:
    - MT5 tick exports (time_msc, bid, ask, volume_real)
    - MT5 OHLC bar exports (time, open, high, low, close, tick_volume, spread, real_volume)
    - Synthetic data with embedded regime shifts (for backtesting without live data)

Symbol Metadata:
    Contains pip sizes, contract sizes, typical spreads for common instruments.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List

# ============================================================================
# SYMBOL METADATA
# ============================================================================

SYMBOL_METADATA: Dict[str, Dict] = {
    "XAUUSD": {
        "pip_size": 0.01,
        "contract_size": 100,
        "typical_spread": 0.30,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": "Gold vs US Dollar",
        "base_price": 2000.0,
        "volatility_pips": 150,  # typical daily range in pips
    },
    "EURUSD": {
        "pip_size": 0.0001,
        "contract_size": 100000,
        "typical_spread": 0.00012,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": "Euro vs US Dollar",
        "base_price": 1.0800,
        "volatility_pips": 80,
    },
    "GBPJPY": {
        "pip_size": 0.01,
        "contract_size": 100000,
        "typical_spread": 0.03,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": "British Pound vs Japanese Yen",
        "base_price": 190.00,
        "volatility_pips": 120,
    },
    "USDJPY": {
        "pip_size": 0.01,
        "contract_size": 100000,
        "typical_spread": 0.015,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": "US Dollar vs Japanese Yen",
        "base_price": 150.00,
        "volatility_pips": 80,
    },
    "NAS100": {
        "pip_size": 0.01,
        "contract_size": 1,
        "typical_spread": 1.5,
        "min_lot": 0.1,
        "lot_step": 0.1,
        "description": "Nasdaq 100 Index",
        "base_price": 18000.0,
        "volatility_pips": 20000,
    },
    "BTCUSD": {
        "pip_size": 0.01,
        "contract_size": 1,
        "typical_spread": 30.0,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": "Bitcoin vs US Dollar",
        "base_price": 65000.0,
        "volatility_pips": 200000,
    },
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ticks(path: Union[str, Path], symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load MT5 tick CSV exports.

    Expected columns: time_msc, bid, ask, volume_real
    Additional columns are ignored.

    Parameters
    ----------
    path : str or Path
        Path to the tick CSV file.
    symbol : str, optional
        Symbol name for metadata enrichment.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, bid, ask, mid, spread, volume_real
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tick data not found at {path}")

    # Detect file format by reading header
    df = pd.read_csv(path, nrows=0)
    columns = [c.lower().strip() for c in df.columns]

    # Read with appropriate columns
    usecols = []
    for col in ["time_msc", "bid", "ask", "volume_real"]:
        if col in columns:
            usecols.append(col)

    if "time_msc" not in columns:
        # Try alternative column names
        if "time" in columns:
            df = pd.read_csv(path)
            if "time" in df.columns:
                df["time_msc"] = pd.to_datetime(df["time"]).astype(np.int64) // 10**6
        else:
            raise ValueError(f"Cannot find time column in {path}. Columns: {list(df.columns)}")
    else:
        df = pd.read_csv(path)

    # Process timestamps
    if "time_msc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    elif "timestamp" not in df.columns:
        raise ValueError("Cannot parse timestamps from tick data")

    # Compute mid and spread
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["spread"] = df["ask"] - df["bid"]
    elif "mid" not in df.columns:
        raise ValueError("Need bid/ask or mid column in tick data")

    if "volume_real" not in df.columns:
        df["volume_real"] = 1.0

    # Filter invalid ticks
    if "bid" in df.columns:
        mask = (df["bid"] > 0) & (df["ask"] > 0)
        df = df[mask].reset_index(drop=True)

    # Sort chronologically
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep essential columns
    keep_cols = ["timestamp", "bid", "ask", "mid", "spread", "volume_real"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def load_bars(path: Union[str, Path], symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load MT5 OHLC bar CSV exports.

    Expected columns: time, open, high, low, close, tick_volume (or volume)
    Handles multiple common MT5 export formats.

    Parameters
    ----------
    path : str or Path
        Path to the bar CSV file.
    symbol : str, optional
        Symbol name for metadata enrichment.

    Returns
    -------
    pd.DataFrame
        OHLC bars with columns: open, high, low, close, volume, tick_count
        Index is DatetimeIndex.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bar data not found at {path}")

    df = pd.read_csv(path)
    columns = [c.lower().strip() for c in df.columns]
    df.columns = columns

    # Parse timestamp
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    elif "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    else:
        # Try first column as timestamp
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0])

    df.set_index("timestamp", inplace=True)

    # Standardize column names
    rename_map = {}
    if "tick_volume" in df.columns:
        rename_map["tick_volume"] = "tick_count"
    if "real_volume" in df.columns:
        rename_map["real_volume"] = "volume"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "volume" not in df.columns:
        if "tick_count" in df.columns:
            df["volume"] = df["tick_count"]
        else:
            df["volume"] = 100.0

    if "tick_count" not in df.columns:
        df["tick_count"] = df["volume"]

    # Keep essential columns
    keep_cols = ["open", "high", "low", "close", "volume", "tick_count"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Drop NaN rows
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    return df


def build_ohlc_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Build OHLC bars from tick data at specified frequency.

    Parameters
    ----------
    ticks : pd.DataFrame
        Tick data with 'timestamp', 'mid' (or 'bid'/'ask'), 'volume_real'.
    freq : str
        Resampling frequency (e.g., '1min', '5min', '15min', '1h').

    Returns
    -------
    pd.DataFrame
        OHLC bars with columns: open, high, low, close, volume, tick_count.
    """
    df = ticks.copy()

    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
        else:
            raise ValueError("Need 'mid' or 'bid'/'ask' columns")

    if "volume_real" not in df.columns:
        df["volume_real"] = 1.0

    df = df.set_index("timestamp")

    bars = df["mid"].resample(freq).ohlc()
    bars.columns = ["open", "high", "low", "close"]
    bars["tick_count"] = df["mid"].resample(freq).count()
    bars["volume"] = df["volume_real"].resample(freq).sum()

    # Drop bars with no ticks
    bars = bars.dropna(subset=["open"])
    bars = bars[bars["tick_count"] > 0]

    return bars


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data(
    n_bars: int = 2000,
    symbols: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generate realistic synthetic multi-symbol bar data with embedded regime shifts.

    Creates data that has clear trending, ranging, and volatile breakout periods
    so the adaptive system can demonstrate its regime detection and strategy switching.

    Uses ADDITIVE price changes (not multiplicative returns) to keep prices
    in realistic ranges throughout the simulation.

    Parameters
    ----------
    n_bars : int
        Number of bars to generate per symbol.
    symbols : list of str, optional
        Symbols to generate. Defaults to ["XAUUSD", "EURUSD", "GBPJPY"].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping symbol names to DataFrames with OHLCV columns.
    """
    if symbols is None:
        symbols = ["XAUUSD", "EURUSD", "GBPJPY"]

    rng = np.random.default_rng(seed)
    result = {}

    for sym_idx, symbol in enumerate(symbols):
        meta = SYMBOL_METADATA.get(symbol, SYMBOL_METADATA["XAUUSD"])
        base_price = meta["base_price"]
        pip_size = meta["pip_size"]

        # Volatility per bar in PRICE UNITS (not percentage returns)
        # Calibrate so that 1.5*ATR stop with min_lot gives ~$3-8 risk per trade
        # This ensures the position sizer can generate reasonable lot sizes
        contract_size = meta.get("contract_size", 100)
        target_trade_risk = 5.0  # $5 risk at minimum lot
        min_lot = meta.get("min_lot", 0.01)
        target_sl = target_trade_risk / (contract_size * min_lot)
        # ATR should be about target_sl / 1.5
        # Bar volatility is lower than ATR (ATR accumulates over multiple bars)
        bar_volatility = target_sl / 4.0
        # Ensure minimum bar_volatility of 2 pips
        bar_volatility = max(bar_volatility, pip_size * 2.0)

        # Generate regime schedule
        regime_schedule = _generate_regime_schedule(n_bars, rng, offset=sym_idx * 50)

        # Generate prices bar by bar using ADDITIVE changes
        close_prices = np.zeros(n_bars, dtype=np.float64)
        high_prices = np.zeros(n_bars, dtype=np.float64)
        low_prices = np.zeros(n_bars, dtype=np.float64)
        open_prices = np.zeros(n_bars, dtype=np.float64)
        volume = np.zeros(n_bars, dtype=np.float64)

        close_prices[0] = base_price
        open_prices[0] = base_price
        high_prices[0] = base_price + bar_volatility * 0.5
        low_prices[0] = base_price - bar_volatility * 0.5
        volume[0] = 100 + rng.poisson(50)

        for i in range(1, n_bars):
            regime = regime_schedule[i]
            prev_close = close_prices[i - 1]

            # Generate ADDITIVE price change based on regime
            change, vol_mult, vol_spike = _generate_regime_change(
                regime, bar_volatility, pip_size, rng, i
            )

            # Apply change (additive, not multiplicative)
            new_close = prev_close + change

            # Generate OHLC
            bar_range = abs(change) + bar_volatility * vol_mult
            bar_range = max(bar_range, pip_size * 3)

            open_price = prev_close + rng.normal(0, bar_range * 0.1)

            if change > 0:
                high_price = new_close + rng.uniform(0, bar_range * 0.3)
                low_price = min(open_price, new_close) - rng.uniform(0, bar_range * 0.4)
            else:
                high_price = max(open_price, new_close) + rng.uniform(0, bar_range * 0.4)
                low_price = new_close - rng.uniform(0, bar_range * 0.3)

            # Ensure OHLC consistency
            high_price = max(high_price, open_price, new_close)
            low_price = min(low_price, open_price, new_close)

            close_prices[i] = new_close
            open_prices[i] = open_price
            high_prices[i] = high_price
            low_prices[i] = low_price

            # Volume
            base_vol_amount = 100 + rng.poisson(50)
            if vol_spike:
                volume[i] = base_vol_amount * rng.uniform(2.0, 4.0)
            else:
                volume[i] = base_vol_amount * vol_mult

        # Build DataFrame
        timestamps = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=n_bars,
            freq="1min",
            tz="UTC",
        )

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
            "tick_count": volume,
        }, index=timestamps)

        # Store regime labels for verification
        df["_regime_label"] = regime_schedule

        result[symbol] = df

    return result


def _generate_regime_schedule(n_bars: int, rng: np.random.Generator,
                              offset: int = 0) -> np.ndarray:
    """
    Generate a regime schedule with clear regime transitions.

    Creates blocks of different regimes that the detector should be able to identify.
    Regimes: 0=trending_up, 1=trending_down, 2=ranging_narrow,
             3=ranging_wide, 4=volatile_breakout, 5=mean_reverting
    """
    schedule = np.zeros(n_bars, dtype=np.int32)

    # Define regime blocks with variety
    regime_blocks = [
        (0, 200),   # Trending up (200 bars)
        (2, 150),   # Ranging narrow (150 bars)
        (4, 80),    # Volatile breakout (80 bars)
        (1, 180),   # Trending down (180 bars)
        (5, 120),   # Mean reverting (120 bars)
        (3, 150),   # Ranging wide (150 bars)
        (0, 160),   # Trending up again (160 bars)
        (4, 60),    # Volatile breakout (60 bars)
        (1, 140),   # Trending down (140 bars)
        (5, 100),   # Mean reverting (100 bars)
        (2, 130),   # Ranging narrow (130 bars)
        (0, 180),   # Trending up (180 bars)
        (3, 120),   # Ranging wide (120 bars)
        (4, 70),    # Volatile breakout (70 bars)
        (1, 160),   # Trending down (160 bars)
    ]

    # Apply offset for symbol diversity
    pos = offset % 100

    for regime_id, duration in regime_blocks:
        actual_duration = duration + rng.integers(-20, 20)
        actual_duration = max(30, actual_duration)
        end_pos = min(pos + actual_duration, n_bars)
        schedule[pos:end_pos] = regime_id
        pos = end_pos
        if pos >= n_bars:
            break

    # Fill any remaining bars with trending up
    if pos < n_bars:
        schedule[pos:] = 0

    return schedule


def _generate_regime_change(regime: int, bar_volatility: float, pip_size: float,
                            rng: np.random.Generator, bar_idx: int):
    """
    Generate a single bar ADDITIVE price change based on the current regime.

    Returns (price_change, vol_multiplier, volume_spike_flag).
    All changes are in price units (not percentage returns).
    """
    vol_spike = False

    if regime == 0:  # TRENDING_UP
        # Very strong consistent positive drift with minimal noise
        # Designed so TrendFollower catches 3:1 RR trades
        drift = bar_volatility * 1.2
        noise = rng.normal(0, bar_volatility * 0.2)
        change = drift + noise
        vol_mult = 0.8
    elif regime == 1:  # TRENDING_DOWN
        # Very strong consistent negative drift
        drift = -bar_volatility * 1.2
        noise = rng.normal(0, bar_volatility * 0.2)
        change = drift + noise
        vol_mult = 0.8
    elif regime == 2:  # RANGING_NARROW
        # Small oscillations, very low volatility
        noise = rng.normal(0, bar_volatility * 0.1)
        cycle = np.sin(bar_idx * 0.05) * bar_volatility * 0.3
        change = noise + cycle
        vol_mult = 0.3
    elif regime == 3:  # RANGING_WIDE
        # Larger oscillations centered at zero (mean-reverting)
        noise = rng.normal(0, bar_volatility * 0.4)
        cycle = np.sin(bar_idx * 0.03) * bar_volatility * 0.8
        change = noise + cycle
        vol_mult = 1.0
    elif regime == 4:  # VOLATILE_BREAKOUT
        # High volatility with strong directional bias
        direction = 1.0 if rng.random() > 0.4 else -1.0
        change = direction * bar_volatility * rng.uniform(2.0, 4.0) + rng.normal(0, bar_volatility * 0.8)
        vol_mult = 2.5
        vol_spike = rng.random() > 0.5
    elif regime == 5:  # MEAN_REVERTING
        # Strong mean reversion: oscillate predictably
        noise = rng.normal(0, bar_volatility * 0.15)
        cycle = -np.sin(bar_idx * 0.08) * bar_volatility * 0.6
        change = noise + cycle
        vol_mult = 0.6
    else:
        change = rng.normal(0, bar_volatility * 0.5)
        vol_mult = 1.0

    return change, vol_mult, vol_spike


def get_symbol_metadata(symbol: str) -> Dict:
    """
    Get metadata for a symbol.

    Parameters
    ----------
    symbol : str
        Symbol name (e.g., "XAUUSD").

    Returns
    -------
    Dict
        Symbol metadata including pip_size, contract_size, typical_spread, etc.
    """
    if symbol in SYMBOL_METADATA:
        return SYMBOL_METADATA[symbol].copy()
    # Return default (gold-like) for unknown symbols
    return {
        "pip_size": 0.01,
        "contract_size": 100,
        "typical_spread": 0.30,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "description": f"Unknown symbol: {symbol}",
        "base_price": 1000.0,
        "volatility_pips": 100,
    }
