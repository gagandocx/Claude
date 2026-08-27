#!/usr/bin/env python3
"""
=============================================================
  NAS100 ML Trainer - IPDA-Based Feature Engineering
  Neural Network Training Pipeline for MetaTrader 5 EA
=============================================================

DESCRIPTION:
    This script processes NAS100 (US Tech 100) M1 OHLCV data,
    engineers features based on ICT/IPDA methodology, trains a
    feedforward neural network using only numpy, and exports
    model weights as CSV files for the MQL5 EA to load.

DATA FORMAT:
    Input CSV must have columns (no header row expected by default):
        datetime, open, high, low, close, volume

    Datetime format: YYYY.MM.DD HH:MM (MetaTrader export format)
    or YYYY-MM-DD HH:MM:SS

    Example row:
        2024.01.02 09:30,17650.25,17655.50,17648.00,17653.75,1250

USAGE:
    1. Export M1 data from MetaTrader 5:
       - Open NAS100 M1 chart
       - Tools -> History Center -> Export
       - Save as CSV

    2. Set CSV_FILE path in configuration below

    3. Run: python3 nas100_ml_trainer.py

    4. Model files are saved to NAS100_ML_Model/ directory

    5. Copy NAS100_ML_Model/ folder to MT5 Files directory:
       MQL5/Files/NAS100_ML_Model/

OUTPUT FILES (in NAS100_ML_Model/ directory):
    - weights_layer1.csv    (input_size x 128 matrix)
    - weights_layer2.csv    (128 x 64 matrix)
    - weights_layer3.csv    (64 x 32 matrix)
    - weights_output.csv    (32 x 3 matrix)
    - biases_layer1.csv     (128 values)
    - biases_layer2.csv     (64 values)
    - biases_layer3.csv     (32 values)
    - biases_output.csv     (3 values)
    - normalization_params.csv  (feature_name,min,max per feature)
    - feature_names.csv     (ordered list of feature names)

FEATURE VECTOR ORDER (must match MQL5 EA computation):
    See feature_names.csv for the exact ordered list.
    The MQL5 EA must compute features in this same order
    and apply the same min-max normalization.

DEPENDENCIES:
    - Python 3.7+
    - numpy
    - pandas

=============================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta


# =============================================================
#  CONFIGURATION
# =============================================================

# --- Data ---
CSV_FILE = "NAS100_M1.csv"          # Path to NAS100 M1 OHLCV CSV
CHUNK_SIZE = 200_000                 # Rows per chunk for reading large files
HAS_HEADER = False                   # Set True if CSV has a header row

# --- Model Hyperparameters ---
LEARNING_RATE = 0.001                # Initial learning rate
EPOCHS = 200                         # Maximum training epochs
BATCH_SIZE = 256                     # Mini-batch size for gradient descent
HIDDEN_SIZES = [128, 64, 32]         # Hidden layer neuron counts
TRAIN_SPLIT = 0.8                    # Train/test split ratio
EARLY_STOP_PATIENCE = 20             # Epochs without improvement before stopping
LR_DECAY = 0.999                     # Learning rate decay per epoch

# --- IPDA Parameters ---
SWING_LOOKBACK = 10                  # Bars to look back for swing highs/lows
OB_DISPLACEMENT_ATR = 1.5           # ATR multiplier for Order Block displacement
FVG_MIN_SIZE_ATR = 0.3              # Minimum FVG size as fraction of ATR
LIQUIDITY_SWEEP_BUFFER = 0.2        # Buffer for liquidity sweep detection (ATR fraction)
BOS_MIN_MOVE_ATR = 1.0              # Minimum move for BOS confirmation

# --- Label Parameters ---
LABEL_LOOKAHEAD_MIN = 5              # Minimum bars to look ahead for labeling
LABEL_LOOKAHEAD_MAX = 15             # Maximum bars to look ahead for labeling
LABEL_ATR_MULTIPLIER = 1.5          # ATR multiplier for BUY/SELL threshold

# --- Session Times (Eastern Time, 24h format) ---
# NAS100 sessions
PREMARKET_START = 8                  # 08:00 ET
PREMARKET_END = 9                    # 09:30 ET (uses half-hour)
REGULAR_START = 9                    # 09:30 ET
REGULAR_END = 16                     # 16:00 ET
POWER_HOUR_START = 15                # 15:00 ET
POWER_HOUR_END = 16                  # 16:00 ET

# --- Output ---
MODEL_DIR = "NAS100_ML_Model"         # Output directory for model files
UTC_OFFSET_ET = -5                   # UTC offset for Eastern Time (adjust for DST)


# =============================================================
#  UTILITY FUNCTIONS
# =============================================================

def print_header(msg):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def print_progress(current, total, prefix=""):
    """Print progress bar."""
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  {prefix} [{bar}] {pct:.1f}%", end="", flush=True)
    if current == total:
        print()


# =============================================================
#  DATA LOADING
# =============================================================

def load_data(csv_file, chunk_size=CHUNK_SIZE, has_header=HAS_HEADER):
    """
    Load NAS100 M1 OHLCV data from CSV file.
    Supports chunked reading for large files.

    Parameters:
        csv_file: Path to CSV file
        chunk_size: Number of rows per chunk
        has_header: Whether CSV has a header row

    Returns:
        pandas DataFrame with columns: datetime, open, high, low, close, volume
    """
    print_header("LOADING DATA")

    if not os.path.exists(csv_file):
        print(f"  ERROR: File '{csv_file}' not found.")
        print(f"  Please set CSV_FILE at the top of this script.")
        sys.exit(1)

    file_size_mb = os.path.getsize(csv_file) / (1024**2)
    print(f"  File: {csv_file}")
    print(f"  Size: {file_size_mb:.1f} MB")

    col_names = ["datetime", "open", "high", "low", "close", "volume"]
    header_opt = 0 if has_header else None

    chunks = []
    chunk_num = 0

    reader = pd.read_csv(
        csv_file,
        header=header_opt,
        names=col_names if not has_header else None,
        chunksize=chunk_size,
        dtype={"open": np.float64, "high": np.float64,
               "low": np.float64, "close": np.float64,
               "volume": np.float64}
    )

    for chunk in reader:
        chunk_num += 1
        print(f"  Reading chunk {chunk_num}...", end="\r")

        if has_header and chunk_num == 1:
            chunk.columns = col_names

        chunk.dropna(inplace=True)
        chunk = chunk[chunk["high"] >= chunk["low"]]
        chunk = chunk[chunk["close"] > 0]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Loaded {len(df):,} bars")
    print(f"  Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    return df


# =============================================================
#  TECHNICAL INDICATORS
# =============================================================

def compute_ema(series, period):
    """Compute Exponential Moving Average."""
    ema = np.zeros(len(series))
    multiplier = 2.0 / (period + 1)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def compute_rsi(close, period=14):
    """Compute Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, 50.0)

    if n < period + 1:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial values
    for i in range(1, period + 1):
        rsi[i] = 50.0

    return rsi


def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))

    # Initial ATR is simple average
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # Fill initial values with expanding average
    for i in range(period - 1):
        atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]

    return atr


def compute_vwap_proxy(high, low, close, volume, window=50):
    """
    Compute a VWAP proxy using typical price * volume over a rolling window.
    Uses a 50-bar rolling window to match the MQL5 EA's real-time computation.
    """
    n = len(close)
    typical_price = (high + low + close) / 3.0
    vwap = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        window_vol = volume[start:i+1]
        window_tp = typical_price[start:i+1]
        cum_vol = np.sum(window_vol)
        cum_tp_vol = np.sum(window_tp * window_vol)

        if cum_vol > 0:
            vwap[i] = cum_tp_vol / cum_vol
        else:
            vwap[i] = typical_price[i]

    return vwap


def compute_volume_profile_ratio(volume, period=20):
    """
    Compute volume relative to its rolling average.
    Values > 1 indicate above-average volume.
    """
    n = len(volume)
    ratio = np.ones(n)

    for i in range(period, n):
        avg_vol = np.mean(volume[i-period:i])
        if avg_vol > 0:
            ratio[i] = volume[i] / avg_vol
        else:
            ratio[i] = 1.0

    return ratio


# =============================================================
#  IPDA FEATURE ENGINEERING
# =============================================================

def detect_swing_points(high, low, lookback=SWING_LOOKBACK):
    """
    Detect swing highs and swing lows.
    A swing high is a bar whose high is the highest in +/- lookback bars.
    A swing low is a bar whose low is the lowest in +/- lookback bars.

    Returns:
        swing_high: array of 1s where swing high detected, 0 otherwise
        swing_low: array of 1s where swing low detected, 0 otherwise
        last_swing_high_price: rolling last swing high price
        last_swing_low_price: rolling last swing low price
    """
    n = len(high)
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    last_sh_price = np.zeros(n)
    last_sl_price = np.zeros(n)

    current_sh = high[0]
    current_sl = low[0]

    for i in range(lookback, n - lookback):
        # Swing high: highest in window
        window_high = high[i - lookback:i + lookback + 1]
        if high[i] == np.max(window_high):
            swing_high[i] = 1.0
            current_sh = high[i]

        # Swing low: lowest in window
        window_low = low[i - lookback:i + lookback + 1]
        if low[i] == np.min(window_low):
            swing_low[i] = 1.0
            current_sl = low[i]

        last_sh_price[i] = current_sh
        last_sl_price[i] = current_sl

    # Forward fill last values
    for i in range(n):
        if last_sh_price[i] == 0 and i > 0:
            last_sh_price[i] = last_sh_price[i-1]
        if last_sl_price[i] == 0 and i > 0:
            last_sl_price[i] = last_sl_price[i-1]

    return swing_high, swing_low, last_sh_price, last_sl_price


def detect_market_structure(close, high, low, swing_high, swing_low,
                           last_sh_price, last_sl_price, atr):
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: Price breaks a swing high (bullish) or swing low (bearish)
         in the direction of the current trend.
    CHoCH: Price breaks structure against the prevailing trend,
           signaling potential reversal.

    Returns:
        bos_bullish: 1 when bullish BOS detected
        bos_bearish: 1 when bearish BOS detected
        choch_bullish: 1 when bullish CHoCH detected
        choch_bearish: 1 when bearish CHoCH detected
        trend: 1 for bullish, -1 for bearish, 0 for neutral
    """
    n = len(close)
    bos_bullish = np.zeros(n)
    bos_bearish = np.zeros(n)
    choch_bullish = np.zeros(n)
    choch_bearish = np.zeros(n)
    trend = np.zeros(n)

    current_trend = 0  # 0=neutral, 1=bullish, -1=bearish
    prev_sh = last_sh_price[0] if last_sh_price[0] > 0 else high[0]
    prev_sl = last_sl_price[0] if last_sl_price[0] > 0 else low[0]

    for i in range(1, n):
        # Update swing references
        if last_sh_price[i] != prev_sh and last_sh_price[i] > 0:
            prev_sh = last_sh_price[i]
        if last_sl_price[i] != prev_sl and last_sl_price[i] > 0:
            prev_sl = last_sl_price[i]

        min_move = atr[i] * BOS_MIN_MOVE_ATR if atr[i] > 0 else 0

        # Bullish break: close above swing high
        if close[i] > prev_sh + min_move * 0.1:
            if current_trend >= 0:
                bos_bullish[i] = 1.0
            else:
                choch_bullish[i] = 1.0
            current_trend = 1

        # Bearish break: close below swing low
        elif close[i] < prev_sl - min_move * 0.1:
            if current_trend <= 0:
                bos_bearish[i] = 1.0
            else:
                choch_bearish[i] = 1.0
            current_trend = -1

        trend[i] = current_trend

    return bos_bullish, bos_bearish, choch_bullish, choch_bearish, trend


def detect_order_blocks(open_p, close, high, low, atr):
    """
    Detect Order Blocks (OB) based on IPDA methodology.

    An Order Block is the last opposing candle before a strong
    displacement move (> OB_DISPLACEMENT_ATR * ATR).

    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move

    Returns:
        ob_bullish_active: 1 if price is near/in a bullish OB zone
        ob_bearish_active: 1 if price is near/in a bearish OB zone
        ob_distance_bull: normalized distance to nearest bullish OB
        ob_distance_bear: normalized distance to nearest bearish OB
    """
    n = len(close)
    ob_bullish_active = np.zeros(n)
    ob_bearish_active = np.zeros(n)
    ob_distance_bull = np.zeros(n)
    ob_distance_bear = np.zeros(n)

    # Store active order blocks as (high, low, index)
    bullish_obs = []  # Bullish OBs (demand zones)
    bearish_obs = []  # Bearish OBs (supply zones)
    max_obs = 10  # Maximum active OBs to track

    for i in range(2, n):
        displacement = atr[i] * OB_DISPLACEMENT_ATR
        if displacement == 0:
            continue

        # Check for bullish displacement (strong up move)
        if close[i] - close[i-2] > displacement:
            # Find last bearish candle before this move
            for j in range(i-1, max(i-5, 0), -1):
                if close[j] < open_p[j]:  # Bearish candle
                    bullish_obs.append((high[j], low[j], j))
                    if len(bullish_obs) > max_obs:
                        bullish_obs.pop(0)
                    break

        # Check for bearish displacement (strong down move)
        if close[i-2] - close[i] > displacement:
            # Find last bullish candle before this move
            for j in range(i-1, max(i-5, 0), -1):
                if close[j] > open_p[j]:  # Bullish candle
                    bearish_obs.append((high[j], low[j], j))
                    if len(bearish_obs) > max_obs:
                        bearish_obs.pop(0)
                    break

        # Check if current price is in any OB zone
        current_price = close[i]

        # Bullish OB (demand) - price returns to OB zone
        min_dist_bull = 999.0
        for ob_high, ob_low, ob_idx in bullish_obs:
            if current_price >= ob_low and current_price <= ob_high:
                ob_bullish_active[i] = 1.0
            dist = abs(current_price - (ob_high + ob_low) / 2.0) / atr[i] if atr[i] > 0 else 10.0
            min_dist_bull = min(min_dist_bull, dist)

        ob_distance_bull[i] = min(min_dist_bull, 10.0) / 10.0  # Normalize to 0-1

        # Bearish OB (supply) - price returns to OB zone
        min_dist_bear = 999.0
        for ob_high, ob_low, ob_idx in bearish_obs:
            if current_price >= ob_low and current_price <= ob_high:
                ob_bearish_active[i] = 1.0
            dist = abs(current_price - (ob_high + ob_low) / 2.0) / atr[i] if atr[i] > 0 else 10.0
            min_dist_bear = min(min_dist_bear, dist)

        ob_distance_bear[i] = min(min_dist_bear, 10.0) / 10.0  # Normalize to 0-1

        # Remove mitigated OBs (price has traded through them)
        bullish_obs = [(h, l, idx) for h, l, idx in bullish_obs if current_price > l - atr[i]]
        bearish_obs = [(h, l, idx) for h, l, idx in bearish_obs if current_price < h + atr[i]]

    return ob_bullish_active, ob_bearish_active, ob_distance_bull, ob_distance_bear


def detect_fair_value_gaps(high, low, close, atr):
    """
    Detect Fair Value Gaps (FVG) - 3-candle imbalance patterns.

    Bullish FVG: Gap between candle 1 high and candle 3 low (price skips up)
    Bearish FVG: Gap between candle 1 low and candle 3 high (price skips down)

    Returns:
        fvg_bullish_active: 1 if price is in/near a bullish FVG
        fvg_bearish_active: 1 if price is in/near a bearish FVG
        fvg_bullish_size: size of nearest bullish FVG (normalized by ATR)
        fvg_bearish_size: size of nearest bearish FVG (normalized by ATR)
    """
    n = len(high)
    fvg_bullish_active = np.zeros(n)
    fvg_bearish_active = np.zeros(n)
    fvg_bullish_size = np.zeros(n)
    fvg_bearish_size = np.zeros(n)

    # Active FVGs: (top, bottom, index)
    bullish_fvgs = []
    bearish_fvgs = []
    max_fvgs = 10

    for i in range(2, n):
        min_gap = atr[i] * FVG_MIN_SIZE_ATR

        # Bullish FVG: candle[i] low > candle[i-2] high (gap up)
        gap_up = low[i] - high[i-2]
        if gap_up > min_gap:
            bullish_fvgs.append((low[i], high[i-2], i))
            if len(bullish_fvgs) > max_fvgs:
                bullish_fvgs.pop(0)

        # Bearish FVG: candle[i-2] low > candle[i] high (gap down)
        gap_down = low[i-2] - high[i]
        if gap_down > min_gap:
            bearish_fvgs.append((low[i-2], high[i], i))
            if len(bearish_fvgs) > max_fvgs:
                bearish_fvgs.pop(0)

        current_price = close[i]

        # Check proximity to bullish FVGs
        for fvg_top, fvg_bot, fvg_idx in bullish_fvgs:
            if fvg_bot <= current_price <= fvg_top:
                fvg_bullish_active[i] = 1.0
                if atr[i] > 0:
                    fvg_bullish_size[i] = max(fvg_bullish_size[i],
                                              (fvg_top - fvg_bot) / atr[i])

        # Check proximity to bearish FVGs
        for fvg_top, fvg_bot, fvg_idx in bearish_fvgs:
            if fvg_bot <= current_price <= fvg_top:
                fvg_bearish_active[i] = 1.0
                if atr[i] > 0:
                    fvg_bearish_size[i] = max(fvg_bearish_size[i],
                                              (fvg_top - fvg_bot) / atr[i])

        # Remove filled FVGs
        bullish_fvgs = [(t, b, idx) for t, b, idx in bullish_fvgs
                        if current_price > b]
        bearish_fvgs = [(t, b, idx) for t, b, idx in bearish_fvgs
                        if current_price < t]

    return fvg_bullish_active, fvg_bearish_active, fvg_bullish_size, fvg_bearish_size


def detect_liquidity_sweeps(high, low, close, last_sh_price, last_sl_price, atr):
    """
    Detect Liquidity Sweeps - price takes out previous highs/lows then reverses.

    Bullish sweep: Price dips below swing low then closes back above (stop hunt)
    Bearish sweep: Price spikes above swing high then closes back below (stop hunt)

    Returns:
        sweep_bullish: 1 when bullish liquidity sweep detected
        sweep_bearish: 1 when bearish liquidity sweep detected
        bars_since_bull_sweep: normalized bars since last bullish sweep
        bars_since_bear_sweep: normalized bars since last bearish sweep
    """
    n = len(high)
    sweep_bullish = np.zeros(n)
    sweep_bearish = np.zeros(n)
    bars_since_bull_sweep = np.ones(n)  # 1.0 = far from sweep
    bars_since_bear_sweep = np.ones(n)

    last_bull_sweep = -100
    last_bear_sweep = -100

    for i in range(1, n):
        buffer = atr[i] * LIQUIDITY_SWEEP_BUFFER if atr[i] > 0 else 0

        # Bullish sweep: low goes below swing low, but close is above
        if (last_sl_price[i] > 0 and
            low[i] < last_sl_price[i] - buffer and
            close[i] > last_sl_price[i]):
            sweep_bullish[i] = 1.0
            last_bull_sweep = i

        # Bearish sweep: high goes above swing high, but close is below
        if (last_sh_price[i] > 0 and
            high[i] > last_sh_price[i] + buffer and
            close[i] < last_sh_price[i]):
            sweep_bearish[i] = 1.0
            last_bear_sweep = i

        # Bars since last sweep (decay feature)
        if last_bull_sweep >= 0:
            bars_since_bull_sweep[i] = min((i - last_bull_sweep) / 50.0, 1.0)
        if last_bear_sweep >= 0:
            bars_since_bear_sweep[i] = min((i - last_bear_sweep) / 50.0, 1.0)

    return sweep_bullish, sweep_bearish, bars_since_bull_sweep, bars_since_bear_sweep


def compute_session_features(datetimes):
    """
    Compute session/killzone features for NAS100.

    Sessions (Eastern Time):
        - Pre-market:   08:00 - 09:30 ET
        - Regular:      09:30 - 16:00 ET
        - Power Hour:   15:00 - 16:00 ET

    Returns:
        is_premarket: 1 during pre-market session
        is_regular: 1 during regular session
        is_power_hour: 1 during power hour
        time_of_day: normalized time (0-1) representing position in trading day
        day_of_week: normalized day (0-1) Mon=0, Fri=0.8
    """
    n = len(datetimes)
    is_premarket = np.zeros(n)
    is_regular = np.zeros(n)
    is_power_hour = np.zeros(n)
    time_of_day = np.zeros(n)
    day_of_week = np.zeros(n)

    for i in range(n):
        dt = datetimes[i]
        # Convert UTC to ET (approximate)
        et_hour = (dt.hour + UTC_OFFSET_ET) % 24
        et_minute = dt.minute
        et_time = et_hour + et_minute / 60.0

        # Pre-market: 08:00 - 09:30 ET
        if PREMARKET_START <= et_time < (PREMARKET_END + 0.5):
            is_premarket[i] = 1.0

        # Regular session: 09:30 - 16:00 ET
        if (REGULAR_START + 0.5) <= et_time < REGULAR_END:
            is_regular[i] = 1.0

        # Power hour: 15:00 - 16:00 ET
        if POWER_HOUR_START <= et_time < POWER_HOUR_END:
            is_power_hour[i] = 1.0

        # Normalized time of day (0-1)
        time_of_day[i] = (dt.hour * 60 + dt.minute) / (24 * 60)

        # Day of week normalized (Mon=0, Fri=4) -> 0 to 1
        day_of_week[i] = dt.weekday() / 4.0  # 0-1 for Mon-Fri

    return is_premarket, is_regular, is_power_hour, time_of_day, day_of_week


def compute_premium_discount(close, last_sh_price, last_sl_price):
    """
    Compute Premium/Discount zone features.

    The dealing range is defined by the last swing high and swing low.
    Premium zone: price is in the upper 50% of the range
    Discount zone: price is in the lower 50% of the range

    Returns:
        pd_position: 0.0 (at swing low) to 1.0 (at swing high)
        is_premium: 1.0 if in premium zone (> 0.5)
        is_discount: 1.0 if in discount zone (< 0.5)
        equilibrium_distance: distance from equilibrium (0.5), normalized
    """
    n = len(close)
    pd_position = np.full(n, 0.5)
    is_premium = np.zeros(n)
    is_discount = np.zeros(n)
    equilibrium_distance = np.zeros(n)

    for i in range(n):
        range_size = last_sh_price[i] - last_sl_price[i]
        if range_size > 0:
            pd_position[i] = (close[i] - last_sl_price[i]) / range_size
            pd_position[i] = np.clip(pd_position[i], 0.0, 1.0)

        if pd_position[i] > 0.5:
            is_premium[i] = 1.0
        elif pd_position[i] < 0.5:
            is_discount[i] = 1.0

        equilibrium_distance[i] = abs(pd_position[i] - 0.5) * 2.0  # 0-1

    return pd_position, is_premium, is_discount, equilibrium_distance


def resample_to_timeframe(df, timeframe_minutes):
    """
    Resample M1 data to higher timeframe for multi-timeframe analysis.

    Parameters:
        df: DataFrame with datetime, open, high, low, close, volume
        timeframe_minutes: target timeframe in minutes (5, 15, 60)

    Returns:
        Resampled DataFrame
    """
    df_copy = df.set_index("datetime")
    rule = f"{timeframe_minutes}min"

    resampled = df_copy.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    resampled.reset_index(inplace=True)
    return resampled


# =============================================================
#  MAIN FEATURE ENGINEERING PIPELINE
# =============================================================

def engineer_features(df):
    """
    Engineer all IPDA-based features from M1 OHLCV data.

    This is the main feature engineering function that computes:
    1. Technical indicators (EMA, RSI, ATR, VWAP, volume profile)
    2. Market structure (swing points, BOS, CHoCH) on M1 and higher TFs
    3. Order Blocks
    4. Fair Value Gaps
    5. Liquidity Sweeps
    6. Session/Killzone features
    7. Premium/Discount zones

    Parameters:
        df: DataFrame with datetime, open, high, low, close, volume

    Returns:
        features: numpy array of shape (n_samples, n_features)
        feature_names: list of feature name strings
    """
    print_header("ENGINEERING FEATURES")

    n = len(df)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    open_p = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    datetimes = df["datetime"].values

    # Convert numpy datetime64 to Python datetime for session computation
    dt_list = pd.to_datetime(datetimes).to_pydatetime().tolist()

    feature_dict = {}

    # --- 1. Technical Indicators ---
    print("  Computing technical indicators...")

    feature_dict["ema_8"] = compute_ema(close, 8)
    feature_dict["ema_21"] = compute_ema(close, 21)
    feature_dict["ema_50"] = compute_ema(close, 50)
    feature_dict["ema_200"] = compute_ema(close, 200)

    # EMA relative positions (normalized by ATR)
    atr = compute_atr(high, low, close, 14)
    feature_dict["atr_14"] = atr

    # Price distance from EMAs (normalized)
    for ema_name, ema_period in [("ema_8", 8), ("ema_21", 21), ("ema_50", 50), ("ema_200", 200)]:
        ema_vals = feature_dict[ema_name]
        dist = np.zeros(n)
        for i in range(n):
            if atr[i] > 0:
                dist[i] = (close[i] - ema_vals[i]) / atr[i]
            else:
                dist[i] = 0.0
        feature_dict[f"dist_from_{ema_name}"] = np.clip(dist, -5.0, 5.0)

    # EMA slopes (normalized)
    for ema_name in ["ema_8", "ema_21", "ema_50", "ema_200"]:
        slope = np.zeros(n)
        ema_vals = feature_dict[ema_name]
        for i in range(5, n):
            if atr[i] > 0:
                slope[i] = (ema_vals[i] - ema_vals[i-5]) / (5 * atr[i])
        feature_dict[f"{ema_name}_slope"] = np.clip(slope, -2.0, 2.0)

    feature_dict["rsi_14"] = compute_rsi(close, 14) / 100.0  # Normalize to 0-1

    vwap = compute_vwap_proxy(high, low, close, volume)
    dist_vwap = np.zeros(n)
    for i in range(n):
        if atr[i] > 0:
            dist_vwap[i] = (close[i] - vwap[i]) / atr[i]
    feature_dict["dist_from_vwap"] = np.clip(dist_vwap, -5.0, 5.0)

    feature_dict["volume_ratio"] = compute_volume_profile_ratio(volume, 20)

    # Candle body/wick ratios
    body = np.abs(close - open_p)
    full_range = high - low
    feature_dict["body_ratio"] = np.where(full_range > 0, body / full_range, 0.5)
    feature_dict["is_bullish_candle"] = (close > open_p).astype(np.float64)

    # ATR normalized
    atr_norm = np.zeros(n)
    for i in range(50, n):
        avg_atr = np.mean(atr[i-50:i])
        if avg_atr > 0:
            atr_norm[i] = atr[i] / avg_atr
    feature_dict["atr_normalized"] = atr_norm

    print("  Computing market structure...")

    # --- 2. Market Structure (M1 timeframe) ---
    swing_high, swing_low, last_sh, last_sl = detect_swing_points(high, low, SWING_LOOKBACK)

    bos_bull, bos_bear, choch_bull, choch_bear, trend = detect_market_structure(
        close, high, low, swing_high, swing_low, last_sh, last_sl, atr
    )

    feature_dict["swing_high"] = swing_high
    feature_dict["swing_low"] = swing_low
    feature_dict["bos_bullish"] = bos_bull
    feature_dict["bos_bearish"] = bos_bear
    feature_dict["choch_bullish"] = choch_bull
    feature_dict["choch_bearish"] = choch_bear
    feature_dict["trend_m1"] = trend

    # --- Higher Timeframe Structure ---
    print("  Computing higher timeframe structure (M5, M15, H1)...")

    for tf_name, tf_min in [("m5", 5), ("m15", 15), ("h1", 60)]:
        df_tf = resample_to_timeframe(df, tf_min)
        if len(df_tf) > SWING_LOOKBACK * 2 + 5:
            tf_high = df_tf["high"].values.astype(np.float64)
            tf_low = df_tf["low"].values.astype(np.float64)
            tf_close = df_tf["close"].values.astype(np.float64)
            tf_atr = compute_atr(tf_high, tf_low, tf_close, 14)

            _, _, tf_sh, tf_sl = detect_swing_points(tf_high, tf_low, SWING_LOOKBACK)
            _, _, _, _, tf_trend = detect_market_structure(
                tf_close, tf_high, tf_low,
                np.zeros(len(tf_close)), np.zeros(len(tf_close)),
                tf_sh, tf_sl, tf_atr
            )

            # Map higher TF trend back to M1 bars
            tf_datetimes = df_tf["datetime"].values
            trend_mapped = np.zeros(n)
            tf_idx = 0
            for i in range(n):
                while tf_idx < len(tf_datetimes) - 1 and datetimes[i] >= tf_datetimes[tf_idx + 1]:
                    tf_idx += 1
                if tf_idx < len(tf_trend):
                    trend_mapped[i] = tf_trend[tf_idx]

            feature_dict[f"trend_{tf_name}"] = trend_mapped
        else:
            feature_dict[f"trend_{tf_name}"] = np.zeros(n)

    print("  Detecting order blocks...")

    # --- 3. Order Blocks ---
    ob_bull, ob_bear, ob_dist_bull, ob_dist_bear = detect_order_blocks(
        open_p, close, high, low, atr
    )
    feature_dict["ob_bullish_active"] = ob_bull
    feature_dict["ob_bearish_active"] = ob_bear
    feature_dict["ob_distance_bull"] = ob_dist_bull
    feature_dict["ob_distance_bear"] = ob_dist_bear

    print("  Detecting fair value gaps...")

    # --- 4. Fair Value Gaps ---
    fvg_bull, fvg_bear, fvg_size_bull, fvg_size_bear = detect_fair_value_gaps(
        high, low, close, atr
    )
    feature_dict["fvg_bullish_active"] = fvg_bull
    feature_dict["fvg_bearish_active"] = fvg_bear
    feature_dict["fvg_bullish_size"] = fvg_size_bull
    feature_dict["fvg_bearish_size"] = fvg_size_bear

    print("  Detecting liquidity sweeps...")

    # --- 5. Liquidity Sweeps ---
    sweep_bull, sweep_bear, bars_bull, bars_bear = detect_liquidity_sweeps(
        high, low, close, last_sh, last_sl, atr
    )
    feature_dict["sweep_bullish"] = sweep_bull
    feature_dict["sweep_bearish"] = sweep_bear
    feature_dict["bars_since_bull_sweep"] = bars_bull
    feature_dict["bars_since_bear_sweep"] = bars_bear

    print("  Computing session features...")

    # --- 6. Session/Killzone Features ---
    is_pre, is_reg, is_power, tod, dow = compute_session_features(dt_list)
    feature_dict["is_premarket"] = is_pre
    feature_dict["is_regular_session"] = is_reg
    feature_dict["is_power_hour"] = is_power
    feature_dict["time_of_day"] = tod
    feature_dict["day_of_week"] = dow

    print("  Computing premium/discount zones...")

    # --- 7. Premium/Discount Zones ---
    pd_pos, is_prem, is_disc, eq_dist = compute_premium_discount(close, last_sh, last_sl)
    feature_dict["pd_position"] = pd_pos
    feature_dict["is_premium"] = is_prem
    feature_dict["is_discount"] = is_disc
    feature_dict["equilibrium_distance"] = eq_dist

    # --- Compile feature matrix ---
    feature_names = list(feature_dict.keys())
    features = np.column_stack([feature_dict[name] for name in feature_names])

    print(f"\n  Total features: {len(feature_names)}")
    print(f"  Total samples: {n:,}")
    print(f"  Feature matrix shape: {features.shape}")

    return features, feature_names


# =============================================================
#  LABEL GENERATION
# =============================================================

def generate_labels(close, atr, lookahead_min=LABEL_LOOKAHEAD_MIN,
                   lookahead_max=LABEL_LOOKAHEAD_MAX,
                   atr_multiplier=LABEL_ATR_MULTIPLIER):
    """
    Generate training labels based on future price movement.

    For each bar, look ahead 5-15 bars and determine:
        BUY  (+1): price rises more than 1.5x ATR within lookahead
        SELL (-1): price falls more than 1.5x ATR within lookahead
        NEUTRAL (0): neither condition met

    The label uses the FIRST direction threshold hit within the window.

    Parameters:
        close: array of close prices
        atr: array of ATR values
        lookahead_min: minimum bars to look ahead
        lookahead_max: maximum bars to look ahead
        atr_multiplier: threshold = atr_multiplier * ATR

    Returns:
        labels: array of -1, 0, +1 values
        label_counts: dict with count of each label
    """
    print_header("GENERATING LABELS")

    n = len(close)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n - lookahead_max):
        threshold = atr[i] * atr_multiplier
        if threshold <= 0:
            continue

        label = 0
        for j in range(i + lookahead_min, min(i + lookahead_max + 1, n)):
            move = close[j] - close[i]
            if move > threshold:
                label = 1  # BUY
                break
            elif move < -threshold:
                label = -1  # SELL
                break

        labels[i] = label

    # Count labels
    buy_count = np.sum(labels == 1)
    sell_count = np.sum(labels == -1)
    neutral_count = np.sum(labels == 0)
    total_valid = n - lookahead_max

    label_counts = {
        "BUY": int(buy_count),
        "SELL": int(sell_count),
        "NEUTRAL": int(neutral_count)
    }

    print(f"  Label distribution:")
    print(f"    BUY  (+1): {buy_count:>8,} ({buy_count/total_valid*100:.1f}%)")
    print(f"    SELL (-1): {sell_count:>8,} ({sell_count/total_valid*100:.1f}%)")
    print(f"    NEUTRAL:   {neutral_count:>8,} ({neutral_count/total_valid*100:.1f}%)")

    return labels, label_counts


# =============================================================
#  NEURAL NETWORK (Pure Numpy Implementation)
# =============================================================

class NeuralNetwork:
    """
    Feedforward Neural Network with ReLU activation.
    Architecture: input -> 128 -> 64 -> 32 -> 3 (softmax output)

    Implements:
        - Forward pass with ReLU activations
        - Backpropagation with cross-entropy loss
        - Mini-batch gradient descent
        - Early stopping
        - Learning rate decay
    """

    def __init__(self, input_size, hidden_sizes=HIDDEN_SIZES, output_size=3,
                 learning_rate=LEARNING_RATE, lr_decay=LR_DECAY):
        """
        Initialize network with He initialization for weights.

        Parameters:
            input_size: number of input features
            hidden_sizes: list of hidden layer sizes [128, 64, 32]
            output_size: number of output classes (3: BUY, NEUTRAL, SELL)
            learning_rate: initial learning rate
            lr_decay: learning rate decay factor per epoch
        """
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        # Build layer dimensions
        layer_dims = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases with He initialization
        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            # He initialization: sqrt(2 / fan_in)
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        self.n_layers = len(self.weights)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(np.float64)

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters:
            X: input array of shape (batch_size, input_size)

        Returns:
            output: softmax probabilities (batch_size, 3)
            cache: list of (input_to_layer, pre_activation, activation) for backprop
        """
        cache = []
        current = X.copy()

        # Hidden layers with ReLU
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            cache.append((current, z, a))
            current = a

        # Output layer with softmax
        z_out = current @ self.weights[-1] + self.biases[-1]
        output = self.softmax(z_out)
        cache.append((current, z_out, output))

        return output, cache

    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss.

        Parameters:
            y_pred: predicted probabilities (batch_size, 3)
            y_true: one-hot encoded labels (batch_size, 3)

        Returns:
            loss: scalar cross-entropy loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def backward(self, X, y_true, cache):
        """
        Backpropagation to compute gradients.

        Parameters:
            X: input data
            y_true: one-hot encoded labels
            cache: cached values from forward pass

        Returns:
            gradients: list of (dW, db) for each layer
        """
        batch_size = X.shape[0]
        gradients = []

        # Output layer gradient (softmax + cross-entropy)
        _, _, y_pred = cache[-1]
        dz = (y_pred - y_true) / batch_size

        # Gradients for output layer
        input_to_output = cache[-1][0]
        dW = input_to_output.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        gradients.append((dW, db))

        # Propagate back through hidden layers
        delta = dz
        for i in range(self.n_layers - 2, -1, -1):
            delta = delta @ self.weights[i + 1].T
            _, z, _ = cache[i]
            delta = delta * self.relu_derivative(z)

            input_to_layer = cache[i][0]
            dW = input_to_layer.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            gradients.append((dW, db))

        gradients.reverse()
        return gradients

    def update_weights(self, gradients):
        """Update weights using gradient descent."""
        for i in range(self.n_layers):
            dW, db = gradients[i]
            # Gradient clipping
            dW = np.clip(dW, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters:
            X: input features (n_samples, n_features)

        Returns:
            predictions: array of class indices (0, 1, 2)
            probabilities: softmax output (n_samples, 3)
        """
        probabilities, _ = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities


# =============================================================
#  TRAINING PIPELINE
# =============================================================

def prepare_data(features, labels, train_split=TRAIN_SPLIT):
    """
    Prepare data for training: normalize features, split train/test.

    Parameters:
        features: numpy array (n_samples, n_features)
        labels: numpy array of labels (-1, 0, +1)
        train_split: fraction for training

    Returns:
        X_train, X_test, y_train, y_test: split and normalized data
        y_train_onehot, y_test_onehot: one-hot encoded labels
        norm_min, norm_max: normalization parameters
    """
    print_header("PREPARING DATA")

    # Remove samples where labels are at the boundary (last LOOKAHEAD_MAX bars)
    valid_mask = np.ones(len(labels), dtype=bool)
    # Remove first 200 bars (indicators need warmup)
    valid_mask[:200] = False
    # Remove last LOOKAHEAD_MAX bars (no valid label)
    valid_mask[-LABEL_LOOKAHEAD_MAX:] = False

    # Remove any rows with NaN or inf
    nan_mask = np.any(np.isnan(features) | np.isinf(features), axis=1)
    valid_mask &= ~nan_mask

    features_valid = features[valid_mask]
    labels_valid = labels[valid_mask]

    print(f"  Valid samples: {len(features_valid):,} (removed {len(features) - len(features_valid):,} invalid)")

    # Min-max normalization
    norm_min = np.min(features_valid, axis=0)
    norm_max = np.max(features_valid, axis=0)

    # Avoid division by zero for constant features
    range_vals = norm_max - norm_min
    range_vals[range_vals == 0] = 1.0

    features_normalized = (features_valid - norm_min) / range_vals

    # Convert labels to 0-indexed classes: SELL(-1)->0, NEUTRAL(0)->1, BUY(+1)->2
    labels_indexed = labels_valid + 1  # Now: 0=SELL, 1=NEUTRAL, 2=BUY

    # One-hot encode
    n_classes = 3
    labels_onehot = np.zeros((len(labels_indexed), n_classes))
    for i in range(len(labels_indexed)):
        labels_onehot[i, labels_indexed[i]] = 1.0

    # Time-series split (not random, to avoid data leakage)
    split_idx = int(len(features_normalized) * train_split)

    X_train = features_normalized[:split_idx]
    X_test = features_normalized[split_idx:]
    y_train = labels_indexed[:split_idx]
    y_test = labels_indexed[split_idx:]
    y_train_onehot = labels_onehot[:split_idx]
    y_test_onehot = labels_onehot[split_idx:]

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples:  {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}")

    # Class distribution in train set
    for cls, name in [(0, "SELL"), (1, "NEUTRAL"), (2, "BUY")]:
        count = np.sum(y_train == cls)
        print(f"    Train {name}: {count:,} ({count/len(y_train)*100:.1f}%)")

    return (X_train, X_test, y_train, y_test,
            y_train_onehot, y_test_onehot, norm_min, norm_max)


def train_model(X_train, y_train_onehot, X_test, y_test, y_test_onehot,
                epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EARLY_STOP_PATIENCE):
    """
    Train the neural network with mini-batch gradient descent and early stopping.

    Parameters:
        X_train: normalized training features
        y_train_onehot: one-hot training labels
        X_test: normalized test features
        y_test: integer test labels
        y_test_onehot: one-hot test labels
        epochs: maximum training epochs
        batch_size: mini-batch size
        patience: early stopping patience

    Returns:
        model: trained NeuralNetwork instance
        history: dict with training metrics per epoch
    """
    print_header("TRAINING NEURAL NETWORK")

    input_size = X_train.shape[1]
    print(f"  Architecture: {input_size} -> {' -> '.join(map(str, HIDDEN_SIZES))} -> 3")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print()

    model = NeuralNetwork(input_size)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
    best_test_loss = float("inf")
    best_weights = None
    best_biases = None
    patience_counter = 0

    n_train = len(X_train)
    n_batches = max(1, n_train // batch_size)

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        # Mini-batch training
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            output, cache = model.forward(X_batch)
            batch_loss = model.compute_loss(output, y_batch)
            epoch_loss += batch_loss

            # Backward pass
            gradients = model.backward(X_batch, y_batch, cache)

            # Update weights
            model.update_weights(gradients)

        epoch_loss /= n_batches

        # Evaluate on test set
        test_output, _ = model.forward(X_test)
        test_loss = model.compute_loss(test_output, y_test_onehot)
        test_preds = np.argmax(test_output, axis=1)
        test_accuracy = np.mean(test_preds == y_test) * 100

        history["train_loss"].append(epoch_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        # Learning rate decay
        model.learning_rate *= model.lr_decay

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = [w.copy() for w in model.weights]
            best_biases = [b.copy() for b in model.biases]
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}% | "
                  f"LR: {model.learning_rate:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases

    print(f"\n  Best test loss: {best_test_loss:.4f}")

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with detailed metrics.

    Parameters:
        model: trained NeuralNetwork
        X_test: test features
        y_test: integer test labels (0=SELL, 1=NEUTRAL, 2=BUY)

    Returns:
        metrics: dict with accuracy, precision, recall per class
    """
    print_header("MODEL EVALUATION")

    predictions, probabilities = model.predict(X_test)

    class_names = ["SELL", "NEUTRAL", "BUY"]

    # Overall accuracy
    accuracy = np.mean(predictions == y_test) * 100
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print()

    metrics = {"accuracy": accuracy, "classes": {}}

    # Per-class metrics
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*50}")

    for cls_idx, cls_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = np.sum((predictions == cls_idx) & (y_test == cls_idx))
        fp = np.sum((predictions == cls_idx) & (y_test != cls_idx))
        fn = np.sum((predictions != cls_idx) & (y_test == cls_idx))
        support = np.sum(y_test == cls_idx)

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  {cls_name:<10} {precision:>9.2f}% {recall:>9.2f}% {f1:>9.2f}% {support:>10,}")

        metrics["classes"][cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support)
        }

    print()

    # Confusion matrix
    print("  Confusion Matrix:")
    print(f"  {'':>10} {'Pred SELL':>10} {'Pred NEUT':>10} {'Pred BUY':>10}")
    for true_idx, true_name in enumerate(class_names):
        row = []
        for pred_idx in range(3):
            count = np.sum((predictions == pred_idx) & (y_test == true_idx))
            row.append(count)
        print(f"  {true_name:<10} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,}")

    return metrics


# =============================================================
#  MODEL EXPORT
# =============================================================

def export_model(model, norm_min, norm_max, feature_names, output_dir=MODEL_DIR):
    """
    Export trained model weights, biases, and normalization parameters
    as CSV files for the MQL5 EA to load.

    File format is designed for MQL5 FileReadString() parsing:
    - Weights: one row per input neuron, comma-separated values for each output neuron
    - Biases: single row with comma-separated values
    - Normalization: header row, then one row per feature with name,min,max values
    - Feature names: one feature name per line

    Parameters:
        model: trained NeuralNetwork
        norm_min: min values per feature for normalization
        norm_max: max values per feature for normalization
        feature_names: list of feature name strings
        output_dir: directory to save model files
    """
    print_header("EXPORTING MODEL")

    os.makedirs(output_dir, exist_ok=True)

    # Export weights
    weight_files = ["weights_layer1.csv", "weights_layer2.csv",
                    "weights_layer3.csv", "weights_output.csv"]

    for i, (weights, filename) in enumerate(zip(model.weights, weight_files)):
        filepath = os.path.join(output_dir, filename)
        # Save as CSV: one row per input neuron, columns = output neurons
        np.savetxt(filepath, weights, delimiter=",", fmt="%.8f")
        print(f"  Saved {filename} (shape: {weights.shape})")

    # Export biases
    bias_files = ["biases_layer1.csv", "biases_layer2.csv",
                  "biases_layer3.csv", "biases_output.csv"]

    for i, (biases, filename) in enumerate(zip(model.biases, bias_files)):
        filepath = os.path.join(output_dir, filename)
        # Save as single row CSV
        np.savetxt(filepath, biases, delimiter=",", fmt="%.8f")
        print(f"  Saved {filename} (shape: {biases.shape})")

    # Export normalization parameters (3 columns: feature_name,min,max)
    # This matches the MQL5 LoadNormParams parser which reads: name, min, max per row
    norm_filepath = os.path.join(output_dir, "normalization_params.csv")
    with open(norm_filepath, "w") as f:
        f.write("feature_name,min,max\n")
        for i, name in enumerate(feature_names):
            f.write(f"{name},{norm_min[i]:.8f},{norm_max[i]:.8f}\n")
    print(f"  Saved normalization_params.csv ({len(norm_min)} features)")

    # Export feature names
    names_filepath = os.path.join(output_dir, "feature_names.csv")
    with open(names_filepath, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"  Saved feature_names.csv ({len(feature_names)} features)")

    # Export model metadata as JSON
    metadata = {
        "input_size": int(model.weights[0].shape[0]),
        "hidden_sizes": HIDDEN_SIZES,
        "output_size": 3,
        "output_classes": ["SELL", "NEUTRAL", "BUY"],
        "activation": "relu",
        "output_activation": "softmax",
        "n_features": len(feature_names),
        "training_config": {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_split": TRAIN_SPLIT,
            "label_lookahead_min": LABEL_LOOKAHEAD_MIN,
            "label_lookahead_max": LABEL_LOOKAHEAD_MAX,
            "label_atr_multiplier": LABEL_ATR_MULTIPLIER
        },
        "ipda_config": {
            "swing_lookback": SWING_LOOKBACK,
            "ob_displacement_atr": OB_DISPLACEMENT_ATR,
            "fvg_min_size_atr": FVG_MIN_SIZE_ATR,
            "liquidity_sweep_buffer": LIQUIDITY_SWEEP_BUFFER,
            "bos_min_move_atr": BOS_MIN_MOVE_ATR
        }
    }

    meta_filepath = os.path.join(output_dir, "model_metadata.json")
    with open(meta_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved model_metadata.json")

    print(f"\n  All model files saved to: {output_dir}/")
    print(f"  Copy this folder to MT5: MQL5/Files/{output_dir}/")


# =============================================================
#  MAIN EXECUTION
# =============================================================

def main():
    """
    Main training pipeline execution.

    Steps:
        1. Load M1 OHLCV data from CSV
        2. Engineer IPDA-based features
        3. Generate labels from future price movement
        4. Prepare data (normalize, split)
        5. Train neural network
        6. Evaluate model
        7. Export weights and parameters
    """
    print("\n" + "=" * 60)
    print("  NAS100 ML Trainer - IPDA Feature Engineering")
    print("  Neural Network Training Pipeline")
    print("=" * 60)
    print(f"\n  Configuration:")
    print(f"    CSV File: {CSV_FILE}")
    print(f"    Model Output: {MODEL_DIR}/")
    print(f"    Architecture: input -> {' -> '.join(map(str, HIDDEN_SIZES))} -> 3")
    print(f"    Learning Rate: {LEARNING_RATE}")
    print(f"    Epochs: {EPOCHS}")
    print(f"    Batch Size: {BATCH_SIZE}")

    # Step 1: Load data
    df = load_data(CSV_FILE)

    # Step 2: Engineer features
    features, feature_names = engineer_features(df)

    # Step 3: Generate labels
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr = compute_atr(high, low, close, 14)
    labels, label_counts = generate_labels(close, atr)

    # Step 4: Prepare data
    (X_train, X_test, y_train, y_test,
     y_train_onehot, y_test_onehot, norm_min, norm_max) = prepare_data(features, labels)

    # Step 5: Train model
    model, history = train_model(X_train, y_train_onehot, X_test, y_test, y_test_onehot)

    # Step 6: Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 7: Export model
    export_model(model, norm_min, norm_max, feature_names)

    # Final summary
    print_header("TRAINING COMPLETE")
    print(f"  Final Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Model saved to: {MODEL_DIR}/")
    print(f"\n  Next steps:")
    print(f"  1. Copy {MODEL_DIR}/ to MT5: MQL5/Files/{MODEL_DIR}/")
    print(f"  2. The EA will load weights and normalize live data")
    print(f"  3. Feature computation order is in feature_names.csv")
    print(f"\n  Feature vector ({len(feature_names)} features):")
    for i, name in enumerate(feature_names):
        print(f"    [{i:>2}] {name}")

    return model, metrics, feature_names


if __name__ == "__main__":
    main()
