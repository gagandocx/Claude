"""
Technical Indicators - Pure NumPy Implementation
=================================================
All indicators take numpy arrays and return numpy arrays.
Vectorized where possible, loops where necessary for accuracy.

Indicator List:
    - RSI (Wilder smoothing)
    - ATR (Average True Range)
    - EMA (Exponential Moving Average)
    - SMA (Simple Moving Average)
    - Bollinger Bands
    - ADX with DI+/DI-
    - Stochastic Oscillator (%K, %D)
    - VWAP (Volume Weighted Average Price)
    - Keltner Channels
    - Hurst Exponent (fractal dimension estimate)
"""

import numpy as np
from typing import Tuple


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute RSI with Wilder smoothing (exponential moving average of gains/losses).

    Parameters
    ----------
    close : np.ndarray
        Close prices array.
    period : int
        RSI lookback period (default 14).

    Returns
    -------
    np.ndarray
        RSI values (0-100). Pre-warmup values are set to 50.0.
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing: initial average then exponential decay
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """
    Compute Average True Range using Wilder smoothing.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    period : int
        ATR lookback period.

    Returns
    -------
    np.ndarray
        ATR values. First (period-1) values are forward-filled with the first valid ATR.
    """
    n = len(high)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)

    # True Range calculation
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    if n < period:
        atr[:] = np.mean(tr[:n])
        return atr

    # Wilder smoothing
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Forward-fill warmup period
    atr[:period - 1] = atr[period - 1]
    return atr


def compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Exponential Moving Average.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    period : int
        EMA period (span).

    Returns
    -------
    np.ndarray
        EMA values. First value seeded with SMA of first `period` values.
    """
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema

    alpha = 2.0 / (period + 1.0)

    if n < period:
        ema[0] = data[0]
        for i in range(1, n):
            ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]
        return ema

    # Seed with SMA of first `period` values
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]

    # Fill initial values with SMA seed
    ema[:period - 1] = ema[period - 1]
    return ema


def compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Simple Moving Average using cumulative sum for speed.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    period : int
        SMA window.

    Returns
    -------
    np.ndarray
        SMA values. First (period-1) values are NaN.
    """
    n = len(data)
    sma = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return sma

    cumsum = np.cumsum(data)
    sma[period - 1:] = (cumsum[period - 1:] - np.concatenate(
        [[0.0], cumsum[:-period]]
    )) / period
    return sma


def compute_bollinger_bands(close: np.ndarray, period: int = 20,
                            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    period : int
        SMA/std lookback period.
    num_std : float
        Number of standard deviations for bands.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (upper_band, middle_band, lower_band)
    """
    n = len(close)
    middle = compute_sma(close, period)
    std = np.full(n, np.nan, dtype=np.float64)

    # Vectorized rolling std for valid range
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        std[i] = np.std(window, ddof=1)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ADX (Average Directional Index) with DI+ and DI-.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    period : int
        ADX smoothing period.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (adx, di_plus, di_minus) - all values 0-100.
    """
    n = len(high)
    adx = np.zeros(n, dtype=np.float64)
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)

    if n < period + 1:
        return adx, di_plus, di_minus

    # Directional Movement
    up_move = np.zeros(n, dtype=np.float64)
    down_move = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move[i] = high[i] - high[i - 1]
        down_move[i] = low[i - 1] - low[i]
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if up_move[i] > down_move[i] and up_move[i] > 0:
            plus_dm[i] = up_move[i]
        if down_move[i] > up_move[i] and down_move[i] > 0:
            minus_dm[i] = down_move[i]

    # Wilder smoothing of TR, +DM, -DM
    smooth_tr = np.zeros(n, dtype=np.float64)
    smooth_plus_dm = np.zeros(n, dtype=np.float64)
    smooth_minus_dm = np.zeros(n, dtype=np.float64)

    # Initial sum for first period
    smooth_tr[period] = np.sum(tr[1:period + 1])
    smooth_plus_dm[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - smooth_plus_dm[i - 1] / period + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - smooth_minus_dm[i - 1] / period + minus_dm[i]

    # DI+ and DI-
    for i in range(period, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            di_minus[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]

    # DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    # ADX: Wilder smoothed DX
    adx_start = 2 * period
    if adx_start < n:
        adx[adx_start] = np.mean(dx[period:adx_start + 1])
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, di_plus, di_minus


def compute_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    k_period : int
        %K lookback period.
    d_period : int
        %D smoothing period (SMA of %K).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (%K, %D) both in range 0-100.
    """
    n = len(close)
    k_values = np.full(n, 50.0, dtype=np.float64)

    for i in range(k_period - 1, n):
        window_high = np.max(high[i - k_period + 1:i + 1])
        window_low = np.min(low[i - k_period + 1:i + 1])
        range_val = window_high - window_low
        if range_val > 1e-10:
            k_values[i] = 100.0 * (close[i] - window_low) / range_val
        else:
            k_values[i] = 50.0

    # %D is SMA of %K
    d_values = np.full(n, 50.0, dtype=np.float64)
    for i in range(k_period + d_period - 2, n):
        d_values[i] = np.mean(k_values[i - d_period + 1:i + 1])

    return k_values, d_values


def compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Compute rolling VWAP (Volume Weighted Average Price).

    For intraday use, this computes a rolling VWAP over `period` bars
    rather than session-anchored VWAP.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    volume : np.ndarray
        Volume (tick_count or real volume).
    period : int
        Rolling window for VWAP calculation.

    Returns
    -------
    np.ndarray
        Rolling VWAP values.
    """
    n = len(close)
    vwap = np.zeros(n, dtype=np.float64)

    # Typical price
    typical_price = (high + low + close) / 3.0

    # Ensure volume has no zeros (use 1 as minimum)
    vol = np.maximum(volume, 1.0)

    tp_vol = typical_price * vol

    for i in range(period - 1, n):
        window_tpv = np.sum(tp_vol[i - period + 1:i + 1])
        window_vol = np.sum(vol[i - period + 1:i + 1])
        if window_vol > 0:
            vwap[i] = window_tpv / window_vol
        else:
            vwap[i] = close[i]

    # Fill warmup with first valid
    if n >= period:
        vwap[:period - 1] = vwap[period - 1]

    return vwap


def compute_keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             ema_period: int = 20, atr_period: int = 14,
                             atr_mult: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Keltner Channels (EMA +/- ATR multiplier).

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    ema_period : int
        EMA period for middle line.
    atr_period : int
        ATR calculation period.
    atr_mult : float
        Multiplier for ATR to determine channel width.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (upper_channel, middle_ema, lower_channel)
    """
    middle = compute_ema(close, ema_period)
    atr = compute_atr(high, low, close, atr_period)
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    return upper, middle, lower


def compute_hurst_exponent(close: np.ndarray, max_lag: int = 40) -> float:
    """
    Compute Hurst Exponent using Rescaled Range (R/S) analysis.

    H < 0.5: Mean-reverting (anti-persistent)
    H = 0.5: Random walk
    H > 0.5: Trending (persistent)

    Parameters
    ----------
    close : np.ndarray
        Close prices (at least 2*max_lag data points recommended).
    max_lag : int
        Maximum lag for R/S calculation.

    Returns
    -------
    float
        Hurst exponent estimate (clamped to [0.01, 0.99]).
    """
    n = len(close)
    if n < 20:
        return 0.5  # Not enough data, assume random walk

    # Use log returns for stationarity
    returns = np.diff(np.log(np.maximum(close, 1e-10)))
    n_ret = len(returns)

    if n_ret < 20:
        return 0.5

    # Compute R/S for different lag sizes
    lags = []
    rs_values = []

    min_lag = 10
    lag = min_lag
    while lag <= min(max_lag, n_ret // 2):
        rs_list = []
        num_segments = n_ret // lag
        for seg in range(num_segments):
            segment = returns[seg * lag:(seg + 1) * lag]
            if len(segment) < 2:
                continue
            mean_seg = np.mean(segment)
            deviations = np.cumsum(segment - mean_seg)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

        lag = int(lag * 1.4) + 1  # Logarithmic spacing

    if len(lags) < 3:
        return 0.5

    # Linear regression of log(R/S) vs log(lag)
    log_lags = np.log(np.array(lags, dtype=np.float64))
    log_rs = np.log(np.array(rs_values, dtype=np.float64))

    # Simple least squares: slope = Hurst exponent
    n_pts = len(log_lags)
    mean_x = np.mean(log_lags)
    mean_y = np.mean(log_rs)
    numerator = np.sum((log_lags - mean_x) * (log_rs - mean_y))
    denominator = np.sum((log_lags - mean_x) ** 2)

    if abs(denominator) < 1e-10:
        return 0.5

    hurst = numerator / denominator
    return float(np.clip(hurst, 0.01, 0.99))


def compute_hurst_rolling(close: np.ndarray, window: int = 100,
                          max_lag: int = 40) -> np.ndarray:
    """
    Compute rolling Hurst Exponent over a sliding window.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    window : int
        Rolling window size.
    max_lag : int
        Max lag for R/S analysis within each window.

    Returns
    -------
    np.ndarray
        Rolling Hurst values. Pre-warmup values are 0.5.
    """
    n = len(close)
    hurst = np.full(n, 0.5, dtype=np.float64)

    for i in range(window - 1, n):
        segment = close[i - window + 1:i + 1]
        hurst[i] = compute_hurst_exponent(segment, max_lag=max_lag)

    return hurst


def compute_volatility_ratio(atr: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Compute volatility ratio: current ATR / average ATR over lookback.

    Values > 1 indicate above-average volatility, < 1 indicates compression.

    Parameters
    ----------
    atr : np.ndarray
        ATR values.
    lookback : int
        Period over which to compute average ATR.

    Returns
    -------
    np.ndarray
        Volatility ratio values.
    """
    n = len(atr)
    ratio = np.ones(n, dtype=np.float64)

    for i in range(lookback, n):
        avg_atr = np.mean(atr[i - lookback:i])
        if avg_atr > 1e-10:
            ratio[i] = atr[i] / avg_atr

    return ratio


def compute_ema_slope(ema: np.ndarray, atr: np.ndarray,
                      lookback: int = 5) -> np.ndarray:
    """
    Compute EMA slope normalized by ATR (dimensionless trend strength).

    Positive = uptrend, Negative = downtrend.
    Magnitude indicates strength relative to volatility.

    Parameters
    ----------
    ema : np.ndarray
        EMA values.
    atr : np.ndarray
        ATR values for normalization.
    lookback : int
        Number of bars for slope calculation.

    Returns
    -------
    np.ndarray
        Normalized slope values.
    """
    n = len(ema)
    slope = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        raw_slope = (ema[i] - ema[i - lookback]) / lookback
        if atr[i] > 1e-10:
            slope[i] = raw_slope / atr[i]

    return slope
