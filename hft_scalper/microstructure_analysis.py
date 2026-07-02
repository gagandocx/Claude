"""
Microstructure analysis module for XAUUSD tick data.

Computes comprehensive market microstructure metrics:
- Spread distribution statistics
- Volatility clustering (realized vol, GARCH-like metrics)
- Autocorrelation at multiple lags (tick and bar level)
- Mean-reversion half-life estimation (Ornstein-Uhlenbeck)
- Momentum persistence at different horizons
- Time-of-day patterns (volume, volatility, spread)
- Order flow imbalance from bid/ask changes
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression


def compute_spread_stats(ticks: pd.DataFrame) -> dict:
    """Compute comprehensive spread distribution statistics."""
    spread = ticks["spread"].values
    spread_positive = spread[spread > 0]

    return {
        "mean_spread": float(np.mean(spread_positive)),
        "median_spread": float(np.median(spread_positive)),
        "std_spread": float(np.std(spread_positive)),
        "min_spread": float(np.min(spread_positive)),
        "max_spread": float(np.max(spread_positive)),
        "p5_spread": float(np.percentile(spread_positive, 5)),
        "p25_spread": float(np.percentile(spread_positive, 25)),
        "p75_spread": float(np.percentile(spread_positive, 75)),
        "p95_spread": float(np.percentile(spread_positive, 95)),
        "p99_spread": float(np.percentile(spread_positive, 99)),
        "spread_skew": float(stats.skew(spread_positive)),
        "spread_kurtosis": float(stats.kurtosis(spread_positive)),
        "pct_tight_spread_below_median": float(
            np.mean(spread_positive <= np.median(spread_positive)) * 100
        ),
    }


def compute_volatility_metrics(bars: pd.DataFrame) -> dict:
    """
    Compute volatility clustering and realized volatility metrics.

    Uses bar-level returns to measure:
    - Realized volatility at different windows
    - Volatility autocorrelation (clustering measure)
    - Parkinson volatility estimator
    - Garman-Klass volatility
    """
    returns = bars["returns"].dropna().values
    log_returns = bars["log_returns"].dropna().values

    # Realized volatility at different windows
    rv_1min = np.std(returns) * np.sqrt(60)  # annualized approx (bars per hour)
    rv_5min = np.std(pd.Series(returns).rolling(5).sum().dropna().values) * np.sqrt(12)
    rv_15min = np.std(pd.Series(returns).rolling(15).sum().dropna().values) * np.sqrt(4)

    # Volatility clustering: autocorrelation of squared returns
    sq_returns = returns ** 2
    vol_autocorr_1 = float(np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1])
    vol_autocorr_5 = float(
        np.corrcoef(sq_returns[:-5], sq_returns[5:])[0, 1]
    ) if len(sq_returns) > 10 else 0.0
    vol_autocorr_10 = float(
        np.corrcoef(sq_returns[:-10], sq_returns[10:])[0, 1]
    ) if len(sq_returns) > 20 else 0.0

    # Parkinson volatility (uses high-low range)
    hl = np.log(bars["high"].values / bars["low"].values)
    hl = hl[np.isfinite(hl) & (hl > 0)]
    parkinson_vol = float(np.sqrt(np.mean(hl ** 2) / (4 * np.log(2))))

    # Garman-Klass volatility
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[valid], h[valid], l[valid], c[valid]
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    gk_vol = float(np.sqrt(np.mean(gk[np.isfinite(gk)])))

    return {
        "realized_vol_1min_annualized": float(rv_1min),
        "realized_vol_5min_annualized": float(rv_5min),
        "realized_vol_15min_annualized": float(rv_15min),
        "daily_vol_pct": float(np.std(returns) * np.sqrt(1440) * 100),
        "vol_autocorr_lag1": vol_autocorr_1,
        "vol_autocorr_lag5": vol_autocorr_5,
        "vol_autocorr_lag10": vol_autocorr_10,
        "vol_clustering_strength": float((vol_autocorr_1 + vol_autocorr_5) / 2),
        "parkinson_vol": parkinson_vol,
        "garman_klass_vol": gk_vol,
        "mean_abs_return": float(np.mean(np.abs(returns))),
        "max_return": float(np.max(returns)),
        "min_return": float(np.min(returns)),
        "return_kurtosis": float(stats.kurtosis(returns)),
        "return_skew": float(stats.skew(returns)),
    }


def compute_autocorrelation(ticks: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """
    Compute return autocorrelation at multiple lags.

    Positive autocorrelation = momentum tendency
    Negative autocorrelation = mean-reversion tendency
    """
    # Tick-level autocorrelations
    mid = ticks["mid"].values
    tick_returns = np.diff(mid) / mid[:-1]
    tick_returns = tick_returns[np.isfinite(tick_returns)]

    # Subsample for efficiency (every 10th tick for large datasets)
    if len(tick_returns) > 1_000_000:
        sample = tick_returns[::10]
    else:
        sample = tick_returns

    tick_ac = {}
    for lag in [1, 5, 10, 20, 50, 100]:
        if len(sample) > lag + 100:
            corr = np.corrcoef(sample[:-lag], sample[lag:])[0, 1]
            tick_ac[f"tick_autocorr_lag{lag}"] = float(corr)

    # Bar-level autocorrelations
    bar_returns = bars["returns"].dropna().values
    bar_ac = {}
    for lag in [1, 2, 3, 5, 10, 20, 30, 60]:
        if len(bar_returns) > lag + 50:
            corr = np.corrcoef(bar_returns[:-lag], bar_returns[lag:])[0, 1]
            bar_ac[f"bar_autocorr_lag{lag}"] = float(corr)

    # Determine dominant regime
    short_term_ac = np.mean([v for k, v in tick_ac.items() if "lag1" in k or "lag5" in k] or [0])
    bar_short_ac = np.mean([v for k, v in bar_ac.items() if "lag1" in k or "lag2" in k] or [0])

    if short_term_ac < -0.01:
        tick_regime = "mean_reverting"
    elif short_term_ac > 0.01:
        tick_regime = "momentum"
    else:
        tick_regime = "random_walk"

    if bar_short_ac < -0.01:
        bar_regime = "mean_reverting"
    elif bar_short_ac > 0.01:
        bar_regime = "momentum"
    else:
        bar_regime = "random_walk"

    return {
        **tick_ac,
        **bar_ac,
        "tick_regime": tick_regime,
        "bar_regime": bar_regime,
        "short_term_tick_ac_mean": float(short_term_ac),
        "short_term_bar_ac_mean": float(bar_short_ac),
    }


def compute_mean_reversion_halflife(bars: pd.DataFrame) -> dict:
    """
    Estimate mean-reversion half-life using Ornstein-Uhlenbeck model.

    Uses ADF-style regression: delta_y = alpha + beta * y_lag + epsilon
    Half-life = -ln(2) / beta
    """
    price = bars["close"].dropna().values

    # Log price for stationarity
    log_price = np.log(price)

    # ADF regression: delta_y = alpha + beta * y_{t-1}
    y = np.diff(log_price)
    x = log_price[:-1].reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(x, y)
    beta = reg.coef_[0]

    if beta >= 0:
        half_life_bars = np.inf
        half_life_minutes = np.inf
    else:
        half_life_bars = -np.log(2) / beta
        half_life_minutes = half_life_bars  # 1-min bars

    # Also compute for different lookback windows
    half_lives = {}
    for window_name, window in [("1h", 60), ("4h", 240), ("1d", 1440)]:
        if len(log_price) > window + 10:
            segment = log_price[-window:]
            y_seg = np.diff(segment)
            x_seg = segment[:-1].reshape(-1, 1)
            reg_seg = LinearRegression()
            reg_seg.fit(x_seg, y_seg)
            b = reg_seg.coef_[0]
            if b < 0:
                half_lives[f"half_life_{window_name}"] = float(-np.log(2) / b)
            else:
                half_lives[f"half_life_{window_name}"] = None

    # Hurst exponent estimation (simple R/S method)
    returns = np.diff(log_price)
    n = len(returns)
    max_k = min(1000, n // 4)
    rs_values = []
    ns = []
    for k in [10, 20, 50, 100, 200, 500, max_k]:
        if k > n:
            continue
        num_segments = n // k
        rs_list = []
        for i in range(num_segments):
            segment = returns[i * k:(i + 1) * k]
            mean_seg = np.mean(segment)
            cumdev = np.cumsum(segment - mean_seg)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns.append(k)

    if len(ns) > 2:
        log_n = np.log(ns)
        log_rs = np.log(rs_values)
        hurst_reg = LinearRegression()
        hurst_reg.fit(log_n.reshape(-1, 1), log_rs)
        hurst_exponent = float(hurst_reg.coef_[0])
    else:
        hurst_exponent = 0.5

    return {
        "mean_reversion_half_life_bars": float(half_life_bars) if np.isfinite(half_life_bars) else None,
        "mean_reversion_half_life_minutes": float(half_life_minutes) if np.isfinite(half_life_minutes) else None,
        "ou_beta": float(beta),
        "hurst_exponent": float(hurst_exponent),
        "hurst_interpretation": (
            "strongly_mean_reverting" if hurst_exponent < 0.4 else
            "mean_reverting" if hurst_exponent < 0.45 else
            "random_walk" if hurst_exponent < 0.55 else
            "trending" if hurst_exponent < 0.65 else
            "strongly_trending"
        ),
        **half_lives,
    }


def compute_momentum_persistence(bars: pd.DataFrame) -> dict:
    """
    Measure momentum persistence at different horizons.

    Checks if returns over period N predict returns over period N+1.
    """
    returns = bars["returns"].dropna().values
    results = {}

    for horizon in [1, 3, 5, 10, 15, 30, 60]:
        if len(returns) < horizon * 3:
            continue

        # Rolling returns over horizon
        rolling_ret = pd.Series(returns).rolling(horizon).sum().dropna().values

        # Correlation between consecutive non-overlapping windows
        n_pairs = len(rolling_ret) // horizon
        if n_pairs < 10:
            continue

        past = rolling_ret[::horizon][:-1]
        future = rolling_ret[::horizon][1:]
        min_len = min(len(past), len(future))
        past = past[:min_len]
        future = future[:min_len]

        if len(past) > 5:
            corr = np.corrcoef(past, future)[0, 1]
            # Win rate: same sign continuation
            same_sign = np.mean(np.sign(past) == np.sign(future))
            results[f"momentum_{horizon}bar_corr"] = float(corr)
            results[f"momentum_{horizon}bar_continuation_pct"] = float(same_sign * 100)

    # Overall momentum score
    momentum_corrs = [v for k, v in results.items() if "corr" in k]
    results["overall_momentum_score"] = float(np.mean(momentum_corrs)) if momentum_corrs else 0.0

    return results


def compute_time_of_day_patterns(bars: pd.DataFrame) -> dict:
    """
    Analyze how volume, volatility, and spread vary by hour of day (UTC).
    """
    bars_copy = bars.copy()
    bars_copy["hour"] = bars_copy.index.hour

    hourly_stats = {}
    for hour in range(24):
        hour_bars = bars_copy[bars_copy["hour"] == hour]
        if len(hour_bars) < 10:
            continue

        hourly_stats[f"hour_{hour:02d}"] = {
            "avg_tick_count": float(hour_bars["tick_count"].mean()),
            "avg_range": float(hour_bars["range"].mean()),
            "avg_spread": float(hour_bars["avg_spread"].mean()),
            "volatility": float(hour_bars["returns"].std()),
            "num_bars": int(len(hour_bars)),
        }

    # Find most active hours
    if hourly_stats:
        hours_by_vol = sorted(
            hourly_stats.items(),
            key=lambda x: x[1]["volatility"],
            reverse=True
        )
        peak_hours = [h[0] for h in hours_by_vol[:5]]
        quiet_hours = [h[0] for h in hours_by_vol[-5:]]
    else:
        peak_hours = []
        quiet_hours = []

    return {
        "hourly_stats": hourly_stats,
        "peak_volatility_hours": peak_hours,
        "quiet_hours": quiet_hours,
        "total_hours_covered": len(hourly_stats),
    }


def compute_order_flow_imbalance(ticks: pd.DataFrame) -> dict:
    """
    Compute order flow imbalance metrics from bid/ask changes.

    Classifies ticks based on:
    - Uptick: bid increased
    - Downtick: bid decreased
    - Bid/Ask change asymmetry
    """
    bid = ticks["bid"].values
    ask = ticks["ask"].values

    bid_change = np.diff(bid)
    ask_change = np.diff(ask)

    # Tick classification
    upticks = np.sum(bid_change > 0)
    downticks = np.sum(bid_change < 0)
    unchanged = np.sum(bid_change == 0)
    total = len(bid_change)

    # Order flow imbalance (OFI)
    # Positive OFI = buying pressure, Negative OFI = selling pressure
    ofi = bid_change - ask_change
    cumulative_ofi = np.cumsum(ofi)

    # Rolling OFI for different windows
    ofi_series = pd.Series(ofi)
    ofi_1000 = ofi_series.rolling(1000).sum().dropna().values
    ofi_5000 = ofi_series.rolling(5000).sum().dropna().values

    # OFI autocorrelation (persistence of flow)
    if len(ofi) > 2000:
        ofi_ac_1 = float(np.corrcoef(ofi[:-1], ofi[1:])[0, 1])
        ofi_blocks = ofi_series.rolling(100).sum().dropna().values
        if len(ofi_blocks) > 200:
            ofi_ac_block = float(np.corrcoef(ofi_blocks[:-100], ofi_blocks[100:])[0, 1])
        else:
            ofi_ac_block = 0.0
    else:
        ofi_ac_1 = 0.0
        ofi_ac_block = 0.0

    # VPIN-like metric (Volume-synchronized Probability of Informed Trading)
    bucket_size = 10000
    n_buckets = len(bid_change) // bucket_size
    buy_volume = []
    sell_volume = []
    for i in range(n_buckets):
        chunk = bid_change[i * bucket_size:(i + 1) * bucket_size]
        buy_volume.append(np.sum(chunk > 0))
        sell_volume.append(np.sum(chunk < 0))

    buy_volume = np.array(buy_volume)
    sell_volume = np.array(sell_volume)
    if len(buy_volume) > 0:
        vpin = np.mean(np.abs(buy_volume - sell_volume) / (buy_volume + sell_volume + 1))
    else:
        vpin = 0.0

    return {
        "uptick_pct": float(upticks / total * 100),
        "downtick_pct": float(downticks / total * 100),
        "unchanged_pct": float(unchanged / total * 100),
        "net_tick_imbalance": float((upticks - downticks) / total),
        "ofi_mean": float(np.mean(ofi)),
        "ofi_std": float(np.std(ofi)),
        "ofi_skew": float(stats.skew(ofi)),
        "cumulative_ofi_final": float(cumulative_ofi[-1]) if len(cumulative_ofi) > 0 else 0.0,
        "ofi_autocorr_tick": ofi_ac_1,
        "ofi_autocorr_block": ofi_ac_block,
        "vpin_estimate": float(vpin),
        "ofi_1000tick_std": float(np.std(ofi_1000)) if len(ofi_1000) > 0 else 0.0,
        "ofi_5000tick_std": float(np.std(ofi_5000)) if len(ofi_5000) > 0 else 0.0,
    }


def run_full_analysis(ticks: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """
    Run the complete microstructure analysis pipeline.

    Returns a dictionary with all computed metrics.
    """
    print("\n=== MICROSTRUCTURE ANALYSIS ===\n")

    print("1/6 Computing spread statistics...")
    spread_stats = compute_spread_stats(ticks)

    print("2/6 Computing volatility metrics...")
    vol_metrics = compute_volatility_metrics(bars)

    print("3/6 Computing autocorrelation structure...")
    autocorr = compute_autocorrelation(ticks, bars)

    print("4/6 Estimating mean-reversion half-life...")
    mr_metrics = compute_mean_reversion_halflife(bars)

    print("5/6 Computing momentum persistence...")
    momentum = compute_momentum_persistence(bars)

    print("6/6 Computing order flow imbalance...")
    ofi = compute_order_flow_imbalance(ticks)

    print("\n Computing time-of-day patterns...")
    tod = compute_time_of_day_patterns(bars)

    # Summary
    report = {
        "num_ticks": len(ticks),
        "num_bars": len(bars),
        "price_range_low": float(ticks["mid"].min()),
        "price_range_high": float(ticks["mid"].max()),
        "data_start": str(ticks["timestamp"].iloc[0]),
        "data_end": str(ticks["timestamp"].iloc[-1]),
        "spread_stats": spread_stats,
        "volatility_metrics": vol_metrics,
        "autocorrelation": autocorr,
        "mean_reversion": mr_metrics,
        "momentum": momentum,
        "time_of_day": tod,
        "order_flow": ofi,
    }

    # Add top-level convenience fields
    report["mean_spread"] = spread_stats["mean_spread"]
    report["hurst_exponent"] = mr_metrics["hurst_exponent"]
    report["dominant_tick_regime"] = autocorr["tick_regime"]
    report["dominant_bar_regime"] = autocorr["bar_regime"]
    report["vol_clustering_strength"] = vol_metrics["vol_clustering_strength"]
    report["half_life_bars"] = mr_metrics["mean_reversion_half_life_bars"]

    return report
