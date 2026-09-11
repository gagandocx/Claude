"""
Data sources.

Three ways in:

1. `load_csv` (in bars.py) for M1 exports. This is the one to use for real work.
   MT5: Tools > Options > Charts, raise "Max bars", then right-click the M1 chart
   and Save As. Or use the repo's ExportRealTicks.mq5.
2. `fetch_yfinance` for a quick look. Requires network + pandas + yfinance, and is
   capped at 7 days of M1 / 60 days of M15 by Yahoo, which is not enough data to
   conclude anything (see AUDIT.md, "Sample size and horizon").
3. `synthetic` for testing the machinery itself, offline.

On the synthetic generator: it exists to verify that the engine does not
manufacture profit, not to validate the strategy. `mode="martingale"` produces a
driftless random walk in which no path-based strategy can have a positive
expectancy after costs. If the engine reports a profit there, the engine is
broken. That test is the point.
"""

from __future__ import annotations

import math
import random

from .bars import Bar, Series, load_csv, tf_seconds, write_csv

# Rough intraday liquidity profile for gold, by UTC hour.
_SESSION_VOL = {
    **{h: 0.45 for h in range(0, 7)},    # Asia / off hours
    **{h: 1.25 for h in range(7, 12)},   # London
    **{h: 1.55 for h in range(12, 17)},  # London/NY overlap
    **{h: 1.00 for h in range(17, 21)},  # NY afternoon
    21: 0.35,
    22: 0.30,
    23: 0.40,
}


def synthetic(
    days: int = 120,
    start_price: float = 2000.0,
    annual_vol: float = 0.16,
    mode: str = "regime",
    seed: int = 7,
    start_ts: int = 1_700_000_000,
) -> Series:
    """
    Generate M1 bars.

    mode="martingale": zero drift everywhere. The null hypothesis.
    mode="regime":     Markov-switching drift (trend up / trend down / range) with
                       persistent volatility clustering. Has structure, so a
                       trend-following strategy can look good on it. That says
                       nothing about gold.
    """
    if mode not in ("martingale", "regime"):
        raise ValueError("mode must be 'martingale' or 'regime'")
    rng = random.Random(seed)
    per_min_sigma = annual_vol / math.sqrt(252 * 1440)

    # align start to midnight UTC Monday-ish
    start_ts -= start_ts % 86400

    bars: list[Bar] = []
    price = start_price
    state = 2  # 0 = up, 1 = down, 2 = range
    # Drift is expressed as a fraction of ONE DAY's sigma per day, then converted
    # to a per-minute rate. Setting it as a fraction of per-minute sigma instead
    # (the obvious-looking mistake) makes the daily drift ~13x the daily noise and
    # produces charts where gold falls from 2000 to 62 in a year.
    trend_per_day_in_sigmas = {0: 0.30, 1: -0.30, 2: 0.0}
    minutes_per_day = 1440.0
    vol_scale = {0: 1.15, 1: 1.25, 2: 0.85}
    switch_p = 1.0 / (3 * 1440)  # regimes last ~3 trading days
    vol_shock = 1.0

    ts = start_ts
    day = 0
    while day < days:
        weekday = ((ts // 86400) + 4) % 7  # 1970-01-01 was a Thursday
        if weekday >= 5:                    # Sat/Sun: no bars
            ts += 86400
            continue
        for minute in range(1440):
            bar_ts = ts + minute * 60
            hour = (bar_ts % 86400) // 3600
            if hour == 22:                  # CME maintenance break
                continue
            if mode == "regime" and rng.random() < switch_p:
                state = rng.choice([0, 1, 2, 2])
            # slow-moving volatility shock (clustering)
            vol_shock += (1.0 - vol_shock) * 0.001 + rng.gauss(0, 0.01)
            vol_shock = max(0.4, min(2.5, vol_shock))

            sigma = per_min_sigma * _SESSION_VOL.get(hour, 0.8) * vol_shock
            if mode == "regime":
                sigma *= vol_scale[state]
                mu = (
                    trend_per_day_in_sigmas[state]
                    * sigma
                    / math.sqrt(minutes_per_day)
                )
            else:
                mu = 0.0

            o = price
            steps = 6
            step_sigma = sigma / math.sqrt(steps)
            step_mu = mu / steps
            p = o
            hi = lo = o
            for _ in range(steps):
                p *= math.exp(step_mu + rng.gauss(0.0, step_sigma))
                hi = max(hi, p)
                lo = min(lo, p)
            c = p
            price = c
            bars.append(Bar(bar_ts, o, hi, lo, c, float(rng.randint(50, 400))))
        ts += 86400
        day += 1

    return Series(60, bars)


def fetch_yfinance(
    symbol: str = "GC=F",
    interval: str = "1m",
    period: str = "7d",
    out_csv: str | None = None,
) -> Series:
    """
    Optional convenience path. Imports are local so the package stays usable with
    no third-party dependencies installed.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "yfinance is not installed. `pip install yfinance pandas`, or export "
            "M1 bars from MT5 and use --csv."
        ) from exc

    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol} {interval}/{period}")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    secs = tf_seconds(interval)
    bars: list[Bar] = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp())
        try:
            o, h, l, c = (
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if any(map(lambda x: x != x, (o, h, l, c))):  # NaN
            continue
        v = float(row.get("Volume", 0.0) or 0.0)
        bars.append(Bar(ts, o, h, l, c, v))
    series = Series(secs, bars)
    if out_csv:
        write_csv(series, out_csv)
    return series


def split(series: Series, n: int) -> list[Series]:
    """Contiguous, equal-length-by-bar-count folds for walk-forward."""
    if n < 2:
        return [series]
    size = len(series) // n
    out = []
    for k in range(n):
        lo = k * size
        hi = (k + 1) * size if k < n - 1 else len(series)
        out.append(Series(series.tf, series.bars[lo:hi]))
    return out


def train_test(series: Series, train_frac: float = 0.6) -> tuple[Series, Series]:
    cut = int(len(series) * train_frac)
    return (
        Series(series.tf, series.bars[:cut]),
        Series(series.tf, series.bars[cut:]),
    )
