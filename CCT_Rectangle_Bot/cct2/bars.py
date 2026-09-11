"""
Bar primitives, resampling and causal indicators.

Dependency-free on purpose: the whole engine must be runnable and testable in an
environment with no network and no pandas/numpy. pandas is only used at the edges
(optional yfinance fetch in `data.py`).

Timestamp convention, enforced everywhere:
    Bar.ts is the bar's OPEN time (epoch seconds, UTC).
    A bar's information becomes available at ts + tf, its CLOSE time.
Nothing in this package is allowed to act on a bar before its close time. That
single rule is what defect #1 in AUDIT.md violated.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone

TIMEFRAMES = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def tf_seconds(name: str) -> int:
    if name not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {name!r}; known: {sorted(TIMEFRAMES)}")
    return TIMEFRAMES[name]


def tf_name(seconds: int) -> str:
    for name, secs in TIMEFRAMES.items():
        if secs == seconds:
            return name
    return f"{seconds}s"


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def parse_ts(raw: str) -> int:
    """Accept epoch seconds/ms or common ISO-ish datetime strings (assumed UTC)."""
    raw = raw.strip()
    if not raw:
        raise ValueError("empty timestamp")
    # epoch?
    try:
        val = float(raw)
    except ValueError:
        pass
    else:
        if val > 1e11:  # milliseconds
            val /= 1000.0
        return int(val)
    txt = raw.replace("T", " ")
    for suffix in ("+00:00", "Z", " UTC", "+0000"):
        if txt.endswith(suffix):
            txt = txt[: -len(suffix)]
            break
    txt = txt.strip()
    fmts = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(txt, fmt)
        except ValueError:
            continue
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    raise ValueError(f"unparseable timestamp {raw!r}")


@dataclass(frozen=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    def close_position(self) -> float:
        """Where the close sits inside the range: 0.0 = at the low, 1.0 = at the high."""
        rng = self.range
        if rng <= 0:
            return 0.5
        return (self.close - self.low) / rng


class Series:
    """An ordered, gap-tolerant list of same-timeframe bars."""

    __slots__ = ("tf", "bars")

    def __init__(self, tf: int, bars: list[Bar]):
        self.tf = tf
        self.bars = bars

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, i):
        return self.bars[i]

    def __iter__(self):
        return iter(self.bars)

    @property
    def name(self) -> str:
        return tf_name(self.tf)

    def close_ts(self, i: int) -> int:
        """Time at which bar i's information becomes usable."""
        return self.bars[i].ts + self.tf

    def slice_ts(self, start_ts: int, end_ts: int) -> "Series":
        return Series(self.tf, [b for b in self.bars if start_ts <= b.ts < end_ts])

    def span(self) -> tuple[int, int]:
        if not self.bars:
            return (0, 0)
        return (self.bars[0].ts, self.bars[-1].ts + self.tf)

    def days(self) -> float:
        lo, hi = self.span()
        return (hi - lo) / 86400.0

    def describe(self) -> str:
        if not self.bars:
            return f"{self.name}: empty"
        lo, hi = self.span()
        return (
            f"{self.name}: {len(self.bars)} bars  {fmt_ts(lo)} -> {fmt_ts(hi)}  "
            f"({(hi - lo) / 86400.0:.1f} calendar days)"
        )


def resample(src: Series, tf: int) -> Series:
    """
    Aggregate to a higher timeframe. Buckets are aligned to the epoch, which for
    4h means 00:00/04:00/08:00... UTC (same alignment pandas `resample("4h")` uses).

    A bucket is emitted only if at least one source bar fell in it; partial final
    buckets are dropped so that no bar is ever treated as closed early.
    """
    if tf % src.tf != 0:
        raise ValueError(f"{tf_name(tf)} is not a multiple of {src.name}")
    if tf == src.tf:
        return Series(src.tf, list(src.bars))

    out: list[Bar] = []
    cur_key = None
    o = h = l = c = 0.0
    vol = 0.0
    for b in src.bars:
        key = b.ts - (b.ts % tf)
        if key != cur_key:
            if cur_key is not None:
                out.append(Bar(cur_key, o, h, l, c, vol))
            cur_key = key
            o, h, l, c, vol = b.open, b.high, b.low, b.close, b.volume
        else:
            h = max(h, b.high)
            l = min(l, b.low)
            c = b.close
            vol += b.volume
    if cur_key is not None:
        last_src_close = src.bars[-1].ts + src.tf
        if last_src_close >= cur_key + tf:
            out.append(Bar(cur_key, o, h, l, c, vol))
    return Series(tf, out)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

_TIME_KEYS = ("time", "timestamp", "date", "datetime", "ts", "open_time")
_OHLC_KEYS = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "close/last"),
    "volume": ("volume", "v", "vol", "tickvol", "tick_volume"),
}


def load_csv(path: str, tf: str | int) -> Series:
    """
    Load OHLCV from CSV. Header names are matched case-insensitively against the
    usual spellings; a separate 'date' + 'time' pair is also supported (MT5 export).
    """
    secs = tf_seconds(tf) if isinstance(tf, str) else tf
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row")
        lower = {(name or "").strip().lower(): name for name in reader.fieldnames}

        time_col = next((lower[k] for k in _TIME_KEYS if k in lower), None)
        date_col = lower.get("date")
        clock_col = lower.get("time")
        split_time = time_col is None and date_col is not None and clock_col is not None
        if time_col is None and not split_time:
            raise ValueError(f"{path}: no recognisable time column in {reader.fieldnames}")

        cols = {}
        for field, aliases in _OHLC_KEYS.items():
            cols[field] = next((lower[a] for a in aliases if a in lower), None)
        for field in ("open", "high", "low", "close"):
            if cols[field] is None:
                raise ValueError(f"{path}: missing '{field}' column")

        for row in reader:
            try:
                if split_time:
                    ts = parse_ts(f"{row[date_col]} {row[clock_col]}")
                else:
                    ts = parse_ts(row[time_col])
                o = float(row[cols["open"]])
                h = float(row[cols["high"]])
                l = float(row[cols["low"]])
                c = float(row[cols["close"]])
            except (ValueError, TypeError, KeyError):
                continue  # header repeats, blank lines, 'null' rows
            v = 0.0
            if cols["volume"]:
                try:
                    v = float(row[cols["volume"]])
                except (ValueError, TypeError):
                    v = 0.0
            if not (h >= max(o, c) and l <= min(o, c)):
                continue  # inconsistent bar
            bars.append(Bar(ts, o, h, l, c, v))

    bars.sort(key=lambda b: b.ts)
    deduped: list[Bar] = []
    for b in bars:
        if deduped and deduped[-1].ts == b.ts:
            deduped[-1] = b
            continue
        deduped.append(b)
    return Series(secs, deduped)


def write_csv(series: Series, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in series.bars:
            w.writerow(
                [
                    datetime.fromtimestamp(b.ts, tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    f"{b.open:.5f}",
                    f"{b.high:.5f}",
                    f"{b.low:.5f}",
                    f"{b.close:.5f}",
                    f"{b.volume:.0f}",
                ]
            )


# --------------------------------------------------------------------------- #
# Causal indicators
#
# Each returns a list aligned to `bars`. Element i uses bars[0..i] only, so it is
# safe to read element i at bar i's close time and never before.
# --------------------------------------------------------------------------- #


def ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if not values:
        return out
    alpha = 2.0 / (period + 1.0)
    acc = values[0]
    for i, v in enumerate(values):
        acc = v if i == 0 else acc + alpha * (v - acc)
        # withhold until there is enough history for the average to mean anything
        out[i] = acc if i >= period - 1 else None
    return out


def atr(bars: list[Bar], period: int) -> list[float | None]:
    """Wilder's ATR."""
    out: list[float | None] = [None] * len(bars)
    if len(bars) < 2:
        return out
    trs: list[float] = [bars[0].range]
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        b = bars[i]
        trs.append(max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close)))
    acc = None
    for i in range(len(bars)):
        if i < period - 1:
            continue
        if acc is None:
            acc = sum(trs[i - period + 1 : i + 1]) / period
        else:
            acc = (acc * (period - 1) + trs[i]) / period
        out[i] = acc
    return out


def rolling_percentile_rank(values: list[float | None], window: int) -> list[float | None]:
    """
    Rank of values[i] within the trailing `window` values, in [0, 1].
    Used for the volatility-regime filter without leaking the full-sample
    distribution (which is what a global percentile would do).
    """
    out: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        v = values[i]
        if v is None:
            continue
        lo = max(0, i - window + 1)
        hist = [x for x in values[lo : i + 1] if x is not None]
        if len(hist) < max(10, window // 4):
            continue
        below = sum(1 for x in hist if x <= v)
        out[i] = below / len(hist)
    return out


@dataclass(frozen=True)
class Swing:
    idx: int          # bar index of the pivot itself
    known_idx: int    # first bar index at whose close the pivot is confirmed
    ts: int           # pivot bar open time
    price: float
    kind: str         # 'high' | 'low'


def swings(bars: list[Bar], k: int) -> list[Swing]:
    """
    Fractal pivots with `k` bars on each side.

    A pivot at i cannot be known until bar i+k has closed, so `known_idx = i + k`
    is carried with it and the engine filters on that. v1 ignored this and read
    pivots the moment they existed in the array.
    """
    out: list[Swing] = []
    n = len(bars)
    for i in range(k, n - k):
        hi = bars[i].high
        lo = bars[i].low
        is_high = True
        is_low = True
        for j in range(1, k + 1):
            if hi <= bars[i - j].high or hi <= bars[i + j].high:
                is_high = False
            if lo >= bars[i - j].low or lo >= bars[i + j].low:
                is_low = False
            if not is_high and not is_low:
                break
        if is_high:
            out.append(Swing(i, i + k, bars[i].ts, hi, "high"))
        if is_low:
            out.append(Swing(i, i + k, bars[i].ts, lo, "low"))
    out.sort(key=lambda s: s.known_idx)
    return out


def fair_value_gaps(bars: list[Bar], min_size: float) -> list[tuple[int, str, float, float]]:
    """
    Three-bar imbalances, returned as (known_idx, kind, bottom, top).
    Known at the close of the third bar, which is the index reported.
    """
    out = []
    for i in range(2, len(bars)):
        a, c = bars[i - 2], bars[i]
        if c.low - a.high >= min_size:
            out.append((i, "bullish", a.high, c.low))
        elif a.low - c.high >= min_size:
            out.append((i, "bearish", c.high, a.low))
    return out


def stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)
