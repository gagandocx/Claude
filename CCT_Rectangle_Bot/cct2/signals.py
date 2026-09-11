"""
The three strategy steps, with information timing made explicit.

Step 1  (4H)  direction candle: a sweep-and-engulf of the previous bar.
Step 2  (15M) weakness: a liquidity sweep of a confirmed pivot, rejected.
Step 3  (1M)  entry: a close through the rejection candle's rectangle.

Every object carries `known_ts`: the earliest wall-clock time at which a live
system could have known about it. The engine refuses to act on anything whose
`known_ts` is in the future. v1 had no such concept, which is how a 4H signal
ended up authorising M15 scans that began 8 hours earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bars import Bar, Series, atr, ema, rolling_percentile_rank, swings, Swing
from .settings import BiasFlags, StrategyConfig


@dataclass
class DirectionSignal:
    known_ts: int
    bar_ts: int
    direction: str          # 'long' | 'short'
    swept_level: float      # prior bar's extreme that was taken out
    bar_high: float
    bar_low: float
    bar_close: float
    atr: float
    expires_ts: int
    setups_spawned: int = 0

    def active_at(self, now: int) -> bool:
        return self.known_ts <= now <= self.expires_ts


@dataclass
class Setup:
    known_ts: int
    direction: str
    level: float            # the swept pivot
    rect_top: float
    rect_bottom: float
    expires_ts: int
    atr: float
    dir_bar_ts: int
    sweep_depth_atr: float
    dead: bool = False
    dead_reason: str = ""

    @property
    def trigger_price(self) -> float:
        return self.rect_top if self.direction == "long" else self.rect_bottom

    @property
    def invalidation_price(self) -> float:
        return self.rect_bottom if self.direction == "long" else self.rect_top

    def kill(self, reason: str) -> None:
        self.dead = True
        self.dead_reason = reason


class DirectionDetector:
    """Step 1. Emits at most one signal per 4H bar, at that bar's close."""

    def __init__(self, series: Series, cfg: StrategyConfig, bias: BiasFlags):
        self.bars = series.bars
        self.tf = series.tf
        self.cfg = cfg
        self.bias = bias
        self.ema = ema([b.close for b in self.bars], cfg.trend_ema)
        self.atr = atr(self.bars, cfg.direction_atr_period)

    def detect(self, i: int) -> DirectionSignal | None:
        if i < 1:
            return None
        cur, prev = self.bars[i], self.bars[i - 1]
        a = self.atr[i]
        if a is None or a <= 0:
            return None

        cfg = self.cfg
        swept_low = cur.low < prev.low
        swept_high = cur.high > prev.high
        prev_mid = (prev.high + prev.low) / 2.0

        engulf_long = swept_low and cur.close > prev.high
        engulf_short = swept_high and cur.close < prev.low
        reclaim_long = swept_low and cur.close > prev_mid and cur.close_position() >= 0.5
        reclaim_short = swept_high and cur.close < prev_mid and cur.close_position() <= 0.5

        mode = cfg.direction_mode
        if mode == "engulf":
            is_long, is_short = engulf_long, engulf_short
        elif mode == "reclaim":
            is_long, is_short = reclaim_long, reclaim_short
        elif mode == "engulf_or_reclaim":
            is_long = engulf_long or reclaim_long
            is_short = engulf_short or reclaim_short
        else:
            raise ValueError(f"unknown direction_mode {mode!r}")

        if is_long and is_short:
            # Outside bar that closed both beyond and below: no directional claim.
            return None
        if not is_long and not is_short:
            return None

        direction = "long" if is_long else "short"

        if cur.body < cfg.min_direction_body_atr * a:
            return None

        if cfg.use_trend_filter:
            e = self.ema[i]
            if e is None:
                return None
            if direction == "long" and cur.close < e:
                return None
            if direction == "short" and cur.close > e:
                return None

        # THE timing rule. v1 used `self.df.index[idx]`, which for a resampled 4H
        # frame is the bar's OPEN, and then also read bars[i+1].open.
        known_ts = cur.ts if self.bias.lookahead_direction else cur.ts + self.tf

        return DirectionSignal(
            known_ts=known_ts,
            bar_ts=cur.ts,
            direction=direction,
            swept_level=prev.low if direction == "long" else prev.high,
            bar_high=cur.high,
            bar_low=cur.low,
            bar_close=cur.close,
            atr=a,
            expires_ts=known_ts + int(cfg.direction_valid_hours * 3600),
        )


@dataclass
class _Level:
    pivot_idx: int
    price: float
    kind: str
    consumed: bool = False


class WeaknessDetector:
    """
    Step 2. Pivot levels enter the pool only once confirmed (`known_idx`), which
    is `swing_k` bars after the pivot itself. Sweep depth, rejection quality and
    rectangle size are all measured in ATR so the same numbers work on gold at
    1800 and at 3600.
    """

    def __init__(self, series: Series, cfg: StrategyConfig, bias: BiasFlags):
        self.bars = series.bars
        self.tf = series.tf
        self.cfg = cfg
        self.bias = bias
        self.atr = atr(self.bars, cfg.atr_period)
        self.vol_rank = rolling_percentile_rank(self.atr, cfg.vol_rank_window)
        self._swings: list[Swing] = swings(self.bars, cfg.swing_k)
        self._by_reveal: list[Swing] = sorted(
            self._swings,
            key=lambda s: (s.idx if bias.lookahead_swings else s.known_idx),
        )
        self._ptr = 0
        self.levels: list[_Level] = []

    def _reveal_index(self, s: Swing) -> int:
        return s.idx if self.bias.lookahead_swings else s.known_idx

    def advance(self, i: int) -> None:
        """Admit every pivot confirmed at or before bar i's close."""
        a = self.atr[i] or 0.0
        tol = self.cfg.level_dedupe_atr * a
        while self._ptr < len(self._by_reveal) and self._reveal_index(self._by_reveal[self._ptr]) <= i:
            s = self._by_reveal[self._ptr]
            self._ptr += 1
            dup = False
            if tol > 0:
                for lv in self.levels:
                    if lv.kind == s.kind and abs(lv.price - s.price) <= tol:
                        # Keep the more recent pivot at an equivalent price.
                        if s.idx > lv.pivot_idx:
                            lv.pivot_idx = s.idx
                            lv.price = s.price
                            lv.consumed = False
                        dup = True
                        break
            if not dup:
                self.levels.append(_Level(s.idx, s.price, s.kind))
        # bound memory / staleness
        cutoff = i - self.cfg.level_max_age_bars
        if len(self.levels) > 64:
            self.levels = [lv for lv in self.levels if lv.pivot_idx >= cutoff]

    def regime_ok(self, i: int) -> bool:
        r = self.vol_rank[i]
        if r is None:
            return True
        return self.cfg.vol_rank_min <= r <= self.cfg.vol_rank_max

    def detect(self, i: int, dsig: DirectionSignal) -> Setup | None:
        """
        Look for a sweep-and-reject on bar i, in the direction `dsig` allows.
        Returns at most one setup per bar (the deepest qualifying sweep).
        """
        cfg = self.cfg
        a = self.atr[i]
        if a is None or a <= 0:
            return None
        if not self.regime_ok(i):
            return None

        bar = self.bars[i]
        now = bar.ts + self.tf
        want_kind = "low" if dsig.direction == "long" else "high"
        min_depth = cfg.sweep_min_atr * a
        max_depth = cfg.sweep_max_atr * a
        cutoff = i - cfg.level_max_age_bars

        best: tuple[float, _Level] | None = None
        for lv in self.levels:
            if lv.consumed or lv.kind != want_kind or lv.pivot_idx < cutoff:
                continue
            if lv.pivot_idx >= i:
                continue  # pivot must predate the sweep bar
            if dsig.direction == "long":
                depth = lv.price - bar.low
                if depth < min_depth or depth > max_depth:
                    continue
                if bar.close <= lv.price:
                    continue  # did not reclaim the level
            else:
                depth = bar.high - lv.price
                if depth < min_depth or depth > max_depth:
                    continue
                if bar.close >= lv.price:
                    continue
            if best is None or depth > best[0]:
                best = (depth, lv)

        if best is None:
            return None
        depth, lv = best

        pos = bar.close_position()
        if dsig.direction == "long":
            if pos < cfg.min_close_position:
                return None
            rect_top, rect_bottom = bar.close, bar.low
        else:
            if (1.0 - pos) < cfg.min_close_position:
                return None
            rect_top, rect_bottom = bar.high, bar.close

        size = rect_top - rect_bottom
        if size < cfg.rect_min_atr * a or size > cfg.rect_max_atr * a:
            return None

        lv.consumed = True
        return Setup(
            known_ts=now,
            direction=dsig.direction,
            level=lv.price,
            rect_top=rect_top,
            rect_bottom=rect_bottom,
            expires_ts=now + cfg.entry_window_min * 60,
            atr=a,
            dir_bar_ts=dsig.bar_ts,
            sweep_depth_atr=depth / a,
        )
