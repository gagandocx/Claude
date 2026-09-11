"""
Event-driven backtester.

Design constraints, each one a direct answer to a defect in AUDIT.md:

* One pass, forward in time, over the entry timeframe. Higher timeframe bars are
  admitted only when their close time has been reached, so a decision can never
  consume a bar that has not finished. (#1)
* Setups are killed the moment price trades to the far side of the rectangle. (#2)
* No trade outcome is ever invented. If data runs out while a position is open,
  it closes at the last available price and is tagged `data_end`. (#3)
* Equity changes at exit time, in chronological order, because that is the only
  order the loop can produce. Position sizing therefore only ever sees P&L that
  had actually been realised. (#4)
* Inside a bar, when both the stop and the target are touched, the stop wins.
  Trailing stops move at bar close and take effect from the next bar. (#5)
* Spread, slippage and commission are charged on every side of every fill. (#6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bars import Bar, Series, fmt_ts, resample, tf_seconds
from .settings import BiasFlags, CostModel, RiskConfig, RunConfig, StrategyConfig
from .signals import DirectionDetector, DirectionSignal, Setup, WeaknessDetector


@dataclass
class Trade:
    tid: int
    direction: str
    entry_ts: int
    entry_price: float          # actual fill, costs included
    signal_price: float         # the close the decision was made on
    initial_stop: float
    stop: float
    target: float | None
    partial_level: float | None
    lots_initial: float
    lots_open: float
    risk_per_unit: float
    risk_dollars: float
    atr: float
    entry_bar_idx: int
    dir_bar_ts: int
    sweep_depth_atr: float
    equity_at_entry: float
    best_price: float
    worst_price: float
    trail_active: bool = False
    partial_done: bool = False
    commission_paid: float = 0.0
    pnl: float = 0.0
    exit_ts: int | None = None
    exit_price: float | None = None
    bars_held: int = 0
    reason: str = ""
    mfe_r: float = 0.0
    mae_r: float = 0.0

    @property
    def r(self) -> float:
        return self.pnl / self.risk_dollars if self.risk_dollars > 0 else 0.0

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_daily: list[tuple[int, float]] = field(default_factory=list)
    equity_by_trade: list[tuple[int, float]] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    start_ts: int = 0
    end_ts: int = 0
    setups_created: int = 0
    signals_created: int = 0
    rejects: dict[str, int] = field(default_factory=dict)
    ruined: bool = False
    config: dict = field(default_factory=dict)
    label: str = ""

    @property
    def days(self) -> float:
        return max((self.end_ts - self.start_ts) / 86400.0, 1e-9)


class Backtester:
    def __init__(
        self,
        entry: Series,
        cfg: RunConfig,
        weakness: Series | None = None,
        direction: Series | None = None,
        label: str = "",
        trade_from_ts: int | None = None,
    ):
        s = cfg.strategy
        self.cfg = cfg
        self.s = s
        self.costs = cfg.costs
        self.risk = cfg.risk
        self.bias = cfg.bias
        self.label = label
        self.trade_from_ts = trade_from_ts
        """Bars before this are used to warm indicators up but produce no trades.
        Walk-forward folds need history to compute a 4H EMA50; without this the
        first two weeks of every fold would be silently untradeable."""

        self.entry = entry
        self.wk = weakness or resample(entry, tf_seconds(s.tf_weakness))
        self.dr = direction or resample(entry, tf_seconds(s.tf_direction))
        if len(self.entry) < 10 or len(self.wk) < 50 or len(self.dr) < 5:
            raise ValueError(
                f"not enough data: entry={len(self.entry)} "
                f"weakness={len(self.wk)} direction={len(self.dr)}"
            )

        self.dirdet = DirectionDetector(self.dr, s, self.bias)
        self.weakdet = WeaknessDetector(self.wk, s, self.bias)

        self.equity = self.risk.initial_capital
        self.peak_equity = self.equity
        self.open: list[Trade] = []
        self.trades: list[Trade] = []
        self.setups: list[Setup] = []
        self.dirs: list[DirectionSignal] = []

        self._d_ptr = 0
        self._w_ptr = 0
        self._tid = 0
        self._day = None
        self._day_start_equity = self.equity
        self._day_blocked = False
        self._streak_losses = 0
        self._cooldown_until = 0
        self._ruined = False
        self._equity_daily: list[tuple[int, float]] = []
        self._equity_by_trade: list[tuple[int, float]] = []
        self._signals = 0
        self._setups = 0
        self.rejects: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _reject(self, why: str) -> None:
        self.rejects[why] = self.rejects.get(why, 0) + 1

    def _cost_per_side(self) -> float:
        return 0.0 if self.bias.zero_costs else self.costs.per_side()

    def _stop_extra(self) -> float:
        return 0.0 if self.bias.zero_costs else self.costs.stop_slippage

    def _commission(self, lots: float) -> float:
        if self.bias.zero_costs:
            return 0.0
        return lots * self.costs.commission_per_lot

    def _in_session(self, ts: int) -> bool:
        if not self.s.sessions:
            return True
        hour = (ts % 86400) // 3600
        return any(start <= hour < end for start, end in self.s.sessions)

    def _round_lots(self, lots: float) -> float:
        step = self.costs.lot_step
        if step <= 0:
            return lots
        return math.floor(lots / step + 1e-9) * step

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def run(self) -> BacktestResult:
        tf = self.entry.tf
        bars = self.entry.bars

        for i, bar in enumerate(bars):
            now = bar.ts + tf                      # this bar has just closed
            self._roll_day(now)
            self._admit_directions(now)
            self._admit_weakness(now)
            self._manage_open(i, bar, now)
            self._scan_setups(i, bar, now)
            if i % 2048 == 0:
                self._prune(now)

        # Close whatever is still open at the last available price. No invented
        # outcomes (AUDIT #3).
        if bars:
            last = bars[-1]
            last_ts = last.ts + tf
            for pos in list(self.open):
                self._close(pos, last_ts, last.close, "data_end", 1.0)

        res = BacktestResult(
            trades=self.trades,
            equity_daily=self._equity_daily,
            equity_by_trade=self._equity_by_trade,
            initial_capital=self.risk.initial_capital,
            final_capital=self.equity,
            start_ts=bars[0].ts if bars else 0,
            end_ts=(bars[-1].ts + tf) if bars else 0,
            setups_created=self._setups,
            signals_created=self._signals,
            rejects=self.rejects,
            ruined=self._ruined,
            config=self.cfg.to_dict(),
            label=self.label,
        )
        return res

    def _roll_day(self, now: int) -> None:
        day = now // 86400
        if self._day is None:
            self._day = day
            self._day_start_equity = self.equity
            self._equity_daily.append((now, self.equity))
            return
        if day != self._day:
            self._day = day
            self._day_start_equity = self.equity
            self._day_blocked = False
            self._equity_daily.append((now, self.equity))

    def _prune(self, now: int) -> None:
        self.setups = [s for s in self.setups if not s.dead]
        self.dirs = [d for d in self.dirs if d.expires_ts >= now]

    # --- step 1 admission ---------------------------------------------- #

    def _admit_directions(self, now: int) -> None:
        bars = self.dr.bars
        while self._d_ptr < len(bars):
            b = bars[self._d_ptr]
            # Honest: a 4H bar is knowable at ts + 4h. With the bias flag on, it
            # is treated as knowable at ts, which is what v1 did.
            reveal = b.ts if self.bias.lookahead_direction else b.ts + self.dr.tf
            if reveal > now:
                break
            idx = self._d_ptr
            self._d_ptr += 1
            sig = self.dirdet.detect(idx)
            if sig is None:
                continue
            self._signals += 1
            self.dirs.append(sig)
            if self.s.cancel_on_opposing_direction:
                for st in self.setups:
                    if not st.dead and st.direction != sig.direction:
                        st.kill("opposing_direction")
                        self._reject("setup_opposing_direction")
            if len(self.dirs) > self.s.max_active_directions:
                self.dirs = self.dirs[-self.s.max_active_directions :]

    # --- step 2 admission ---------------------------------------------- #

    def _admit_weakness(self, now: int) -> None:
        bars = self.wk.bars
        while self._w_ptr < len(bars) and bars[self._w_ptr].ts + self.wk.tf <= now:
            i = self._w_ptr
            self._w_ptr += 1
            bar_close = bars[i].ts + self.wk.tf
            self.weakdet.advance(i)
            for dsig in self.dirs:
                if not dsig.active_at(bar_close):
                    continue
                if dsig.setups_spawned >= self.s.max_setups_per_direction:
                    continue
                setup = self.weakdet.detect(i, dsig)
                if setup is None:
                    continue
                dsig.setups_spawned += 1
                self._setups += 1
                self.setups.append(setup)

    # --- step 3: manage open positions --------------------------------- #

    def _manage_open(self, i: int, bar: Bar, now: int) -> None:
        for pos in list(self.open):
            if pos.entry_bar_idx >= i:
                continue  # never resolve a position on its own entry bar
            pos.bars_held += 1
            long = pos.direction == "long"

            if self.bias.optimistic_intrabar:
                # v1 raised the trail using this bar's own extreme and then tested
                # this bar's opposite extreme against the raised stop.
                self._update_trail(pos, bar)

            # excursions (mid prices)
            if long:
                pos.best_price = max(pos.best_price, bar.high)
                pos.worst_price = min(pos.worst_price, bar.low)
                pos.mfe_r = max(pos.mfe_r, (pos.best_price - pos.entry_price) / pos.risk_per_unit)
                pos.mae_r = max(pos.mae_r, (pos.entry_price - pos.worst_price) / pos.risk_per_unit)
            else:
                pos.best_price = min(pos.best_price, bar.low)
                pos.worst_price = max(pos.worst_price, bar.high)
                pos.mfe_r = max(pos.mfe_r, (pos.entry_price - pos.best_price) / pos.risk_per_unit)
                pos.mae_r = max(pos.mae_r, (pos.worst_price - pos.entry_price) / pos.risk_per_unit)

            tgt = None if (self.s.disable_tp_when_trailing and pos.trail_active) else pos.target

            # 1) gaps: an open beyond a level fills at the open, not the level
            if long and bar.open <= pos.stop:
                self._close(pos, now, bar.open, "gap_stop", 1.0, stop_fill=True)
                continue
            if not long and bar.open >= pos.stop:
                self._close(pos, now, bar.open, "gap_stop", 1.0, stop_fill=True)
                continue
            if tgt is not None:
                if long and bar.open >= tgt:
                    self._close(pos, now, bar.open, "gap_target", 1.0)
                    continue
                if not long and bar.open <= tgt:
                    self._close(pos, now, bar.open, "gap_target", 1.0)
                    continue

            hit_stop = bar.low <= pos.stop if long else bar.high >= pos.stop
            hit_tgt = tgt is not None and (bar.high >= tgt if long else bar.low <= tgt)
            plevel = pos.partial_level if not pos.partial_done else None
            hit_partial = plevel is not None and (
                bar.high >= plevel if long else bar.low <= plevel
            )

            if hit_stop and not self.bias.optimistic_intrabar:
                # Conservative resolution of an ambiguous bar (AUDIT #5).
                self._close(pos, now, pos.stop, "stop", 1.0, stop_fill=True)
                continue

            if hit_partial:
                self._close(pos, now, plevel, "partial", self.s.partial_frac)
                pos.partial_done = True
                if self.s.move_to_breakeven_after_partial:
                    be = pos.entry_price
                    pos.stop = max(pos.stop, be) if long else min(pos.stop, be)
                if pos.lots_open <= 1e-9:
                    continue

            if hit_tgt:
                self._close(pos, now, tgt, "target", 1.0)
                continue

            if hit_stop:  # only reachable in optimistic mode
                self._close(pos, now, pos.stop, "stop", 1.0, stop_fill=True)
                continue

            # Wall clock, not bar count: weekends and session breaks mean the two
            # differ, and a live position ages in wall-clock time.
            if now - pos.entry_ts >= self.s.max_hold_min * 60:
                self._close(pos, now, bar.close, "time_stop", 1.0)
                continue

            if not self.bias.optimistic_intrabar:
                # Honest: the trail moves at this bar's close and binds from the
                # next bar onward.
                self._update_trail(pos, bar)

    def _update_trail(self, pos: Trade, bar: Bar) -> None:
        if not self.s.use_trail:
            return
        long = pos.direction == "long"
        best = max(pos.best_price, bar.high) if long else min(pos.best_price, bar.low)
        pos.best_price = best
        profit_r = (
            (best - pos.entry_price) if long else (pos.entry_price - best)
        ) / pos.risk_per_unit
        if profit_r < self.s.trail_start_r:
            return
        pos.trail_active = True
        dist = (
            self.s.trail_atr * pos.atr
            if self.s.trail_atr > 0
            else self.s.trail_r * pos.risk_per_unit
        )
        if dist <= 0:
            return
        new_stop = best - dist if long else best + dist
        pos.stop = max(pos.stop, new_stop) if long else min(pos.stop, new_stop)

    # --- step 3: setups -> entries ------------------------------------- #

    def _scan_setups(self, i: int, bar: Bar, now: int) -> None:
        for st in self.setups:
            if st.dead or st.known_ts > now:
                continue
            if now > st.expires_ts:
                st.kill("expired")
                self._reject("setup_expired")
                continue

            long = st.direction == "long"

            if self.s.invalidate_on_rect_breach and not self.bias.skip_invalidation:
                # The far side of the rectangle is the future stop. If price has
                # been there, the setup no longer exists. v1 traded it anyway,
                # which is where a large slice of the 775% came from.
                breached = (
                    bar.low <= st.invalidation_price
                    if long
                    else bar.high >= st.invalidation_price
                )
                if breached:
                    st.kill("invalidated")
                    self._reject("setup_invalidated")
                    continue

            buf = self.s.entry_buffer_atr * st.atr
            triggered = (
                bar.close > st.rect_top + buf if long else bar.close < st.rect_bottom - buf
            )
            if not triggered:
                continue

            if self._open_position(st, i, bar, now):
                st.kill("filled")
            # a rejected trigger still consumes the setup: the breakout happened
            else:
                st.kill("trigger_rejected")

    def _blocked_reason(self, direction: str) -> str | None:
        if self._ruined:
            return "ruined"
        if self.equity <= self.risk.initial_capital * self.risk.ruin_floor_frac:
            self._ruined = True
            return "ruined"
        if self._day_blocked:
            return "daily_loss_cap"
        if len(self.open) >= self.risk.max_concurrent:
            return "max_concurrent"
        same = sum(1 for p in self.open if p.direction == direction)
        if same >= self.risk.max_concurrent_same_direction:
            return "max_concurrent_same_direction"
        return None

    def _open_position(self, st: Setup, i: int, bar: Bar, now: int) -> bool:
        if self.trade_from_ts is not None and now < self.trade_from_ts:
            self._reject("warmup")
            return False
        if not self._in_session(now):
            self._reject("out_of_session")
            return False
        if self._cooldown_until and now < self._cooldown_until:
            self._reject("cooldown")
            return False
        blocked = self._blocked_reason(st.direction)
        if blocked:
            self._reject(blocked)
            return False

        long = st.direction == "long"
        cost = self._cost_per_side()
        signal_price = bar.close
        fill = signal_price + cost if long else signal_price - cost

        buf = self.s.sl_buffer_atr * st.atr
        stop = st.rect_bottom - buf if long else st.rect_top + buf
        risk_per_unit = (fill - stop) if long else (stop - fill)
        if risk_per_unit <= 0:
            self._reject("non_positive_risk")
            return False
        # A breakout that has already run far from the rectangle makes the stop
        # so wide that the setup's geometry no longer applies.
        if risk_per_unit > (st.rect_top - st.rect_bottom) + buf + 2.0 * st.atr:
            self._reject("stop_too_wide")
            return False

        equity_base = self.equity if self.risk.compounding else self.risk.initial_capital
        risk_frac = min(self.risk.risk_frac, self.risk.max_risk_frac)
        risk_amount = equity_base * risk_frac

        lots = self._round_lots(risk_amount / (risk_per_unit * self.costs.contract_size))
        lots = min(lots, self.costs.max_lot)
        max_lots_by_leverage = (
            equity_base * self.risk.max_leverage / (fill * self.costs.contract_size)
        )
        lots = min(lots, self._round_lots(max_lots_by_leverage))
        if lots < self.costs.min_lot - 1e-12:
            self._reject("below_min_lot")
            return False

        risk_dollars = lots * self.costs.contract_size * risk_per_unit
        target = None
        if self.s.tp_r > 0:
            target = fill + self.s.tp_r * risk_per_unit if long else fill - self.s.tp_r * risk_per_unit
        partial_level = None
        if self.s.use_partial and self.s.partial_at_r > 0:
            partial_level = (
                fill + self.s.partial_at_r * risk_per_unit
                if long
                else fill - self.s.partial_at_r * risk_per_unit
            )

        self._tid += 1
        entry_comm = self._commission(lots)
        pos = Trade(
            tid=self._tid,
            direction=st.direction,
            entry_ts=now,
            entry_price=fill,
            signal_price=signal_price,
            initial_stop=stop,
            stop=stop,
            target=target,
            partial_level=partial_level,
            lots_initial=lots,
            lots_open=lots,
            risk_per_unit=risk_per_unit,
            risk_dollars=risk_dollars,
            atr=st.atr,
            entry_bar_idx=i,
            dir_bar_ts=st.dir_bar_ts,
            sweep_depth_atr=st.sweep_depth_atr,
            equity_at_entry=self.equity,
            best_price=fill,
            worst_price=fill,
            commission_paid=entry_comm,
            pnl=-entry_comm,
        )
        self.equity -= entry_comm
        self.open.append(pos)
        return True

    # --- closing ------------------------------------------------------- #

    def _close(
        self,
        pos: Trade,
        ts: int,
        level: float,
        reason: str,
        frac: float,
        stop_fill: bool = False,
    ) -> None:
        long = pos.direction == "long"
        cost = self._cost_per_side() + (self._stop_extra() if stop_fill else 0.0)
        fill = level - cost if long else level + cost

        lots = pos.lots_open if frac >= 1.0 else min(pos.lots_open, pos.lots_initial * frac)
        lots = max(lots, 0.0)
        if lots <= 0:
            return
        units = lots * self.costs.contract_size
        gross = (fill - pos.entry_price) * units if long else (pos.entry_price - fill) * units
        comm = self._commission(lots)
        pos.pnl += gross - comm
        pos.commission_paid += comm
        pos.lots_open -= lots
        self.equity += gross - comm
        if self.equity < 0:
            self.equity = 0.0
        self.peak_equity = max(self.peak_equity, self.equity)

        if pos.lots_open > 1e-9 and frac < 1.0:
            return  # partial: position stays open

        pos.exit_ts = ts
        pos.exit_price = fill
        pos.reason = reason if not pos.partial_done else f"{reason}+partial"
        self.trades.append(pos)
        if pos in self.open:
            self.open.remove(pos)
        self._equity_by_trade.append((ts, self.equity))

        # risk governors
        if pos.pnl < 0:
            self._streak_losses += 1
            if self._streak_losses >= self.risk.max_consecutive_losses:
                self._cooldown_until = ts + int(self.risk.cooldown_hours_after_streak * 3600)
                self._streak_losses = 0
                self._reject("streak_cooldown_triggered")
        else:
            self._streak_losses = 0

        if self._day_start_equity > 0:
            day_dd = (self.equity - self._day_start_equity) / self._day_start_equity
            if day_dd <= -self.risk.daily_loss_cap_frac:
                self._day_blocked = True
        if self.equity <= self.risk.initial_capital * self.risk.ruin_floor_frac:
            self._ruined = True


def run_backtest(
    entry: Series,
    cfg: RunConfig,
    label: str = "",
    weakness: Series | None = None,
    direction: Series | None = None,
    trade_from_ts: int | None = None,
) -> BacktestResult:
    bt = Backtester(
        entry,
        cfg,
        weakness=weakness,
        direction=direction,
        label=label,
        trade_from_ts=trade_from_ts,
    )
    res = bt.run()
    if trade_from_ts is not None:
        res.start_ts = max(res.start_ts, trade_from_ts)
    return res
