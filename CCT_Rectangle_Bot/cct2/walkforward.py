"""
Walk-forward parameter selection.

v1 tuned parameters against the whole sample and reported that same sample's
return, then relaxed the parameters in a retry loop until trades appeared. Any
number produced that way is an in-sample number.

Here: parameters are chosen on a training window and then applied, untouched, to
the next window, which the selection never saw. Only the out-of-sample windows are
reported. Each fold gets a warm-up prefix so that indicators are primed and no
fold is quietly untradeable for its first fortnight.

The selection score deliberately is not "highest return":

    score = expectancy_R * sqrt(n)     with a hard minimum trade count

That is the t-statistic up to a constant, so it prefers a small edge measured
many times over a large edge measured five times. Return-maximising selection on
short windows picks whichever parameter set caught the biggest trend, which is the
thing that does not repeat.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

from .bars import Series, fmt_ts
from .engine import BacktestResult, Trade, run_backtest
from .metrics import Metrics, compute
from .settings import RunConfig

# Kept small on purpose. Every extra axis multiplies the number of chances the
# search gets to fit noise; the walk-forward split limits the damage but does not
# eliminate it.
DEFAULT_GRID: dict[str, list] = {
    # This one governs how many signals exist at all (0.4 removes ~83% of them on
    # gold-like bars), so it is the first thing the search should be allowed to
    # decide rather than something to fix by hand.
    "min_direction_body_atr": [0.0, 0.15, 0.40],
    "sweep_min_atr": [0.10, 0.25],
    "min_close_position": [0.50, 0.65],
    "tp_r": [2.0, 3.0],
    "trail_start_r": [1.5, 2.5],
}

QUICK_GRID: dict[str, list] = {
    "sweep_min_atr": [0.15, 0.30],
    "tp_r": [2.0, 3.0],
}


@dataclass
class Candidate:
    params: dict
    score: float
    metrics: Metrics

    def __str__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (
            f"score {self.score:+.2f}  n={self.metrics.trades:<4} "
            f"E={self.metrics.expectancy_r:+.3f}R  PF={self.metrics.profit_factor:.2f}  "
            f"DD={self.metrics.max_dd_pct:.1f}%  [{p}]"
        )


@dataclass
class Fold:
    index: int
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int
    chosen: dict = field(default_factory=dict)
    train_metrics: Metrics | None = None
    test_metrics: Metrics | None = None
    test_result: BacktestResult | None = None
    candidates_evaluated: int = 0


def score_of(m: Metrics, min_trades: int, max_dd_pct: float) -> float:
    if m.trades < min_trades:
        return float("-inf")
    if m.max_dd_pct > max_dd_pct:
        return float("-inf")
    if m.ruined:
        return float("-inf")
    return m.expectancy_r * math.sqrt(m.trades)


def grid_search(
    series: Series,
    base: RunConfig,
    grid: dict[str, list],
    trade_from_ts: int | None = None,
    min_trades: int = 12,
    max_dd_pct: float = 40.0,
    progress: bool = False,
) -> list[Candidate]:
    keys = list(grid)
    out: list[Candidate] = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    for n, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        cfg = RunConfig(
            strategy=base.strategy.with_(**params),
            costs=base.costs,
            risk=base.risk,
            bias=base.bias,
        )
        res = run_backtest(series, cfg, label="train", trade_from_ts=trade_from_ts)
        m = compute(res, bootstrap=False)
        out.append(Candidate(params, score_of(m, min_trades, max_dd_pct), m))
        if progress:
            print(f"    [{n}/{len(combos)}] {out[-1]}")
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def walk_forward(
    series: Series,
    base: RunConfig,
    folds: int = 4,
    grid: dict[str, list] | None = None,
    warmup_days: float = 15.0,
    min_train_days: float = 120.0,
    min_trades: int = 12,
    verbose: bool = True,
) -> tuple[list[Fold], BacktestResult]:
    """
    Anchored walk-forward: fold k trains on everything before it and is tested on
    its own window only. Equity is chained across the out-of-sample windows, so
    the final curve is what an account that re-tuned at each boundary would have
    experienced.
    """
    grid = grid or DEFAULT_GRID
    tf = series.tf
    warmup_bars = int(warmup_days * 86400 / tf)
    min_train_bars = int(min_train_days * 86400 / tf)
    n = len(series)

    # The first test window cannot start until the indicators are warm AND there is
    # a training period long enough to contain trades. Starting it right after
    # warm-up (the obvious implementation) means fold 1 selects parameters on a
    # window in which the strategy never traded.
    first_test = warmup_bars + min_train_bars
    usable = n - first_test
    if usable <= 0:
        raise ValueError(
            f"need more than {warmup_days + min_train_days:.0f} days of data to "
            f"walk forward ({warmup_days:.0f}d warm-up + {min_train_days:.0f}d "
            f"minimum training window); got {series.days():.1f} days. "
            f"Lower --warmup-days/--min-train-days, or get more history."
        )
    fold_size = usable // folds
    if fold_size < int(14 * 86400 / tf):
        raise ValueError(
            f"folds of {fold_size * tf / 86400:.1f} days are too short to mean "
            f"anything at 1-3 trades per month; use fewer folds or more data"
        )

    results: list[Fold] = []
    all_trades: list[Trade] = []
    equity_daily: list[tuple[int, float]] = []
    equity_by_trade: list[tuple[int, float]] = []
    equity = base.risk.initial_capital
    start_equity = equity
    oos_start_ts = None
    oos_end_ts = None
    total_setups = 0
    total_signals = 0
    rejects: dict[str, int] = {}

    for k in range(folds):
        test_lo = first_test + k * fold_size
        test_hi = first_test + (k + 1) * fold_size if k < folds - 1 else n
        if test_hi - test_lo < 10:
            continue

        train_lo = 0
        train_hi = test_lo
        train_series = Series(tf, series.bars[train_lo:train_hi])
        train_from_ts = series.bars[min(train_lo + warmup_bars, train_hi - 1)].ts

        if verbose:
            print(
                f"\n  fold {k + 1}/{folds}: "
                f"train {fmt_ts(train_series.bars[0].ts)} -> {fmt_ts(train_series.bars[-1].ts)} "
                f"({train_series.days():.0f}d), "
                f"test {fmt_ts(series.bars[test_lo].ts)} -> {fmt_ts(series.bars[test_hi - 1].ts)} "
                f"({(test_hi - test_lo) * tf / 86400:.0f}d)"
            )

        try:
            cands = grid_search(
                train_series,
                base,
                grid,
                trade_from_ts=train_from_ts,
                min_trades=min_trades,
            )
        except ValueError as exc:
            if verbose:
                print(f"    train window unusable: {exc}")
            continue

        viable = [c for c in cands if c.score > float("-inf")]
        if not viable:
            if verbose:
                best_n = max((c.metrics.trades for c in cands), default=0)
                print(
                    f"    no parameter set cleared the bar (min {min_trades} trades, "
                    f"best had {best_n}). Fold skipped rather than forced."
                )
            results.append(
                Fold(k + 1, train_series.bars[0].ts, train_series.bars[-1].ts,
                     series.bars[test_lo].ts, series.bars[test_hi - 1].ts,
                     candidates_evaluated=len(cands))
            )
            continue

        best = viable[0]
        if verbose:
            print(f"    chosen: {best}")
            for c in viable[1:3]:
                print(f"    runner: {c}")

        # Test window, with a warm-up prefix that produces no trades.
        prefix_lo = max(0, test_lo - warmup_bars)
        test_series = Series(tf, series.bars[prefix_lo:test_hi])
        test_from_ts = series.bars[test_lo].ts
        fold_cfg = RunConfig(
            strategy=base.strategy.with_(**best.params),
            costs=base.costs,
            risk=replace(base.risk, initial_capital=equity),
            bias=base.bias,
        )
        test_res = run_backtest(
            test_series, fold_cfg, label=f"OOS fold {k + 1}", trade_from_ts=test_from_ts
        )
        tm = compute(test_res)
        equity = test_res.final_capital
        all_trades.extend(test_res.trades)
        equity_daily.extend(test_res.equity_daily)
        equity_by_trade.extend(test_res.equity_by_trade)
        total_setups += test_res.setups_created
        total_signals += test_res.signals_created
        for kk, vv in test_res.rejects.items():
            rejects[kk] = rejects.get(kk, 0) + vv
        oos_start_ts = oos_start_ts or test_from_ts
        oos_end_ts = test_res.end_ts

        if verbose:
            print(
                f"    OOS: n={tm.trades:<4} E={tm.expectancy_r:+.3f}R  "
                f"win={tm.win_rate:.0f}%  PF={tm.profit_factor:.2f}  "
                f"ret={tm.total_return_pct:+.1f}%  DD={tm.max_dd_pct:.1f}%  "
                f"equity ${equity:,.0f}"
            )

        results.append(
            Fold(
                index=k + 1,
                train_start_ts=train_series.bars[0].ts,
                train_end_ts=train_series.bars[-1].ts,
                test_start_ts=test_from_ts,
                test_end_ts=test_res.end_ts,
                chosen=best.params,
                train_metrics=best.metrics,
                test_metrics=tm,
                test_result=test_res,
                candidates_evaluated=len(cands),
            )
        )

    combined = BacktestResult(
        trades=all_trades,
        equity_daily=equity_daily,
        equity_by_trade=equity_by_trade,
        initial_capital=start_equity,
        final_capital=equity,
        start_ts=oos_start_ts or series.bars[0].ts,
        end_ts=oos_end_ts or series.bars[-1].ts,
        setups_created=total_setups,
        signals_created=total_signals,
        rejects=rejects,
        config=base.to_dict(),
        label="WALK-FORWARD (out-of-sample only)",
    )
    return results, combined


def parameter_stability(folds: list[Fold]) -> str:
    """
    If the chosen parameters jump around between folds, the search is fitting
    noise and the OOS aggregate is optimistic even so. Worth printing.
    """
    picked = [f for f in folds if f.chosen]
    if not picked:
        return "  no fold produced a parameter choice"
    keys = sorted({k for f in picked for k in f.chosen})
    lines = ["  parameter choices by fold (stability check):"]
    for key in keys:
        vals = [f.chosen.get(key) for f in picked]
        distinct = len(set(map(str, vals)))
        flag = "  <-- unstable" if distinct == len(vals) and len(vals) > 2 else ""
        lines.append(f"    {key:<22}" + "  ".join(f"{v!s:>6}" for v in vals) + flag)
    return "\n".join(lines)
