"""
Invariant tests for the engine. No third-party dependencies, so this runs anywhere.

These are not unit tests of arithmetic. Each one asserts a property that, if it
broke, would let the backtest report profit it could not have earned. The most
important is `test_null_hypothesis`: on a driftless random walk the engine must
NOT find an edge. v1 would have, because its look-ahead does not care whether the
data has structure.

    python -m cct2.run selftest
"""

from __future__ import annotations

import time
from dataclasses import replace

from .bars import Bar, Series, atr, ema, resample, swings, tf_seconds
from .data import synthetic
from .engine import run_backtest
from .metrics import compute
from .settings import BiasFlags, RiskConfig, RunConfig, StrategyConfig, gold_costs

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSES
    if condition:
        _PASSES += 1
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def _honest_cfg(**strategy_kw) -> RunConfig:
    return RunConfig(
        strategy=StrategyConfig(**strategy_kw),
        costs=gold_costs(),
        risk=RiskConfig(),
        bias=BiasFlags(),
    )


# --------------------------------------------------------------------------- #


def test_resample():
    bars = [
        Bar(0, 10, 12, 9, 11),
        Bar(60, 11, 15, 10, 14),
        Bar(120, 14, 14, 8, 9),
        Bar(180, 9, 11, 7, 10),      # completes the 4-minute bucket 0..240
        Bar(240, 10, 10, 10, 10),    # starts an incomplete bucket
    ]
    src = Series(60, bars)
    out = resample(src, 240)
    check("resample drops the incomplete trailing bucket", len(out) == 1, f"got {len(out)}")
    b = out[0]
    check(
        "resample aggregates OHLC correctly",
        (b.ts, b.open, b.high, b.low, b.close) == (0, 10, 15, 7, 10),
        f"got {(b.ts, b.open, b.high, b.low, b.close)}",
    )

    s1m = synthetic(days=6, mode="martingale", seed=3)
    s15 = resample(s1m, 900)
    ok = True
    for hb in s15.bars[:50]:
        members = [x for x in s1m.bars if hb.ts <= x.ts < hb.ts + 900]
        if not members:
            continue
        if (
            abs(hb.open - members[0].open) > 1e-9
            or abs(hb.close - members[-1].close) > 1e-9
            or abs(hb.high - max(m.high for m in members)) > 1e-9
            or abs(hb.low - min(m.low for m in members)) > 1e-9
        ):
            ok = False
            break
    check("resample 1m->15m matches members", ok)


def test_swing_confirmation_lag():
    bars = [Bar(i * 60, 10, 10 + (1 if i == 5 else 0), 9, 10) for i in range(12)]
    sw = swings(bars, 3)
    highs = [s for s in sw if s.kind == "high"]
    check("swing high located at the pivot bar", len(highs) == 1 and highs[0].idx == 5,
          f"got {[(s.idx, s.kind) for s in sw]}")
    check(
        "swing cannot be known until k bars later",
        highs and highs[0].known_idx == 8,
        f"known_idx={highs[0].known_idx if highs else None}",
    )


def test_indicator_causality():
    vals = [float(i) for i in range(100)]
    e = ema(vals, 10)
    e_trunc = ema(vals[:60], 10)
    check(
        "EMA at index i does not depend on data after i",
        all(
            (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12)
            for a, b in zip(e[:60], e_trunc)
        ),
    )
    s = synthetic(days=5, mode="martingale", seed=11)
    a_full = atr(s.bars, 14)
    a_trunc = atr(s.bars[:500], 14)
    check(
        "ATR at index i does not depend on data after i",
        all(
            (x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-9)
            for x, y in zip(a_full[:500], a_trunc)
        ),
    )


def test_no_lookahead_in_trades():
    s = synthetic(days=70, mode="regime", seed=5)
    cfg = _honest_cfg()
    res = run_backtest(s, cfg, label="causality")
    dir_tf = tf_seconds(cfg.strategy.tf_direction)
    bad = [t for t in res.trades if t.entry_ts < t.dir_bar_ts + dir_tf]
    check(
        "every entry postdates the close of its 4H direction bar",
        not bad,
        f"{len(bad)} of {len(res.trades)} trades violate it",
    )

    biased = RunConfig(
        strategy=cfg.strategy, costs=cfg.costs, risk=cfg.risk,
        bias=BiasFlags(lookahead_direction=True),
    )
    res_b = run_backtest(s, biased, label="lookahead")
    bad_b = [t for t in res_b.trades if t.entry_ts < t.dir_bar_ts + dir_tf]
    check(
        "the lookahead flag actually reintroduces the defect (audit is meaningful)",
        len(bad_b) > 0,
        f"flag produced {len(bad_b)} violations out of {len(res_b.trades)}",
    )
    return s, res


def test_trade_integrity(s: Series):
    cfg = _honest_cfg()
    res = run_backtest(s, cfg, label="integrity")
    t_bad_time = [t for t in res.trades if not t.exit_ts or t.exit_ts <= t.entry_ts]
    check("exits strictly follow entries", not t_bad_time, f"{len(t_bad_time)} bad")

    step = cfg.costs.lot_step
    bad_lots = [t for t in res.trades if abs(round(t.lots_initial / step) * step - t.lots_initial) > 1e-9]
    check("lot sizes respect the broker step", not bad_lots, f"{len(bad_lots)} bad")

    hold_cap = cfg.strategy.max_hold_min * 60 + s.tf
    over = [t for t in res.trades if (t.exit_ts - t.entry_ts) > hold_cap and t.reason != "data_end"]
    check("time stop is honoured", not over, f"{len(over)} trades held too long")

    # equity conservation
    total = sum(t.pnl for t in res.trades)
    expect = res.initial_capital + total
    check(
        "equity == initial + sum(trade P&L)",
        abs(expect - res.final_capital) < 0.01,
        f"{expect:.2f} vs {res.final_capital:.2f}",
    )

    # chronological ledger (AUDIT #4)
    ts_list = [ts for ts, _ in res.equity_by_trade]
    check(
        "equity ledger is chronological",
        all(a <= b for a, b in zip(ts_list, ts_list[1:])),
        "out-of-order equity updates found",
    )

    # concurrency
    limit = cfg.risk.max_concurrent
    events = []
    for t in res.trades:
        events.append((t.entry_ts, 1))
        events.append((t.exit_ts, -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    check(f"max concurrent positions <= {limit}", peak <= limit, f"peak was {peak}")

    stops = [t for t in res.trades if t.reason == "stop"]
    bad_r = [t for t in stops if not (-2.2 < t.r < -0.85)]
    check(
        "a pure stop-out loses about 1R",
        not bad_r,
        f"{len(bad_r)}/{len(stops)} outside [-2.2, -0.85] R",
    )
    return res


def test_costs_are_charged(s: Series):
    cfg = _honest_cfg()
    res = run_backtest(s, cfg, label="costs")
    longs = [t for t in res.trades if t.direction == "long"]
    shorts = [t for t in res.trades if t.direction == "short"]
    check(
        "long entries fill above the signal close",
        all(t.entry_price > t.signal_price for t in longs),
        f"{sum(1 for t in longs if t.entry_price <= t.signal_price)} bad of {len(longs)}",
    )
    check(
        "short entries fill below the signal close",
        all(t.entry_price < t.signal_price for t in shorts),
        f"{sum(1 for t in shorts if t.entry_price >= t.signal_price)} bad of {len(shorts)}",
    )
    check(
        "commission is actually deducted",
        all(t.commission_paid > 0 for t in res.trades) if res.trades else False,
        "some trades paid no commission",
    )

    free = RunConfig(
        strategy=cfg.strategy, costs=cfg.costs, risk=cfg.risk,
        bias=BiasFlags(zero_costs=True),
    )
    res_free = run_backtest(s, free, label="free")
    m, mf = compute(res, bootstrap=False), compute(res_free, bootstrap=False)
    check(
        "removing costs improves the result (they were binding)",
        mf.expectancy_r > m.expectancy_r,
        f"{mf.expectancy_r:+.3f} vs {m.expectancy_r:+.3f}",
    )


def test_invalidation_binds(s: Series):
    cfg = _honest_cfg()
    honest = compute(run_backtest(s, cfg, label="inv-on"), bootstrap=False)
    skip = RunConfig(
        strategy=cfg.strategy, costs=cfg.costs, risk=cfg.risk,
        bias=BiasFlags(skip_invalidation=True),
    )
    loose = compute(run_backtest(s, skip, label="inv-off"), bootstrap=False)
    check(
        "skipping invalidation admits more trades",
        loose.trades > honest.trades,
        f"{loose.trades} vs {honest.trades}",
    )
    check(
        "invalidation is not free (it changes measured expectancy)",
        abs(loose.expectancy_r - honest.expectancy_r) > 1e-9,
        "identical expectancy, flag may be inert",
    )


def test_risk_governors(s: Series):
    cfg = RunConfig(
        strategy=StrategyConfig(),
        costs=gold_costs(),
        risk=RiskConfig(risk_frac=0.02, max_risk_frac=0.02, daily_loss_cap_frac=0.005),
        bias=BiasFlags(),
    )
    res = run_backtest(s, cfg, label="governors")
    check(
        "daily loss cap engages",
        res.rejects.get("daily_loss_cap", 0) > 0,
        f"rejects={res.rejects}",
    )

    # v1's 25% bet on a sequence with no edge. This is the ruin test: identical
    # signals, identical trade order, only the stake changes.
    noise = synthetic(days=240, mode="martingale", seed=77)

    def dd_at(frac: float, max_leverage: float = 1e9):
        c = RunConfig(
            strategy=StrategyConfig(),
            costs=gold_costs(),
            risk=RiskConfig(
                risk_frac=frac,
                max_risk_frac=frac,
                daily_loss_cap_frac=1.0,
                max_consecutive_losses=99,
                max_leverage=max_leverage,
                ruin_floor_frac=0.02,
            ),
            bias=BiasFlags(),
        )
        r = run_backtest(noise, c, label=f"risk {frac}")
        return compute(r, bootstrap=False), r

    small_m, _ = dd_at(0.0075)
    big_m, big_res = dd_at(0.25)
    print(
        f"        same {small_m.trades} trades, no leverage cap: "
        f"0.75% risk -> {small_m.max_dd_pct:.1f}% DD, "
        f"25% risk -> {big_m.max_dd_pct:.1f}% DD"
    )
    check(
        "v1's 25% stake turns a no-edge sequence into a catastrophic drawdown",
        big_m.max_dd_pct > 50.0 or big_res.ruined,
        f"maxDD {big_m.max_dd_pct:.1f}%, ruined={big_res.ruined}, n={big_m.trades}",
    )
    check(
        "drawdown grows far faster than the stake",
        # Drawdown saturates at 100%, so the ratio cannot keep scaling; either it
        # is 8x the small-stake figure or it has already pinned the account.
        big_m.max_dd_pct > min(90.0, 8 * small_m.max_dd_pct),
        f"{big_m.max_dd_pct:.1f}% vs {small_m.max_dd_pct:.1f}% at 1/33rd the stake",
    )

    capped_m, _ = dd_at(0.25, max_leverage=20.0)
    print(
        f"        with a 20x notional cap the same 25% request is clipped to "
        f"{capped_m.max_dd_pct:.1f}% DD"
    )
    check(
        "the notional cap materially limits an over-sized request",
        capped_m.max_dd_pct < big_m.max_dd_pct,
        f"capped {capped_m.max_dd_pct:.1f}% vs uncapped {big_m.max_dd_pct:.1f}%",
    )


def test_determinism(s: Series):
    cfg = _honest_cfg()
    a = run_backtest(s, cfg, label="det-a")
    b = run_backtest(s, cfg, label="det-b")
    check(
        "runs are reproducible",
        abs(a.final_capital - b.final_capital) < 1e-9 and len(a.trades) == len(b.trades),
        f"{a.final_capital} vs {b.final_capital}",
    )


def test_null_hypothesis():
    """
    The one that matters. Driftless geometric random walk, so there is no
    path-dependent edge to find; after spread and commission the honest engine must
    come out at or below zero. A backtester that reports a positive, statistically
    significant expectancy here is measuring itself.
    """
    from .metrics import _bootstrap_mean_ci

    worst_ci = None
    pooled: list[float] = []
    for seed in (101, 202, 303, 404, 505, 606):
        s = synthetic(days=150, mode="martingale", seed=seed)
        cfg = _honest_cfg()
        res = run_backtest(s, cfg, label=f"null-{seed}")
        m = compute(res)
        pooled.extend(t.r for t in res.trades)
        print(f"        seed {seed}: n={m.trades:<4} E={m.expectancy_r:+.3f}R  "
              f"CI_low={m.ci_low_r:+.3f}")
        if m.trades >= 20:
            worst_ci = m.ci_low_r if worst_ci is None else max(worst_ci, m.ci_low_r)

    lo, hi, p = _bootstrap_mean_ci(pooled)
    mean = sum(pooled) / len(pooled) if pooled else 0.0
    print(
        f"        pooled: n={len(pooled)} E={mean:+.3f}R  CI=[{lo:+.3f},{hi:+.3f}]  "
        f"P(<=0)={p:.2f}"
    )
    check(
        "no individual seed shows a significant edge in noise",
        worst_ci is None or worst_ci <= 0.0,
        f"a seed produced CI lower bound {worst_ci:+.3f} R > 0",
    )
    check(
        "pooled expectancy on noise is not distinguishable from zero",
        lo <= 0.0,
        f"pooled CI [{lo:+.3f},{hi:+.3f}] excludes zero",
    )
    check(
        "pooled expectancy on noise is not materially positive",
        mean <= 0.05,
        f"pooled mean {mean:+.3f} R",
    )


def main() -> int:
    t0 = time.time()
    print("\ncct2 self-test\n" + "=" * 78)
    print("\n[bars]")
    test_resample()
    test_swing_confirmation_lag()
    test_indicator_causality()
    print("\n[causality]")
    s, _ = test_no_lookahead_in_trades()
    print("\n[accounting]")
    test_trade_integrity(s)
    print("\n[costs]")
    test_costs_are_charged(s)
    print("\n[setup invalidation]")
    test_invalidation_binds(s)
    print("\n[risk governors]")
    test_risk_governors(s)
    print("\n[determinism]")
    test_determinism(s)
    print("\n[null hypothesis: no edge in noise]")
    test_null_hypothesis()

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"{_PASSES} passed, {len(_FAILURES)} FAILED  ({time.time() - t0:.1f}s)")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"all {_PASSES} checks passed  ({time.time() - t0:.1f}s)")
    return 0
