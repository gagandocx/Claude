"""
Command line entry point.

    python -m cct2.run selftest
    python -m cct2.run audit      [--synthetic-days 180] [--csv FILE]
    python -m cct2.run backtest   --csv FILE [--risk 0.0075] [--log]
    python -m cct2.run walkforward --csv FILE [--folds 4] [--quick]
    python -m cct2.run null       [--seeds 5]
    python -m cct2.run fetch      --symbol GC=F --interval 1m --period 7d --out gold.csv

Data: `--csv` expects M1 (or any single timeframe) OHLC. Without it, commands fall
back to synthetic bars, which test the machinery and tell you nothing about gold.
Every command that runs on synthetic data says so in its output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from . import audit as audit_mod
from .bars import Series, load_csv, resample, tf_seconds, write_csv
from .data import fetch_yfinance, split, synthetic, train_test
from .engine import run_backtest
from .metrics import compute, summary, trade_log
from .settings import (
    BiasFlags,
    RiskConfig,
    RunConfig,
    StrategyConfig,
    gold_costs,
    v1_equivalent,
)
from .walkforward import (
    DEFAULT_GRID,
    QUICK_GRID,
    parameter_stability,
    walk_forward,
)

SYNTHETIC_WARNING = (
    "  NOTE: running on SYNTHETIC bars. This validates the engine, not the strategy.\n"
    "        Any profit shown here is a property of the generator. Use --csv with real\n"
    "        M1 history before believing a number."
)


def _load(args) -> tuple[Series, bool]:
    if getattr(args, "csv", None):
        series = load_csv(args.csv, args.tf)
        if len(series) < 500:
            raise SystemExit(f"{args.csv}: only {len(series)} usable bars")
        return series, False
    days = getattr(args, "synthetic_days", 180)
    mode = getattr(args, "synthetic_mode", "regime")
    seed = getattr(args, "seed", 7)
    return synthetic(days=days, mode=mode, seed=seed), True


def _base_cfg(args) -> RunConfig:
    risk = RiskConfig()
    if getattr(args, "risk", None) is not None:
        risk = replace(risk, risk_frac=args.risk, max_risk_frac=max(args.risk, risk.max_risk_frac))
    if getattr(args, "capital", None) is not None:
        risk = replace(risk, initial_capital=args.capital)
    costs = gold_costs()
    if getattr(args, "spread", None) is not None:
        costs = replace(costs, spread=args.spread)
    if getattr(args, "commission", None) is not None:
        costs = replace(costs, commission_per_lot=args.commission)
    return RunConfig(strategy=StrategyConfig(), costs=costs, risk=risk, bias=BiasFlags())


# --------------------------------------------------------------------------- #


def cmd_selftest(args) -> int:
    from .selftest import main as selftest_main

    return selftest_main()


def cmd_backtest(args) -> int:
    series, is_synth = _load(args)
    print(f"\n  data: {series.describe()}")
    if is_synth:
        print(SYNTHETIC_WARNING)
    cfg = _base_cfg(args)
    res = run_backtest(series, cfg, label=args.label or "cct2 honest backtest")
    m = compute(res)
    print()
    print(summary(m))
    if args.log:
        print(trade_log(res, limit=args.log_limit))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "metrics": {k: v for k, v in vars(m).items()},
                    "config": res.config,
                    "trades": [
                        {
                            "entry_ts": t.entry_ts,
                            "exit_ts": t.exit_ts,
                            "direction": t.direction,
                            "lots": t.lots_initial,
                            "entry": t.entry_price,
                            "stop": t.initial_stop,
                            "exit": t.exit_price,
                            "r": t.r,
                            "pnl": t.pnl,
                            "reason": t.reason,
                        }
                        for t in res.trades
                    ],
                },
                fh,
                indent=2,
                default=str,
            )
        print(f"\n  wrote {args.json}")
    if m.trades and not m.sample_is_meaningful:
        print(
            "\n  The sample does not support a claim of edge. That is a result, not a\n"
            "  failure: it means do not trade this configuration yet, and do not tune it\n"
            "  against this window."
        )
    return 0


def cmd_audit(args) -> int:
    series, is_synth = _load(args)
    print(f"\n  data: {series.describe()}")
    if is_synth:
        print(SYNTHETIC_WARNING)
    print(
        "\n  Each row runs the SAME strategy on the SAME bars with v1's parameters and a\n"
        "  fixed 0.5% stake. The only thing that changes is which of v1's measurement\n"
        "  defects is switched on, so every difference is pure artefact. Stake is held\n"
        "  small and constant here on purpose; it gets its own table below.\n"
    )
    rows = audit_mod.run_audit(series)
    print(audit_mod.format_table(rows))

    honest = rows[0][1]
    fully_biased = rows[-1][1]
    print(
        f"\n  v1 params measured honestly:  {honest.expectancy_r:+.3f} R/trade, "
        f"{honest.total_return_pct:+.1f}% total"
    )
    print(
        f"  same thing with all defects:  {fully_biased.expectancy_r:+.3f} R/trade, "
        f"{fully_biased.total_return_pct:+.1f}% total"
    )
    delta = fully_biased.expectancy_r - honest.expectancy_r
    print(f"  artefact:                     {delta:+.3f} R/trade of pure measurement error")

    print("\n\n  RISK LADDER (v2 parameters, honest measurement, notional cap lifted)")
    print(
        "  Sizing happens after the entry decision, so the stake does not pick the\n"
        "  trades. Only the bet size changes down the rows.\n"
    )
    cfg = _base_cfg(args)
    print(audit_mod.format_risk_table(audit_mod.risk_ladder(series, cfg)))
    print(
        "\n  v1 used 25%. Whatever the expectancy, that stake is a bet on the ORDER of\n"
        "  the wins, not on the edge."
    )
    return 0


def cmd_walkforward(args) -> int:
    series, is_synth = _load(args)
    print(f"\n  data: {series.describe()}")
    if is_synth:
        print(SYNTHETIC_WARNING)
    cfg = _base_cfg(args)
    grid = QUICK_GRID if args.quick else DEFAULT_GRID
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(
        f"\n  walk-forward: {args.folds} folds, {n_combos} parameter combinations per fold, "
        f"{args.warmup_days:.0f}d warm-up"
    )
    folds, combined = walk_forward(
        series,
        cfg,
        folds=args.folds,
        grid=grid,
        warmup_days=args.warmup_days,
        min_train_days=args.min_train_days,
        min_trades=args.min_trades,
    )
    print()
    print(parameter_stability(folds))
    m = compute(combined)
    print()
    print(summary(m))
    tested = [f for f in folds if f.test_metrics]
    if tested:
        pos = sum(1 for f in tested if f.test_metrics.expectancy_r > 0)
        print(
            f"  folds with positive out-of-sample expectancy: {pos}/{len(tested)}\n"
            "  A strategy that is genuinely stationary should win most folds, not one."
        )
    if args.log:
        print(trade_log(combined, limit=args.log_limit))
    return 0


def cmd_null(args) -> int:
    print(
        "\n  Null test: driftless random walks. There is no edge to find, so an honest\n"
        "  engine must report expectancy <= 0 after costs on every seed.\n"
    )
    cfg = _base_cfg(args)
    rows = []
    for seed in range(101, 101 + args.seeds):
        s = synthetic(days=args.synthetic_days, mode="martingale", seed=seed)
        res = run_backtest(s, cfg, label=f"null seed {seed}")
        m = compute(res)
        rows.append((seed, m))
        print(
            f"    seed {seed}: n={m.trades:<4} E={m.expectancy_r:+.3f}R  "
            f"CI=[{m.ci_low_r:+.3f},{m.ci_high_r:+.3f}]  "
            f"total={m.total_return_pct:+.1f}%  PF={m.profit_factor:.2f}"
        )
    bad = [s for s, m in rows if m.trades >= 20 and m.ci_low_r > 0]
    print()
    if bad:
        print(f"  FAILED: seeds {bad} show significant 'edge' in noise. The engine leaks.")
        return 1
    print("  PASSED: no significant edge found in noise on any seed.")
    return 0


def cmd_portfolio(args) -> int:
    """
    Run the same configuration across several instruments and pool the trades.

    This is the answer to the sample-size problem, not parameter loosening: 2
    trades a month on gold becomes 12 a month across six instruments, and the
    pooled expectancy is measured on observations that were never used to choose
    the parameters.
    """
    from .metrics import pool

    instruments = []
    if args.spec:
        with open(args.spec) as fh:
            spec = json.load(fh)
        instruments = spec["instruments"]
    elif args.csv:
        instruments = [{"csv": p} for p in args.csv]
    if len(instruments) < 2:
        raise SystemExit(
            "portfolio needs at least two instruments: repeat --csv, or pass --spec "
            "with a JSON file (see cct2/instruments.sample.json)"
        )

    base = _base_cfg(args)
    results = []
    rows = []
    for inst in instruments:
        path = inst["csv"]
        name = inst.get("name") or path.rsplit("/", 1)[-1]
        series = load_csv(path, inst.get("tf", args.tf))
        if len(series) < 5000:
            print(f"  skipping {name}: only {len(series)} bars")
            continue
        # Per-instrument costs. A single spread figure cannot be right for gold
        # and for EURUSD at the same time, and cost realism dominates the result
        # at this trade size.
        costs = replace(
            base.costs,
            **{
                k: inst[k]
                for k in (
                    "spread",
                    "slippage",
                    "stop_slippage",
                    "commission_per_lot",
                    "contract_size",
                    "lot_step",
                    "min_lot",
                )
                if k in inst
            },
        )
        cfg = RunConfig(strategy=base.strategy, costs=costs, risk=base.risk, bias=base.bias)
        res = run_backtest(series, cfg, label=name)
        results.append(res)
        rows.append((name, compute(res, bootstrap=False)))
        print(f"  {name:<20} {series.describe()}   spread={costs.spread} "
              f"contract={costs.contract_size:g}")
    if not results:
        raise SystemExit("no usable inputs")
    cfg = base

    print("\n  PER INSTRUMENT")
    print(audit_mod.format_table(rows))

    pooled = pool(results, risk_frac=cfg.risk.risk_frac, initial=cfg.risk.initial_capital,
                  label="POOLED across instruments")
    m = compute(pooled)
    print()
    print(summary(m))
    print(
        "  Pooling assumes no cross-instrument position limit and equal risk per\n"
        "  trade, so the drawdown above is optimistic for correlated instruments.\n"
        "  XAUUSD and XAGUSD are not two independent samples."
    )
    return 0


def cmd_fetch(args) -> int:
    series = fetch_yfinance(
        symbol=args.symbol, interval=args.interval, period=args.period, out_csv=args.out
    )
    print(f"  {series.describe()}")
    print(f"  wrote {args.out}")
    print(
        "\n  Reminder: Yahoo caps intraday history (7d of M1, 60d of M15). Two months of\n"
        "  one symbol cannot establish an edge. Export M1 from MT5 for real work."
    )
    return 0


def cmd_compare(args) -> int:
    """v1 config vs v2 config, both measured honestly."""
    series, is_synth = _load(args)
    print(f"\n  data: {series.describe()}")
    if is_synth:
        print(SYNTHETIC_WARNING)
    v1 = v1_equivalent()
    honest_v1 = RunConfig(
        strategy=v1.strategy, costs=gold_costs(), risk=RiskConfig(), bias=BiasFlags()
    )
    v2 = _base_cfg(args)
    rows = []
    for label, cfg in (("v1 parameters", honest_v1), ("v2 parameters", v2)):
        res = run_backtest(series, cfg, label=label)
        rows.append((label, compute(res)))
    print(
        "\n  Both rows: honest engine, honest costs, 0.75% risk. Only the strategy\n"
        "  parameters differ, so this isolates the value of the filters added in v2.\n"
    )
    print(audit_mod.format_table(rows))
    return 0


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cct2", description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data_args(sp, default_days=180):
        sp.add_argument("--csv", help="OHLC csv (M1 recommended)")
        sp.add_argument("--tf", default="1m", help="timeframe of --csv (default 1m)")
        sp.add_argument("--synthetic-days", type=int, default=default_days)
        sp.add_argument(
            "--synthetic-mode", default="regime", choices=("regime", "martingale")
        )
        sp.add_argument("--seed", type=int, default=7)

    def add_run_args(sp):
        sp.add_argument("--risk", type=float, help="risk fraction per trade, e.g. 0.0075")
        sp.add_argument("--capital", type=float)
        sp.add_argument("--spread", type=float, help="full spread in price units")
        sp.add_argument("--commission", type=float, help="per lot per side")

    sp = sub.add_parser("selftest", help="engine invariants, incl. the no-edge-in-noise test")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("backtest", help="single honest backtest")
    add_data_args(sp)
    add_run_args(sp)
    sp.add_argument("--label", default="")
    sp.add_argument("--log", action="store_true", help="print the trade log")
    sp.add_argument("--log-limit", type=int, default=40)
    sp.add_argument("--json", help="write metrics + trades to this file")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("audit", help="quantify each of v1's measurement defects")
    add_data_args(sp)
    add_run_args(sp)
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("walkforward", help="out-of-sample parameter selection")
    add_data_args(sp, default_days=400)
    add_run_args(sp)
    sp.add_argument("--folds", type=int, default=4)
    sp.add_argument("--warmup-days", type=float, default=15.0)
    sp.add_argument(
        "--min-train-days",
        type=float,
        default=120.0,
        help="shortest acceptable training window; folds start after it",
    )
    sp.add_argument("--min-trades", type=int, default=12)
    sp.add_argument("--quick", action="store_true", help="tiny parameter grid")
    sp.add_argument("--log", action="store_true")
    sp.add_argument("--log-limit", type=int, default=40)
    sp.set_defaults(func=cmd_walkforward)

    sp = sub.add_parser("null", help="prove the engine finds no edge in random data")
    add_data_args(sp, default_days=150)
    add_run_args(sp)
    sp.add_argument("--seeds", type=int, default=5)
    sp.set_defaults(func=cmd_null)

    sp = sub.add_parser("compare", help="v1 vs v2 parameters, both measured honestly")
    add_data_args(sp)
    add_run_args(sp)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser(
        "portfolio", help="run several instruments and pool the trades into one sample"
    )
    sp.add_argument("--csv", action="append", help="repeat once per instrument")
    sp.add_argument("--spec", help="JSON file with per-instrument costs")
    sp.add_argument("--tf", default="1m")
    add_run_args(sp)
    sp.set_defaults(func=cmd_portfolio)

    sp = sub.add_parser("fetch", help="download bars via yfinance (needs network+pandas)")
    sp.add_argument("--symbol", default="GC=F")
    sp.add_argument("--interval", default="1m")
    sp.add_argument("--period", default="7d")
    sp.add_argument("--out", default="bars.csv")
    sp.set_defaults(func=cmd_fetch)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
