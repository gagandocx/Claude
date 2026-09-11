"""
Prices each of v1's defects on identical bars.

Method: hold the strategy parameters and the risk settings fixed at v1's, then
switch the defects on one at a time and cumulatively. Any change in the reported
result is attributable to the defect and to nothing else, because the strategy,
the data and the costs are unchanged.

Read the output as: "this much of the reported performance was an artefact".

Caveat, in the direction of understating: the `lookahead_direction` flag reproduces
only the resample-label half of AUDIT #1 (a 4H bar treated as knowable at its open).
v1 additionally read `bars[i+1].open`, a second 4 hours. The real leak is larger
than what is measured here.
"""

from __future__ import annotations

from dataclasses import replace

from .bars import Series
from .engine import run_backtest
from .metrics import Metrics, compute
from .settings import BiasFlags, RiskConfig, RunConfig, v1_equivalent

FLAG_ORDER = [
    ("lookahead_direction", "4H signal timestamped at bar open (AUDIT #1)"),
    ("lookahead_swings", "pivots used before confirmation (AUDIT, secondary)"),
    ("skip_invalidation", "enter although price already hit the future stop (#2)"),
    ("optimistic_intrabar", "target-before-stop, trail on the current bar (#5)"),
    ("zero_costs", "no spread, slippage or commission (#6)"),
]


def _run(series: Series, cfg: RunConfig, label: str) -> Metrics:
    try:
        res = run_backtest(series, cfg, label=label)
    except ValueError as exc:
        return Metrics(label=f"{label} (failed: {exc})")
    return compute(res)


def _audit_risk() -> RiskConfig:
    """
    Small, fixed stake for every bias variant.

    v1's own 25% is deliberately NOT used here: at that stake the ruin floor and
    the compounding path dominate, and rows stop being comparable. The stake
    question is asked separately by `risk_ladder`. Governors are also switched off
    so that no variant is truncated by a cooldown the others did not hit.
    """
    return RiskConfig(
        risk_frac=0.005,
        max_risk_frac=0.005,
        daily_loss_cap_frac=1.0,
        max_consecutive_losses=10_000,
        max_leverage=1e9,
        ruin_floor_frac=0.0,
    )


def run_audit(series: Series, verbose: bool = True) -> list[tuple[str, Metrics]]:
    base = v1_equivalent()
    risk = _audit_risk()

    def cfg_with(**flags) -> RunConfig:
        return RunConfig(
            strategy=base.strategy,
            costs=base.costs,
            risk=risk,
            bias=BiasFlags(**flags),
        )

    rows: list[tuple[str, Metrics]] = []

    # 1. v1's parameters, measured honestly. The reference row.
    rows.append(("v1 params, no defects", _run(series, cfg_with(), "v1 params, honest")))

    # 2. each defect alone
    for flag, desc in FLAG_ORDER:
        rows.append((f"only {flag}", _run(series, cfg_with(**{flag: True}), desc)))

    # 3. cumulative, in the order above
    acc: dict[str, bool] = {}
    for flag, _ in FLAG_ORDER:
        acc[flag] = True
        rows.append(
            (
                f"cumulative {len(acc)}: +{flag}",
                _run(series, cfg_with(**acc), f"cumulative {len(acc)}"),
            )
        )

    return rows


def risk_ladder(
    series: Series,
    cfg: RunConfig,
    fracs=(0.0025, 0.0075, 0.02, 0.05, 0.25),
    uncapped: bool = True,
):
    """
    Same signals, different bet size.

    Sizing happens after the entry decision, so the stake does not choose the
    trades. It can still change the trade count slightly, because a bigger stake
    clears the broker's minimum lot on setups a small stake cannot afford; the
    `trades` column shows that. With `uncapped=True` the notional limit is lifted
    so the requested stake is actually expressed - which is the situation v1 was
    in, since it sized in raw units with no margin check at all.
    """
    out = []
    for f in fracs:
        risk = replace(
            cfg.risk,
            risk_frac=f,
            max_risk_frac=max(f, cfg.risk.max_risk_frac),
            ruin_floor_frac=0.0,
            daily_loss_cap_frac=1.0,
            max_consecutive_losses=10_000,
        )
        if uncapped:
            risk = replace(risk, max_leverage=1e9)
        c = RunConfig(strategy=cfg.strategy, costs=cfg.costs, risk=risk, bias=cfg.bias)
        out.append((f, _run(series, c, f"risk {f * 100:.2f}%")))
    return out


def format_table(rows: list[tuple[str, Metrics]]) -> str:
    L = []
    w = L.append
    w("-" * 100)
    w(f"{'variant':<34}{'trades':>7}{'win%':>7}{'E[R]':>9}{'PF':>7}"
      f"{'total %':>12}{'monthly %':>12}{'maxDD %':>10}")
    w("-" * 100)
    for label, m in rows:
        pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        w(
            f"{label:<34}{m.trades:>7}{m.win_rate:>7.1f}{m.expectancy_r:>+9.3f}{pf:>7}"
            f"{m.total_return_pct:>+12.1f}{m.monthly_return_pct:>+12.1f}{m.max_dd_pct:>10.1f}"
            + ("  RUIN" if m.ruined else "")
        )
    w("-" * 100)
    return "\n".join(L)


def format_risk_table(rows) -> str:
    L = []
    w = L.append
    w("-" * 88)
    w(f"{'risk/trade':>12}{'trades':>8}{'E[R]':>9}{'total %':>14}{'monthly %':>13}"
      f"{'maxDD %':>11}{'':>6}")
    w("-" * 88)
    for frac, m in rows:
        w(
            f"{frac * 100:>11.2f}%{m.trades:>8}{m.expectancy_r:>+9.3f}"
            f"{m.total_return_pct:>+14.1f}{m.monthly_return_pct:>+13.1f}{m.max_dd_pct:>11.1f}"
            + ("   RUIN" if m.ruined else "")
        )
    w("-" * 88)
    return "\n".join(L)
