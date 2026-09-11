"""
Performance statistics, including the ones that tell you whether the result means
anything.

v1 reported win rate, profit factor and a monthly figure extrapolated from 47
trades over 60 days. None of those answer "could this have been luck?", which for
a 47-trade sample is the only question worth asking first. So the summary here
leads with per-trade expectancy in R, its bootstrap confidence interval, and the
sample size needed to distinguish the result from zero.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .bars import fmt_ts, stdev
from .engine import BacktestResult, Trade

TRADING_DAYS = 252


@dataclass
class Metrics:
    label: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    median_r: float = 0.0
    stdev_r: float = 0.0
    t_stat: float = 0.0
    ci_low_r: float = 0.0
    ci_high_r: float = 0.0
    p_not_positive: float = 1.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    monthly_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_dd_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_consec_losses: int = 0
    trades_per_month: float = 0.0
    trades_needed: int = 0
    months_needed: float = 0.0
    avg_hold_min: float = 0.0
    exposure_pct: float = 0.0
    cost_drag_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    commissions: float = 0.0
    final_capital: float = 0.0
    initial_capital: float = 0.0
    days: float = 0.0
    ruined: bool = False
    exit_reasons: dict[str, int] = field(default_factory=dict)
    rejects: dict[str, int] = field(default_factory=dict)
    setups_created: int = 0
    signals_created: int = 0
    data_end_trades: int = 0

    @property
    def sample_is_meaningful(self) -> bool:
        return self.trades >= 30 and self.ci_low_r > 0

    def to_row(self) -> dict:
        return {
            "label": self.label,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "expectancy_r": self.expectancy_r,
            "ci_low_r": self.ci_low_r,
            "profit_factor": self.profit_factor,
            "total_return_pct": self.total_return_pct,
            "monthly_return_pct": self.monthly_return_pct,
            "max_dd_pct": self.max_dd_pct,
            "sharpe": self.sharpe,
        }


def _bootstrap_mean_ci(
    values: list[float], iters: int = 4000, seed: int = 12345
) -> tuple[float, float, float]:
    """
    Percentile bootstrap for the mean of per-trade R, plus the share of resamples
    that came out <= 0 (a one-sided p-value proxy for "there is no edge").
    """
    n = len(values)
    if n < 5:
        return (float("nan"), float("nan"), 1.0)
    rng = random.Random(seed)
    means = []
    non_positive = 0
    for _ in range(iters):
        acc = 0.0
        for _ in range(n):
            acc += values[rng.randrange(n)]
        m = acc / n
        means.append(m)
        if m <= 0:
            non_positive += 1
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    return (lo, hi, non_positive / iters)


def _max_drawdown(curve: list[tuple[int, float]]) -> float:
    peak = None
    worst = 0.0
    for _, eq in curve:
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = (eq - peak) / peak
            worst = min(worst, dd)
    return abs(worst) * 100.0


def _daily_returns(curve: list[tuple[int, float]]) -> list[float]:
    out = []
    for (_, a), (_, b) in zip(curve, curve[1:]):
        if a > 0:
            out.append(b / a - 1.0)
    return out


def compute(res: BacktestResult, bootstrap: bool = True) -> Metrics:
    m = Metrics(label=res.label)
    m.initial_capital = res.initial_capital
    m.final_capital = res.final_capital
    m.days = res.days
    m.ruined = res.ruined
    m.rejects = dict(sorted(res.rejects.items(), key=lambda kv: -kv[1]))
    m.setups_created = res.setups_created
    m.signals_created = res.signals_created

    trades = res.trades
    m.trades = len(trades)
    if m.trades == 0:
        return m

    rs = [t.r for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    m.wins, m.losses = len(wins), len(losses)
    m.win_rate = 100.0 * m.wins / m.trades
    m.expectancy_r = sum(rs) / len(rs)
    srt = sorted(rs)
    mid = len(srt) // 2
    m.median_r = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
    m.stdev_r = stdev(rs)
    if m.stdev_r > 0:
        m.t_stat = m.expectancy_r / (m.stdev_r / math.sqrt(len(rs)))
    if bootstrap:
        lo, hi, p = _bootstrap_mean_ci(rs)
        m.ci_low_r, m.ci_high_r, m.p_not_positive = lo, hi, p

    m.avg_win_r = sum(t.r for t in wins) / len(wins) if wins else 0.0
    m.avg_loss_r = sum(t.r for t in losses) / len(losses) if losses else 0.0
    m.gross_profit = sum(t.pnl for t in wins)
    m.gross_loss = abs(sum(t.pnl for t in losses))
    m.profit_factor = (
        m.gross_profit / m.gross_loss if m.gross_loss > 0 else float("inf")
    )
    m.commissions = sum(t.commission_paid for t in trades)

    m.total_return_pct = (
        (res.final_capital - res.initial_capital) / res.initial_capital * 100.0
        if res.initial_capital
        else 0.0
    )
    mult = res.final_capital / res.initial_capital if res.initial_capital else 0.0
    months = res.days / 30.44
    if mult > 0 and months > 0.5:
        m.monthly_return_pct = (mult ** (1.0 / months) - 1.0) * 100.0
        m.cagr_pct = (mult ** (365.0 / res.days) - 1.0) * 100.0
    elif mult <= 0:
        m.monthly_return_pct = -100.0
        m.cagr_pct = -100.0

    curve = [(res.start_ts, res.initial_capital)] + res.equity_by_trade
    m.max_dd_pct = _max_drawdown(curve)

    daily = _daily_returns(res.equity_daily)
    if len(daily) > 5:
        mu = sum(daily) / len(daily)
        sd = stdev(daily)
        if sd > 0:
            m.sharpe = mu / sd * math.sqrt(TRADING_DAYS)
        downs = [d for d in daily if d < 0]
        dsd = stdev(downs) if len(downs) > 1 else 0.0
        if dsd > 0:
            m.sortino = mu / dsd * math.sqrt(TRADING_DAYS)

    streak = best = 0
    for t in trades:
        if t.pnl <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    m.max_consec_losses = best

    m.trades_per_month = m.trades / months if months > 0 else 0.0

    # Power calculation. To distinguish an expectancy of E from zero at t = 2 you
    # need n >= (2*sigma/E)^2 trades. For a strategy that fires twice a month this
    # is usually a number of years, which is the single most important fact about
    # the CCT funnel and one that no amount of parameter tuning changes.
    if m.expectancy_r > 0 and m.stdev_r > 0:
        m.trades_needed = int(math.ceil((2.0 * m.stdev_r / m.expectancy_r) ** 2))
        if m.trades_per_month > 0:
            m.months_needed = m.trades_needed / m.trades_per_month
    holds = [
        (t.exit_ts - t.entry_ts) / 60.0 for t in trades if t.exit_ts and t.entry_ts
    ]
    m.avg_hold_min = sum(holds) / len(holds) if holds else 0.0
    total_min = res.days * 24 * 60
    m.exposure_pct = 100.0 * sum(holds) / total_min if total_min > 0 else 0.0

    gross_total = m.gross_profit + m.gross_loss
    if gross_total > 0:
        m.cost_drag_pct = 100.0 * m.commissions / gross_total

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    m.exit_reasons = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    m.data_end_trades = sum(1 for t in trades if t.reason.startswith("data_end"))
    return m


def summary(m: Metrics, verbose: bool = True) -> str:
    L: list[str] = []
    w = L.append
    title = m.label or "backtest"
    w("=" * 78)
    w(f"  {title}")
    w("=" * 78)
    if m.trades == 0:
        w("  no trades")
        w(f"  4H direction signals: {m.signals_created}   15M setups: {m.setups_created}")
        if m.rejects:
            w("  rejects: " + ", ".join(f"{k}={v}" for k, v in m.rejects.items()))
        return "\n".join(L)

    w("")
    w("  IS THERE AN EDGE")
    w(f"    expectancy            {m.expectancy_r:+.3f} R per trade")
    w(f"    bootstrap 95% CI      [{m.ci_low_r:+.3f}, {m.ci_high_r:+.3f}] R")
    w(f"    P(mean R <= 0)        {m.p_not_positive:.3f}      t-stat {m.t_stat:+.2f}")
    w(f"    sample                {m.trades} trades over {m.days:.0f} days "
      f"({m.trades_per_month:.1f}/month)")
    verdict = (
        "edge is statistically distinguishable from zero"
        if m.sample_is_meaningful
        else "NOT distinguishable from zero on this sample"
    )
    w(f"    verdict               {verdict}")
    if m.trades_needed:
        w(f"    to confirm at t=2     {m.trades_needed} trades "
          f"= {m.months_needed:.0f} months at this rate"
          + ("  <-- longer than any usable backtest" if m.months_needed > 60 else ""))

    w("")
    w("  TRADE STATISTICS")
    w(f"    win rate              {m.win_rate:.1f}%   ({m.wins}W / {m.losses}L)")
    w(f"    avg win / avg loss    {m.avg_win_r:+.2f} R / {m.avg_loss_r:+.2f} R")
    w(f"    median trade          {m.median_r:+.2f} R      sigma {m.stdev_r:.2f} R")
    w(f"    profit factor         {m.profit_factor:.2f}")
    w(f"    max consec losses     {m.max_consec_losses}")
    w(f"    avg hold              {m.avg_hold_min:.0f} min      exposure {m.exposure_pct:.1f}%")

    w("")
    w("  MONEY")
    w(f"    capital               ${m.initial_capital:,.0f} -> ${m.final_capital:,.0f}")
    w(f"    total return          {m.total_return_pct:+.1f}%")
    w(f"    compounded monthly    {m.monthly_return_pct:+.1f}%")
    w(f"    max drawdown          {m.max_dd_pct:.1f}%")
    w(f"    Sharpe / Sortino      {m.sharpe:.2f} / {m.sortino:.2f}")
    w(f"    commission paid       ${m.commissions:,.0f} ({m.cost_drag_pct:.1f}% of gross)")
    if m.ruined:
        w("    *** RUIN FLOOR HIT: trading halted mid-run ***")

    if verbose:
        w("")
        w("  PIPELINE")
        w(f"    4H direction signals  {m.signals_created}")
        w(f"    15M setups            {m.setups_created}")
        w(f"    filled                {m.trades}")
        if m.exit_reasons:
            w("    exits                 "
              + ", ".join(f"{k}={v}" for k, v in m.exit_reasons.items()))
        if m.data_end_trades:
            w(f"    closed by data end    {m.data_end_trades} "
              "(unresolved at the end of the sample, not extrapolated)")
        if m.rejects:
            top = list(m.rejects.items())[:8]
            w("    setups dropped        " + ", ".join(f"{k}={v}" for k, v in top))
        min_lot_drops = m.rejects.get("below_min_lot", 0)
        if m.setups_created and min_lot_drops > 0.1 * m.setups_created:
            w("")
            w(f"    {min_lot_drops} setups ({100 * min_lot_drops / m.setups_created:.0f}% "
              "of them) were skipped because the stop distance times the")
            w("    broker's minimum lot exceeds the risk budget. The account is too small")
            w("    for these stops, not the strategy too selective. Raise capital, or trade")
            w("    an instrument whose minimum lot is smaller relative to its volatility.")
    w("=" * 78)
    return "\n".join(L)


def trade_log(res: BacktestResult, limit: int = 40) -> str:
    L = []
    w = L.append
    w("-" * 104)
    w(f"{'#':<4}{'entry (UTC)':<18}{'dir':<6}{'lots':>7}{'entry':>10}{'stop':>10}"
      f"{'exit':>10}{'R':>8}{'P&L $':>12}{'reason':<16}")
    w("-" * 104)
    for i, t in enumerate(res.trades[:limit], 1):
        w(
            f"{i:<4}{fmt_ts(t.entry_ts):<18}{('LONG' if t.direction=='long' else 'SHORT'):<6}"
            f"{t.lots_initial:>7.2f}{t.entry_price:>10.2f}{t.initial_stop:>10.2f}"
            f"{(t.exit_price or 0):>10.2f}{t.r:>8.2f}{t.pnl:>12,.2f}  {t.reason:<16}"
        )
    if len(res.trades) > limit:
        w(f"... {len(res.trades) - limit} more")
    return "\n".join(L)



def pool(
    results: list[BacktestResult],
    risk_frac: float = 0.0075,
    initial: float = 10_000.0,
    label: str = "pooled",
) -> BacktestResult:
    """
    Merge several runs into one sample by replaying their trades in exit order at a
    fixed fractional stake.

    Why this exists: the CCT funnel produces one or two trades a month on a single
    symbol, so a single-symbol backtest can never accumulate the observations
    needed to tell an edge from noise (see `Metrics.trades_needed`). Pooling across
    instruments is the only honest way to get the count up, because it adds
    observations instead of re-using the same ones with looser parameters.

    R is scale-free, so replaying `equity *= 1 + risk_frac * R` is a faithful
    reconstruction of what fractional-risk sizing would have produced.

    Two assumptions, both stated rather than hidden:
      * No cross-symbol concurrency limit, so the drawdown here is optimistic if
        the instruments are correlated. Gold and silver are not two samples.
      * Equal risk per trade across instruments.
    """
    trades = sorted(
        (t for r in results for t in r.trades if t.exit_ts),
        key=lambda t: t.exit_ts,
    )
    if not trades:
        return BacktestResult(initial_capital=initial, final_capital=initial, label=label)

    equity = initial
    by_trade: list[tuple[int, float]] = []
    for t in trades:
        equity *= 1.0 + risk_frac * t.r
        equity = max(equity, 0.0)
        by_trade.append((t.exit_ts, equity))

    # A proper daily curve (carrying equity forward through days with no trades)
    # so that Sharpe is computed on daily returns rather than on trade events.
    start_day = min(r.start_ts for r in results if r.start_ts) // 86400
    end_day = max(t.exit_ts for t in trades) // 86400
    daily: list[tuple[int, float]] = []
    eq = initial
    ptr = 0
    for day in range(start_day, end_day + 1):
        day_end = (day + 1) * 86400
        while ptr < len(trades) and trades[ptr].exit_ts < day_end:
            eq *= 1.0 + risk_frac * trades[ptr].r
            eq = max(eq, 0.0)
            ptr += 1
        daily.append((day * 86400, eq))

    rejects: dict[str, int] = {}
    for r in results:
        for k, v in r.rejects.items():
            rejects[k] = rejects.get(k, 0) + v

    return BacktestResult(
        trades=trades,
        equity_daily=daily,
        equity_by_trade=by_trade,
        initial_capital=initial,
        final_capital=equity,
        start_ts=min(r.start_ts for r in results if r.start_ts),
        end_ts=max(r.end_ts for r in results),
        setups_created=sum(r.setups_created for r in results),
        signals_created=sum(r.signals_created for r in results),
        rejects=rejects,
        label=label,
    )
