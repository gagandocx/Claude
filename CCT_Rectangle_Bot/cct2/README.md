# cct2 - CCT rectangle strategy, measured honestly

Same strategy idea as `CCT_Rectangle_Bot` v1. Rebuilt measurement.

v1 reported **775% per month**. That number is an artefact. `AUDIT.md` lists the six
defects that produced it; this package reproduces each one on demand so the claim is
checkable rather than assertable:

```
python -m cct2.run audit
```

Output on 560 days of synthetic gold-like bars, v1's parameters, fixed 0.5% stake.
Only the defect being switched changes between rows, so every difference is artefact:

```
variant                            trades   win%     E[R]     PF     total %
v1 params, no defects                 249   24.9   -0.033   0.95        -4.4
only lookahead_direction              255   44.7   +0.756   2.31      +150.1
only lookahead_swings                 232   20.3   -0.217   0.71       -21.8
only skip_invalidation                290   26.6   +0.030   1.03        +3.6
only optimistic_intrabar              236   25.4   -0.026   0.96        -3.1
only zero_costs                       249   25.7   +0.028   1.03        +2.6
cumulative 5 defects                  255   42.7   +1.046   2.84      +252.0
```

Read the first and last rows together. Measured honestly the strategy is slightly
negative; with the defects restored it earns +1.05R a trade. **The entire apparent
edge is the look-ahead**: treating a 4H bar as knowable at its open instead of its
close, on its own, moves the win rate from 24.9% to 44.7%.

Then compound +1.05R a trade at 25% risk and any headline you like drops out. The
risk ladder printed underneath shows the other side of that stake:

```
  risk/trade  trades     E[R]       total %    maxDD %
       0.25%     240   -0.107          -6.0        8.0
       0.75%     240   -0.107         -17.4       23.1
       2.00%     240   -0.107         -44.1       53.2
      25.00%     161   -0.002         -99.9       99.9
```

## No dependencies

Pure standard library, Python 3.9+. No pandas, no numpy, no network. `yfinance` is
imported lazily and only by `run.py fetch`.

## Commands

```bash
python -m cct2.run selftest                     # 30 engine invariants (~25s)
python -m cct2.run audit      [--csv FILE]      # price each of v1's defects
python -m cct2.run null                         # find no edge in random data
python -m cct2.run backtest   --csv gold_m1.csv --log
python -m cct2.run walkforward --csv gold_m1.csv --folds 4
python -m cct2.run compare    --csv gold_m1.csv # v1 vs v2 params, both honest
python -m cct2.run portfolio  --spec instruments.json
python -m cct2.run fetch --symbol GC=F --interval 1m --period 7d --out gold.csv
```

Without `--csv`, every command falls back to synthetic bars and says so. Synthetic
results describe the generator, not gold.

## What changed, and why each change matters

| v1 | cct2 |
|---|---|
| 4H signal stamped at the bar's open, and reads the next bar's open | every object carries `known_ts`; nothing is acted on before its bar closes |
| Enters even if price already traded through the future stop | setup dies on rectangle breach (`invalidate_on_rect_breach`) |
| Invents alternating win/loss when exit data is missing | unresolved positions close at the last price, tagged `data_end` |
| Trades simulated to completion in setup order, equity fed forward | single forward pass; equity moves at exit time, in exit order |
| `MAX_CONCURRENT_TRADES` checked against a list that is never populated | concurrency enforced, plus a same-direction limit |
| Trail raised on the current bar's high, then that bar's low tested against it | trail moves at bar close, binds from the next bar; ambiguous bars resolve as stop-first |
| Entry fills at the candle close, exits fill exactly at the level, zero commission | half-spread + slippage per side, extra slippage on stops, commission per lot per side, gap fills at the open |
| Thresholds in absolute price (`SWEEP_MIN_PIPS = 0.10`) | thresholds in ATR multiples, so they transfer across symbols and volatility regimes |
| Filters relaxed in a retry loop until trades appeared, config mutated globally | immutable config per run, echoed into the result; parameters chosen by walk-forward |
| Reports win rate and an extrapolated monthly figure | reports expectancy in R with a bootstrap CI, and how many trades confirmation would take |
| 25% risk per trade | 0.75% default, 2% hard ceiling, daily loss cap, loss-streak cooldown, notional cap, ruin floor |

## The finding that no amount of tuning fixes

Per-trade dispersion for this strategy is around 1.3R. To separate an expectancy of
+0.10R from zero at t = 2 you therefore need `(2 * 1.3 / 0.10)^2` = about 675 trades.

Measured on gold-like bars, v1's strict engulfing rule fires on ~5% of 4H bars, which
through the full 4H -> 15M -> 1M funnel is roughly 6 trades a month; the looser
`reclaim` variant roughly doubles it. Call it 12 a month, and confirmation still
takes **four to five years of trading**.

So every result prints the arithmetic:

```
    sample                157 trades over 420 days (11.4/month)
    verdict               NOT distinguishable from zero on this sample
    to confirm at t=2     657 trades = 58 months at this rate
```

v1's 47 trades over 60 days could not have established anything in either direction,
and no parameter search over that window can. This is why `run.py portfolio` exists:
pooling across instruments adds observations, whereas loosening filters just re-uses
the same ones with more chances to fit them.

If you want a faster verdict the honest options are more instruments, more history, or
a strategy that trades more often. Not a wider grid.

## Getting real data

`fetch` uses Yahoo, which caps intraday history at 7 days of M1 and 60 days of M15.
That is a smoke test, not a study. For real work export M1 from MT5 (raise
Tools > Options > Charts > "Max bars in chart", then right-click the M1 chart and
Save As), or use `ExportRealTicks.mq5` in the repo root.

Costs are per instrument in the portfolio spec, because one spread figure cannot be
right for gold and EURUSD at once. Take the numbers from your broker's contract
specification and round against yourself. After look-ahead, optimistic cost
assumptions are the most common reason a backtest lies.

## Before risking money

1. `selftest` passes (the null test is the one that matters: no edge found in noise).
2. `walkforward` shows positive expectancy in most folds, not one, with stable
   parameter choices across folds.
3. The bootstrap CI lower bound is above zero on a sample you did not tune on.
4. Only then size it, starting below the walk-forward's own numbers, because those
   are still the best of N parameter sets.

If step 2 or 3 fails, the answer is that this configuration is not tradeable yet.
That is a legitimate outcome and it is the one v1's reporting was built to hide.
