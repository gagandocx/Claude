# CCT Rectangle Bot

Two implementations of the same strategy live here.

## `cct2/` - use this one

Corrected, dependency-free (stdlib only, Python 3.9+), event-driven, with a
bias-audit harness and walk-forward validation.

```bash
python -m cct2.run selftest      # 30 engine invariants, ~25s
python -m cct2.run audit         # prices each of v1's measurement defects
python -m cct2.run backtest --csv your_m1_data.csv --log
python -m cct2.run walkforward --csv your_m1_data.csv --folds 4
```

Read `cct2/README.md` for the command reference and `cct2/AUDIT.md` for what was
wrong with v1.

## v1 (`main.py`, `strategy.py`, `backtester.py`, `config.py`, `utils.py`)

Superseded. It reports 775%/month. That figure is the sum of six measurement
defects, and `python -m cct2.run audit` reproduces each one on identical bars so you
can see the contribution of each: the largest single item, timestamping 4H signals at
the bar's open instead of its close, moves the win rate from 31% to 50% by itself.

Left in the tree because the audit reproduces it and because the git history is worth
keeping. Requires `pandas` + `yfinance`; the corrected engine requires nothing.

## The headline finding

Per-trade dispersion here is about 1.3R, so confirming an expectancy of +0.10R needs
roughly 675 trades. The funnel produces something like 6-12 trades a month on one
symbol, which puts confirmation four to five years out. Every result prints that
arithmetic instead of a monthly percentage extrapolated from six weeks.

v1's 47 trades over 60 days could not have settled the question in either direction,
and no parameter search over that window can either. The useful next steps are more
instruments (`run.py portfolio`) and more history, not more tuning.
