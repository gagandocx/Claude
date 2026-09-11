"""
cct2 - a corrected implementation of the CCT rectangle strategy.

Why this exists: see AUDIT.md. The short version is that v1's headline number was
produced by a backtester that could see up to eight hours into the future, entered
setups that had already been stopped out, invented outcomes when data was missing,
and charged no transaction costs. This package keeps the strategy idea and rebuilds
the measurement.

Entry points:
    python -m cct2.run selftest        # prove the engine cannot print free money
    python -m cct2.run audit           # price each of v1's defects
    python -m cct2.run backtest --csv gold_m1.csv
    python -m cct2.run walkforward --csv gold_m1.csv

Nothing here needs pandas or numpy.
"""

from .bars import Bar, Series, load_csv, resample
from .engine import Backtester, BacktestResult, Trade, run_backtest
from .metrics import Metrics, compute, summary
from .settings import (
    BiasFlags,
    CostModel,
    RiskConfig,
    RunConfig,
    StrategyConfig,
    gold_costs,
    v1_equivalent,
)

__all__ = [
    "Bar",
    "Series",
    "load_csv",
    "resample",
    "Backtester",
    "BacktestResult",
    "Trade",
    "run_backtest",
    "Metrics",
    "compute",
    "summary",
    "BiasFlags",
    "CostModel",
    "RiskConfig",
    "RunConfig",
    "StrategyConfig",
    "gold_costs",
    "v1_equivalent",
]
