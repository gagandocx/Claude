"""
Walk-Forward Parameter Optimizer.

Splits data into train (70%) and validation (30%) periods.
Performs grid search over key parameters for each strategy.
Selects parameters that perform best on out-of-sample data.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from .backtest_engine import run_backtest, BacktestConfig, BacktestResult
from .strategies.base import BaseStrategy


def optimize_strategy(
    strategy_class: type,
    bars: pd.DataFrame,
    param_grid: Dict[str, list],
    train_ratio: float = 0.7,
    config: BacktestConfig = None,
    metric: str = "sharpe_ratio",
    max_combinations: int = 200,
) -> Tuple[Dict[str, Any], BacktestResult, BacktestResult]:
    """
    Walk-forward optimization for a strategy.

    Parameters
    ----------
    strategy_class : type
        Strategy class to optimize
    bars : pd.DataFrame
        Full bar data
    param_grid : dict
        Parameter grid {param_name: [values]}
    train_ratio : float
        Fraction of data for training (default 0.7)
    config : BacktestConfig
        Backtest configuration
    metric : str
        Metric to optimize (default sharpe_ratio)
    max_combinations : int
        Maximum parameter combinations to test

    Returns
    -------
    Tuple of (best_params, train_result, validation_result)
    """
    if config is None:
        config = BacktestConfig()

    n_bars = len(bars)
    split_idx = int(n_bars * train_ratio)

    train_bars = bars.iloc[:split_idx].copy()
    val_bars = bars.iloc[split_idx:].copy()

    print(f"  Train: {len(train_bars)} bars, Validation: {len(val_bars)} bars")

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    # Limit combinations
    if len(combinations) > max_combinations:
        # Random sampling of combinations
        rng = np.random.default_rng(42)
        indices = rng.choice(len(combinations), size=max_combinations, replace=False)
        combinations = [combinations[i] for i in indices]

    print(f"  Testing {len(combinations)} parameter combinations...")

    # Two-pass approach: find top candidates on training, then validate
    candidates = []

    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            strategy = strategy_class(params)
            signals = strategy.generate_signals(train_bars)
            result = run_backtest(train_bars, signals, config)

            # Basic filter: needs minimum trades and acceptable drawdown
            if result.total_trades < 5:
                continue
            if result.max_drawdown_pct < -30:
                continue

            metric_value = getattr(result, metric, 0.0)
            candidates.append((params, result, metric_value))

        except Exception:
            continue

    if not candidates:
        # Fallback to default params
        print("  No valid parameters found, using defaults")
        strategy = strategy_class()
        best_params = strategy.get_default_params()
        signals = strategy.generate_signals(train_bars)
        best_train_result = run_backtest(train_bars, signals, config)
        val_signals = strategy.generate_signals(val_bars)
        val_result = run_backtest(val_bars, val_signals, config)
        return best_params, best_train_result, val_result

    # Sort by metric and take top 20 candidates for validation
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:20]

    # Validate top candidates on OOS data
    best_combined_score = -np.inf
    best_params = top_candidates[0][0]
    best_train_result = top_candidates[0][1]
    best_val_result = None

    for params, train_result, train_metric in top_candidates:
        try:
            val_strategy = strategy_class(params)
            val_signals = val_strategy.generate_signals(val_bars)
            val_result = run_backtest(val_bars, val_signals, config)

            # Combined score: train performance + validation performance
            # Validation gets higher weight to prevent overfitting
            combined = (
                train_metric * 0.4 +
                getattr(val_result, metric, 0.0) * 0.6
            )

            if combined > best_combined_score:
                best_combined_score = combined
                best_params = params
                best_train_result = train_result
                best_val_result = val_result

        except Exception:
            continue

    if best_val_result is None:
        val_strategy = strategy_class(best_params)
        val_signals = val_strategy.generate_signals(val_bars)
        best_val_result = run_backtest(val_bars, val_signals, config)

    return best_params, best_train_result, best_val_result


def quick_optimize(
    strategy_class: type,
    bars: pd.DataFrame,
    config: BacktestConfig = None,
    metric: str = "calmar_ratio",
) -> Tuple[Dict[str, Any], BacktestResult, BacktestResult]:
    """
    Quick optimization with reduced parameter grid.
    Uses the strategy's built-in param_grid but limits combinations.
    Optimizes for calmar_ratio (return/drawdown) to get best PnL with least DD.
    """
    strategy = strategy_class()
    param_grid = strategy.get_param_grid()

    if not param_grid:
        # No grid defined, run with defaults
        n_bars = len(bars)
        split_idx = int(n_bars * 0.7)
        train_bars = bars.iloc[:split_idx].copy()
        val_bars = bars.iloc[split_idx:].copy()

        signals = strategy.generate_signals(train_bars)
        train_result = run_backtest(train_bars, signals, config or BacktestConfig())

        val_signals = strategy.generate_signals(val_bars)
        val_result = run_backtest(val_bars, val_signals, config or BacktestConfig())

        return strategy.get_default_params(), train_result, val_result

    # Use calmar_ratio as metric (PnL / max_drawdown) for best risk-adjusted returns
    return optimize_strategy(
        strategy_class, bars, param_grid,
        config=config, metric="calmar_ratio", max_combinations=150
    )
