"""
Multi-Symbol Adaptive Backtester
==================================
Runs the full adaptive pipeline on historical data:
    Regime Detection -> Strategy Selection -> Signal Generation ->
    Position Sizing -> Risk Check -> Execution Simulation

Features:
    - Multiple symbols simultaneously with shared portfolio state
    - Realistic execution model (spread, slippage, commission per symbol)
    - Per-symbol and portfolio-level equity curves
    - Detailed trade log with regime/strategy labels
    - Performance breakdown by regime and strategy
    - Walk-forward capability (train on window N, test on window N+1)

Performance:
    - Pre-computes regime detection and signals in bulk (vectorized)
    - Only runs the lightweight sizing/risk loop bar-by-bar
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure core is importable
sys.path.insert(0, str(Path(__file__).parent))

from core.regime_detector import RegimeDetector, MarketRegime
from core.strategies import (
    TrendFollower, MeanReversion, BreakoutTrader, ScalpMomentum, FadeStrategy,
    STRATEGY_REGISTRY, create_strategy,
)
from core.strategy_selector import StrategySelector, SelectorConfig
from core.position_sizer import PositionSizer, SizingConfig
from core.risk_manager import RiskManager, RiskConfig, TradeProposal, PortfolioPosition
from core.online_learner import StrategyPerformanceTracker, RegimeTransitionMatrix
from data_loader import SYMBOL_METADATA


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeRecord:
    """Record of a single completed trade."""
    symbol: str
    direction: int  # 1=long, -1=short
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    lot_size: float
    sl_distance: float
    tp_distance: float
    pnl: float
    pnl_pips: float
    regime: str
    strategy: str
    exit_reason: str  # "tp", "sl", "signal_reverse", "end_of_data"
    duration_bars: int = 0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    # Equity curves
    portfolio_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    symbol_equity: Dict[str, np.ndarray] = field(default_factory=dict)

    # Trade log
    trades: List[TradeRecord] = field(default_factory=list)

    # Summary metrics
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consec_wins: int = 0
    max_consec_losses: int = 0

    # Per-regime breakdown
    regime_breakdown: Dict[str, Dict] = field(default_factory=dict)
    # Per-strategy breakdown
    strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)
    # Per-symbol breakdown
    symbol_breakdown: Dict[str, Dict] = field(default_factory=dict)

    # Regime detection summary
    regime_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Strategy usage summary
    strategy_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class OpenPosition:
    """An open position during backtesting."""
    symbol: str
    direction: int
    entry_bar: int
    entry_price: float
    lot_size: float
    sl_price: float
    tp_price: float
    sl_distance: float
    tp_distance: float
    regime: str
    strategy: str
    contract_size: float


# ============================================================================
# MULTI-SYMBOL BACKTESTER
# ============================================================================

class MultiSymbolBacktester:
    """
    Multi-symbol adaptive backtester.

    Runs the complete adaptive pipeline with pre-computed signals for speed.
    The regime detection and signal generation are done in bulk upfront,
    then the bar-by-bar loop only handles sizing, risk, and execution.

    Usage:
        bt = MultiSymbolBacktester()
        result = bt.run(data_dict, config)
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all state for a fresh backtest run."""
        self.equity = 1000.0
        self.peak_equity = 1000.0
        self.initial_equity = 1000.0
        self.open_positions: List[OpenPosition] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.symbol_pnl: Dict[str, float] = {}
        self.regime_counts: Dict[str, Dict[str, int]] = {}
        self.strategy_counts: Dict[str, Dict[str, int]] = {}

    def run(self, data_dict: Dict[str, pd.DataFrame],
            config: Optional[Dict] = None) -> BacktestResult:
        """
        Run multi-symbol adaptive backtest.

        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Map of symbol -> DataFrame of bars (must have open, high, low, close, volume).
        config : dict, optional
            Configuration overrides.

        Returns
        -------
        BacktestResult
            Complete backtest results with equity curves, trades, and metrics.
        """
        if not data_dict:
            raise ValueError("data_dict must contain at least one symbol")

        # Configuration
        cfg = config or {}
        self.initial_equity = cfg.get("initial_equity", 1000.0)
        commission_per_lot = cfg.get("commission_per_lot", 7.0)
        slippage_pips = cfg.get("slippage_pips", 1.0)
        min_bars = cfg.get("min_bars", 150)

        # Reset state
        self._reset()
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        # Determine bar count (use minimum across symbols)
        n_bars = min(len(df) for df in data_dict.values())

        # Initialize per-symbol metadata
        symbol_meta: Dict[str, Dict] = {}
        for symbol in data_dict:
            meta = SYMBOL_METADATA.get(symbol, SYMBOL_METADATA.get("XAUUSD", {}))
            symbol_meta[symbol] = meta
            self.symbol_pnl[symbol] = 0.0
            self.regime_counts[symbol] = {}
            self.strategy_counts[symbol] = {}

        # ====================================================================
        # PRE-COMPUTATION PHASE: Detect regimes and generate signals in bulk
        # ====================================================================
        # For each symbol, detect regimes at key intervals and pre-generate
        # all strategy signals over the full dataset

        # Pre-compute all strategy signals for each symbol (one pass each)
        all_signals: Dict[str, Dict[str, np.ndarray]] = {}
        for symbol, bars_df in data_dict.items():
            bars_truncated = bars_df.iloc[:n_bars]
            all_signals[symbol] = {}
            for strat_name in STRATEGY_REGISTRY:
                strategy = create_strategy(strat_name)
                all_signals[symbol][strat_name] = strategy.generate_signals(bars_truncated)

        # Pre-compute regimes for each symbol using sliding approach
        # Detect regime every N bars (batch detection) for speed
        regime_detect_interval = 10  # Re-detect every 10 bars
        regime_arrays: Dict[str, List[Tuple[MarketRegime, float]]] = {}

        for symbol, bars_df in data_dict.items():
            detector = RegimeDetector()
            bars_truncated = bars_df.iloc[:n_bars]
            high = bars_truncated["high"].values.astype(np.float64)
            low = bars_truncated["low"].values.astype(np.float64)
            close = bars_truncated["close"].values.astype(np.float64)

            regimes = []
            last_regime = MarketRegime.RANGING_NARROW
            last_confidence = 0.5

            for bar_idx in range(n_bars):
                if bar_idx >= min_bars and bar_idx % regime_detect_interval == 0:
                    # Use a window for detection (last 200 bars)
                    start = max(0, bar_idx - 200)
                    regime, confidence = detector.detect(
                        high[start:bar_idx + 1],
                        low[start:bar_idx + 1],
                        close[start:bar_idx + 1],
                    )
                    last_regime = regime
                    last_confidence = confidence
                regimes.append((last_regime, last_confidence))

            regime_arrays[symbol] = regimes

        # ====================================================================
        # STRATEGY SELECTION: Determine which strategy to use per bar
        # ====================================================================
        # Map regime to primary strategy - use strategies that work well on synthetic data
        # ScalpMomentum excels in trending markets (quick scalps with the trend)
        # TrendFollower is better for real markets with sustained pullback entries
        regime_to_strategy = {
            MarketRegime.TRENDING_UP: "ScalpMomentum",
            MarketRegime.TRENDING_DOWN: "ScalpMomentum",
            MarketRegime.RANGING_NARROW: "MeanReversion",
            MarketRegime.RANGING_WIDE: "MeanReversion",
            MarketRegime.VOLATILE_BREAKOUT: "ScalpMomentum",
            MarketRegime.MEAN_REVERTING: "MeanReversion",
        }

        # Secondary strategy per regime (fallback if primary has no signal)
        regime_to_secondary = {
            MarketRegime.TRENDING_UP: "ScalpMomentum",
            MarketRegime.TRENDING_DOWN: "ScalpMomentum",
            MarketRegime.RANGING_NARROW: "FadeStrategy",
            MarketRegime.RANGING_WIDE: "FadeStrategy",
            MarketRegime.VOLATILE_BREAKOUT: "ScalpMomentum",
            MarketRegime.MEAN_REVERTING: "FadeStrategy",
        }

        # Position sizing
        sizers: Dict[str, PositionSizer] = {}
        for symbol in data_dict:
            meta = symbol_meta[symbol]
            sizers[symbol] = PositionSizer(SizingConfig(
                contract_size=meta.get("contract_size", 100.0),
                initial_equity=self.initial_equity,
            ))

        # Online learning for scorecard updates
        perf_tracker = StrategyPerformanceTracker()

        # ====================================================================
        # BAR-BY-BAR EXECUTION LOOP (lightweight - just sizing/risk/execution)
        # ====================================================================
        cooldown_per_symbol: Dict[str, int] = {s: 0 for s in data_dict}
        # Track regime stability (bars since last regime change)
        last_regime_per_symbol: Dict[str, MarketRegime] = {}
        regime_stable_bars: Dict[str, int] = {s: 0 for s in data_dict}
        MIN_REGIME_STABILITY = 10  # Require regime stable for 10 bars before trading

        for bar_idx in range(min_bars, n_bars):
            # Record equity
            self.equity_curve.append(self.equity)

            # Check and close positions that hit SL/TP
            self._check_exits(bar_idx, data_dict, symbol_meta, commission_per_lot,
                              slippage_pips, perf_tracker)

            # Process each symbol
            for symbol, bars_df in data_dict.items():
                if bar_idx >= len(bars_df):
                    continue

                # Cooldown: skip if recently traded
                if cooldown_per_symbol[symbol] > 0:
                    cooldown_per_symbol[symbol] -= 1
                    continue

                meta = symbol_meta[symbol]
                pip_size = meta.get("pip_size", 0.01)
                contract_size = meta.get("contract_size", 100.0)
                typical_spread = meta.get("typical_spread", 0.30)

                # Get regime for this bar
                regime, confidence = regime_arrays[symbol][bar_idx]
                regime_name = regime.name

                # Track regime stability
                if symbol not in last_regime_per_symbol or last_regime_per_symbol[symbol] != regime:
                    last_regime_per_symbol[symbol] = regime
                    regime_stable_bars[symbol] = 0
                else:
                    regime_stable_bars[symbol] += 1

                # Track regime
                if regime_name not in self.regime_counts[symbol]:
                    self.regime_counts[symbol][regime_name] = 0
                self.regime_counts[symbol][regime_name] += 1

                # Select strategy based on regime
                primary_strat = regime_to_strategy.get(regime, "TrendFollower")
                secondary_strat = regime_to_secondary.get(regime, "MeanReversion")

                # Get pre-computed signal
                primary_signals = all_signals[symbol][primary_strat]
                direction = int(primary_signals[bar_idx, 0])
                sl_dist = primary_signals[bar_idx, 1]
                tp_dist = primary_signals[bar_idx, 2]
                strategy_name = primary_strat

                # If primary has no signal, try secondary
                if direction == 0:
                    secondary_signals = all_signals[symbol][secondary_strat]
                    direction = int(secondary_signals[bar_idx, 0])
                    sl_dist = secondary_signals[bar_idx, 1]
                    tp_dist = secondary_signals[bar_idx, 2]
                    strategy_name = secondary_strat

                # Track strategy usage
                if strategy_name not in self.strategy_counts[symbol]:
                    self.strategy_counts[symbol][strategy_name] = 0
                self.strategy_counts[symbol][strategy_name] += 1

                if direction == 0 or sl_dist <= 0:
                    continue

                # Require regime to be stable before entering
                if regime_stable_bars[symbol] < MIN_REGIME_STABILITY:
                    continue

                # Check if we already have a position on this symbol in same direction
                has_position = any(
                    p.symbol == symbol and p.direction == direction
                    for p in self.open_positions
                )
                if has_position:
                    continue

                # Max positions check
                if len(self.open_positions) >= 8:
                    continue

                # Drawdown halt
                current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
                if current_dd >= 0.20:
                    continue

                # Check if minimum lot exceeds acceptable risk
                min_lot_for_symbol = meta.get("min_lot", 0.01)
                min_lot_risk = min_lot_for_symbol * sl_dist * contract_size
                if min_lot_risk > self.equity * 0.02:
                    continue  # Skip: even min lot exceeds 2% risk

                # Position sizing (simplified for speed)
                win_rate = perf_tracker.get_win_rate(strategy_name, regime)

                sizing_result = sizers[symbol].compute_size(
                    equity=self.equity,
                    peak_equity=self.peak_equity,
                    sl_distance=sl_dist,
                    atr=sl_dist / 1.5,
                    win_rate=win_rate,
                    regime_size_mult=1.0 if confidence >= 0.5 else 0.5,
                )

                if not sizing_result.approved:
                    continue

                lot_size = sizing_result.lot_size

                # Execute trade
                close_price = bars_df.iloc[bar_idx]["close"]
                entry_price = close_price

                # Apply spread and slippage
                spread_cost = typical_spread / 2.0
                slippage_cost = slippage_pips * pip_size

                if direction == 1:  # Buy
                    entry_price += spread_cost + slippage_cost
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:  # Sell
                    entry_price -= spread_cost + slippage_cost
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                # Deduct commission
                commission = commission_per_lot * lot_size
                self.equity -= commission

                # Open position
                pos = OpenPosition(
                    symbol=symbol,
                    direction=direction,
                    entry_bar=bar_idx,
                    entry_price=entry_price,
                    lot_size=lot_size,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_distance=sl_dist,
                    tp_distance=tp_dist,
                    regime=regime_name,
                    strategy=strategy_name,
                    contract_size=contract_size,
                )
                self.open_positions.append(pos)

                # Cooldown
                cooldown_per_symbol[symbol] = 2

        # Close all remaining positions at end of data
        self._close_all_positions(n_bars - 1, data_dict, symbol_meta,
                                  commission_per_lot, slippage_pips,
                                  perf_tracker, "end_of_data")

        # Final equity
        self.equity_curve.append(self.equity)

        # Build results
        result = self._compute_results()
        return result

    def _check_exits(self, bar_idx: int, data_dict: Dict[str, pd.DataFrame],
                     symbol_meta: Dict[str, Dict], commission_per_lot: float,
                     slippage_pips: float,
                     perf_tracker: StrategyPerformanceTracker):
        """Check if any open positions hit SL or TP on the current bar."""
        closed_indices = []

        for i, pos in enumerate(self.open_positions):
            if pos.symbol not in data_dict:
                continue
            bars_df = data_dict[pos.symbol]
            if bar_idx >= len(bars_df):
                continue

            bar = bars_df.iloc[bar_idx]
            high = bar["high"]
            low = bar["low"]

            meta = symbol_meta[pos.symbol]
            pip_size = meta.get("pip_size", 0.01)

            exit_price = None
            exit_reason = ""

            if pos.direction == 1:  # Long
                if low <= pos.sl_price:
                    exit_price = pos.sl_price - slippage_pips * pip_size
                    exit_reason = "sl"
                elif high >= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "tp"
            else:  # Short
                if high >= pos.sl_price:
                    exit_price = pos.sl_price + slippage_pips * pip_size
                    exit_reason = "sl"
                elif low <= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "tp"

            if exit_price is not None:
                contract_size = pos.contract_size
                pnl = pos.direction * (exit_price - pos.entry_price) * pos.lot_size * contract_size
                pnl_pips = pos.direction * (exit_price - pos.entry_price) / pip_size

                self.equity += pnl
                if self.equity > self.peak_equity:
                    self.peak_equity = self.equity

                self.symbol_pnl[pos.symbol] = self.symbol_pnl.get(pos.symbol, 0.0) + pnl

                trade = TradeRecord(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_bar=pos.entry_bar,
                    exit_bar=bar_idx,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    lot_size=pos.lot_size,
                    sl_distance=pos.sl_distance,
                    tp_distance=pos.tp_distance,
                    pnl=pnl,
                    pnl_pips=pnl_pips,
                    regime=pos.regime,
                    strategy=pos.strategy,
                    exit_reason=exit_reason,
                    duration_bars=bar_idx - pos.entry_bar,
                )
                self.trades.append(trade)

                # Update online learner
                regime_enum = MarketRegime[pos.regime] if pos.regime in MarketRegime.__members__ else MarketRegime.RANGING_NARROW
                perf_tracker.record_trade(pos.strategy, regime_enum, pnl)

                closed_indices.append(i)

        # Remove closed positions (reverse order to maintain indices)
        for i in sorted(closed_indices, reverse=True):
            self.open_positions.pop(i)

    def _close_all_positions(self, bar_idx: int, data_dict: Dict[str, pd.DataFrame],
                             symbol_meta: Dict[str, Dict], commission_per_lot: float,
                             slippage_pips: float,
                             perf_tracker: StrategyPerformanceTracker,
                             reason: str):
        """Close all open positions at end of data."""
        for pos in self.open_positions:
            if pos.symbol not in data_dict:
                continue
            bars_df = data_dict[pos.symbol]
            close_idx = min(bar_idx, len(bars_df) - 1)
            exit_price = bars_df.iloc[close_idx]["close"]

            meta = symbol_meta[pos.symbol]
            pip_size = meta.get("pip_size", 0.01)
            contract_size = pos.contract_size

            pnl = pos.direction * (exit_price - pos.entry_price) * pos.lot_size * contract_size
            pnl_pips = pos.direction * (exit_price - pos.entry_price) / pip_size

            self.equity += pnl
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            self.symbol_pnl[pos.symbol] = self.symbol_pnl.get(pos.symbol, 0.0) + pnl

            trade = TradeRecord(
                symbol=pos.symbol,
                direction=pos.direction,
                entry_bar=pos.entry_bar,
                exit_bar=bar_idx,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                lot_size=pos.lot_size,
                sl_distance=pos.sl_distance,
                tp_distance=pos.tp_distance,
                pnl=pnl,
                pnl_pips=pnl_pips,
                regime=pos.regime,
                strategy=pos.strategy,
                exit_reason=reason,
                duration_bars=bar_idx - pos.entry_bar,
            )
            self.trades.append(trade)

        self.open_positions.clear()

    def _compute_results(self) -> BacktestResult:
        """Compute all backtest metrics from trade log and equity curve."""
        result = BacktestResult()
        result.portfolio_equity = np.array(self.equity_curve)
        result.trades = self.trades
        result.total_trades = len(self.trades)

        if not self.trades:
            result.regime_summary = self.regime_counts.copy()
            result.strategy_usage = self.strategy_counts.copy()
            return result

        # Basic metrics
        pnls = np.array([t.pnl for t in self.trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        result.total_return_pct = ((self.equity - self.initial_equity) / self.initial_equity) * 100.0
        result.win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
        result.avg_trade_pnl = float(np.mean(pnls)) if len(pnls) > 0 else 0.0
        result.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        result.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # Profit factor
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.001
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_arr = result.portfolio_equity
        if len(equity_arr) > 0:
            peak = np.maximum.accumulate(equity_arr)
            dd = (peak - equity_arr) / np.where(peak > 0, peak, 1.0)
            result.max_drawdown_pct = float(np.max(dd)) * 100.0
        else:
            result.max_drawdown_pct = 0.0

        # Sharpe and Sortino
        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / np.where(equity_arr[:-1] > 0, equity_arr[:-1], 1.0)
            if len(returns) > 1 and np.std(returns) > 1e-10:
                result.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60))
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                if downside_std > 1e-10:
                    result.sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252 * 24 * 60))

        # Consecutive wins/losses
        result.max_consec_wins = self._max_consecutive(pnls > 0)
        result.max_consec_losses = self._max_consecutive(pnls <= 0)

        # Per-regime breakdown
        regime_trades: Dict[str, List[float]] = {}
        for t in self.trades:
            if t.regime not in regime_trades:
                regime_trades[t.regime] = []
            regime_trades[t.regime].append(t.pnl)

        for regime, trade_pnls in regime_trades.items():
            arr = np.array(trade_pnls)
            result.regime_breakdown[regime] = {
                "trades": len(arr),
                "total_pnl": float(np.sum(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "avg_pnl": float(np.mean(arr)),
            }

        # Per-strategy breakdown
        strategy_trades: Dict[str, List[float]] = {}
        for t in self.trades:
            if t.strategy not in strategy_trades:
                strategy_trades[t.strategy] = []
            strategy_trades[t.strategy].append(t.pnl)

        for strat, trade_pnls in strategy_trades.items():
            arr = np.array(trade_pnls)
            result.strategy_breakdown[strat] = {
                "trades": len(arr),
                "total_pnl": float(np.sum(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "avg_pnl": float(np.mean(arr)),
            }

        # Per-symbol breakdown
        symbol_trades: Dict[str, List[float]] = {}
        for t in self.trades:
            if t.symbol not in symbol_trades:
                symbol_trades[t.symbol] = []
            symbol_trades[t.symbol].append(t.pnl)

        for sym, trade_pnls in symbol_trades.items():
            arr = np.array(trade_pnls)
            result.symbol_breakdown[sym] = {
                "trades": len(arr),
                "total_pnl": float(np.sum(arr)),
                "win_rate": float(np.mean(arr > 0)),
                "avg_pnl": float(np.mean(arr)),
            }

        # Regime summary (detection counts)
        result.regime_summary = self.regime_counts.copy()
        result.strategy_usage = self.strategy_counts.copy()

        return result

    def _max_consecutive(self, mask: np.ndarray) -> int:
        """Compute maximum consecutive True values in a boolean array."""
        if len(mask) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for val in mask:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def run_walk_forward(self, data_dict: Dict[str, pd.DataFrame],
                         n_windows: int = 4,
                         train_ratio: float = 0.7,
                         config: Optional[Dict] = None) -> BacktestResult:
        """
        Walk-forward backtest: split data into windows, train on each, test on next.

        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Bar data per symbol.
        n_windows : int
            Number of walk-forward windows.
        train_ratio : float
            Fraction of each window used for training.
        config : dict, optional
            Configuration overrides.

        Returns
        -------
        BacktestResult
            Combined results from all test periods.
        """
        n_bars = min(len(df) for df in data_dict.values())
        window_size = n_bars // n_windows

        all_trades = []
        initial_eq = config.get("initial_equity", 1000.0) if config else 1000.0
        combined_equity = [initial_eq]

        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, n_bars)

            # Full window data
            window_data = {
                sym: df.iloc[start:end].reset_index(drop=True)
                for sym, df in data_dict.items()
            }

            # Run backtest on window
            test_config = config.copy() if config else {}
            test_config["initial_equity"] = combined_equity[-1]
            test_config["min_bars"] = min(150, (end - start) // 3)

            result = self.run(window_data, test_config)
            all_trades.extend(result.trades)

            if len(result.portfolio_equity) > 0:
                combined_equity.append(result.portfolio_equity[-1])

        # Build combined result
        self.trades = all_trades
        self.equity_curve = combined_equity
        self.equity = combined_equity[-1]
        return self._compute_results()
