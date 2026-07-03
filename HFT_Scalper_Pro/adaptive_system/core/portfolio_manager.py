"""
Portfolio Manager - Multi-Symbol Coordination
===============================================
Manages the multi-currency portfolio: symbol universe, capital allocation,
cross-symbol correlation, signal aggregation, and portfolio state tracking.

Features:
    1. Symbol universe management (add/remove symbols with metadata)
    2. Rolling correlation matrix (50-bar rolling between symbol returns)
    3. Capital allocation proportional to (confidence * inverse_correlation_penalty)
    4. Signal aggregation: collect signals from all symbols, pass through risk_manager
    5. Portfolio state tracking: open positions, P&L per symbol, total equity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from .regime_detector import MarketRegime, RegimeDetector
from .strategy_selector import StrategySelector
from .position_sizer import PositionSizer, SizingConfig, SizingResult
from .risk_manager import RiskManager, TradeProposal, RiskDecision, PortfolioPosition
from .online_learner import (
    StrategyPerformanceTracker,
    RegimeTransitionMatrix,
    MarketProfiler,
)


@dataclass
class SymbolConfig:
    """Configuration and metadata for a tradeable symbol."""
    symbol: str
    pip_size: float            # Size of one pip (e.g., 0.01 for gold, 0.0001 for forex)
    contract_size: float       # Contract size (100 for gold, 100000 for forex)
    typical_spread: float      # Average spread in price units
    min_lot: float = 0.01
    max_lot: float = 200.0
    lot_step: float = 0.01
    active: bool = True
    # Symbol-specific session hours (UTC)
    active_hours: List[int] = field(default_factory=lambda: list(range(1, 22)))
    # Blackout hours
    blackout_hours: List[int] = field(default_factory=lambda: [23, 0])


@dataclass
class SymbolState:
    """Runtime state for a symbol."""
    symbol: str
    current_regime: MarketRegime = MarketRegime.RANGING_NARROW
    regime_confidence: float = 0.5
    current_strategy: str = ""
    open_positions: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    last_signal_bar: int = -999
    # Recent returns for correlation computation
    recent_returns: List[float] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Overall portfolio state."""
    equity: float = 1000.0
    peak_equity: float = 1000.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    open_position_count: int = 0
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    # Per-symbol states
    symbol_states: Dict[str, SymbolState] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Aggregated trade signal ready for execution."""
    symbol: str
    direction: int
    lot_size: float
    sl_distance: float
    tp_distance: float
    strategy_name: str
    regime: MarketRegime
    regime_confidence: float
    risk_decision: Optional[RiskDecision] = None


class PortfolioManager:
    """
    Multi-symbol portfolio coordination engine.

    Manages the full lifecycle:
    1. Maintains symbol universe with metadata
    2. Detects regime per symbol
    3. Selects strategy per symbol based on regime
    4. Generates and aggregates signals
    5. Sizes positions with portfolio-level awareness
    6. Passes through risk manager for final approval
    7. Tracks P&L and updates online learner

    Usage:
        pm = PortfolioManager(initial_equity=1000.0)
        pm.add_symbol(SymbolConfig(symbol="XAUUSD", pip_size=0.01, contract_size=100, typical_spread=0.30))
        pm.add_symbol(SymbolConfig(symbol="EURUSD", pip_size=0.0001, contract_size=100000, typical_spread=0.00012))

        signals = pm.process_bars({"XAUUSD": bars_df, "EURUSD": bars_df2})
        for signal in signals:
            if signal.risk_decision and signal.risk_decision.approved:
                execute_trade(signal)
    """

    def __init__(
        self,
        initial_equity: float = 1000.0,
        risk_manager: Optional[RiskManager] = None,
        strategy_selector: Optional[StrategySelector] = None,
    ):
        self.state = PortfolioState(equity=initial_equity, peak_equity=initial_equity)

        # Core components
        self.risk_manager = risk_manager or RiskManager()
        self.strategy_selector = strategy_selector or StrategySelector()

        # Per-symbol components
        self._symbol_configs: Dict[str, SymbolConfig] = {}
        self._regime_detectors: Dict[str, RegimeDetector] = {}
        self._position_sizers: Dict[str, PositionSizer] = {}
        self._market_profilers: Dict[str, MarketProfiler] = {}

        # Portfolio-level learning
        self._performance_tracker = StrategyPerformanceTracker()
        self._transition_matrix = RegimeTransitionMatrix()

        # Correlation tracking
        self._correlation_window: int = 50
        self._symbol_returns: Dict[str, List[float]] = defaultdict(list)
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}

        # Open positions tracking
        self._open_positions: List[PortfolioPosition] = []

    def add_symbol(self, config: SymbolConfig):
        """
        Add a symbol to the trading universe.

        Parameters
        ----------
        config : SymbolConfig
            Symbol configuration and metadata.
        """
        symbol = config.symbol
        self._symbol_configs[symbol] = config
        self._regime_detectors[symbol] = RegimeDetector()
        self._position_sizers[symbol] = PositionSizer(SizingConfig(
            contract_size=config.contract_size,
        ))
        self._market_profilers[symbol] = MarketProfiler()
        self.state.symbol_states[symbol] = SymbolState(symbol=symbol)

    def remove_symbol(self, symbol: str):
        """Remove a symbol from the trading universe."""
        self._symbol_configs.pop(symbol, None)
        self._regime_detectors.pop(symbol, None)
        self._position_sizers.pop(symbol, None)
        self._market_profilers.pop(symbol, None)
        self.state.symbol_states.pop(symbol, None)
        self._symbol_returns.pop(symbol, None)

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        return [
            sym for sym, cfg in self._symbol_configs.items()
            if cfg.active
        ]

    def process_bars(self, bars_by_symbol: Dict[str, pd.DataFrame],
                     current_hour: int = 12) -> List[TradeSignal]:
        """
        Process new bars for all symbols and generate approved trade signals.

        This is the main entry point called on each bar update.

        Parameters
        ----------
        bars_by_symbol : Dict[str, pd.DataFrame]
            Bar data for each symbol. Keys are symbol names.
            Each DataFrame must have: open, high, low, close, and volume/tick_count.
        current_hour : int
            Current hour (UTC) for session filtering.

        Returns
        -------
        List[TradeSignal]
            List of approved trade signals ready for execution.
        """
        signals = []

        # Update correlation matrix
        self._update_correlations(bars_by_symbol)

        # Update risk manager correlations
        self.risk_manager.update_correlations(self._correlation_matrix)

        # Process each symbol
        for symbol in self.get_active_symbols():
            if symbol not in bars_by_symbol:
                continue

            bars = bars_by_symbol[symbol]
            if len(bars) < 100:
                continue  # Not enough data

            config = self._symbol_configs[symbol]
            sym_state = self.state.symbol_states[symbol]

            # Check if symbol is in blackout
            if current_hour in config.blackout_hours:
                continue

            # Step 1: Detect regime
            high = bars["high"].values.astype(np.float64)
            low = bars["low"].values.astype(np.float64)
            close = bars["close"].values.astype(np.float64)

            detector = self._regime_detectors[symbol]
            regime, confidence = detector.detect(high, low, close)
            sym_state.current_regime = regime
            sym_state.regime_confidence = confidence

            # Track regime transitions
            self._transition_matrix.observe(regime)

            # Step 2: Select strategy for this regime
            strategy, size_mult = self.strategy_selector.select(regime, confidence)
            sym_state.current_strategy = strategy.get_name()

            # Step 3: Generate signals
            raw_signals = strategy.generate_signals(bars)

            # Check for signal on latest bar
            latest_idx = len(bars) - 1
            direction = raw_signals[latest_idx, 0]
            sl_dist = raw_signals[latest_idx, 1]
            tp_dist = raw_signals[latest_idx, 2]

            if direction == 0 or sl_dist <= 0:
                continue  # No signal

            # Step 4: Position sizing
            sizer = self._position_sizers[symbol]

            # Get performance stats for Kelly
            perf_stats = self._performance_tracker.get_pnl_stats(
                strategy.get_name(), regime
            )
            win_rate = self._performance_tracker.get_win_rate(
                strategy.get_name(), regime
            )
            avg_win = max(perf_stats.mean if perf_stats.mean > 0 else tp_dist, 0.01)
            avg_loss = max(perf_stats.std() if perf_stats.std() > 0 else sl_dist, 0.01)

            # Existing risk on this symbol
            existing_symbol_risk = self._compute_symbol_risk(symbol)
            existing_total_risk = self._compute_total_risk()

            # Compute correlation with portfolio
            max_corr = self._get_max_correlation(symbol)

            sizing_result = sizer.compute_size(
                equity=self.state.equity,
                peak_equity=self.state.peak_equity,
                sl_distance=sl_dist,
                atr=sl_dist / 1.5,  # Approximate ATR from SL
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                consec_wins=0,
                consec_losses=0,
                correlation_with_portfolio=max_corr,
                regime_size_mult=size_mult,
                existing_symbol_risk=existing_symbol_risk,
                existing_total_risk=existing_total_risk,
            )

            if not sizing_result.approved:
                continue

            lot_size = sizing_result.lot_size

            # Step 5: Risk manager approval
            proposal = TradeProposal(
                symbol=symbol,
                direction=int(direction),
                lot_size=lot_size,
                sl_distance=sl_dist,
                tp_distance=tp_dist,
                contract_size=config.contract_size,
                strategy_name=strategy.get_name(),
                hour=current_hour,
            )

            risk_decision = self.risk_manager.approve_trade(
                proposal=proposal,
                current_positions=self._open_positions,
                equity=self.state.equity,
                peak_equity=self.state.peak_equity,
                daily_pnl=self.state.daily_pnl,
            )

            # Build trade signal
            trade_signal = TradeSignal(
                symbol=symbol,
                direction=int(direction),
                lot_size=risk_decision.adjusted_lot_size if risk_decision.approved else 0.0,
                sl_distance=sl_dist,
                tp_distance=tp_dist,
                strategy_name=strategy.get_name(),
                regime=regime,
                regime_confidence=confidence,
                risk_decision=risk_decision,
            )

            if risk_decision.approved:
                signals.append(trade_signal)
                sym_state.last_signal_bar = latest_idx

        return signals

    def record_trade_open(self, symbol: str, direction: int, lot_size: float,
                          entry_price: float, sl_distance: float,
                          strategy_name: str = ""):
        """Record a new position being opened."""
        config = self._symbol_configs.get(symbol)
        contract_size = config.contract_size if config else 100.0

        position = PortfolioPosition(
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            entry_price=entry_price,
            sl_distance=sl_distance,
            contract_size=contract_size,
            strategy_name=strategy_name,
        )
        self._open_positions.append(position)
        self.state.open_position_count = len(self._open_positions)

        sym_state = self.state.symbol_states.get(symbol)
        if sym_state:
            sym_state.open_positions += 1
            sym_state.total_trades += 1

    def record_trade_close(self, symbol: str, direction: int, pnl: float,
                           strategy_name: str = "", regime: Optional[MarketRegime] = None):
        """
        Record a position being closed and update learning.

        Parameters
        ----------
        symbol : str
            Symbol of the closed trade.
        direction : int
            Direction of the closed trade.
        pnl : float
            Realized P&L.
        strategy_name : str
            Strategy that produced the trade.
        regime : MarketRegime, optional
            Regime when trade was entered.
        """
        # Remove from open positions
        for i, pos in enumerate(self._open_positions):
            if pos.symbol == symbol and pos.direction == direction:
                self._open_positions.pop(i)
                break

        self.state.open_position_count = len(self._open_positions)
        self.state.total_realized_pnl += pnl
        self.state.daily_pnl += pnl

        # Update equity
        self.state.equity += pnl
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        # Update drawdown
        if self.state.peak_equity > 0:
            self.state.drawdown_pct = (
                (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            )

        # Update symbol state
        sym_state = self.state.symbol_states.get(symbol)
        if sym_state:
            sym_state.open_positions = max(0, sym_state.open_positions - 1)
            sym_state.realized_pnl += pnl
            if pnl > 0:
                sym_state.winning_trades += 1

        # Update online learner
        if strategy_name and regime:
            self._performance_tracker.record_trade(strategy_name, regime, pnl)
            self.strategy_selector.record_result(strategy_name, regime, pnl)

        # Update risk manager daily PnL
        self.risk_manager.record_daily_pnl(pnl)

    def reset_daily(self):
        """Reset daily tracking (call at start of each trading day)."""
        self.state.daily_pnl = 0.0
        self.risk_manager.reset_daily_pnl()

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        return self.state

    def get_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get the current cross-symbol correlation matrix."""
        return self._correlation_matrix.copy()

    def get_capital_allocation(self) -> Dict[str, float]:
        """
        Compute capital allocation weights per symbol.

        Allocation proportional to:
            strategy_confidence * inverse_correlation_penalty

        Returns
        -------
        Dict[str, float]
            Normalized allocation weights (sum to 1.0).
        """
        active_symbols = self.get_active_symbols()
        if not active_symbols:
            return {}

        raw_weights = {}
        for symbol in active_symbols:
            sym_state = self.state.symbol_states.get(symbol)
            if sym_state is None:
                continue

            confidence = sym_state.regime_confidence
            # Inverse correlation penalty: reduce allocation for highly correlated symbols
            max_corr = self._get_max_correlation(symbol)
            corr_penalty = 1.0 - 0.5 * max_corr  # 0.5 to 1.0

            weight = confidence * corr_penalty
            raw_weights[symbol] = max(0.1, weight)  # Floor at 0.1

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            return {sym: w / total for sym, w in raw_weights.items()}
        else:
            n = len(active_symbols)
            return {sym: 1.0 / n for sym in active_symbols}

    def _update_correlations(self, bars_by_symbol: Dict[str, pd.DataFrame]):
        """Update rolling correlation matrix from latest returns."""
        # Collect latest return for each symbol
        for symbol, bars in bars_by_symbol.items():
            if len(bars) < 2:
                continue
            close = bars["close"].values
            ret = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0.0
            self._symbol_returns[symbol].append(ret)
            # Keep only last N returns
            if len(self._symbol_returns[symbol]) > self._correlation_window:
                self._symbol_returns[symbol] = self._symbol_returns[symbol][-self._correlation_window:]

        # Compute pairwise correlations
        symbols = list(self._symbol_returns.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]
                returns_a = self._symbol_returns[sym_a]
                returns_b = self._symbol_returns[sym_b]

                # Need at least 20 common data points
                min_len = min(len(returns_a), len(returns_b))
                if min_len < 20:
                    continue

                arr_a = np.array(returns_a[-min_len:])
                arr_b = np.array(returns_b[-min_len:])

                std_a = np.std(arr_a)
                std_b = np.std(arr_b)
                if std_a < 1e-10 or std_b < 1e-10:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0

                self._correlation_matrix[(sym_a, sym_b)] = corr
                self._correlation_matrix[(sym_b, sym_a)] = corr

    def _get_max_correlation(self, symbol: str) -> float:
        """Get maximum correlation of a symbol with any currently held position."""
        max_corr = 0.0
        for pos in self._open_positions:
            if pos.symbol == symbol:
                continue
            key = (symbol, pos.symbol)
            corr = abs(self._correlation_matrix.get(key, 0.0))
            max_corr = max(max_corr, corr)
        return max_corr

    def _compute_symbol_risk(self, symbol: str) -> float:
        """Compute current risk allocated to a symbol."""
        if self.state.equity <= 0:
            return 0.0
        total = 0.0
        for pos in self._open_positions:
            if pos.symbol == symbol:
                total += (pos.lot_size * pos.sl_distance * pos.contract_size) / self.state.equity
        return total

    def _compute_total_risk(self) -> float:
        """Compute total portfolio risk."""
        if self.state.equity <= 0:
            return 0.0
        total = 0.0
        for pos in self._open_positions:
            total += (pos.lot_size * pos.sl_distance * pos.contract_size) / self.state.equity
        return total
