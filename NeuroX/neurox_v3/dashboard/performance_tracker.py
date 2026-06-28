"""
=============================================================
  Performance Tracker - Professional Prop Trading Analytics
  
  Tracks all trades in real-time and computes institutional-grade
  performance metrics: Sharpe ratio, Sortino ratio, profit factor,
  max drawdown, per-model breakdown, and per-regime analysis.
  
  Designed like a top-tier prop desk dashboard where every metric
  matters for evaluating a smart trading system's edge.
=============================================================
"""

import math
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete record of a single trade for performance analysis."""
    trade_id: str
    entry_time: str                   # ISO format timestamp
    exit_time: str                    # ISO format timestamp
    direction: str                    # BUY or SELL
    pnl: float                        # Realized profit/loss
    model: str                        # transformer / lstm / gradient_boost / ensemble
    regime: str                       # trending / ranging / volatile / crash
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    confidence: float = 0.0
    hold_bars: int = 0


class PerformanceTracker:
    """
    Professional-grade performance tracking system.
    
    Maintains a running ledger of all executed trades and computes
    real-time risk-adjusted performance metrics used by institutional
    trading desks:
    
    - Sharpe Ratio (annualized): risk-adjusted returns vs risk-free
    - Sortino Ratio: penalizes only downside volatility
    - Profit Factor: gross wins / gross losses (must be > 1.0 to survive)
    - Max Drawdown: peak-to-trough equity curve decline
    - Win Rate, Expectancy, Recovery Factor, Consecutive stats
    - Per-Model Breakdown: which model generates the best edge
    - Per-Regime Breakdown: how strategy performs in each market state
    
    These are the metrics that separate profitable traders from noise.
    """

    # Annualization factor (approximate trading days in a year)
    TRADING_DAYS_PER_YEAR = 252
    # Assume average trades-per-day for annualization when using trade returns
    DEFAULT_TRADES_PER_DAY = 3
    # Risk-free rate (annualized) for Sharpe calculation
    RISK_FREE_RATE = 0.05  # 5% (T-bill rate)

    def __init__(self, min_trades_for_stats: int = 10):
        """
        Initialize the performance tracker.
        
        Args:
            min_trades_for_stats: Minimum number of trades before computing
                                  meaningful statistics (avoids noise).
        """
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[float] = [0.0]  # Cumulative PnL
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0
        self._max_consecutive_wins: int = 0
        self._max_consecutive_losses: int = 0
        self._min_trades_for_stats: int = min_trades_for_stats

        # Per-model tracking
        self._model_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        # Per-regime tracking
        self._regime_trades: Dict[str, List[TradeRecord]] = defaultdict(list)

    @property
    def total_trades(self) -> int:
        """Total number of recorded trades."""
        return len(self._trades)

    @property
    def has_sufficient_data(self) -> bool:
        """Whether we have enough trades for meaningful stats."""
        return len(self._trades) >= self._min_trades_for_stats

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade and update all running metrics.
        
        This is the primary entry point. Call this each time a position
        is closed by MT5 confirmation.
        """
        self._trades.append(trade)

        # Update equity curve
        new_equity = self._equity_curve[-1] + trade.pnl
        self._equity_curve.append(new_equity)

        # Update peak and drawdown
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity
        current_dd = self._peak_equity - new_equity
        if current_dd > self._max_drawdown:
            self._max_drawdown = current_dd

        # Update consecutive counts
        if trade.pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            if self._consecutive_wins > self._max_consecutive_wins:
                self._max_consecutive_wins = self._consecutive_wins
        elif trade.pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            if self._consecutive_losses > self._max_consecutive_losses:
                self._max_consecutive_losses = self._consecutive_losses
        # pnl == 0 breaks both streaks
        else:
            self._consecutive_wins = 0
            self._consecutive_losses = 0

        # Per-model and per-regime tracking
        self._model_trades[trade.model].append(trade)
        self._regime_trades[trade.regime].append(trade)

        logger.debug(
            f"[PerfTracker] Recorded trade {trade.trade_id}: "
            f"PnL={trade.pnl:+.2f} model={trade.model} regime={trade.regime} "
            f"equity={new_equity:.2f}"
        )

    def record_trade_simple(self, trade_id: str, pnl: float, model: str,
                            regime: str, direction: str = "BUY",
                            entry_time: Optional[str] = None,
                            exit_time: Optional[str] = None,
                            hold_bars: int = 0,
                            confidence: float = 0.0) -> None:
        """
        Convenience method to record a trade without building TradeRecord manually.
        """
        now = datetime.now().isoformat()
        trade = TradeRecord(
            trade_id=trade_id,
            entry_time=entry_time or now,
            exit_time=exit_time or now,
            direction=direction,
            pnl=pnl,
            model=model,
            regime=regime,
            hold_bars=hold_bars,
            confidence=confidence,
        )
        self.record_trade(trade)

    # ─────────────────────────────────────────────
    #  CORE METRICS
    # ─────────────────────────────────────────────

    def get_pnl_series(self) -> List[float]:
        """Get the raw PnL series for all trades."""
        return [t.pnl for t in self._trades]

    def win_rate(self) -> float:
        """
        Win rate: percentage of trades that were profitable.
        Returns 0.0 if no trades.
        """
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.pnl > 0)
        return wins / len(self._trades)

    def loss_rate(self) -> float:
        """Loss rate: 1 - win_rate."""
        return 1.0 - self.win_rate()

    def profit_factor(self) -> float:
        """
        Profit factor = gross_wins / gross_losses.
        
        A profit factor > 1.0 means the system is profitable.
        Professional desks target > 1.5 for live deployment.
        Returns float('inf') if no losing trades, 0.0 if no trades.
        """
        if not self._trades:
            return 0.0
        gross_wins = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in self._trades if t.pnl < 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    def avg_win(self) -> float:
        """Average winning trade PnL."""
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    def avg_loss(self) -> float:
        """Average losing trade PnL (negative value)."""
        losses = [t.pnl for t in self._trades if t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    def expectancy(self) -> float:
        """
        Expectancy per trade = (win_rate * avg_win) + (loss_rate * avg_loss).
        
        This is the expected value per trade. Positive = edge exists.
        Professional traders live and die by this number.
        """
        wr = self.win_rate()
        lr = self.loss_rate()
        aw = self.avg_win()
        al = self.avg_loss()
        return (wr * aw) + (lr * al)

    def sharpe_ratio(self) -> float:
        """
        Annualized Sharpe Ratio.
        
        Sharpe = (mean_return - risk_free_per_trade) / std_returns * sqrt(N)
        where N = trades_per_year (annualization factor).
        
        Institutional benchmark: > 1.0 is acceptable, > 2.0 is excellent.
        Returns 0.0 if insufficient data or zero standard deviation.
        """
        if len(self._trades) < 2:
            return 0.0

        returns = [t.pnl for t in self._trades]
        mean_return = sum(returns) / len(returns)
        std_return = self._std(returns)

        if std_return == 0:
            return 0.0

        # Risk-free rate per trade
        trades_per_year = self.TRADING_DAYS_PER_YEAR * self.DEFAULT_TRADES_PER_DAY
        rf_per_trade = self.RISK_FREE_RATE / trades_per_year

        # Annualized Sharpe
        sharpe = (mean_return - rf_per_trade) / std_return
        annualized = sharpe * math.sqrt(trades_per_year)
        return annualized

    def sortino_ratio(self) -> float:
        """
        Annualized Sortino Ratio.
        
        Like Sharpe but only penalizes downside volatility.
        Professional traders prefer Sortino because upside volatility
        is desirable (big wins don't hurt the ratio).
        
        Sortino = (mean_return - rf) / downside_deviation * sqrt(N)
        """
        if len(self._trades) < 2:
            return 0.0

        returns = [t.pnl for t in self._trades]
        mean_return = sum(returns) / len(returns)

        # Downside deviation: std of returns below zero
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_dev = self._std(downside_returns)
        if downside_dev == 0:
            return 0.0

        trades_per_year = self.TRADING_DAYS_PER_YEAR * self.DEFAULT_TRADES_PER_DAY
        rf_per_trade = self.RISK_FREE_RATE / trades_per_year

        sortino = (mean_return - rf_per_trade) / downside_dev
        annualized = sortino * math.sqrt(trades_per_year)
        return annualized

    def max_drawdown(self) -> float:
        """
        Maximum drawdown: peak-to-trough equity curve decline.
        
        This is the single most important risk metric. Professional
        risk desks will shut down a strategy exceeding max DD limits.
        """
        return self._max_drawdown

    def max_drawdown_pct(self, account_balance: float = 10000.0) -> float:
        """Max drawdown as percentage of account balance."""
        if account_balance <= 0:
            return 0.0
        return (self._max_drawdown / account_balance) * 100.0

    def recovery_factor(self) -> float:
        """
        Recovery factor = net_profit / max_drawdown.
        
        How well the system recovers from drawdowns.
        Professional threshold: > 3.0 for live deployment.
        """
        net_profit = self._equity_curve[-1] if self._equity_curve else 0.0
        if self._max_drawdown == 0:
            return float('inf') if net_profit > 0 else 0.0
        return net_profit / self._max_drawdown

    def best_trade(self) -> float:
        """Largest single winning trade PnL."""
        if not self._trades:
            return 0.0
        return max(t.pnl for t in self._trades)

    def worst_trade(self) -> float:
        """Largest single losing trade PnL (most negative)."""
        if not self._trades:
            return 0.0
        return min(t.pnl for t in self._trades)

    def consecutive_wins(self) -> int:
        """Maximum consecutive winning streak."""
        return self._max_consecutive_wins

    def consecutive_losses(self) -> int:
        """Maximum consecutive losing streak."""
        return self._max_consecutive_losses

    def avg_hold_time(self) -> float:
        """
        Average hold time in hours.
        Parses entry/exit time ISO strings to compute duration.
        Returns 0.0 if no trades with valid timestamps.
        """
        durations = []
        for t in self._trades:
            try:
                entry = datetime.fromisoformat(t.entry_time)
                exit_ = datetime.fromisoformat(t.exit_time)
                duration_hours = (exit_ - entry).total_seconds() / 3600.0
                if duration_hours >= 0:
                    durations.append(duration_hours)
            except (ValueError, TypeError):
                continue
        return sum(durations) / len(durations) if durations else 0.0

    def net_profit(self) -> float:
        """Total net profit across all trades."""
        return self._equity_curve[-1] if len(self._equity_curve) > 1 else 0.0

    def total_wins(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self._trades if t.pnl > 0)

    def total_losses(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self._trades if t.pnl < 0)

    def payoff_ratio(self) -> float:
        """
        Payoff ratio = avg_win / abs(avg_loss).
        
        How much you win on average vs how much you lose.
        Combined with win rate, determines if the edge is real.
        """
        aw = self.avg_win()
        al = abs(self.avg_loss())
        if al == 0:
            return float('inf') if aw > 0 else 0.0
        return aw / al

    def calmar_ratio(self, account_balance: float = 10000.0) -> float:
        """
        Calmar Ratio = annualized_return / max_drawdown.
        
        Measures return efficiency relative to worst-case risk.
        Professional benchmark: > 1.0.
        """
        if self._max_drawdown == 0 or not self._trades:
            return 0.0
        annualized_return = self.net_profit() / account_balance * 100
        dd_pct = self.max_drawdown_pct(account_balance)
        if dd_pct == 0:
            return 0.0
        return annualized_return / dd_pct

    # ─────────────────────────────────────────────
    #  PER-MODEL BREAKDOWN
    # ─────────────────────────────────────────────

    def get_model_stats(self, model: str) -> Dict:
        """
        Get performance statistics for a specific model.
        
        Args:
            model: One of 'transformer', 'lstm', 'gradient_boost', 'ensemble'
            
        Returns:
            Dict with win_rate, profit_factor, avg_pnl, trade_count, expectancy
        """
        trades = self._model_trades.get(model, [])
        if not trades:
            return {
                "model": model,
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "expectancy": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))

        wr = len(wins) / len(trades)
        pf = gross_wins / gross_losses if gross_losses > 0 else (
            float('inf') if gross_wins > 0 else 0.0
        )
        avg_w = gross_wins / len(wins) if wins else 0.0
        avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        exp = (wr * avg_w) + ((1 - wr) * avg_l)

        return {
            "model": model,
            "trade_count": len(trades),
            "win_rate": wr,
            "profit_factor": pf,
            "avg_pnl": sum(t.pnl for t in trades) / len(trades),
            "total_pnl": sum(t.pnl for t in trades),
            "expectancy": exp,
            "best_trade": max(t.pnl for t in trades),
            "worst_trade": min(t.pnl for t in trades),
        }

    def get_all_model_stats(self) -> Dict[str, Dict]:
        """Get stats for all tracked models."""
        all_models = ["transformer", "lstm", "gradient_boost", "ensemble"]
        result = {}
        for model in all_models:
            result[model] = self.get_model_stats(model)
        # Also include any other models that appeared
        for model in self._model_trades:
            if model not in result:
                result[model] = self.get_model_stats(model)
        return result

    # ─────────────────────────────────────────────
    #  PER-REGIME BREAKDOWN
    # ─────────────────────────────────────────────

    def get_regime_stats(self, regime: str) -> Dict:
        """
        Get performance statistics for a specific market regime.
        
        Args:
            regime: One of 'trending', 'ranging', 'volatile', 'crash'
            
        Returns:
            Dict with win_rate, profit_factor, avg_pnl, trade_count
        """
        trades = self._regime_trades.get(regime, [])
        if not trades:
            return {
                "regime": regime,
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "expectancy": 0.0,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))

        wr = len(wins) / len(trades)
        pf = gross_wins / gross_losses if gross_losses > 0 else (
            float('inf') if gross_wins > 0 else 0.0
        )
        avg_w = gross_wins / len(wins) if wins else 0.0
        avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        exp = (wr * avg_w) + ((1 - wr) * avg_l)

        return {
            "regime": regime,
            "trade_count": len(trades),
            "win_rate": wr,
            "profit_factor": pf,
            "avg_pnl": sum(t.pnl for t in trades) / len(trades),
            "total_pnl": sum(t.pnl for t in trades),
            "expectancy": exp,
        }

    def get_all_regime_stats(self) -> Dict[str, Dict]:
        """Get stats for all tracked regimes."""
        all_regimes = ["trending", "ranging", "volatile", "crash"]
        result = {}
        for regime in all_regimes:
            result[regime] = self.get_regime_stats(regime)
        # Include any other regimes
        for regime in self._regime_trades:
            if regime not in result:
                result[regime] = self.get_regime_stats(regime)
        return result

    # ─────────────────────────────────────────────
    #  FULL SUMMARY
    # ─────────────────────────────────────────────

    def get_full_summary(self) -> Dict:
        """
        Get complete performance summary - all metrics in one dict.
        This is what the dashboard renderer uses.
        """
        return {
            "total_trades": self.total_trades,
            "net_profit": self.net_profit(),
            "win_rate": self.win_rate(),
            "loss_rate": self.loss_rate(),
            "profit_factor": self.profit_factor(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "recovery_factor": self.recovery_factor(),
            "expectancy": self.expectancy(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "payoff_ratio": self.payoff_ratio(),
            "best_trade": self.best_trade(),
            "worst_trade": self.worst_trade(),
            "consecutive_wins": self.consecutive_wins(),
            "consecutive_losses": self.consecutive_losses(),
            "avg_hold_time_hours": self.avg_hold_time(),
            "total_wins": self.total_wins(),
            "total_losses": self.total_losses(),
            "equity_curve": self._equity_curve,
            "per_model": self.get_all_model_stats(),
            "per_regime": self.get_all_regime_stats(),
        }

    # ─────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────

    @staticmethod
    def _std(values: List[float]) -> float:
        """Population standard deviation."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return math.sqrt(variance)

    def reset(self) -> None:
        """Reset all tracked data. Use for testing or strategy restart."""
        self._trades.clear()
        self._equity_curve = [0.0]
        self._peak_equity = 0.0
        self._max_drawdown = 0.0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
        self._model_trades.clear()
        self._regime_trades.clear()
