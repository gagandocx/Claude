"""
Portfolio Risk Management
==========================
Comprehensive risk management at the portfolio level.

Features:
    1. Per-symbol maximum exposure check
    2. Total portfolio exposure check
    3. Correlation-based position limiting
    4. Drawdown circuit breaker (halt all trading)
    5. Intraday loss limit
    6. Session blackout hours per symbol (configurable)
    7. Maximum concurrent positions (configurable, default 6)
    8. approve_trade(proposed_trade, current_portfolio) -> (approved, adjusted_size)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Exposure limits
    max_risk_per_symbol: float = 0.05        # 5% max risk on any single symbol
    max_total_risk: float = 0.15             # 15% total portfolio risk
    max_concurrent_positions: int = 6         # Max open positions across all symbols

    # Correlation limits
    max_correlation: float = 0.7              # If new position corr > this with existing, reduce
    correlation_reduction_factor: float = 0.5 # Reduce size by this factor when correlated

    # Drawdown management
    dd_halt_threshold: float = 0.20           # Halt ALL trading at 20% drawdown
    dd_reduce_threshold: float = 0.10         # Start reducing size at 10% drawdown
    dd_reduce_factor: float = 0.5             # Reduce by this factor at dd_reduce_threshold

    # Intraday loss limit
    daily_loss_limit: float = 0.05            # Stop trading after 5% daily loss
    daily_loss_reset_hour: int = 0            # UTC hour to reset daily P&L tracking

    # Session blackout (hours when trading is restricted per symbol)
    # Dict of symbol -> list of blackout hours (UTC)
    session_blackouts: Dict[str, List[int]] = field(default_factory=dict)

    # Default blackout hours (applied to all symbols if not overridden)
    default_blackout_hours: List[int] = field(default_factory=lambda: [23, 0])

    # Position size floor (minimum to be worth trading)
    min_lot_size: float = 0.01


@dataclass
class TradeProposal:
    """Proposed trade for risk approval."""
    symbol: str
    direction: int          # 1=buy, -1=sell
    lot_size: float
    sl_distance: float      # Stop distance in price units
    tp_distance: float      # Take profit distance
    contract_size: float    # Contract size for this symbol
    strategy_name: str = ""
    entry_price: float = 0.0
    hour: int = 12          # Current hour (UTC) for blackout check


@dataclass
class RiskDecision:
    """Risk manager decision on a trade proposal."""
    approved: bool = True
    adjusted_lot_size: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0=no risk, 1=max risk
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioPosition:
    """Represents an open position in the portfolio."""
    symbol: str
    direction: int
    lot_size: float
    entry_price: float
    sl_distance: float
    contract_size: float
    unrealized_pnl: float = 0.0
    strategy_name: str = ""


class RiskManager:
    """
    Portfolio-level risk management engine.

    Evaluates trade proposals against the current portfolio state and
    risk limits, returning approval/rejection decisions with adjusted sizes.

    Usage:
        risk_mgr = RiskManager(RiskConfig())
        decision = risk_mgr.approve_trade(proposal, portfolio_state)
        if decision.approved:
            execute_trade(decision.adjusted_lot_size)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        # Track daily P&L
        self._daily_pnl: float = 0.0
        self._daily_pnl_start_equity: float = 0.0
        self._last_reset_day: int = -1
        # Correlation matrix (symbol_i, symbol_j) -> correlation
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}

    def approve_trade(
        self,
        proposal: TradeProposal,
        current_positions: List[PortfolioPosition],
        equity: float,
        peak_equity: float,
        daily_pnl: float = 0.0,
    ) -> RiskDecision:
        """
        Evaluate a trade proposal against all risk limits.

        Parameters
        ----------
        proposal : TradeProposal
            The proposed trade.
        current_positions : List[PortfolioPosition]
            All currently open positions.
        equity : float
            Current account equity.
        peak_equity : float
            Peak equity (for drawdown calculation).
        daily_pnl : float
            Today's realized + unrealized P&L.

        Returns
        -------
        RiskDecision
            Approval decision with adjusted size and reasons.
        """
        cfg = self.config
        decision = RiskDecision()
        decision.adjusted_lot_size = proposal.lot_size
        rejection_reasons = []
        warnings = []

        # --- Check 1: Drawdown circuit breaker ---
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if current_dd >= cfg.dd_halt_threshold:
            rejection_reasons.append(
                f"Drawdown halt: {current_dd:.2%} >= {cfg.dd_halt_threshold:.2%}"
            )
            decision.approved = False
            decision.rejection_reasons = rejection_reasons
            return decision

        # Drawdown reduction (partial)
        dd_reduction = 1.0
        if current_dd >= cfg.dd_reduce_threshold:
            # Linear interpolation from reduce_threshold to halt_threshold
            dd_severity = (current_dd - cfg.dd_reduce_threshold) / (
                cfg.dd_halt_threshold - cfg.dd_reduce_threshold
            )
            dd_reduction = 1.0 - dd_severity * (1.0 - cfg.dd_reduce_factor)
            dd_reduction = max(cfg.dd_reduce_factor, dd_reduction)
            decision.adjusted_lot_size *= dd_reduction
            warnings.append(f"DD reduction applied: {dd_reduction:.2f}x")

        # --- Check 2: Daily loss limit ---
        if equity > 0:
            daily_loss_pct = -daily_pnl / equity if daily_pnl < 0 else 0.0
            if daily_loss_pct >= cfg.daily_loss_limit:
                rejection_reasons.append(
                    f"Daily loss limit: {daily_loss_pct:.2%} >= {cfg.daily_loss_limit:.2%}"
                )
                decision.approved = False
                decision.rejection_reasons = rejection_reasons
                return decision

        # --- Check 3: Maximum concurrent positions ---
        if len(current_positions) >= cfg.max_concurrent_positions:
            rejection_reasons.append(
                f"Max positions reached: {len(current_positions)} >= {cfg.max_concurrent_positions}"
            )
            decision.approved = False
            decision.rejection_reasons = rejection_reasons
            return decision

        # --- Check 4: Session blackout ---
        blackout_hours = cfg.session_blackouts.get(
            proposal.symbol, cfg.default_blackout_hours
        )
        if proposal.hour in blackout_hours:
            rejection_reasons.append(
                f"Session blackout: hour {proposal.hour} for {proposal.symbol}"
            )
            decision.approved = False
            decision.rejection_reasons = rejection_reasons
            return decision

        # --- Check 5: Per-symbol exposure ---
        symbol_risk = self._compute_symbol_risk(
            proposal.symbol, current_positions, equity
        )
        new_trade_risk = self._compute_trade_risk(proposal, equity)
        total_symbol_risk = symbol_risk + new_trade_risk

        if total_symbol_risk > cfg.max_risk_per_symbol:
            # Try to reduce size to fit
            available_risk = max(0.0, cfg.max_risk_per_symbol - symbol_risk)
            if available_risk < 0.001:
                rejection_reasons.append(
                    f"Symbol risk limit: {proposal.symbol} at {symbol_risk:.2%}"
                )
                decision.approved = False
                decision.rejection_reasons = rejection_reasons
                return decision
            else:
                # Scale down to fit
                scale = available_risk / new_trade_risk if new_trade_risk > 0 else 0.0
                decision.adjusted_lot_size *= scale
                warnings.append(f"Reduced for symbol limit: {scale:.2f}x")

        # --- Check 6: Total portfolio exposure ---
        total_risk = self._compute_total_risk(current_positions, equity)
        if total_risk + new_trade_risk > cfg.max_total_risk:
            available_total = max(0.0, cfg.max_total_risk - total_risk)
            if available_total < 0.001:
                rejection_reasons.append(
                    f"Total portfolio risk limit: {total_risk:.2%} >= {cfg.max_total_risk:.2%}"
                )
                decision.approved = False
                decision.rejection_reasons = rejection_reasons
                return decision
            else:
                scale = available_total / new_trade_risk if new_trade_risk > 0 else 0.0
                decision.adjusted_lot_size = min(decision.adjusted_lot_size,
                                                  proposal.lot_size * scale)
                warnings.append(f"Reduced for portfolio limit: {scale:.2f}x")

        # --- Check 7: Correlation limit ---
        max_corr = self._compute_max_correlation(proposal.symbol, current_positions)
        if max_corr > cfg.max_correlation:
            decision.adjusted_lot_size *= cfg.correlation_reduction_factor
            warnings.append(
                f"Correlation reduction: corr={max_corr:.2f} with existing position"
            )

        # --- Final checks ---
        # Ensure minimum lot size
        if decision.adjusted_lot_size < cfg.min_lot_size:
            if rejection_reasons:
                decision.approved = False
            else:
                decision.adjusted_lot_size = cfg.min_lot_size
                warnings.append("Size floored to minimum lot")

        # Round lot size
        decision.adjusted_lot_size = round(decision.adjusted_lot_size, 2)

        # Compute risk score
        decision.risk_score = self._compute_risk_score(
            current_dd, total_risk, len(current_positions), max_corr
        )

        decision.rejection_reasons = rejection_reasons
        decision.warnings = warnings
        decision.approved = len(rejection_reasons) == 0
        return decision

    def update_correlations(self, correlation_matrix: Dict[Tuple[str, str], float]):
        """
        Update the correlation matrix between symbols.

        Parameters
        ----------
        correlation_matrix : Dict[Tuple[str, str], float]
            Mapping of (symbol_a, symbol_b) -> correlation coefficient.
        """
        self._correlation_matrix = correlation_matrix

    def record_daily_pnl(self, pnl_change: float):
        """Record a P&L change for daily tracking."""
        self._daily_pnl += pnl_change

    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of new trading day)."""
        self._daily_pnl = 0.0

    def get_daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._daily_pnl

    def _compute_symbol_risk(self, symbol: str,
                             positions: List[PortfolioPosition],
                             equity: float) -> float:
        """Compute total risk allocated to a symbol."""
        if equity <= 0:
            return 0.0
        total_risk = 0.0
        for pos in positions:
            if pos.symbol == symbol:
                risk = (pos.lot_size * pos.sl_distance * pos.contract_size) / equity
                total_risk += risk
        return total_risk

    def _compute_trade_risk(self, proposal: TradeProposal, equity: float) -> float:
        """Compute risk of a single proposed trade."""
        if equity <= 0:
            return 0.0
        return (proposal.lot_size * proposal.sl_distance * proposal.contract_size) / equity

    def _compute_total_risk(self, positions: List[PortfolioPosition],
                            equity: float) -> float:
        """Compute total portfolio risk."""
        if equity <= 0:
            return 0.0
        total = 0.0
        for pos in positions:
            risk = (pos.lot_size * pos.sl_distance * pos.contract_size) / equity
            total += risk
        return total

    def _compute_max_correlation(self, symbol: str,
                                 positions: List[PortfolioPosition]) -> float:
        """Find maximum correlation between new symbol and existing positions."""
        max_corr = 0.0
        for pos in positions:
            if pos.symbol == symbol:
                # Same symbol = correlation 1.0
                max_corr = 1.0
                break
            # Check correlation matrix
            key1 = (symbol, pos.symbol)
            key2 = (pos.symbol, symbol)
            corr = self._correlation_matrix.get(key1,
                   self._correlation_matrix.get(key2, 0.0))
            max_corr = max(max_corr, abs(corr))
        return max_corr

    def _compute_risk_score(self, drawdown: float, total_risk: float,
                            num_positions: int, max_correlation: float) -> float:
        """
        Compute overall risk score (0=safe, 1=maximum risk).
        Used for dashboards and alerting.
        """
        cfg = self.config
        score = 0.0

        # Drawdown component (0-0.35)
        score += 0.35 * min(1.0, drawdown / cfg.dd_halt_threshold)

        # Total risk component (0-0.25)
        score += 0.25 * min(1.0, total_risk / cfg.max_total_risk)

        # Position count component (0-0.2)
        score += 0.2 * min(1.0, num_positions / cfg.max_concurrent_positions)

        # Correlation component (0-0.2)
        score += 0.2 * min(1.0, max_correlation)

        return min(1.0, score)
