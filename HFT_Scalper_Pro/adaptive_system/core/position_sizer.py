"""
Adaptive Position Sizing Engine
=================================
Advanced position sizing that adapts to market conditions, portfolio state,
and equity curve dynamics.

Features:
    1. Kelly Criterion with half-Kelly cap
    2. Volatility-adjusted sizing (ATR-normalized risk)
    3. Equity curve feedback (grow/protect dual-mode, enhanced)
    4. Correlation penalty (reduce size for correlated positions)
    5. Drawdown-reactive exponential decay with adaptive dd_power
    6. Max risk per symbol and total portfolio caps
    7. Streak awareness (boost after losses, reduce after wins)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    # Account parameters
    initial_equity: float = 1000.0
    leverage: float = 500.0
    contract_size: float = 100.0  # Default for gold; forex=100000

    # Risk parameters
    risk_grow: float = 0.15        # Risk when at equity high (grow mode)
    risk_protect: float = 0.02     # Base risk in protect mode
    max_risk_cap: float = 0.25     # Maximum risk per trade (absolute cap)
    min_risk_floor: float = 0.003  # Minimum risk per trade

    # Drawdown parameters
    dd_power: float = 12.0         # Exponential decay power in drawdown
    dd_power_adaptive: bool = True # Adapt dd_power based on DD depth
    at_high_thresh: float = 0.01   # Below this DD% = at-high mode
    dd_halt: float = 0.20          # Halt trading at this DD%

    # Kelly Criterion
    use_kelly: bool = True
    kelly_fraction: float = 0.5    # Half-Kelly for safety

    # Streak adjustments
    loss_boost: float = 1.8        # Multiply risk after consecutive losses (Martingale-lite)
    win_reduce: float = 0.5        # Multiply risk after consecutive wins
    streak_threshold: int = 2       # Consecutive count to trigger adjustment

    # Correlation penalty
    correlation_penalty_factor: float = 0.3  # Reduce by this * correlation
    max_correlation_reduction: float = 0.5   # Never reduce by more than 50%

    # Position limits
    max_risk_per_symbol: float = 0.05   # 5% max risk on any single symbol
    max_risk_total: float = 0.15         # 15% max total portfolio risk


@dataclass
class SizingResult:
    """Result of position size calculation."""
    lot_size: float = 0.01
    risk_pct: float = 0.01
    mode: str = "protect"  # "grow" or "protect"
    size_multiplier: float = 1.0
    kelly_size: float = 0.0
    correlation_penalty: float = 0.0
    dd_scale: float = 1.0
    approved: bool = True
    rejection_reason: str = ""


class PositionSizer:
    """
    Adaptive position sizing engine.

    Computes optimal lot size considering:
    - Current equity and drawdown state
    - Volatility (ATR-based stop distance)
    - Win rate and reward/risk ratio (Kelly)
    - Portfolio correlation
    - Consecutive win/loss streaks

    Usage:
        sizer = PositionSizer(SizingConfig(contract_size=100))
        result = sizer.compute_size(
            equity=1500, peak_equity=1600,
            sl_distance=2.5, atr=3.0,
            win_rate=0.55, avg_win=5.0, avg_loss=3.0,
            consec_wins=0, consec_losses=2,
            correlation_with_portfolio=0.3,
            regime_size_mult=1.0
        )
        print(f"Lot size: {result.lot_size}, Risk: {result.risk_pct:.4f}")
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()

    def compute_size(
        self,
        equity: float,
        peak_equity: float,
        sl_distance: float,
        atr: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        consec_wins: int = 0,
        consec_losses: int = 0,
        correlation_with_portfolio: float = 0.0,
        regime_size_mult: float = 1.0,
        existing_symbol_risk: float = 0.0,
        existing_total_risk: float = 0.0,
    ) -> SizingResult:
        """
        Compute position size with all adaptive factors.

        Parameters
        ----------
        equity : float
            Current account equity.
        peak_equity : float
            Peak equity (all-time high).
        sl_distance : float
            Stop-loss distance in price units.
        atr : float
            Current ATR for volatility reference.
        win_rate : float
            Estimated win rate (0-1).
        avg_win : float
            Average winning trade size.
        avg_loss : float
            Average losing trade size.
        consec_wins : int
            Consecutive winning trades.
        consec_losses : int
            Consecutive losing trades.
        correlation_with_portfolio : float
            Max correlation of new position with existing portfolio (0-1).
        regime_size_mult : float
            Multiplier from strategy selector (e.g., 0.5 for low-confidence regime).
        existing_symbol_risk : float
            Current risk already allocated to this symbol (fraction of equity).
        existing_total_risk : float
            Current total portfolio risk (fraction of equity).

        Returns
        -------
        SizingResult
            Complete sizing result with lot size and metadata.
        """
        cfg = self.config
        result = SizingResult()

        # Validate inputs
        if equity <= 0 or sl_distance <= 0 or atr <= 0:
            result.approved = False
            result.rejection_reason = "Invalid inputs (equity/sl/atr <= 0)"
            return result

        if peak_equity <= 0:
            peak_equity = equity

        # --- Step 1: Compute drawdown and base risk ---
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        current_dd = max(0.0, current_dd)

        # Check drawdown halt
        if current_dd >= cfg.dd_halt:
            result.approved = False
            result.rejection_reason = f"Drawdown halt: {current_dd:.4f} >= {cfg.dd_halt}"
            return result

        # Two-mode risk: grow vs protect
        if current_dd <= cfg.at_high_thresh:
            # GROW mode: at or near equity high
            base_risk = cfg.risk_grow
            result.mode = "grow"
        else:
            # PROTECT mode: exponential decay with drawdown
            eq_ratio = equity / peak_equity if peak_equity > 0 else 1.0

            # Adaptive dd_power: increase power as DD deepens (faster reduction)
            dd_power = cfg.dd_power
            if cfg.dd_power_adaptive:
                # Linearly increase dd_power as DD gets worse
                dd_severity = current_dd / cfg.dd_halt  # 0 to 1
                dd_power = cfg.dd_power * (1.0 + dd_severity * 0.5)

            base_risk = cfg.risk_protect * (eq_ratio ** dd_power)
            result.mode = "protect"
            result.dd_scale = eq_ratio ** dd_power

        # --- Step 2: Kelly Criterion adjustment ---
        kelly_risk = base_risk
        if cfg.use_kelly and avg_loss > 1e-10:
            # Kelly formula: f* = (bp - q) / b
            # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1.0 - p
            kelly_full = (b * p - q) / b if b > 0 else 0.0
            kelly_half = kelly_full * cfg.kelly_fraction

            if kelly_half > 0:
                # Use Kelly as an upper bound, not the sole determinant
                kelly_risk = min(base_risk, kelly_half)
                # But don't let Kelly reduce below protect risk in grow mode
                if result.mode == "grow":
                    kelly_risk = max(kelly_risk, cfg.risk_protect)
            else:
                # Negative Kelly means edge is negative - reduce aggressively
                kelly_risk = cfg.min_risk_floor

            result.kelly_size = kelly_half

        risk = kelly_risk

        # --- Step 3: Streak adjustment ---
        if consec_losses >= cfg.streak_threshold:
            # After losses: boost risk (Martingale-lite, expecting mean reversion)
            risk *= cfg.loss_boost
        elif consec_wins >= cfg.streak_threshold:
            # After wins: reduce risk (lock profits, expect regression to mean)
            risk *= cfg.win_reduce

        # --- Step 4: Correlation penalty ---
        corr_penalty = 0.0
        if correlation_with_portfolio > 0.3:
            corr_penalty = cfg.correlation_penalty_factor * correlation_with_portfolio
            corr_penalty = min(corr_penalty, cfg.max_correlation_reduction)
            risk *= (1.0 - corr_penalty)
        result.correlation_penalty = corr_penalty

        # --- Step 5: Regime confidence multiplier ---
        risk *= regime_size_mult
        result.size_multiplier = regime_size_mult

        # --- Step 6: Apply caps ---
        risk = max(cfg.min_risk_floor, min(cfg.max_risk_cap, risk))

        # Check per-symbol risk limit
        if existing_symbol_risk + risk > cfg.max_risk_per_symbol:
            available = max(0.0, cfg.max_risk_per_symbol - existing_symbol_risk)
            if available < cfg.min_risk_floor:
                result.approved = False
                result.rejection_reason = f"Symbol risk limit reached: {existing_symbol_risk:.4f}"
                return result
            risk = available

        # Check total portfolio risk limit
        if existing_total_risk + risk > cfg.max_risk_total:
            available = max(0.0, cfg.max_risk_total - existing_total_risk)
            if available < cfg.min_risk_floor:
                result.approved = False
                result.rejection_reason = f"Total portfolio risk limit: {existing_total_risk:.4f}"
                return result
            risk = available

        # --- Step 7: Convert risk to lot size ---
        # lot = (equity * risk) / (sl_distance * contract_size)
        lot = (equity * risk) / (sl_distance * cfg.contract_size)
        lot = max(0.01, min(200.0, lot))
        lot = round(lot, 2)

        result.lot_size = lot
        result.risk_pct = risk
        result.approved = True
        return result

    def compute_volatility_adjusted_size(
        self,
        equity: float,
        target_risk: float,
        current_atr: float,
        normal_atr: float,
        contract_size: float,
    ) -> float:
        """
        Quick volatility-adjusted lot size computation.

        Reduces size when volatility is above normal, increases when below.

        Parameters
        ----------
        equity : float
            Current equity.
        target_risk : float
            Target risk fraction.
        current_atr : float
            Current ATR.
        normal_atr : float
            Average/normal ATR for this instrument.
        contract_size : float
            Contract size for the instrument.

        Returns
        -------
        float
            Adjusted lot size.
        """
        if current_atr < 1e-10 or normal_atr < 1e-10:
            return 0.01

        # Scale SL with current volatility
        sl_distance = current_atr * 1.5  # 1.5x ATR stop

        # Volatility adjustment: reduce when vol is high relative to normal
        vol_adj = normal_atr / current_atr  # < 1 when vol high, > 1 when low
        vol_adj = np.clip(vol_adj, 0.5, 2.0)  # Cap adjustment range

        lot = (equity * target_risk * vol_adj) / (sl_distance * contract_size)
        lot = max(0.01, min(200.0, round(lot, 2)))
        return lot
