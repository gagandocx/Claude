"""
=============================================================
  Python ML Bridge - Smart Exit Manager
  AI-driven position management using reinforcement learning.
  Implements dynamic trailing stops, partial closes, and
  intelligent exit decisions like an experienced prop trader.

  Key principles (from professional trading):
  - Cut losers fast: Never let a small loss become a large one
  - Let winners run: Use trailing stops, not fixed targets
  - Scale out: Take partial profits at key levels
  - Adapt: Tighten stops as confidence decays over time
  - Be decisive: Avoid holding without conviction
=============================================================
"""

import os
import sys
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SmartExitConfig, RLConfig
from models.rl_agent import (
    RLAgent, ExitAction, ACTION_NAMES, PositionState
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  EXIT DECISION
# ─────────────────────────────────────────────
@dataclass
class ExitDecision:
    """Output from the smart exit evaluation.

    Represents what action to take on a position, like a prop trader's
    order ticket for position modification.
    """
    action: str                   # HOLD, CLOSE_FULL, CLOSE_PARTIAL, MODIFY_SL
    lot_pct_to_close: float       # Percentage of position to close (0.0 to 1.0)
    new_sl_price: float           # New stop loss price (0.0 if no change)
    reason: str                   # Human-readable explanation
    confidence: float             # Agent's confidence in this decision
    q_values: Optional[Dict[str, float]] = None  # Q-values for analysis


# ─────────────────────────────────────────────
#  SMART EXIT MANAGER
# ─────────────────────────────────────────────
class SmartExitManager:
    """AI-driven exit management system.

    Combines reinforcement learning with rule-based prop trading logic:
    1. RL agent suggests action based on learned Q-values
    2. Rule-based system validates and adjusts (trailing stops, partials)
    3. Dynamic confidence decay reduces trail width over time
    4. Hard rules prevent catastrophic losses (max hold, min confidence)

    Think of it as an AI assistant to a prop trader, where the AI
    suggests the action but hard risk rules always have final say.
    """

    def __init__(self, config: Optional[SmartExitConfig] = None,
                 rl_config: Optional[RLConfig] = None):
        self.config = config or SmartExitConfig()
        self.rl_agent = RLAgent(rl_config or RLConfig())

        # Track positions being managed
        self._position_history: Dict[str, List[ExitDecision]] = {}
        self._decisions_made = 0

    def evaluate_exit(self, position: PositionState,
                      market_features: Optional[np.ndarray] = None,
                      training: bool = True) -> ExitDecision:
        """Evaluate whether to exit/modify an open position.

        This is the main decision function called each bar for each
        open position. Combines RL agent suggestion with rule-based
        validation.

        Args:
            position: Current position state information
            market_features: Latest model features for context
            training: Whether agent is in training mode (exploration)

        Returns:
            ExitDecision with action to take
        """
        # Build state for RL agent
        state = self.rl_agent.state_from_position(position, market_features)

        # Get RL agent's action
        action_idx = self.rl_agent.select_action(state, training=training)
        q_values = self.rl_agent.get_q_values(state)

        # Convert to named Q-values for analysis
        q_dict = {ACTION_NAMES[ExitAction(i)]: float(q_values[i])
                  for i in range(len(q_values))}

        # Apply rule-based overrides (hard risk rules always win)
        decision = self._apply_rules(position, action_idx, q_dict)

        # Track decision
        self._decisions_made += 1
        ticket = position.ticket or "unknown"
        if ticket not in self._position_history:
            self._position_history[ticket] = []
        self._position_history[ticket].append(decision)

        return decision

    def _apply_rules(self, position: PositionState,
                     rl_action: int, q_values: Dict[str, float]) -> ExitDecision:
        """Apply rule-based overrides to RL agent suggestions.

        Hard rules that override RL (non-negotiable risk management):
        1. Max hold time exceeded -> CLOSE_FULL
        2. Confidence below minimum -> CLOSE_FULL
        3. Loss exceeds 2 ATR -> CLOSE_FULL (stop out)
        4. At 2R profit with partial_close_at_2r -> suggest partial
        5. At 1R profit with break_even enabled -> move to BE
        """
        # RULE 1: Max hold time exceeded
        if position.hold_bars >= self.config.max_hold_bars:
            return ExitDecision(
                action="CLOSE_FULL",
                lot_pct_to_close=1.0 - position.partial_closed_pct,
                new_sl_price=0.0,
                reason=f"Max hold time exceeded ({position.hold_bars} bars)",
                confidence=1.0,
                q_values=q_values
            )

        # RULE 2: Confidence below minimum threshold
        if position.confidence < self.config.min_confidence_to_hold:
            return ExitDecision(
                action="CLOSE_FULL",
                lot_pct_to_close=1.0 - position.partial_closed_pct,
                new_sl_price=0.0,
                reason=f"Confidence too low ({position.confidence:.3f})",
                confidence=1.0,
                q_values=q_values
            )

        # RULE 3: Loss exceeds 2 ATR (catastrophic stop)
        if position.unrealized_pnl_atr < -2.0:
            return ExitDecision(
                action="CLOSE_FULL",
                lot_pct_to_close=1.0 - position.partial_closed_pct,
                new_sl_price=0.0,
                reason=f"Loss exceeds 2 ATR ({position.unrealized_pnl_atr:.2f}R)",
                confidence=1.0,
                q_values=q_values
            )

        # RULE 4: Partial close at 2R profit
        if (self.config.partial_close_at_2r
                and position.unrealized_pnl_atr >= 2.0
                and position.partial_closed_pct < self.config.partial_close_pct):
            return ExitDecision(
                action="CLOSE_PARTIAL",
                lot_pct_to_close=self.config.partial_close_pct - position.partial_closed_pct,
                new_sl_price=position.entry_price,  # Move to break-even
                reason=f"Taking partial profit at 2R ({position.unrealized_pnl_atr:.2f}R)",
                confidence=0.9,
                q_values=q_values
            )

        # RULE 5: Partial close at 3R profit
        if (self.config.partial_close_at_3r
                and position.unrealized_pnl_atr >= 3.0
                and position.partial_closed_pct < (self.config.partial_close_pct + self.config.partial_close_3r_pct)):
            close_pct = min(
                self.config.partial_close_3r_pct,
                1.0 - position.partial_closed_pct - 0.25  # Keep at least 25%
            )
            if close_pct > 0:
                return ExitDecision(
                    action="CLOSE_PARTIAL",
                    lot_pct_to_close=close_pct,
                    new_sl_price=0.0,
                    reason=f"Taking more profit at 3R ({position.unrealized_pnl_atr:.2f}R)",
                    confidence=0.85,
                    q_values=q_values
                )

        # RULE 6: Break-even at 1R
        if (self.config.break_even_at_1r
                and position.unrealized_pnl_atr >= 1.0
                and position.partial_closed_pct == 0.0):
            # Check if stop is not already at break-even
            be_distance = abs(position.current_price - position.entry_price) / max(position.atr, 0.01)
            if position.sl_distance_atr > be_distance * 0.5:
                return ExitDecision(
                    action="MODIFY_SL",
                    lot_pct_to_close=0.0,
                    new_sl_price=position.entry_price,
                    reason="Moving stop to break-even at 1R profit",
                    confidence=0.8,
                    q_values=q_values
                )

        # No hard rule triggered: use RL agent decision
        return self._convert_rl_action(position, rl_action, q_values)

    def _convert_rl_action(self, position: PositionState,
                           action_idx: int,
                           q_values: Dict[str, float]) -> ExitDecision:
        """Convert RL action index to ExitDecision."""
        action = ExitAction(action_idx)

        if action == ExitAction.HOLD:
            return ExitDecision(
                action="HOLD",
                lot_pct_to_close=0.0,
                new_sl_price=0.0,
                reason="RL agent: Hold position (conviction intact)",
                confidence=float(q_values.get("HOLD_POSITION", 0)),
                q_values=q_values
            )

        elif action == ExitAction.CLOSE_FULL:
            return ExitDecision(
                action="CLOSE_FULL",
                lot_pct_to_close=1.0 - position.partial_closed_pct,
                new_sl_price=0.0,
                reason="RL agent: Close full position",
                confidence=float(q_values.get("CLOSE_FULL", 0)),
                q_values=q_values
            )

        elif action == ExitAction.CLOSE_PARTIAL_25:
            remaining = 1.0 - position.partial_closed_pct
            close_pct = min(0.25, remaining - 0.25)  # Keep at least 25%
            if close_pct <= 0:
                # Can't partial close, treat as hold
                return ExitDecision(
                    action="HOLD",
                    lot_pct_to_close=0.0,
                    new_sl_price=0.0,
                    reason="RL agent: Cannot partial close (too little remaining)",
                    confidence=0.5,
                    q_values=q_values
                )
            return ExitDecision(
                action="CLOSE_PARTIAL",
                lot_pct_to_close=close_pct,
                new_sl_price=0.0,
                reason="RL agent: Close 25% (scale out)",
                confidence=float(q_values.get("CLOSE_PARTIAL_25", 0)),
                q_values=q_values
            )

        elif action == ExitAction.CLOSE_PARTIAL_50:
            remaining = 1.0 - position.partial_closed_pct
            close_pct = min(0.50, remaining - 0.25)  # Keep at least 25%
            if close_pct <= 0:
                return ExitDecision(
                    action="HOLD",
                    lot_pct_to_close=0.0,
                    new_sl_price=0.0,
                    reason="RL agent: Cannot partial close 50% (too little remaining)",
                    confidence=0.5,
                    q_values=q_values
                )
            return ExitDecision(
                action="CLOSE_PARTIAL",
                lot_pct_to_close=close_pct,
                new_sl_price=0.0,
                reason="RL agent: Close 50% (lock profits)",
                confidence=float(q_values.get("CLOSE_PARTIAL_50", 0)),
                q_values=q_values
            )

        elif action == ExitAction.TIGHTEN_STOP:
            new_sl = self.compute_trailing_stop(
                entry_price=position.entry_price,
                current_price=position.current_price,
                atr=position.atr,
                direction=position.direction,
                confidence=position.confidence
            )
            return ExitDecision(
                action="MODIFY_SL",
                lot_pct_to_close=0.0,
                new_sl_price=new_sl,
                reason="RL agent: Tighten stop loss",
                confidence=float(q_values.get("TIGHTEN_STOP", 0)),
                q_values=q_values
            )

        # Fallback
        return ExitDecision(
            action="HOLD",
            lot_pct_to_close=0.0,
            new_sl_price=0.0,
            reason="Default hold",
            confidence=0.0,
            q_values=q_values
        )

    def compute_trailing_stop(self, entry_price: float, current_price: float,
                              atr: float, direction: int,
                              confidence: float) -> float:
        """Compute dynamic trailing stop level.

        Professional trailing stop logic:
        - High confidence: wide trail (let profits run)
        - Low/decaying confidence: tight trail (protect capital)
        - Linear interpolation between tight and wide based on confidence

        Args:
            entry_price: Original entry price
            current_price: Current market price
            atr: Current ATR value
            direction: 1 for LONG, -1 for SHORT
            confidence: Current confidence level (0 to 1)

        Returns:
            New stop loss price
        """
        if atr <= 0:
            atr = 1.0  # Safety fallback

        # Interpolate trail width based on confidence
        if confidence <= self.config.confidence_decay_threshold:
            trail_mult = self.config.trailing_atr_mult_tight
        elif confidence >= self.config.confidence_strong_threshold:
            trail_mult = self.config.trailing_atr_mult_wide
        else:
            # Linear interpolation
            t = (confidence - self.config.confidence_decay_threshold) / (
                self.config.confidence_strong_threshold - self.config.confidence_decay_threshold
            )
            trail_mult = (self.config.trailing_atr_mult_tight +
                          t * (self.config.trailing_atr_mult_wide -
                               self.config.trailing_atr_mult_tight))

        trail_distance = atr * trail_mult

        if direction == 1:  # LONG
            trailing_sl = current_price - trail_distance
            # Never move stop backwards
            return max(trailing_sl, entry_price) if current_price > entry_price else trailing_sl
        else:  # SHORT
            trailing_sl = current_price + trail_distance
            # Never move stop backwards (for shorts, lower is better)
            return min(trailing_sl, entry_price) if current_price < entry_price else trailing_sl

    def compute_partial_close(self, unrealized_pnl_atr: float,
                              confidence: float,
                              hold_bars: int,
                              partial_closed_pct: float) -> Tuple[bool, float]:
        """Determine if partial close is warranted.

        Prop trader partial close rules:
        - At 2R: close 50% to lock profits
        - At 3R: close 25% more
        - Keep final 25% running with tight trail
        - If confidence decays below threshold: close more

        Args:
            unrealized_pnl_atr: Current PnL in ATR multiples
            confidence: Current confidence level
            hold_bars: Bars held
            partial_closed_pct: Already closed percentage

        Returns:
            Tuple of (should_close, percentage_to_close)
        """
        remaining = 1.0 - partial_closed_pct

        if remaining <= 0.25:
            # Already scaled out enough, let remainder run
            return False, 0.0

        # 2R partial close
        if (self.config.partial_close_at_2r
                and unrealized_pnl_atr >= 2.0
                and partial_closed_pct < self.config.partial_close_pct):
            close_pct = self.config.partial_close_pct - partial_closed_pct
            return True, min(close_pct, remaining - 0.25)

        # 3R partial close
        if (self.config.partial_close_at_3r
                and unrealized_pnl_atr >= 3.0
                and partial_closed_pct < (self.config.partial_close_pct + self.config.partial_close_3r_pct)):
            target = self.config.partial_close_pct + self.config.partial_close_3r_pct
            close_pct = target - partial_closed_pct
            return True, min(close_pct, remaining - 0.25)

        # Confidence decay partial close
        if confidence < self.config.confidence_decay_threshold and unrealized_pnl_atr > 0.5:
            # Take some off when confidence wanes but still in profit
            close_pct = 0.25
            if partial_closed_pct + close_pct <= 0.75:
                return True, close_pct

        return False, 0.0

    def feed_reward(self, position: PositionState,
                    action_taken: int, realized_pnl: float,
                    next_state: np.ndarray, done: bool,
                    market_features: Optional[np.ndarray] = None):
        """Feed trade outcome back to RL agent for learning.

        Called when a position is closed (full or partial) to provide
        the reward signal. This is how the agent learns from experience.

        Args:
            position: Position state when action was taken
            action_taken: The action that was executed
            realized_pnl: Actual PnL from the action
            next_state: State after action
            done: Whether position is fully closed
            market_features: Market features at time of action
        """
        state = self.rl_agent.state_from_position(position, market_features)
        reward = self.rl_agent.compute_reward(
            position, action_taken, realized_pnl, trade_closed=done
        )
        self.rl_agent.store_transition(state, action_taken, reward, next_state, done)

        # Train if enough experience
        loss = self.rl_agent.train_step()
        if loss is not None:
            logger.debug(f"[SmartExit] RL train step loss={loss:.4f}, "
                         f"epsilon={self.rl_agent.epsilon:.4f}")

    def decay_confidence(self, initial_confidence: float,
                         hold_bars: int) -> float:
        """Compute decayed confidence based on holding time.

        Confidence naturally decays over time because market conditions
        change. A signal that was strong 5 bars ago may no longer be valid.
        """
        return initial_confidence * (self.config.confidence_time_decay ** hold_bars)

    def get_stats(self) -> Dict:
        """Get exit manager statistics."""
        return {
            "decisions_made": self._decisions_made,
            "positions_tracked": len(self._position_history),
            "rl_stats": self.rl_agent.get_stats(),
        }
