"""
=============================================================
  Python ML Bridge - Auto Optimizer
  Self-tuning parameter optimization engine that analyzes live
  trade results and gradually shifts parameters toward optimal
  values. Implements adaptive SL distances, session multipliers,
  confidence thresholds, momentum lookback, RSI levels, trailing
  distances, cooldown periods, and max position limits.
=============================================================
"""

import json
import os
import copy
import logging
from typing import Dict, List, Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import AutoOptimizerConfig

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """
    Self-tuning parameter optimizer that analyzes live trade results
    and gradually shifts trading parameters toward optimal values.

    The optimizer records every closed trade with full context (session,
    confidence level, momentum lookback used, SL distance, P/L result, etc.)
    and after every N trades (default 50), groups trades by parameter values,
    calculates win rate and average profit per value, then shifts each
    parameter 10-20% toward the winning value.

    Safety constraints:
    - Never jumps more than 1 step per optimization cycle
    - All parameters are clamped to configured ranges
    - Rollback if performance drops 20% after optimization
    - State persists to JSON file (survives restarts)
    """

    def __init__(self, config: Optional[AutoOptimizerConfig] = None,
                 state_dir: Optional[str] = None):
        self.config = config or AutoOptimizerConfig()
        self._state_dir = state_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._state_file = os.path.join(self._state_dir, self.config.state_file)

        # Current optimized parameters
        self._params: Dict = self._default_params()

        # Previous params (for rollback)
        self._prev_params: Optional[Dict] = None

        # Trade history for analysis
        self._trades: List[Dict] = []

        # Trades since last optimization
        self._trades_since_optimize: int = 0

        # Total optimization cycles completed
        self._cycle_count: int = 0

        # Performance tracking for rollback
        self._pre_optimize_performance: Optional[Dict] = None
        self._post_optimize_trades: List[Dict] = []
        self._rollback_pending: bool = False

        # Load saved state if available
        self.load_state()

    def _default_params(self) -> Dict:
        """Return default parameter values (optimized baselines)."""
        cfg = self.config
        return {
            "sl_distance": 5.0,  # $5 SL balanced for gold M1
            "session_multipliers": {
                "asian": 1.0,
                "london": 1.2,
                "newyork": 1.0,
                "overlap": 1.2,
                "off_session": 1.0,
            },
            "min_confidence": 0.25,
            "momentum_lookback": 8,
            "rsi_overbought": 62,
            "rsi_oversold": 38,
            "trail_distances": {
                "tight": 0.5,
                "medium": 1.0,
                "wide": 2.0,
            },
            "cooldown_seconds": 10,
            "max_positions": 3,
        }

    def record_trade(self, trade_context: Dict) -> None:
        """
        Record a closed trade with full context for analysis.

        Args:
            trade_context: Dict containing trade details:
                - session: str (asian/london/newyork/overlap/off_session)
                - confidence: float (entry confidence level)
                - momentum_lookback: int (lookback bars used)
                - sl_distance: float (stop loss distance in dollars)
                - result_pnl: float (realized P/L)
                - direction: str (BUY/SELL)
                - rsi_at_entry: float (RSI value at entry)
                - trail_tier: str (tight/medium/wide)
                - cooldown_used: float (cooldown seconds at entry)
                - max_positions_at_entry: int
                - entry_time: str (ISO timestamp)
                - exit_time: str (ISO timestamp)
        """
        if not self.config.enabled:
            return

        self._trades.append(trade_context)
        self._trades_since_optimize += 1

        # If rollback is pending, track post-optimize trades
        if self._rollback_pending:
            self._post_optimize_trades.append(trade_context)
            # Check rollback after enough post-optimize trades
            if len(self._post_optimize_trades) >= self.config.optimize_frequency:
                self._check_rollback()

        # Auto-trigger optimization when threshold reached
        if self._trades_since_optimize >= self.config.optimize_frequency:
            if len(self._trades) >= self.config.min_trades_before_tuning:
                self.optimize()

        logger.debug("[AutoOpt] Recorded trade #%d (PnL=%.2f, session=%s)",
                     len(self._trades),
                     trade_context.get("result_pnl", 0.0),
                     trade_context.get("session", "unknown"))

    def optimize(self) -> Dict:
        """
        Analyze recent trades and shift parameters toward optimal values.

        Groups trades by each parameter value, calculates win rate and
        average profit per group, then shifts current parameter 10-20%
        toward the best-performing value. Never jumps more than 1 step
        per cycle for safety.

        Returns:
            Dict with optimization results per parameter
        """
        if not self.config.enabled:
            return {"status": "disabled"}

        if len(self._trades) < self.config.min_trades_before_tuning:
            return {"status": "insufficient_trades",
                    "count": len(self._trades),
                    "required": self.config.min_trades_before_tuning}

        # Save pre-optimize performance for rollback comparison
        recent_trades = self._trades[-self.config.optimize_frequency:]
        self._pre_optimize_performance = self._calculate_performance(recent_trades)

        # Save previous params for potential rollback
        self._prev_params = copy.deepcopy(self._params)

        results = {}

        # Optimize each parameter
        results["sl_distance"] = self._optimize_sl_distance()
        results["session_multipliers"] = self._optimize_session_multipliers()
        results["min_confidence"] = self._optimize_confidence()
        results["momentum_lookback"] = self._optimize_momentum_lookback()
        results["rsi_levels"] = self._optimize_rsi_levels()
        results["trail_distances"] = self._optimize_trail_distances()
        results["cooldown_seconds"] = self._optimize_cooldown()
        results["max_positions"] = self._optimize_max_positions()

        # Reset counter and start rollback monitoring
        self._trades_since_optimize = 0
        self._post_optimize_trades = []
        self._rollback_pending = True
        self._cycle_count += 1

        # Persist state
        self.save_state()

        logger.info("[AutoOpt] Optimization cycle #%d complete. Results: %s",
                    self._cycle_count, results)

        return {"status": "optimized", "cycle": self._cycle_count, "results": results}

    def _optimize_sl_distance(self) -> Dict:
        """Optimize SL distance by finding the value with highest win rate."""
        trades_with_sl = [t for t in self._trades if "sl_distance" in t]
        if not trades_with_sl:
            return {"status": "no_data"}

        # Group by SL distance (bucket into $0.50 increments)
        buckets = defaultdict(list)
        for t in trades_with_sl:
            sl = t["sl_distance"]
            bucket = round(sl * 2) / 2  # Round to nearest $0.50
            bucket = max(self.config.sl_range[0], min(self.config.sl_range[1], bucket))
            buckets[bucket].append(t)

        # Find bucket with highest win rate
        best_bucket = None
        best_win_rate = -1.0
        for bucket_val, bucket_trades in buckets.items():
            wins = sum(1 for t in bucket_trades if t.get("result_pnl", 0) > 0)
            wr = wins / len(bucket_trades) if bucket_trades else 0
            if wr > best_win_rate:
                best_win_rate = wr
                best_bucket = bucket_val

        if best_bucket is None:
            return {"status": "no_optimal"}

        # Shift toward best bucket
        current = self._params["sl_distance"]
        shift = (best_bucket - current) * self.config.shift_rate
        # Clamp shift to max 1 step ($0.50)
        shift = max(-0.5, min(0.5, shift))
        new_val = current + shift
        new_val = max(self.config.sl_range[0], min(self.config.sl_range[1], new_val))
        self._params["sl_distance"] = round(new_val, 2)

        return {"old": current, "new": self._params["sl_distance"],
                "best_bucket": best_bucket, "best_win_rate": best_win_rate}

    def _optimize_session_multipliers(self) -> Dict:
        """Optimize session multipliers based on session profitability."""
        trades_with_session = [t for t in self._trades if "session" in t]
        if not trades_with_session:
            return {"status": "no_data"}

        results = {}
        session_groups = defaultdict(list)
        for t in trades_with_session:
            session_groups[t["session"]].append(t)

        for session, session_trades in session_groups.items():
            if session not in self._params["session_multipliers"]:
                continue

            avg_pnl = sum(t.get("result_pnl", 0) for t in session_trades) / len(session_trades)
            current_mult = self._params["session_multipliers"][session]

            # If session is profitable, increase multiplier; if losing, decrease
            if avg_pnl > 0:
                shift = self.config.shift_rate * 0.1  # Small positive shift
            elif avg_pnl < 0:
                shift = -self.config.shift_rate * 0.1  # Small negative shift
            else:
                shift = 0.0

            # Clamp shift to max 0.1 per cycle
            shift = max(-0.1, min(0.1, shift))
            new_mult = current_mult + shift
            new_mult = max(self.config.session_mult_range[0],
                           min(self.config.session_mult_range[1], new_mult))
            new_mult = round(new_mult, 2)

            results[session] = {"old": current_mult, "new": new_mult, "avg_pnl": avg_pnl}
            self._params["session_multipliers"][session] = new_mult

        return results

    def _optimize_confidence(self) -> Dict:
        """Optimize min confidence threshold based on low-confidence trade results."""
        trades_with_conf = [t for t in self._trades if "confidence" in t]
        if not trades_with_conf:
            return {"status": "no_data"}

        # Check if low-confidence trades are losing
        current_threshold = self._params["min_confidence"]
        low_conf_trades = [t for t in trades_with_conf
                           if t["confidence"] < current_threshold + 0.10]
        high_conf_trades = [t for t in trades_with_conf
                            if t["confidence"] >= current_threshold + 0.10]

        if not low_conf_trades:
            return {"status": "no_low_conf_trades"}

        low_conf_wr = (sum(1 for t in low_conf_trades if t.get("result_pnl", 0) > 0)
                       / len(low_conf_trades))
        high_conf_wr = (sum(1 for t in high_conf_trades if t.get("result_pnl", 0) > 0)
                        / len(high_conf_trades)) if high_conf_trades else 0.5

        # If low-confidence trades are losing significantly, raise threshold
        if low_conf_wr < 0.40:
            shift = self.config.shift_rate * 0.05  # Raise by ~0.75%
        elif low_conf_wr > 0.60:
            shift = -self.config.shift_rate * 0.03  # Lower slightly if they're winning
        else:
            shift = 0.0

        # Clamp shift to max 0.05 per cycle
        shift = max(-0.05, min(0.05, shift))
        new_val = current_threshold + shift
        new_val = max(self.config.confidence_range[0],
                      min(self.config.confidence_range[1], new_val))
        new_val = round(new_val, 3)

        old = self._params["min_confidence"]
        self._params["min_confidence"] = new_val
        return {"old": old, "new": new_val,
                "low_conf_wr": low_conf_wr, "high_conf_wr": high_conf_wr}

    def _optimize_momentum_lookback(self) -> Dict:
        """Optimize momentum lookback by finding which value gives best results."""
        trades_with_momentum = [t for t in self._trades if "momentum_lookback" in t]
        if not trades_with_momentum:
            return {"status": "no_data"}

        # Group by lookback value
        lookback_groups = defaultdict(list)
        for t in trades_with_momentum:
            lb = int(t["momentum_lookback"])
            lookback_groups[lb].append(t)

        # Find best-performing lookback
        best_lb = None
        best_metric = -float("inf")
        for lb_val, lb_trades in lookback_groups.items():
            avg_pnl = sum(t.get("result_pnl", 0) for t in lb_trades) / len(lb_trades)
            if avg_pnl > best_metric:
                best_metric = avg_pnl
                best_lb = lb_val

        if best_lb is None:
            return {"status": "no_optimal"}

        current = self._params["momentum_lookback"]
        # Shift toward best lookback (max 1 bar per cycle)
        if best_lb > current:
            new_val = current + 1
        elif best_lb < current:
            new_val = current - 1
        else:
            new_val = current

        new_val = max(self.config.momentum_range[0],
                      min(self.config.momentum_range[1], new_val))
        old = self._params["momentum_lookback"]
        self._params["momentum_lookback"] = new_val
        return {"old": old, "new": new_val, "best_lookback": best_lb}

    def _optimize_rsi_levels(self) -> Dict:
        """Optimize RSI overbought/oversold levels."""
        trades_with_rsi = [t for t in self._trades if "rsi_at_entry" in t]
        if not trades_with_rsi:
            return {"status": "no_data"}

        current_ob = self._params["rsi_overbought"]
        current_os = self._params["rsi_oversold"]

        # Analyze trades where RSI was extreme
        ob_trades = [t for t in trades_with_rsi if t["rsi_at_entry"] > current_ob - 5]
        os_trades = [t for t in trades_with_rsi if t["rsi_at_entry"] < current_os + 5]

        result = {}

        # Overbought: if filtering too strictly (blocking good trades), lower it
        if ob_trades:
            ob_wr = sum(1 for t in ob_trades if t.get("result_pnl", 0) > 0) / len(ob_trades)
            if ob_wr > 0.55:
                # RSI filter is too strict, lower OB threshold
                shift = -1
            elif ob_wr < 0.40:
                # RSI filter not strict enough, raise OB threshold
                shift = 1
            else:
                shift = 0
            new_ob = max(self.config.rsi_ob_range[0],
                         min(self.config.rsi_ob_range[1], current_ob + shift))
            result["overbought"] = {"old": current_ob, "new": new_ob, "ob_wr": ob_wr}
            self._params["rsi_overbought"] = new_ob

        # Oversold: similar logic
        if os_trades:
            os_wr = sum(1 for t in os_trades if t.get("result_pnl", 0) > 0) / len(os_trades)
            if os_wr > 0.55:
                # RSI filter is too strict, raise OS threshold
                shift = 1
            elif os_wr < 0.40:
                # RSI filter not strict enough, lower OS threshold
                shift = -1
            else:
                shift = 0
            new_os = max(self.config.rsi_os_range[0],
                         min(self.config.rsi_os_range[1], current_os + shift))
            result["oversold"] = {"old": current_os, "new": new_os, "os_wr": os_wr}
            self._params["rsi_oversold"] = new_os

        return result

    def _optimize_trail_distances(self) -> Dict:
        """Optimize trailing stop distances per tier."""
        trades_with_trail = [t for t in self._trades if "trail_tier" in t]
        if not trades_with_trail:
            return {"status": "no_data"}

        results = {}
        tier_groups = defaultdict(list)
        for t in trades_with_trail:
            tier_groups[t["trail_tier"]].append(t)

        for tier, tier_trades in tier_groups.items():
            if tier not in self._params["trail_distances"]:
                continue

            current_dist = self._params["trail_distances"][tier]
            avg_pnl = sum(t.get("result_pnl", 0) for t in tier_trades) / len(tier_trades)

            # If trailing at this tier is profitable, widen slightly (let winners run)
            # If losing, tighten (protect capital)
            if avg_pnl > 0:
                shift = 0.1 * self.config.shift_rate
            elif avg_pnl < 0:
                shift = -0.1 * self.config.shift_rate
            else:
                shift = 0.0

            # Clamp shift
            shift = max(-0.2, min(0.2, shift))
            new_dist = current_dist + shift
            new_dist = max(0.3, min(3.0, new_dist))
            new_dist = round(new_dist, 2)

            results[tier] = {"old": current_dist, "new": new_dist, "avg_pnl": avg_pnl}
            self._params["trail_distances"][tier] = new_dist

        return results

    def _optimize_cooldown(self) -> Dict:
        """Optimize cooldown seconds based on fast entry profitability."""
        trades_with_cooldown = [t for t in self._trades if "cooldown_used" in t]
        if not trades_with_cooldown:
            return {"status": "no_data"}

        current = self._params["cooldown_seconds"]

        # Group into fast vs slow entries
        fast_trades = [t for t in trades_with_cooldown if t["cooldown_used"] <= current]
        slow_trades = [t for t in trades_with_cooldown if t["cooldown_used"] > current]

        fast_wr = (sum(1 for t in fast_trades if t.get("result_pnl", 0) > 0)
                   / len(fast_trades)) if fast_trades else 0.5

        # If fast entries are profitable, reduce cooldown
        if fast_wr > 0.55:
            shift = -1  # Reduce by 1 second
        elif fast_wr < 0.40:
            shift = 1   # Increase by 1 second
        else:
            shift = 0

        new_val = current + shift
        new_val = max(self.config.cooldown_range[0],
                      min(self.config.cooldown_range[1], new_val))
        old = self._params["cooldown_seconds"]
        self._params["cooldown_seconds"] = new_val
        return {"old": old, "new": new_val, "fast_wr": fast_wr}

    def _optimize_max_positions(self) -> Dict:
        """Optimize max positions based on drawdown vs profit tradeoff."""
        trades_with_positions = [t for t in self._trades if "max_positions_at_entry" in t]
        if not trades_with_positions:
            return {"status": "no_data"}

        current = self._params["max_positions"]

        # Calculate max drawdown from recent trades
        running_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0
        total_profit = 0.0

        for t in trades_with_positions[-self.config.optimize_frequency:]:
            pnl = t.get("result_pnl", 0)
            running_pnl += pnl
            total_profit += pnl
            peak_pnl = max(peak_pnl, running_pnl)
            dd = peak_pnl - running_pnl
            max_dd = max(max_dd, dd)

        # If low drawdown and profitable, allow more positions
        # If high drawdown, reduce positions
        if max_dd > 0 and total_profit > 0:
            dd_ratio = max_dd / max(total_profit, 1.0)
            if dd_ratio < 0.3 and total_profit > 0:
                shift = 1  # Low DD relative to profit -> allow more
            elif dd_ratio > 0.7:
                shift = -1  # High DD -> reduce
            else:
                shift = 0
        elif total_profit <= 0:
            shift = -1  # Losing overall -> reduce positions
        else:
            shift = 0

        new_val = current + shift
        new_val = max(self.config.max_positions_range[0],
                      min(self.config.max_positions_range[1], new_val))
        old = self._params["max_positions"]
        self._params["max_positions"] = new_val
        return {"old": old, "new": new_val, "max_dd": max_dd, "total_profit": total_profit}

    def _calculate_performance(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics for a set of trades."""
        if not trades:
            return {"win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0, "count": 0}

        wins = sum(1 for t in trades if t.get("result_pnl", 0) > 0)
        total_pnl = sum(t.get("result_pnl", 0) for t in trades)
        return {
            "win_rate": wins / len(trades),
            "avg_pnl": total_pnl / len(trades),
            "total_pnl": total_pnl,
            "count": len(trades),
        }

    def _check_rollback(self) -> None:
        """Check if new params perform worse and rollback if needed."""
        if not self._rollback_pending or not self._prev_params:
            return

        if not self._pre_optimize_performance:
            self._rollback_pending = False
            return

        post_perf = self._calculate_performance(self._post_optimize_trades)
        pre_perf = self._pre_optimize_performance

        # Compare win rates - rollback if new params are 20% worse
        pre_wr = pre_perf["win_rate"]
        post_wr = post_perf["win_rate"]

        if pre_wr > 0 and post_wr < pre_wr * (1 - self.config.rollback_threshold):
            logger.warning(
                "[AutoOpt] ROLLBACK triggered! Post-optimize WR=%.2f vs pre=%.2f "
                "(%.1f%% drop exceeds %.0f%% threshold)",
                post_wr, pre_wr,
                (1 - post_wr / pre_wr) * 100,
                self.config.rollback_threshold * 100
            )
            self.rollback()
        else:
            logger.info("[AutoOpt] Post-optimize performance OK (WR: %.2f -> %.2f)",
                        pre_wr, post_wr)
            self._rollback_pending = False
            self._prev_params = None
            self._post_optimize_trades = []

    def rollback(self) -> None:
        """Revert to previous parameters if new params perform 20% worse."""
        if self._prev_params is None:
            logger.warning("[AutoOpt] No previous params to rollback to")
            return

        self._params = copy.deepcopy(self._prev_params)
        self._prev_params = None
        self._rollback_pending = False
        self._post_optimize_trades = []
        self.save_state()
        logger.info("[AutoOpt] Rolled back to previous parameters")

    def get_current_params(self) -> Dict:
        """
        Return the current optimized parameter values.

        Returns:
            Dict with all optimized parameters:
            - sl_distance: float
            - session_multipliers: Dict[str, float]
            - min_confidence: float
            - momentum_lookback: int
            - rsi_overbought: int
            - rsi_oversold: int
            - trail_distances: Dict[str, float]
            - cooldown_seconds: int
            - max_positions: int
        """
        return copy.deepcopy(self._params)

    def save_state(self) -> None:
        """Persist optimizer state to JSON file (survives restarts)."""
        state = {
            "params": self._params,
            "prev_params": self._prev_params,
            "trades_since_optimize": self._trades_since_optimize,
            "cycle_count": self._cycle_count,
            "rollback_pending": self._rollback_pending,
            "trade_count": len(self._trades),
            "trades": self._trades[-500:],  # Keep last 500 trades
            "post_optimize_trades": self._post_optimize_trades,
            "pre_optimize_performance": self._pre_optimize_performance,
        }

        try:
            # Atomic write: write to temp file then rename
            tmp_file = self._state_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_file, self._state_file)
            logger.debug("[AutoOpt] State saved to %s", self._state_file)
        except Exception as e:
            logger.error("[AutoOpt] Failed to save state: %s", e)

    def load_state(self) -> bool:
        """
        Load optimizer state from JSON file.

        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not os.path.exists(self._state_file):
            logger.info("[AutoOpt] No saved state found, using defaults")
            return False

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)

            self._params = state.get("params", self._default_params())
            self._prev_params = state.get("prev_params")
            self._trades_since_optimize = state.get("trades_since_optimize", 0)
            self._cycle_count = state.get("cycle_count", 0)
            self._rollback_pending = state.get("rollback_pending", False)
            self._trades = state.get("trades", [])
            self._post_optimize_trades = state.get("post_optimize_trades", [])
            self._pre_optimize_performance = state.get("pre_optimize_performance")

            logger.info("[AutoOpt] Loaded state: cycle=%d, trades=%d, params=%s",
                        self._cycle_count, len(self._trades), self._params)
            return True
        except Exception as e:
            logger.error("[AutoOpt] Failed to load state: %s", e)
            return False

    @property
    def trade_count(self) -> int:
        """Total number of recorded trades."""
        return len(self._trades)

    @property
    def cycle_count(self) -> int:
        """Total number of optimization cycles completed."""
        return self._cycle_count

    @property
    def is_enabled(self) -> bool:
        """Whether the auto-optimizer is enabled."""
        return self.config.enabled
