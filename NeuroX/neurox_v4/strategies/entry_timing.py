"""
=============================================================
  NeuroX v7.4 - Smart Entry Timing (Micro-Pullback)

  After a signal fires (BUY/SELL), waits for a micro-pullback
  before executing the entry. This reduces average entry slippage
  by entering at a slightly better price when available.

  Logic:
    - After signal, wait up to timeout_seconds for a pullback
    - If pullback of pullback_points occurs -> enter (better price)
    - If timeout reached -> enter at market (don't miss the move)
    - If price moves away by breakout_threshold -> enter immediately

  v7.5: Adaptive timeout based on ATR. In high volatility,
  price moves faster so timeout is shorter. In low volatility,
  price moves slower so timeout is longer.

  Non-blocking: uses time comparison, not sleep. Each call to
  evaluate_entry checks elapsed time and current price.
=============================================================
"""

import os
import sys
import time
import logging
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EntryTimingConfig

logger = logging.getLogger(__name__)


class EntryTimingManager:
    """
    Smart entry timing using micro-pullback detection.

    After a trade signal fires, this manager waits for a small pullback
    in price before confirming the entry. This is non-blocking and uses
    time-based comparison on each evaluation call.

    v7.5: Timeout adapts to volatility (ATR). High ATR -> shorter timeout.
    """

    def __init__(self, config: Optional[EntryTimingConfig] = None):
        self.config = config or EntryTimingConfig()

        # Pending signal state
        self._signal_action: Optional[str] = None
        self._signal_price: Optional[float] = None
        self._signal_time: Optional[float] = None
        self._best_pullback_price: Optional[float] = None
        # Full signal details for reconstruction during HOLD-cycle triggers
        self._signal_details: Optional[Dict] = None
        # Adaptive timeout for this specific signal
        self._active_timeout: float = self.config.timeout_seconds

    @property
    def has_pending_signal(self) -> bool:
        """Check if there is a pending signal waiting for pullback."""
        return self._signal_action is not None

    def _compute_adaptive_timeout(self, atr: float, avg_atr: float) -> float:
        """
        Compute adaptive timeout based on current volatility vs average.

        In high volatility (ATR > avg), price moves faster, so we use a
        shorter timeout (no time to wait). In low volatility (ATR < avg),
        price moves slower, so we allow a longer timeout.

        Formula: base_timeout * (avg_atr / max(atr, 0.01))
        Clamped to [adaptive_timeout_min, adaptive_timeout_max].

        Args:
            atr: Current ATR value
            avg_atr: Average (normal) ATR value

        Returns:
            Adapted timeout in seconds
        """
        if avg_atr <= 0 or atr <= 0:
            return self.config.timeout_seconds

        ratio = avg_atr / max(atr, 0.01)
        adaptive_timeout = self.config.timeout_seconds * ratio

        # Clamp to configured bounds
        adaptive_timeout = max(self.config.adaptive_timeout_min, adaptive_timeout)
        adaptive_timeout = min(self.config.adaptive_timeout_max, adaptive_timeout)

        return adaptive_timeout

    def set_pending_signal(self, action: str, price: float, timestamp: Optional[float] = None,
                           signal_details: Optional[Dict] = None,
                           atr: float = 0.0, avg_atr: float = 0.0) -> None:
        """
        Register a new signal waiting for micro-pullback entry.

        Args:
            action: "BUY" or "SELL"
            price: Price at signal generation time
            timestamp: Signal time (defaults to current time)
            signal_details: Full signal attributes (confidence, sl_pips, tp_pips,
                            lot_size, model_name, regime, symbol) for reconstruction
                            when the pullback triggers during a HOLD cycle.
            atr: Current ATR for adaptive timeout calculation
            avg_atr: Average ATR for adaptive timeout calculation
        """
        self._signal_action = action
        self._signal_price = price
        self._signal_time = timestamp or time.monotonic()
        self._best_pullback_price = price
        self._signal_details = signal_details

        # Compute adaptive timeout if enabled and ATR data available
        if self.config.adaptive_timeout and atr > 0 and avg_atr > 0:
            self._active_timeout = self._compute_adaptive_timeout(atr, avg_atr)
            logger.info(
                f"[EntryTiming] Pending {action} signal @ ${price:.2f} - "
                f"adaptive timeout: {self._active_timeout:.1f}s "
                f"(ATR={atr:.2f}, avg={avg_atr:.2f}), "
                f"pullback target: ${self.config.pullback_points:.2f}"
            )
        else:
            self._active_timeout = self.config.timeout_seconds
            logger.info(
                f"[EntryTiming] Pending {action} signal @ ${price:.2f} - "
                f"waiting for pullback of ${self.config.pullback_points:.2f} "
                f"(timeout: {self._active_timeout:.1f}s)"
            )

    def evaluate_entry(self, action: str, current_price: float) -> Dict:
        """
        Evaluate whether to enter now based on micro-pullback logic.

        Args:
            action: The pending action ("BUY" or "SELL")
            current_price: Current market price

        Returns:
            Dict with keys:
                'should_enter': bool - True if entry should happen now
                'reason': str - Why the entry decision was made
                'adjusted_price': float - The price to enter at
        """
        if not self.has_pending_signal:
            return {
                'should_enter': False,
                'reason': 'No pending signal',
                'adjusted_price': current_price,
            }

        elapsed = time.monotonic() - self._signal_time
        signal_price = self._signal_price
        timeout = self._active_timeout

        # Calculate pullback amount based on direction
        if self._signal_action == "BUY":
            # For BUY: pullback means price went DOWN from signal price
            pullback_amount = signal_price - current_price
            # Track best pullback (lowest price seen)
            if current_price < self._best_pullback_price:
                self._best_pullback_price = current_price
            # Breakout: price moved UP significantly from signal
            breakout_amount = current_price - signal_price
        else:
            # For SELL: pullback means price went UP from signal price
            pullback_amount = current_price - signal_price
            # Track best pullback (highest price seen)
            if current_price > self._best_pullback_price:
                self._best_pullback_price = current_price
            # Breakout: price moved DOWN significantly from signal
            breakout_amount = signal_price - current_price

        # Check conditions in priority order:

        # 1. Pullback detected - enter at better price
        if pullback_amount >= self.config.pullback_points:
            reason = (
                f"Pullback detected: ${pullback_amount:.2f} "
                f"(target: ${self.config.pullback_points:.2f}) "
                f"after {elapsed:.1f}s"
            )
            logger.info(f"[EntryTiming] ENTER - {reason}")
            self._clear_pending()
            return {
                'should_enter': True,
                'reason': reason,
                'adjusted_price': current_price,
            }

        # 2. Breakout - price moving away fast, enter immediately
        if breakout_amount >= self.config.breakout_threshold_points:
            reason = (
                f"Breakout detected: ${breakout_amount:.2f} away "
                f"(threshold: ${self.config.breakout_threshold_points:.2f}) "
                f"after {elapsed:.1f}s"
            )
            logger.info(f"[EntryTiming] ENTER (breakout) - {reason}")
            self._clear_pending()
            return {
                'should_enter': True,
                'reason': reason,
                'adjusted_price': current_price,
            }

        # 3. Timeout - enter at market to avoid missing the move
        if elapsed >= timeout:
            reason = (
                f"Timeout ({timeout:.1f}s) reached - "
                f"entering at market. Best pullback seen: "
                f"${abs(self._best_pullback_price - signal_price):.2f}"
            )
            logger.info(f"[EntryTiming] ENTER (timeout) - {reason}")
            self._clear_pending()
            return {
                'should_enter': True,
                'reason': reason,
                'adjusted_price': current_price,
            }

        # 4. Still waiting for pullback
        return {
            'should_enter': False,
            'reason': (
                f"Waiting for pullback: {elapsed:.1f}s / {timeout:.1f}s, "
                f"pullback so far: ${max(0, pullback_amount):.2f} / "
                f"${self.config.pullback_points:.2f}"
            ),
            'adjusted_price': current_price,
        }

    def _clear_pending(self) -> None:
        """Clear the pending signal state."""
        self._signal_action = None
        self._signal_price = None
        self._signal_time = None
        self._best_pullback_price = None
        self._signal_details = None
        self._active_timeout = self.config.timeout_seconds

    def cancel_pending(self) -> None:
        """Cancel any pending signal (e.g., if conditions changed)."""
        if self.has_pending_signal:
            logger.debug(
                f"[EntryTiming] Cancelled pending {self._signal_action} signal"
            )
        self._clear_pending()

    def get_status(self) -> Dict:
        """Return current entry timing status."""
        if not self.has_pending_signal:
            return {"pending": False}
        elapsed = time.monotonic() - self._signal_time
        return {
            "pending": True,
            "action": self._signal_action,
            "signal_price": self._signal_price,
            "elapsed_seconds": elapsed,
            "timeout_seconds": self._active_timeout,
            "best_pullback_price": self._best_pullback_price,
            "signal_details": self._signal_details,
        }
