"""
MT5 Tick Monitor - Real-Time Tick Stream Processing.

Handles:
- Real-time tick monitoring and candle close detection
- Pre-computation of direction and weakness signals between candle closes
- Signal readiness state management (IDLE -> SCANNING -> ARMED)
- Instant execution trigger when new 1M candle closes

The goal: when a 1M candle closes, the direction and weakness signals
are already computed. Only the rectangle entry check needs to run,
achieving signal-to-execution time under 500ms.
"""

import time
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import pandas as pd

import mt5_config
from mt5_data_feed import MT5DataFeed


class SignalState(Enum):
    """
    Signal readiness state for the tick monitor.

    IDLE: No direction signal detected, waiting for 4H setup.
    SCANNING: Direction confirmed, scanning 15M for weakness.
    ARMED: Direction + weakness confirmed, waiting for 1M candle close
           to check rectangle entry. This is the hot state where
           execution must be instant.
    """
    IDLE = "idle"
    SCANNING = "scanning"
    ARMED = "armed"


@dataclass
class PreComputedSignals:
    """Cached pre-computed signals for instant execution."""
    direction_signal: Optional[Any] = None
    direction_computed_at: float = 0.0
    weakness_signal: Optional[Any] = None
    weakness_computed_at: float = 0.0
    df_4h: Optional[pd.DataFrame] = None
    df_15m: Optional[pd.DataFrame] = None
    state: SignalState = SignalState.IDLE


class TickMonitor:
    """
    Real-time tick stream processor for CCT Rectangle Bot.

    Monitors the tick stream to:
    1. Detect the exact moment a new 1M candle closes
    2. Pre-compute direction (4H) and weakness (15M) signals
       between candle closes so only the entry check runs at close time
    3. Track signal readiness state for instant execution

    Usage:
        monitor = TickMonitor(data_feed, logger)
        monitor.start()

        # In main loop (every ~1 second):
        new_candle = monitor.check_candle_close()
        if new_candle:
            # Execute immediately - signals are pre-computed
            signals = monitor.get_precomputed_signals()
    """

    def __init__(
        self,
        data_feed: MT5DataFeed,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize TickMonitor.

        Args:
            data_feed: MT5DataFeed instance for fetching market data.
            logger: Optional logger instance.
        """
        self.data_feed = data_feed
        self.logger = logger or logging.getLogger("mt5_tick_monitor")

        # Candle close detection state
        self._last_1m_candle_time: Optional[int] = None  # Unix timestamp of last known 1M bar
        self._last_tick_time: float = 0.0

        # Pre-computed signals cache
        self._signals = PreComputedSignals()

        # Performance tracking
        self._last_candle_close_detected_at: float = 0.0
        self._candle_close_count: int = 0

    @property
    def signal_state(self) -> SignalState:
        """Current signal readiness state."""
        return self._signals.state

    @property
    def is_armed(self) -> bool:
        """True when direction + weakness are confirmed and waiting for entry."""
        return self._signals.state == SignalState.ARMED

    def start(self):
        """
        Initialize the tick monitor.

        Fetches the initial 1M candle time reference and performs
        first pre-computation of signals.
        """
        self.logger.info("TickMonitor starting...")

        # Get initial 1M candle time reference
        latest_candles = self.data_feed.get_latest_1m_candle()
        if latest_candles is not None and len(latest_candles) >= 1:
            # Use the timestamp of the most recent completed candle
            self._last_1m_candle_time = int(latest_candles.index[-1].timestamp())
            self.logger.info(
                f"Initial 1M candle reference: {latest_candles.index[-1]}"
            )

        # Perform initial signal pre-computation
        if mt5_config.PRE_COMPUTE_SIGNALS:
            self._precompute_direction()
            self._precompute_weakness()

        self.logger.info(
            f"TickMonitor started. State: {self._signals.state.value}"
        )

    def check_candle_close(self) -> bool:
        """
        Check if a new 1M candle has closed since last check.

        Uses the tick timestamp and candle data to detect the exact moment
        a new 1M bar appears, indicating the previous bar has closed.

        Returns:
            True if a new 1M candle has just closed, False otherwise.
        """
        # Get latest tick for timing reference
        tick = self.data_feed.get_latest_tick()
        if tick is None:
            return False

        # Check if tick is stale
        tick_time = tick.get("time", 0)
        now = time.time()
        if now - tick_time > mt5_config.TICK_STALE_THRESHOLD_SECONDS:
            self.logger.debug("Tick is stale, skipping candle close check")
            return False

        self._last_tick_time = tick_time

        # Fetch latest 1M candle(s) to check for new bar
        latest_candles = self.data_feed.get_latest_1m_candle()
        if latest_candles is None or len(latest_candles) < 1:
            return False

        # Get the timestamp of the most recent candle
        current_candle_time = int(latest_candles.index[-1].timestamp())

        # Compare with our stored reference
        if self._last_1m_candle_time is None:
            # First check - just store the reference
            self._last_1m_candle_time = current_candle_time
            return False

        if current_candle_time > self._last_1m_candle_time:
            # New candle detected - the previous candle has closed
            detection_time = time.time()
            self._last_candle_close_detected_at = detection_time
            self._candle_close_count += 1

            self.logger.info(
                f"New 1M candle detected! Previous bar closed. "
                f"New bar time: {datetime.fromtimestamp(current_candle_time, tz=timezone.utc)}"
            )

            # Update reference
            self._last_1m_candle_time = current_candle_time
            return True

        return False

    def get_current_tick_price(self) -> Optional[Dict[str, float]]:
        """
        Get the current bid/ask prices from the latest tick.

        Returns:
            Dictionary with 'bid' and 'ask' prices, or None.
        """
        tick = self.data_feed.get_latest_tick()
        if tick is None:
            return None
        return {"bid": tick["bid"], "ask": tick["ask"]}

    def is_price_in_rectangle_zone(
        self, rectangle_top: float, rectangle_bottom: float
    ) -> Optional[str]:
        """
        Check if current price is near or inside a rectangle zone.

        Used for tick-level monitoring to prepare for breakout execution.
        When price is near the rectangle boundary, the system stays alert
        for a breakout candle close.

        Args:
            rectangle_top: Upper boundary of the rectangle.
            rectangle_bottom: Lower boundary of the rectangle.

        Returns:
            'above' if price is above rectangle (potential bullish breakout),
            'below' if price is below rectangle (potential bearish breakout),
            'inside' if price is within the rectangle,
            None if price data unavailable.
        """
        tick = self.data_feed.get_latest_tick()
        if tick is None:
            return None

        mid_price = (tick["bid"] + tick["ask"]) / 2

        if mid_price > rectangle_top:
            return "above"
        elif mid_price < rectangle_bottom:
            return "below"
        else:
            return "inside"

    def precompute_signals(self):
        """
        Run pre-computation of direction and weakness signals.

        Called between candle closes to have signals ready before
        the next 1M candle closes. This is the key optimization that
        enables sub-500ms signal-to-execution time.

        Only re-computes if the cached signals have expired based on
        configured cache durations.
        """
        if not mt5_config.PRE_COMPUTE_SIGNALS:
            return

        now = time.time()

        # Re-compute direction if cache expired (every 4 hours)
        if now - self._signals.direction_computed_at > mt5_config.DIRECTION_CACHE_SECONDS:
            self._precompute_direction()

        # Re-compute weakness if cache expired (every 15 minutes)
        if now - self._signals.weakness_computed_at > mt5_config.WEAKNESS_CACHE_SECONDS:
            self._precompute_weakness()

        # Update state based on what signals are available
        self._update_signal_state()

    def get_precomputed_signals(self) -> PreComputedSignals:
        """
        Get the current pre-computed signals.

        Returns:
            PreComputedSignals dataclass with cached direction/weakness data.
        """
        return self._signals

    def get_cached_dataframes(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get the cached DataFrames from pre-computation.

        Returns pre-computed 4H and 15M DataFrames so they don't need
        to be re-fetched at candle close time. Only the 1M data needs
        a fresh fetch for the entry check.

        Returns:
            Dictionary with 'df_4h' and 'df_15m' or None if not cached.
        """
        if self._signals.df_4h is None or self._signals.df_15m is None:
            return None

        return {
            "df_4h": self._signals.df_4h,
            "df_15m": self._signals.df_15m,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tick monitor performance statistics.

        Returns:
            Dictionary with monitoring stats.
        """
        return {
            "state": self._signals.state.value,
            "candle_closes_detected": self._candle_close_count,
            "last_candle_close_at": self._last_candle_close_detected_at,
            "last_tick_time": self._last_tick_time,
            "direction_cached": self._signals.direction_signal is not None,
            "weakness_cached": self._signals.weakness_signal is not None,
            "direction_age_s": (
                time.time() - self._signals.direction_computed_at
                if self._signals.direction_computed_at > 0 else -1
            ),
            "weakness_age_s": (
                time.time() - self._signals.weakness_computed_at
                if self._signals.weakness_computed_at > 0 else -1
            ),
        }

    def _precompute_direction(self):
        """
        Pre-compute 4H direction signal.

        Fetches fresh 4H data and runs DirectionDetector to cache
        the current directional bias. This changes very rarely
        (only when a new 4H candle forms with the right pattern).
        """
        self.logger.debug("Pre-computing 4H direction signal...")

        df_4h = self.data_feed.get_4h_data()
        if df_4h is None:
            self.logger.warning("Failed to fetch 4H data for pre-computation")
            return

        self._signals.df_4h = df_4h
        self._signals.direction_computed_at = time.time()

        try:
            # Import here to avoid circular imports at module level
            import sys
            import os
            _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from strategy import DirectionDetector

            detector = DirectionDetector(df_4h)
            detector.detect()
            signals = detector.signals

            if signals:
                # Cache the most recent direction signal
                self._signals.direction_signal = signals[-1]
                self.logger.info(
                    f"Direction pre-computed: {signals[-1].direction} "
                    f"(from {signals[-1].time})"
                )
            else:
                self._signals.direction_signal = None
                self.logger.debug("No active direction signal found")
        except Exception as e:
            self.logger.error(f"Direction pre-computation error: {e}", exc_info=True)

    def _precompute_weakness(self):
        """
        Pre-compute 15M weakness signal.

        Fetches fresh 15M data and runs WeaknessDetector to cache
        weakness signals. This should be refreshed every 15 minutes
        (on each new 15M candle close).
        """
        self.logger.debug("Pre-computing 15M weakness signal...")

        # Need direction signal to compute weakness
        if self._signals.direction_signal is None:
            self.logger.debug("No direction signal - skipping weakness pre-computation")
            return

        df_15m = self.data_feed.get_15m_data()
        if df_15m is None:
            self.logger.warning("Failed to fetch 15M data for pre-computation")
            return

        self._signals.df_15m = df_15m
        self._signals.weakness_computed_at = time.time()

        try:
            import sys
            import os
            _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from strategy import WeaknessDetector

            detector = WeaknessDetector(
                df_15m,
                direction_signal=self._signals.direction_signal,
            )
            detector.detect()
            signals = detector.signals

            if signals:
                self._signals.weakness_signal = signals[-1]
                self.logger.info(
                    f"Weakness pre-computed: rectangle "
                    f"[{signals[-1].rectangle_bottom:.5f} - "
                    f"{signals[-1].rectangle_top:.5f}]"
                )
            else:
                self._signals.weakness_signal = None
                self.logger.debug("No active weakness signal found")
        except Exception as e:
            self.logger.error(f"Weakness pre-computation error: {e}", exc_info=True)

    def _update_signal_state(self):
        """
        Update the signal readiness state based on cached signals.

        State transitions:
        - IDLE: No direction signal
        - SCANNING: Direction confirmed, looking for weakness
        - ARMED: Both direction + weakness confirmed, ready for instant entry
        """
        if self._signals.direction_signal is None:
            self._signals.state = SignalState.IDLE
        elif self._signals.weakness_signal is None:
            self._signals.state = SignalState.SCANNING
        else:
            self._signals.state = SignalState.ARMED

    def reset(self):
        """Reset the tick monitor state (e.g., after a trade is executed)."""
        self._signals = PreComputedSignals()
        self._last_1m_candle_time = None
        self.logger.info("TickMonitor state reset")
