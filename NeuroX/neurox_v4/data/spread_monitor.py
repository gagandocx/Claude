"""
=============================================================
  Python ML Bridge v3 - Spread Monitor (Tier 2)

  Reads the spread CSV file written by the EA every tick.
  Format: timestamp,spread_points,ask,bid

  Gates trade entries based on spread conditions:
    - Only enter when current spread < max_spread_multiplier * average
    - Provides spread features for model input
    - Tracks spread velocity and percentiles for regime awareness

  This prevents entries during high-spread periods (news,
  low liquidity, Asian session widening) that eat into profits.
=============================================================
"""

import os
import sys
import time
import logging
from typing import Dict, Optional
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SpreadGateConfig, SPREAD_FILE

logger = logging.getLogger(__name__)


class SpreadMonitor:
    """
    Monitors real-time spread data written by the EA.

    The EA writes a CSV row every tick with: timestamp,spread_points,ask,bid
    Python reads this file to determine if spread conditions are acceptable
    for entry. This prevents slippage-heavy entries during illiquid periods.

    Attributes:
        config: SpreadGateConfig with thresholds and file path
        _spread_buffer: Rolling buffer of recent spread readings
        _last_read_time: Timestamp of last successful file read
        _last_mtime: File modification time of last read (avoid re-reading unchanged file)
    """

    def __init__(self, config: Optional[SpreadGateConfig] = None):
        self.config = config or SpreadGateConfig()

        # Rolling spread buffer for averaging
        self._spread_buffer: deque = deque(maxlen=500)
        self._last_read_time: float = 0.0
        self._last_mtime: float = 0.0

        # Current spread data
        self._current_spread: float = 0.0
        self._current_ask: float = 0.0
        self._current_bid: float = 0.0
        self._last_timestamp: str = ""

        logger.info("[SpreadMonitor] Initialized. file=%s, "
                    "max_multiplier=%.2f, update_interval=%dms",
                    self.config.spread_file,
                    self.config.max_spread_multiplier,
                    self.config.update_interval_ms)

    def _read_spread_file(self) -> bool:
        """
        Read the latest spread data from the EA-written CSV file.

        File format (one row, overwritten each tick by EA):
            timestamp,spread_points,ask,bid

        Returns:
            True if successfully read new data, False otherwise.
        """
        spread_path = self.config.spread_file

        if not os.path.exists(spread_path):
            return False

        try:
            # Check if file was modified since last read
            mtime = os.path.getmtime(spread_path)
            if mtime <= self._last_mtime:
                return False  # No new data

            with open(spread_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                return False

            # Read the last line (most recent)
            last_line = lines[-1].strip()
            if not last_line:
                if len(lines) > 1:
                    last_line = lines[-2].strip()
                else:
                    return False

            parts = last_line.split(',')
            if len(parts) < 4:
                # Try reading all lines into buffer for historical data
                for line in lines[-100:]:
                    line = line.strip()
                    if not line:
                        continue
                    p = line.split(',')
                    if len(p) >= 4:
                        try:
                            spread_val = float(p[1])
                            self._spread_buffer.append(spread_val)
                        except (ValueError, IndexError):
                            continue
                return False

            timestamp_str = parts[0].strip()
            spread_points = float(parts[1].strip())
            ask_price = float(parts[2].strip())
            bid_price = float(parts[3].strip())

            self._current_spread = spread_points
            self._current_ask = ask_price
            self._current_bid = bid_price
            self._last_timestamp = timestamp_str
            self._last_read_time = time.time()
            self._last_mtime = mtime

            # Add to rolling buffer
            self._spread_buffer.append(spread_points)

            # Also read historical lines into buffer if available
            for line in lines[-100:]:
                line = line.strip()
                if not line or line == last_line:
                    continue
                p = line.split(',')
                if len(p) >= 2:
                    try:
                        sv = float(p[1].strip())
                        if sv not in list(self._spread_buffer)[-5:]:
                            self._spread_buffer.append(sv)
                    except (ValueError, IndexError):
                        continue

            return True

        except (IOError, OSError, ValueError) as e:
            logger.debug("[SpreadMonitor] Read error: %s", e)
            return False

    def get_current_spread(self) -> float:
        """
        Get the most recent spread value in points.

        Reads the spread file if update interval has elapsed.

        Returns:
            Current spread in points. Returns 0.0 if no data available.
        """
        # Respect update interval to avoid excessive I/O
        elapsed_ms = (time.time() - self._last_read_time) * 1000
        if elapsed_ms >= self.config.update_interval_ms:
            self._read_spread_file()

        return self._current_spread

    def get_average_spread(self, window: int = 100) -> float:
        """
        Get the average spread over the last N readings.

        Args:
            window: Number of recent spread values to average.
                    Defaults to 100.

        Returns:
            Average spread in points. Returns current spread if
            insufficient history (graceful degradation).
        """
        # Ensure we have recent data
        self.get_current_spread()

        if not self._spread_buffer:
            return self._current_spread if self._current_spread > 0 else 0.0

        # Use at most 'window' recent values
        recent = list(self._spread_buffer)[-window:]
        if not recent:
            return self._current_spread

        return float(np.mean(recent))

    def is_spread_acceptable(self) -> bool:
        """
        Check if current spread is acceptable for trade entry.

        Acceptable means: current_spread < max_spread_multiplier * average_spread

        If no spread data is available (EA not running / file doesn't exist),
        returns True (fail-open: don't block trades without evidence).

        Returns:
            True if spread is OK for entry, False if too wide.
        """
        current = self.get_current_spread()

        # No data available - fail open (allow trading)
        if current <= 0:
            return True

        average = self.get_average_spread()

        # If average is zero or very small, can't make a meaningful comparison
        if average <= 0:
            return True

        ratio = current / average
        is_ok = ratio <= self.config.max_spread_multiplier

        if not is_ok:
            logger.info("[SpreadMonitor] Spread TOO WIDE: current=%.1f pts, "
                        "avg=%.1f pts, ratio=%.2f (max=%.2f)",
                        current, average, ratio, self.config.max_spread_multiplier)

        return is_ok

    def get_spread_features(self) -> Dict[str, float]:
        """
        Get spread-related features for model input.

        Returns a dictionary with:
            - spread_ratio: current_spread / average_spread (normalized)
            - spread_percentile: where current spread sits in recent history [0,1]
            - spread_velocity: rate of change of spread (widening/narrowing)

        All values are normalized to be suitable for model input.
        """
        features = {
            'spread_ratio': 0.0,
            'spread_percentile': 0.5,
            'spread_velocity': 0.0,
        }

        current = self.get_current_spread()
        if current <= 0 or not self._spread_buffer:
            return features

        average = self.get_average_spread()

        # Spread ratio: current/average (1.0 = normal, >1 = wide, <1 = tight)
        if average > 0:
            features['spread_ratio'] = current / average
        else:
            features['spread_ratio'] = 1.0

        # Spread percentile: where current spread sits in recent history
        buffer_arr = np.array(list(self._spread_buffer))
        if len(buffer_arr) >= 5:
            percentile = float(np.sum(buffer_arr <= current)) / len(buffer_arr)
            features['spread_percentile'] = percentile

        # Spread velocity: change over last N readings
        if len(self._spread_buffer) >= 10:
            recent_10 = list(self._spread_buffer)[-10:]
            first_half = np.mean(recent_10[:5])
            second_half = np.mean(recent_10[5:])
            if first_half > 0:
                features['spread_velocity'] = (second_half - first_half) / first_half
            else:
                features['spread_velocity'] = 0.0

        return features

    def get_spread_info(self) -> Dict[str, float]:
        """
        Get full spread state for logging/debugging.

        Returns:
            Dict with current_spread, average_spread, ratio, ask, bid, is_acceptable
        """
        current = self.get_current_spread()
        average = self.get_average_spread()
        ratio = current / average if average > 0 else 0.0

        return {
            'current_spread': current,
            'average_spread': average,
            'spread_ratio': ratio,
            'ask': self._current_ask,
            'bid': self._current_bid,
            'is_acceptable': self.is_spread_acceptable(),
            'buffer_size': len(self._spread_buffer),
        }
