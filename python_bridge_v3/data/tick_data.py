"""
=============================================================
  Python ML Bridge v3 - Tick Data Processor (Tier 1)

  Reads real-time tick data written by MT5 EA to CSV:
    Format: timestamp,bid,ask,volume,flags

  Computes institutional order flow features:
    - bid_ask_imbalance: ratio of bid vs ask volume
    - volume_delta: cumulative delta of buy vs sell volume
    - trade_flow_intensity: ticks per second (activity rate)
    - large_trade_detection: volume > 3x average (institutional)

  Rolling window of max 5000 ticks for memory efficiency.
=============================================================
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, List
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TickDataConfig

logger = logging.getLogger(__name__)


class TickDataProcessor:
    """
    Processes real-time tick data from MT5 EA for order flow features.

    The EA writes tick data to a CSV file in the Common Files folder.
    This class polls the file, parses ticks, and computes features
    suitable for model input (numpy arrays).

    Tick CSV format (written by EA):
        timestamp,bid,ask,volume,flags
        2024.01.15 10:30:01.123,2025.50,2025.70,5,6
        ...

    Flags (MT5 TICK_FLAG):
        1 = TICK_FLAG_BID   (bid changed)
        2 = TICK_FLAG_ASK   (ask changed)
        4 = TICK_FLAG_LAST  (last deal price changed)
        8 = TICK_FLAG_VOLUME (volume changed)
        16 = TICK_FLAG_BUY  (buy trade)
        32 = TICK_FLAG_SELL (sell trade)
    """

    # MT5 tick flags
    TICK_FLAG_BID = 1
    TICK_FLAG_ASK = 2
    TICK_FLAG_LAST = 4
    TICK_FLAG_VOLUME = 8
    TICK_FLAG_BUY = 16
    TICK_FLAG_SELL = 32

    def __init__(self, config: Optional[TickDataConfig] = None):
        self.config = config or TickDataConfig()

        # Rolling window of ticks (max_ticks capacity)
        self._ticks: deque = deque(maxlen=self.config.max_ticks)
        self._last_file_mtime: float = 0.0
        self._last_poll_time: float = 0.0
        self._features_cache: Optional[Dict[str, float]] = None
        self._features_cache_time: float = 0.0

        logger.info("[TickData] Initialized. File: %s, Max ticks: %d",
                    self.config.tick_file, self.config.max_ticks)

    def poll_tick_file(self) -> int:
        """
        Read latest ticks from the CSV file written by MT5 EA.

        The EA rewrites the entire tick buffer (up to 5000 ticks) on each
        write cycle (~100ms). Therefore, we read the entire file contents
        on each poll and replace our internal deque. Incremental seek-based
        reads would be invalid since the EA truncates and rewrites the file.

        Handles file rotation gracefully.

        Returns:
            Number of new ticks read
        """
        tick_file = self.config.tick_file
        if not os.path.exists(tick_file):
            return 0

        try:
            # Check if file was modified
            mtime = os.path.getmtime(tick_file)
            if mtime <= self._last_file_mtime:
                return 0

            new_ticks = 0
            parsed_ticks = []
            with open(tick_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('timestamp'):
                        continue  # Skip header or empty lines

                    tick = self._parse_tick_line(line)
                    if tick is not None:
                        parsed_ticks.append(tick)
                        new_ticks += 1

            self._last_file_mtime = mtime
            self._last_poll_time = time.time()

            if parsed_ticks:
                # Replace entire deque with fresh file contents
                self._ticks.clear()
                for tick in parsed_ticks:
                    self._ticks.append(tick)

                logger.debug("[TickData] Read %d ticks from file (total: %d)",
                             new_ticks, len(self._ticks))
                # Invalidate features cache
                self._features_cache = None

            return new_ticks

        except Exception as e:
            logger.warning("[TickData] Error reading tick file: %s", e)
            return 0

    def _parse_tick_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single tick CSV line.

        Expected format: timestamp,bid,ask,volume,flags
        Example: 2024.01.15 10:30:01.123,2025.50,2025.70,5,6

        Returns:
            Dict with keys: timestamp, bid, ask, volume, flags, spread
            or None if parsing fails
        """
        try:
            parts = line.split(',')
            if len(parts) < 5:
                return None

            timestamp_str = parts[0].strip()
            bid = float(parts[1].strip())
            ask = float(parts[2].strip())
            volume = float(parts[3].strip())
            flags = int(parts[4].strip())

            return {
                'timestamp': timestamp_str,
                'time': time.time(),  # System time for rate calculations
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'flags': flags,
                'spread': ask - bid,
                'mid': (bid + ask) / 2.0,
                'is_buy': bool(flags & self.TICK_FLAG_BUY),
                'is_sell': bool(flags & self.TICK_FLAG_SELL),
            }
        except (ValueError, IndexError) as e:
            logger.debug("[TickData] Parse error on line: %s (%s)", line[:50], e)
            return None

    def compute_tick_features(self) -> Dict[str, float]:
        """
        Compute order flow features from the tick rolling window.

        Features computed:
            - bid_ask_imbalance: (buy_volume - sell_volume) / total_volume
              Range [-1, 1]. Positive = buying pressure, negative = selling.
            - volume_delta: Cumulative net volume (buys - sells) over window.
              Normalized by window size. Indicates persistent order flow.
            - trade_flow_intensity: Ticks per second over recent window.
              High = active market, low = quiet market.
            - large_trade_detection: Fraction of recent ticks with volume > 3x avg.
              High = institutional activity detected.

        Returns:
            Dict with feature names as keys and float values.
            Returns zeros if insufficient tick data.
        """
        features = {
            'bid_ask_imbalance': 0.0,
            'volume_delta': 0.0,
            'trade_flow_intensity': 0.0,
            'large_trade_detection': 0.0,
        }

        if len(self._ticks) < 10:
            return features

        # Use cached if still valid (within poll interval)
        if (self._features_cache is not None and
                time.time() - self._features_cache_time < self.config.poll_interval_ms / 1000.0):
            return self._features_cache

        ticks = list(self._ticks)
        n = len(ticks)

        # --- Bid/Ask Imbalance ---
        # Compute from flag-based buy/sell classification
        buy_volume = 0.0
        sell_volume = 0.0
        for tick in ticks:
            vol = tick['volume']
            if tick['is_buy']:
                buy_volume += vol
            elif tick['is_sell']:
                sell_volume += vol
            else:
                # If no buy/sell flag, classify by price movement
                # Uptick = buy, downtick = sell
                buy_volume += vol * 0.5
                sell_volume += vol * 0.5

        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            features['bid_ask_imbalance'] = (buy_volume - sell_volume) / total_volume
        else:
            features['bid_ask_imbalance'] = 0.0

        # --- Volume Delta ---
        # Cumulative net volume normalized by tick count
        volume_delta = buy_volume - sell_volume
        features['volume_delta'] = volume_delta / max(n, 1)

        # --- Trade Flow Intensity (ticks per second) ---
        if n >= 2:
            time_span = ticks[-1]['time'] - ticks[0]['time']
            if time_span > 0:
                features['trade_flow_intensity'] = n / time_span
            else:
                features['trade_flow_intensity'] = float(n)
        else:
            features['trade_flow_intensity'] = 0.0

        # --- Large Trade Detection ---
        # Fraction of ticks with volume > 3x average
        volumes = np.array([t['volume'] for t in ticks])
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
        if avg_volume > 0:
            large_threshold = avg_volume * 3.0
            large_count = np.sum(volumes > large_threshold)
            features['large_trade_detection'] = float(large_count) / n
        else:
            features['large_trade_detection'] = 0.0

        # Cache the result
        self._features_cache = features
        self._features_cache_time = time.time()

        return features

    def get_tick_count(self) -> int:
        """Return current number of ticks in the rolling window."""
        return len(self._ticks)

    def get_recent_ticks(self, n: int = 100) -> List[Dict]:
        """Get the most recent N ticks."""
        ticks = list(self._ticks)
        return ticks[-n:] if len(ticks) > n else ticks

    def get_spread_stats(self) -> Dict[str, float]:
        """
        Compute spread statistics from recent ticks.

        Returns:
            Dict with avg_spread, min_spread, max_spread, current_spread
        """
        if len(self._ticks) < 2:
            return {'avg_spread': 0.0, 'min_spread': 0.0,
                    'max_spread': 0.0, 'current_spread': 0.0}

        spreads = np.array([t['spread'] for t in self._ticks])
        return {
            'avg_spread': float(np.mean(spreads)),
            'min_spread': float(np.min(spreads)),
            'max_spread': float(np.max(spreads)),
            'current_spread': float(spreads[-1]),
        }

    def reset(self):
        """Reset the tick processor state."""
        self._ticks.clear()
        self._last_file_mtime = 0.0
        self._features_cache = None
        logger.info("[TickData] State reset")
