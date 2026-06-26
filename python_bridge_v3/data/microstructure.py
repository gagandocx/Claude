"""
=============================================================
  Python ML Bridge v3 - Microstructure Analyzer (Tier 2)

  Computes institutional-grade microstructure signals from
  tick data stream:
    - tick_rate: ticks per second over configurable window
    - bid_ask_bounce_rate: alternation between bid-side/ask-side hits
    - large_order_flow: net volume from trades > threshold * avg
    - spread_velocity: rate of spread change

  These feed into the feature vector alongside existing technical
  indicators for enhanced model input.
=============================================================
"""

import os
import sys
import logging
from typing import Dict, Optional, List
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MicrostructureConfig

logger = logging.getLogger(__name__)


class MicrostructureAnalyzer:
    """
    Computes microstructure features from tick-level data.

    Works in conjunction with TickDataProcessor. Receives tick data
    (list of tick dicts) and computes higher-order features that
    capture institutional order flow patterns invisible on OHLCV.

    Features:
        - tick_rate: Activity level (ticks/sec). High = active, low = quiet.
        - bid_ask_bounce_rate: How often price alternates between bid-side
          and ask-side hits. High bounce rate indicates ranging/indecision.
          Low bounce rate indicates trending (one-sided pressure).
        - large_order_flow: Net signed volume from unusually large ticks.
          Positive = institutional buying, negative = institutional selling.
        - spread_velocity: Rate of spread change (d(spread)/dt).
          Widening spread = upcoming volatility/news. Narrowing = stable.
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self.config = config or MicrostructureConfig()

        # Rolling history for trend calculations
        self._spread_history: deque = deque(maxlen=200)
        self._tick_rate_history: deque = deque(maxlen=100)

        logger.info("[Microstructure] Initialized. tick_rate_window=%ds, "
                    "large_order_threshold=%.1f sigma, bounce_window=%d ticks",
                    self.config.tick_rate_window,
                    self.config.large_order_threshold,
                    self.config.bounce_rate_window)

    def compute_features(self, ticks: List[Dict]) -> Dict[str, float]:
        """
        Compute all microstructure features from a list of ticks.

        Args:
            ticks: List of tick dicts from TickDataProcessor.get_recent_ticks()
                   Each tick has: time, bid, ask, volume, flags, is_buy, is_sell, spread, mid

        Returns:
            Dict with feature names as keys and float values.
            Returns zeros if insufficient data.
        """
        features = {
            'tick_rate': 0.0,
            'bid_ask_bounce_rate': 0.0,
            'large_order_flow': 0.0,
            'spread_velocity': 0.0,
        }

        if not ticks or len(ticks) < 5:
            return features

        features['tick_rate'] = self._compute_tick_rate(ticks)
        features['bid_ask_bounce_rate'] = self._compute_bounce_rate(ticks)
        features['large_order_flow'] = self._compute_large_order_flow(ticks)
        features['spread_velocity'] = self._compute_spread_velocity(ticks)

        return features

    def _compute_tick_rate(self, ticks: List[Dict]) -> float:
        """
        Compute ticks per second over the configured window.

        Uses only ticks within the last tick_rate_window seconds
        for an accurate real-time activity measurement.

        Returns:
            Float: ticks per second
        """
        if len(ticks) < 2:
            return 0.0

        window = self.config.tick_rate_window
        current_time = ticks[-1]['time']
        cutoff_time = current_time - window

        # Count ticks within window
        recent_ticks = [t for t in ticks if t['time'] >= cutoff_time]
        n_recent = len(recent_ticks)

        if n_recent < 2:
            return 0.0

        time_span = recent_ticks[-1]['time'] - recent_ticks[0]['time']
        if time_span <= 0:
            return float(n_recent)

        rate = n_recent / time_span
        self._tick_rate_history.append(rate)
        return rate

    def _compute_bounce_rate(self, ticks: List[Dict]) -> float:
        """
        Compute bid-ask bounce rate over the configured window.

        Bounce rate = fraction of ticks where the price direction alternated
        (went from bid-side to ask-side or vice versa). High bounce rate
        indicates choppy/ranging conditions; low bounce rate indicates
        one-directional flow (trending).

        Classification:
            - If mid price moved up from previous tick: ask-side hit
            - If mid price moved down from previous tick: bid-side hit

        Returns:
            Float in [0, 1]: fraction of alternations in the window
        """
        window = min(self.config.bounce_rate_window, len(ticks))
        if window < 3:
            return 0.0

        recent = ticks[-window:]
        alternations = 0
        total_moves = 0

        prev_direction = None  # True = up (ask-side), False = down (bid-side)
        for i in range(1, len(recent)):
            price_change = recent[i]['mid'] - recent[i - 1]['mid']
            if abs(price_change) < 1e-10:
                continue  # Skip if no price change

            current_direction = price_change > 0
            total_moves += 1

            if prev_direction is not None and current_direction != prev_direction:
                alternations += 1

            prev_direction = current_direction

        if total_moves < 2:
            return 0.0

        return alternations / total_moves

    def _compute_large_order_flow(self, ticks: List[Dict]) -> float:
        """
        Compute net signed volume from unusually large ticks.

        Large orders are defined as ticks with volume > threshold * avg_volume.
        The net flow (buy_large - sell_large) indicates institutional direction.

        Returns:
            Float: normalized net large order volume.
            Positive = institutional buying, negative = institutional selling.
        """
        if len(ticks) < 10:
            return 0.0

        volumes = np.array([t['volume'] for t in ticks])
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        if avg_volume <= 0 or std_volume <= 0:
            return 0.0

        threshold = avg_volume + self.config.large_order_threshold * std_volume

        large_buy_volume = 0.0
        large_sell_volume = 0.0

        for tick in ticks:
            if tick['volume'] > threshold:
                if tick['is_buy']:
                    large_buy_volume += tick['volume']
                elif tick['is_sell']:
                    large_sell_volume += tick['volume']
                else:
                    # Classify by price movement relative to mid
                    # If closer to ask, likely a buy; if closer to bid, likely a sell
                    mid = tick['mid']
                    last_price = tick.get('bid', mid)
                    if last_price >= mid:
                        large_buy_volume += tick['volume']
                    else:
                        large_sell_volume += tick['volume']

        total_large = large_buy_volume + large_sell_volume
        if total_large <= 0:
            return 0.0

        # Normalize to [-1, 1]
        net_flow = (large_buy_volume - large_sell_volume) / total_large
        return net_flow

    def _compute_spread_velocity(self, ticks: List[Dict]) -> float:
        """
        Compute rate of spread change over recent ticks.

        Uses linear regression slope of spread values over time.
        Positive = spread widening (risk increasing).
        Negative = spread narrowing (conditions improving).

        Returns:
            Float: spread change per second (normalized)
        """
        if len(ticks) < 5:
            return 0.0

        # Use last 50 ticks or all available
        window = min(50, len(ticks))
        recent = ticks[-window:]

        spreads = np.array([t['spread'] for t in recent])
        times = np.array([t['time'] for t in recent])

        # Normalize time to start from 0
        times = times - times[0]
        time_span = times[-1]
        if time_span <= 0:
            return 0.0

        # Update spread history
        self._spread_history.append(spreads[-1])

        # Linear regression: slope of spread over time
        # spread = a * time + b -> a is velocity
        n = len(times)
        if n < 3:
            return 0.0

        mean_t = np.mean(times)
        mean_s = np.mean(spreads)

        numerator = np.sum((times - mean_t) * (spreads - mean_s))
        denominator = np.sum((times - mean_t) ** 2)

        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator

        # Normalize by average spread to make it scale-independent
        avg_spread = np.mean(spreads)
        if avg_spread > 0:
            normalized_velocity = slope / avg_spread
        else:
            normalized_velocity = slope

        return float(normalized_velocity)

    def get_tick_rate_percentile(self) -> float:
        """
        Get current tick rate as percentile of recent history.

        Returns:
            Float in [0, 1]: 0 = very quiet, 1 = very active
        """
        if len(self._tick_rate_history) < 5:
            return 0.5  # Not enough history, assume average

        current = self._tick_rate_history[-1]
        history = np.array(list(self._tick_rate_history))
        percentile = np.sum(history <= current) / len(history)
        return float(percentile)

    def is_high_activity(self) -> bool:
        """Check if current tick rate is in the top 20% of recent history."""
        return self.get_tick_rate_percentile() > 0.80

    def is_low_activity(self) -> bool:
        """Check if current tick rate is in the bottom 20% of recent history."""
        return self.get_tick_rate_percentile() < 0.20

    def get_feature_vector(self, ticks: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Return a fixed-size numpy array of microstructure features suitable
        for concatenation with existing model input features.

        Features (6 elements):
            [0] tick_rate_normalized: ticks/sec normalized by history percentile
            [1] bid_ask_bounce_rate: fraction of alternations [0, 1]
            [2] large_order_flow_normalized: net large flow [-1, 1]
            [3] spread_velocity: normalized spread change rate
            [4] volume_acceleration: rate of change of tick volume
            [5] tick_imbalance_ratio: buy_ticks / total_ticks - 0.5 (centered)

        If ticks data is unavailable or insufficient, returns zeros (safe default).

        Args:
            ticks: Optional list of tick dicts. If None, returns zeros.

        Returns:
            np.ndarray of shape (6,) with float32 values.
        """
        # Fixed size: always 6 elements
        vector = np.zeros(6, dtype=np.float32)

        if not ticks or len(ticks) < 5:
            return vector

        # Compute base features
        features = self.compute_features(ticks)

        # [0] tick_rate_normalized: use percentile (0-1 range)
        vector[0] = self.get_tick_rate_percentile()

        # [1] bid_ask_bounce_rate: already in [0, 1]
        vector[1] = features.get('bid_ask_bounce_rate', 0.0)

        # [2] large_order_flow_normalized: already in [-1, 1]
        vector[2] = features.get('large_order_flow', 0.0)

        # [3] spread_velocity: clip to [-1, 1] for safety
        sv = features.get('spread_velocity', 0.0)
        vector[3] = float(np.clip(sv, -1.0, 1.0))

        # [4] volume_acceleration: rate of change of tick volume
        vector[4] = self._compute_volume_acceleration(ticks)

        # [5] tick_imbalance_ratio: buy fraction - 0.5 (centered around 0)
        vector[5] = self._compute_tick_imbalance(ticks)

        return vector

    def _compute_volume_acceleration(self, ticks: List[Dict]) -> float:
        """
        Compute volume acceleration (second derivative of cumulative volume).

        Compares volume in the most recent window vs the previous window
        to detect whether activity is increasing or decreasing.

        Returns:
            Float in roughly [-1, 1]: positive = accelerating, negative = decelerating
        """
        if len(ticks) < 20:
            return 0.0

        half = len(ticks) // 2
        first_half_vol = sum(t.get('volume', 0) for t in ticks[:half])
        second_half_vol = sum(t.get('volume', 0) for t in ticks[half:])

        total = first_half_vol + second_half_vol
        if total <= 0:
            return 0.0

        # Normalized acceleration: (recent - old) / total
        accel = (second_half_vol - first_half_vol) / total
        return float(np.clip(accel, -1.0, 1.0))

    def _compute_tick_imbalance(self, ticks: List[Dict]) -> float:
        """
        Compute tick imbalance ratio (buy fraction - 0.5).

        Counts buy ticks vs total classified ticks. Returns value
        centered around 0: positive = more buys, negative = more sells.

        Returns:
            Float in [-0.5, 0.5]
        """
        if len(ticks) < 5:
            return 0.0

        buy_count = 0
        sell_count = 0

        for tick in ticks[-100:]:  # Use last 100 ticks max
            if tick.get('is_buy', False):
                buy_count += 1
            elif tick.get('is_sell', False):
                sell_count += 1

        total = buy_count + sell_count
        if total == 0:
            return 0.0

        buy_fraction = buy_count / total
        return float(buy_fraction - 0.5)
