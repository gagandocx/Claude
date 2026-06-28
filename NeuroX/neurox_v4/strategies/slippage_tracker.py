"""
=============================================================
  NeuroX - Slippage / Execution Quality Tracker
  Maintains a rolling window of slippage values per direction
  (BUY/SELL), computes average slippage, detects degradation,
  and exposes a fill quality score (0-1).
=============================================================
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class SlippageTrackerConfig:
    """Slippage tracker configuration."""
    # Rolling window size (number of fills to track)
    window_size: int = 100
    # Degradation threshold: slippage > N x rolling average = degraded
    degradation_multiplier: float = 2.0
    # Minimum fills before quality scoring is active
    min_fills_for_scoring: int = 5
    # Quality warning threshold (log warning below this)
    quality_warning_threshold: float = 0.7
    # Maximum expected slippage in points (for XAUUSD, typically <$1)
    max_expected_slippage: float = 1.0


# ─────────────────────────────────────────────
#  SLIPPAGE RECORD
# ─────────────────────────────────────────────
@dataclass
class SlippageRecord:
    """Single slippage measurement."""
    timestamp: float
    direction: str        # "BUY" or "SELL"
    requested_price: float
    fill_price: float
    slippage: float       # abs(fill - requested)
    ticket: str = ""


# ─────────────────────────────────────────────
#  SLIPPAGE TRACKER
# ─────────────────────────────────────────────
class SlippageTracker:
    """
    Rolling execution quality tracker.

    Tracks fill slippage per direction (BUY/SELL), computes rolling
    statistics, detects quality degradation, and provides a normalized
    fill quality score from 0 (worst) to 1 (perfect fills).

    Fill quality formula:
        quality = max(0, 1 - avg_slippage / max_expected_slippage)

    Degradation is detected when the latest slippage exceeds
    degradation_multiplier times the rolling average.
    """

    def __init__(self, config: Optional[SlippageTrackerConfig] = None):
        self.config = config or SlippageTrackerConfig()
        self.logger = logging.getLogger("NeuroX.SlippageTracker")

        # Rolling windows per direction
        self._buy_slippage: deque = deque(maxlen=self.config.window_size)
        self._sell_slippage: deque = deque(maxlen=self.config.window_size)
        self._all_slippage: deque = deque(maxlen=self.config.window_size)

        # Statistics
        self._total_fills = 0
        self._degradation_count = 0
        self._last_quality_score = 1.0

    def record_fill(self, direction: str, slippage: float,
                    requested_price: float = 0.0, fill_price: float = 0.0,
                    ticket: str = "") -> Dict:
        """
        Record a new fill and compute quality metrics.

        Args:
            direction: "BUY" or "SELL"
            slippage: Absolute slippage value (always positive)
            requested_price: Price requested at order submission
            fill_price: Actual fill price from broker
            ticket: Position ticket for tracking

        Returns:
            Dict with quality assessment:
                - slippage: the recorded slippage
                - avg_slippage: rolling average
                - quality_score: 0-1 score
                - is_degraded: True if this fill shows degradation
                - direction_avg: average for this direction
        """
        slippage = abs(slippage)
        record = SlippageRecord(
            timestamp=time.time(),
            direction=direction.upper(),
            requested_price=requested_price,
            fill_price=fill_price,
            slippage=slippage,
            ticket=ticket,
        )

        # Store in appropriate deque
        self._all_slippage.append(record)
        if direction.upper() == "BUY":
            self._buy_slippage.append(record)
        else:
            self._sell_slippage.append(record)

        self._total_fills += 1

        # Compute metrics
        result = self._compute_metrics(record)
        self._last_quality_score = result["quality_score"]

        return result

    def _compute_metrics(self, latest: SlippageRecord) -> Dict:
        """Compute rolling quality metrics after a new fill."""
        # Overall average
        if len(self._all_slippage) == 0:
            return {
                "slippage": latest.slippage,
                "avg_slippage": 0.0,
                "quality_score": 1.0,
                "is_degraded": False,
                "direction_avg": 0.0,
            }

        all_values = [r.slippage for r in self._all_slippage]
        avg_slippage = sum(all_values) / len(all_values)

        # Direction-specific average
        if latest.direction == "BUY":
            dir_values = [r.slippage for r in self._buy_slippage]
        else:
            dir_values = [r.slippage for r in self._sell_slippage]
        direction_avg = sum(dir_values) / len(dir_values) if dir_values else 0.0

        # Quality score (0-1, higher is better)
        quality_score = self.get_fill_quality()

        # Degradation detection
        is_degraded = False
        if (len(self._all_slippage) >= self.config.min_fills_for_scoring
                and avg_slippage > 0):
            threshold = avg_slippage * self.config.degradation_multiplier
            if latest.slippage > threshold:
                is_degraded = True
                self._degradation_count += 1

        return {
            "slippage": latest.slippage,
            "avg_slippage": avg_slippage,
            "quality_score": quality_score,
            "is_degraded": is_degraded,
            "direction_avg": direction_avg,
        }

    def get_fill_quality(self) -> float:
        """
        Get current fill quality score (0-1).

        1.0 = perfect fills (zero slippage)
        0.0 = maximum expected slippage exceeded

        Returns:
            Float between 0.0 and 1.0
        """
        if len(self._all_slippage) < self.config.min_fills_for_scoring:
            return 1.0  # Assume good quality until enough data

        all_values = [r.slippage for r in self._all_slippage]
        avg_slippage = sum(all_values) / len(all_values)

        quality = max(0.0, 1.0 - avg_slippage / self.config.max_expected_slippage)
        return round(quality, 4)

    def get_direction_quality(self, direction: str) -> float:
        """Get fill quality for a specific direction (BUY or SELL)."""
        if direction.upper() == "BUY":
            records = self._buy_slippage
        else:
            records = self._sell_slippage

        if len(records) < self.config.min_fills_for_scoring:
            return 1.0

        values = [r.slippage for r in records]
        avg = sum(values) / len(values)
        quality = max(0.0, 1.0 - avg / self.config.max_expected_slippage)
        return round(quality, 4)

    def get_stats(self) -> Dict:
        """Get comprehensive slippage statistics."""
        all_values = [r.slippage for r in self._all_slippage]
        buy_values = [r.slippage for r in self._buy_slippage]
        sell_values = [r.slippage for r in self._sell_slippage]

        return {
            "total_fills": self._total_fills,
            "window_fills": len(self._all_slippage),
            "avg_slippage": sum(all_values) / len(all_values) if all_values else 0.0,
            "max_slippage": max(all_values) if all_values else 0.0,
            "min_slippage": min(all_values) if all_values else 0.0,
            "buy_avg": sum(buy_values) / len(buy_values) if buy_values else 0.0,
            "sell_avg": sum(sell_values) / len(sell_values) if sell_values else 0.0,
            "buy_fills": len(buy_values),
            "sell_fills": len(sell_values),
            "quality_score": self.get_fill_quality(),
            "buy_quality": self.get_direction_quality("BUY"),
            "sell_quality": self.get_direction_quality("SELL"),
            "degradation_count": self._degradation_count,
        }

    def is_degraded(self) -> bool:
        """Check if recent fill quality is degraded."""
        return self._last_quality_score < self.config.quality_warning_threshold

    def reset(self):
        """Reset all tracked data."""
        self._buy_slippage.clear()
        self._sell_slippage.clear()
        self._all_slippage.clear()
        self._total_fills = 0
        self._degradation_count = 0
        self._last_quality_score = 1.0
