"""
=============================================================
  Python ML Bridge v3 - Adversarial Signal Filter (Tier 1)

  Before trading, checks if recent similar signals (same direction,
  similar time, similar price level) won or lost. If last N similar
  signals had a high loss rate, skips the current signal.

  Prevents repeated losses in adverse market conditions by learning
  from recent outcomes of similar setups.
=============================================================
"""

import os
import sys
import time
import logging
import hashlib
from typing import Dict, Optional, List, Tuple
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import AdversarialFilterConfig

logger = logging.getLogger(__name__)


class AdversarialFilter:
    """
    Adversarial signal filtering to prevent repeated losses.

    Institutional insight: If the market is consistently stopping out
    a particular type of trade (same direction, time of day, price zone),
    it's likely that conditions are adversarial for that setup.

    The filter maintains a rolling history of recent signals with their
    outcomes. Before generating a new signal, it finds similar past
    signals using cosine similarity on a feature vector and checks
    their win/loss rate. If the loss rate exceeds the threshold,
    the signal is skipped.

    Signal context vector components:
        - direction (buy=1, sell=-1)
        - hour_sin, hour_cos (time of day, circular)
        - price_level_normalized (relative to recent range)
        - confidence
        - regime_encoded (trending=1, ranging=0, volatile=-1)

    This captures the "personality" of a signal without overfitting
    to exact price values.
    """

    def __init__(self, config: Optional[AdversarialFilterConfig] = None):
        self.config = config or AdversarialFilterConfig()

        # Signal history: list of dicts with context + outcome
        self._signal_history: deque = deque(maxlen=self.config.lookback_signals)

        # Counter for signal IDs
        self._signal_counter: int = 0

        # Stats
        self._total_checked: int = 0
        self._total_skipped: int = 0

        logger.info("[AdversarialFilter] Initialized. lookback=%d, "
                    "similarity_threshold=%.2f, min_similar=%d, "
                    "loss_rate_threshold=%.2f",
                    self.config.lookback_signals,
                    self.config.similarity_threshold,
                    self.config.min_similar_signals,
                    self.config.loss_rate_threshold)

    def record_signal(self, direction: str, confidence: float,
                      regime: str, time_of_day: Optional[float] = None,
                      price_level: float = 0.0,
                      features_hash: Optional[str] = None) -> int:
        """
        Record a signal that was generated (before outcome is known).

        Args:
            direction: "BUY" or "SELL"
            confidence: Model confidence [0, 1]
            regime: Current regime name (e.g., "trending", "ranging")
            time_of_day: Hour of day (0-23.99) or None for current time
            price_level: Current price level (for zone detection)
            features_hash: Optional hash of feature vector for exact matching

        Returns:
            signal_id: Unique ID for this signal (used in record_outcome)
        """
        self._signal_counter += 1
        signal_id = self._signal_counter

        if time_of_day is None:
            now = datetime.now()
            time_of_day = now.hour + now.minute / 60.0

        # Build context vector for similarity comparison
        context_vector = self._build_context_vector(
            direction, confidence, regime, time_of_day, price_level
        )

        entry = {
            'signal_id': signal_id,
            'direction': direction,
            'confidence': confidence,
            'regime': regime,
            'time_of_day': time_of_day,
            'price_level': price_level,
            'context_vector': context_vector,
            'features_hash': features_hash,
            'timestamp': time.time(),
            'outcome': None,  # Will be filled by record_outcome
            'pnl': None,
        }

        self._signal_history.append(entry)
        return signal_id

    def record_outcome(self, signal_id: int, won: bool, pnl: float = 0.0) -> None:
        """
        Update a signal's outcome after the trade closes.

        Args:
            signal_id: ID returned by record_signal()
            won: True if trade was profitable
            pnl: Realized profit/loss
        """
        for entry in self._signal_history:
            if entry['signal_id'] == signal_id:
                entry['outcome'] = won
                entry['pnl'] = pnl
                logger.debug("[AdversarialFilter] Recorded outcome for signal %d: "
                             "won=%s, pnl=%.2f", signal_id, won, pnl)
                return

        logger.debug("[AdversarialFilter] Signal %d not found in history "
                     "(may have been evicted)", signal_id)

    def should_skip_signal(self, direction: str, confidence: float,
                           regime: str, time_of_day: Optional[float] = None,
                           price_level: float = 0.0) -> Tuple[bool, str]:
        """
        Check if a proposed signal should be skipped based on adversarial history.

        Finds similar past signals (cosine similarity > threshold) and
        checks their win/loss rate. Skips if loss rate exceeds threshold.

        Args:
            direction: Proposed direction ("BUY" or "SELL")
            confidence: Proposed confidence
            regime: Current regime
            time_of_day: Current time of day
            price_level: Current price level

        Returns:
            Tuple of (skip: bool, reason: str)
            skip=True means the signal should NOT be traded.
        """
        self._total_checked += 1

        if time_of_day is None:
            now = datetime.now()
            time_of_day = now.hour + now.minute / 60.0

        # Build context vector for the proposed signal
        query_vector = self._build_context_vector(
            direction, confidence, regime, time_of_day, price_level
        )

        # Find similar past signals with known outcomes
        similar_signals = self._find_similar_signals(query_vector)

        # Not enough history to make a decision
        if len(similar_signals) < self.config.min_similar_signals:
            return False, "insufficient_history"

        # Check loss rate among similar signals
        outcomes = [s['outcome'] for s in similar_signals]
        losses = sum(1 for o in outcomes if o is False)
        total = len(outcomes)
        loss_rate = losses / total

        if loss_rate > self.config.loss_rate_threshold:
            self._total_skipped += 1
            reason = (f"adversarial_skip: {losses}/{total} similar signals lost "
                      f"(loss_rate={loss_rate:.2f} > threshold={self.config.loss_rate_threshold:.2f})")
            logger.info("[AdversarialFilter] SKIP SIGNAL: %s %s conf=%.2f | %s",
                        direction, regime, confidence, reason)
            return True, reason

        return False, "pass"

    def _build_context_vector(self, direction: str, confidence: float,
                              regime: str, time_of_day: float,
                              price_level: float) -> np.ndarray:
        """
        Build a normalized context vector for similarity comparison.

        Components:
            [0] direction: BUY=1, SELL=-1
            [1] hour_sin: sin(2*pi*hour/24) for circular time
            [2] hour_cos: cos(2*pi*hour/24)
            [3] confidence: [0, 1]
            [4] regime: trending=1, ranging=0, volatile=-1
            [5] price_level_norm: normalized by a fixed scale

        Returns:
            np.ndarray of shape (6,)
        """
        # Direction
        dir_val = 1.0 if direction == "BUY" else -1.0

        # Circular time encoding
        hour_rad = 2 * np.pi * time_of_day / 24.0
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # Regime encoding
        regime_lower = regime.lower()
        if 'trend' in regime_lower or 'strong' in regime_lower:
            regime_val = 1.0
        elif 'rang' in regime_lower or 'sideways' in regime_lower:
            regime_val = 0.0
        elif 'volat' in regime_lower:
            regime_val = -1.0
        else:
            regime_val = 0.0

        # Price level: normalize to a reasonable scale
        # For gold (~$2000), divide by 1000 to get values around 2.0
        # This is a rough normalization; the cosine similarity handles
        # the comparison correctly regardless of absolute scale.
        price_norm = price_level / 1000.0 if price_level > 0 else 0.0

        vector = np.array([
            dir_val,
            hour_sin,
            hour_cos,
            confidence,
            regime_val,
            price_norm,
        ], dtype=np.float64)

        return vector

    def _find_similar_signals(self, query_vector: np.ndarray) -> List[Dict]:
        """
        Find past signals similar to the query using cosine similarity.

        Only returns signals with known outcomes (outcome is not None).

        Args:
            query_vector: Context vector for the proposed signal

        Returns:
            List of signal history entries with similarity > threshold
        """
        similar = []
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return similar

        for entry in self._signal_history:
            # Only compare signals with known outcomes
            if entry['outcome'] is None:
                continue

            hist_vector = entry['context_vector']
            hist_norm = np.linalg.norm(hist_vector)
            if hist_norm < 1e-10:
                continue

            # Cosine similarity
            similarity = np.dot(query_vector, hist_vector) / (query_norm * hist_norm)

            if similarity >= self.config.similarity_threshold:
                similar.append(entry)

        return similar

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        total_with_outcome = sum(
            1 for s in self._signal_history if s['outcome'] is not None
        )
        total_wins = sum(
            1 for s in self._signal_history
            if s['outcome'] is True
        )

        return {
            'total_checked': self._total_checked,
            'total_skipped': self._total_skipped,
            'skip_rate': self._total_skipped / max(self._total_checked, 1),
            'history_size': len(self._signal_history),
            'signals_with_outcome': total_with_outcome,
            'historical_win_rate': total_wins / max(total_with_outcome, 1),
        }

    def reset(self):
        """Reset the filter state."""
        self._signal_history.clear()
        self._total_checked = 0
        self._total_skipped = 0
        self._signal_counter = 0
        logger.info("[AdversarialFilter] State reset")
