"""
=============================================================
  Circuit Breaker for Correlated Model Failures (Groupthink Detection)

  Tracks agreement ratio across all 17 models each prediction cycle.
  If agreement >= 95% for 3+ consecutive cycles: DANGER level
  If agreement >= 85% for 2+ consecutive cycles: CAUTION level

  Groupthink = all models voting the same direction = systemic risk.
  When detected, applies a confidence penalty to the Bayesian calculation.
=============================================================
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for the groupthink circuit breaker."""
    # Thresholds
    danger_agreement_threshold: float = 0.95    # 95% agreement = DANGER
    caution_agreement_threshold: float = 0.85   # 85% agreement = CAUTION
    # Consecutive cycles required
    danger_consecutive_cycles: int = 3           # 3 cycles at 95%+ = DANGER
    caution_consecutive_cycles: int = 2          # 2 cycles at 85%+ = CAUTION
    # Confidence penalties
    danger_confidence_penalty: float = 0.30      # Severe penalty in DANGER
    caution_confidence_penalty: float = 0.10     # Mild penalty in CAUTION
    # History size
    history_size: int = 50                       # Keep last 50 agreement ratios
    # Number of models in ensemble
    num_models: int = 17


class CircuitBreakerState:
    """Tracks the current state of the circuit breaker."""

    def __init__(self):
        self.level: str = "NORMAL"           # NORMAL, CAUTION, DANGER
        self.consecutive_high: int = 0       # Consecutive cycles above caution threshold
        self.consecutive_extreme: int = 0    # Consecutive cycles above danger threshold
        self.current_agreement: float = 0.0
        self.confidence_penalty: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "consecutive_high": self.consecutive_high,
            "consecutive_extreme": self.consecutive_extreme,
            "current_agreement": self.current_agreement,
            "confidence_penalty": self.confidence_penalty,
        }


class GroupthinkCircuitBreaker:
    """
    Monitors ensemble model agreement for groupthink detection.

    When all models start voting in the same direction, it is usually
    a sign of:
    1. Overfitting to recent data (all models trained on same patterns)
    2. Market regime that looks obvious but is about to reverse
    3. Correlated failure mode (all models fooled by same data artifact)

    Professional trading desks watch for this - when ALL analysts agree,
    something is usually wrong. Contrarian indicator at extreme agreement.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._agreement_history: deque = deque(maxlen=self.config.history_size)
        self._prediction_history: deque = deque(maxlen=self.config.history_size)

    def update(self, individual_predictions: Dict[str, np.ndarray]) -> CircuitBreakerState:
        """
        Update the circuit breaker with new ensemble predictions.

        Args:
            individual_predictions: Dict mapping model_name to probability array
                                   [p_buy, p_sell, p_hold] for each model.

        Returns:
            Updated CircuitBreakerState with level and penalty.
        """
        if not individual_predictions:
            return self.state

        # Compute agreement ratio
        agreement = self._compute_agreement(individual_predictions)
        self._agreement_history.append(agreement)
        self.state.current_agreement = agreement

        # Track dominant direction votes
        directions = []
        for model_name, probs in individual_predictions.items():
            if isinstance(probs, np.ndarray) and len(probs) >= 3:
                directions.append(int(np.argmax(probs)))
            elif isinstance(probs, (list, tuple)) and len(probs) >= 3:
                directions.append(int(np.argmax(probs)))

        # Store for analysis
        self._prediction_history.append(directions)

        # Update consecutive counters
        if agreement >= self.config.danger_agreement_threshold:
            self.state.consecutive_extreme += 1
            self.state.consecutive_high += 1
        elif agreement >= self.config.caution_agreement_threshold:
            self.state.consecutive_extreme = 0
            self.state.consecutive_high += 1
        else:
            self.state.consecutive_extreme = 0
            self.state.consecutive_high = 0

        # Determine level
        old_level = self.state.level

        if (self.state.consecutive_extreme
                >= self.config.danger_consecutive_cycles):
            self.state.level = "DANGER"
            self.state.confidence_penalty = self.config.danger_confidence_penalty
            if old_level != "DANGER":
                logger.warning(
                    "[CircuitBreaker] DANGER: %d consecutive cycles with "
                    "%.1f%% agreement. Groupthink detected! "
                    "Applying %.0f%% confidence penalty.",
                    self.state.consecutive_extreme,
                    agreement * 100,
                    self.config.danger_confidence_penalty * 100
                )
        elif (self.state.consecutive_high
              >= self.config.caution_consecutive_cycles):
            self.state.level = "CAUTION"
            self.state.confidence_penalty = self.config.caution_confidence_penalty
            if old_level == "NORMAL":
                logger.warning(
                    "[CircuitBreaker] CAUTION: %d consecutive cycles with "
                    "%.1f%% agreement. Possible groupthink.",
                    self.state.consecutive_high,
                    agreement * 100
                )
        else:
            self.state.level = "NORMAL"
            self.state.confidence_penalty = 0.0
            if old_level != "NORMAL":
                logger.info(
                    "[CircuitBreaker] Returned to NORMAL. "
                    "Agreement: %.1f%%", agreement * 100
                )

        return self.state

    def _compute_agreement(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Compute the agreement ratio across all models.

        Agreement = fraction of models voting for the same direction (argmax).
        E.g., if 16/17 models say BUY, agreement = 16/17 = 0.94

        Args:
            predictions: Dict of model_name -> probability array [buy, sell, hold]

        Returns:
            Agreement ratio in [0, 1]
        """
        if not predictions:
            return 0.0

        votes = []
        for model_name, probs in predictions.items():
            if isinstance(probs, np.ndarray) and len(probs) >= 3:
                votes.append(int(np.argmax(probs)))
            elif isinstance(probs, (list, tuple)) and len(probs) >= 3:
                votes.append(int(np.argmax(probs)))

        if not votes:
            return 0.0

        # Count the most common vote
        vote_counts = np.bincount(votes, minlength=3)
        max_votes = int(np.max(vote_counts))
        total = len(votes)

        return max_votes / total if total > 0 else 0.0

    def get_penalty(self) -> float:
        """Get the current confidence penalty (0.0 = no penalty)."""
        return self.state.confidence_penalty

    def get_status(self) -> Dict:
        """Get the full circuit breaker status for logging/monitoring."""
        avg_agreement = (
            float(np.mean(list(self._agreement_history)))
            if self._agreement_history else 0.0
        )
        return {
            "level": self.state.level,
            "current_agreement": self.state.current_agreement,
            "avg_agreement": avg_agreement,
            "consecutive_high": self.state.consecutive_high,
            "consecutive_extreme": self.state.consecutive_extreme,
            "confidence_penalty": self.state.confidence_penalty,
            "history_length": len(self._agreement_history),
        }

    def get_bayesian_evidence_key(self) -> Optional[str]:
        """
        Get the Bayesian evidence key for the current circuit breaker state.
        Integrates into the Brain's Bayesian confidence calculation as
        'groupthink_penalty' evidence.

        Returns:
            Evidence key string or None if no penalty applies.
        """
        if self.state.level == "DANGER":
            return "groupthink_danger"
        elif self.state.level == "CAUTION":
            return "groupthink_caution"
        return None

    def reset(self):
        """Reset the circuit breaker state (e.g., on daily reset)."""
        self.state = CircuitBreakerState()
        logger.info("[CircuitBreaker] State reset to NORMAL")
