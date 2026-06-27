"""
=============================================================
  Python ML Bridge v8.0 - Multi-Model Disagreement Signal
  Tier 3: Institutional-Grade Feature

  When models strongly disagree on direction, it predicts an
  upcoming volatility spike. Used for:
    - Position sizing reduction (smaller lots in uncertainty)
    - Entry timing (wait for convergence)
    - TP widening (capture larger moves during expected vol)

  Tracks disagreement history and correlates with realized
  volatility for ongoing calibration.
=============================================================
"""

import logging
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DisagreementConfig

logger = logging.getLogger(__name__)


class DisagreementSignal:
    """
    Multi-model disagreement as a volatility forecasting signal.

    When the 17-model ensemble disagrees strongly on trade direction,
    it indicates market uncertainty which typically precedes volatility
    spikes. This class:

    1. Computes disagreement from individual model predictions
    2. Predicts volatility spikes based on disagreement + ATR
    3. Adjusts position size (reduce during high disagreement)
    4. Provides timing signals (wait / enter / reduce)
    5. Calibrates predictions against realized outcomes
    """

    def __init__(self, config: Optional[DisagreementConfig] = None):
        self.config = config or DisagreementConfig()

        # History tracking for calibration
        self._disagreement_history: deque = deque(maxlen=500)
        self._realized_vol_history: deque = deque(maxlen=500)
        self._calibration_pairs: deque = deque(maxlen=200)

        # Running statistics for normalization
        self._disagreement_mean: float = 0.3
        self._disagreement_std: float = 0.15
        self._vol_mean: float = 1.0
        self._vol_std: float = 0.5

        # Calibration coefficient (how much disagreement predicts vol)
        self._calibration_coeff: float = 1.0
        self._last_calibration_time: float = 0.0
        self._calibration_interval: float = 300.0  # Recalibrate every 5 min

        logger.info("[DisagreementSignal] Initialized with threshold=%.2f, "
                    "vol_scale=%.1f, reduction=%.0f%%",
                    self.config.strong_disagreement_threshold,
                    self.config.volatility_scale_factor,
                    self.config.position_reduction_pct * 100)

    def compute_disagreement(self, individual_predictions: dict) -> float:
        """
        Compute disagreement level from individual model predictions.

        Disagreement is measured as the standard deviation of model outputs
        normalized to [0, 1] range where:
            0.0 = total agreement (all models predict same direction)
            1.0 = maximum disagreement (models split evenly)

        Args:
            individual_predictions: Dict mapping model_name -> prediction array.
                Each prediction is shape (1, 3) with probabilities for [SELL, HOLD, BUY].

        Returns:
            Float in [0, 1] representing disagreement level.
        """
        if not individual_predictions:
            return 0.0

        # Extract the predicted direction for each model
        # Direction = argmax of the 3-class probability
        directions = []
        confidences = []

        for model_name, pred in individual_predictions.items():
            try:
                if isinstance(pred, np.ndarray):
                    if pred.ndim == 2:
                        pred = pred[0]  # shape (1,3) -> (3,)
                    direction = int(np.argmax(pred))
                    conf = float(np.max(pred))
                elif isinstance(pred, (list, tuple)):
                    pred_arr = np.array(pred).flatten()
                    direction = int(np.argmax(pred_arr))
                    conf = float(np.max(pred_arr))
                else:
                    continue

                directions.append(direction)
                confidences.append(conf)
            except (ValueError, TypeError, IndexError):
                continue

        if len(directions) < 2:
            return 0.0

        # Method 1: Entropy-based disagreement
        # Count votes for each direction
        direction_counts = np.zeros(3)
        for d in directions:
            if 0 <= d <= 2:
                direction_counts[d] += 1

        # Normalize to probabilities
        total = direction_counts.sum()
        if total == 0:
            return 0.0
        probs = direction_counts / total

        # Compute entropy (max entropy = log(3) for 3 classes)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p)
        max_entropy = np.log(3)
        entropy_disagreement = entropy / max_entropy if max_entropy > 0 else 0.0

        # Method 2: Confidence variance
        # Low average confidence + high variance = disagreement
        conf_array = np.array(confidences)
        conf_std = float(np.std(conf_array)) if len(conf_array) > 1 else 0.0
        conf_mean = float(np.mean(conf_array))
        # Invert confidence: low mean confidence = more disagreement
        confidence_disagreement = (1.0 - conf_mean) * 0.5 + conf_std

        # Combine both measures (weighted average)
        disagreement = 0.6 * entropy_disagreement + 0.4 * confidence_disagreement
        disagreement = float(np.clip(disagreement, 0.0, 1.0))

        # Update history for calibration
        self._disagreement_history.append(disagreement)
        self._update_running_stats()

        logger.debug("[DisagreementSignal] Disagreement=%.4f (entropy=%.4f, "
                     "conf_disagree=%.4f) from %d models",
                     disagreement, entropy_disagreement,
                     confidence_disagreement, len(directions))

        return disagreement

    def predict_volatility_spike(self, disagreement: float,
                                  current_atr: float) -> Tuple[float, float]:
        """
        Predict probability of a volatility spike based on disagreement level.

        Uses the calibrated relationship between past disagreement and
        subsequent realized volatility to estimate:
        1. Probability of a spike (disagreement > threshold historically
           preceded spikes X% of the time)
        2. Expected ATR multiplier (how much larger the next move might be)

        Args:
            disagreement: Current disagreement level [0, 1]
            current_atr: Current ATR value in price units

        Returns:
            Tuple of (spike_probability, expected_atr_multiplier)
            - spike_probability: [0, 1] chance of a vol spike
            - expected_atr_multiplier: expected ATR multiple (1.0 = normal)
        """
        if current_atr <= 0:
            return 0.0, 1.0

        # Base spike probability from disagreement (sigmoid mapping)
        # Strong disagreement -> high spike probability
        threshold = self.config.strong_disagreement_threshold
        # Sigmoid centered at threshold
        x = (disagreement - threshold) / 0.15  # steepness
        spike_prob = 1.0 / (1.0 + np.exp(-x))
        spike_prob = float(np.clip(spike_prob, 0.0, 1.0))

        # Expected ATR multiplier: linear interpolation
        # At disagreement=0: multiplier=1.0 (normal vol)
        # At disagreement=1: multiplier=volatility_scale_factor
        scale = self.config.volatility_scale_factor
        atr_multiplier = 1.0 + (scale - 1.0) * disagreement
        atr_multiplier = float(np.clip(atr_multiplier, 1.0, scale))

        # Apply calibration coefficient
        atr_multiplier = 1.0 + (atr_multiplier - 1.0) * self._calibration_coeff

        # Record for calibration
        self._record_prediction(disagreement, current_atr)

        logger.debug("[DisagreementSignal] Spike prediction: prob=%.3f, "
                     "ATR_mult=%.2f (disagreement=%.3f, ATR=%.2f)",
                     spike_prob, atr_multiplier, disagreement, current_atr)

        return spike_prob, atr_multiplier

    def get_position_adjustment(self, disagreement: float) -> float:
        """
        Get position size adjustment multiplier based on disagreement.

        High disagreement = smaller position (uncertainty = less risk).
        Low disagreement = full position (consensus = confidence).

        The mapping is:
            disagreement < threshold * 0.5  -> 1.0 (full size)
            disagreement = threshold        -> 0.75 (reduced)
            disagreement > threshold        -> position_reduction_pct (e.g. 0.5)

        Args:
            disagreement: Current disagreement level [0, 1]

        Returns:
            Float multiplier for lot size (0.0 to 1.0)
        """
        threshold = self.config.strong_disagreement_threshold
        reduction = self.config.position_reduction_pct

        if disagreement <= threshold * 0.5:
            # Low disagreement: full position
            return 1.0
        elif disagreement >= threshold:
            # High disagreement: reduced position
            # Linear interpolation from 0.75 at threshold to reduction at 1.0
            if disagreement >= 1.0:
                return reduction
            t = (disagreement - threshold) / (1.0 - threshold)
            multiplier = 0.75 - (0.75 - reduction) * t
            return float(np.clip(multiplier, reduction, 0.75))
        else:
            # Moderate disagreement: slight reduction
            # Linear from 1.0 to 0.75 between threshold*0.5 and threshold
            t = (disagreement - threshold * 0.5) / (threshold * 0.5)
            multiplier = 1.0 - 0.25 * t
            return float(np.clip(multiplier, 0.75, 1.0))

    def get_timing_signal(self, disagreement: float) -> str:
        """
        Get entry timing recommendation based on disagreement level.

        Three possible signals:
            'enter_now'   - Low disagreement, models agree, safe to enter
            'reduce_size' - Moderate disagreement, enter with reduced size
            'wait'        - High disagreement, wait for convergence

        Args:
            disagreement: Current disagreement level [0, 1]

        Returns:
            One of: 'enter_now', 'wait', 'reduce_size'
        """
        threshold = self.config.strong_disagreement_threshold

        if disagreement < threshold * 0.6:
            return 'enter_now'
        elif disagreement < threshold:
            return 'reduce_size'
        else:
            return 'wait'

    def record_realized_volatility(self, realized_atr: float):
        """
        Record the realized volatility after a disagreement measurement.
        Used for calibration of the prediction model.

        Args:
            realized_atr: The actual ATR that materialized after the prediction
        """
        self._realized_vol_history.append(realized_atr)

        # Pair with the most recent disagreement for calibration
        if self._disagreement_history:
            latest_disagree = self._disagreement_history[-1]
            self._calibration_pairs.append(
                (latest_disagree, realized_atr)
            )

        # Trigger recalibration periodically
        now = time.time()
        if now - self._last_calibration_time > self._calibration_interval:
            self._recalibrate()
            self._last_calibration_time = now

    def _record_prediction(self, disagreement: float, current_atr: float):
        """Store prediction for later calibration comparison."""
        self._disagreement_history.append(disagreement)

    def _update_running_stats(self):
        """Update running mean/std of disagreement for normalization."""
        if len(self._disagreement_history) < 10:
            return
        recent = list(self._disagreement_history)[-50:]
        self._disagreement_mean = float(np.mean(recent))
        self._disagreement_std = float(np.std(recent)) + 1e-8

    def _recalibrate(self):
        """
        Recalibrate the prediction model using collected disagreement/vol pairs.

        Computes the linear correlation between disagreement level and
        subsequent realized volatility. Updates calibration coefficient.
        """
        if len(self._calibration_pairs) < 20:
            return

        pairs = list(self._calibration_pairs)[-100:]
        disagree_vals = np.array([p[0] for p in pairs])
        vol_vals = np.array([p[1] for p in pairs])

        if np.std(disagree_vals) < 1e-6 or np.std(vol_vals) < 1e-6:
            return

        # Simple linear regression: vol = a * disagreement + b
        correlation = np.corrcoef(disagree_vals, vol_vals)[0, 1]

        if not np.isnan(correlation):
            # Calibration coefficient: how predictive disagreement is
            # High positive correlation = disagreement predicts vol well
            self._calibration_coeff = float(np.clip(
                0.5 + 0.5 * correlation, 0.3, 2.0
            ))
            logger.info("[DisagreementSignal] Recalibrated: correlation=%.3f, "
                        "coeff=%.3f (from %d pairs)",
                        correlation, self._calibration_coeff, len(pairs))

    def get_stats(self) -> Dict:
        """Get current disagreement signal statistics."""
        return {
            "history_size": len(self._disagreement_history),
            "mean_disagreement": self._disagreement_mean,
            "std_disagreement": self._disagreement_std,
            "calibration_coeff": self._calibration_coeff,
            "calibration_pairs": len(self._calibration_pairs),
        }
