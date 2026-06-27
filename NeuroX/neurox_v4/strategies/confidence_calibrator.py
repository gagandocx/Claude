"""
=============================================================
  NeuroX v7.4 - Confidence Calibration (Platt Scaling)

  Live calibration of model confidence using Platt scaling.
  Maintains a sliding window of (raw_logit, actual_outcome) pairs
  and fits sigmoid parameters A, B via logistic regression so that:
      P_calibrated = 1 / (1 + exp(-(A * logit + B)))

  Falls back to pass-through when insufficient samples are available.
  State is persisted to disk for continuity across restarts.
=============================================================
"""

import os
import sys
import json
import logging
import numpy as np
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PlattCalibrationConfig

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Platt scaling confidence calibrator.

    Maintains a sliding window of recent (raw_logit, actual_outcome) pairs
    and fits sigmoid calibration parameters A and B using scipy.optimize.minimize.
    """

    def __init__(self, config: Optional[PlattCalibrationConfig] = None):
        self.config = config or PlattCalibrationConfig()
        self._window: deque = deque(maxlen=self.config.window_size)
        self._A: float = 1.0  # Scale parameter (identity by default)
        self._B: float = 0.0  # Bias parameter (zero by default)
        self._fitted: bool = False

        # Load persisted state if available
        self.load_state()

    @property
    def has_sufficient_samples(self) -> bool:
        """Check if we have enough data to perform calibration."""
        return len(self._window) >= self.config.min_samples

    def record_outcome(self, raw_prob: float, actual_outcome: int) -> None:
        """
        Record a new (raw_probability, actual_outcome) pair.

        Args:
            raw_prob: The raw model probability/confidence (0 to 1)
            actual_outcome: 1 if the prediction was correct (win), 0 otherwise
        """
        # Convert probability to logit space for Platt scaling
        # Clip to avoid log(0) or log(inf)
        prob_clipped = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
        logit = np.log(prob_clipped / (1.0 - prob_clipped))

        self._window.append((logit, actual_outcome))

        # Refit if we have sufficient samples
        if self.has_sufficient_samples:
            self._fit()

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling calibration to raw probabilities.

        Args:
            raw_probs: Array of raw model probabilities

        Returns:
            Calibrated probabilities. Pass-through if insufficient data.
        """
        if not self._fitted or not self.has_sufficient_samples:
            return raw_probs

        # Convert to logit space
        probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))

        # Apply Platt scaling: P_calibrated = sigmoid(A * logit + B)
        calibrated = 1.0 / (1.0 + np.exp(-(self._A * logits + self._B)))

        return calibrated

    def _fit(self) -> None:
        """Fit Platt scaling parameters A and B using logistic regression."""
        try:
            from scipy.optimize import minimize

            data = list(self._window)
            logits = np.array([d[0] for d in data])
            outcomes = np.array([d[1] for d in data])

            # Negative log-likelihood for logistic regression
            def neg_log_likelihood(params):
                a, b = params
                z = a * logits + b
                # Numerically stable sigmoid
                z_clipped = np.clip(z, -500, 500)
                probs = 1.0 / (1.0 + np.exp(-z_clipped))
                probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
                # Binary cross-entropy
                nll = -np.mean(
                    outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
                )
                return nll

            # Optimize
            result = minimize(
                neg_log_likelihood,
                x0=[self._A, self._B],
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )

            if result.success or result.fun < 10.0:
                self._A = float(result.x[0])
                self._B = float(result.x[1])
                self._fitted = True
                logger.debug(
                    f"[PlattCal] Fitted: A={self._A:.4f}, B={self._B:.4f} "
                    f"(NLL={result.fun:.4f}, samples={len(self._window)})"
                )
            else:
                logger.debug(f"[PlattCal] Optimization did not converge: {result.message}")

        except ImportError:
            logger.warning("[PlattCal] scipy not available - calibration disabled")
            self._fitted = False
        except Exception as e:
            logger.warning(f"[PlattCal] Fit error: {e}")

    def save_state(self) -> None:
        """Persist calibration state to JSON file."""
        try:
            state = {
                "A": self._A,
                "B": self._B,
                "fitted": self._fitted,
                "window": list(self._window),
                "window_size": self.config.window_size,
            }
            state_path = self.config.state_file
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"[PlattCal] State saved to {state_path}")
        except Exception as e:
            logger.warning(f"[PlattCal] Could not save state: {e}")

    def load_state(self) -> None:
        """Load calibration state from JSON file."""
        try:
            state_path = self.config.state_file
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self._A = state.get("A", 1.0)
                self._B = state.get("B", 0.0)
                self._fitted = state.get("fitted", False)
                window_data = state.get("window", [])
                self._window = deque(
                    [tuple(item) for item in window_data],
                    maxlen=self.config.window_size
                )
                logger.info(
                    f"[PlattCal] Loaded state: A={self._A:.4f}, B={self._B:.4f}, "
                    f"samples={len(self._window)}, fitted={self._fitted}"
                )
        except Exception as e:
            logger.debug(f"[PlattCal] Could not load state: {e}")

    def get_stats(self) -> dict:
        """Return current calibration statistics."""
        return {
            "A": self._A,
            "B": self._B,
            "fitted": self._fitted,
            "samples": len(self._window),
            "min_samples": self.config.min_samples,
            "sufficient": self.has_sufficient_samples,
        }
