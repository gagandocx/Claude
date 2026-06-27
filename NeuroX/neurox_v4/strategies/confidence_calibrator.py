"""
=============================================================
  NeuroX v7.4 - Confidence Calibration (Platt Scaling)

  Live calibration of model confidence using Platt scaling.
  Maintains a sliding window of (raw_logit, actual_outcome) pairs
  and fits sigmoid parameters A, B via logistic regression so that:
      P_calibrated = 1 / (1 + exp(-(A * logit + B)))

  v7.5: Regime-conditional calibration. Maintains separate (A, B)
  parameters per market regime. Falls back to global calibration
  if a regime has fewer than min_samples observations.

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
from typing import Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PlattCalibrationConfig

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Platt scaling confidence calibrator with regime-conditional support.

    Maintains a sliding window of recent (raw_logit, actual_outcome) pairs
    and fits sigmoid calibration parameters A and B using scipy.optimize.minimize.

    v7.5: Per-regime calibration parameters allow different regimes (trending,
    ranging, volatile) to have their own calibration curves, since model
    confidence behaves differently in each regime.
    """

    def __init__(self, config: Optional[PlattCalibrationConfig] = None):
        self.config = config or PlattCalibrationConfig()

        # Global calibration (fallback)
        self._window: deque = deque(maxlen=self.config.window_size)
        self._A: float = 1.0  # Scale parameter (identity by default)
        self._B: float = 0.0  # Bias parameter (zero by default)
        self._fitted: bool = False

        # Per-regime calibration parameters
        # Format: {regime_name: {"A": float, "B": float, "fitted": bool, "window": deque}}
        self._regime_params: Dict[str, Dict] = {}
        self._regime_min_samples: int = 20  # Min samples per regime before using regime-specific params

        # Load persisted state if available
        self.load_state()

    @property
    def has_sufficient_samples(self) -> bool:
        """Check if we have enough data to perform global calibration."""
        return len(self._window) >= self.config.min_samples

    def _get_or_create_regime(self, regime: str) -> Dict:
        """Get or create regime-specific calibration state."""
        if regime not in self._regime_params:
            self._regime_params[regime] = {
                "A": 1.0,
                "B": 0.0,
                "fitted": False,
                "window": deque(maxlen=self.config.window_size),
            }
        return self._regime_params[regime]

    def record_outcome(self, raw_prob: float, actual_outcome: int,
                       regime: Optional[str] = None) -> None:
        """
        Record a new (raw_probability, actual_outcome) pair.

        Args:
            raw_prob: The raw model probability/confidence (0 to 1)
            actual_outcome: 1 if the prediction was correct (win), 0 otherwise
            regime: Optional regime name for regime-conditional calibration
        """
        # Convert probability to logit space for Platt scaling
        # Clip to avoid log(0) or log(inf)
        prob_clipped = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
        logit = np.log(prob_clipped / (1.0 - prob_clipped))

        # Always record in global window
        self._window.append((logit, actual_outcome))

        # Also record in regime-specific window if regime provided
        if regime:
            regime_state = self._get_or_create_regime(regime)
            regime_state["window"].append((logit, actual_outcome))
            # Fit regime-specific params if sufficient data
            if len(regime_state["window"]) >= self._regime_min_samples:
                self._fit_regime(regime)

        # Refit global if we have sufficient samples
        if self.has_sufficient_samples:
            self._fit()

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling calibration to raw probabilities (global).

        Args:
            raw_probs: Array of raw model probabilities

        Returns:
            Calibrated probabilities. Pass-through if insufficient data.
        """
        if not self._fitted or not self.has_sufficient_samples:
            return raw_probs

        return self._apply_platt(raw_probs, self._A, self._B)

    def calibrate_for_regime(self, raw_probs: np.ndarray,
                             regime: Optional[str] = None) -> np.ndarray:
        """
        Apply regime-conditional Platt scaling calibration.

        Uses regime-specific parameters if available and fitted.
        Falls back to global calibration if regime has insufficient data.
        Falls back to pass-through if no calibration is available.

        Args:
            raw_probs: Array of raw model probabilities
            regime: Market regime name (e.g. "trending", "ranging", "volatile")

        Returns:
            Calibrated probabilities.
        """
        # Try regime-specific calibration first
        if regime and regime in self._regime_params:
            regime_state = self._regime_params[regime]
            if (regime_state["fitted"] and
                    len(regime_state["window"]) >= self._regime_min_samples):
                logger.debug(
                    f"[PlattCal] Using regime-specific calibration for '{regime}' "
                    f"(A={regime_state['A']:.4f}, B={regime_state['B']:.4f}, "
                    f"samples={len(regime_state['window'])})"
                )
                return self._apply_platt(
                    raw_probs, regime_state["A"], regime_state["B"]
                )

        # Fall back to global calibration
        return self.calibrate(raw_probs)

    @staticmethod
    def _apply_platt(raw_probs: np.ndarray, A: float, B: float) -> np.ndarray:
        """Apply Platt scaling with given A and B parameters."""
        probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(probs_clipped / (1.0 - probs_clipped))
        calibrated = 1.0 / (1.0 + np.exp(-(A * logits + B)))
        return calibrated

    def _fit(self) -> None:
        """Fit global Platt scaling parameters A and B using logistic regression."""
        result_params = self._fit_from_window(self._window)
        if result_params is not None:
            self._A, self._B = result_params
            self._fitted = True

    def _fit_regime(self, regime: str) -> None:
        """Fit regime-specific Platt scaling parameters."""
        regime_state = self._regime_params[regime]
        result_params = self._fit_from_window(regime_state["window"])
        if result_params is not None:
            regime_state["A"], regime_state["B"] = result_params
            regime_state["fitted"] = True
            logger.debug(
                f"[PlattCal] Regime '{regime}' fitted: "
                f"A={regime_state['A']:.4f}, B={regime_state['B']:.4f} "
                f"(samples={len(regime_state['window'])})"
            )

    def _fit_from_window(self, window: deque) -> Optional[Tuple[float, float]]:
        """
        Fit Platt scaling parameters from a data window.

        Args:
            window: Deque of (logit, outcome) tuples.

        Returns:
            Tuple (A, B) if fit succeeded, None otherwise.
        """
        try:
            from scipy.optimize import minimize

            data = list(window)
            logits = np.array([d[0] for d in data])
            outcomes = np.array([d[1] for d in data])

            # Need at least 2 classes
            if len(np.unique(outcomes)) < 2:
                return None

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

            # Optimize starting from current global params
            result = minimize(
                neg_log_likelihood,
                x0=[1.0, 0.0],
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )

            if result.success or result.fun < 0.75:
                return (float(result.x[0]), float(result.x[1]))
            else:
                logger.debug(f"[PlattCal] Optimization did not converge: {result.message}")
                return None

        except ImportError:
            logger.warning("[PlattCal] scipy not available - calibration disabled")
            return None
        except Exception as e:
            logger.warning(f"[PlattCal] Fit error: {e}")
            return None

    def save_state(self) -> None:
        """Persist calibration state (global + per-regime) to JSON file."""
        try:
            # Serialize regime params
            regime_serialized = {}
            for regime_name, rstate in self._regime_params.items():
                regime_serialized[regime_name] = {
                    "A": rstate["A"],
                    "B": rstate["B"],
                    "fitted": rstate["fitted"],
                    "window": list(rstate["window"]),
                }

            state = {
                "A": self._A,
                "B": self._B,
                "fitted": self._fitted,
                "window": list(self._window),
                "window_size": self.config.window_size,
                "regime_params": regime_serialized,
            }
            state_path = self.config.state_file
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"[PlattCal] State saved to {state_path}")
        except Exception as e:
            logger.warning(f"[PlattCal] Could not save state: {e}")

    def load_state(self) -> None:
        """Load calibration state (global + per-regime) from JSON file."""
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

                # Load per-regime params
                regime_data = state.get("regime_params", {})
                for regime_name, rstate in regime_data.items():
                    self._regime_params[regime_name] = {
                        "A": rstate.get("A", 1.0),
                        "B": rstate.get("B", 0.0),
                        "fitted": rstate.get("fitted", False),
                        "window": deque(
                            [tuple(item) for item in rstate.get("window", [])],
                            maxlen=self.config.window_size
                        ),
                    }

                logger.info(
                    f"[PlattCal] Loaded state: A={self._A:.4f}, B={self._B:.4f}, "
                    f"samples={len(self._window)}, fitted={self._fitted}, "
                    f"regimes={list(self._regime_params.keys())}"
                )
        except Exception as e:
            logger.debug(f"[PlattCal] Could not load state: {e}")

    def get_stats(self) -> dict:
        """Return current calibration statistics."""
        regime_stats = {}
        for regime_name, rstate in self._regime_params.items():
            regime_stats[regime_name] = {
                "A": rstate["A"],
                "B": rstate["B"],
                "fitted": rstate["fitted"],
                "samples": len(rstate["window"]),
            }

        return {
            "A": self._A,
            "B": self._B,
            "fitted": self._fitted,
            "samples": len(self._window),
            "min_samples": self.config.min_samples,
            "sufficient": self.has_sufficient_samples,
            "regime_calibrations": regime_stats,
        }
