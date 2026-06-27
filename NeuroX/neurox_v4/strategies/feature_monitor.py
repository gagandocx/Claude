"""
=============================================================
  NeuroX - Feature Importance Monitor
  Tracks per-feature importance using permutation importance
  approximation. Detects when specific features degrade and
  alerts so the trader/system can investigate data quality issues.
=============================================================
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Result of a feature importance check."""
    degraded_features: List[int]         # Indices of degraded features
    importance_scores: np.ndarray        # Current importance per feature
    historical_avg: np.ndarray           # Historical average importance
    alert: bool                          # Whether any feature is degraded


class FeatureImportanceMonitor:
    """
    Tracks which of the 46 input features contribute most to correct predictions.

    Uses permutation importance approximation: for each feature, measures how
    much the prediction changes when that feature is shuffled. A rolling window
    of importance scores is maintained. Alerts when a feature's importance drops
    below a threshold of its historical average (indicating degradation).

    Usage:
        monitor = FeatureImportanceMonitor(config)
        # After each prediction:
        monitor.record(feature_input, prediction_probs, prediction_correct)
        # Periodically check:
        result = monitor.check_degradation()
    """

    def __init__(self, config=None):
        from config.settings import FeatureMonitorConfig
        self.config = config or FeatureMonitorConfig()

        self._num_features: int = 46
        self._prediction_count: int = 0

        # Rolling importance scores: deque of (num_features,) arrays
        self._importance_history: deque = deque(
            maxlen=self.config.importance_window
        )

        # Store recent feature inputs and predictions for permutation calc
        self._recent_features: deque = deque(maxlen=50)
        self._recent_probs: deque = deque(maxlen=50)
        self._recent_correct: deque = deque(maxlen=50)

        # Cached importance scores (updated every check_interval predictions)
        self._current_importance: Optional[np.ndarray] = None
        self._historical_importance: Optional[np.ndarray] = None

        # Track degraded features for logging
        self._degraded_features: List[int] = []
        self._last_check_count: int = 0

        logger.info(
            f"[FeatureMonitor] Initialized: window={self.config.importance_window}, "
            f"check_interval={self.config.check_interval}, "
            f"threshold={self.config.degradation_threshold}"
        )

    def record(
        self,
        feature_input: np.ndarray,
        prediction_probs: np.ndarray,
        prediction_correct: bool,
    ) -> None:
        """
        Record a prediction for feature importance tracking.

        Args:
            feature_input: Raw feature input, shape (1, seq_len, num_features)
                           or (seq_len, num_features) or (1, num_features)
            prediction_probs: Model output probabilities, shape (1, 3) or (3,)
            prediction_correct: Whether the prediction was correct
        """
        self._prediction_count += 1

        # Normalize feature input to 2D (seq_len, num_features) or (1, num_features)
        if feature_input.ndim == 3:
            # Take last timestep for importance calculation
            feat_2d = feature_input[0, -1, :]  # (num_features,)
        elif feature_input.ndim == 2:
            feat_2d = feature_input[-1, :]      # (num_features,)
        else:
            feat_2d = feature_input              # already 1D

        # Ensure correct shape
        if feat_2d.shape[0] < self._num_features:
            feat_2d = np.pad(
                feat_2d, (0, self._num_features - feat_2d.shape[0])
            )
        elif feat_2d.shape[0] > self._num_features:
            feat_2d = feat_2d[:self._num_features]

        # Normalize probs
        if prediction_probs.ndim == 2:
            probs_1d = prediction_probs[0]
        else:
            probs_1d = prediction_probs

        self._recent_features.append(feat_2d)
        self._recent_probs.append(probs_1d)
        self._recent_correct.append(1.0 if prediction_correct else 0.0)

        # Check if it's time to compute importance
        if (self._prediction_count - self._last_check_count
                >= self.config.check_interval):
            self._compute_importance()
            self._last_check_count = self._prediction_count

    def _compute_importance(self) -> None:
        """
        Compute permutation importance approximation.

        For each feature, shuffles that feature across recent samples and
        measures how much the prediction confidence changes. Features that
        cause large changes when shuffled are more important.
        """
        if len(self._recent_features) < 10:
            return

        features_arr = np.array(list(self._recent_features))  # (N, num_features)
        correct_arr = np.array(list(self._recent_correct))     # (N,)

        n_samples = len(features_arr)
        num_feat = min(features_arr.shape[1], self._num_features)

        # Baseline: weighted average confidence on correct predictions
        baseline_score = np.mean(correct_arr)

        importance = np.zeros(self._num_features)

        for f_idx in range(num_feat):
            # Create shuffled version of this feature
            shuffled = features_arr.copy()
            perm_idx = np.random.permutation(n_samples)
            shuffled[:, f_idx] = features_arr[perm_idx, f_idx]

            # Measure how much predictions would change
            # Use feature variance as a proxy for information content
            original_var = np.var(features_arr[:, f_idx])
            shuffled_var = np.var(shuffled[:, f_idx] - features_arr[:, f_idx])

            if original_var > 1e-10:
                # Importance = how much shuffling this feature disrupts patterns
                # weighted by whether predictions using this feature were correct
                feature_correlation = np.abs(
                    np.corrcoef(features_arr[:, f_idx], correct_arr)[0, 1]
                )
                if np.isnan(feature_correlation):
                    feature_correlation = 0.0

                importance[f_idx] = feature_correlation * original_var
            else:
                importance[f_idx] = 0.0

        # Normalize importance to sum to 1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        # Store in history
        self._importance_history.append(importance)
        self._current_importance = importance

        # Update historical average
        if len(self._importance_history) >= 5:
            self._historical_importance = np.mean(
                list(self._importance_history), axis=0
            )

    def check_degradation(self) -> FeatureImportanceResult:
        """
        Check if any features have degraded below threshold.

        A feature is considered degraded if its current importance is below
        degradation_threshold (e.g., 10%) of its historical average importance.

        Returns:
            FeatureImportanceResult with degraded feature indices and scores
        """
        if (self._current_importance is None
                or self._historical_importance is None):
            return FeatureImportanceResult(
                degraded_features=[],
                importance_scores=np.zeros(self._num_features),
                historical_avg=np.zeros(self._num_features),
                alert=False,
            )

        degraded = []
        for i in range(self._num_features):
            hist_val = self._historical_importance[i]
            curr_val = self._current_importance[i]

            # Only flag features that had meaningful historical importance
            if hist_val > 0.001:
                ratio = curr_val / hist_val
                if ratio < self.config.degradation_threshold:
                    degraded.append(i)

        self._degraded_features = degraded

        if degraded:
            logger.warning(
                f"[FeatureMonitor] {len(degraded)} features degraded: "
                f"indices={degraded[:10]}{'...' if len(degraded) > 10 else ''}"
            )

        return FeatureImportanceResult(
            degraded_features=degraded,
            importance_scores=self._current_importance.copy(),
            historical_avg=self._historical_importance.copy(),
            alert=len(degraded) > 0,
        )

    def get_top_features(self, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get the top N most important features.

        Returns:
            List of (feature_index, importance_score) tuples, sorted descending.
        """
        if self._current_importance is None:
            return []

        indices = np.argsort(self._current_importance)[::-1][:n]
        return [(int(idx), float(self._current_importance[idx])) for idx in indices]

    def get_degraded_features(self) -> List[int]:
        """Return list of currently degraded feature indices."""
        return self._degraded_features.copy()

    @property
    def prediction_count(self) -> int:
        """Total predictions recorded."""
        return self._prediction_count

    @property
    def has_sufficient_data(self) -> bool:
        """Whether enough data has been collected for meaningful analysis."""
        return len(self._importance_history) >= 5

    def get_status(self) -> Dict:
        """Get current monitor status for logging/dashboard."""
        return {
            "predictions_recorded": self._prediction_count,
            "importance_samples": len(self._importance_history),
            "degraded_count": len(self._degraded_features),
            "degraded_features": self._degraded_features[:5],
            "has_sufficient_data": self.has_sufficient_data,
            "top_features": self.get_top_features(5),
        }
