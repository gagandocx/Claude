"""
=============================================================
  Python ML Bridge v3 - Regime Model Router (Tier 1)

  Routes predictions to regime-specific model subsets:
    - Trending: boosts momentum models (Transformer/LSTM/TCN)
    - Ranging:  boosts mean-reversion (DLinear/NHiTS/GradBoost)
    - Volatile: boosts adaptive models (Mamba/TimesNet/xLSTM)

  Maintains per-regime performance tracking for dynamic weight
  adjustment. Integrates with ensemble.set_regime_weights().
=============================================================
"""

import os
import sys
import logging
from typing import Dict, Optional, List
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RegimeRoutingConfig

logger = logging.getLogger(__name__)

# Full model names in ensemble order (17 models)
_MODEL_NAMES = [
    "transformer", "lstm", "tcn",
    "patch_tst", "tft", "nhits",
    "itransformer", "mamba", "dlinear",
    "xlstm", "timesnet",
    "chronos", "timemixer", "softs",
    "gradient_boost", "xgboost", "catboost",
]
_NUM_MODELS = 17


class RegimeModelRouter:
    """
    Routes predictions through regime-specific model weight configurations.

    Instead of using uniform ensemble weights for all market conditions,
    this router maintains 3 optimized weight sets:

    1. Trending regime:
       - Boosts momentum-following models that excel in trends
       - Transformer, LSTM, TCN, PatchTST, TFT, Mamba
       - Reduces weight on mean-reversion models

    2. Ranging regime:
       - Boosts mean-reversion models that excel in ranges
       - DLinear, NHiTS, GradBoost, SOFTS, xLSTM
       - Reduces weight on momentum models

    3. Volatile regime:
       - Boosts adaptive/robust models for volatile conditions
       - Mamba, iTransformer, Transformer, Chronos, TimeMixer
       - Increases diversity (more even weights)

    Performance tracking per model per regime enables dynamic
    weight adjustment over time (learns which models work in which
    regime from actual trade outcomes).
    """

    def __init__(self, config: Optional[RegimeRoutingConfig] = None):
        self.config = config or RegimeRoutingConfig()

        # Build model name -> index mapping
        self._model_idx = {name: i for i, name in enumerate(_MODEL_NAMES)}

        # Initialize regime-specific weight arrays
        self._weights = {
            'trending': self._build_trending_weights(),
            'ranging': self._build_ranging_weights(),
            'volatile': self._build_volatile_weights(),
        }

        # Per-model per-regime accuracy tracking for dynamic adjustment
        self._accuracy: Dict[str, Dict[str, deque]] = {
            regime: {model: deque(maxlen=100) for model in _MODEL_NAMES}
            for regime in ['trending', 'ranging', 'volatile']
        }

        # Count of predictions routed per regime
        self._route_count = {'trending': 0, 'ranging': 0, 'volatile': 0}

        logger.info("[RegimeRouter] Initialized with 3 regime configurations")
        logger.info("[RegimeRouter]   Trending models: %s", self.config.trending_models)
        logger.info("[RegimeRouter]   Ranging models: %s", self.config.ranging_models)
        logger.info("[RegimeRouter]   Volatile models: %s", self.config.volatile_models)

    def _build_trending_weights(self) -> np.ndarray:
        """
        Build weight array for trending regime.

        Boosts momentum-following models, reduces mean-reversion models.
        """
        weights = np.ones(_NUM_MODELS) * 0.03  # Base weight for all
        boost = 0.12  # Extra weight for favored models

        for model_name in self.config.trending_models:
            if model_name in self._model_idx:
                weights[self._model_idx[model_name]] += boost

        # Also give moderate boost to related models
        # Transformer family excels at capturing trends
        trending_related = ["patch_tst", "tft", "itransformer"]
        for model_name in trending_related:
            if model_name in self._model_idx and model_name not in self.config.trending_models:
                weights[self._model_idx[model_name]] += boost * 0.5

        # Normalize to sum to 1
        weights = weights / weights.sum()
        return weights

    def _build_ranging_weights(self) -> np.ndarray:
        """
        Build weight array for ranging regime.

        Boosts mean-reversion models, reduces momentum models.
        """
        weights = np.ones(_NUM_MODELS) * 0.03  # Base weight for all
        boost = 0.12

        for model_name in self.config.ranging_models:
            if model_name in self._model_idx:
                weights[self._model_idx[model_name]] += boost

        # Tree models are good at ranging (capture local patterns)
        ranging_related = ["gradient_boost", "xgboost", "catboost"]
        for model_name in ranging_related:
            if model_name in self._model_idx and model_name not in self.config.ranging_models:
                weights[self._model_idx[model_name]] += boost * 0.5

        weights = weights / weights.sum()
        return weights

    def _build_volatile_weights(self) -> np.ndarray:
        """
        Build weight array for volatile regime.

        More diversified (less concentrated) to handle uncertainty.
        Slightly boosts adaptive models that handle regime changes.
        """
        weights = np.ones(_NUM_MODELS) * 0.04  # Higher base for diversification
        boost = 0.08  # Smaller boost since we want more diversity

        for model_name in self.config.volatile_models:
            if model_name in self._model_idx:
                weights[self._model_idx[model_name]] += boost

        weights = weights / weights.sum()
        return weights

    def get_regime_weights(self, regime: str) -> np.ndarray:
        """
        Get optimized model weights for the given regime.

        Args:
            regime: One of 'trending', 'ranging', 'volatile'

        Returns:
            np.ndarray of shape (17,) with weights summing to 1.0
        """
        # Map regime names from RegimeDetector to our categories
        regime_lower = regime.lower()
        if 'trend' in regime_lower or 'strong' in regime_lower:
            key = 'trending'
        elif 'rang' in regime_lower or 'sideways' in regime_lower or 'consolid' in regime_lower:
            key = 'ranging'
        elif 'volat' in regime_lower or 'high_vol' in regime_lower or 'crisis' in regime_lower:
            key = 'volatile'
        else:
            # Default: use slightly trending-biased weights
            key = 'trending'

        return self._weights[key].copy()

    def route_prediction(self, features: np.ndarray, regime: str,
                         ensemble) -> Dict[str, np.ndarray]:
        """
        Route prediction through the ensemble with regime-specific weights.

        Sets regime weights on the ensemble, runs prediction, then
        restores default weights.

        Args:
            features: Input features array for ensemble prediction
            regime: Current market regime name
            ensemble: EnsembleManager instance

        Returns:
            Prediction dict from ensemble.predict() with regime-weighted output
        """
        regime_lower = regime.lower()
        if 'trend' in regime_lower or 'strong' in regime_lower:
            key = 'trending'
        elif 'rang' in regime_lower or 'sideways' in regime_lower or 'consolid' in regime_lower:
            key = 'ranging'
        elif 'volat' in regime_lower or 'high_vol' in regime_lower or 'crisis' in regime_lower:
            key = 'volatile'
        else:
            key = 'trending'

        # Apply regime weights
        weights = self._weights[key]
        ensemble.set_regime_weights(weights)

        # Run prediction with regime weights
        prediction = ensemble.predict(features)

        # Track routing
        self._route_count[key] += 1

        logger.debug("[RegimeRouter] Routed to '%s' (count: %d)",
                     key, self._route_count[key])

        return prediction

    def update_regime_performance(self, regime: str, model_name: str,
                                  correct: bool) -> None:
        """
        Track per-model accuracy within each regime for dynamic adjustment.

        Called after trade outcome is known. Over time, this shifts
        weights toward models that actually perform well in each regime.

        Args:
            regime: The regime during this prediction
            model_name: Which model to update
            correct: Whether the model's prediction was correct
        """
        regime_lower = regime.lower()
        if 'trend' in regime_lower or 'strong' in regime_lower:
            key = 'trending'
        elif 'rang' in regime_lower or 'sideways' in regime_lower or 'consolid' in regime_lower:
            key = 'ranging'
        elif 'volat' in regime_lower or 'high_vol' in regime_lower or 'crisis' in regime_lower:
            key = 'volatile'
        else:
            key = 'trending'

        if model_name in self._accuracy[key]:
            self._accuracy[key][model_name].append(1.0 if correct else 0.0)

        # Periodically update weights based on accumulated accuracy data
        total_tracked = sum(len(d) for d in self._accuracy[key].values())
        if total_tracked > 50 and total_tracked % 20 == 0:
            self._adjust_weights(key)

    def _adjust_weights(self, regime_key: str) -> None:
        """
        Adjust weights for a regime based on tracked accuracy.

        Blends the initial regime bias (60%) with actual performance (40%)
        to prevent wild swings while still adapting.
        """
        accuracy_scores = np.zeros(_NUM_MODELS)
        for i, model_name in enumerate(_MODEL_NAMES):
            history = self._accuracy[regime_key][model_name]
            if len(history) >= 5:
                accuracy_scores[i] = np.mean(list(history))
            else:
                accuracy_scores[i] = 1.0 / _NUM_MODELS  # Prior (uniform)

        # Normalize accuracy scores to form a valid weight distribution
        if accuracy_scores.sum() > 0:
            perf_weights = accuracy_scores / accuracy_scores.sum()
        else:
            perf_weights = np.ones(_NUM_MODELS) / _NUM_MODELS

        # Blend: 60% original regime bias + 40% performance-based
        base_weights = self._weights[regime_key]
        self._weights[regime_key] = 0.6 * base_weights + 0.4 * perf_weights

        # Re-normalize
        self._weights[regime_key] /= self._weights[regime_key].sum()

        logger.info("[RegimeRouter] Adjusted '%s' weights from performance data "
                    "(top 3: %s)",
                    regime_key,
                    sorted(zip(_MODEL_NAMES, self._weights[regime_key]),
                           key=lambda x: x[1], reverse=True)[:3])

    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        return {
            'route_counts': self._route_count.copy(),
            'total_routed': sum(self._route_count.values()),
            'regime_weights': {k: v.tolist() for k, v in self._weights.items()},
        }
