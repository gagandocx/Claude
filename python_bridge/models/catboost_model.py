"""
=============================================================
  Python ML Bridge - CatBoost Model Wrapper
  Yandex CatBoost (Prokhorenkova et al., 2018)
  https://arxiv.org/abs/1706.09516

  Why CatBoost completes the gradient boosting trio:

    LightGBM   — Leaf-wise tree growth, histogram binning, fastest training
    XGBoost    — Column/row subsampling, L1+L2 regularisation, industry standard
    CatBoost   — Ordered boosting (prevents target leakage), symmetric trees,
                 most resistant to overfitting on SMALL noisy datasets

  CatBoost's "ordered boosting" is specifically designed to prevent
  overfitting on time series by computing gradients using only the
  historical subset of data that would have been available at each
  training step — exactly how live trading works.

  Install:
    pip install catboost

  Fallback:
    If CatBoost is not installed, automatically falls back to
    HistGradientBoostingClassifier (already a dependency).
=============================================================
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── Optional CatBoost detection ──────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


class CatBoostModel:
    """
    CatBoost wrapper with HistGradientBoosting fallback.

    Exposes the same interface as GradBoostExtra so it plugs into
    EnsembleManager without any changes to the calling code.

    Input:
        X: np.ndarray (n_samples, seq_len, features) — flattened to 2D internally
        y: np.ndarray (n_samples,)                   — integer labels 0/1/2

    Output:
        predict_proba(X) → np.ndarray (n_samples, 3)
    """

    def __init__(self, config=None):
        from config.settings import CatBoostConfig
        self.config = config or CatBoostConfig()
        self.fitted = False
        self._backend: str = "unknown"
        self._model = None
        self._init_model()

    # ── model init ────────────────────────────────────────────────────────────

    def _init_model(self):
        if _CATBOOST_AVAILABLE:
            self._backend = "catboost"
            self._model = CatBoostClassifier(
                iterations=self.config.iterations,
                depth=self.config.depth,
                learning_rate=self.config.learning_rate,
                l2_leaf_reg=self.config.l2_leaf_reg,
                random_strength=self.config.random_strength,
                bagging_temperature=self.config.bagging_temperature,
                border_count=self.config.border_count,
                random_seed=self.config.random_seed,
                loss_function="MultiClass",
                eval_metric="Accuracy",
                od_type="Iter",                          # Early stopping type
                od_wait=self.config.early_stopping_rounds,
                verbose=False,
                allow_writing_files=False,               # No temp files
                thread_count=-1,                         # All CPU cores
            )
            logger.info("CatBoostModel: backend=CatBoost (ordered boosting)")
        else:
            self._backend = "histgb_fallback"
            self._model = HistGradientBoostingClassifier(
                max_iter=self.config.iterations,
                max_depth=self.config.depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=20,
                l2_regularization=self.config.l2_leaf_reg,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=self.config.early_stopping_rounds,
            )
            logger.warning(
                "CatBoostModel: CatBoost not installed — using HistGradientBoosting fallback. "
                "Install for best performance: pip install catboost"
            )

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostModel":
        """
        Train on sequence data (flattened to 2D internally).

        Args:
            X: (n_samples, seq_len, features)
            y: (n_samples,) integer labels 0/1/2
        """
        X_flat = X.reshape(X.shape[0], -1)

        if self._backend == "catboost":
            # CatBoost has built-in eval set support
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_flat, y, test_size=0.1, random_state=42, stratify=y
            )
            self._model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False,
            )
            best_iter = self._model.get_best_iteration()
            logger.info(f"  CatBoost best iteration: {best_iter}")
        else:
            self._model.fit(X_flat, y)

        self.fitted = True
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities.

        Args:
            X: (batch, seq_len, features)
        Returns:
            probs: (batch, 3)
        """
        if not self.fitted:
            logger.debug("CatBoostModel not fitted yet — returning uniform probs")
            return np.full((X.shape[0], 3), 1.0 / 3.0)

        X_flat = X.reshape(X.shape[0], -1)
        probs = self._model.predict_proba(X_flat)

        # Ensure exactly 3 columns
        if probs.shape[1] < 3:
            pad = np.zeros((probs.shape[0], 3 - probs.shape[1]))
            probs = np.hstack([probs, pad])
        elif probs.shape[1] > 3:
            probs = probs[:, :3]

        return probs

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(
            {"backend": self._backend, "model": self._model, "fitted": self.fitted},
            path,
        )
        logger.info(f"CatBoostModel saved → {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        if not os.path.exists(path):
            logger.warning(f"CatBoostModel checkpoint not found: {path}")
            return
        data = joblib.load(path)
        self._backend = data["backend"]
        self._model = data["model"]
        self.fitted = data["fitted"]
        logger.info(f"CatBoostModel loaded from {path}  (backend={self._backend})")

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active backend: 'catboost' | 'histgb_fallback'"""
        return self._backend

    def feature_importance(self, top_n: int = 20) -> np.ndarray:
        """Return top feature importances (CatBoost only)."""
        if not self.fitted or self._backend != "catboost":
            return np.array([])
        return self._model.get_feature_importance()[:top_n]
