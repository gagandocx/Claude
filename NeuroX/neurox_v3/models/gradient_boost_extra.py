"""
=============================================================
  Python ML Bridge - Advanced Gradient Boosting Model
  LightGBM → XGBoost → HistGradientBoosting (auto-fallback)

  Why this complements HistGradientBoosting:
    - LightGBM uses leaf-wise tree growth + histogram binning
      → faster training, often higher accuracy on financial data
    - XGBoost adds L1/L2 regularisation + column subsampling
      → reduces overfitting on noisy M1 price features
    - Both handle class imbalance natively
    - Falls back to sklearn HistGradientBoosting if neither installed

  Install (recommended):
    pip install lightgbm         # preferred
    pip install xgboost          # fallback
=============================================================
"""

import os
import sys
import logging
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── Optional dependency detection ────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
#  GRAD BOOST EXTRA
# ─────────────────────────────────────────────
class GradBoostExtra:
    """
    Advanced gradient boosting wrapper.

    Priority: LightGBM > XGBoost > HistGradientBoosting (sklearn).

    Exposes the same interface as EnsembleManager's gradient_boost so
    it can be dropped in alongside the existing model without refactoring.

    Input:
        X: np.ndarray of shape (n_samples, seq_len, features)
           Flattened to 2-D internally — no sequence awareness needed;
           tree models capture feature interactions across the flattened window.
        y: np.ndarray of shape (n_samples,)  — integer class labels (0/1/2)

    Output:
        predict_proba(X) → np.ndarray of shape (n_samples, 3)
    """

    def __init__(self, config=None):
        from config.settings import XGBoostConfig
        self.config = config or XGBoostConfig()
        self.fitted = False
        self._backend: str = "unknown"
        self._model = None
        self._init_model()

    # ── model initialisation ─────────────────────────────────────────────────

    def _init_model(self):
        """Select and configure the best available backend."""
        if self.config.use_lightgbm and _LGBM_AVAILABLE:
            self._backend = "lightgbm"
            self._model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_samples=self.config.min_child_weight,
                num_leaves=self.config.num_leaves,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            logger.info("GradBoostExtra: backend=LightGBM")

        elif _XGB_AVAILABLE:
            self._backend = "xgboost"
            self._model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_weight=self.config.min_child_weight,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
            )
            logger.info("GradBoostExtra: backend=XGBoost")

        else:
            self._backend = "histgb"
            self._model = HistGradientBoostingClassifier(
                max_iter=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_child_weight,
                l2_regularization=self.config.reg_lambda,
                random_state=self.config.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
            )
            logger.warning(
                "GradBoostExtra: LightGBM and XGBoost not installed. "
                "Falling back to HistGradientBoosting. "
                "Install lightgbm for best performance: pip install lightgbm"
            )

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradBoostExtra":
        """
        Train on sequence data (flattened internally).

        Args:
            X: (n_samples, seq_len, features)
            y: (n_samples,) integer labels 0/1/2
        Returns:
            self
        """
        X_flat = X.reshape(X.shape[0], -1)

        if self._backend == "lightgbm":
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_flat, y, test_size=0.1, random_state=42, stratify=y
            )
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),     # silent
            ]
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
            best_iter = self._model.best_iteration_
            logger.info(f"  LightGBM best iteration: {best_iter}")

        elif self._backend == "xgboost":
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_flat, y, test_size=0.1, random_state=42, stratify=y
            )
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )

        else:  # histgb — sklearn handles early stopping internally
            self._model.fit(X_flat, y)

        self.fitted = True
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities for all samples.

        Args:
            X: (batch, seq_len, features)
        Returns:
            probs: (batch, 3)  — columns: SELL / HOLD / BUY
        """
        if not self.fitted:
            batch_size = X.shape[0]
            logger.debug("GradBoostExtra not fitted yet — returning uniform probs")
            return np.full((batch_size, 3), 1.0 / 3.0)

        X_flat = X.reshape(X.shape[0], -1)
        probs = self._model.predict_proba(X_flat)

        # Guard: ensure exactly 3 columns (in case a class is missing in small data)
        if probs.shape[1] < 3:
            pad = np.zeros((probs.shape[0], 3 - probs.shape[1]))
            probs = np.hstack([probs, pad])
        elif probs.shape[1] > 3:
            probs = probs[:, :3]

        return probs

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise model to disk (joblib)."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        joblib.dump(
            {
                "backend": self._backend,
                "model": self._model,
                "fitted": self.fitted,
            },
            path,
        )
        logger.info(f"GradBoostExtra saved → {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        if not os.path.exists(path):
            logger.warning(f"GradBoostExtra checkpoint not found: {path}")
            return
        data = joblib.load(path)
        self._backend = data["backend"]
        self._model = data["model"]
        self.fitted = data["fitted"]
        logger.info(f"GradBoostExtra loaded from {path}  (backend={self._backend})")

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Which backend is active: 'lightgbm' | 'xgboost' | 'histgb'."""
        return self._backend

    def feature_importance(self, top_n: int = 20) -> Optional[np.ndarray]:
        """
        Return feature importances if the backend supports them.
        Useful for debugging which features drive decisions.
        """
        if not self.fitted:
            return None
        try:
            if self._backend == "lightgbm":
                return self._model.feature_importances_[:top_n]
            elif self._backend == "xgboost":
                return self._model.feature_importances_[:top_n]
            else:
                return None
        except AttributeError:
            return None
