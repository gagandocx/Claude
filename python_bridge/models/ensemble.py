"""
=============================================================
  Python ML Bridge - Ensemble Model Manager  (v2 — 5 models)

  Stack:
    1. MarketTransformer      — global self-attention patterns   (weight 0.28)
    2. MarketLSTM             — sequential / gating patterns     (weight 0.24)
    3. MarketTCN              — multi-scale local patterns       (weight 0.22)
    4. HistGradientBoosting   — tabular feature interactions     (weight 0.14)
    5. GradBoostExtra         — LightGBM / XGBoost               (weight 0.12)

  Meta-learner: HistGradientBoostingClassifier on 15-dim stacked predictions
                (upgraded from LogisticRegression for non-linear combinations)

  Confidence formula (improved):
    confidence = max_prob  ×  (1 − entropy_norm)  ×  agreement_norm

    • max_prob        — how strongly the ensemble leans toward one class
    • 1 − entropy_norm — how "peaked" the full distribution is
                         (penalises near-uniform outputs more than max_prob alone)
    • agreement_norm  — inter-model consensus, normalised for chance level
                         agreement_norm = (raw_agreement − 1/N) / (1 − 1/N)

  Example — strong clean signal (5/5 agree, probs [0.90, 0.05, 0.05]):
    old formula:  0.90 × 1.00 = 0.90   (over-confident for noisy markets)
    new formula:  0.90 × 0.64 × 1.00 = 0.58   (more conservative, better calibrated)

  Example — weak signal (2/5 agree, probs [0.40, 0.35, 0.25]):
    old formula:  0.40 × 0.40 = 0.16   (passes 0.10 threshold)
    new formula:  0.40 × 0.03 × 0.25 = 0.003  (correctly filtered as uncertain)
=============================================================
"""

import os
import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Optional
from collections import deque
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    EnsembleConfig, TransformerConfig, LSTMConfig, TCNConfig, XGBoostConfig
)
from models.transformer_model import MarketTransformer
from models.lstm_model import MarketLSTM
from models.tcn_model import MarketTCN
from models.gradient_boost_extra import GradBoostExtra

logger = logging.getLogger(__name__)

# Number of base models — used in agreement normalisation
_NUM_MODELS = 5


class EnsembleManager:
    """
    Five-model ensemble with stacking meta-learner and entropy-based confidence.

    Models:
        1. MarketTransformer     — attention over full sequence
        2. MarketLSTM            — bidirectional LSTM + attention
        3. MarketTCN             — dilated 1-D convolutions (NEW)
        4. HistGradientBoosting  — sklearn tabular baseline
        5. GradBoostExtra        — LightGBM / XGBoost (NEW)

    Meta-learner: HistGradientBoostingClassifier on 15-dim stacked probs
    Dynamic weight adjustment based on rolling per-model accuracy.
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        lstm_config: Optional[LSTMConfig] = None,
        tcn_config: Optional[TCNConfig] = None,
        xgb_config: Optional[XGBoostConfig] = None,
    ):
        self.config = config or EnsembleConfig()

        # ── base models ───────────────────────────────────────────────────────
        self.transformer = MarketTransformer(transformer_config or TransformerConfig())
        self.lstm = MarketLSTM(lstm_config or LSTMConfig())
        self.tcn = MarketTCN(tcn_config or TCNConfig())

        self.gradient_boost = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.xgboost_model = GradBoostExtra(xgb_config or XGBoostConfig())

        # ── meta-learner (upgraded: HistGradBoost instead of LogReg) ─────────
        # Learns non-linear combinations of the 5 base model outputs.
        # Input: 15-dim vector (5 models × 3 class probs)
        self.meta_learner = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )

        # ── dynamic weights (order matches _MODEL_NAMES below) ───────────────
        self.weights = np.array([
            self.config.transformer_weight,
            self.config.lstm_weight,
            self.config.tcn_weight,
            self.config.gradient_boost_weight,
            self.config.xgboost_weight,
        ])

        # ── state flags ───────────────────────────────────────────────────────
        self.gb_fitted = False
        self.xgb_fitted = False
        self.meta_fitted = False
        self.models_loaded = False
        self.device = torch.device("cpu")

        # ── accuracy tracking for dynamic weights ─────────────────────────────
        _MODEL_NAMES = ["transformer", "lstm", "tcn", "gradient_boost", "xgboost"]
        self._model_names: List[str] = _MODEL_NAMES
        self._accuracy_tracker: Dict[str, deque] = {
            name: deque(maxlen=self.config.weight_lookback)
            for name in _MODEL_NAMES
        }

    # ── device management ─────────────────────────────────────────────────────

    def to_device(self, device: str = "cpu") -> "EnsembleManager":
        """Move neural network models to the specified device."""
        self.device = torch.device(device)
        self.transformer = self.transformer.to(self.device)
        self.lstm = self.lstm.to(self.device)
        self.tcn = self.tcn.to(self.device)
        return self

    # ── individual predictors ─────────────────────────────────────────────────

    def predict_transformer(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, 3) probabilities."""
        self.transformer.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(self.device)
            return self.transformer.predict(t).cpu().numpy()

    def predict_lstm(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, 3) probabilities."""
        self.lstm.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(self.device)
            return self.lstm.predict(t).cpu().numpy()

    def predict_tcn(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, 3) probabilities."""
        self.tcn.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(self.device)
            return self.tcn.predict(t).cpu().numpy()

    def predict_gradient_boost(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, 3) probabilities (flattened internally)."""
        if not self.gb_fitted:
            return np.full((x.shape[0], 3), 1.0 / 3.0)
        x_flat = x.reshape(x.shape[0], -1)
        return self.gradient_boost.predict_proba(x_flat)

    def predict_xgboost(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, 3) probabilities (flattened internally)."""
        return self.xgboost_model.predict_proba(x)

    # ── fitting helpers ───────────────────────────────────────────────────────

    def fit_gradient_boost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the sklearn HistGradientBoosting baseline."""
        X_flat = X.reshape(X.shape[0], -1)
        self.gradient_boost.fit(X_flat, y)
        self.gb_fitted = True

    def fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM / XGBoost model."""
        self.xgboost_model.fit(X, y)
        self.xgb_fitted = True

    def fit_meta_learner(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the meta-learner on stacked predictions.

        Args:
            X: Stacked predictions from all 5 models — shape (n_samples, 15)
            y: True labels (n_samples,)
        """
        self.meta_learner.fit(X, y)
        self.meta_fitted = True

    # ── main prediction ───────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full ensemble prediction combining all 5 models.

        Args:
            x: (batch, seq_len, features)

        Returns dict with keys:
            'probabilities'   — (batch, 3)  final ensemble class probs
            'confidence'      — (batch,)    entropy-weighted confidence score
            'agreement'       — (batch,)    raw fraction of models agreeing
            'individual_preds' — dict of model_name → (batch, 3)
        """
        # ── collect individual predictions ────────────────────────────────────
        t_probs  = self.predict_transformer(x)
        l_probs  = self.predict_lstm(x)
        c_probs  = self.predict_tcn(x)
        gb_probs = self.predict_gradient_boost(x)
        xb_probs = self.predict_xgboost(x)

        # ── stacked input for meta-learner (15-dim) ───────────────────────────
        stacked = np.concatenate(
            [t_probs, l_probs, c_probs, gb_probs, xb_probs], axis=1
        )  # (batch, 15)

        # ── ensemble probabilities ─────────────────────────────────────────────
        if self.meta_fitted:
            ensemble_probs = self.meta_learner.predict_proba(stacked)
        else:
            # Weighted average fallback before meta-learner is trained
            ensemble_probs = (
                self.weights[0] * t_probs +
                self.weights[1] * l_probs +
                self.weights[2] * c_probs +
                self.weights[3] * gb_probs +
                self.weights[4] * xb_probs
            )

        # ── agreement (fraction of models voting for the majority class) ───────
        all_preds = np.stack([
            np.argmax(t_probs,  axis=1),
            np.argmax(l_probs,  axis=1),
            np.argmax(c_probs,  axis=1),
            np.argmax(gb_probs, axis=1),
            np.argmax(xb_probs, axis=1),
        ], axis=1)  # (batch, 5)

        agreement = np.array([
            np.max(np.bincount(all_preds[i], minlength=3)) / float(_NUM_MODELS)
            for i in range(all_preds.shape[0])
        ])

        # ── entropy-based confidence (IMPROVED) ───────────────────────────────
        confidence = self._compute_confidence(ensemble_probs, agreement)

        return {
            "probabilities": ensemble_probs,
            "confidence": confidence,
            "agreement": agreement,
            "individual_preds": {
                "transformer":    t_probs,
                "lstm":           l_probs,
                "tcn":            c_probs,
                "gradient_boost": gb_probs,
                "xgboost":        xb_probs,
            },
        }

    # ── confidence computation ────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        ensemble_probs: np.ndarray,
        agreement: np.ndarray,
        num_classes: int = 3,
        num_models: int = _NUM_MODELS,
    ) -> np.ndarray:
        """
        Entropy-weighted confidence score in [0, 1].

        Formula:
            confidence = max_prob × (1 − entropy_norm) × agreement_norm

        Components:
            max_prob        — strength of directional prediction
            1 − entropy_norm — certainty of the full distribution
                               (near-uniform → entropy → 1 → penalised heavily)
            agreement_norm  — inter-model consensus, corrected for chance level
                               = (raw_agreement − 1/N) / (1 − 1/N)

        Why better than the old `max_prob × agreement`:
          A distribution like [0.45, 0.30, 0.25] has max_prob=0.45 which looks
          moderate, but its normalised entropy ≈ 0.97 → certainty ≈ 0.03,
          so confidence collapses to near-zero and the signal is correctly filtered.
        """
        # Shannon entropy, normalised to [0, 1]
        probs_safe = np.clip(ensemble_probs, 1e-10, 1.0)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
        entropy_norm = entropy / np.log(num_classes)          # divide by log(3)

        # Directional strength
        max_prob = np.max(ensemble_probs, axis=1)

        # Agreement normalised to [0, 1] relative to chance level (1/N)
        chance = 1.0 / num_models
        agreement_norm = np.clip(
            (agreement - chance) / (1.0 - chance), 0.0, 1.0
        )

        return max_prob * (1.0 - entropy_norm) * (0.5 + 0.5 * agreement_norm)

    # ── dynamic weight update ─────────────────────────────────────────────────

    def update_weights(
        self, true_label: int, predictions: Dict[str, int]
    ) -> None:
        """
        Update per-model weights based on rolling accuracy.

        Args:
            true_label:  Actual outcome (0=SELL, 1=HOLD, 2=BUY)
            predictions: {model_name: predicted_label} for all 5 models
        """
        if not self.config.dynamic_weights:
            return

        for name, pred in predictions.items():
            if name in self._accuracy_tracker:
                self._accuracy_tracker[name].append(
                    1.0 if pred == true_label else 0.0
                )

        # Softmax-like weight reassignment from rolling accuracy
        accuracies = np.array([
            np.mean(list(self._accuracy_tracker[n]))
            if len(self._accuracy_tracker[n]) > 0
            else 1.0 / _NUM_MODELS
            for n in self._model_names
        ])

        total = accuracies.sum()
        if total > 0:
            self.weights = accuracies / total
        else:
            self.weights = np.array([
                self.config.transformer_weight,
                self.config.lstm_weight,
                self.config.tcn_weight,
                self.config.gradient_boost_weight,
                self.config.xgboost_weight,
            ])

    # ── uncertainty helper ────────────────────────────────────────────────────

    def get_disagreement_signal(self, x: np.ndarray) -> float:
        """
        Disagreement score as an uncertainty indicator (0=full agreement, 1=max discord).
        Higher value → smaller position size recommended.
        """
        result = self.predict(x)
        return float(1.0 - result["agreement"].mean())

    # ── checkpoint I/O ────────────────────────────────────────────────────────

    def save_models(self, path: str) -> None:
        """Save all model checkpoints to directory."""
        import joblib
        os.makedirs(path, exist_ok=True)

        # Neural networks
        torch.save(self.transformer.state_dict(),
                   os.path.join(path, "transformer.pth"))
        torch.save(self.lstm.state_dict(),
                   os.path.join(path, "lstm.pth"))
        torch.save(self.tcn.state_dict(),
                   os.path.join(path, "tcn.pth"))

        # Tree models
        if self.gb_fitted:
            joblib.dump(self.gradient_boost,
                        os.path.join(path, "gradient_boost.joblib"))
        self.xgboost_model.save(os.path.join(path, "xgboost_extra.joblib"))

        # Meta-learner
        if self.meta_fitted:
            joblib.dump(self.meta_learner,
                        os.path.join(path, "meta_learner.joblib"))

        logger.info(f"[Ensemble] All 5 models saved → {path}")

    def load_models(self, path: str) -> None:
        """Load all model checkpoints from directory."""
        import joblib
        nn_loaded = False

        # Neural networks
        for fname, attr in [
            ("transformer.pth", "transformer"),
            ("lstm.pth",        "lstm"),
            ("tcn.pth",         "tcn"),
        ]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                getattr(self, attr).load_state_dict(
                    torch.load(fpath, map_location=self.device,
                               weights_only=True),
                    strict=False,
                )
                nn_loaded = True
                logger.info(f"[Ensemble] Loaded {fname}")

        # Tree models
        gb_path = os.path.join(path, "gradient_boost.joblib")
        if os.path.exists(gb_path):
            # NOTE: joblib.load uses pickle. Only load from trusted local paths.
            self.gradient_boost = joblib.load(gb_path)
            self.gb_fitted = True

        xgb_path = os.path.join(path, "xgboost_extra.joblib")
        if os.path.exists(xgb_path):
            self.xgboost_model.load(xgb_path)
            self.xgb_fitted = self.xgboost_model.fitted

        # Meta-learner
        meta_path = os.path.join(path, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.meta_fitted = True

        # Backward compat: load old 3-model meta-learner if new one missing
        # (will fall back to weighted average until retrained)
        if not self.meta_fitted:
            logger.warning(
                "[Ensemble] meta_learner.joblib not found — "
                "will use weighted average until retrained."
            )

        if nn_loaded:
            self.models_loaded = True
            logger.info(
                f"[Ensemble] Models loaded. "
                f"gb={self.gb_fitted} xgb={self.xgb_fitted} "
                f"meta={self.meta_fitted}"
            )
