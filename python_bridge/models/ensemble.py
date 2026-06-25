"""
=============================================================
  Python ML Bridge - Ensemble Model Manager  (v3 — 9 models)

  Complete model stack:
    1. MarketTransformer   — global self-attention              (weight 0.12)
    2. MarketLSTM          — bidirectional LSTM + attention     (weight 0.10)
    3. MarketTCN           — dilated temporal convolutions      (weight 0.10)
    4. MarketPatchTST      — patch-based SOTA transformer 2023  (weight 0.15)
    5. MarketTFT           — Temporal Fusion Transformer        (weight 0.15)
    6. MarketNHiTS         — hierarchical multi-scale MLP       (weight 0.10)
    7. HistGradientBoosting — sklearn tabular baseline          (weight 0.10)
    8. GradBoostExtra      — LightGBM / XGBoost                 (weight 0.09)
    9. CatBoostModel       — ordered boosting                   (weight 0.09)

  Meta-learner : HistGradientBoostingClassifier on 27-dim stacked predictions
                 (9 models × 3 class probabilities)

  Confidence formula (entropy-weighted):
    confidence = max_prob  ×  (1 − entropy_norm)  ×  (0.5 + 0.5 × agreement_norm)

    agreement_norm = (raw_agreement − 1/9) / (1 − 1/9)

  Why 9 models?
    Each model captures a structurally different signal:
    - Transformer/PatchTST  : global temporal dependencies
    - LSTM/TFT              : sequential context with learned gating
    - TCN/N-HiTS            : multi-scale local patterns
    - GB/XGB/CatBoost       : non-linear feature interactions from flattened window
    Disagreement between these model families is informative:
    high disagreement → low confidence → signal filtered → fewer bad trades.
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
    EnsembleConfig,
    TransformerConfig, LSTMConfig, TCNConfig,
    PatchTSTConfig, TFTConfig, NHiTSConfig,
    XGBoostConfig, CatBoostConfig,
)
from models.transformer_model import MarketTransformer
from models.lstm_model import MarketLSTM
from models.tcn_model import MarketTCN
from models.patch_tst import MarketPatchTST
from models.tft_model import MarketTFT
from models.nhits_model import MarketNHiTS
from models.gradient_boost_extra import GradBoostExtra
from models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)

_NUM_MODELS = 9   # used for agreement normalisation
_MODEL_NAMES = [
    "transformer", "lstm", "tcn",
    "patch_tst", "tft", "nhits",
    "gradient_boost", "xgboost", "catboost",
]


class EnsembleManager:
    """
    9-model ensemble with stacking meta-learner and entropy-based confidence.
    Backward-compatible: loads any subset of checkpoints that exist on disk.
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        lstm_config: Optional[LSTMConfig] = None,
        tcn_config: Optional[TCNConfig] = None,
        patch_tst_config: Optional[PatchTSTConfig] = None,
        tft_config: Optional[TFTConfig] = None,
        nhits_config: Optional[NHiTSConfig] = None,
        xgb_config: Optional[XGBoostConfig] = None,
        catboost_config: Optional[CatBoostConfig] = None,
    ):
        self.config = config or EnsembleConfig()

        # ── neural models ──────────────────────────────────────────────────
        self.transformer  = MarketTransformer(transformer_config or TransformerConfig())
        self.lstm         = MarketLSTM(lstm_config or LSTMConfig())
        self.tcn          = MarketTCN(tcn_config or TCNConfig())
        self.patch_tst    = MarketPatchTST(patch_tst_config or PatchTSTConfig())
        self.tft          = MarketTFT(tft_config or TFTConfig())
        self.nhits        = MarketNHiTS(nhits_config or NHiTSConfig())

        # ── tree models ────────────────────────────────────────────────────
        self.gradient_boost = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
        )
        self.xgboost_model  = GradBoostExtra(xgb_config or XGBoostConfig())
        self.catboost_model = CatBoostModel(catboost_config or CatBoostConfig())

        # ── meta-learner: 27-dim stacked probs → final class ──────────────
        self.meta_learner = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
        )

        # ── initial weights (order matches _MODEL_NAMES) ──────────────────
        self.weights = np.array([
            self.config.transformer_weight,
            self.config.lstm_weight,
            self.config.tcn_weight,
            self.config.patch_tst_weight,
            self.config.tft_weight,
            self.config.nhits_weight,
            self.config.gradient_boost_weight,
            self.config.xgboost_weight,
            self.config.catboost_weight,
        ])

        # ── state flags ────────────────────────────────────────────────────
        self.gb_fitted       = False
        self.xgb_fitted      = False
        self.catboost_fitted = False
        self.meta_fitted     = False
        self.models_loaded   = False
        self.device          = torch.device("cpu")

        # ── rolling accuracy trackers for dynamic weight update ────────────
        self._model_names: List[str] = _MODEL_NAMES
        self._accuracy_tracker: Dict[str, deque] = {
            name: deque(maxlen=self.config.weight_lookback)
            for name in _MODEL_NAMES
        }

    # ── device management ──────────────────────────────────────────────────

    def to_device(self, device: str = "cpu") -> "EnsembleManager":
        self.device = torch.device(device)
        for m in [self.transformer, self.lstm, self.tcn,
                  self.patch_tst, self.tft, self.nhits]:
            m.to(self.device)
        return self

    # ── individual predictors ──────────────────────────────────────────────

    def _nn_predict(self, model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
        """Generic neural-net predict: eval mode, no_grad, numpy output."""
        model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(self.device)
            return model.predict(t).cpu().numpy()

    def predict_transformer(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.transformer, x)

    def predict_lstm(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.lstm, x)

    def predict_tcn(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.tcn, x)

    def predict_patch_tst(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.patch_tst, x)

    def predict_tft(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.tft, x)

    def predict_nhits(self, x: np.ndarray) -> np.ndarray:
        return self._nn_predict(self.nhits, x)

    def predict_gradient_boost(self, x: np.ndarray) -> np.ndarray:
        if not self.gb_fitted:
            return np.full((x.shape[0], 3), 1.0 / 3.0)
        return self.gradient_boost.predict_proba(x.reshape(x.shape[0], -1))

    def predict_xgboost(self, x: np.ndarray) -> np.ndarray:
        return self.xgboost_model.predict_proba(x)

    def predict_catboost(self, x: np.ndarray) -> np.ndarray:
        return self.catboost_model.predict_proba(x)

    # ── fitting helpers ────────────────────────────────────────────────────

    def fit_gradient_boost(self, X: np.ndarray, y: np.ndarray) -> None:
        self.gradient_boost.fit(X.reshape(X.shape[0], -1), y)
        self.gb_fitted = True

    def fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        self.xgboost_model.fit(X, y)
        self.xgb_fitted = True

    def fit_catboost(self, X: np.ndarray, y: np.ndarray) -> None:
        self.catboost_model.fit(X, y)
        self.catboost_fitted = True

    def fit_meta_learner(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: stacked predictions (n_samples, 27)"""
        self.meta_learner.fit(X, y)
        self.meta_fitted = True

    # ── main prediction ────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full 9-model ensemble prediction.

        Args:
            x: (batch, seq_len, features)

        Returns dict:
            'probabilities'    — (batch, 3)
            'confidence'       — (batch,)
            'agreement'        — (batch,)
            'individual_preds' — {model_name: (batch, 3)}
        """
        t_probs   = self.predict_transformer(x)
        l_probs   = self.predict_lstm(x)
        c_probs   = self.predict_tcn(x)
        pt_probs  = self.predict_patch_tst(x)
        tft_probs = self.predict_tft(x)
        nh_probs  = self.predict_nhits(x)
        gb_probs  = self.predict_gradient_boost(x)
        xb_probs  = self.predict_xgboost(x)
        cb_probs  = self.predict_catboost(x)

        # 27-dim stacked for meta-learner
        stacked = np.concatenate(
            [t_probs, l_probs, c_probs, pt_probs, tft_probs,
             nh_probs, gb_probs, xb_probs, cb_probs], axis=1
        )

        if self.meta_fitted:
            ensemble_probs = self.meta_learner.predict_proba(stacked)
        else:
            # Weighted average fallback
            ensemble_probs = (
                self.weights[0] * t_probs  +
                self.weights[1] * l_probs  +
                self.weights[2] * c_probs  +
                self.weights[3] * pt_probs +
                self.weights[4] * tft_probs+
                self.weights[5] * nh_probs +
                self.weights[6] * gb_probs +
                self.weights[7] * xb_probs +
                self.weights[8] * cb_probs
            )

        # Agreement: fraction of 9 models voting for the majority class
        all_preds = np.stack([
            np.argmax(t_probs,   axis=1),
            np.argmax(l_probs,   axis=1),
            np.argmax(c_probs,   axis=1),
            np.argmax(pt_probs,  axis=1),
            np.argmax(tft_probs, axis=1),
            np.argmax(nh_probs,  axis=1),
            np.argmax(gb_probs,  axis=1),
            np.argmax(xb_probs,  axis=1),
            np.argmax(cb_probs,  axis=1),
        ], axis=1)  # (batch, 9)

        agreement = np.array([
            np.max(np.bincount(all_preds[i], minlength=3)) / float(_NUM_MODELS)
            for i in range(all_preds.shape[0])
        ])

        confidence = self._compute_confidence(ensemble_probs, agreement)

        return {
            "probabilities": ensemble_probs,
            "confidence":    confidence,
            "agreement":     agreement,
            "individual_preds": {
                "transformer":    t_probs,
                "lstm":           l_probs,
                "tcn":            c_probs,
                "patch_tst":      pt_probs,
                "tft":            tft_probs,
                "nhits":          nh_probs,
                "gradient_boost": gb_probs,
                "xgboost":        xb_probs,
                "catboost":       cb_probs,
            },
        }

    # ── confidence ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        ensemble_probs: np.ndarray,
        agreement: np.ndarray,
        num_classes: int = 3,
        num_models: int = _NUM_MODELS,
    ) -> np.ndarray:
        """
        confidence = max_prob × (1 − entropy_norm) × (0.5 + 0.5 × agreement_norm)

        With 9 models the agreement signal is much richer than 3 or 5 models:
        7/9 models agreeing is very different from 5/9 — both get correct scores.
        """
        probs_safe   = np.clip(ensemble_probs, 1e-10, 1.0)
        entropy_norm = (
            -np.sum(probs_safe * np.log(probs_safe), axis=1) / np.log(num_classes)
        )
        max_prob      = np.max(ensemble_probs, axis=1)
        chance        = 1.0 / num_models
        agreement_norm = np.clip(
            (agreement - chance) / (1.0 - chance), 0.0, 1.0
        )
        return max_prob * (1.0 - entropy_norm) * (0.5 + 0.5 * agreement_norm)

    # ── dynamic weights ────────────────────────────────────────────────────

    def update_weights(self, true_label: int, predictions: Dict[str, int]) -> None:
        if not self.config.dynamic_weights:
            return
        for name, pred in predictions.items():
            if name in self._accuracy_tracker:
                self._accuracy_tracker[name].append(
                    1.0 if pred == true_label else 0.0
                )
        accuracies = np.array([
            np.mean(list(self._accuracy_tracker[n]))
            if len(self._accuracy_tracker[n]) > 0 else 1.0 / _NUM_MODELS
            for n in self._model_names
        ])
        total = accuracies.sum()
        self.weights = accuracies / total if total > 0 else np.array([
            self.config.transformer_weight, self.config.lstm_weight,
            self.config.tcn_weight, self.config.patch_tst_weight,
            self.config.tft_weight, self.config.nhits_weight,
            self.config.gradient_boost_weight, self.config.xgboost_weight,
            self.config.catboost_weight,
        ])

    def get_disagreement_signal(self, x: np.ndarray) -> float:
        return float(1.0 - self.predict(x)["agreement"].mean())

    # ── checkpoint I/O ─────────────────────────────────────────────────────

    def save_models(self, path: str) -> None:
        import joblib
        os.makedirs(path, exist_ok=True)

        # Neural networks
        nn_map = {
            "transformer.pth": self.transformer,
            "lstm.pth":        self.lstm,
            "tcn.pth":         self.tcn,
            "patch_tst.pth":   self.patch_tst,
            "tft.pth":         self.tft,
            "nhits.pth":       self.nhits,
        }
        for fname, model in nn_map.items():
            torch.save(model.state_dict(), os.path.join(path, fname))

        # Tree models
        if self.gb_fitted:
            joblib.dump(self.gradient_boost,
                        os.path.join(path, "gradient_boost.joblib"))
        self.xgboost_model.save(os.path.join(path, "xgboost_extra.joblib"))
        self.catboost_model.save(os.path.join(path, "catboost.joblib"))

        # Meta-learner
        if self.meta_fitted:
            joblib.dump(self.meta_learner,
                        os.path.join(path, "meta_learner.joblib"))

        logger.info(f"[Ensemble] All 9 models saved → {path}")

    def load_models(self, path: str) -> None:
        import joblib
        nn_loaded = False

        # Neural networks — load whatever checkpoints exist
        nn_map = {
            "transformer.pth": "transformer",
            "lstm.pth":        "lstm",
            "tcn.pth":         "tcn",
            "patch_tst.pth":   "patch_tst",
            "tft.pth":         "tft",
            "nhits.pth":       "nhits",
        }
        for fname, attr in nn_map.items():
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                getattr(self, attr).load_state_dict(
                    torch.load(fpath, map_location=self.device, weights_only=True),
                    strict=False,
                )
                nn_loaded = True
                logger.info(f"[Ensemble] Loaded {fname}")

        # Tree models
        gb_path = os.path.join(path, "gradient_boost.joblib")
        if os.path.exists(gb_path):
            # NOTE: joblib.load uses pickle — only load from trusted local paths.
            self.gradient_boost = joblib.load(gb_path)
            self.gb_fitted = True

        xgb_path = os.path.join(path, "xgboost_extra.joblib")
        if os.path.exists(xgb_path):
            self.xgboost_model.load(xgb_path)
            self.xgb_fitted = self.xgboost_model.fitted

        cb_path = os.path.join(path, "catboost.joblib")
        if os.path.exists(cb_path):
            self.catboost_model.load(cb_path)
            self.catboost_fitted = self.catboost_model.fitted

        # Meta-learner
        meta_path = os.path.join(path, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.meta_fitted = True
        else:
            logger.warning(
                "[Ensemble] meta_learner.joblib not found — "
                "using weighted average until retrained."
            )

        if nn_loaded:
            self.models_loaded = True
            logger.info(
                f"[Ensemble] Load complete. "
                f"gb={self.gb_fitted} xgb={self.xgb_fitted} "
                f"cb={self.catboost_fitted} meta={self.meta_fitted}"
            )
