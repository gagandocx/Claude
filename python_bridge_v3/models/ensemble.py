"""
=============================================================
  Python ML Bridge - Ensemble Model Manager  (v6 — 17 models)

  Complete model stack:
    1.  MarketTransformer    global self-attention              (0.06)
    2.  MarketLSTM           BiLSTM + attention                 (0.05)
    3.  MarketTCN            dilated temporal convolutions      (0.05)
    4.  MarketPatchTST       patch-based SOTA 2023              (0.08)
    5.  MarketTFT            Temporal Fusion Transformer        (0.08)
    6.  MarketNHiTS          hierarchical multi-scale MLP       (0.05)
    7.  MarketITransformer   feature-space attention 2024       (0.08)
    8.  MarketMamba          selective state space S6 2023      (0.07)
    9.  MarketDLinear        trend/residual decomposition       (0.03)
    10. MarketXLSTM          matrix memory LSTM 2024            (0.08)
    11. MarketTimesNet       2D temporal via FFT 2023           (0.06)
    12. MarketChronos        Amazon pre-trained foundation NEW  (0.09)
    13. MarketTimeMixer      multi-scale decomp mixing NEW      (0.07)
    14. MarketSOFTS          star aggregate O(N) fusion NEW     (0.05)
    15. HistGradientBoosting sklearn tabular baseline           (0.04)
    16. GradBoostExtra       LightGBM / XGBoost                 (0.03)
    17. CatBoostModel        ordered boosting                   (0.03)

  Meta-learner : HistGradientBoostingClassifier on 51-dim stacked probs
  Confidence   : max_prob × (1−entropy) × (0.5 + 0.5 × agreement_norm)
  Agreement    : normalised against 1/17 chance baseline
=============================================================
"""
import os, sys, logging
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
    ITransformerConfig, MambaConfig, DLinearConfig,
    xLSTMConfig, TimesNetConfig,
    ChronosConfig, TimeMixerConfig, SOFTSConfig,
    XGBoostConfig, CatBoostConfig,
)
from models.transformer_model import MarketTransformer
from models.lstm_model        import MarketLSTM
from models.tcn_model         import MarketTCN
from models.patch_tst         import MarketPatchTST
from models.tft_model         import MarketTFT
from models.nhits_model       import MarketNHiTS
from models.itransformer      import MarketITransformer
from models.mamba_model       import MarketMamba
from models.dlinear_model     import MarketDLinear
from models.xlstm_model       import MarketXLSTM
from models.timesnet_model    import MarketTimesNet
from models.chronos_model     import MarketChronos
from models.timemixer_model   import MarketTimeMixer
from models.softs_model       import MarketSOFTS
from models.gradient_boost_extra import GradBoostExtra
from models.catboost_model    import CatBoostModel

logger = logging.getLogger(__name__)

_NUM_MODELS  = 17
_MODEL_NAMES = [
    "transformer", "lstm", "tcn",
    "patch_tst", "tft", "nhits",
    "itransformer", "mamba", "dlinear",
    "xlstm", "timesnet",
    "chronos", "timemixer", "softs",
    "gradient_boost", "xgboost", "catboost",
]


class EnsembleManager:
    """
    17-model ensemble with stacking meta-learner and entropy-based confidence.
    Backward-compatible — any subset of checkpoints loads fine.
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig]       = None,
        transformer_config:  Optional[TransformerConfig]  = None,
        lstm_config:         Optional[LSTMConfig]         = None,
        tcn_config:          Optional[TCNConfig]          = None,
        patch_tst_config:    Optional[PatchTSTConfig]     = None,
        tft_config:          Optional[TFTConfig]          = None,
        nhits_config:        Optional[NHiTSConfig]        = None,
        itransformer_config: Optional[ITransformerConfig] = None,
        mamba_config:        Optional[MambaConfig]        = None,
        dlinear_config:      Optional[DLinearConfig]      = None,
        xlstm_config:        Optional[xLSTMConfig]        = None,
        timesnet_config:     Optional[TimesNetConfig]     = None,
        chronos_config:      Optional[ChronosConfig]      = None,
        timemixer_config:    Optional[TimeMixerConfig]    = None,
        softs_config:        Optional[SOFTSConfig]        = None,
        xgb_config:          Optional[XGBoostConfig]      = None,
        catboost_config:     Optional[CatBoostConfig]     = None,
    ):
        self.config = config or EnsembleConfig()

        # ── neural models ──────────────────────────────────────────────────
        self.transformer  = MarketTransformer(transformer_config  or TransformerConfig())
        self.lstm         = MarketLSTM(lstm_config                or LSTMConfig())
        self.tcn          = MarketTCN(tcn_config                  or TCNConfig())
        self.patch_tst    = MarketPatchTST(patch_tst_config       or PatchTSTConfig())
        self.tft          = MarketTFT(tft_config                  or TFTConfig())
        self.nhits        = MarketNHiTS(nhits_config              or NHiTSConfig())
        self.itransformer = MarketITransformer(itransformer_config or ITransformerConfig())
        self.mamba        = MarketMamba(mamba_config              or MambaConfig())
        self.dlinear      = MarketDLinear(dlinear_config          or DLinearConfig())
        self.xlstm        = MarketXLSTM(xlstm_config              or xLSTMConfig())
        self.timesnet     = MarketTimesNet(timesnet_config        or TimesNetConfig())
        self.chronos      = MarketChronos(chronos_config          or ChronosConfig())
        self.timemixer    = MarketTimeMixer(timemixer_config      or TimeMixerConfig())
        self.softs        = MarketSOFTS(softs_config              or SOFTSConfig())

        # ── tree models ────────────────────────────────────────────────────
        self.gradient_boost = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
        )
        self.xgboost_model  = GradBoostExtra(xgb_config      or XGBoostConfig())
        self.catboost_model = CatBoostModel(catboost_config   or CatBoostConfig())

        # ── meta-learner: 51-dim → final class ────────────────────────────
        self.meta_learner = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
        )

        # ── initial weights (order = _MODEL_NAMES) ────────────────────────
        self.weights = np.array([
            self.config.transformer_weight, self.config.lstm_weight,
            self.config.tcn_weight, self.config.patch_tst_weight,
            self.config.tft_weight, self.config.nhits_weight,
            self.config.itransformer_weight, self.config.mamba_weight,
            self.config.dlinear_weight, self.config.xlstm_weight,
            self.config.timesnet_weight, self.config.chronos_weight,
            self.config.timemixer_weight, self.config.softs_weight,
            self.config.gradient_boost_weight, self.config.xgboost_weight,
            self.config.catboost_weight,
        ])

        # ── state ──────────────────────────────────────────────────────────
        self.gb_fitted = self.xgb_fitted = self.catboost_fitted = False
        self.meta_fitted = self.models_loaded = False
        self.device = torch.device("cpu")

        self._model_names: List[str] = _MODEL_NAMES
        self._accuracy_tracker: Dict[str, deque] = {
            n: deque(maxlen=self.config.weight_lookback) for n in _MODEL_NAMES
        }

    # ── device ────────────────────────────────────────────────────────────

    def to_device(self, device: str = "cpu") -> "EnsembleManager":
        self.device = torch.device(device)
        for m in [self.transformer, self.lstm, self.tcn,
                  self.patch_tst, self.tft, self.nhits,
                  self.itransformer, self.mamba, self.dlinear,
                  self.xlstm, self.timesnet,
                  self.chronos, self.timemixer, self.softs]:
            m.to(self.device)
        return self

    # ── per-model predictors ──────────────────────────────────────────────

    def _nn_predict(self, model, x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model.predict(
                torch.FloatTensor(x).to(self.device)
            ).cpu().numpy()

    def predict_transformer(self, x):  return self._nn_predict(self.transformer,  x)
    def predict_lstm(self, x):         return self._nn_predict(self.lstm,          x)
    def predict_tcn(self, x):          return self._nn_predict(self.tcn,           x)
    def predict_patch_tst(self, x):    return self._nn_predict(self.patch_tst,     x)
    def predict_tft(self, x):          return self._nn_predict(self.tft,           x)
    def predict_nhits(self, x):        return self._nn_predict(self.nhits,         x)
    def predict_itransformer(self, x): return self._nn_predict(self.itransformer,  x)
    def predict_mamba(self, x):        return self._nn_predict(self.mamba,         x)
    def predict_dlinear(self, x):      return self._nn_predict(self.dlinear,       x)
    def predict_xlstm(self, x):        return self._nn_predict(self.xlstm,         x)
    def predict_timesnet(self, x):     return self._nn_predict(self.timesnet,      x)
    def predict_chronos(self, x):      return self._nn_predict(self.chronos,       x)
    def predict_timemixer(self, x):    return self._nn_predict(self.timemixer,     x)
    def predict_softs(self, x):        return self._nn_predict(self.softs,         x)

    def predict_gradient_boost(self, x: np.ndarray) -> np.ndarray:
        if not self.gb_fitted:
            return np.full((x.shape[0], 3), 1.0 / 3.0)
        return self.gradient_boost.predict_proba(x.reshape(x.shape[0], -1))

    def predict_xgboost(self, x):  return self.xgboost_model.predict_proba(x)
    def predict_catboost(self, x): return self.catboost_model.predict_proba(x)

    # ── fitting ───────────────────────────────────────────────────────────

    def fit_gradient_boost(self, X, y):
        self.gradient_boost.fit(X.reshape(X.shape[0], -1), y)
        self.gb_fitted = True

    def fit_xgboost(self, X, y):
        self.xgboost_model.fit(X, y); self.xgb_fitted = True

    def fit_catboost(self, X, y):
        self.catboost_model.fit(X, y); self.catboost_fitted = True

    def fit_meta_learner(self, X, y):
        """X: stacked predictions (n_samples, 51)"""
        self.meta_learner.fit(X, y); self.meta_fitted = True

    # ── main predict ──────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Full 17-model ensemble prediction."""
        all_probs = [
            self.predict_transformer(x), self.predict_lstm(x),
            self.predict_tcn(x),         self.predict_patch_tst(x),
            self.predict_tft(x),         self.predict_nhits(x),
            self.predict_itransformer(x),self.predict_mamba(x),
            self.predict_dlinear(x),     self.predict_xlstm(x),
            self.predict_timesnet(x),    self.predict_chronos(x),
            self.predict_timemixer(x),   self.predict_softs(x),
            self.predict_gradient_boost(x),
            self.predict_xgboost(x),     self.predict_catboost(x),
        ]
        stacked = np.concatenate(all_probs, axis=1)   # (batch, 51)

        ensemble_probs = (
            self.meta_learner.predict_proba(stacked) if self.meta_fitted
            else sum(w * p for w, p in zip(self.weights, all_probs))
        )

        all_preds  = np.stack([np.argmax(p, axis=1) for p in all_probs], axis=1)
        agreement  = np.array([
            np.max(np.bincount(all_preds[i], minlength=3)) / _NUM_MODELS
            for i in range(all_preds.shape[0])
        ])
        confidence = self._compute_confidence(ensemble_probs, agreement)

        return {
            "probabilities":   ensemble_probs,
            "confidence":      confidence,
            "agreement":       agreement,
            "individual_preds": dict(zip(_MODEL_NAMES, all_probs)),
        }

    # ── confidence ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(ensemble_probs, agreement,
                            num_classes=3, num_models=_NUM_MODELS):
        p = np.clip(ensemble_probs, 1e-10, 1.0)
        entropy_norm   = -np.sum(p * np.log(p), axis=1) / np.log(num_classes)
        max_prob       = np.max(ensemble_probs, axis=1)
        chance         = 1.0 / num_models
        agreement_norm = np.clip((agreement - chance) / (1.0 - chance), 0.0, 1.0)
        return max_prob * (1.0 - entropy_norm) * (0.5 + 0.5 * agreement_norm)

    # ── regime weight override ────────────────────────────────────────────

    def set_regime_weights(self, weights: np.ndarray) -> None:
        """
        Temporarily override model weights with regime-specific values.

        Used by RegimeModelRouter to apply regime-optimized weights
        before calling predict(). Weights persist until next call to
        set_regime_weights() or update_weights().

        Args:
            weights: np.ndarray of shape (17,) summing to ~1.0
        """
        if len(weights) != _NUM_MODELS:
            logger.warning("[Ensemble] set_regime_weights: expected %d weights, got %d",
                           _NUM_MODELS, len(weights))
            return
        # Normalize to ensure valid distribution
        total = weights.sum()
        if total > 0:
            self.weights = weights / total
        else:
            self.weights = np.ones(_NUM_MODELS) / _NUM_MODELS
        logger.debug("[Ensemble] Regime weights set. Top 3: %s",
                     sorted(zip(_MODEL_NAMES, self.weights),
                            key=lambda x: x[1], reverse=True)[:3])

    def get_individual_predictions(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get all 17 model predictions without combining them.

        Returns raw probability arrays from each model. Used by:
        - RegimeModelRouter for selective model weighting
        - DisagreementSignal for volatility detection
        - Any component needing per-model outputs

        Args:
            x: Input features array (batch, seq_len, features)

        Returns:
            Dict mapping model name -> probability array (batch, 3)
        """
        all_probs = [
            self.predict_transformer(x), self.predict_lstm(x),
            self.predict_tcn(x),         self.predict_patch_tst(x),
            self.predict_tft(x),         self.predict_nhits(x),
            self.predict_itransformer(x),self.predict_mamba(x),
            self.predict_dlinear(x),     self.predict_xlstm(x),
            self.predict_timesnet(x),    self.predict_chronos(x),
            self.predict_timemixer(x),   self.predict_softs(x),
            self.predict_gradient_boost(x),
            self.predict_xgboost(x),     self.predict_catboost(x),
        ]
        return dict(zip(_MODEL_NAMES, all_probs))

    # ── dynamic weights ───────────────────────────────────────────────────

    def update_weights(self, true_label: int, predictions: Dict[str, int]) -> None:
        if not self.config.dynamic_weights:
            return
        for name, pred in predictions.items():
            if name in self._accuracy_tracker:
                self._accuracy_tracker[name].append(
                    1.0 if pred == true_label else 0.0
                )
        accs  = np.array([
            np.mean(list(self._accuracy_tracker[n]))
            if self._accuracy_tracker[n] else 1.0 / _NUM_MODELS
            for n in self._model_names
        ])
        total = accs.sum()
        self.weights = accs / total if total > 0 else self.weights.copy()

    def get_disagreement_signal(self, x: np.ndarray) -> float:
        return float(1.0 - self.predict(x)["agreement"].mean())

    # ── checkpoint I/O ────────────────────────────────────────────────────

    def save_models(self, path: str) -> None:
        import joblib
        os.makedirs(path, exist_ok=True)
        nn_map = {
            "transformer.pth":  self.transformer,  "lstm.pth":  self.lstm,
            "tcn.pth":          self.tcn,           "patch_tst.pth": self.patch_tst,
            "tft.pth":          self.tft,           "nhits.pth": self.nhits,
            "itransformer.pth": self.itransformer,  "mamba.pth": self.mamba,
            "dlinear.pth":      self.dlinear,       "xlstm.pth": self.xlstm,
            "timesnet.pth":     self.timesnet,      "chronos.pth": self.chronos,
            "timemixer.pth":    self.timemixer,     "softs.pth": self.softs,
        }
        for fname, model in nn_map.items():
            torch.save(model.state_dict(), os.path.join(path, fname))
        if self.gb_fitted:
            joblib.dump(self.gradient_boost,
                        os.path.join(path, "gradient_boost.joblib"))
        self.xgboost_model.save(os.path.join(path, "xgboost_extra.joblib"))
        self.catboost_model.save(os.path.join(path, "catboost.joblib"))
        if self.meta_fitted:
            joblib.dump(self.meta_learner, os.path.join(path, "meta_learner.joblib"))
        logger.info(f"[Ensemble] All 17 models saved → {path}")

    def load_models(self, path: str) -> None:
        import joblib
        nn_loaded = False
        nn_map = {
            "transformer.pth":  "transformer",  "lstm.pth":  "lstm",
            "tcn.pth":          "tcn",           "patch_tst.pth": "patch_tst",
            "tft.pth":          "tft",           "nhits.pth": "nhits",
            "itransformer.pth": "itransformer",  "mamba.pth": "mamba",
            "dlinear.pth":      "dlinear",       "xlstm.pth": "xlstm",
            "timesnet.pth":     "timesnet",      "chronos.pth": "chronos",
            "timemixer.pth":    "timemixer",     "softs.pth": "softs",
        }
        for fname, attr in nn_map.items():
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                getattr(self, attr).load_state_dict(
                    torch.load(fpath, map_location=self.device,
                               weights_only=True),
                    strict=False,
                )
                nn_loaded = True
                logger.info(f"[Ensemble] Loaded {fname}")
        # NOTE: joblib.load uses pickle — only load from trusted local paths.
        for fpath, attr, flag in [
            (os.path.join(path, "gradient_boost.joblib"),
             "gradient_boost", "gb_fitted"),
        ]:
            if os.path.exists(fpath):
                setattr(self, attr, joblib.load(fpath))
                setattr(self, flag, True)
        xgb_path = os.path.join(path, "xgboost_extra.joblib")
        if os.path.exists(xgb_path):
            self.xgboost_model.load(xgb_path)
            self.xgb_fitted = self.xgboost_model.fitted
        cb_path = os.path.join(path, "catboost.joblib")
        if os.path.exists(cb_path):
            self.catboost_model.load(cb_path)
            self.catboost_fitted = self.catboost_model.fitted
        meta_path = os.path.join(path, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.meta_fitted = True
        else:
            logger.warning("[Ensemble] meta_learner.joblib missing — "
                           "weighted average fallback until retrained.")
        if nn_loaded:
            self.models_loaded = True
            logger.info(f"[Ensemble] Load complete. "
                        f"gb={self.gb_fitted} xgb={self.xgb_fitted} "
                        f"cb={self.catboost_fitted} meta={self.meta_fitted}")
