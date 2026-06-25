"""
=============================================================
  Python ML Bridge - Ensemble Model Manager  (v5 — 14 models)

  Complete model stack:
    1.  MarketTransformer    global self-attention              (0.07)
    2.  MarketLSTM           BiLSTM + attention                 (0.06)
    3.  MarketTCN            dilated temporal convolutions      (0.06)
    4.  MarketPatchTST       patch-based SOTA 2023              (0.10)
    5.  MarketTFT            Temporal Fusion Transformer        (0.10)
    6.  MarketNHiTS          hierarchical multi-scale MLP       (0.06)
    7.  MarketITransformer   feature-space attention 2024       (0.10)
    8.  MarketMamba          selective state space S6 2023      (0.08)
    9.  MarketDLinear        trend/residual decomposition       (0.04)
    10. MarketXLSTM          matrix memory LSTM 2024  NEW       (0.11)
    11. MarketTimesNet       2D temporal variation 2023 NEW     (0.09)
    12. HistGradientBoosting sklearn tabular baseline           (0.05)
    13. GradBoostExtra       LightGBM / XGBoost                 (0.04)
    14. CatBoostModel        ordered boosting                   (0.04)

  Meta-learner: HistGradientBoostingClassifier on 42-dim stack
                (14 models × 3 class probabilities)

  Confidence formula:
    confidence = max_prob × (1 − entropy_norm) × (0.5 + 0.5 × agreement_norm)
    agreement_norm = (raw_agreement − 1/14) / (1 − 1/14)
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
from models.gradient_boost_extra import GradBoostExtra
from models.catboost_model    import CatBoostModel

logger = logging.getLogger(__name__)

_NUM_MODELS  = 14
_MODEL_NAMES = [
    "transformer", "lstm", "tcn",
    "patch_tst", "tft", "nhits",
    "itransformer", "mamba", "dlinear",
    "xlstm", "timesnet",
    "gradient_boost", "xgboost", "catboost",
]


class EnsembleManager:
    """
    14-model ensemble with stacking meta-learner and entropy-based confidence.
    Backward-compatible: any subset of checkpoints loads fine —
    missing models fall back to weighted average until retrained.
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
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

        # ── tree models ────────────────────────────────────────────────────
        self.gradient_boost = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
        )
        self.xgboost_model  = GradBoostExtra(xgb_config      or XGBoostConfig())
        self.catboost_model = CatBoostModel(catboost_config   or CatBoostConfig())

        # ── meta-learner: 42-dim → final class ────────────────────────────
        self.meta_learner = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=0.1,
            random_state=42, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
        )

        # ── initial weights ────────────────────────────────────────────────
        self.weights = np.array([
            self.config.transformer_weight,
            self.config.lstm_weight,
            self.config.tcn_weight,
            self.config.patch_tst_weight,
            self.config.tft_weight,
            self.config.nhits_weight,
            self.config.itransformer_weight,
            self.config.mamba_weight,
            self.config.dlinear_weight,
            self.config.xlstm_weight,
            self.config.timesnet_weight,
            self.config.gradient_boost_weight,
            self.config.xgboost_weight,
            self.config.catboost_weight,
        ])

        # ── state ──────────────────────────────────────────────────────────
        self.gb_fitted       = False
        self.xgb_fitted      = False
        self.catboost_fitted = False
        self.meta_fitted     = False
        self.models_loaded   = False
        self.device          = torch.device("cpu")

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
                  self.xlstm, self.timesnet]:
            m.to(self.device)
        return self

    # ── individual predictors ──────────────────────────────────────────────

    def _nn_predict(self, model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
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
        """X: stacked predictions (n_samples, 42)"""
        self.meta_learner.fit(X, y); self.meta_fitted = True

    # ── main predict ──────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Full 14-model ensemble prediction."""
        t_p   = self.predict_transformer(x)
        l_p   = self.predict_lstm(x)
        c_p   = self.predict_tcn(x)
        pt_p  = self.predict_patch_tst(x)
        tf_p  = self.predict_tft(x)
        nh_p  = self.predict_nhits(x)
        it_p  = self.predict_itransformer(x)
        mb_p  = self.predict_mamba(x)
        dl_p  = self.predict_dlinear(x)
        xl_p  = self.predict_xlstm(x)
        tn_p  = self.predict_timesnet(x)
        gb_p  = self.predict_gradient_boost(x)
        xb_p  = self.predict_xgboost(x)
        cb_p  = self.predict_catboost(x)

        all_probs = [t_p, l_p, c_p, pt_p, tf_p, nh_p,
                     it_p, mb_p, dl_p, xl_p, tn_p,
                     gb_p, xb_p, cb_p]

        stacked = np.concatenate(all_probs, axis=1)   # (batch, 42)

        if self.meta_fitted:
            ensemble_probs = self.meta_learner.predict_proba(stacked)
        else:
            ensemble_probs = sum(w * p for w, p in zip(self.weights, all_probs))

        all_preds = np.stack(
            [np.argmax(p, axis=1) for p in all_probs], axis=1
        )  # (batch, 14)
        agreement = np.array([
            np.max(np.bincount(all_preds[i], minlength=3)) / _NUM_MODELS
            for i in range(all_preds.shape[0])
        ])

        return {
            "probabilities": ensemble_probs,
            "confidence":    self._compute_confidence(ensemble_probs, agreement),
            "agreement":     agreement,
            "individual_preds": {
                "transformer": t_p, "lstm": l_p, "tcn": c_p,
                "patch_tst": pt_p, "tft": tf_p, "nhits": nh_p,
                "itransformer": it_p, "mamba": mb_p, "dlinear": dl_p,
                "xlstm": xl_p, "timesnet": tn_p,
                "gradient_boost": gb_p, "xgboost": xb_p, "catboost": cb_p,
            },
        }

    # ── confidence ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        ensemble_probs: np.ndarray,
        agreement: np.ndarray,
        num_classes: int = 3,
        num_models: int = _NUM_MODELS,
    ) -> np.ndarray:
        probs_safe   = np.clip(ensemble_probs, 1e-10, 1.0)
        entropy_norm = (
            -np.sum(probs_safe * np.log(probs_safe), axis=1) / np.log(num_classes)
        )
        max_prob       = np.max(ensemble_probs, axis=1)
        chance         = 1.0 / num_models
        agreement_norm = np.clip((agreement - chance) / (1.0 - chance), 0.0, 1.0)
        return max_prob * (1.0 - entropy_norm) * (0.5 + 0.5 * agreement_norm)

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
            "transformer.pth":  self.transformer,
            "lstm.pth":         self.lstm,
            "tcn.pth":          self.tcn,
            "patch_tst.pth":    self.patch_tst,
            "tft.pth":          self.tft,
            "nhits.pth":        self.nhits,
            "itransformer.pth": self.itransformer,
            "mamba.pth":        self.mamba,
            "dlinear.pth":      self.dlinear,
            "xlstm.pth":        self.xlstm,
            "timesnet.pth":     self.timesnet,
        }
        for fname, model in nn_map.items():
            torch.save(model.state_dict(), os.path.join(path, fname))

        if self.gb_fitted:
            joblib.dump(self.gradient_boost,
                        os.path.join(path, "gradient_boost.joblib"))
        self.xgboost_model.save(os.path.join(path, "xgboost_extra.joblib"))
        self.catboost_model.save(os.path.join(path, "catboost.joblib"))
        if self.meta_fitted:
            joblib.dump(self.meta_learner,
                        os.path.join(path, "meta_learner.joblib"))
        logger.info(f"[Ensemble] All 14 models saved → {path}")

    def load_models(self, path: str) -> None:
        import joblib
        nn_loaded = False
        nn_map = {
            "transformer.pth":  "transformer",
            "lstm.pth":         "lstm",
            "tcn.pth":          "tcn",
            "patch_tst.pth":    "patch_tst",
            "tft.pth":          "tft",
            "nhits.pth":        "nhits",
            "itransformer.pth": "itransformer",
            "mamba.pth":        "mamba",
            "dlinear.pth":      "dlinear",
            "xlstm.pth":        "xlstm",
            "timesnet.pth":     "timesnet",
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
        gb_path = os.path.join(path, "gradient_boost.joblib")
        if os.path.exists(gb_path):
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
