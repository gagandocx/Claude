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
import torch.nn as nn
import torch.optim as optim
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
    SharpeWeightConfig,
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

        # ── Sharpe-ratio-based model weighting ─────────────────────────────
        self._sharpe_config = SharpeWeightConfig()
        self._pnl_tracker: Dict[str, deque] = {
            n: deque(maxlen=self._sharpe_config.lookback_trades) for n in _MODEL_NAMES
        }
        self._sharpe_trade_counter: int = 0

        # ── Online learning state ─────────────────────────────────────────
        self._online_labels_buffer: List[tuple] = []  # (x_input, true_label)
        self._last_stacked_predictions: Optional[np.ndarray] = None

        # ── Meta-learner data accumulation ─────────────────────────────────
        self._meta_data_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "meta_learner_data.npz"
        )

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
        """Full 17-model ensemble prediction with feature schema validation."""
        # ── Feature schema validation ─────────────────────────────────────
        expected_features = 46
        if x.ndim == 3:
            actual_features = x.shape[2]
        elif x.ndim == 2:
            actual_features = x.shape[1]
        else:
            actual_features = None

        if actual_features is not None and actual_features != expected_features:
            logger.warning(
                f"[Ensemble] Feature count mismatch: got {actual_features}, "
                f"expected {expected_features}. Adjusting input."
            )
            if x.ndim == 3:
                if actual_features < expected_features:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, 0), (0, expected_features - actual_features))
                    x = np.pad(x, pad_width, mode='constant', constant_values=0.0)
                else:
                    # Truncate
                    x = x[:, :, :expected_features]
            elif x.ndim == 2:
                if actual_features < expected_features:
                    pad_width = ((0, 0), (0, expected_features - actual_features))
                    x = np.pad(x, pad_width, mode='constant', constant_values=0.0)
                else:
                    x = x[:, :expected_features]

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

        # Store stacked predictions for meta-learner data accumulation
        if self.config.accumulate_meta_data:
            self._last_stacked_predictions = stacked.copy()

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

    # ── Sharpe-ratio-based model weighting ────────────────────────────────

    def update_pnl_attribution(self, model_name: str, pnl: float) -> None:
        """
        Feed a per-trade PnL result attributed to a specific model.

        Args:
            model_name: Name of the model (must be in _MODEL_NAMES)
            pnl: Profit/loss for this trade attributed to this model
        """
        if model_name in self._pnl_tracker:
            self._pnl_tracker[model_name].append(pnl)

    def compute_sharpe_weights(self) -> np.ndarray:
        """
        Compute weights based on rolling Sharpe ratio per model.

        For each model, Sharpe = mean(pnl) / std(pnl).
        Normalizes to sum=1 with a minimum floor per model.

        Returns:
            Numpy array of normalized weights (length = _NUM_MODELS)
        """
        sharpe_ratios = np.zeros(_NUM_MODELS)

        for i, name in enumerate(_MODEL_NAMES):
            pnl_data = list(self._pnl_tracker[name])
            if len(pnl_data) < self._sharpe_config.min_trades_per_model:
                # Insufficient data - use neutral weight
                sharpe_ratios[i] = 0.0
            else:
                pnl_arr = np.array(pnl_data)
                mean_pnl = np.mean(pnl_arr)
                std_pnl = np.std(pnl_arr)
                if std_pnl < 1e-10:
                    # No variance - use mean as indicator
                    sharpe_ratios[i] = mean_pnl if mean_pnl > 0 else 0.0
                else:
                    sharpe_ratios[i] = mean_pnl / std_pnl

        # Shift to positive domain (add offset so all values are > 0)
        min_sharpe = np.min(sharpe_ratios)
        if min_sharpe < 0:
            sharpe_ratios = sharpe_ratios - min_sharpe + 0.01

        # Apply minimum floor
        floor = self._sharpe_config.min_weight_floor
        weights = np.maximum(sharpe_ratios, floor)

        # Normalize to sum = 1
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(_NUM_MODELS) / _NUM_MODELS

        # Ensure minimum floor is maintained after normalization
        weights = np.maximum(weights, floor)
        weights = weights / weights.sum()

        return weights

    def sharpe_reweight(self) -> None:
        """
        Increment trade counter and recompute Sharpe-based weights
        every N trades (reweight_interval).

        Keeps existing update_weights() for win-rate as fallback
        when insufficient Sharpe data is available.
        """
        if not self._sharpe_config.enabled:
            return

        self._sharpe_trade_counter += 1

        if self._sharpe_trade_counter % self._sharpe_config.reweight_interval != 0:
            return

        # Check if we have sufficient data for at least some models
        models_with_data = sum(
            1 for name in _MODEL_NAMES
            if len(self._pnl_tracker[name]) >= self._sharpe_config.min_trades_per_model
        )

        if models_with_data < 3:
            # Not enough models with sufficient Sharpe data - keep win-rate weights
            logger.debug(
                f"[Sharpe] Only {models_with_data} models have enough data, "
                f"keeping win-rate weights"
            )
            return

        new_weights = self.compute_sharpe_weights()
        self.weights = new_weights
        logger.info(
            f"[Sharpe] Reweighted at trade #{self._sharpe_trade_counter}. "
            f"Top 3: {', '.join(f'{_MODEL_NAMES[i]}={new_weights[i]:.3f}' for i in np.argsort(new_weights)[-3:][::-1])}"
        )

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
            # Attempt auto-retrain from accumulated data
            if self.config.accumulate_meta_data:
                self.retrain_meta_learner_from_accumulated()
        if nn_loaded:
            self.models_loaded = True
            logger.info(f"[Ensemble] Load complete. "
                        f"gb={self.gb_fitted} xgb={self.xgb_fitted} "
                        f"cb={self.catboost_fitted} meta={self.meta_fitted}")

    # ── Online Learning Adaptation ────────────────────────────────────────

    def online_update(self, x_batch: np.ndarray, y_true: np.ndarray) -> bool:
        """
        Perform lightweight gradient updates on neural model classification heads.

        Only updates the last layer (classification head) while keeping the
        backbone frozen. This bridges the gap between weekly full retrains by
        adapting to recent market conditions.

        Args:
            x_batch: Input features, shape (batch, seq_len, features)
            y_true: True labels, shape (batch,) with values in {0, 1, 2}

        Returns:
            True if update was performed, False if skipped (insufficient data
            or online learning disabled).
        """
        if not self.config.online_learning_enabled:
            return False

        if len(x_batch) < self.config.online_batch_size:
            return False

        # Convert to tensors
        x_tensor = torch.FloatTensor(x_batch).to(self.device)
        y_tensor = torch.LongTensor(y_true).to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        updated_count = 0

        # List of neural models to update (their classification heads only)
        nn_models = [
            self.transformer, self.lstm, self.tcn,
            self.patch_tst, self.tft, self.nhits,
            self.itransformer, self.mamba, self.dlinear,
            self.xlstm, self.timesnet,
            self.chronos, self.timemixer, self.softs,
        ]

        for model in nn_models:
            try:
                model.train()

                # Freeze all parameters first
                for param in model.parameters():
                    param.requires_grad = False

                # Unfreeze only the last linear layer (classification head)
                # Most models have a 'fc' or 'classifier' or 'head' attribute
                head_params = []
                for name, param in model.named_parameters():
                    # Look for the final classification layer
                    if any(key in name for key in ['fc.', 'classifier.', 'head.', 'output_layer.']):
                        param.requires_grad = True
                        head_params.append(param)

                # If no explicit head found, unfreeze last 2 layers
                if not head_params:
                    all_params = list(model.parameters())
                    for param in all_params[-2:]:
                        param.requires_grad = True
                        head_params.append(param)

                if not head_params:
                    continue

                # Single gradient step with small learning rate
                optimizer = optim.Adam(head_params, lr=self.config.online_lr)
                optimizer.zero_grad()

                with torch.enable_grad():
                    output = model.predict(x_tensor)
                    if isinstance(output, np.ndarray):
                        # Model returned numpy, need to re-run forward pass
                        # Skip this model for online update
                        continue
                    # output should be (batch, 3) logits or probabilities
                    if output.requires_grad:
                        loss = loss_fn(output, y_tensor)
                        loss.backward()
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
                        optimizer.step()
                        updated_count += 1

                # Re-freeze everything
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()

            except Exception as e:
                # Silently skip models that fail online update
                # (e.g., models without standard forward pass)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                logger.debug(f"[Online] Skip model update: {e}")
                continue

        if updated_count > 0:
            logger.info(
                f"[Online] Updated {updated_count} model heads "
                f"(batch={len(x_batch)}, lr={self.config.online_lr})"
            )

        return updated_count > 0

    def record_true_label(self, true_label: int) -> None:
        """
        Record the true label for the most recent prediction.

        When accumulate_meta_data is enabled, this saves the stacked predictions
        (from the last predict() call) paired with the true label for future
        meta-learner retraining.

        Args:
            true_label: True class label (0=SELL, 1=HOLD, 2=BUY)
        """
        if not self.config.accumulate_meta_data:
            return

        if self._last_stacked_predictions is None:
            logger.debug("[Ensemble] No stacked predictions to pair with label")
            return

        # Save to disk for future meta-learner training
        self.save_stacked_predictions(self._last_stacked_predictions, true_label)

    def save_stacked_predictions(
        self, stacked_preds: np.ndarray, true_label: int
    ) -> None:
        """
        Append (stacked_51_dim, true_label) to accumulated meta-learner data.

        Data is stored in checkpoints/meta_learner_data.npz as two arrays:
        - 'X': stacked predictions, shape (N, 51)
        - 'y': true labels, shape (N,)

        Args:
            stacked_preds: Stacked model predictions, shape (1, 51) or (51,)
            true_label: True class label (0, 1, or 2)
        """
        if stacked_preds.ndim == 1:
            stacked_preds = stacked_preds.reshape(1, -1)
        elif stacked_preds.ndim == 2:
            stacked_preds = stacked_preds[:1]  # Take first sample only

        label_arr = np.array([true_label])

        try:
            os.makedirs(os.path.dirname(self._meta_data_path), exist_ok=True)

            # Load existing data if available
            if os.path.exists(self._meta_data_path):
                existing = np.load(self._meta_data_path)
                X_existing = existing['X']
                y_existing = existing['y']
                X_new = np.vstack([X_existing, stacked_preds])
                y_new = np.concatenate([y_existing, label_arr])
            else:
                X_new = stacked_preds
                y_new = label_arr

            np.savez(self._meta_data_path, X=X_new, y=y_new)
            logger.debug(
                f"[Ensemble] Saved meta-learner sample "
                f"(total: {len(y_new)} samples)"
            )
        except Exception as e:
            logger.debug(f"[Ensemble] Error saving meta-learner data: {e}")

    def retrain_meta_learner_from_accumulated(self) -> bool:
        """
        Retrain the meta-learner from accumulated stacked prediction data.

        Loads meta_learner_data.npz, fits self.meta_learner on the data,
        and saves to meta_learner.joblib. Called from load_models() when
        .joblib is missing and sufficient data exists, and also callable
        from walk_forward.py during retraining.

        Returns:
            True if retrain was successful, False if insufficient data or error.
        """
        if not os.path.exists(self._meta_data_path):
            logger.info(
                "[Ensemble] No accumulated meta-learner data found at "
                f"{self._meta_data_path}"
            )
            return False

        try:
            data = np.load(self._meta_data_path)
            X = data['X']
            y = data['y']

            if len(y) < self.config.meta_data_min_samples:
                logger.info(
                    f"[Ensemble] Insufficient meta-learner data: "
                    f"{len(y)} samples (need {self.config.meta_data_min_samples})"
                )
                return False

            # Verify shape
            if X.shape[1] != 51:
                logger.warning(
                    f"[Ensemble] Meta-learner data shape mismatch: "
                    f"X.shape={X.shape}, expected (N, 51)"
                )
                return False

            # Verify labels are valid
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                logger.warning(
                    "[Ensemble] Meta-learner data has fewer than 2 classes, "
                    "cannot retrain"
                )
                return False

            logger.info(
                f"[Ensemble] Retraining meta-learner from "
                f"{len(y)} accumulated samples..."
            )

            # Fit the meta-learner
            self.meta_learner.fit(X, y)
            self.meta_fitted = True

            # Save to joblib
            import joblib
            meta_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints", "meta_learner.joblib"
            )
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            joblib.dump(self.meta_learner, meta_path)

            logger.info(
                f"[Ensemble] Meta-learner retrained successfully "
                f"({len(y)} samples, {len(unique_labels)} classes). "
                f"Saved to {meta_path}"
            )
            return True

        except Exception as e:
            logger.warning(f"[Ensemble] Meta-learner retrain failed: {e}")
            return False
