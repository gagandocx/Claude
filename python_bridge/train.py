"""
=============================================================
  Python ML Bridge - Model Training Script  (v3 — 9 models)

  Pipeline:
    1.  Download + prepare multi-timeframe training data
    2.  Train Transformer
    3.  Train LSTM
    4.  Train TCN
    5.  Train PatchTST          (NEW)
    6.  Train TFT               (NEW)
    7.  Train N-HiTS            (NEW)
    8.  Train Gradient Boosting (sklearn)
    9.  Train LightGBM / XGBoost
    10. Train CatBoost          (NEW)
    11. Fit 27-dim meta-learner on stacked val predictions
    12. Evaluate all 9 models + ensemble on held-out test set
    13. Save all checkpoints

  Install optional backends for best performance:
    pip install lightgbm    # step 9
    pip install catboost    # step 10

  Run locally:   python train.py
  Run on Colab:  !python train.py   (GPU auto-detected)
=============================================================
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    TransformerConfig, LSTMConfig, TCNConfig,
    PatchTSTConfig, TFTConfig, NHiTSConfig,
    XGBoostConfig, CatBoostConfig,
    EnsembleConfig, DataConfig, MODEL_DIR,
)
from data.market_data import MarketDataFetcher
from models.transformer_model import MarketTransformer
from models.lstm_model import MarketLSTM
from models.tcn_model import MarketTCN
from models.patch_tst import MarketPatchTST
from models.tft_model import MarketTFT
from models.nhits_model import MarketNHiTS
from models.ensemble import EnsembleManager


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Training")

# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data(
    config: DataConfig = None, seq_length: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Download historical data from multiple timeframes and prepare sequences.

    Fetches 7d/1m + 60d/15m + 2y/1h data → targets 25,000-30,000+ sequences.
    Each timeframe is feature-engineered independently, then all X/y arrays
    are concatenated and normalised with shared stats saved for inference.

    Returns:
        (X, y) — X shape (N, seq_len, features), y shape (N,)
    """
    config  = config or DataConfig()
    fetcher = MarketDataFetcher(config)
    all_X, all_y = [], []

    for tf_spec in config.training_periods:
        period, interval = tf_spec["period"], tf_spec["interval"]
        logger.info(f"Downloading {period} @ {interval}...")
        df = fetcher.fetch_ohlcv(period=period, interval=interval)
        if df.empty:
            logger.warning(f"  No data for {period}/{interval} — skipping")
            continue

        features = fetcher.compute_features(df)
        if features.empty:
            logger.warning(f"  Feature computation failed for {period}/{interval}")
            continue

        X, y = fetcher.prepare_model_input(
            features, seq_length=seq_length, normalize=False
        )
        if len(X) == 0:
            continue

        logger.info(f"  {len(X)} sequences from {period}/{interval}")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        logger.error("No training data — check internet connection or data config.")
        return np.array([]), np.array([])

    X = np.vstack(all_X)
    y = np.hstack(all_y)

    # Normalise once across all timeframes; save stats for live inference
    means = np.mean(X, axis=(0, 1))
    stds  = np.std(X, axis=(0, 1)) + 1e-10
    X     = (X - means) / stds
    fetcher._save_normalization_stats(
        feature_cols=[f"feat_{i}" for i in range(X.shape[2])],
        means=means, stds=stds,
    )

    logger.info(f"Total sequences: {len(X)} | Features: {X.shape[2]}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    return X, y


# ─────────────────────────────────────────────
#  GENERIC NEURAL MODEL TRAINER
# ─────────────────────────────────────────────
def _train_neural_model(
    model_class: Type[nn.Module],
    config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label: str,
) -> nn.Module:
    """
    Shared training loop for ALL 6 neural models.

    All models share the same recipe:
      AdamW + CosineAnnealingLR + class-weighted CrossEntropy
      + gradient clipping (max_norm=1.0) + early stopping.

    Each model class must accept `config` as its constructor argument
    and implement `forward(x) → logits`.

    Args:
        model_class : One of MarketTransformer, MarketLSTM, MarketTCN,
                      MarketPatchTST, MarketTFT, MarketNHiTS
        config      : Matching config dataclass (input_features pre-set)
        X_train/val : (N, seq_len, features) arrays
        y_train/val : (N,) integer labels
        label       : Display name for logging

    Returns:
        Trained model on CPU (device-agnostic checkpoint)
    """
    model = model_class(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Class-weighted loss — handles BUY/HOLD/SELL imbalance
    counts = np.bincount(y_train, minlength=3)
    total  = counts.sum()
    cw = torch.FloatTensor(
        [total / (3 * c) if c > 0 else 1.0 for c in counts]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=cw, label_smoothing=config.label_smoothing
    )

    batch_size   = config.batch_size_gpu if device.type == "cuda" else config.batch_size
    loader_kw    = (
        {"num_workers": 4, "pin_memory": True}
        if device.type == "cuda" else {"num_workers": 0}
    )
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=batch_size, **loader_kw,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Training {label}: {n_params:,} params | device={device} | bs={batch_size}")

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    for epoch in range(config.epochs):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out     = model(bx)
                val_loss += criterion(out, by).item()
                correct  += (out.argmax(1) == by).sum().item()
                total    += by.size(0)
        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  [{label}] Epoch {epoch+1}/{config.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"acc={correct/total:.4f}"
            )

        # ── early stopping ──────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= config.patience:
                logger.info(f"  [{label}] Early stop at epoch {epoch+1}")
                break

    model = model.cpu()
    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
#  PER-MODEL WRAPPERS  (thin — just set input_features)
# ─────────────────────────────────────────────
def train_transformer(X_tr, y_tr, X_val, y_val,
                      config: Optional[TransformerConfig] = None) -> MarketTransformer:
    cfg = config or TransformerConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketTransformer, cfg, X_tr, y_tr, X_val, y_val,
                               "Transformer")


def train_lstm(X_tr, y_tr, X_val, y_val,
               config: Optional[LSTMConfig] = None) -> MarketLSTM:
    cfg = config or LSTMConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketLSTM, cfg, X_tr, y_tr, X_val, y_val, "LSTM")


def train_tcn(X_tr, y_tr, X_val, y_val,
              config: Optional[TCNConfig] = None) -> MarketTCN:
    cfg = config or TCNConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketTCN, cfg, X_tr, y_tr, X_val, y_val, "TCN")


def train_patch_tst(X_tr, y_tr, X_val, y_val,
                    config: Optional[PatchTSTConfig] = None) -> MarketPatchTST:
    """
    PatchTST — divides 64-bar window into 8-bar patches, runs Transformer
    over 8 patch tokens instead of 64 bar tokens. Pre-norm architecture.
    """
    cfg = config or PatchTSTConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketPatchTST, cfg, X_tr, y_tr, X_val, y_val,
                               "PatchTST")


def train_tft(X_tr, y_tr, X_val, y_val,
              config: Optional[TFTConfig] = None) -> MarketTFT:
    """
    Temporal Fusion Transformer — Variable Selection Networks learn which
    of the 46 features matter most at each bar; GRN gates suppress noise.
    """
    cfg = config or TFTConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketTFT, cfg, X_tr, y_tr, X_val, y_val, "TFT")


def train_nhits(X_tr, y_tr, X_val, y_val,
                config: Optional[NHiTSConfig] = None) -> MarketNHiTS:
    """
    N-HiTS — 4 hierarchical blocks (pool=8/4/2/1) decompose the sequence
    from macro trend down to micro price action simultaneously.
    """
    cfg = config or NHiTSConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketNHiTS, cfg, X_tr, y_tr, X_val, y_val, "N-HiTS")


# ─────────────────────────────────────────────
#  FULL TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_all():
    """
    Full 9-model training pipeline. Trains all models sequentially,
    then fits a 27-dim HistGradientBoosting meta-learner on stacked
    validation predictions. Logs per-model and ensemble accuracy.
    """
    logger.info("=" * 70)
    logger.info("  Python ML Bridge — 9-Model Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    t0 = time.time()

    # ── 1. Data ────────────────────────────────────────────────────────────
    X, y = prepare_data()
    if len(X) == 0:
        logger.error("No data — aborting.")
        return

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, shuffle=False
    )
    n_feat = X_train.shape[2]
    logger.info(
        f"Splits — train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
        f"features={n_feat}"
    )

    # ── 2. Neural models ───────────────────────────────────────────────────
    logger.info("\n--- [1/6] Transformer ---")
    transformer = train_transformer(
        X_train, y_train, X_val, y_val,
        TransformerConfig(input_features=n_feat)
    )

    logger.info("\n--- [2/6] LSTM ---")
    lstm = train_lstm(
        X_train, y_train, X_val, y_val,
        LSTMConfig(input_features=n_feat)
    )

    logger.info("\n--- [3/6] TCN ---")
    tcn = train_tcn(
        X_train, y_train, X_val, y_val,
        TCNConfig(input_features=n_feat)
    )

    logger.info("\n--- [4/6] PatchTST ---")
    patch_tst = train_patch_tst(
        X_train, y_train, X_val, y_val,
        PatchTSTConfig(input_features=n_feat)
    )

    logger.info("\n--- [5/6] TFT ---")
    tft = train_tft(
        X_train, y_train, X_val, y_val,
        TFTConfig(input_features=n_feat)
    )

    logger.info("\n--- [6/6] N-HiTS ---")
    nhits = train_nhits(
        X_train, y_train, X_val, y_val,
        NHiTSConfig(input_features=n_feat)
    )

    # ── 3. Wire trained neural models into EnsembleManager ─────────────────
    ensemble = EnsembleManager(
        transformer_config = TransformerConfig(input_features=n_feat),
        lstm_config        = LSTMConfig(input_features=n_feat),
        tcn_config         = TCNConfig(input_features=n_feat),
        patch_tst_config   = PatchTSTConfig(input_features=n_feat),
        tft_config         = TFTConfig(input_features=n_feat),
        nhits_config       = NHiTSConfig(input_features=n_feat),
    )
    ensemble.transformer = transformer
    ensemble.lstm        = lstm
    ensemble.tcn         = tcn
    ensemble.patch_tst   = patch_tst
    ensemble.tft         = tft
    ensemble.nhits       = nhits

    # ── 4. Tree models ─────────────────────────────────────────────────────
    logger.info("\n--- Gradient Boosting (sklearn) ---")
    ensemble.fit_gradient_boost(X_train, y_train)
    logger.info("  HistGradientBoosting fitted")

    logger.info("\n--- LightGBM / XGBoost ---")
    ensemble.fit_xgboost(X_train, y_train)
    logger.info(f"  XGBoost fitted (backend={ensemble.xgboost_model.backend})")

    logger.info("\n--- CatBoost ---")
    ensemble.fit_catboost(X_train, y_train)
    logger.info(f"  CatBoost fitted (backend={ensemble.catboost_model.backend})")

    # ── 5. Build 27-dim stacked predictions on validation set ──────────────
    logger.info("\n--- Fitting Meta-Learner (27-dim stack) ---")
    for m in [transformer, lstm, tcn, patch_tst, tft, nhits]:
        m.eval()

    with torch.no_grad():
        Xt  = torch.FloatTensor(X_val)
        t_v   = transformer.predict(Xt).numpy()
        l_v   = lstm.predict(Xt).numpy()
        c_v   = tcn.predict(Xt).numpy()
        pt_v  = patch_tst.predict(Xt).numpy()
        tf_v  = tft.predict(Xt).numpy()
        nh_v  = nhits.predict(Xt).numpy()

    gb_v  = ensemble.predict_gradient_boost(X_val)
    xb_v  = ensemble.predict_xgboost(X_val)
    cb_v  = ensemble.predict_catboost(X_val)

    stacked_val = np.concatenate(
        [t_v, l_v, c_v, pt_v, tf_v, nh_v, gb_v, xb_v, cb_v], axis=1
    )  # (n_val, 27)

    ensemble.fit_meta_learner(stacked_val, y_val)
    logger.info("  Meta-learner fitted on 27-dim stacked predictions")

    # ── 6. Evaluate on held-out test set ────────────────────────────────────
    logger.info("\n--- Test Set Evaluation (9 models) ---")
    with torch.no_grad():
        Xt_test = torch.FloatTensor(X_test)
        t_te  = transformer.predict(Xt_test).numpy()
        l_te  = lstm.predict(Xt_test).numpy()
        c_te  = tcn.predict(Xt_test).numpy()
        pt_te = patch_tst.predict(Xt_test).numpy()
        tf_te = tft.predict(Xt_test).numpy()
        nh_te = nhits.predict(Xt_test).numpy()

    gb_te  = ensemble.predict_gradient_boost(X_test)
    xb_te  = ensemble.predict_xgboost(X_test)
    cb_te  = ensemble.predict_catboost(X_test)

    def acc(probs): return np.mean(np.argmax(probs, axis=1) == y_test)

    t_acc  = acc(t_te);   l_acc  = acc(l_te);   c_acc  = acc(c_te)
    pt_acc = acc(pt_te);  tf_acc = acc(tf_te);  nh_acc = acc(nh_te)
    gb_acc = acc(gb_te);  xb_acc = acc(xb_te);  cb_acc = acc(cb_te)

    stacked_test = np.concatenate(
        [t_te, l_te, c_te, pt_te, tf_te, nh_te, gb_te, xb_te, cb_te], axis=1
    )
    ens_acc = np.mean(ensemble.meta_learner.predict(stacked_test) == y_test)
    best_ind = max(t_acc, l_acc, c_acc, pt_acc, tf_acc, nh_acc,
                   gb_acc, xb_acc, cb_acc)

    logger.info(f"  Transformer acc:        {t_acc:.4f}")
    logger.info(f"  LSTM acc:               {l_acc:.4f}")
    logger.info(f"  TCN acc:                {c_acc:.4f}")
    logger.info(f"  PatchTST acc:           {pt_acc:.4f}  ← SOTA 2023")
    logger.info(f"  TFT acc:                {tf_acc:.4f}  ← financial-specific")
    logger.info(f"  N-HiTS acc:             {nh_acc:.4f}  ← multi-scale")
    logger.info(f"  GradBoost acc:          {gb_acc:.4f}")
    logger.info(f"  LightGBM/XGBoost acc:   {xb_acc:.4f}")
    logger.info(f"  CatBoost acc:           {cb_acc:.4f}  ← ordered boosting")
    logger.info(f"  ─────────────────────────────────────────")
    logger.info(f"  Best individual:        {best_ind:.4f}")
    logger.info(f"  9-model ensemble:       {ens_acc:.4f}  ← target metric")
    logger.info(f"  Ensemble lift:          {ens_acc - best_ind:+.4f}")

    # ── 7. Save all checkpoints ─────────────────────────────────────────────
    logger.info(f"\n--- Saving Checkpoints → {MODEL_DIR} ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensemble.save_models(MODEL_DIR)

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed/60:.1f} min")
    logger.info("=" * 70)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all()
