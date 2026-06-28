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
    ITransformerConfig, MambaConfig, DLinearConfig,
    xLSTMConfig, TimesNetConfig,
    ChronosConfig, TimeMixerConfig, SOFTSConfig,
    XGBoostConfig, CatBoostConfig,
    EnsembleConfig, DataConfig, MODEL_DIR,
)
from data.market_data import MarketDataFetcher
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
from models.ensemble          import EnsembleManager


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
    """N-HiTS — 4 hierarchical blocks (pool=8/4/2/1), macro→micro decomposition."""
    cfg = config or NHiTSConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketNHiTS, cfg, X_tr, y_tr, X_val, y_val, "N-HiTS")


def train_itransformer(X_tr, y_tr, X_val, y_val,
                       config: Optional[ITransformerConfig] = None) -> MarketITransformer:
    """
    iTransformer — inverts the input so Transformer attention runs over
    features (RSI, MACD, ATR…) rather than time steps.
    Captures cross-indicator correlations no other model in the stack models.
    """
    cfg = config or ITransformerConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketITransformer, cfg, X_tr, y_tr, X_val, y_val,
                               "iTransformer")


def train_mamba(X_tr, y_tr, X_val, y_val,
                config: Optional[MambaConfig] = None) -> MarketMamba:
    """
    Mamba S6 — pure PyTorch selective state space model.
    Input-dependent Δ/B/C parameters let the model learn what price
    history to remember and what to discard at each bar.
    """
    cfg = config or MambaConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketMamba, cfg, X_tr, y_tr, X_val, y_val, "Mamba")


def train_dlinear(X_tr, y_tr, X_val, y_val,
                  config: Optional[DLinearConfig] = None) -> MarketDLinear:
    """
    DLinear — decomposes sequence into trend + residual components,
    applies channel-independent linear projection to each.
    Adds ensemble diversity: captures clean linear signals others miss.
    """
    cfg = config or DLinearConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketDLinear, cfg, X_tr, y_tr, X_val, y_val, "DLinear")


def train_xlstm(X_tr, y_tr, X_val, y_val,
                config: Optional[xLSTMConfig] = None) -> MarketXLSTM:
    """
    xLSTM mLSTM — matrix memory cells (d×d per head) with exponential
    gates stabilised in log-space. The LSTM inventor's 2024 rewrite:
    no sigmoid saturation, no vanishing gradients, far higher capacity
    than standard BiLSTM.
    """
    cfg = config or xLSTMConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketXLSTM, cfg, X_tr, y_tr, X_val, y_val, "xLSTM")


def train_timesnet(X_tr, y_tr, X_val, y_val,
                   config: Optional[TimesNetConfig] = None) -> MarketTimesNet:
    """
    TimesNet — discovers dominant periods via FFT, reshapes each period
    into a 2D image, applies Inception-style 2D convolutions (horizontal
    = within-cycle, vertical = same-phase cross-cycle, diagonal = mixed).
    The only 2D model in the stack.
    """
    cfg = config or TimesNetConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketTimesNet, cfg, X_tr, y_tr, X_val, y_val,
                               "TimesNet")


def train_chronos(X_tr, y_tr, X_val, y_val,
                  config: Optional[ChronosConfig] = None) -> MarketChronos:
    """
    Chronos — Amazon pre-trained T5 foundation model.
    The T5 encoder is FROZEN; only the classification head trains.
    Install: pip install git+https://github.com/amazon-science/chronos-forecasting.git
    Falls back to lightweight learned encoder if not installed.
    """
    cfg = config or ChronosConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketChronos, cfg, X_tr, y_tr, X_val, y_val, "Chronos")


def train_timemixer(X_tr, y_tr, X_val, y_val,
                    config: Optional[TimeMixerConfig] = None) -> MarketTimeMixer:
    """
    TimeMixer — decomposes input at 4 scales (pool=1,2,4,8) and mixes
    trend + seasonal components bottom-up between scales.
    """
    cfg = config or TimeMixerConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketTimeMixer, cfg, X_tr, y_tr, X_val, y_val,
                               "TimeMixer")


def train_softs(X_tr, y_tr, X_val, y_val,
                config: Optional[SOFTSConfig] = None) -> MarketSOFTS:
    """
    SOFTS — Star aggregate: O(N) cross-series interaction via one central
    'market state' node. More efficient than iTransformer, captures
    complementary global context.
    """
    cfg = config or SOFTSConfig()
    cfg.input_features = X_tr.shape[2]
    return _train_neural_model(MarketSOFTS, cfg, X_tr, y_tr, X_val, y_val, "SOFTS")


# ─────────────────────────────────────────────
#  FULL TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_all():
    """
    Full 14-model training pipeline. Trains all models sequentially,
    fits a 42-dim HistGradientBoosting meta-learner on stacked val
    predictions, then evaluates and saves all checkpoints.
    """
    logger.info("=" * 70)
    logger.info("  Python ML Bridge — 17-Model Training Pipeline")
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
        f"Splits — train={len(X_train):,} val={len(X_val):,} "
        f"test={len(X_test):,} | features={n_feat}"
    )

    # ── 2. Train all 11 neural models ──────────────────────────────────────
    logger.info("\n--- [ 1/11 neural] Transformer ---")
    transformer = train_transformer(X_train, y_train, X_val, y_val,
                                    TransformerConfig(input_features=n_feat))

    logger.info("\n--- [ 2/11 neural] LSTM ---")
    lstm = train_lstm(X_train, y_train, X_val, y_val,
                      LSTMConfig(input_features=n_feat))

    logger.info("\n--- [ 3/11 neural] TCN ---")
    tcn = train_tcn(X_train, y_train, X_val, y_val,
                    TCNConfig(input_features=n_feat))

    logger.info("\n--- [ 4/11 neural] PatchTST ---")
    patch_tst = train_patch_tst(X_train, y_train, X_val, y_val,
                                 PatchTSTConfig(input_features=n_feat))

    logger.info("\n--- [ 5/11 neural] TFT ---")
    tft = train_tft(X_train, y_train, X_val, y_val,
                    TFTConfig(input_features=n_feat))

    logger.info("\n--- [ 6/11 neural] N-HiTS ---")
    nhits = train_nhits(X_train, y_train, X_val, y_val,
                        NHiTSConfig(input_features=n_feat))

    logger.info("\n--- [ 7/11 neural] iTransformer ---")
    itransformer = train_itransformer(X_train, y_train, X_val, y_val,
                                      ITransformerConfig(input_features=n_feat))

    logger.info("\n--- [ 8/11 neural] Mamba ---")
    mamba = train_mamba(X_train, y_train, X_val, y_val,
                        MambaConfig(input_features=n_feat))

    logger.info("\n--- [ 9/11 neural] DLinear ---")
    dlinear = train_dlinear(X_train, y_train, X_val, y_val,
                            DLinearConfig(input_features=n_feat))

    logger.info("\n--- [10/14 neural] xLSTM ---")
    xlstm = train_xlstm(X_train, y_train, X_val, y_val,
                        xLSTMConfig(input_features=n_feat))

    logger.info("\n--- [11/14 neural] TimesNet ---")
    timesnet = train_timesnet(X_train, y_train, X_val, y_val,
                              TimesNetConfig(input_features=n_feat))

    logger.info("\n--- [12/14 neural] Chronos (NEW — pre-trained) ---")
    chronos = train_chronos(X_train, y_train, X_val, y_val,
                            ChronosConfig(input_features=n_feat))

    logger.info("\n--- [13/14 neural] TimeMixer (NEW) ---")
    timemixer = train_timemixer(X_train, y_train, X_val, y_val,
                                TimeMixerConfig(input_features=n_feat))

    logger.info("\n--- [14/14 neural] SOFTS (NEW) ---")
    softs = train_softs(X_train, y_train, X_val, y_val,
                        SOFTSConfig(input_features=n_feat))

    # ── 3. Wire into EnsembleManager ──────────────────────────────────────
    ensemble = EnsembleManager(
        transformer_config  = TransformerConfig(input_features=n_feat),
        lstm_config         = LSTMConfig(input_features=n_feat),
        tcn_config          = TCNConfig(input_features=n_feat),
        patch_tst_config    = PatchTSTConfig(input_features=n_feat),
        tft_config          = TFTConfig(input_features=n_feat),
        nhits_config        = NHiTSConfig(input_features=n_feat),
        itransformer_config = ITransformerConfig(input_features=n_feat),
        mamba_config        = MambaConfig(input_features=n_feat),
        dlinear_config      = DLinearConfig(input_features=n_feat),
        xlstm_config        = xLSTMConfig(input_features=n_feat),
        timesnet_config     = TimesNetConfig(input_features=n_feat),
        chronos_config      = ChronosConfig(input_features=n_feat),
        timemixer_config    = TimeMixerConfig(input_features=n_feat),
        softs_config        = SOFTSConfig(input_features=n_feat),
    )
    neural_models_map = {
        "transformer": transformer, "lstm": lstm, "tcn": tcn,
        "patch_tst": patch_tst, "tft": tft, "nhits": nhits,
        "itransformer": itransformer, "mamba": mamba, "dlinear": dlinear,
        "xlstm": xlstm, "timesnet": timesnet,
        "chronos": chronos, "timemixer": timemixer, "softs": softs,
    }
    for attr, model in neural_models_map.items():
        setattr(ensemble, attr, model)

    # ── 4. Tree models ─────────────────────────────────────────────────────
    logger.info("\n--- Gradient Boosting (sklearn) ---")
    ensemble.fit_gradient_boost(X_train, y_train)

    logger.info("\n--- LightGBM / XGBoost ---")
    ensemble.fit_xgboost(X_train, y_train)
    logger.info(f"  backend={ensemble.xgboost_model.backend}")

    logger.info("\n--- CatBoost ---")
    ensemble.fit_catboost(X_train, y_train)
    logger.info(f"  backend={ensemble.catboost_model.backend}")

    # ── 5. 51-dim meta-learner ─────────────────────────────────────────────
    logger.info("\n--- Fitting Meta-Learner (51-dim stack, 17 × 3) ---")
    neural_models = [
        transformer, lstm, tcn, patch_tst, tft, nhits,
        itransformer, mamba, dlinear, xlstm, timesnet,
        chronos, timemixer, softs,
    ]
    for m in neural_models:
        m.eval()

    with torch.no_grad():
        Xv = torch.FloatTensor(X_val)
        nn_val = [m.predict(Xv).numpy() for m in neural_models]

    tree_val = [
        ensemble.predict_gradient_boost(X_val),
        ensemble.predict_xgboost(X_val),
        ensemble.predict_catboost(X_val),
    ]
    stacked_val = np.concatenate(nn_val + tree_val, axis=1)  # (n_val, 36)
    ensemble.fit_meta_learner(stacked_val, y_val)
    logger.info("  Meta-learner fitted on 36-dim stacked predictions")

    # ── 6. Evaluate on held-out test set ────────────────────────────────────
    logger.info("\n--- Test Set Evaluation (12 models) ---")
    with torch.no_grad():
        Xt = torch.FloatTensor(X_test)
        nn_test = [m.predict(Xt).numpy() for m in neural_models]

    tree_test = [
        ensemble.predict_gradient_boost(X_test),
        ensemble.predict_xgboost(X_test),
        ensemble.predict_catboost(X_test),
    ]
    all_test   = nn_test + tree_test
    model_lbls = ["Transformer", "LSTM", "TCN", "PatchTST", "TFT", "N-HiTS",
                  "iTransformer", "Mamba", "DLinear", "xLSTM", "TimesNet",
                  "Chronos*", "TimeMixer*", "SOFTS*",
                  "GradBoost", "LightGBM", "CatBoost"]

    def acc(p): return np.mean(np.argmax(p, axis=1) == y_test)
    accs     = [acc(p) for p in all_test]
    best_ind = max(accs)

    stacked_test = np.concatenate(all_test, axis=1)
    ens_acc      = np.mean(ensemble.meta_learner.predict(stacked_test) == y_test)

    logger.info("")
    for lbl, a in zip(model_lbls, accs):
        flag = " ← NEW" if "*" in lbl else ""
        logger.info(f"  {lbl.replace('*',''):<18} {a:.4f}{flag}")
    logger.info(f"  {'─' * 38}")
    logger.info(f"  Best individual:    {best_ind:.4f}")
    logger.info(f"  17-model ensemble:  {ens_acc:.4f}  ← target")
    logger.info(f"  Ensemble lift:      {ens_acc - best_ind:+.4f}")

    # ── 7. Save checkpoints ─────────────────────────────────────────────────
    logger.info(f"\n--- Saving → {MODEL_DIR} ---")
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
