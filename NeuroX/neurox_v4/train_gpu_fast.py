"""
=============================================================
  NeuroX v4 - FAST GPU Training Script (Google Colab T4)
  ============================================================
  Trains ALL 17 models in 5-15 minutes on a free Colab T4 GPU.
  Downloads data from yfinance - NO CSV upload needed!

  QUICKSTART (Google Colab):
  ============================================================
  Open a new Colab notebook, set Runtime -> GPU (T4), then:

  # Cell 1 - Clone and install (run once, ~2 min)
  !git clone -b fast-gpu-training --single-branch https://github.com/gagandocx/Claude.git /content/Claude
  %cd /content/Claude/NeuroX/neurox_v4
  !pip install -q torch scikit-learn==1.9.0 lightgbm catboost \
      xgboost yfinance ta pandas numpy tqdm joblib hmmlearn scipy

  # Cell 2 - Train all 17 models on GPU (~5-10 min on T4)
  !python train_gpu_fast.py

  # Cell 3 - Download checkpoints to your PC
  from google.colab import files
  import shutil
  shutil.make_archive('/content/checkpoints', 'zip', '/content/Claude/NeuroX/neurox_v4/checkpoints')
  files.download('/content/checkpoints.zip')

  OR mount Google Drive:
  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r /content/Claude/NeuroX/neurox_v4/checkpoints /content/drive/MyDrive/neurox_checkpoints

  Then on your trading PC, copy the checkpoints/ folder into:
    neurox_v4/checkpoints/
  ============================================================

  WHY THIS IS FAST:
  - T4 GPU: 8.1 TFLOPS FP32 vs ~0.1 TFLOPS on CPU = 80x speedup
  - batch_size_gpu=256 (vs 32 on CPU) = 8x fewer gradient steps
  - GPU parallelism: matrix multiplications run in parallel
  - Combined: each model trains in 30-90 seconds instead of 30-120 min

  EXPECTED TIMES ON T4 (approximate):
  - Transformer (3.2M params):  ~90 sec (was 3.5 hours on CPU)
  - LSTM (1.1M params):         ~45 sec (was 1.2 hours)
  - TCN (156K params):          ~20 sec (was 26 min)
  - PatchTST (459K params):     ~25 sec (was 9 min)
  - TFT (320K params):          ~30 sec (was 31 min)
  - N-HiTS (2.2M params):       ~15 sec (was 6 min)
  - iTransformer (419K params): ~25 sec (was 24 min)
  - Mamba (486K params):        ~30 sec (was 25 min)
  - DLinear (small):            ~10 sec (was 5 min)
  - xLSTM (large):             ~35 sec (was 30 min)
  - TimesNet:                   ~20 sec (was 15 min)
  - Chronos:                    ~15 sec
  - TimeMixer:                  ~20 sec
  - SOFTS:                      ~15 sec
  - GradBoost/LightGBM/CatBoost: ~30 sec total (CPU, tree models)
  - Meta-learner:               ~5 sec
  TOTAL:                        ~7-12 min (was 12+ hours!)
=============================================================
"""

import os
import sys
import time
import logging
import warnings

warnings.filterwarnings("ignore")

# Ensure we import from the correct location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FastGPU")

# ─────────────────────────────────────────────
#  DEVICE DETECTION
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_banner():
    """Print startup banner with device info."""
    logger.info("=" * 65)
    logger.info("  NeuroX v4 - FAST GPU Training")
    logger.info("  All 17 models in minutes, not hours!")
    logger.info("=" * 65)
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  GPU: {gpu_name}")
        logger.info(f"  VRAM: {vram:.1f} GB")
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info("")
        logger.info("  GPU detected! Training will be FAST (5-15 min)")
    else:
        logger.info("")
        logger.info("  WARNING: No GPU detected!")
        logger.info("  Go to Runtime -> Change runtime type -> T4 GPU")
        logger.info("  Then restart and re-run this script.")
        logger.info("")
        logger.info("  Continuing on CPU (will be slow)...")
    logger.info("=" * 65)


# ─────────────────────────────────────────────
#  DATA PREPARATION (yfinance - no CSV needed)
# ─────────────────────────────────────────────
def prepare_data(seq_length: int = 64):
    """
    Download training data from yfinance across multiple timeframes.
    No CSV upload needed - works directly from internet data.

    Fetches:
      - 7 days @ 1-minute bars (~7,997 sequences)
      - 60 days @ 15-minute bars (~4,282 sequences)
      - 2 years @ 1-hour bars (~11,161 sequences)
    Total: ~23,440 sequences with 46 features each.
    """
    from config.settings import DataConfig
    from data.market_data import MarketDataFetcher

    config = DataConfig()
    fetcher = MarketDataFetcher(config)
    all_X, all_y = [], []

    training_periods = [
        {"period": "7d", "interval": "1m"},
        {"period": "60d", "interval": "15m"},
        {"period": "2y", "interval": "1h"},
    ]

    for tf_spec in training_periods:
        period, interval = tf_spec["period"], tf_spec["interval"]
        logger.info(f"  Downloading {period} @ {interval}...")

        df = fetcher.fetch_ohlcv(period=period, interval=interval)
        if df.empty:
            logger.warning(f"    No data for {period}/{interval} - skipping")
            continue

        features = fetcher.compute_features(df)
        if features.empty:
            logger.warning(f"    Feature computation failed for {period}/{interval}")
            continue

        X, y = fetcher.prepare_model_input(
            features, seq_length=seq_length, normalize=False
        )
        if len(X) == 0:
            continue

        logger.info(f"    {len(X):,} sequences from {period}/{interval}")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        logger.error("No training data! Check internet connection.")
        return np.array([]), np.array([])

    X = np.vstack(all_X)
    y = np.hstack(all_y)

    # Normalize across all timeframes
    means = np.mean(X, axis=(0, 1))
    stds = np.std(X, axis=(0, 1)) + 1e-10
    X = (X - means) / stds

    # Save normalization stats for live inference
    fetcher._save_normalization_stats(
        feature_cols=[f"feat_{i}" for i in range(X.shape[2])],
        means=means, stds=stds,
    )

    logger.info(f"  Total: {len(X):,} sequences | {X.shape[2]} features | 3 classes")
    logger.info(f"  Class distribution: {np.bincount(y)}")
    return X, y


# ─────────────────────────────────────────────
#  GPU-OPTIMIZED TRAINING LOOP
# ─────────────────────────────────────────────
def train_model_fast(model_class, config, X_train, y_train, X_val, y_val, label):
    """
    Fast GPU training loop with progress bars and early stopping.
    Uses batch_size_gpu (256) for maximum GPU throughput.
    """
    t_start = time.time()
    model = model_class(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Class-weighted loss
    counts = np.bincount(y_train, minlength=3)
    total = counts.sum()
    cw = torch.FloatTensor(
        [total / (3 * c) if c > 0 else 1.0 for c in counts]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=cw, label_smoothing=config.label_smoothing
    )

    # Use GPU batch size when on GPU
    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size
    loader_kw = (
        {"num_workers": 2, "pin_memory": True, "persistent_workers": True}
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
    logger.info(f"    {n_params:,} params | bs={batch_size} | {device}")

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    # Progress bar
    epoch_range = range(config.epochs)
    if HAS_TQDM:
        epoch_range = tqdm(
            epoch_range, desc=f"    {label}",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True
        )

    for epoch in epoch_range:
        # Train
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss, correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                out = model(bx)
                val_loss += criterion(out, by).item()
                correct += (out.argmax(1) == by).sum().item()
                total_samples += by.size(0)
        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total_samples, 1)
        scheduler.step()

        # Update progress bar
        if HAS_TQDM:
            epoch_range.set_postfix(loss=f"{val_loss:.4f}", acc=f"{val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= config.patience:
                if HAS_TQDM:
                    epoch_range.set_description(f"    {label} [early stop ep {epoch+1}]")
                break

    model = model.cpu()
    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t_start
    logger.info(f"    Done in {elapsed:.0f}s | val_acc={val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def train_all_fast():
    """
    Full 17-model training pipeline optimized for GPU speed.
    Downloads data from yfinance, trains all models, saves checkpoints.
    """
    print_banner()
    t0 = time.time()

    # ── 1. Download and prepare data ──────────────────────────────────────
    logger.info("\n[Step 1/5] Downloading training data from yfinance...")
    X, y = prepare_data()
    if len(X) == 0:
        logger.error("No data available. Check your internet connection.")
        return

    # Split: 70% train, 15% val, 15% test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, shuffle=False
    )
    n_feat = X_train.shape[2]
    logger.info(
        f"  Splits: train={len(X_train):,} | val={len(X_val):,} | "
        f"test={len(X_test):,} | features={n_feat}"
    )

    # ── 2. Import all model classes and configs ───────────────────────────
    from config.settings import (
        TransformerConfig, LSTMConfig, TCNConfig,
        PatchTSTConfig, TFTConfig, NHiTSConfig,
        ITransformerConfig, MambaConfig, DLinearConfig,
        xLSTMConfig, TimesNetConfig,
        ChronosConfig, TimeMixerConfig, SOFTSConfig,
    )
    from models.transformer_model import MarketTransformer
    from models.lstm_model import MarketLSTM
    from models.tcn_model import MarketTCN
    from models.patch_tst import MarketPatchTST
    from models.tft_model import MarketTFT
    from models.nhits_model import MarketNHiTS
    from models.itransformer import MarketITransformer
    from models.mamba_model import MarketMamba
    from models.dlinear_model import MarketDLinear
    from models.xlstm_model import MarketXLSTM
    from models.timesnet_model import MarketTimesNet
    from models.chronos_model import MarketChronos
    from models.timemixer_model import MarketTimeMixer
    from models.softs_model import MarketSOFTS
    from models.ensemble import EnsembleManager

    # Model training order (largest first for GPU warmup)
    NEURAL_MODELS = [
        ("Transformer",  MarketTransformer,  TransformerConfig(input_features=n_feat)),
        ("LSTM",         MarketLSTM,         LSTMConfig(input_features=n_feat)),
        ("TCN",          MarketTCN,          TCNConfig(input_features=n_feat)),
        ("PatchTST",     MarketPatchTST,     PatchTSTConfig(input_features=n_feat)),
        ("TFT",          MarketTFT,          TFTConfig(input_features=n_feat)),
        ("N-HiTS",       MarketNHiTS,        NHiTSConfig(input_features=n_feat)),
        ("iTransformer", MarketITransformer, ITransformerConfig(input_features=n_feat)),
        ("Mamba",        MarketMamba,        MambaConfig(input_features=n_feat)),
        ("DLinear",      MarketDLinear,      DLinearConfig(input_features=n_feat)),
        ("xLSTM",        MarketXLSTM,        xLSTMConfig(input_features=n_feat)),
        ("TimesNet",     MarketTimesNet,     TimesNetConfig(input_features=n_feat)),
        ("Chronos",      MarketChronos,      ChronosConfig(input_features=n_feat)),
        ("TimeMixer",    MarketTimeMixer,    TimeMixerConfig(input_features=n_feat)),
        ("SOFTS",        MarketSOFTS,        SOFTSConfig(input_features=n_feat)),
    ]

    # ── 3. Train all 14 neural models ─────────────────────────────────────
    logger.info(f"\n[Step 2/5] Training 14 neural models on {device}...")
    if device.type == "cuda":
        logger.info(f"  Using GPU batch size: 256 (8x faster than CPU batch=32)")
    logger.info("")

    trained = {}
    for idx, (label, cls, cfg) in enumerate(NEURAL_MODELS, 1):
        logger.info(f"  [{idx:2d}/14] {label}")
        trained[label] = train_model_fast(
            cls, cfg, X_train, y_train, X_val, y_val, label
        )
        # Free GPU memory between models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 4. Train tree models + meta-learner ───────────────────────────────
    logger.info(f"\n[Step 3/5] Training tree models (GradBoost, LightGBM, CatBoost)...")

    ensemble = EnsembleManager(
        transformer_config=TransformerConfig(input_features=n_feat),
        lstm_config=LSTMConfig(input_features=n_feat),
        tcn_config=TCNConfig(input_features=n_feat),
        patch_tst_config=PatchTSTConfig(input_features=n_feat),
        tft_config=TFTConfig(input_features=n_feat),
        nhits_config=NHiTSConfig(input_features=n_feat),
        itransformer_config=ITransformerConfig(input_features=n_feat),
        mamba_config=MambaConfig(input_features=n_feat),
        dlinear_config=DLinearConfig(input_features=n_feat),
        xlstm_config=xLSTMConfig(input_features=n_feat),
        timesnet_config=TimesNetConfig(input_features=n_feat),
        chronos_config=ChronosConfig(input_features=n_feat),
        timemixer_config=TimeMixerConfig(input_features=n_feat),
        softs_config=SOFTSConfig(input_features=n_feat),
    )

    # Wire neural models into ensemble
    attr_map = {
        "Transformer": "transformer", "LSTM": "lstm", "TCN": "tcn",
        "PatchTST": "patch_tst", "TFT": "tft", "N-HiTS": "nhits",
        "iTransformer": "itransformer", "Mamba": "mamba", "DLinear": "dlinear",
        "xLSTM": "xlstm", "TimesNet": "timesnet", "Chronos": "chronos",
        "TimeMixer": "timemixer", "SOFTS": "softs",
    }
    for label, model in trained.items():
        attr = attr_map.get(label)
        if attr and hasattr(ensemble, attr):
            setattr(ensemble, attr, model)

    # Train tree models
    logger.info("  Training Gradient Boosting (sklearn)...")
    ensemble.fit_gradient_boost(X_train, y_train)

    logger.info("  Training LightGBM / XGBoost...")
    ensemble.fit_xgboost(X_train, y_train)
    logger.info(f"    backend: {ensemble.xgboost_model.backend}")

    logger.info("  Training CatBoost...")
    ensemble.fit_catboost(X_train, y_train)
    logger.info(f"    backend: {ensemble.catboost_model.backend}")

    # ── 5. Fit meta-learner ───────────────────────────────────────────────
    logger.info(f"\n[Step 4/5] Fitting meta-learner (51-dim, 17x3 stacked)...")

    neural_models = [trained[n] for n, _, _ in NEURAL_MODELS]
    for m in neural_models:
        m.eval()

    with torch.no_grad():
        Xv = torch.FloatTensor(X_val)
        val_preds = [m.predict(Xv).numpy() for m in neural_models]

    val_preds += [
        ensemble.predict_gradient_boost(X_val),
        ensemble.predict_xgboost(X_val),
        ensemble.predict_catboost(X_val),
    ]
    stacked_val = np.concatenate(val_preds, axis=1)
    ensemble.fit_meta_learner(stacked_val, y_val)
    logger.info("  Meta-learner fitted!")

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    logger.info(f"\n[Step 5/5] Evaluating all 17 models on test set...")

    with torch.no_grad():
        Xt = torch.FloatTensor(X_test)
        test_preds = [m.predict(Xt).numpy() for m in neural_models]

    test_preds += [
        ensemble.predict_gradient_boost(X_test),
        ensemble.predict_xgboost(X_test),
        ensemble.predict_catboost(X_test),
    ]

    all_names = [n for n, _, _ in NEURAL_MODELS] + ["GradBoost", "LightGBM", "CatBoost"]
    accs = [np.mean(np.argmax(p, axis=1) == y_test) for p in test_preds]
    best_ind = max(accs)

    stacked_test = np.concatenate(test_preds, axis=1)
    ens_acc = np.mean(ensemble.meta_learner.predict(stacked_test) == y_test)
    random_bl = max(np.bincount(y_test)) / len(y_test)

    logger.info("")
    logger.info("  " + "-" * 50)
    logger.info(f"  {'Model':<18} {'Accuracy':>10}")
    logger.info("  " + "-" * 50)
    for name, acc in zip(all_names, accs):
        bar = "#" * int(acc * 30)
        logger.info(f"  {name:<18} {acc:.4f}   {bar}")
    logger.info("  " + "-" * 50)
    logger.info(f"  {'Random baseline':<18} {random_bl:.4f}")
    logger.info(f"  {'Best individual':<18} {best_ind:.4f}  (+{best_ind-random_bl:.4f} vs random)")
    logger.info(f"  {'17-MODEL ENSEMBLE':<18} {ens_acc:.4f}  <-- TARGET")
    logger.info(f"  {'Ensemble lift':<18} {ens_acc-best_ind:+.4f} vs best individual")
    logger.info("  " + "-" * 50)

    # ── 7. Save all checkpoints ───────────────────────────────────────────
    from config.settings import MODEL_DIR
    logger.info(f"\n  Saving checkpoints to: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensemble.save_models(MODEL_DIR)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 65)
    logger.info(f"  DONE! Total training time: {elapsed/60:.1f} minutes")
    logger.info("=" * 65)
    logger.info("")
    logger.info("  NEXT STEPS:")
    logger.info(f"  1. Download checkpoints/ folder from: {MODEL_DIR}")
    logger.info("  2. Copy to your trading PC: neurox_v4/checkpoints/")
    logger.info("  3. Restart your EA - it will use the new models!")
    logger.info("")
    if elapsed < 900:  # < 15 min
        speedup = (12 * 60) / (elapsed / 60)  # vs 12 hours on CPU
        logger.info(f"  Speed: {speedup:.0f}x faster than CPU training!")
    logger.info("=" * 65)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all_fast()
