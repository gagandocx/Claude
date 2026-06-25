"""
=============================================================
  Python ML Bridge — Google Colab Training Script
  Trains all 12 models on YOUR real Fusion Markets XAUUSD data.
  Saves checkpoints to Google Drive for use on your trading PC.

  HOW TO USE (Google Colab):
  ─────────────────────────────────────────────────────────
  1. Open Google Colab:  https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Upload this file or paste into a cell with %%writefile
  4. Run the setup cell below, then:  !python train_colab.py
  5. Checkpoints saved to: /content/drive/MyDrive/claude_ea/checkpoints/
  6. Copy the checkpoints/ folder to your PC at:
       C:\\Users\\gagan\\AppData\\Roaming\\MetaQuotes\\Terminal\\...
       ...\\Common\\Files\\  (or wherever your python_bridge/ folder is)

  CSV FORMAT (Fusion Markets / MetaTrader 5 export):
  ─────────────────────────────────────────────────────────
  The script auto-detects these MT5 export formats:
    Format A (tab-separated):  Date  Open  High  Low  Close  Volume
    Format B (comma + time):   2026.03.13,00:01,3961.12,3962.00,...
    Format C (space-separated headers): <DATE> <TIME> <OPEN> ...
  Download from MT5: History → XAUUSD M1 → Right-click → Save As CSV

  COLAB SETUP CELL (run this first):
  ─────────────────────────────────────────────────────────
  !pip install scikit-learn==1.9.0 lightgbm catboost torch torchvision
  !git clone https://github.com/gagandocx/Claude.git
  %cd Claude/python_bridge
  from google.colab import drive
  drive.mount('/content/drive')
=============================================================
"""

import os, sys, time, logging, warnings
warnings.filterwarnings("ignore")

# ── must run from python_bridge/ directory ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Colab")


# ─────────────────────────────────────────────
#  CONFIGURATION — edit these paths
# ─────────────────────────────────────────────
# Path to your Fusion Markets MT5 CSV export (bars_export_XAUUSD.csv)
# Option A: upload to Colab session  →  "/content/bars_export_XAUUSD.csv"
# Option B: put in Google Drive      →  "/content/drive/MyDrive/bars_export_XAUUSD.csv"
# Option C: download from your repo  →  auto-downloaded below if USE_GITHUB=True
CSV_PATH       = "/content/bars_export_XAUUSD.csv"
USE_GITHUB_CSV = True     # True = auto-download from gagandocx/Uploads repo
GITHUB_CSV_URL = "https://raw.githubusercontent.com/gagandocx/Uploads/main/bars_export_XAUUSD.csv"

# Where to save trained checkpoints (Google Drive recommended)
CHECKPOINT_DIR = "/content/drive/MyDrive/claude_ea/checkpoints"
SEQ_LENGTH     = 64       # Sequence length — must match model configs
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  CSV LOADER — handles all Fusion Markets / MT5 export formats
# ─────────────────────────────────────────────
def load_fusion_markets_csv(path: str) -> pd.DataFrame:
    """
    Load Fusion Markets / MetaTrader 5 XAUUSD M1 CSV export.

    Auto-detects format:
      A: tab-separated with headers (Date, Open, High, Low, Close, Volume)
      B: comma-separated with date+time columns (2026.03.13,00:01,O,H,L,C,V)
      C: angle-bracket headers (<DATE>,<TIME>,<OPEN>,...) from MT5 History Export

    Returns DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex.
    """
    log.info(f"Loading CSV: {path}")
    raw = open(path, "r").read(2000)       # peek at first 2KB

    # ── detect separator ────────────────────────────────────────────────────
    sep = "\t" if "\t" in raw[:200] else ","

    # ── detect header style ──────────────────────────────────────────────────
    first_line = raw.split("\n")[0].lower()
    has_angle  = "<date>" in first_line or "<open>" in first_line
    has_time   = "<time>" in first_line or ",time," in first_line or "\ttime\t" in first_line

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip("<>").strip().lower() for c in df.columns]

    # ── build datetime index ─────────────────────────────────────────────────
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M", errors="coerce",
        )
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Cannot find date column in CSV. Check file format.")

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # ── rename to standard OHLCV ─────────────────────────────────────────────
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "o"):         rename[col] = "open"
        elif lc in ("high", "h"):       rename[col] = "high"
        elif lc in ("low", "l"):        rename[col] = "low"
        elif lc in ("close", "c"):      rename[col] = "close"
        elif lc in ("tickvol", "vol", "volume", "v"): rename[col] = "volume"
    df = df.rename(columns=rename)

    required = ["open", "high", "low", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}. Columns found: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 1.0               # synthetic volume if not present

    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.dropna()

    log.info(f"  Loaded {len(df):,} M1 bars  |  "
             f"{df.index[0].date()} → {df.index[-1].date()}  |  "
             f"Price range: ${df['close'].min():.0f}–${df['close'].max():.0f}")
    return df



# ─────────────────────────────────────────────
#  DATA PREPARATION FROM CSV
# ─────────────────────────────────────────────
def prepare_data_from_csv(
    csv_path: str, seq_length: int = SEQ_LENGTH
):
    """
    Load Fusion Markets CSV → feature engineering → training sequences.

    Also synthesises M5 and M15 bars by resampling the M1 data,
    computing HTF features, and merging them back — giving the models
    the same multi-timeframe view they get during live trading.

    Returns: X (N, seq_length, n_features), y (N,) integer labels
    """
    from data.market_data import MarketDataFetcher
    from config.settings import DataConfig

    cfg     = DataConfig()
    fetcher = MarketDataFetcher(cfg)

    # ── 1. Load M1 bars ──────────────────────────────────────────────────────
    m1 = load_fusion_markets_csv(csv_path)

    # ── 2. Compute M1 features ───────────────────────────────────────────────
    log.info("Computing M1 features...")
    feats_m1 = fetcher.compute_features(m1)

    # ── 3. Resample to M5 / M15 and compute HTF features ────────────────────
    def resample_ohlcv(df, rule):
        r = df.resample(rule).agg({
            "open":   "first", "high": "max",
            "low":    "min",   "close": "last", "volume": "sum",
        }).dropna()
        return r

    log.info("Computing M5 HTF features...")
    m5  = resample_ohlcv(m1, "5min")
    feats_m5 = fetcher.compute_features(m5).add_prefix("m5_")

    log.info("Computing M15 HTF features...")
    m15 = resample_ohlcv(m1, "15min")
    feats_m15 = fetcher.compute_features(m15).add_prefix("m15_")

    # ── 4. Forward-fill HTF features to M1 index ────────────────────────────
    feats_m5_ff  = feats_m5.reindex(feats_m1.index, method="ffill")
    feats_m15_ff = feats_m15.reindex(feats_m1.index, method="ffill")

    # ── 5. Merge all features ────────────────────────────────────────────────
    feats = pd.concat([feats_m1, feats_m5_ff, feats_m15_ff], axis=1)
    feats = feats.ffill().dropna()
    log.info(f"Total features (M1+M5+M15): {feats.shape[1]}")

    # ── 6. Prepare sequences ─────────────────────────────────────────────────
    log.info("Building training sequences...")
    X, y = fetcher.prepare_model_input(feats, seq_length=seq_length, normalize=False)
    if len(X) == 0:
        raise RuntimeError("No sequences generated — check feature pipeline.")

    # ── 7. Normalise and save stats ──────────────────────────────────────────
    means = np.mean(X, axis=(0, 1))
    stds  = np.std(X, axis=(0, 1)) + 1e-10
    X     = (X - means) / stds
    fetcher._save_normalization_stats(
        feature_cols=[f"feat_{i}" for i in range(X.shape[2])],
        means=means, stds=stds,
    )

    log.info(f"Sequences: {len(X):,}  |  Shape: {X.shape}  |  Classes: {np.bincount(y)}")
    return X, y



# ─────────────────────────────────────────────
#  GENERIC NEURAL MODEL TRAINER
# ─────────────────────────────────────────────
def _train_model(model_class, config, X_tr, y_tr, X_val, y_val, label):
    """Shared AdamW + CosineAnnealing + early stopping loop for all 9 neural models."""
    model = model_class(config).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    counts = np.bincount(y_tr, minlength=3)
    total  = counts.sum()
    cw = torch.FloatTensor([total / (3 * c) if c > 0 else 1.0 for c in counts]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=config.label_smoothing)

    bs = config.batch_size_gpu if DEVICE.type == "cuda" else config.batch_size
    kw = {"num_workers": 2, "pin_memory": True} if DEVICE.type == "cuda" else {}

    tr_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=bs, shuffle=True, **kw)
    va_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=bs, **kw)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  {label}: {n_params:,} params | bs={bs} | device={DEVICE}")

    best_loss, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(config.epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                vl += criterion(out, by).item()
                vc += (out.argmax(1) == by).sum().item()
                vt += by.size(0)
        vl /= len(va_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            log.info(f"    ep {epoch+1:3d}/{config.epochs} | val_loss={vl:.4f} val_acc={vc/vt:.4f}")

        if vl < best_loss:
            best_loss, patience_cnt = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= config.patience:
                log.info(f"    Early stop at epoch {epoch+1}")
                break

    model = model.cpu()
    if best_state:
        model.load_state_dict(best_state)
    return model



# ─────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_all_colab():
    log.info("=" * 65)
    log.info("  Fusion Markets XAUUSD — 12-Model Training Pipeline")
    log.info("=" * 65)
    log.info(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    t0 = time.time()

    # ── 0. Download CSV if needed ────────────────────────────────────────────
    if USE_GITHUB_CSV and not os.path.exists(CSV_PATH):
        log.info(f"Downloading bars_export_XAUUSD.csv from GitHub...")
        import urllib.request
        os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
        urllib.request.urlretrieve(GITHUB_CSV_URL, CSV_PATH)
        log.info(f"  Saved to {CSV_PATH}")

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    X, y   = prepare_data_from_csv(CSV_PATH, SEQ_LENGTH)
    n_feat = X.shape[2]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3,
                                                  random_state=42, shuffle=False)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                  random_state=42, shuffle=False)
    log.info(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")

    # ── Import configs + model classes ───────────────────────────────────────
    from config.settings import (
        TransformerConfig, LSTMConfig, TCNConfig,
        PatchTSTConfig, TFTConfig, NHiTSConfig,
        ITransformerConfig, MambaConfig, DLinearConfig,
        xLSTMConfig, TimesNetConfig,
    )
    from models.transformer_model import MarketTransformer
    from models.lstm_model         import MarketLSTM
    from models.tcn_model          import MarketTCN
    from models.patch_tst          import MarketPatchTST
    from models.tft_model          import MarketTFT
    from models.nhits_model        import MarketNHiTS
    from models.itransformer       import MarketITransformer
    from models.mamba_model        import MarketMamba
    from models.dlinear_model      import MarketDLinear
    from models.xlstm_model        import MarketXLSTM
    from models.timesnet_model     import MarketTimesNet
    from models.ensemble           import EnsembleManager

    NEURAL_MODELS = [
        ("Transformer",    MarketTransformer,  TransformerConfig(input_features=n_feat)),
        ("LSTM",           MarketLSTM,         LSTMConfig(input_features=n_feat)),
        ("TCN",            MarketTCN,          TCNConfig(input_features=n_feat)),
        ("PatchTST",       MarketPatchTST,     PatchTSTConfig(input_features=n_feat)),
        ("TFT",            MarketTFT,          TFTConfig(input_features=n_feat)),
        ("N-HiTS",         MarketNHiTS,        NHiTSConfig(input_features=n_feat)),
        ("iTransformer",   MarketITransformer, ITransformerConfig(input_features=n_feat)),
        ("Mamba",          MarketMamba,        MambaConfig(input_features=n_feat)),
        ("DLinear",        MarketDLinear,      DLinearConfig(input_features=n_feat)),
        ("xLSTM",          MarketXLSTM,        xLSTMConfig(input_features=n_feat)),
        ("TimesNet",       MarketTimesNet,     TimesNetConfig(input_features=n_feat)),
    ]

    _NEW_MODELS = {"iTransformer", "Mamba", "DLinear", "xLSTM", "TimesNet"}

    # ── 2. Train all neural models ───────────────────────────────────────────
    trained = {}
    for label, cls, cfg in NEURAL_MODELS:
        tag = " (NEW)" if label in _NEW_MODELS else ""
        log.info(f"\n─── {label}{tag} ───")
        trained[label] = _train_model(cls, cfg, X_tr, y_tr, X_val, y_val, label)

    # ── 3. Wire into EnsembleManager ────────────────────────────────────────
    ens = EnsembleManager(
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
    )
    for label, _, _ in NEURAL_MODELS:
        key = label.lower().replace("-", "").replace(" ", "_")
        if hasattr(ens, key):
            setattr(ens, key, trained[label])

    # ── 4. Tree models ───────────────────────────────────────────────────────
    log.info("\n─── Gradient Boosting (sklearn) ───")
    ens.fit_gradient_boost(X_tr, y_tr)

    log.info("\n─── LightGBM / XGBoost ───")
    ens.fit_xgboost(X_tr, y_tr)
    log.info(f"  backend: {ens.xgboost_model.backend}")

    log.info("\n─── CatBoost ───")
    ens.fit_catboost(X_tr, y_tr)
    log.info(f"  backend: {ens.catboost_model.backend}")

    # ── 5. 42-dim meta-learner ───────────────────────────────────────────────
    log.info("\n─── Fitting Meta-Learner (42-dim stack, 14 × 3) ───")
    for m in trained.values():
        m.eval()
    with torch.no_grad():
        Xv = torch.FloatTensor(X_val)
        val_preds = [trained[n].predict(Xv).numpy() for n, _, _ in NEURAL_MODELS]
    val_preds += [
        ens.predict_gradient_boost(X_val),
        ens.predict_xgboost(X_val),
        ens.predict_catboost(X_val),
    ]
    stacked_val = np.concatenate(val_preds, axis=1)   # (n_val, 42)
    ens.fit_meta_learner(stacked_val, y_val)
    log.info("  Meta-learner fitted on 42-dim stacked predictions")

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    log.info("\n─── Test Accuracy (all 12 models) ───")
    with torch.no_grad():
        Xt = torch.FloatTensor(X_te)
        test_preds = [trained[n].predict(Xt).numpy() for n, _, _ in NEURAL_MODELS]
    test_preds += [
        ens.predict_gradient_boost(X_te),
        ens.predict_xgboost(X_te),
        ens.predict_catboost(X_te),
    ]
    model_names = [n for n, _, _ in NEURAL_MODELS] + ["GradBoost", "LightGBM", "CatBoost"]
    accs = [np.mean(np.argmax(p, axis=1) == y_te) for p in test_preds]

    stacked_te = np.concatenate(test_preds, axis=1)
    ens_acc    = np.mean(ens.meta_learner.predict(stacked_te) == y_te)
    best_ind   = max(accs)

    log.info("")
    for name, acc in zip(model_names, accs):
        bar = "█" * int(acc * 40)
        log.info(f"  {name:<18} {acc:.4f}  {bar}")
    log.info(f"  {'─'*50}")
    log.info(f"  Best individual:   {best_ind:.4f}")
    log.info(f"  12-model ensemble: {ens_acc:.4f}  ← target")
    log.info(f"  Ensemble lift:     {ens_acc - best_ind:+.4f}")

    # ── 7. Save to Drive ─────────────────────────────────────────────────────
    log.info(f"\n─── Saving checkpoints → {CHECKPOINT_DIR} ───")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ens.save_models(CHECKPOINT_DIR)
    log.info(f"  All 12 models saved ✓")
    log.info(f"\n  Copy this folder to your trading PC:")
    log.info(f"  {CHECKPOINT_DIR}")
    log.info(f"  → python_bridge/checkpoints/")

    elapsed = time.time() - t0
    log.info(f"\nTotal training time: {elapsed/60:.1f} min")
    log.info("=" * 65)


if __name__ == "__main__":
    train_all_colab()
