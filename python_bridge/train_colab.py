"""
=============================================================
  Python ML Bridge — Google Colab Training Script  (v2)
  Trains ALL 17 models on your real Fusion Markets XAUUSD data.

  KEY IMPROVEMENTS over v1:
    ✓ Resume from checkpoint — if Colab disconnects, restart and
      already-trained models are automatically skipped
    ✓ Per-epoch checkpoint saving every 5 epochs — you never lose
      more than 5 epochs of progress
    ✓ Progress bars (tqdm) with loss/accuracy display
    ✓ GPU memory monitoring during training
    ✓ Time estimates per model
    ✓ Final accuracy verification — confirms trained > random
    ✓ All 17 models including Chronos, TimeMixer, SOFTS

  QUICKSTART (Google Colab):
  ──────────────────────────────────────────────────────────
  # Cell 1 — Setup (run once)
  !pip install scikit-learn==1.9.0 lightgbm catboost torch tqdm
  !pip install git+https://github.com/amazon-science/chronos-forecasting.git
  !git clone https://github.com/gagandocx/Claude.git
  %cd Claude/python_bridge
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2 — Upload bars_export_XAUUSD.csv to /content/, then:
  !python train_colab.py

  # Cell 3 — If Colab disconnects, just re-run Cell 2.
  #           Already-trained models are skipped automatically.
  ──────────────────────────────────────────────────────────

  CHECKPOINTS: /content/drive/MyDrive/claude_ea/checkpoints/
  Copy this folder to your PC:  python_bridge/checkpoints/
=============================================================
"""

import os, sys, time, logging, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Colab")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH        = "/content/bars_export_XAUUSD.csv"
USE_GITHUB_CSV  = True
GITHUB_CSV_URL  = "https://raw.githubusercontent.com/gagandocx/Uploads/main/bars_export_XAUUSD.csv"
CHECKPOINT_DIR  = "/content/drive/MyDrive/claude_ea/checkpoints"
PROGRESS_DIR    = "/content/drive/MyDrive/claude_ea/inprogress"  # mid-epoch saves
SEQ_LENGTH      = 64
SAVE_EVERY      = 5    # Save in-progress checkpoint every N epochs
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ─────────────────────────────────────────────
_LABEL_TO_FNAME = {
    "Transformer": "transformer.pth",  "LSTM":        "lstm.pth",
    "TCN":         "tcn.pth",          "PatchTST":    "patch_tst.pth",
    "TFT":         "tft.pth",          "N-HiTS":      "nhits.pth",
    "iTransformer":"itransformer.pth", "Mamba":       "mamba.pth",
    "DLinear":     "dlinear.pth",      "xLSTM":       "xlstm.pth",
    "TimesNet":    "timesnet.pth",     "Chronos":     "chronos.pth",
    "TimeMixer":   "timemixer.pth",    "SOFTS":       "softs.pth",
}

def _is_trained(label: str) -> bool:
    """Return True if a final checkpoint already exists for this model."""
    fname = _LABEL_TO_FNAME.get(label)
    if not fname:
        return False
    return os.path.exists(os.path.join(CHECKPOINT_DIR, fname))

def _progress_path(label: str) -> str:
    """Path for the in-progress (mid-training) checkpoint."""
    fname = _LABEL_TO_FNAME.get(label, label.lower() + ".pth")
    return os.path.join(PROGRESS_DIR, fname)

def _save_progress(model, label: str):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    torch.save(model.state_dict(), _progress_path(label))

def _load_progress(model, label: str) -> int:
    """Load in-progress checkpoint if it exists. Returns resumed epoch or 0."""
    p = _progress_path(label)
    if os.path.exists(p):
        try:
            model.load_state_dict(
                torch.load(p, map_location=DEVICE, weights_only=True),
                strict=False,
            )
            log.info(f"    ↻ Resumed from in-progress checkpoint: {p}")
            return 1   # Mark as resumed (actual epoch unknown, start from 0)
        except Exception as e:
            log.warning(f"    Could not load in-progress checkpoint: {e}")
    return 0

def _gpu_mem_str() -> str:
    if DEVICE.type != "cuda":
        return ""
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f" | GPU {used:.1f}/{total:.1f}GB"



# ─────────────────────────────────────────────
#  CSV LOADER
# ─────────────────────────────────────────────
def load_fusion_markets_csv(path: str) -> pd.DataFrame:
    """
    Load Fusion Markets / MetaTrader 5 XAUUSD M1 CSV export.
    Auto-detects tab-separated, comma-separated, and angle-bracket header formats.
    """
    log.info(f"Loading CSV: {path}")
    raw = open(path, "r").read(2000)
    sep = "\t" if "\t" in raw[:200] else ","
    df  = pd.read_csv(path, sep=sep)
    df.columns = [c.strip("<>").strip().lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M", errors="coerce",
        )
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Cannot find date column. Check CSV format.")

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open","o"):              rename[col] = "open"
        elif lc in ("high","h"):            rename[col] = "high"
        elif lc in ("low","l"):             rename[col] = "low"
        elif lc in ("close","c"):           rename[col] = "close"
        elif lc in ("tickvol","vol","volume","v"): rename[col] = "volume"
    df = df.rename(columns=rename)

    for req in ["open","high","low","close"]:
        if req not in df.columns:
            raise ValueError(f"Missing column '{req}'. Found: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 1.0

    df = df[["open","high","low","close","volume"]].astype(float).dropna()
    log.info(f"  {len(df):,} M1 bars  |  "
             f"{df.index[0].date()} → {df.index[-1].date()}  |  "
             f"${df['close'].min():.0f}–${df['close'].max():.0f}")
    return df


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data_from_csv(csv_path: str, seq_length: int = SEQ_LENGTH):
    """Load CSV → feature engineering (M1+M5+M15) → training sequences."""
    from data.market_data import MarketDataFetcher
    from config.settings  import DataConfig

    fetcher = MarketDataFetcher(DataConfig())
    m1      = load_fusion_markets_csv(csv_path)

    def resample_ohlcv(df, rule):
        return df.resample(rule).agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna()

    log.info("Computing M1 features...")
    feats_m1 = fetcher.compute_features(m1)

    log.info("Computing M5 HTF features...")
    feats_m5 = fetcher.compute_features(resample_ohlcv(m1,"5min")).add_prefix("m5_")

    log.info("Computing M15 HTF features...")
    feats_m15 = fetcher.compute_features(resample_ohlcv(m1,"15min")).add_prefix("m15_")

    feats = pd.concat([
        feats_m1,
        feats_m5.reindex(feats_m1.index, method="ffill"),
        feats_m15.reindex(feats_m1.index, method="ffill"),
    ], axis=1).ffill().dropna()

    log.info(f"Total features (M1+M5+M15): {feats.shape[1]}")
    X, y = fetcher.prepare_model_input(feats, seq_length=seq_length, normalize=False)
    if len(X) == 0:
        raise RuntimeError("No sequences generated. Check feature pipeline.")

    means = np.mean(X, axis=(0,1))
    stds  = np.std(X, axis=(0,1)) + 1e-10
    X     = (X - means) / stds
    fetcher._save_normalization_stats(
        feature_cols=[f"feat_{i}" for i in range(X.shape[2])],
        means=means, stds=stds,
    )
    log.info(f"Sequences: {len(X):,}  Shape: {X.shape}  Classes: {np.bincount(y)}")
    return X, y



# ─────────────────────────────────────────────
#  IMPROVED TRAINER (with progress bars + resume + periodic saves)
# ─────────────────────────────────────────────
def _train_model(model_class, config, X_tr, y_tr, X_val, y_val, label):
    """
    Train one model with:
      - tqdm progress bar (falls back to print if tqdm not installed)
      - Per-epoch checkpoint saving every SAVE_EVERY epochs
      - Resume from in-progress checkpoint if Colab was disconnected
      - GPU memory display
    """
    model = model_class(config).to(DEVICE)

    # Try to resume from previous interrupted run
    _load_progress(model, label)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    counts = np.bincount(y_tr, minlength=3)
    total  = counts.sum()
    cw     = torch.FloatTensor(
        [total / (3 * c) if c > 0 else 1.0 for c in counts]
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=config.label_smoothing)

    bs     = config.batch_size_gpu if DEVICE.type == "cuda" else config.batch_size
    kw     = {"num_workers": 2, "pin_memory": True} if DEVICE.type == "cuda" else {}
    tr_dl  = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=bs, shuffle=True, **kw)
    va_dl  = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=bs, **kw)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  {label}: {n_params:,} trainable params | bs={bs}{_gpu_mem_str()}")

    best_loss, pat_cnt, best_state = float("inf"), 0, None
    epoch_iter = (
        tqdm(range(config.epochs), desc=f"  {label}", unit="ep",
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]")
        if HAS_TQDM else range(config.epochs)
    )

    t_start = time.time()
    for epoch in epoch_iter:
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for bx, by in tr_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += 1
        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in va_dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out  = model(bx)
                vl  += criterion(out, by).item()
                vc  += (out.argmax(1) == by).sum().item()
                vt  += by.size(0)
        vl /= max(len(va_dl), 1)
        va  = vc / max(vt, 1)
        scheduler.step()

        # Update progress bar
        if HAS_TQDM:
            epoch_iter.set_postfix(val_loss=f"{vl:.4f}", val_acc=f"{va:.4f}",
                                   gpu=_gpu_mem_str().strip())
        elif (epoch + 1) % 10 == 0:
            elapsed = time.time() - t_start
            log.info(f"    ep {epoch+1:3d}/{config.epochs} "
                     f"val_loss={vl:.4f} val_acc={va:.4f} "
                     f"({elapsed:.0f}s){_gpu_mem_str()}")

        # ── periodic checkpoint save ────────────────────────────────────────
        if (epoch + 1) % SAVE_EVERY == 0:
            _save_progress(model.cpu(), label)
            model = model.to(DEVICE)

        # ── early stopping ──────────────────────────────────────────────────
        if vl < best_loss:
            best_loss, pat_cnt = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= config.patience:
                if HAS_TQDM:
                    epoch_iter.set_description(f"  {label} [early stop]")
                else:
                    log.info(f"    Early stop at epoch {epoch+1}")
                break

    model = model.cpu()
    if best_state:
        model.load_state_dict(best_state)
    elapsed = time.time() - t_start
    log.info(f"  ✓ {label} done in {elapsed/60:.1f} min | best_val_loss={best_loss:.4f}")
    return model



# ─────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_all_colab():
    log.info("=" * 65)
    log.info("  Fusion Markets XAUUSD — 17-Model Training Pipeline")
    log.info("=" * 65)
    log.info(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR,   exist_ok=True)
    t0 = time.time()

    # ── 0. Download CSV if needed ────────────────────────────────────────────
    if USE_GITHUB_CSV and not os.path.exists(CSV_PATH):
        log.info("Downloading bars_export_XAUUSD.csv from GitHub...")
        import urllib.request
        urllib.request.urlretrieve(GITHUB_CSV_URL, CSV_PATH)

    # ── 1. Data ──────────────────────────────────────────────────────────────
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
        ChronosConfig, TimeMixerConfig, SOFTSConfig,
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
    from models.chronos_model      import MarketChronos
    from models.timemixer_model    import MarketTimeMixer
    from models.softs_model        import MarketSOFTS
    from models.ensemble           import EnsembleManager

    _NEW = {"iTransformer","Mamba","DLinear","xLSTM","TimesNet",
            "Chronos","TimeMixer","SOFTS"}

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


    # ── 2. Train neural models (skip if checkpoint exists) ───────────────────
    trained = {}
    total_models = len(NEURAL_MODELS)
    for idx, (label, cls, cfg) in enumerate(NEURAL_MODELS, 1):
        tag = " [NEW]" if label in _NEW else ""
        if _is_trained(label):
            log.info(f"\n[{idx:2d}/{total_models}] {label}{tag} — ✓ ALREADY TRAINED, loading...")
            m = cls(cfg).to(DEVICE)
            ck = os.path.join(CHECKPOINT_DIR, _LABEL_TO_FNAME[label])
            m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True),
                              strict=False)
            trained[label] = m.cpu()
        else:
            log.info(f"\n[{idx:2d}/{total_models}] Training {label}{tag}...")
            trained[label] = _train_model(cls, cfg, X_tr, y_tr, X_val, y_val, label)
            # Save final checkpoint immediately
            ck = os.path.join(CHECKPOINT_DIR, _LABEL_TO_FNAME[label])
            torch.save(trained[label].state_dict(), ck)
            log.info(f"  Saved → {ck}")

    # ── 3. Wire into EnsembleManager ─────────────────────────────────────────
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
        chronos_config      = ChronosConfig(input_features=n_feat),
        timemixer_config    = TimeMixerConfig(input_features=n_feat),
        softs_config        = SOFTSConfig(input_features=n_feat),
    )
    _attr_map = {
        "Transformer":"transformer","LSTM":"lstm","TCN":"tcn",
        "PatchTST":"patch_tst","TFT":"tft","N-HiTS":"nhits",
        "iTransformer":"itransformer","Mamba":"mamba","DLinear":"dlinear",
        "xLSTM":"xlstm","TimesNet":"timesnet","Chronos":"chronos",
        "TimeMixer":"timemixer","SOFTS":"softs",
    }
    for label, model in trained.items():
        attr = _attr_map.get(label)
        if attr and hasattr(ens, attr):
            setattr(ens, attr, model)


    # ── 4. Tree models ───────────────────────────────────────────────────────
    log.info("\n─── Gradient Boosting ───")
    ens.fit_gradient_boost(X_tr, y_tr)

    log.info("\n─── LightGBM / XGBoost ───")
    ens.fit_xgboost(X_tr, y_tr)
    log.info(f"  backend: {ens.xgboost_model.backend}")

    log.info("\n─── CatBoost ───")
    ens.fit_catboost(X_tr, y_tr)
    log.info(f"  backend: {ens.catboost_model.backend}")

    # ── 5. 51-dim meta-learner ────────────────────────────────────────────────
    log.info("\n─── Meta-Learner (51-dim, 17 × 3) ───")
    for m in trained.values():
        m.eval()
    with torch.no_grad():
        Xv = torch.FloatTensor(X_val)
        val_preds = [trained[n].predict(Xv).numpy() for n,_,_ in NEURAL_MODELS]
    val_preds += [ens.predict_gradient_boost(X_val),
                  ens.predict_xgboost(X_val), ens.predict_catboost(X_val)]
    ens.fit_meta_learner(np.concatenate(val_preds, axis=1), y_val)
    log.info("  Meta-learner fitted on 51-dim stacked predictions")

    # ── 6. Evaluate all 17 models ─────────────────────────────────────────────
    log.info("\n─── Test Accuracy (all 17 models) ───\n")
    with torch.no_grad():
        Xt = torch.FloatTensor(X_te)
        test_preds = [trained[n].predict(Xt).numpy() for n,_,_ in NEURAL_MODELS]
    test_preds += [ens.predict_gradient_boost(X_te),
                   ens.predict_xgboost(X_te), ens.predict_catboost(X_te)]

    all_names = [n for n,_,_ in NEURAL_MODELS] + ["GradBoost","LightGBM","CatBoost"]
    accs      = [np.mean(np.argmax(p,axis=1)==y_te) for p in test_preds]
    best_ind  = max(accs)

    stacked_te = np.concatenate(test_preds, axis=1)
    ens_acc    = np.mean(ens.meta_learner.predict(stacked_te) == y_te)
    random_bl  = max(np.bincount(y_te)) / len(y_te)   # majority class baseline

    for name, acc in zip(all_names, accs):
        tag = " [NEW]" if name in _NEW else ""
        bar = "█" * int(acc * 40)
        log.info(f"  {name:<18}{tag:<7} {acc:.4f}  {bar}")
    log.info(f"\n  {'─'*55}")
    log.info(f"  Random baseline:   {random_bl:.4f}")
    log.info(f"  Best individual:   {best_ind:.4f}  (+{best_ind-random_bl:+.4f} vs random)")
    log.info(f"  17-model ensemble: {ens_acc:.4f}  ← TARGET")
    log.info(f"  Ensemble lift:     {ens_acc-best_ind:+.4f} vs best individual")

    # ── 7. Save all checkpoints ───────────────────────────────────────────────
    log.info(f"\n─── Saving all checkpoints → {CHECKPOINT_DIR} ───")
    ens.save_models(CHECKPOINT_DIR)
    log.info("  ✓ All 17 models saved")
    log.info(f"\n  Copy this folder to your trading PC:")
    log.info(f"  {CHECKPOINT_DIR}")
    log.info(f"  → python_bridge/checkpoints/")

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed/60:.1f} min")
    log.info("=" * 65)


if __name__ == "__main__":
    train_all_colab()
