"""
=============================================================
  Python ML Bridge - Model Training Script
  Downloads historical data, trains transformer and LSTM models,
  trains gradient boosting, fits meta-learner.
  Supports incremental online learning from new data.
=============================================================
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    TransformerConfig, LSTMConfig, TCNConfig, EnsembleConfig,
    XGBoostConfig, DataConfig, MODEL_DIR
)
from data.market_data import MarketDataFetcher
from models.transformer_model import MarketTransformer
from models.lstm_model import MarketLSTM
from models.tcn_model import MarketTCN
from models.ensemble import EnsembleManager


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Training")

# ─────────────────────────────────────────────
#  GPU SUPPORT - AUTO-DETECT CUDA
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  TRAINING FUNCTIONS
# ─────────────────────────────────────────────
def prepare_data(config: DataConfig = None,
                 seq_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Download historical data from multiple timeframes and prepare training sequences.

    Fetches 5y daily + 2y H1 + 60d M15 data to build a large, diverse training
    set targeting 25,000-30,000+ sequences. Each timeframe is processed independently
    through compute_features() and prepare_model_input(), then all X/y arrays
    are concatenated.

    Returns:
        Tuple of (X, y) arrays
    """
    config = config or DataConfig()
    fetcher = MarketDataFetcher(config)

    all_X = []
    all_y = []

    # Fetch and process each timeframe
    for tf_spec in config.training_periods:
        period = tf_spec["period"]
        interval = tf_spec["interval"]

        logger.info(f"Downloading {period} data at {interval} interval...")
        df = fetcher.fetch_ohlcv(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data for period={period}, interval={interval}")
            continue

        logger.info(f"Downloaded {len(df)} bars for {period}/{interval}")

        logger.info("Computing features...")
        features = fetcher.compute_features(df)
        if features.empty:
            logger.warning(f"Feature computation failed for {period}/{interval}")
            continue

        logger.info(f"Computed {len(features.columns)} features over {len(features)} bars")

        logger.info("Preparing sequences...")
        # Pass normalize=False so we get raw (unnormalized) feature sequences.
        # Normalization is applied once after all timeframes are concatenated,
        # ensuring the saved stats reflect the true raw feature distributions
        # and match what inference will see.
        X, y = fetcher.prepare_model_input(features, seq_length=seq_length,
                                           normalize=False)
        if len(X) == 0:
            logger.warning(f"No sequences generated for {period}/{interval}")
            continue

        logger.info(f"Created {len(X)} sequences from {period}/{interval}")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        logger.error("No training data available from any timeframe")
        return np.array([]), np.array([])

    # Concatenate all timeframes
    X = np.vstack(all_X)
    y = np.hstack(all_y)

    # Compute combined normalization stats from the RAW concatenated data.
    # Since prepare_model_input was called with normalize=False, X contains
    # unnormalized features. We compute means/stds across all samples and
    # time steps, normalize X in-place, and save the stats so that inference
    # applies the same transform to raw features.
    combined_means = np.mean(X, axis=(0, 1))  # mean across samples and time steps
    combined_stds = np.std(X, axis=(0, 1)) + 1e-10

    # Normalize X in-place using the combined stats
    X = (X - combined_means) / combined_stds

    fetcher._save_normalization_stats(
        feature_cols=[f"feat_{i}" for i in range(X.shape[2])],
        means=combined_means,
        stds=combined_stds,
    )
    logger.info("Saved combined normalization stats from all %d sequences", len(X))

    logger.info(f"Total training sequences: {len(X)} (target: 25,000+)")
    logger.info(f"Class distribution: {np.bincount(y)}")

    return X, y


def train_transformer(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      config: Optional[TransformerConfig] = None) -> MarketTransformer:
    """
    Train the transformer model.

    Args:
        X_train: Training features (N, seq_len, features)
        y_train: Training labels (N,)
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration

    Returns:
        Trained MarketTransformer model (on CPU for portable checkpoints)
    """
    config = config or TransformerConfig()
    config.input_features = X_train.shape[2]

    model = MarketTransformer(config)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Class weight balancing (handles imbalanced BUY/HOLD/SELL distribution)
    class_counts = np.bincount(y_train, minlength=3)
    total = class_counts.sum()
    class_weights = torch.FloatTensor([total / (3 * c) if c > 0 else 1.0 for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)

    # Select batch size based on device (GPU can handle larger batches)
    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )

    # Use pin_memory and num_workers for faster GPU data transfer
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {"num_workers": 0}
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, **loader_kwargs)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Training Transformer: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"  Device: {device} | Batch size: {batch_size}")

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model on CPU for device-agnostic checkpoints
    model = model.cpu()
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               config: Optional[LSTMConfig] = None) -> MarketLSTM:
    """
    Train the LSTM model.

    Args:
        X_train: Training features (N, seq_len, features)
        y_train: Training labels (N,)
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration

    Returns:
        Trained MarketLSTM model (on CPU for portable checkpoints)
    """
    config = config or LSTMConfig()
    config.input_features = X_train.shape[2]

    model = MarketLSTM(config)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Class weight balancing (handles imbalanced BUY/HOLD/SELL distribution)
    class_counts = np.bincount(y_train, minlength=3)
    total = class_counts.sum()
    class_weights = torch.FloatTensor([total / (3 * c) if c > 0 else 1.0 for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)

    # Select batch size based on device (GPU can handle larger batches)
    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )

    # Use pin_memory and num_workers for faster GPU data transfer
    loader_kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {"num_workers": 0}
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, **loader_kwargs)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Training LSTM: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"  Device: {device} | Batch size: {batch_size}")

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model on CPU for device-agnostic checkpoints
    model = model.cpu()
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    return model


def train_tcn(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: Optional[TCNConfig] = None) -> MarketTCN:
    """
    Train the Temporal Convolutional Network.

    Architecture: 6 dilated conv blocks (dilation 1→32), receptive field
    covers the full 64-bar window. Fully parallelisable — fast on CPU.

    Args:
        X_train: Training features (N, seq_len, features)
        y_train: Training labels (N,)
        X_val:   Validation features
        y_val:   Validation labels
        config:  TCN configuration

    Returns:
        Trained MarketTCN model (on CPU for portable checkpoints)
    """
    config = config or TCNConfig()
    config.input_features = X_train.shape[2]

    model = MarketTCN(config)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Class-weighted loss to handle BUY/HOLD/SELL imbalance
    class_counts = np.bincount(y_train, minlength=3)
    total = class_counts.sum()
    class_weights = torch.FloatTensor(
        [total / (3 * c) if c > 0 else 1.0 for c in class_counts]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config.label_smoothing
    )

    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size
    loader_kwargs = (
        {"num_workers": 4, "pin_memory": True} if device.type == "cuda"
        else {"num_workers": 0}
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              **loader_kwargs)

    best_val_loss = float("inf")
    patience_counter = 0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Training TCN: {n_params:,} parameters")
    logger.info(f"  Device: {device} | Batch size: {batch_size} | "
                f"Dilation: 1→{2**(config.n_layers-1)}")

    for epoch in range(config.epochs):
        # ── train ──
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                out = model(batch_X)
                val_loss += criterion(out, batch_y).item()
                _, pred = torch.max(out, 1)
                total   += batch_y.size(0)
                correct += (pred == batch_y).sum().item()
        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # ── early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    model = model.cpu()
    if "best_state" in locals():
        model.load_state_dict(best_state)

    return model


def train_all():
    """
    Full training pipeline:
        1. Download and prepare data
        2. Train Transformer
        3. Train LSTM
        4. Train TCN              (NEW)
        5. Train Gradient Boosting (sklearn)
        6. Train XGBoost / LightGBM (NEW)
        7. Fit meta-learner on 15-dim stacked predictions
        8. Evaluate all 5 models + ensemble on held-out test set
        9. Save all checkpoints
    """
    logger.info("=" * 60)
    logger.info("  Python ML Bridge - Full Training Pipeline")
    logger.info("=" * 60)

    # Log training device info
    logger.info(f"Training device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    start_time = time.time()

    # 1. Prepare data
    X, y = prepare_data()
    if len(X) == 0:
        logger.error("No training data available. Exiting.")
        return

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
    )

    logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}")
    logger.info(f"Class distribution: {np.bincount(y_train)}")

    # 2. Train Transformer
    logger.info("\n--- Training Transformer ---")
    transformer_config = TransformerConfig(input_features=X_train.shape[2])
    transformer = train_transformer(X_train, y_train, X_val, y_val, transformer_config)

    # 3. Train LSTM
    logger.info("\n--- Training LSTM ---")
    lstm_config = LSTMConfig(input_features=X_train.shape[2])
    lstm = train_lstm(X_train, y_train, X_val, y_val, lstm_config)

    # 4. Train TCN (NEW)
    logger.info("\n--- Training TCN ---")
    tcn_config = TCNConfig(input_features=X_train.shape[2])
    tcn = train_tcn(X_train, y_train, X_val, y_val, tcn_config)

    # 5. Train Gradient Boosting (sklearn)
    logger.info("\n--- Training Gradient Boosting (sklearn) ---")
    ensemble = EnsembleManager(
        transformer_config=transformer_config,
        lstm_config=lstm_config,
        tcn_config=tcn_config,
    )
    ensemble.transformer = transformer
    ensemble.lstm = lstm
    ensemble.tcn = tcn
    ensemble.fit_gradient_boost(X_train, y_train)
    logger.info("  Gradient Boosting fitted")

    # 6. Train XGBoost / LightGBM (NEW)
    logger.info("\n--- Training XGBoost / LightGBM ---")
    ensemble.fit_xgboost(X_train, y_train)
    logger.info(f"  XGBoost fitted (backend={ensemble.xgboost_model.backend})")

    # 7. Fit Meta-Learner on 15-dim stacked predictions
    logger.info("\n--- Fitting Meta-Learner (15-dim stack) ---")
    transformer.eval()
    lstm.eval()
    tcn.eval()
    with torch.no_grad():
        t_preds  = transformer.predict(torch.FloatTensor(X_val)).numpy()
        l_preds  = lstm.predict(torch.FloatTensor(X_val)).numpy()
        c_preds  = tcn.predict(torch.FloatTensor(X_val)).numpy()
    gb_preds  = ensemble.predict_gradient_boost(X_val)
    xgb_preds = ensemble.predict_xgboost(X_val)
    stacked = np.concatenate([t_preds, l_preds, c_preds, gb_preds, xgb_preds], axis=1)
    ensemble.fit_meta_learner(stacked, y_val)
    logger.info("  Meta-learner fitted")

    # 8. Evaluate on test set — all 5 models + ensemble
    logger.info("\n--- Test Set Evaluation (5 models) ---")
    with torch.no_grad():
        t_test   = transformer.predict(torch.FloatTensor(X_test)).numpy()
        l_test   = lstm.predict(torch.FloatTensor(X_test)).numpy()
        c_test   = tcn.predict(torch.FloatTensor(X_test)).numpy()
    gb_test  = ensemble.predict_gradient_boost(X_test)
    xgb_test = ensemble.predict_xgboost(X_test)

    t_acc   = np.mean(np.argmax(t_test,   axis=1) == y_test)
    l_acc   = np.mean(np.argmax(l_test,   axis=1) == y_test)
    c_acc   = np.mean(np.argmax(c_test,   axis=1) == y_test)
    gb_acc  = np.mean(np.argmax(gb_test,  axis=1) == y_test)
    xgb_acc = np.mean(np.argmax(xgb_test, axis=1) == y_test)

    stacked_test = np.concatenate(
        [t_test, l_test, c_test, gb_test, xgb_test], axis=1
    )
    ensemble_preds = ensemble.meta_learner.predict(stacked_test)
    ensemble_acc = np.mean(ensemble_preds == y_test)

    logger.info(f"  Transformer accuracy:   {t_acc:.4f}")
    logger.info(f"  LSTM accuracy:          {l_acc:.4f}")
    logger.info(f"  TCN accuracy:           {c_acc:.4f}  (NEW)")
    logger.info(f"  Gradient Boost acc:     {gb_acc:.4f}")
    logger.info(f"  XGBoost/LightGBM acc:   {xgb_acc:.4f}  (NEW)")
    logger.info(f"  Ensemble accuracy:      {ensemble_acc:.4f}  ← target metric")
    logger.info(f"  Best individual:        {max(t_acc, l_acc, c_acc, gb_acc, xgb_acc):.4f}")
    logger.info(f"  Ensemble lift:          {ensemble_acc - max(t_acc, l_acc, c_acc, gb_acc, xgb_acc):+.4f}")

    # 9. Save checkpoints
    logger.info("\n--- Saving Checkpoints ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensemble.save_models(MODEL_DIR)
    logger.info(f"  Saved to {MODEL_DIR}")

    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed:.1f} seconds")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all()
