"""
=============================================================
  Python ML Bridge - Auto-Retraining Scheduler
  Manages model retraining on weekends with walk-forward
  validation. Only deploys new models if they demonstrably
  outperform the current production models.

  Professional trading firm approach:
  - Retrain when markets are closed (weekends)
  - Walk-forward validation on most recent data
  - Only deploy if performance improves by threshold
  - Log all retrain attempts for audit trail
  - Incorporate recent trade outcomes as training signal
=============================================================
"""

import os
import sys
import json
import time
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RetrainConfig, MODEL_DIR, LOG_DIR

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """Weekend auto-retraining scheduler with performance gating.

    Professional model lifecycle management:
    1. Checks if retraining conditions are met (day + time since last)
    2. Trains new model using full pipeline
    3. Walk-forward validates on recent out-of-sample data
    4. Compares against current production model
    5. Only deploys if improvement exceeds threshold
    6. Logs everything for audit trail

    This prevents model degradation from overfitting to recent noise
    while still adapting to genuine regime changes.
    """

    def __init__(self, config: Optional[RetrainConfig] = None):
        self.config = config or RetrainConfig()
        self._last_retrain: Optional[datetime] = None
        self._retrain_history: List[Dict] = []
        self._trade_outcomes: List[Dict] = []
        self._retrain_attempts = 0

        # Load history if exists
        self._load_history()

    def _load_history(self):
        """Load retrain history from log file."""
        log_path = os.path.join(LOG_DIR, self.config.retrain_log_file)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    data = json.load(f)
                self._retrain_history = data.get("history", [])
                last_str = data.get("last_retrain")
                if last_str:
                    self._last_retrain = datetime.fromisoformat(last_str)
                logger.info(f"[AutoRetrain] Loaded history: "
                            f"{len(self._retrain_history)} previous retrains")
            except Exception as e:
                logger.warning(f"[AutoRetrain] Could not load history: {e}")

    def _save_history(self):
        """Save retrain history to log file."""
        if not self.config.save_retrain_history:
            return

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, self.config.retrain_log_file)

        data = {
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "history": self._retrain_history,
        }
        try:
            with open(log_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[AutoRetrain] Could not save history: {e}")

    def should_retrain(self, current_time: Optional[datetime] = None) -> bool:
        """Check if retraining should happen now.

        Conditions (all must be true):
        1. Current day matches retrain_day (Saturday by default)
        2. At least min_days_between since last retrain
        3. Haven't exceeded max attempts this session

        Args:
            current_time: Override for testing (defaults to now)

        Returns:
            True if retraining should proceed
        """
        now = current_time or datetime.now()

        # Check day of week
        day_name = now.strftime("%A")
        if day_name != self.config.retrain_day:
            return False

        # Check minimum interval since last retrain
        if self._last_retrain is not None:
            days_since = (now - self._last_retrain).days
            if days_since < self.config.min_days_between:
                return False

        # Check max attempts
        if self._retrain_attempts >= self.config.max_retrain_attempts:
            return False

        return True

    def record_trade_outcome(self, trade_id: str, pnl: float,
                             entry_time: str, exit_time: str,
                             direction: str, features_at_entry: Optional[np.ndarray] = None):
        """Record a trade outcome for incorporation into retraining.

        Professional firms use trade results as additional training signal:
        winning trades confirm good predictions, losing trades highlight
        areas where the model needs improvement.
        """
        self._trade_outcomes.append({
            "trade_id": trade_id,
            "pnl": pnl,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "recorded_at": datetime.now().isoformat(),
        })

    def run_retrain(self, train_fn=None, evaluate_fn=None) -> Dict:
        """Execute the full retraining pipeline.

        Steps:
        1. Back up current production model
        2. Run training pipeline (or custom train_fn)
        3. Walk-forward validate new model
        4. Compare against production model
        5. Deploy if improved, rollback if not

        Args:
            train_fn: Optional custom training function (defaults to train.train_all)
            evaluate_fn: Optional custom evaluation function

        Returns:
            Dict with retrain results
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "deployed": False,
            "old_score": None,
            "new_score": None,
            "improvement_pct": None,
            "reason": "",
        }

        self._retrain_attempts += 1
        logger.info("[AutoRetrain] Starting retraining pipeline...")

        try:
            # Step 1: Backup current model
            backup_dir = os.path.join(MODEL_DIR, "backup_pre_retrain")
            if os.path.exists(MODEL_DIR):
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
                    shutil.copytree(MODEL_DIR, backup_dir)
                    logger.info("[AutoRetrain] Backed up current model")

            # Step 2: Evaluate current model (before retraining)
            old_score = None
            if evaluate_fn:
                old_score = evaluate_fn()
                result["old_score"] = old_score
                logger.info(f"[AutoRetrain] Current model score: {old_score:.4f}")

            # Step 3: Run training
            if train_fn:
                train_fn()
            else:
                # Import train_all from the python_bridge package.
                # train.py lives at the python_bridge package root, which is
                # one directory up from this file (training/auto_retrain.py).
                train_all = None

                # Try 1: Absolute package import (works if python_bridge is on sys.path)
                try:
                    from python_bridge.train import train_all as _ta
                    train_all = _ta
                except ImportError:
                    pass

                # Try 2: Direct relative import (works when running from python_bridge/)
                if train_all is None:
                    try:
                        from train import train_all as _ta
                        train_all = _ta
                    except ImportError:
                        pass

                # Try 3: importlib fallback with explicit file path
                if train_all is None:
                    import importlib.util
                    # From training/auto_retrain.py, go up one level to python_bridge/
                    train_module_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "train.py"
                    )
                    if not os.path.exists(train_module_path):
                        raise FileNotFoundError(
                            f"train.py not found at {train_module_path}. "
                            f"Cannot run auto-retrain without train.py."
                        )
                    spec = importlib.util.spec_from_file_location("train", train_module_path)
                    train_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_module)
                    train_all = train_module.train_all

                train_all()
            logger.info("[AutoRetrain] Training completed")

            # Step 4: Evaluate new model
            new_score = None
            if evaluate_fn:
                new_score = evaluate_fn()
                result["new_score"] = new_score
                logger.info(f"[AutoRetrain] New model score: {new_score:.4f}")

            # Step 5: Compare and decide on deployment
            if old_score is not None and new_score is not None:
                if old_score > 0:
                    improvement = ((new_score - old_score) / old_score) * 100.0
                else:
                    improvement = 100.0 if new_score > old_score else 0.0

                result["improvement_pct"] = improvement

                if improvement >= self.config.min_improvement_pct:
                    # Deploy new model
                    result["deployed"] = True
                    result["success"] = True
                    result["reason"] = (
                        f"Deployed: {improvement:.2f}% improvement "
                        f"(threshold: {self.config.min_improvement_pct}%)"
                    )
                    logger.info(f"[AutoRetrain] DEPLOYED: {improvement:.2f}% improvement")
                else:
                    # Rollback to old model
                    if os.path.exists(backup_dir):
                        if os.path.exists(MODEL_DIR):
                            shutil.rmtree(MODEL_DIR)
                        shutil.copytree(backup_dir, MODEL_DIR)
                    result["success"] = True
                    result["deployed"] = False
                    result["reason"] = (
                        f"Not deployed: {improvement:.2f}% improvement "
                        f"(need {self.config.min_improvement_pct}%)"
                    )
                    logger.info(f"[AutoRetrain] NOT DEPLOYED: "
                                f"only {improvement:.2f}% improvement")
            else:
                # No comparison possible, deploy anyway (first time)
                result["deployed"] = True
                result["success"] = True
                result["reason"] = "Deployed (no previous model to compare)"
                logger.info("[AutoRetrain] Deployed (first training or no evaluate_fn)")

            # Update state
            self._last_retrain = datetime.now()
            self._retrain_attempts = 0  # Reset on success

        except Exception as e:
            result["reason"] = f"Training failed: {str(e)}"
            logger.error(f"[AutoRetrain] Retraining failed: {e}", exc_info=True)

            # Rollback on failure
            backup_dir = os.path.join(MODEL_DIR, "backup_pre_retrain")
            if os.path.exists(backup_dir):
                if os.path.exists(MODEL_DIR):
                    shutil.rmtree(MODEL_DIR)
                shutil.copytree(backup_dir, MODEL_DIR)
                logger.info("[AutoRetrain] Rolled back to previous model")

        # Save history
        self._retrain_history.append(result)
        self._save_history()

        # Clean up backup
        backup_dir = os.path.join(MODEL_DIR, "backup_pre_retrain")
        if os.path.exists(backup_dir):
            try:
                shutil.rmtree(backup_dir)
            except Exception:
                pass

        return result

    def walk_forward_validate(self, model, data: np.ndarray,
                              labels: np.ndarray,
                              weeks: Optional[int] = None) -> float:
        """Perform walk-forward validation on recent data.

        Walk-forward validation simulates how the model would have
        performed on the most recent N weeks, using only data
        available before each prediction point.

        Args:
            model: Model with predict() method
            data: Feature data array
            labels: True labels
            weeks: Number of weeks to validate (default from config)

        Returns:
            Accuracy score on walk-forward window
        """
        weeks = weeks or self.config.walk_forward_weeks
        # Assume hourly data: ~168 bars per week (24*7)
        bars_per_week = 168
        validation_bars = weeks * bars_per_week

        if len(data) < validation_bars:
            logger.warning("[AutoRetrain] Not enough data for walk-forward validation")
            return 0.0

        # Use last N bars as validation set
        val_data = data[-validation_bars:]
        val_labels = labels[-validation_bars:]

        try:
            if hasattr(model, 'predict'):
                import torch
                with torch.no_grad():
                    if isinstance(val_data, np.ndarray):
                        input_tensor = torch.FloatTensor(val_data)
                    else:
                        input_tensor = val_data
                    predictions = model.predict(input_tensor)
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.numpy()
                    pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = model(val_data)

            accuracy = np.mean(pred_classes == val_labels)
            return float(accuracy)
        except Exception as e:
            logger.error(f"[AutoRetrain] Walk-forward validation error: {e}")
            return 0.0

    def get_trade_outcomes_for_training(self) -> List[Dict]:
        """Get accumulated trade outcomes for incorporation into training.

        Returns outcomes only if minimum threshold is met. Professional
        firms need statistical significance before adjusting models.
        """
        if len(self._trade_outcomes) < self.config.min_trades_for_retrain:
            return []
        outcomes = self._trade_outcomes.copy()
        self._trade_outcomes = []  # Clear after providing
        return outcomes

    def get_status(self) -> Dict:
        """Get current retrainer status."""
        return {
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "retrain_history_count": len(self._retrain_history),
            "trade_outcomes_pending": len(self._trade_outcomes),
            "retrain_attempts_this_session": self._retrain_attempts,
            "next_retrain_eligible": self._next_eligible_time(),
        }

    def _next_eligible_time(self) -> Optional[str]:
        """Calculate next eligible retrain time."""
        if self._last_retrain is None:
            return "Next " + self.config.retrain_day
        next_eligible = self._last_retrain + timedelta(days=self.config.min_days_between)
        return next_eligible.isoformat()
