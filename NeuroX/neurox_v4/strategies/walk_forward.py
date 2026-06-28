"""
=============================================================
  Python ML Bridge v3 - Walk-Forward Retrainer (Tier 1)

  Automated weekly retraining pipeline:
    1. save_training_data() - accumulate features and labels
    2. retrain_with_validation() - train on recent data with
       walk-forward validation split
    3. deploy_if_improved() - only deploy if model beats baseline
    4. schedule_weekly() - runs every Sunday at configurable time

  Extends the existing AutoRetrainer with proper walk-forward splits
  and data management. Backward-compatible with existing checkpoints.
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
from config.settings import WalkForwardConfig, MODEL_DIR, LOG_DIR

logger = logging.getLogger(__name__)


class WalkForwardRetrainer:
    """
    Walk-forward retraining pipeline for continuous model improvement.

    Professional quantitative trading firms retrain models on a regular
    schedule using walk-forward validation to prevent lookahead bias.
    This class implements that workflow:

    1. Data accumulation: Training data (features + labels) saved
       incrementally as the system runs.
    2. Walk-forward splits: When retraining, data is split into expanding
       training window + fixed validation window. Only the most recent
       data is used (no stale patterns).
    3. Performance gating: New model must beat the old model by a minimum
       improvement percentage on the validation set.
    4. Automatic scheduling: Runs weekly (configurable) to adapt to
       evolving market conditions.

    Integration:
        Called from main.py's main loop. Checks schedule each cycle.
        On trigger, runs full retrain pipeline in the main thread
        (blocking but bounded by max_retrain_duration_min).
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

        # Data storage path
        self._data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.config.data_save_path
        )
        os.makedirs(self._data_dir, exist_ok=True)

        # State tracking
        self._last_retrain_time: Optional[float] = None
        self._retrain_history: List[Dict] = []
        self._accumulated_samples: int = 0

        # Load state if exists
        self._load_state()

        logger.info("[WalkForward] Initialized. interval=%dh, validation=%d bars, "
                    "min_improvement=%.1f%%, data_path=%s",
                    self.config.retrain_interval_hours,
                    self.config.validation_window_bars,
                    self.config.min_improvement_pct,
                    self._data_dir)

    def _load_state(self):
        """Load saved state from disk."""
        state_file = os.path.join(self._data_dir, 'walk_forward_state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self._last_retrain_time = state.get('last_retrain_time')
                self._retrain_history = state.get('retrain_history', [])
                self._accumulated_samples = state.get('accumulated_samples', 0)
                logger.info("[WalkForward] Loaded state: last_retrain=%s, "
                            "history_count=%d, samples=%d",
                            self._last_retrain_time,
                            len(self._retrain_history),
                            self._accumulated_samples)
            except Exception as e:
                logger.warning("[WalkForward] Could not load state: %s", e)

    def _save_state(self):
        """Persist state to disk."""
        state_file = os.path.join(self._data_dir, 'walk_forward_state.json')
        state = {
            'last_retrain_time': self._last_retrain_time,
            'retrain_history': self._retrain_history[-50:],  # Keep last 50
            'accumulated_samples': self._accumulated_samples,
        }
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning("[WalkForward] Could not save state: %s", e)

    def save_training_data(self, features: np.ndarray, labels: np.ndarray,
                           metadata: Optional[Dict] = None) -> None:
        """
        Save a batch of training data for future retraining.

        Data is saved as .npz files with timestamps. Older files
        are cleaned up when total exceeds a threshold.

        Args:
            features: Feature array (n_samples, seq_len, n_features)
            labels: Label array (n_samples,)
            metadata: Optional metadata dict (regime, timestamp, etc.)
        """
        if features is None or labels is None:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_batch_{timestamp}.npz'
        filepath = os.path.join(self._data_dir, filename)

        try:
            save_dict = {'features': features, 'labels': labels}
            if metadata:
                save_dict['metadata'] = np.array([json.dumps(metadata)])
            np.savez_compressed(filepath, **save_dict)

            self._accumulated_samples += len(labels)
            logger.debug("[WalkForward] Saved %d samples to %s (total: %d)",
                         len(labels), filename, self._accumulated_samples)

            # Cleanup old files if too many accumulated
            self._cleanup_old_data()
        except Exception as e:
            logger.warning("[WalkForward] Error saving training data: %s", e)

    def _cleanup_old_data(self, max_files: int = 100):
        """Remove oldest data files if exceeding max count."""
        try:
            files = sorted([
                f for f in os.listdir(self._data_dir)
                if f.startswith('training_batch_') and f.endswith('.npz')
            ])
            while len(files) > max_files:
                oldest = files.pop(0)
                os.remove(os.path.join(self._data_dir, oldest))
                logger.debug("[WalkForward] Removed old data file: %s", oldest)
        except Exception as e:
            logger.debug("[WalkForward] Cleanup error: %s", e)

    def _load_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load all accumulated training data.

        Returns:
            Tuple of (features, labels) or (None, None) if insufficient data
        """
        try:
            files = sorted([
                f for f in os.listdir(self._data_dir)
                if f.startswith('training_batch_') and f.endswith('.npz')
            ])

            if not files:
                return None, None

            all_features = []
            all_labels = []

            for filename in files:
                filepath = os.path.join(self._data_dir, filename)
                data = np.load(filepath, allow_pickle=True)
                all_features.append(data['features'])
                all_labels.append(data['labels'])

            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            logger.info("[WalkForward] Loaded %d total samples from %d files",
                        len(labels), len(files))
            return features, labels

        except Exception as e:
            logger.error("[WalkForward] Error loading training data: %s", e)
            return None, None

    def schedule_weekly(self) -> bool:
        """
        Check if it's time for weekly retraining.

        Returns True if retraining should happen now, based on:
        1. Enough time has passed since last retrain
        2. We have accumulated enough data

        Returns:
            bool: True if retraining should proceed
        """
        now = time.time()
        interval_seconds = self.config.retrain_interval_hours * 3600

        # Check time since last retrain
        if self._last_retrain_time is not None:
            elapsed = now - self._last_retrain_time
            if elapsed < interval_seconds:
                return False

        # Check minimum data requirement
        if self._accumulated_samples < self.config.validation_window_bars * 2:
            logger.debug("[WalkForward] Not enough data for retrain: %d < %d",
                         self._accumulated_samples,
                         self.config.validation_window_bars * 2)
            return False

        return True

    def retrain_with_validation(self, ensemble, train_fn=None) -> Dict:
        """
        Execute walk-forward retraining with validation split.

        Splits accumulated data into train/val using walk-forward:
        - Train on all data except the last validation_window_bars
        - Validate on the last validation_window_bars
        - Only deploy if improvement exceeds threshold

        Args:
            ensemble: EnsembleManager instance to retrain
            train_fn: Optional custom training function

        Returns:
            Dict with results: deployed, old_score, new_score, improvement
        """
        start_time = time.time()
        result = {
            'timestamp': datetime.now().isoformat(),
            'deployed': False,
            'old_score': None,
            'new_score': None,
            'improvement_pct': None,
            'reason': '',
        }

        logger.info("[WalkForward] Starting walk-forward retraining...")

        # Load accumulated data
        features, labels = self._load_training_data()
        if features is None or len(features) < self.config.validation_window_bars * 2:
            result['reason'] = "Insufficient training data"
            logger.warning("[WalkForward] %s", result['reason'])
            return result

        # Walk-forward split
        val_size = self.config.validation_window_bars
        train_features = features[:-val_size]
        train_labels = labels[:-val_size]
        val_features = features[-val_size:]
        val_labels = labels[-val_size:]

        logger.info("[WalkForward] Split: train=%d, val=%d",
                    len(train_labels), len(val_labels))

        # Evaluate current model on validation set
        try:
            old_prediction = ensemble.predict(val_features)
            old_probs = old_prediction['probabilities']
            old_preds = np.argmax(old_probs, axis=1)
            old_score = float(np.mean(old_preds == val_labels))
            result['old_score'] = old_score
            logger.info("[WalkForward] Current model validation accuracy: %.4f", old_score)
        except Exception as e:
            logger.warning("[WalkForward] Could not evaluate current model: %s", e)
            old_score = 0.0
            result['old_score'] = old_score

        # Check time limit
        elapsed = time.time() - start_time
        max_seconds = self.config.max_retrain_duration_min * 60
        if elapsed > max_seconds * 0.8:
            result['reason'] = "Time limit approaching before training"
            logger.warning("[WalkForward] %s", result['reason'])
            return result

        # Backup current model
        backup_dir = os.path.join(os.path.dirname(MODEL_DIR), "wf_checkpoint_backup")
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(MODEL_DIR, backup_dir)

        # Run training
        try:
            if train_fn:
                train_fn(train_features, train_labels)
            else:
                # Use ensemble's built-in training capabilities
                # Fit tree models on flattened features
                flat_features = train_features.reshape(len(train_features), -1)
                ensemble.fit_gradient_boost(train_features, train_labels)
                ensemble.fit_xgboost(train_features, train_labels)
                ensemble.fit_catboost(train_features, train_labels)
                logger.info("[WalkForward] Retrained tree models on %d samples",
                            len(train_labels))
        except Exception as e:
            result['reason'] = f"Training failed: {e}"
            logger.error("[WalkForward] %s", result['reason'])
            # Rollback
            if os.path.exists(backup_dir):
                if os.path.exists(MODEL_DIR):
                    shutil.rmtree(MODEL_DIR)
                shutil.copytree(backup_dir, MODEL_DIR)
            return result

        # Evaluate new model on validation set
        try:
            new_prediction = ensemble.predict(val_features)
            new_probs = new_prediction['probabilities']
            new_preds = np.argmax(new_probs, axis=1)
            new_score = float(np.mean(new_preds == val_labels))
            result['new_score'] = new_score
            logger.info("[WalkForward] New model validation accuracy: %.4f", new_score)
        except Exception as e:
            result['reason'] = f"Evaluation failed: {e}"
            logger.error("[WalkForward] %s", result['reason'])
            # Rollback
            if os.path.exists(backup_dir):
                if os.path.exists(MODEL_DIR):
                    shutil.rmtree(MODEL_DIR)
                shutil.copytree(backup_dir, MODEL_DIR)
            return result

        # Deploy decision
        result['deployed'] = self.deploy_if_improved(
            new_score, old_score, backup_dir
        )

        if result['deployed']:
            improvement = ((new_score - old_score) / max(old_score, 0.001)) * 100
            result['improvement_pct'] = improvement
            result['reason'] = (
                f"Deployed: {improvement:.2f}% improvement "
                f"({old_score:.4f} -> {new_score:.4f})"
            )
            logger.info("[WalkForward] DEPLOYED: %s", result['reason'])
        else:
            improvement = ((new_score - old_score) / max(old_score, 0.001)) * 100
            result['improvement_pct'] = improvement
            result['reason'] = (
                f"Not deployed: {improvement:.2f}% improvement "
                f"(need {self.config.min_improvement_pct}%)"
            )
            logger.info("[WalkForward] NOT DEPLOYED: %s", result['reason'])

        # Update state
        self._last_retrain_time = time.time()
        self._retrain_history.append(result)
        self._save_state()

        # Cleanup backup
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        return result

    def deploy_if_improved(self, new_score: float, old_score: float,
                           backup_dir: str) -> bool:
        """
        Compare new model against old and deploy only if improved.

        Args:
            new_score: Validation accuracy of new model
            old_score: Validation accuracy of current model
            backup_dir: Path to backup for rollback

        Returns:
            bool: True if new model was deployed
        """
        if old_score <= 0:
            # No valid baseline, deploy new model
            return True

        improvement = ((new_score - old_score) / old_score) * 100

        if improvement >= self.config.min_improvement_pct:
            # New model is better, keep it (already deployed by training)
            return True
        else:
            # Rollback to old model
            if os.path.exists(backup_dir):
                if os.path.exists(MODEL_DIR):
                    shutil.rmtree(MODEL_DIR)
                shutil.copytree(backup_dir, MODEL_DIR)
            return False

    def get_status(self) -> Dict:
        """Get current walk-forward retrainer status."""
        now = time.time()
        next_eligible = None
        if self._last_retrain_time:
            next_time = self._last_retrain_time + (self.config.retrain_interval_hours * 3600)
            next_eligible = datetime.fromtimestamp(next_time).isoformat()

        return {
            'last_retrain': datetime.fromtimestamp(self._last_retrain_time).isoformat()
            if self._last_retrain_time else None,
            'next_eligible': next_eligible,
            'accumulated_samples': self._accumulated_samples,
            'total_retrains': len(self._retrain_history),
            'data_files': len([
                f for f in os.listdir(self._data_dir)
                if f.endswith('.npz')
            ]) if os.path.exists(self._data_dir) else 0,
        }
