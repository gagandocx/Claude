"""
=============================================================
  Walk-Forward Validation Harness for 17-Model Ensemble

  Splits historical data into train/validate windows.
  For each window:
    - Train models on the training portion
    - Predict on out-of-sample (validation)
    - Measure accuracy metrics (accuracy, precision, recall, F1)
  Compare ensemble vs individual models vs naive baseline (always BUY).
  Output summary report as JSON + console log.

  This is the gold standard for avoiding overfitting in time-series ML.
  Professional quant funds require walk-forward results before deploying
  any model update.
=============================================================
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    # Window settings
    train_window_bars: int = 5000         # Bars in each training window
    validate_window_bars: int = 1000      # Bars in each validation window
    step_size_bars: int = 500             # How far to slide the window each step
    min_windows: int = 3                  # Minimum validation windows required

    # Model settings
    num_models: int = 17
    model_names: List[str] = field(default_factory=lambda: [
        "transformer", "lstm", "tcn",
        "patch_tst", "tft", "nhits",
        "itransformer", "mamba", "dlinear",
        "xlstm", "timesnet",
        "chronos", "timemixer", "softs",
        "gradient_boost", "xgboost", "catboost",
    ])

    # Output
    output_dir: str = "validation/results"
    report_file: str = "walk_forward_report.json"


@dataclass
class WindowResult:
    """Result for a single validation window."""
    window_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    # Per-model accuracy on this window
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    # Ensemble accuracy
    ensemble_accuracy: float = 0.0
    # Naive baseline (always BUY = class 0)
    baseline_accuracy: float = 0.0
    # Additional metrics
    ensemble_precision: float = 0.0
    ensemble_recall: float = 0.0
    ensemble_f1: float = 0.0
    n_samples: int = 0


class WalkForwardValidator:
    """
    Walk-forward validation harness for the 17-model ensemble.

    Simulates real-world deployment: train on past data, predict on
    unseen future data, measure performance honestly (no lookahead bias).
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.results: List[WindowResult] = []
        self._report: Optional[Dict] = None

    def run(self, features: np.ndarray, labels: np.ndarray,
            model_predict_fn=None) -> Dict:
        """
        Execute walk-forward validation.

        Args:
            features: Full feature array, shape (n_samples, seq_len, n_features)
                     or (n_samples, n_features) for tree models.
            labels: True labels array, shape (n_samples,) with values 0/1/2.
            model_predict_fn: Optional callable(train_X, train_y, val_X) -> Dict
                             that returns {model_name: predictions_array}.
                             If None, uses a simple k-NN baseline for demonstration.

        Returns:
            Summary report dict.
        """
        n_samples = len(features)
        cfg = self.config
        logger.info(
            f"[WalkForward] Starting validation: "
            f"{n_samples} samples, train={cfg.train_window_bars}, "
            f"val={cfg.validate_window_bars}, step={cfg.step_size_bars}"
        )

        # Calculate number of windows
        total_needed = cfg.train_window_bars + cfg.validate_window_bars
        if n_samples < total_needed:
            logger.warning(
                f"[WalkForward] Not enough data: {n_samples} < "
                f"{total_needed} (train + val). Reducing window sizes."
            )
            # Adjust proportionally
            cfg.train_window_bars = int(n_samples * 0.7)
            cfg.validate_window_bars = int(n_samples * 0.2)
            cfg.step_size_bars = max(cfg.validate_window_bars // 2, 1)

        # Generate windows
        windows = self._generate_windows(n_samples)
        if len(windows) < 1:
            logger.warning("[WalkForward] No valid windows. Aborting.")
            return {"error": "insufficient_data", "n_samples": n_samples}

        logger.info(f"[WalkForward] Generated {len(windows)} validation windows")

        # Run validation for each window
        self.results = []
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            logger.info(
                f"[WalkForward] Window {i+1}/{len(windows)}: "
                f"train[{train_start}:{train_end}] "
                f"val[{val_start}:{val_end}]"
            )

            train_X = features[train_start:train_end]
            train_y = labels[train_start:train_end]
            val_X = features[val_start:val_end]
            val_y = labels[val_start:val_end]

            if len(val_y) == 0:
                continue

            # Get predictions from all models
            if model_predict_fn:
                predictions = model_predict_fn(train_X, train_y, val_X)
            else:
                predictions = self._default_predict(train_X, train_y, val_X)

            # Calculate metrics
            result = self._evaluate_window(
                i, train_start, train_end, val_start, val_end,
                predictions, val_y
            )
            self.results.append(result)

        # Generate report
        self._report = self._generate_report()
        self._log_report()
        self._save_report()

        return self._report

    def _generate_windows(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        """Generate (train_start, train_end, val_start, val_end) tuples."""
        cfg = self.config
        windows = []
        start = 0

        while True:
            train_start = start
            train_end = start + cfg.train_window_bars
            val_start = train_end
            val_end = val_start + cfg.validate_window_bars

            if val_end > n_samples:
                break

            windows.append((train_start, train_end, val_start, val_end))
            start += cfg.step_size_bars

        return windows

    def _default_predict(self, train_X: np.ndarray, train_y: np.ndarray,
                         val_X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Default prediction function using simple models when no external
        model_predict_fn is provided.

        Uses:
        - Majority class from training set (as simplest baseline per "model")
        - Random predictions (to simulate diversity)
        - Nearest-neighbor for a slightly better baseline
        """
        n_val = len(val_X)
        predictions = {}

        # Flatten features for simple models
        if train_X.ndim == 3:
            train_flat = train_X.reshape(len(train_X), -1)
            val_flat = val_X.reshape(n_val, -1)
        else:
            train_flat = train_X
            val_flat = val_X

        # Class distribution from training
        unique, counts = np.unique(train_y, return_counts=True)
        majority_class = unique[np.argmax(counts)]

        for model_name in self.config.model_names:
            # Each "model" gets slightly different predictions for diversity
            np.random.seed(hash(model_name) % (2**31))

            if "gradient_boost" in model_name or "xgboost" in model_name or "catboost" in model_name:
                # Tree models: use simple histogram-based prediction
                predictions[model_name] = self._histogram_predict(
                    train_flat, train_y, val_flat
                )
            else:
                # Neural models: use nearest neighbor as proxy
                predictions[model_name] = self._knn_predict(
                    train_flat, train_y, val_flat, k=5
                )

        return predictions

    def _histogram_predict(self, train_X: np.ndarray, train_y: np.ndarray,
                           val_X: np.ndarray) -> np.ndarray:
        """Simple histogram-based classification (bins features)."""
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            # Use a small subset for speed
            max_train = min(len(train_X), 2000)
            idx = np.random.choice(len(train_X), max_train, replace=False)
            # Limit features to prevent memory issues
            max_feats = min(train_X.shape[1], 50)
            clf = HistGradientBoostingClassifier(
                max_iter=50, max_depth=4, random_state=42
            )
            clf.fit(train_X[idx, :max_feats], train_y[idx])
            return clf.predict(val_X[:, :max_feats])
        except Exception:
            # Fallback: majority class
            unique, counts = np.unique(train_y, return_counts=True)
            return np.full(len(val_X), unique[np.argmax(counts)])

    def _knn_predict(self, train_X: np.ndarray, train_y: np.ndarray,
                     val_X: np.ndarray, k: int = 5) -> np.ndarray:
        """K-nearest neighbors prediction (simplified)."""
        # Subsample training for speed
        max_train = min(len(train_X), 1000)
        idx = np.random.choice(len(train_X), max_train, replace=False)
        train_sub = train_X[idx]
        labels_sub = train_y[idx]

        # Limit features
        max_feats = min(train_sub.shape[1], 50)
        train_sub = train_sub[:, :max_feats]
        val_sub = val_X[:, :max_feats]

        predictions = np.zeros(len(val_sub), dtype=int)
        for i in range(len(val_sub)):
            # Euclidean distance to all training points
            dists = np.sqrt(np.sum((train_sub - val_sub[i]) ** 2, axis=1))
            nearest_idx = np.argsort(dists)[:k]
            nearest_labels = labels_sub[nearest_idx]
            # Majority vote
            counts = np.bincount(nearest_labels.astype(int), minlength=3)
            predictions[i] = np.argmax(counts)

        return predictions

    def _evaluate_window(self, window_idx: int,
                         train_start: int, train_end: int,
                         val_start: int, val_end: int,
                         predictions: Dict[str, np.ndarray],
                         val_y: np.ndarray) -> WindowResult:
        """Evaluate all models + ensemble on a single validation window."""
        n_samples = len(val_y)

        # Per-model accuracy
        model_accuracies = {}
        for model_name, preds in predictions.items():
            if len(preds) == n_samples:
                acc = float(np.mean(preds == val_y))
                model_accuracies[model_name] = acc

        # Ensemble: majority vote across all models
        if predictions:
            all_preds = np.array(list(predictions.values()))
            ensemble_preds = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                votes = all_preds[:, i].astype(int)
                counts = np.bincount(votes, minlength=3)
                ensemble_preds[i] = np.argmax(counts)

            ensemble_accuracy = float(np.mean(ensemble_preds == val_y))

            # Precision, recall, F1 (macro-averaged)
            precision, recall, f1 = self._compute_metrics(
                ensemble_preds, val_y
            )
        else:
            ensemble_accuracy = 0.0
            precision = recall = f1 = 0.0
            ensemble_preds = np.zeros(n_samples, dtype=int)

        # Naive baseline: always predict BUY (class 0)
        baseline_preds = np.zeros(n_samples, dtype=int)
        baseline_accuracy = float(np.mean(baseline_preds == val_y))

        return WindowResult(
            window_idx=window_idx,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            model_accuracies=model_accuracies,
            ensemble_accuracy=ensemble_accuracy,
            baseline_accuracy=baseline_accuracy,
            ensemble_precision=precision,
            ensemble_recall=recall,
            ensemble_f1=f1,
            n_samples=n_samples,
        )

    def _compute_metrics(self, predictions: np.ndarray,
                         labels: np.ndarray) -> Tuple[float, float, float]:
        """Compute macro-averaged precision, recall, F1."""
        classes = np.unique(np.concatenate([predictions, labels]))
        precisions = []
        recalls = []

        for cls in classes:
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0 else 0.0
        )

        return avg_precision, avg_recall, f1

    def _generate_report(self) -> Dict:
        """Generate the final walk-forward validation report."""
        if not self.results:
            return {"error": "no_results", "windows": 0}

        # Aggregate metrics across all windows
        ensemble_accs = [r.ensemble_accuracy for r in self.results]
        baseline_accs = [r.baseline_accuracy for r in self.results]

        # Per-model average accuracy across windows
        model_avg_accs = {}
        for model_name in self.config.model_names:
            accs = [
                r.model_accuracies.get(model_name, 0.0)
                for r in self.results
            ]
            model_avg_accs[model_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
            }

        # Rank models by average accuracy
        model_ranking = sorted(
            model_avg_accs.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "train_window": self.config.train_window_bars,
                "validate_window": self.config.validate_window_bars,
                "step_size": self.config.step_size_bars,
                "num_windows": len(self.results),
            },
            "summary": {
                "ensemble_accuracy_mean": float(np.mean(ensemble_accs)),
                "ensemble_accuracy_std": float(np.std(ensemble_accs)),
                "baseline_accuracy_mean": float(np.mean(baseline_accs)),
                "ensemble_vs_baseline": float(
                    np.mean(ensemble_accs) - np.mean(baseline_accs)
                ),
                "ensemble_precision": float(
                    np.mean([r.ensemble_precision for r in self.results])
                ),
                "ensemble_recall": float(
                    np.mean([r.ensemble_recall for r in self.results])
                ),
                "ensemble_f1": float(
                    np.mean([r.ensemble_f1 for r in self.results])
                ),
            },
            "model_ranking": [
                {"model": name, **stats}
                for name, stats in model_ranking
            ],
            "per_window": [
                {
                    "window": r.window_idx,
                    "ensemble_acc": r.ensemble_accuracy,
                    "baseline_acc": r.baseline_accuracy,
                    "n_samples": r.n_samples,
                    "best_model": max(
                        r.model_accuracies.items(),
                        key=lambda x: x[1]
                    )[0] if r.model_accuracies else "none",
                }
                for r in self.results
            ],
        }

        return report

    def _log_report(self):
        """Log the report summary to console."""
        if not self._report or "error" in self._report:
            logger.warning("[WalkForward] No valid report to log")
            return

        summary = self._report["summary"]
        logger.info("=" * 60)
        logger.info("[WalkForward] VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(
            f"  Windows: {self._report['config']['num_windows']}"
        )
        logger.info(
            f"  Ensemble Accuracy: "
            f"{summary['ensemble_accuracy_mean']:.4f} "
            f"(+/- {summary['ensemble_accuracy_std']:.4f})"
        )
        logger.info(
            f"  Baseline Accuracy: "
            f"{summary['baseline_accuracy_mean']:.4f}"
        )
        logger.info(
            f"  Ensemble vs Baseline: "
            f"{summary['ensemble_vs_baseline']:+.4f}"
        )
        logger.info(
            f"  Ensemble F1: {summary['ensemble_f1']:.4f}"
        )
        logger.info("-" * 60)
        logger.info("  Top 5 Models:")
        for item in self._report["model_ranking"][:5]:
            logger.info(
                f"    {item['model']:20s} "
                f"acc={item['mean']:.4f} (std={item['std']:.4f})"
            )
        logger.info("=" * 60)

    def _save_report(self):
        """Save the report as JSON."""
        if not self._report:
            return

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, self.config.report_file)
        try:
            with open(output_path, "w") as f:
                json.dump(self._report, f, indent=2)
            logger.info(f"[WalkForward] Report saved to: {output_path}")
        except Exception as e:
            logger.warning(f"[WalkForward] Could not save report: {e}")

    def get_report(self) -> Optional[Dict]:
        """Get the generated report (None if not yet run)."""
        return self._report
