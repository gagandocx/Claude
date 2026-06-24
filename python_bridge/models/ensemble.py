"""
=============================================================
  Python ML Bridge - Ensemble Model Manager
  Combines Transformer + LSTM + Gradient Boosting predictions
  using stacking meta-learner with dynamic weight adjustment.
=============================================================
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EnsembleConfig, TransformerConfig, LSTMConfig
from models.transformer_model import MarketTransformer
from models.lstm_model import MarketLSTM


class EnsembleManager:
    """
    Ensemble model that combines multiple predictors with a meta-learner.

    Models:
        1. MarketTransformer - Attention-based pattern recognition
        2. MarketLSTM - Sequential pattern recognition
        3. HistGradientBoostingClassifier - Fast histogram-based ML baseline

    Meta-learner: Logistic Regression on stacked predictions
    Dynamic weight adjustment based on recent accuracy.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None,
                 transformer_config: Optional[TransformerConfig] = None,
                 lstm_config: Optional[LSTMConfig] = None):
        self.config = config or EnsembleConfig()

        # Initialize models
        self.transformer = MarketTransformer(transformer_config or TransformerConfig())
        self.lstm = MarketLSTM(lstm_config or LSTMConfig())
        self.gradient_boost = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

        # Meta-learner (stacking)
        self.meta_learner = LogisticRegression(
            max_iter=1000, random_state=42
        )

        # Dynamic weights
        self.weights = np.array([
            self.config.transformer_weight,
            self.config.lstm_weight,
            self.config.gradient_boost_weight
        ])

        # Performance tracking
        self._prediction_history: deque = deque(maxlen=self.config.weight_lookback)
        self._accuracy_tracker: Dict[str, deque] = {
            "transformer": deque(maxlen=self.config.weight_lookback),
            "lstm": deque(maxlen=self.config.weight_lookback),
            "gradient_boost": deque(maxlen=self.config.weight_lookback),
        }

        # State
        self.gb_fitted = False
        self.meta_fitted = False
        self.models_loaded = False  # True only after successful checkpoint load
        self.device = torch.device("cpu")

    def to_device(self, device: str = "cpu"):
        """Move neural network models to specified device."""
        self.device = torch.device(device)
        self.transformer = self.transformer.to(self.device)
        self.lstm = self.lstm.to(self.device)
        return self

    def predict_transformer(self, x: np.ndarray) -> np.ndarray:
        """
        Get transformer predictions.

        Args:
            x: Input array of shape (batch, seq_len, features)

        Returns:
            Probabilities array of shape (batch, 3)
        """
        self.transformer.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(x).to(self.device)
            probs = self.transformer.predict(tensor)
            return probs.cpu().numpy()

    def predict_lstm(self, x: np.ndarray) -> np.ndarray:
        """
        Get LSTM predictions.

        Args:
            x: Input array of shape (batch, seq_len, features)

        Returns:
            Probabilities array of shape (batch, 3)
        """
        self.lstm.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(x).to(self.device)
            probs = self.lstm.predict(tensor)
            return probs.cpu().numpy()

    def predict_gradient_boost(self, x: np.ndarray) -> np.ndarray:
        """
        Get gradient boosting predictions.

        Args:
            x: Input array of shape (batch, seq_len, features)
               Flattened to 2D for sklearn

        Returns:
            Probabilities array of shape (batch, 3)
        """
        if not self.gb_fitted:
            # Return uniform distribution if not fitted
            batch_size = x.shape[0]
            return np.full((batch_size, 3), 1.0 / 3.0)

        # Flatten sequences for gradient boosting
        x_flat = x.reshape(x.shape[0], -1)
        probs = self.gradient_boost.predict_proba(x_flat)
        return probs

    def fit_gradient_boost(self, X: np.ndarray, y: np.ndarray):
        """
        Train the gradient boosting model.

        Args:
            X: Training features (num_samples, seq_len, features)
            y: Labels (num_samples,)
        """
        X_flat = X.reshape(X.shape[0], -1)
        self.gradient_boost.fit(X_flat, y)
        self.gb_fitted = True

    def fit_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """
        Train the meta-learner on stacked predictions.

        Args:
            X: Stacked predictions from all models (num_samples, 9)
            y: True labels (num_samples,)
        """
        self.meta_learner.fit(X, y)
        self.meta_fitted = True

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get ensemble prediction combining all models.

        Args:
            x: Input array of shape (batch, seq_len, features)

        Returns:
            Dict with 'probabilities' (batch, 3), 'confidence' (batch,),
            'agreement' (batch,), 'individual_preds' dict
        """
        # Get individual predictions
        transformer_probs = self.predict_transformer(x)
        lstm_probs = self.predict_lstm(x)
        gb_probs = self.predict_gradient_boost(x)

        # Stack predictions for meta-learner
        stacked = np.concatenate([transformer_probs, lstm_probs, gb_probs], axis=1)

        # Use meta-learner if fitted, else weighted average
        if self.meta_fitted:
            ensemble_probs = self.meta_learner.predict_proba(stacked)
        else:
            # Weighted average
            ensemble_probs = (
                self.weights[0] * transformer_probs +
                self.weights[1] * lstm_probs +
                self.weights[2] * gb_probs
            )

        # Compute agreement (how much models agree)
        predictions = np.stack([
            np.argmax(transformer_probs, axis=1),
            np.argmax(lstm_probs, axis=1),
            np.argmax(gb_probs, axis=1),
        ], axis=1)

        agreement = np.array([
            np.max(np.bincount(predictions[i], minlength=3)) / 3.0
            for i in range(predictions.shape[0])
        ])

        # Confidence = max probability * agreement
        confidence = np.max(ensemble_probs, axis=1) * agreement

        return {
            "probabilities": ensemble_probs,
            "confidence": confidence,
            "agreement": agreement,
            "individual_preds": {
                "transformer": transformer_probs,
                "lstm": lstm_probs,
                "gradient_boost": gb_probs,
            }
        }

    def update_weights(self, true_label: int, predictions: Dict[str, int]):
        """
        Update dynamic weights based on prediction accuracy.

        Args:
            true_label: The actual outcome (0=SELL, 1=HOLD, 2=BUY)
            predictions: Dict of model_name -> predicted_label
        """
        if not self.config.dynamic_weights:
            return

        for name, pred in predictions.items():
            correct = 1.0 if pred == true_label else 0.0
            self._accuracy_tracker[name].append(correct)

        # Recompute weights based on recent accuracy
        accuracies = []
        for name in ["transformer", "lstm", "gradient_boost"]:
            if len(self._accuracy_tracker[name]) > 0:
                acc = np.mean(list(self._accuracy_tracker[name]))
            else:
                acc = 1.0 / 3.0  # Default
            accuracies.append(acc)

        # Softmax-like weight assignment
        accuracies = np.array(accuracies)
        if accuracies.sum() > 0:
            self.weights = accuracies / accuracies.sum()
        else:
            self.weights = np.array([
                self.config.transformer_weight,
                self.config.lstm_weight,
                self.config.gradient_boost_weight,
            ])

    def get_disagreement_signal(self, x: np.ndarray) -> float:
        """
        Compute model disagreement as uncertainty indicator.

        Higher disagreement = less certainty = smaller position size.

        Args:
            x: Input array

        Returns:
            Disagreement score 0-1 (0 = full agreement, 1 = max disagreement)
        """
        result = self.predict(x)
        return float(1.0 - result["agreement"].mean())

    def save_models(self, path: str):
        """Save all model checkpoints."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.transformer.state_dict(),
                   os.path.join(path, "transformer.pth"))
        torch.save(self.lstm.state_dict(),
                   os.path.join(path, "lstm.pth"))
        # Gradient boost saved via joblib
        import joblib
        if self.gb_fitted:
            joblib.dump(self.gradient_boost,
                        os.path.join(path, "gradient_boost.joblib"))
        if self.meta_fitted:
            joblib.dump(self.meta_learner,
                        os.path.join(path, "meta_learner.joblib"))

    def load_models(self, path: str):
        """Load all model checkpoints."""
        nn_loaded = False

        transformer_path = os.path.join(path, "transformer.pth")
        if os.path.exists(transformer_path):
            self.transformer.load_state_dict(
                torch.load(transformer_path, map_location=self.device,
                           weights_only=True),
                strict=False
            )
            nn_loaded = True

        lstm_path = os.path.join(path, "lstm.pth")
        if os.path.exists(lstm_path):
            self.lstm.load_state_dict(
                torch.load(lstm_path, map_location=self.device,
                           weights_only=True),
                strict=False
            )
            nn_loaded = True

        import joblib
        # NOTE: joblib.load uses pickle internally and can execute arbitrary
        # code. Only load checkpoint files from trusted sources (local training
        # output). If checkpoints are ever sourced from external/shared storage,
        # add integrity verification before loading.
        gb_path = os.path.join(path, "gradient_boost.joblib")
        if os.path.exists(gb_path):
            self.gradient_boost = joblib.load(gb_path)
            self.gb_fitted = True

        meta_path = os.path.join(path, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.meta_fitted = True

        # Only mark as loaded if at least one neural network loaded successfully
        if nn_loaded:
            self.models_loaded = True
