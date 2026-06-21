"""
=============================================================
  Python ML Bridge - Signal Generator
  Main signal generation logic: runs models, combines predictions,
  applies risk filters, regime detection, and outputs final
  trade signals with confidence and risk parameters.
=============================================================
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SignalConfig, DataConfig
from models.ensemble import EnsembleManager
from strategies.risk_manager import RiskManager
from strategies.regime_detector import RegimeDetector, MarketRegime


class TradeSignal:
    """Represents a generated trade signal."""

    def __init__(self, timestamp: str, symbol: str, action: str,
                 confidence: float, sl_pips: float, tp_pips: float,
                 lot_size: float, model_name: str, regime: str):
        self.timestamp = timestamp
        self.symbol = symbol
        self.action = action          # BUY, SELL, HOLD
        self.confidence = confidence
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.lot_size = lot_size
        self.model_name = model_name
        self.regime = regime

    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "lot_size": self.lot_size,
            "model_name": self.model_name,
            "regime": self.regime,
        }

    def to_csv_row(self) -> str:
        """Convert signal to CSV row format."""
        return (
            f"{self.timestamp},{self.symbol},{self.action},"
            f"{self.confidence:.4f},{self.sl_pips:.1f},{self.tp_pips:.1f},"
            f"{self.lot_size:.2f},{self.model_name},{self.regime}"
        )


class SignalGenerator:
    """
    Generates trade signals by combining model predictions with risk filters.

    Pipeline:
        1. Get model predictions from ensemble
        2. Check regime and apply adjustments
        3. Apply risk filters (drawdown, correlation, time)
        4. Compute stop loss and take profit from ATR
        5. Calculate position size using Kelly criterion
        6. Generate final signal if all filters pass
    """

    def __init__(self, signal_config: Optional[SignalConfig] = None,
                 data_config: Optional[DataConfig] = None):
        self.signal_config = signal_config or SignalConfig()
        self.data_config = data_config or DataConfig()

        self.ensemble = EnsembleManager()
        self.risk_manager = RiskManager()
        self.regime_detector = RegimeDetector()

        self._last_signal_time = 0.0
        self._signal_count = 0

    def generate_signal(self, features: np.ndarray,
                        prices: Optional[object] = None,
                        atr: float = 0.0,
                        current_price: float = 0.0,
                        adx_series: Optional[object] = None,
                        vix_level: Optional[float] = None) -> TradeSignal:
        """
        Generate a trade signal from input features.

        Args:
            features: Input features array (1, seq_len, num_features)
            prices: Price DataFrame for regime detection
            atr: Current ATR value
            current_price: Current market price
            adx_series: ADX series for regime detection
            vix_level: Current VIX level

        Returns:
            TradeSignal object
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Default HOLD signal
        hold_signal = TradeSignal(
            timestamp=timestamp,
            symbol=self.data_config.symbol,
            action="HOLD",
            confidence=0.0,
            sl_pips=0.0,
            tp_pips=0.0,
            lot_size=0.0,
            model_name="ensemble",
            regime="ranging"
        )

        # Cooldown check
        elapsed = time.time() - self._last_signal_time
        if elapsed < self.signal_config.cooldown_seconds:
            return hold_signal

        # 1. Get ensemble prediction
        try:
            prediction = self.ensemble.predict(features)
        except Exception as e:
            print(f"[SignalGen] Model prediction error: {e}")
            return hold_signal

        probabilities = prediction["probabilities"][0]  # (3,)
        confidence = float(prediction["confidence"][0])
        agreement = float(prediction["agreement"][0])

        # 2. Detect regime
        import pandas as pd
        regime_info = self.regime_detector.detect_regime(
            prices if prices is not None else pd.DataFrame({"Close": [current_price]}),
            adx=adx_series,
            vix=vix_level
        )
        regime = regime_info["regime"]
        regime_name = regime_info["regime_name"]
        regime_adjustments = self.regime_detector.get_regime_adjustments(regime)

        # 3. Determine action from probabilities
        # 0=SELL, 1=HOLD, 2=BUY
        action_idx = int(np.argmax(probabilities))
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map[action_idx]

        # 4. Apply confidence threshold (adjusted by regime)
        min_confidence = regime_adjustments.get(
            "confidence_threshold", self.signal_config.min_confidence
        )
        if confidence < min_confidence:
            return hold_signal

        # 5. If HOLD, return hold signal
        if action == "HOLD":
            return hold_signal

        # 6. Check model agreement
        if agreement < self.ensemble.config.min_agreement:
            return hold_signal

        # 7. Check risk manager
        if not self.risk_manager.is_trading_allowed():
            return hold_signal

        # 8. Check correlation limits
        if not self.risk_manager.check_correlation_limit(action):
            return hold_signal

        # 9. Calculate SL/TP based on ATR
        if atr <= 0:
            atr = 2.0  # Default ATR for gold

        sl_mult = self.signal_config.atr_sl_multiplier * regime_adjustments.get("sl_mult", 1.0)
        tp_mult = self.signal_config.atr_tp_multiplier * regime_adjustments.get("tp_mult", 1.0)
        levels = self.risk_manager.calculate_sl_tp(
            atr=atr, direction=action,
            current_price=current_price,
            sl_mult=sl_mult, tp_mult=tp_mult
        )

        # 10. Calculate position size
        position_mult = regime_adjustments.get("position_size_mult", 1.0)
        lot_size = self.risk_manager.calculate_position_size(
            confidence=confidence,
            atr=atr,
            win_rate=self.risk_manager.get_win_rate(),
            avg_win_loss_ratio=self.risk_manager.get_avg_win_loss_ratio(),
            regime_mult=position_mult
        )

        if lot_size <= 0:
            return hold_signal

        # 11. Determine which model contributed most
        individual = prediction["individual_preds"]
        model_contributions = {
            "transformer": float(np.max(individual["transformer"][0])),
            "lstm": float(np.max(individual["lstm"][0])),
            "gradient_boost": float(np.max(individual["gradient_boost"][0])),
        }
        top_model = max(model_contributions, key=model_contributions.get)

        # 12. Generate signal
        signal = TradeSignal(
            timestamp=timestamp,
            symbol=self.data_config.symbol,
            action=action,
            confidence=confidence,
            sl_pips=levels["sl_pips"],
            tp_pips=levels["tp_pips"],
            lot_size=lot_size,
            model_name=top_model,
            regime=regime_name
        )

        self._last_signal_time = time.time()
        self._signal_count += 1

        return signal

    def update_from_execution(self, trade_id: str, pnl: float,
                              predicted_action: int, actual_outcome: int):
        """
        Update models based on trade execution results.

        Args:
            trade_id: Unique trade identifier
            pnl: Realized P&L
            predicted_action: What was predicted (0, 1, 2)
            actual_outcome: What actually happened (0, 1, 2)
        """
        # Update risk manager
        self.risk_manager.close_trade(trade_id, pnl)

        # Update ensemble weights
        self.ensemble.update_weights(actual_outcome, {
            "transformer": predicted_action,
            "lstm": predicted_action,
            "gradient_boost": predicted_action,
        })

    @property
    def signal_count(self) -> int:
        """Total signals generated."""
        return self._signal_count
