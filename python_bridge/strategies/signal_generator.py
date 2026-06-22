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
import logging
from datetime import datetime
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SignalConfig, DataConfig, RLConfig, SmartExitConfig, MODEL_DIR
from models.ensemble import EnsembleManager
from models.rl_agent import RLAgent, PositionState, ExitAction
from strategies.risk_manager import RiskManager
from strategies.regime_detector import RegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


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

        # RL agent for exit management feedback loop
        self._rl_agent = RLAgent(RLConfig())
        # Track open positions for RL agent state
        self._open_positions: Dict[str, Dict] = {}

        # Optional performance tracker reference (set from main.py)
        self._performance_tracker = None

        self._last_signal_time = 0.0
        self._signal_count = 0

    def generate_signal(self, features: np.ndarray,
                        prices: Optional[object] = None,
                        atr: float = 0.0,
                        current_price: float = 0.0,
                        adx_series: Optional[object] = None,
                        vix_level: Optional[float] = None,
                        htf_bias: Optional[Dict] = None,
                        cross_pair_info: Optional[Dict] = None) -> TradeSignal:
        """
        Generate a trade signal from input features.

        Professional signal generation with multi-timeframe confirmation
        and cross-pair context:
        - HTF bias filters: Only take longs if H1/H4 bias is bullish
        - USD strength: Adjusts confidence for gold (inverse correlation)
        - Model input shape remains fixed (46 features); auxiliary data
          is used for filtering and confidence scaling, not model input.

        Args:
            features: Input features array (1, seq_len, num_features)
            prices: Price DataFrame for regime detection
            atr: Current ATR value
            current_price: Current market price
            adx_series: ADX series for regime detection
            vix_level: Current VIX level
            htf_bias: Higher timeframe trend bias dict (e.g. {'1h': 0.7, '4h': 0.3})
            cross_pair_info: Cross-pair context (e.g. {'usd_strength': 0.5})

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
            logger.info("[SignalGen] HOLD - cooldown active (%.1fs remaining)",
                        self.signal_config.cooldown_seconds - elapsed)
            return hold_signal

        # Guard: refuse to generate non-HOLD signals if models are not loaded
        # Fallback: if checkpoint files exist on disk, allow signal generation
        if not self.ensemble.models_loaded:
            checkpoint_dir = MODEL_DIR
            has_checkpoints = os.path.isdir(checkpoint_dir) and any(
                f.endswith('.pt') or f.endswith('.pth') or f.endswith('.pkl')
                for f in os.listdir(checkpoint_dir)
            ) if os.path.isdir(checkpoint_dir) else False

            if has_checkpoints:
                logger.info("[SignalGen] models_loaded=False but checkpoint files found - "
                            "allowing signal generation")
            else:
                logger.info("[SignalGen] HOLD - models not loaded and no checkpoints found")
                return hold_signal

        # 1. Get ensemble prediction
        try:
            prediction = self.ensemble.predict(features)
        except Exception as e:
            logger.error("[SignalGen] Model prediction error: %s", e)
            return hold_signal

        probabilities = prediction["probabilities"][0]  # (3,)
        confidence = float(prediction["confidence"][0])
        agreement = float(prediction["agreement"][0])

        logger.info("[SignalGen] Prediction probs: SELL=%.4f HOLD=%.4f BUY=%.4f",
                    probabilities[0], probabilities[1], probabilities[2])
        logger.info("[SignalGen] Confidence=%.4f, Agreement=%.4f", confidence, agreement)

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

        # 3. Determine action from probabilities - HOLD bias fix
        # Instead of argmax across all 3 classes (which always picks HOLD due to
        # imbalanced training data), only compare BUY vs SELL probabilities.
        # 0=SELL, 1=HOLD, 2=BUY
        sell_prob = float(probabilities[0])
        buy_prob = float(probabilities[2])

        # Only HOLD if both BUY and SELL are extremely weak
        if buy_prob < 0.10 and sell_prob < 0.10:
            action = "HOLD"
            confidence = 0.0
        elif buy_prob >= sell_prob:
            action = "BUY"
            confidence = buy_prob
        else:
            action = "SELL"
            confidence = sell_prob

        logger.info("[SignalGen] Action decision: %s (buy_prob=%.4f, sell_prob=%.4f, conf=%.4f), regime=%s",
                    action, buy_prob, sell_prob, confidence, regime_name)

        # 4a. PROFESSIONAL SIGNAL FILTERING: HTF trend confirmation
        # Use higher timeframe bias to filter counter-trend trades and
        # boost confidence for trend-aligned signals. This is how
        # professional prop traders use multi-timeframe analysis.
        htf_confidence_adj = 0.0
        if htf_bias and action != "HOLD":
            h1_bias = htf_bias.get("1h", 0.0)
            h4_bias = htf_bias.get("4h", 0.0)
            # Combined HTF score: H4 weighted more (institutional timeframe)
            htf_score = h1_bias * 0.3 + h4_bias * 0.7

            if action == "BUY" and htf_score < -0.3:
                # Buying against strong bearish HTF trend: penalize confidence
                htf_confidence_adj = -0.10
                logger.info("[SignalGen] HTF FILTER: Penalizing BUY (HTF bearish, score=%.2f)", htf_score)
            elif action == "SELL" and htf_score > 0.3:
                # Selling against strong bullish HTF trend: penalize confidence
                htf_confidence_adj = -0.10
                logger.info("[SignalGen] HTF FILTER: Penalizing SELL (HTF bullish, score=%.2f)", htf_score)
            elif (action == "BUY" and htf_score > 0.3) or (action == "SELL" and htf_score < -0.3):
                # Signal aligned with HTF trend: boost confidence
                htf_confidence_adj = 0.05
                logger.info("[SignalGen] HTF CONFIRM: Signal aligned with HTF trend (score=%.2f)", htf_score)

        # 4b. PROFESSIONAL SIGNAL FILTERING: Cross-pair USD strength context
        # Gold is inversely correlated with USD. Strong USD = bearish gold,
        # Weak USD = bullish gold. Adjust confidence accordingly.
        xpair_confidence_adj = 0.0
        if cross_pair_info and action != "HOLD":
            usd_strength = cross_pair_info.get("usd_strength", 0.0)
            if action == "BUY" and usd_strength > 0.5:
                # Buying gold while USD is strong: reduce confidence
                xpair_confidence_adj = -0.05
                logger.info("[SignalGen] XPAIR FILTER: Penalizing BUY (USD strong=%.3f)", usd_strength)
            elif action == "SELL" and usd_strength < -0.5:
                # Selling gold while USD is weak: reduce confidence
                xpair_confidence_adj = -0.05
                logger.info("[SignalGen] XPAIR FILTER: Penalizing SELL (USD weak=%.3f)", usd_strength)
            elif (action == "BUY" and usd_strength < -0.3) or (action == "SELL" and usd_strength > 0.3):
                # Signal aligned with USD/gold correlation: boost
                xpair_confidence_adj = 0.03
                logger.info("[SignalGen] XPAIR CONFIRM: Signal aligned with USD (strength=%.3f)", usd_strength)

        # Apply auxiliary feature adjustments to confidence
        confidence = confidence + htf_confidence_adj + xpair_confidence_adj
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # 4. Apply confidence threshold (adjusted by regime)
        min_confidence = regime_adjustments.get(
            "confidence_threshold", self.signal_config.min_confidence
        )
        if confidence < min_confidence:
            logger.info("[SignalGen] HOLD - confidence %.4f < threshold %.4f (regime: %s)",
                        confidence, min_confidence, regime_name)
            return hold_signal

        # 5. If HOLD, return hold signal
        if action == "HOLD":
            logger.info("[SignalGen] HOLD - model predicted HOLD action")
            return hold_signal

        # 6. Model agreement check - BYPASSED for aggressive trading
        # The user wants signals on nearly every candle, so we skip the
        # agreement filter to maximize signal frequency.
        logger.info("[SignalGen] Agreement=%.4f (check bypassed for aggressive mode)",
                    agreement)

        # 7. Check risk manager
        if not self.risk_manager.is_trading_allowed():
            logger.info("[SignalGen] HOLD - risk manager blocked trading (drawdown/daily limit)")
            return hold_signal

        # 8. Check correlation limits
        if not self.risk_manager.check_correlation_limit(action):
            logger.info("[SignalGen] HOLD - correlation limit reached for %s", action)
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
            logger.info("[SignalGen] HOLD - calculated lot_size is 0")
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

        logger.info("[SignalGen] SIGNAL GENERATED: %s %s conf=%.4f lot=%.2f regime=%s",
                    signal.action, signal.symbol, signal.confidence,
                    signal.lot_size, signal.regime)

        return signal

    def set_performance_tracker(self, tracker) -> None:
        """
        Set a reference to the PerformanceTracker for trade result recording.
        Called from main.py during initialization.
        """
        self._performance_tracker = tracker

    def update_from_execution(self, trade_id: str, pnl: float,
                              predicted_action: int, actual_outcome: int):
        """
        Update models based on trade execution results.

        Feeds reward signal to the RL agent for learning position
        management, and updates ensemble weights for prediction quality.

        Note: Performance tracking is handled exclusively by main.py from
        MT5 confirmations to prevent double-counting (Fix #6).

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

        # Feed RL agent with trade outcome (learning only, no perf tracking)
        if trade_id in self._open_positions:
            pos_info = self._open_positions[trade_id]
            # Create terminal state for RL agent
            position = PositionState(
                direction=pos_info.get("direction", 1),
                unrealized_pnl=pnl,
                unrealized_pnl_atr=pnl / max(pos_info.get("atr", 2.0), 0.01),
                hold_bars=pos_info.get("hold_bars", 0),
                entry_price=pos_info.get("entry_price", 0.0),
                current_price=pos_info.get("current_price", 0.0),
                atr=pos_info.get("atr", 2.0),
                confidence=pos_info.get("confidence", 0.5),
                initial_confidence=pos_info.get("initial_confidence", 0.5),
                sl_distance_atr=pos_info.get("sl_distance_atr", 1.5),
                tp_distance_atr=pos_info.get("tp_distance_atr", 2.5),
                max_favorable=max(pnl, pos_info.get("max_favorable", 0.0)),
                max_adverse=min(pnl, pos_info.get("max_adverse", 0.0)),
                partial_closed_pct=pos_info.get("partial_closed_pct", 0.0),
                regime_changed=pos_info.get("regime_changed", False),
                ticket=trade_id
            )
            state = self._rl_agent.state_from_position(position)
            reward = self._rl_agent.compute_reward(
                position, ExitAction.CLOSE_FULL, realized_pnl=pnl, trade_closed=True
            )
            # Terminal transition
            self._rl_agent.store_transition(
                state, ExitAction.CLOSE_FULL, reward, state, done=True
            )
            self._rl_agent.train_step()
            # Remove from tracked positions
            del self._open_positions[trade_id]
            logger.info(f"[SignalGen] RL agent received reward={reward:.4f} "
                        f"for trade {trade_id} (PnL={pnl:.2f})")

    def register_open_position(self, trade_id: str, direction: int,
                               entry_price: float, confidence: float,
                               atr: float, sl_pips: float, tp_pips: float):
        """Register a new open position for RL agent tracking.

        Called when MT5 confirms a trade execution so we can track
        the position state for the RL learning loop.
        """
        self._open_positions[trade_id] = {
            "direction": direction,
            "entry_price": entry_price,
            "current_price": entry_price,
            "confidence": confidence,
            "initial_confidence": confidence,
            "atr": atr,
            "sl_distance_atr": sl_pips * 0.1 / max(atr, 0.01),  # Convert pips to ATR
            "tp_distance_atr": tp_pips * 0.1 / max(atr, 0.01),
            "hold_bars": 0,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "partial_closed_pct": 0.0,
            "regime_changed": False,
            "open_time": datetime.now().isoformat(),
        }

    @property
    def signal_count(self) -> int:
        """Total signals generated."""
        return self._signal_count
