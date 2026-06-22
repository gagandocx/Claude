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

    Pipeline (Rapid HF Scalper):
        1. Compute short-term momentum direction from last 3-5 M1 candles
        2. Get model predictions from ensemble (used for TIMING only)
        3. Check regime and apply adjustments
        4. Apply risk filters (drawdown, correlation, time)
        5. Compute stop loss and take profit from ATR
        6. Calculate position size using Kelly criterion
        7. Generate final signal if all filters pass

    Direction comes from MOMENTUM, not from model predictions.
    AI confidence is used only for entry timing (confidence gate).
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

    def _analyze_candle_patterns(self, prices_df) -> Dict:
        """
        Analyze the most recent candle(s) for candlestick patterns that indicate
        potential reversals or continuations.

        Detects:
            - Bullish engulfing (previous bearish, current bullish body engulfs previous)
            - Bearish engulfing (previous bullish, current bearish body engulfs previous)
            - Hammer / pin bar (long lower wick = bullish reversal signal)
            - Shooting star (long upper wick = bearish reversal signal)
            - Doji (very small body relative to range = indecision)
            - Strong bullish candle (body > 60% of range)
            - Strong bearish candle (body > 60% of range)
            - Exhaustion candle (body > 2x ATR = likely reversal coming)

        Args:
            prices_df: DataFrame with Open, High, Low, Close columns.
                       Uses iloc[-1] for current candle, iloc[-2] for previous.

        Returns:
            Dict with keys:
                "pattern": str - name of detected pattern
                "bias": str - "bullish", "bearish", or "neutral"
                "block_buy": bool - whether to block BUY signals
                "block_sell": bool - whether to block SELL signals
        """
        import pandas as pd

        result = {
            "pattern": "none",
            "bias": "neutral",
            "block_buy": False,
            "block_sell": False,
        }

        if not isinstance(prices_df, pd.DataFrame):
            return result

        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(prices_df.columns):
            return result

        if len(prices_df) < 2:
            return result

        # Current candle (most recent)
        curr = prices_df.iloc[-1]
        prev = prices_df.iloc[-2]

        curr_open = float(curr["Open"])
        curr_high = float(curr["High"])
        curr_low = float(curr["Low"])
        curr_close = float(curr["Close"])

        prev_open = float(prev["Open"])
        prev_high = float(prev["High"])
        prev_low = float(prev["Low"])
        prev_close = float(prev["Close"])

        # Candle metrics for current candle
        curr_range = curr_high - curr_low
        curr_body = abs(curr_close - curr_open)
        curr_upper_wick = curr_high - max(curr_open, curr_close)
        curr_lower_wick = min(curr_open, curr_close) - curr_low
        curr_is_bullish = curr_close > curr_open
        curr_is_bearish = curr_close < curr_open

        # Previous candle metrics
        prev_body = abs(prev_close - prev_open)
        prev_is_bullish = prev_close > prev_open
        prev_is_bearish = prev_close < prev_open

        # Avoid division by zero - use relative threshold based on price level
        # Gold: price ~4000, min range 0.01 is fine
        # Forex: price ~1.09, min range needs to be much smaller (0.00001)
        min_range = max(curr_close * 0.000001, 0.000001)  # 0.0001% of price or absolute minimum
        if curr_range < min_range:
            # Essentially a zero-range candle (doji-like)
            result["pattern"] = "doji"
            result["bias"] = "neutral"
            result["block_buy"] = True
            result["block_sell"] = True
            logger.info("[CandlePattern] Doji detected (zero range) - blocking all trades")
            return result

        body_ratio = curr_body / curr_range

        # Compute ATR from recent bars for exhaustion detection
        atr_value = 0.0
        if len(prices_df) >= 14:
            recent_ranges = (prices_df["High"].iloc[-14:] - prices_df["Low"].iloc[-14:]).values
            atr_value = float(np.mean(recent_ranges))

        # --- Pattern Detection (priority order) ---

        # 1. Doji: body is < 3% of range (very strict - only true dojis block)
        if body_ratio < 0.03:
            result["pattern"] = "doji"
            result["bias"] = "neutral"
            result["block_buy"] = True
            result["block_sell"] = True
            logger.info("[CandlePattern] Doji detected (body=%.1f%% of range) - "
                        "blocking all trades (indecision)", body_ratio * 100)
            return result

        # 2. Bullish Engulfing: previous bearish, current bullish body engulfs previous body
        if (prev_is_bearish and curr_is_bullish and
                curr_close > prev_open and curr_open < prev_close):
            result["pattern"] = "bullish_engulfing"
            result["bias"] = "bullish"
            result["block_buy"] = False
            result["block_sell"] = True  # Block sells during bullish engulfing
            logger.info("[CandlePattern] Bullish Engulfing detected - "
                        "confirms BUY, blocks SELL")
            return result

        # 3. Bearish Engulfing: previous bullish, current bearish body engulfs previous body
        if (prev_is_bullish and curr_is_bearish and
                curr_close < prev_open and curr_open > prev_close):
            result["pattern"] = "bearish_engulfing"
            result["bias"] = "bearish"
            result["block_buy"] = True  # Block buys during bearish engulfing
            result["block_sell"] = False
            logger.info("[CandlePattern] Bearish Engulfing detected - "
                        "confirms SELL, blocks BUY")
            return result

        # 4. Hammer / Pin Bar: long lower wick (> 60% of range), small upper wick
        if (curr_lower_wick > 0.6 * curr_range and
                curr_upper_wick < 0.2 * curr_range and
                body_ratio < 0.35):
            result["pattern"] = "hammer"
            result["bias"] = "bullish"
            result["block_buy"] = False
            result["block_sell"] = True  # Block sells - hammer signals bullish reversal
            logger.info("[CandlePattern] Hammer/Pin Bar detected (lower wick=%.1f%%) - "
                        "confirms BUY, blocks SELL", (curr_lower_wick / curr_range) * 100)
            return result

        # 5. Shooting Star: long upper wick (> 60% of range), small lower wick
        if (curr_upper_wick > 0.6 * curr_range and
                curr_lower_wick < 0.2 * curr_range and
                body_ratio < 0.35):
            result["pattern"] = "shooting_star"
            result["bias"] = "bearish"
            result["block_buy"] = True  # Block buys - shooting star signals bearish reversal
            result["block_sell"] = False
            logger.info("[CandlePattern] Shooting Star detected (upper wick=%.1f%%) - "
                        "confirms SELL, blocks BUY", (curr_upper_wick / curr_range) * 100)
            return result

        # 6. Exhaustion candle: body > 2x ATR (overextended, reversal likely)
        if atr_value > 0 and curr_body > 2.0 * atr_value:
            if curr_is_bullish:
                result["pattern"] = "exhaustion_bullish"
                result["bias"] = "bearish"  # Reversal expected
                result["block_buy"] = True  # Don't chase the exhaustion move
                result["block_sell"] = False
                logger.info("[CandlePattern] Bullish Exhaustion candle (body=%.2f > 2xATR=%.2f) - "
                            "reversal likely, blocks BUY", curr_body, 2.0 * atr_value)
            else:
                result["pattern"] = "exhaustion_bearish"
                result["bias"] = "bullish"  # Reversal expected
                result["block_buy"] = False
                result["block_sell"] = True  # Don't chase the exhaustion move
                logger.info("[CandlePattern] Bearish Exhaustion candle (body=%.2f > 2xATR=%.2f) - "
                            "reversal likely, blocks SELL", curr_body, 2.0 * atr_value)
            return result

        # 7. Strong bullish candle: body > 60% of range, bullish
        if body_ratio > 0.60 and curr_is_bullish:
            result["pattern"] = "strong_bullish"
            result["bias"] = "bullish"
            result["block_buy"] = False
            result["block_sell"] = True  # Don't sell against a strong bullish candle
            logger.info("[CandlePattern] Strong Bullish candle (body=%.1f%% of range) - "
                        "confirms BUY, blocks SELL", body_ratio * 100)
            return result

        # 8. Strong bearish candle: body > 60% of range, bearish
        if body_ratio > 0.60 and curr_is_bearish:
            result["pattern"] = "strong_bearish"
            result["bias"] = "bearish"
            result["block_buy"] = True  # Don't buy against a strong bearish candle
            result["block_sell"] = False
            logger.info("[CandlePattern] Strong Bearish candle (body=%.1f%% of range) - "
                        "confirms SELL, blocks BUY", body_ratio * 100)
            return result

        # No significant pattern detected - neutral, no blocking
        logger.info("[CandlePattern] No significant pattern - neutral (body=%.1f%% of range)",
                    body_ratio * 100)
        return result

    def _compute_momentum_direction(self, prices) -> str:
        """
        Compute short-term momentum direction from the last few M1 candles.

        Uses a 3-bar comparison (close[-1] vs close[-4]) to determine the
        direction of short-term momentum. Requires at least 5 bars of data
        to ensure the reference price is meaningful.

        Args:
            prices: DataFrame with 'Close' column or array-like of close prices

        Returns:
            "BUY" if price rose over last 3 bars (close[-1] > close[-4] by > $0.50)
            "SELL" if price fell over last 3 bars (close[-1] < close[-4] by > $0.50)
            "FLAT" if movement is below threshold ($0.50 for gold)
        """
        import pandas as pd

        # Extract close prices
        if isinstance(prices, pd.DataFrame):
            if "Close" not in prices.columns:
                return "FLAT"
            close = prices["Close"].values
        elif isinstance(prices, np.ndarray):
            close = prices
        else:
            return "FLAT"

        if len(close) < 5:
            return "FLAT"

        # Compare current close to close 3 bars ago (fixed 3-bar momentum)
        current_close = close[-1]
        reference_close = close[-4]

        diff = current_close - reference_close
        threshold = 0.5  # $0.50 minimum move for gold (5 pips)

        if abs(diff) < threshold:
            return "FLAT"
        elif diff > 0:
            return "BUY"
        else:
            return "SELL"

    def generate_signal(self, features: np.ndarray,
                        prices: Optional[object] = None,
                        atr: float = 0.0,
                        current_price: float = 0.0,
                        adx_series: Optional[object] = None,
                        vix_level: Optional[float] = None,
                        htf_bias: Optional[Dict] = None,
                        cross_pair_info: Optional[Dict] = None) -> TradeSignal:
        """
        Generate a trade signal using momentum for DIRECTION and AI for TIMING.

        Rapid HF Scalper approach:
        - Direction is determined by short-term momentum (last 3-5 M1 candles)
        - AI model confidence is used as a timing gate (only enter when confident)
        - HTF bias is reduced (only penalize if H4 magnitude > 0.8, and only by 0.05)
        - Targets: SL ~$3 (30 pips), TP ~$0.50 (5 pips) for 85%+ win rate

        Args:
            features: Input features array (1, seq_len, num_features)
            prices: Price DataFrame for regime detection and momentum calculation
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

        # 1. Compute momentum direction from recent price action
        momentum_direction = self._compute_momentum_direction(prices)
        logger.info("[SignalGen] Momentum direction: %s", momentum_direction)

        # If momentum is FLAT, do not trade (no clear short-term trend)
        if momentum_direction == "FLAT":
            logger.info("[SignalGen] HOLD - momentum is FLAT (no clear direction)")
            return hold_signal

        # 1b. Analyze candlestick patterns to detect pullbacks / reversals
        candle_info = self._analyze_candle_patterns(prices)
        candle_pattern = candle_info["pattern"]
        candle_bias = candle_info["bias"]

        # Block trades that go against the candle structure
        if momentum_direction == "BUY" and candle_info["block_buy"]:
            logger.info("[SignalGen] HOLD - momentum BUY blocked by candle pattern '%s' "
                        "(bias=%s). Pullback/reversal detected!",
                        candle_pattern, candle_bias)
            return hold_signal

        if momentum_direction == "SELL" and candle_info["block_sell"]:
            logger.info("[SignalGen] HOLD - momentum SELL blocked by candle pattern '%s' "
                        "(bias=%s). Pullback/reversal detected!",
                        candle_pattern, candle_bias)
            return hold_signal

        logger.info("[SignalGen] Candle pattern: '%s' (bias=%s) - %s",
                    candle_pattern, candle_bias,
                    "CONFIRMS momentum" if candle_bias != "neutral" else "neutral/no block")

        # 2. Get ensemble prediction for TIMING (confidence gate)
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

        # 3. Detect regime
        import pandas as pd
        regime_info = self.regime_detector.detect_regime(
            prices if prices is not None else pd.DataFrame({"Close": [current_price]}),
            adx=adx_series,
            vix=vix_level
        )
        regime = regime_info["regime"]
        regime_name = regime_info["regime_name"]
        regime_adjustments = self.regime_detector.get_regime_adjustments(regime)

        # 4. Use momentum for DIRECTION, AI confidence for TIMING
        # The action comes from momentum, not from model probabilities
        action = momentum_direction  # BUY or SELL from momentum

        # Use overall model confidence as the timing gate
        # Higher confidence = better timing for entry
        # We take the max of buy_prob and sell_prob as base confidence
        sell_prob = float(probabilities[0])
        buy_prob = float(probabilities[2])
        timing_confidence = max(buy_prob, sell_prob, confidence)

        logger.info("[SignalGen] Momentum action: %s, Timing confidence: %.4f",
                    action, timing_confidence)

        # 5. HTF bias - REDUCED penalty for scalper
        # Only penalize if H4 is STRONGLY against momentum (magnitude > 0.8)
        # and only reduce confidence by 0.05 instead of blocking
        htf_confidence_adj = 0.0
        if htf_bias:
            h4_bias = htf_bias.get("4h", 0.0)
            if action == "BUY" and h4_bias < -0.8:
                htf_confidence_adj = -0.05
                logger.info("[SignalGen] HTF: Slight penalty for BUY against strong H4 bearish (%.2f)", h4_bias)
            elif action == "SELL" and h4_bias > 0.8:
                htf_confidence_adj = -0.05
                logger.info("[SignalGen] HTF: Slight penalty for SELL against strong H4 bullish (%.2f)", h4_bias)

        # Apply HTF adjustment
        timing_confidence = timing_confidence + htf_confidence_adj
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 6. Apply confidence threshold (timing gate)
        min_confidence = regime_adjustments.get(
            "confidence_threshold", self.signal_config.min_confidence
        )
        if timing_confidence < min_confidence:
            logger.info("[SignalGen] HOLD - timing confidence %.4f < threshold %.4f (regime: %s)",
                        timing_confidence, min_confidence, regime_name)
            return hold_signal

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
            atr = 3.0  # Default ATR for gold M1 scalping

        # Cap ATR to prevent H1 ATR values inflating M1 scalp SL/TP
        atr = min(atr, 5.0)

        sl_mult = self.signal_config.atr_sl_multiplier * regime_adjustments.get("sl_mult", 1.0)
        tp_mult = self.signal_config.atr_tp_multiplier * regime_adjustments.get("tp_mult", 1.0)
        levels = self.risk_manager.calculate_sl_tp(
            atr=atr, direction=action,
            current_price=current_price,
            sl_mult=sl_mult, tp_mult=tp_mult
        )

        # 9b. Dynamic trailing mode: if tp_pips == 0 (atr_tp_multiplier=0),
        # set tp_pips=9999 to signal the EA to manage exit dynamically (no fixed TP)
        if levels["tp_pips"] == 0:
            levels["tp_pips"] = 9999

        # 10. Calculate position size
        position_mult = regime_adjustments.get("position_size_mult", 1.0)
        lot_size = self.risk_manager.calculate_position_size(
            confidence=timing_confidence,
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
            confidence=timing_confidence,
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
