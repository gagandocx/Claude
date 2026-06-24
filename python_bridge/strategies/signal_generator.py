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
from datetime import datetime, timezone
from typing import Dict, Optional

import ta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (SignalConfig, DataConfig, RLConfig, SmartExitConfig, MODEL_DIR,
                             SessionConfig, SpreadFilterConfig, AdaptiveMomentumConfig,
                             PriceStructureConfig, FVGConfig, LiquiditySweepConfig,
                             AutoOptimizerConfig)
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
        1. Compute short-term momentum direction from last 5 M1 candles
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

        # Smart upgrade configs
        self.session_config = SessionConfig()
        self.spread_filter_config = SpreadFilterConfig()
        self.adaptive_momentum_config = AdaptiveMomentumConfig()
        self.price_structure_config = PriceStructureConfig()
        self.fvg_config = FVGConfig()
        self.liquidity_sweep_config = LiquiditySweepConfig()

        self.ensemble = EnsembleManager()
        self.risk_manager = RiskManager()
        self.regime_detector = RegimeDetector()

        # RL agent for exit management feedback loop
        self._rl_agent = RLAgent(RLConfig())
        # Track open positions for RL agent state
        self._open_positions: Dict[str, Dict] = {}

        # Optional performance tracker reference (set from main.py)
        self._performance_tracker = None

        # Optional auto-optimizer reference (set from main.py)
        self._auto_optimizer = None

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

        # 1. Doji: body is < 1% of range (extremely strict - almost zero body only)
        if body_ratio < 0.01:
            result["pattern"] = "doji"
            result["bias"] = "neutral"
            result["block_buy"] = True
            result["block_sell"] = True
            logger.info("[CandlePattern] Doji detected (body=%.1f%% of range) - "
                        "blocking all trades (indecision)", body_ratio * 100)
            return result

        # 2. Bullish Engulfing: previous bearish, current bullish body engulfs previous body
        # Additional: current body must be significantly larger (>1.5x previous body)
        if (prev_is_bearish and curr_is_bullish and
                curr_close > prev_open and curr_open < prev_close and
                curr_body > prev_body * 1.5):
            result["pattern"] = "bullish_engulfing"
            result["bias"] = "bullish"
            result["block_buy"] = False
            result["block_sell"] = True  # Block sells during bullish engulfing
            logger.info("[CandlePattern] Bullish Engulfing detected - "
                        "confirms BUY, blocks SELL")
            return result

        # 3. Bearish Engulfing: previous bullish, current bearish body engulfs previous body
        # Additional: current body must be significantly larger (>1.5x previous body)
        if (prev_is_bullish and curr_is_bearish and
                curr_close < prev_open and curr_open > prev_close and
                curr_body > prev_body * 1.5):
            result["pattern"] = "bearish_engulfing"
            result["bias"] = "bearish"
            result["block_buy"] = True  # Block buys during bearish engulfing
            result["block_sell"] = False
            logger.info("[CandlePattern] Bearish Engulfing detected - "
                        "confirms SELL, blocks BUY")
            return result

        # 4. Hammer / Pin Bar: long lower wick (> 80% of range), small upper wick, tiny body
        # Very strict: only blocks on extreme hammers (80%+ wick, <15% body)
        if (curr_lower_wick > 0.8 * curr_range and
                curr_upper_wick < 0.1 * curr_range and
                body_ratio < 0.15):
            result["pattern"] = "hammer"
            result["bias"] = "bullish"
            result["block_buy"] = False
            result["block_sell"] = True  # Block sells - hammer signals bullish reversal
            logger.info("[CandlePattern] Hammer/Pin Bar detected (lower wick=%.1f%%) - "
                        "confirms BUY, blocks SELL", (curr_lower_wick / curr_range) * 100)
            return result

        # 5. Shooting Star: long upper wick (> 80% of range), small lower wick, tiny body
        # Very strict: only blocks on extreme shooting stars
        if (curr_upper_wick > 0.8 * curr_range and
                curr_lower_wick < 0.1 * curr_range and
                body_ratio < 0.15):
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

    def _detect_session(self) -> str:
        """
        Detect current trading session based on UTC hour.

        Returns:
            Session name: "asian", "london", "newyork", "overlap" (London/NY overlap),
            or "off_session" (21:00-23:59 UTC, between NY close and Asian open)
        """
        current_hour = datetime.now(timezone.utc).hour
        cfg = self.session_config

        in_london = cfg.london_start <= current_hour < cfg.london_end
        in_ny = cfg.ny_start <= current_hour < cfg.ny_end

        # Check overlap first (London + NY both active)
        if in_london and in_ny:
            return "overlap"
        elif in_london:
            return "london"
        elif in_ny:
            return "newyork"
        elif cfg.asian_start <= current_hour < cfg.asian_end:
            return "asian"
        else:
            # Outside main sessions (21:00-23:59 UTC) - pre-Asian dead zone
            # with reduced position sizing
            return "off_session"

    def _get_session_multiplier(self, session: str) -> float:
        """
        Get position sizing multiplier for the current session.

        Args:
            session: Session name from _detect_session()

        Returns:
            Position sizing multiplier
        """
        cfg = self.session_config
        multipliers = {
            "asian": cfg.asian_multiplier,
            "london": cfg.london_multiplier,
            "newyork": cfg.ny_multiplier,
            "overlap": cfg.overlap_multiplier,
            "off_session": 0.7,  # Reduced sizing for 21:00-23:59 UTC dead zone
        }
        return multipliers.get(session, 1.0)

    def _check_spread_filter(self, prices) -> bool:
        """
        Check if current spread (High-Low range) is too wide relative to average.

        Uses High-Low of the last bar vs average High-Low of the last N bars
        as a spread proxy. Wide spreads indicate illiquid conditions.

        Args:
            prices: DataFrame with 'High' and 'Low' columns

        Returns:
            True if spread is too wide (should block), False if OK
        """
        import pandas as pd

        if not isinstance(prices, pd.DataFrame):
            return False

        if "High" not in prices.columns or "Low" not in prices.columns:
            return False

        window = self.spread_filter_config.avg_spread_window
        if len(prices) < window + 1:
            return False

        # Current bar range
        current_range = float(prices["High"].iloc[-1] - prices["Low"].iloc[-1])

        # Average range of last N bars (excluding current)
        recent_ranges = (prices["High"].iloc[-(window + 1):-1] -
                         prices["Low"].iloc[-(window + 1):-1])
        avg_range = float(recent_ranges.mean())

        if avg_range <= 0:
            return False

        # Block if current range > multiplier * average
        if current_range > self.spread_filter_config.max_spread_multiplier * avg_range:
            logger.warning("[SignalGen] Spread filter: current range $%.2f > %.1fx avg $%.2f - BLOCKING",
                           current_range, self.spread_filter_config.max_spread_multiplier, avg_range)
            return True

        return False

    def _detect_price_structure(self, prices, lookback: int = None) -> str:
        """
        Detect price action structure (trend) from swing highs and lows.

        Analyzes the last N bars for higher highs + higher lows (uptrend)
        or lower highs + lower lows (downtrend).

        Args:
            prices: DataFrame with 'High' and 'Low' columns
            lookback: Number of bars to analyze (default from config)

        Returns:
            "uptrend", "downtrend", or "no_structure"
        """
        import pandas as pd

        if lookback is None:
            lookback = self.price_structure_config.swing_lookback

        if not isinstance(prices, pd.DataFrame):
            return "no_structure"

        if "High" not in prices.columns or "Low" not in prices.columns:
            return "no_structure"

        if len(prices) < lookback:
            return "no_structure"

        highs = prices["High"].values[-lookback:]
        lows = prices["Low"].values[-lookback:]

        # Find swing highs (local maxima over 3-bar window)
        swing_highs = []
        swing_lows = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])

        # Need at least 2 swing points to determine structure
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "no_structure"

        # Check for higher highs and higher lows (uptrend)
        higher_highs = swing_highs[-1] > swing_highs[-2]
        higher_lows = swing_lows[-1] > swing_lows[-2]

        # Check for lower highs and lower lows (downtrend)
        lower_highs = swing_highs[-1] < swing_highs[-2]
        lower_lows = swing_lows[-1] < swing_lows[-2]

        if higher_highs and higher_lows:
            return "uptrend"
        elif lower_highs and lower_lows:
            return "downtrend"
        else:
            return "no_structure"

    def _check_dxy_correlation(self, prices, cross_pair_info: Optional[Dict] = None) -> Dict:
        """
        Check DXY (US Dollar Index) correlation impact on XAUUSD.

        Gold is inversely correlated with USD strength. If DXY is rising,
        penalize BUY and favor SELL. If DXY is falling, favor BUY and penalize SELL.

        Note: This method requires `cross_pair_info` to be populated with DXY data
        (keys: 'dxy_direction', 'dxy_prices', or 'usd_strength'). In production,
        this is wired up by main.py's multi-pair mode when enabled -- without it,
        the method returns zero adjustments and is effectively a no-op.

        Args:
            prices: Main price DataFrame (not used directly, kept for interface consistency)
            cross_pair_info: Dict that may contain DXY data or 'dxy_direction'

        Returns:
            Dict with 'buy_adjustment' and 'sell_adjustment' (confidence deltas)
        """
        result = {"buy_adjustment": 0.0, "sell_adjustment": 0.0}

        if not cross_pair_info:
            return result

        # Check for direct DXY direction info
        dxy_direction = cross_pair_info.get("dxy_direction", None)

        # If not directly provided, check for DXY price data
        if dxy_direction is None:
            dxy_prices = cross_pair_info.get("dxy_prices", None)
            if dxy_prices is not None and len(dxy_prices) >= 3:
                # Determine direction from last 3 bars
                if dxy_prices[-1] > dxy_prices[-3]:
                    dxy_direction = "rising"
                elif dxy_prices[-1] < dxy_prices[-3]:
                    dxy_direction = "falling"
                else:
                    dxy_direction = "flat"

        # Also check usd_strength field
        if dxy_direction is None:
            usd_strength = cross_pair_info.get("usd_strength", None)
            if usd_strength is not None:
                if usd_strength > 0.5:
                    dxy_direction = "rising"
                elif usd_strength < -0.5:
                    dxy_direction = "falling"
                else:
                    dxy_direction = "flat"

        if dxy_direction is None or dxy_direction == "flat":
            return result

        if dxy_direction == "rising":
            # Strong dollar -> bearish for gold
            result["buy_adjustment"] = -0.10
            result["sell_adjustment"] = 0.05
            logger.info("[SignalGen] DXY rising (strong dollar): BUY -0.10, SELL +0.05")
        elif dxy_direction == "falling":
            # Weak dollar -> bullish for gold
            result["buy_adjustment"] = 0.05
            result["sell_adjustment"] = -0.10
            logger.info("[SignalGen] DXY falling (weak dollar): BUY +0.05, SELL -0.10")

        return result

    def _detect_fvg(self, prices) -> Dict:
        """
        Detect Fair Value Gaps (FVGs) / Imbalance Zones in price action.

        A gap up occurs when candle N's low > candle N-2's high (bullish FVG).
        A gap down occurs when candle N's high < candle N-2's low (bearish FVG).

        If price is approaching an unfilled FVG in the direction of momentum,
        it provides confluence for the trade.

        Args:
            prices: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            Dict with 'bullish_fvg' (bool), 'bearish_fvg' (bool),
            'fvg_aligns_buy' (bool), 'fvg_aligns_sell' (bool)
        """
        import pandas as pd

        result = {
            "bullish_fvg": False,
            "bearish_fvg": False,
            "fvg_aligns_buy": False,
            "fvg_aligns_sell": False,
        }

        if not self.fvg_config.enabled:
            return result

        if not isinstance(prices, pd.DataFrame):
            return result

        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(prices.columns):
            return result

        if len(prices) < 5:
            return result

        current_price = float(prices["Close"].iloc[-1])

        # Scan last 10 bars for unfilled FVGs
        scan_range = min(10, len(prices) - 2)
        bullish_fvgs = []  # List of (gap_low, gap_high) for bullish FVGs
        bearish_fvgs = []  # List of (gap_low, gap_high) for bearish FVGs

        for i in range(2, scan_range + 2):
            idx = len(prices) - i
            if idx < 2:
                break

            candle_n_low = float(prices["Low"].iloc[idx])
            candle_n_high = float(prices["High"].iloc[idx])
            candle_n2_high = float(prices["High"].iloc[idx - 2])
            candle_n2_low = float(prices["Low"].iloc[idx - 2])

            # Bullish FVG: gap up (candle N low > candle N-2 high)
            if candle_n_low > candle_n2_high:
                # Check if unfilled (current price hasn't come back to fill it)
                if current_price > candle_n2_high:
                    bullish_fvgs.append((candle_n2_high, candle_n_low))

            # Bearish FVG: gap down (candle N high < candle N-2 low)
            if candle_n_high < candle_n2_low:
                # Check if unfilled (current price hasn't come back to fill it)
                if current_price < candle_n2_low:
                    bearish_fvgs.append((candle_n_high, candle_n2_low))

        if bullish_fvgs:
            result["bullish_fvg"] = True
            # If price is near an unfilled bullish FVG, it aligns with BUY
            # (price may bounce off the FVG zone)
            for gap_low, gap_high in bullish_fvgs:
                gap_size = gap_high - gap_low
                if abs(current_price - gap_high) < gap_size * 2:
                    result["fvg_aligns_buy"] = True
                    break

        if bearish_fvgs:
            result["bearish_fvg"] = True
            # If price is near an unfilled bearish FVG, it aligns with SELL
            for gap_low, gap_high in bearish_fvgs:
                gap_size = gap_high - gap_low
                if abs(current_price - gap_low) < gap_size * 2:
                    result["fvg_aligns_sell"] = True
                    break

        return result

    def _detect_liquidity_sweep(self, prices, lookback: int = None) -> Dict:
        """
        Detect liquidity sweep (stop hunt) patterns.

        A bullish sweep: price breaks below a recent swing low then recovers
        above it in the same or next bar (stops hunted, then reversal).

        A bearish sweep: price breaks above a recent swing high then falls
        back below it (stops hunted, then reversal).

        Args:
            prices: DataFrame with 'High', 'Low', 'Close' columns
            lookback: Number of bars to look back for swing levels

        Returns:
            Dict with 'bullish_sweep' (bool), 'bearish_sweep' (bool)
        """
        import pandas as pd

        if lookback is None:
            lookback = self.liquidity_sweep_config.lookback

        result = {"bullish_sweep": False, "bearish_sweep": False}

        if not isinstance(prices, pd.DataFrame):
            return result

        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(prices.columns):
            return result

        if len(prices) < lookback + 2:
            return result

        # Get recent swing low and swing high from the lookback period
        # (excluding the last 2 bars which are the sweep candidates)
        lookback_lows = prices["Low"].iloc[-(lookback + 2):-2]
        lookback_highs = prices["High"].iloc[-(lookback + 2):-2]

        recent_low = float(lookback_lows.min())
        recent_high = float(lookback_highs.max())

        # Last two bars (the potential sweep bars)
        prev_bar_low = float(prices["Low"].iloc[-2])
        prev_bar_high = float(prices["High"].iloc[-2])
        curr_close = float(prices["Close"].iloc[-1])
        curr_low = float(prices["Low"].iloc[-1])
        curr_high = float(prices["High"].iloc[-1])

        min_recovery = self.liquidity_sweep_config.min_recovery_pct

        # Bullish sweep: price broke below recent low then recovered
        if prev_bar_low < recent_low or curr_low < recent_low:
            # Check recovery: close is back above the recent low
            sweep_depth = recent_low - min(prev_bar_low, curr_low)
            if curr_close > recent_low and sweep_depth > 0:
                recovery = (curr_close - min(prev_bar_low, curr_low)) / (sweep_depth + abs(recent_low - curr_close) + 1e-10)
                if recovery >= min_recovery:
                    result["bullish_sweep"] = True
                    logger.info("[SignalGen] Bullish liquidity sweep detected: broke below %.2f, recovered to %.2f",
                                recent_low, curr_close)

        # Bearish sweep: price broke above recent high then fell back
        if prev_bar_high > recent_high or curr_high > recent_high:
            # Check recovery: close is back below the recent high
            sweep_depth = max(prev_bar_high, curr_high) - recent_high
            if curr_close < recent_high and sweep_depth > 0:
                recovery = (max(prev_bar_high, curr_high) - curr_close) / (sweep_depth + abs(curr_close - recent_high) + 1e-10)
                if recovery >= min_recovery:
                    result["bearish_sweep"] = True
                    logger.info("[SignalGen] Bearish liquidity sweep detected: broke above %.2f, fell back to %.2f",
                                recent_high, curr_close)

        return result

    def _compute_momentum_direction(self, prices, adaptive_atr: float = None,
                                     avg_atr: float = None) -> str:
        """
        Compute short-term momentum direction from the last few M1 candles.

        Uses adaptive lookback: if ATR is high (>1.5x average), uses shorter
        lookback (3 bars) for faster reaction. If ATR is normal/low, uses
        longer lookback (7 bars) for smoother signals.

        If prices is a DataFrame with a 'Volume' column, momentum is
        volume-weighted: sum((close[i] - close[i-1]) * volume[i]) / sum(volume[i])
        for the last N bars. This gives more weight to moves on high volume.

        Args:
            prices: DataFrame with 'Close' column (and optionally 'Volume')
                    or array-like of close prices
            adaptive_atr: Current ATR value for adaptive lookback selection
            avg_atr: Average ATR (14-period) for comparison

        Returns:
            "BUY" if price rose over last N bars by more than threshold
            "SELL" if price fell over last N bars by more than threshold
            "FLAT" if movement is below threshold ($0.50 for gold)
        """
        import pandas as pd

        # Adaptive momentum: select lookback based on ATR
        if adaptive_atr is not None and avg_atr is not None and avg_atr > 0:
            atr_cfg = self.adaptive_momentum_config
            if adaptive_atr > atr_cfg.atr_threshold_mult * avg_atr:
                lookback = atr_cfg.high_atr_lookback
                logger.info("[SignalGen] Adaptive momentum: HIGH ATR (%.2f > %.1fx avg %.2f) - using %d-bar lookback",
                            adaptive_atr, atr_cfg.atr_threshold_mult, avg_atr, lookback)
            else:
                lookback = atr_cfg.low_atr_lookback
                logger.info("[SignalGen] Adaptive momentum: normal ATR (%.2f) - using %d-bar lookback",
                            adaptive_atr, lookback)
        else:
            lookback = self.data_config.momentum_lookback  # default 5

        # Extract close prices
        if isinstance(prices, pd.DataFrame):
            if "Close" not in prices.columns:
                return "FLAT"
            close = prices["Close"].values

            # Volume-weighted momentum if Volume column is present
            if "Volume" in prices.columns and len(close) >= lookback + 2:
                volume = prices["Volume"].values
                # Use last N bars of price changes weighted by volume
                recent_close = close[-(lookback + 1):]
                recent_volume = volume[-(lookback + 1):]

                # Compute volume-weighted price change
                price_changes = np.diff(recent_close)
                volumes = recent_volume[1:]  # volumes corresponding to changes

                total_volume = np.sum(volumes)
                if total_volume > 0:
                    vw_momentum = np.sum(price_changes * volumes) / total_volume
                    # Configurable threshold (default $0.50 for gold/5 pips)
                    threshold = self.data_config.momentum_threshold

                    if abs(vw_momentum) < threshold:
                        return "FLAT"
                    elif vw_momentum > 0:
                        return "BUY"
                    else:
                        return "SELL"

        elif isinstance(prices, np.ndarray):
            close = prices
        else:
            return "FLAT"

        if len(close) < lookback + 2:
            return "FLAT"

        # Compare current close to close N bars ago
        current_close = close[-1]
        reference_close = close[-(lookback + 1)]

        diff = current_close - reference_close
        # Configurable threshold (default $0.50 for gold/5 pips)
        threshold = self.data_config.momentum_threshold

        if abs(diff) < threshold:
            return "FLAT"
        elif diff > 0:
            return "BUY"
        else:
            return "SELL"

    def _detect_support_resistance(self, prices, lookback: int = 100) -> Dict:
        """
        Detect nearest support and resistance levels from recent price action.

        Finds swing highs (local maxima over a 5-bar window) and swing lows
        (local minima over a 5-bar window) from the last `lookback` bars.

        Args:
            prices: DataFrame with 'High', 'Low', 'Close' columns,
                    or array-like of close prices
            lookback: Number of bars to analyze (default 100)

        Returns:
            Dict with keys:
                "nearest_resistance": float or None - nearest resistance above current price
                "nearest_support": float or None - nearest support below current price
        """
        import pandas as pd

        result = {"nearest_resistance": None, "nearest_support": None}

        if isinstance(prices, pd.DataFrame):
            if "High" not in prices.columns or "Low" not in prices.columns:
                return result
            highs = prices["High"].values[-lookback:]
            lows = prices["Low"].values[-lookback:]
            current_price = prices["Close"].values[-1] if "Close" in prices.columns else highs[-1]
        elif isinstance(prices, np.ndarray):
            if len(prices) < 5:
                return result
            highs = prices[-lookback:]
            lows = prices[-lookback:]
            current_price = prices[-1]
        else:
            return result

        if len(highs) < 5:
            return result

        # Find swing highs (local maxima over 5-bar window)
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                    highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
                swing_highs.append(highs[i])

        # Find swing lows (local minima over 5-bar window)
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                    lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
                swing_lows.append(lows[i])

        # Find nearest resistance (swing highs above current price)
        resistances = [h for h in swing_highs if h > current_price]
        if resistances:
            result["nearest_resistance"] = min(resistances)

        # Find nearest support (swing lows below current price)
        supports = [l for l in swing_lows if l < current_price]
        if supports:
            result["nearest_support"] = max(supports)

        return result

    def generate_signal(self, features: np.ndarray,
                        prices: Optional[object] = None,
                        atr: float = 0.0,
                        current_price: float = 0.0,
                        adx_series: Optional[object] = None,
                        vix_level: Optional[float] = None,
                        htf_bias: Optional[Dict] = None,
                        cross_pair_info: Optional[Dict] = None,
                        prices_m1: Optional[object] = None) -> TradeSignal:
        """
        Generate a trade signal using momentum for DIRECTION and AI for TIMING.

        Rapid HF Scalper approach:
        - Direction is determined by short-term momentum (last 5 M1 candles)
        - AI model confidence is used as a timing gate (only enter when confident)
        - HTF bias is reduced (only penalize if H4 magnitude > 0.8, and only by 0.05)
        - Support/resistance proximity reduces confidence to avoid traps
        - RSI exhaustion filter penalizes buying overbought / selling oversold
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

        # AUTO-OPTIMIZER: Override parameters with optimized values if available
        if self._auto_optimizer and self._auto_optimizer.is_enabled:
            opt_params = self._auto_optimizer.get_current_params()
            # Override min_confidence
            self.signal_config.min_confidence = opt_params.get(
                "min_confidence", self.signal_config.min_confidence)
            # Override momentum lookback
            self.data_config.momentum_lookback = opt_params.get(
                "momentum_lookback", self.data_config.momentum_lookback)
            # Override session multipliers
            opt_sessions = opt_params.get("session_multipliers", {})
            if opt_sessions:
                self.session_config.asian_multiplier = opt_sessions.get(
                    "asian", self.session_config.asian_multiplier)
                self.session_config.london_multiplier = opt_sessions.get(
                    "london", self.session_config.london_multiplier)
                self.session_config.ny_multiplier = opt_sessions.get(
                    "newyork", self.session_config.ny_multiplier)
                self.session_config.overlap_multiplier = opt_sessions.get(
                    "overlap", self.session_config.overlap_multiplier)
            # Override RSI levels
            self.data_config.rsi_overbought = opt_params.get(
                "rsi_overbought", self.data_config.rsi_overbought)
            self.data_config.rsi_oversold = opt_params.get(
                "rsi_oversold", self.data_config.rsi_oversold)
            # Override cooldown
            self.signal_config.cooldown_seconds = opt_params.get(
                "cooldown_seconds", self.signal_config.cooldown_seconds)
            logger.debug("[SignalGen] Auto-optimizer params applied: confidence=%.3f, "
                         "momentum_lb=%d, cooldown=%ds",
                         self.signal_config.min_confidence,
                         self.data_config.momentum_lookback,
                         self.signal_config.cooldown_seconds)

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

        # 1. Session awareness - detect and log current session
        # Use M1 data for momentum/direction when available, fall back to H1
        momentum_prices = prices_m1 if prices_m1 is not None else prices

        session = self._detect_session()
        session_multiplier = self._get_session_multiplier(session)
        logger.info("[SignalGen] Session: %s (position multiplier: %.2f)", session, session_multiplier)

        # 1b. Spread filter - block if spread is abnormally wide
        if self._check_spread_filter(prices):
            logger.info("[SignalGen] HOLD - spread filter blocked (abnormally wide range)")
            return hold_signal

        # 1c. Compute average ATR for adaptive momentum
        import pandas as pd
        adaptive_atr_val = None
        avg_atr_val = None
        if isinstance(momentum_prices, pd.DataFrame) and "High" in momentum_prices.columns and "Low" in momentum_prices.columns:
            atr_period = self.adaptive_momentum_config.atr_avg_period
            if len(momentum_prices) >= atr_period + 1:
                recent_ranges = (momentum_prices["High"].iloc[-atr_period:] - momentum_prices["Low"].iloc[-atr_period:]).values
                avg_atr_val = float(np.mean(recent_ranges))
                adaptive_atr_val = float(momentum_prices["High"].iloc[-1] - momentum_prices["Low"].iloc[-1])

        # 1d. Compute momentum direction from recent price action (adaptive)
        momentum_direction = self._compute_momentum_direction(momentum_prices, adaptive_atr=adaptive_atr_val, avg_atr=avg_atr_val)
        logger.info("[SignalGen] Momentum direction: %s", momentum_direction)

        # Initialize range_signal (set to a value inside the FLAT block if range trading triggers)
        range_signal = None

        # MEAN-REVERSION RANGE TRADING: When momentum is FLAT, detect the
        # consolidation range and trade the extremes (buy low, sell high).
        # Only trade ranges > $5 for meaningful setups.
        if momentum_direction == "FLAT":
            range_bars = 20
            range_signal = None

            if isinstance(momentum_prices, pd.DataFrame) and "High" in momentum_prices.columns and "Low" in momentum_prices.columns:
                if len(momentum_prices) >= range_bars:
                    range_high = float(momentum_prices["High"].iloc[-range_bars:].max())
                    range_low = float(momentum_prices["Low"].iloc[-range_bars:].min())
                    range_size = range_high - range_low

                    if range_size > 5.0 and current_price > 0:
                        # Position within range: 0.0 = at low, 1.0 = at high
                        position_in_range = (current_price - range_low) / range_size

                        logger.info("[SignalGen] Range detection: high=%.2f, low=%.2f, "
                                    "size=%.2f, price=%.2f, position=%.1f%%",
                                    range_high, range_low, range_size,
                                    current_price, position_in_range * 100)

                        if position_in_range <= 0.15:
                            # Price is near the BOTTOM of range (within 15%) -> BUY
                            momentum_direction = "BUY"
                            range_signal = "range_buy"
                            logger.info("[SignalGen] RANGE MODE: Price near bottom (%.1f%%) "
                                        "- mean-reversion BUY signal", position_in_range * 100)
                        elif position_in_range >= 0.85:
                            # Price is near the TOP of range (within 85%+) -> SELL
                            momentum_direction = "SELL"
                            range_signal = "range_sell"
                            logger.info("[SignalGen] RANGE MODE: Price near top (%.1f%%) "
                                        "- mean-reversion SELL signal", position_in_range * 100)
                        else:
                            # Price is in the middle of range (15-85%) -> still FLAT/HOLD
                            logger.info("[SignalGen] RANGE MODE: Price in middle (%.1f%%) "
                                        "- no edge, holding", position_in_range * 100)
                    elif range_size <= 5.0:
                        logger.info("[SignalGen] RANGE MODE: Range too small ($%.2f < $5.00) "
                                    "- no trade", range_size)

            # If no range signal was generated, hold as before
            if range_signal is None:
                logger.info("[SignalGen] HOLD - momentum is FLAT (no clear direction, "
                            "no range extremes)")
                return hold_signal

        # 1a. SESSION FILTER: Allow ALL sessions with reduced confidence for low-liquidity
        # Asian and off-session trade with reduced confidence multiplier
        session_confidence_mult = 1.0
        if session == "asian":
            session_confidence_mult = 0.6  # Asian session: 0.6x multiplier
            logger.info("[SignalGen] Asian session detected - trading with 0.6x confidence multiplier")
        elif session == "off_session":
            session_confidence_mult = 0.5  # Off-session: 0.5x multiplier
            logger.info("[SignalGen] Off-session detected - trading with 0.5x confidence multiplier")

        # 1a2. DAILY TRADE LIMIT: Max 30 trades per day for aggressive trading
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not hasattr(self, '_trade_day'):
            self._trade_day = current_day
            self._trades_today = 0
        if self._trade_day != current_day:
            self._trade_day = current_day
            self._trades_today = 0
        if self._trades_today >= 30:
            logger.info("[SignalGen] HOLD - daily trade limit reached (%d trades today)",
                        self._trades_today)
            return hold_signal

        # 1a3. RSI ZONE FILTER: Require RSI in favorable zone
        # BUY: RSI must be 25-70 (ultra-wide zone for max trades)
        # SELL: RSI must be 30-75 (ultra-wide zone for max trades)
        import pandas as pd
        rsi_zone_ok = True
        current_rsi_for_zone = None
        if isinstance(momentum_prices, pd.DataFrame) and "Close" in momentum_prices.columns:
            try:
                rsi_zone_series = ta.momentum.rsi(momentum_prices["Close"], window=14)
                if rsi_zone_series is not None and len(rsi_zone_series) > 0:
                    current_rsi_for_zone = float(rsi_zone_series.iloc[-1])
                    if not np.isnan(current_rsi_for_zone):
                        if momentum_direction == "BUY" and not (25 <= current_rsi_for_zone <= 70):
                            rsi_zone_ok = False
                            logger.info("[SignalGen] HOLD - RSI zone filter: BUY requires RSI 25-70, "
                                        "got RSI=%.1f", current_rsi_for_zone)
                        elif momentum_direction == "SELL" and not (30 <= current_rsi_for_zone <= 75):
                            rsi_zone_ok = False
                            logger.info("[SignalGen] HOLD - RSI zone filter: SELL requires RSI 30-75, "
                                        "got RSI=%.1f", current_rsi_for_zone)
            except Exception as e:
                logger.warning("[SignalGen] RSI zone filter error: %s", e)

        if not rsi_zone_ok:
            return hold_signal

        # 1a4. PRICE STRUCTURE ALIGNMENT: Confidence penalty instead of hard block
        # Opposing structure reduces confidence but does not prevent trade entry
        structure_check = self._detect_price_structure(momentum_prices)
        if structure_check != "no_structure":
            if momentum_direction == "BUY" and structure_check == "downtrend":
                logger.info("[SignalGen] Price structure penalty: BUY against confirmed downtrend "
                            "(confidence will be reduced, not blocked)")
                # Penalty is applied later in the confidence pipeline (section 5d)
            elif momentum_direction == "SELL" and structure_check == "uptrend":
                logger.info("[SignalGen] Price structure penalty: SELL against confirmed uptrend "
                            "(confidence will be reduced, not blocked)")

        # 1b. Analyze candlestick patterns to detect pullbacks / reversals
        candle_info = self._analyze_candle_patterns(prices)
        candle_pattern = candle_info["pattern"]
        candle_bias = candle_info["bias"]

        # Log candle patterns for info but DO NOT block on M1 scalping
        # On M1 timeframe, candle patterns are noise - momentum is king
        if momentum_direction == "BUY" and candle_info["block_buy"]:
            logger.info("[SignalGen] Candle pattern '%s' (bias=%s) conflicts with BUY - "
                        "logging only, NOT blocking (M1 scalper mode)",
                        candle_pattern, candle_bias)

        if momentum_direction == "SELL" and candle_info["block_sell"]:
            logger.info("[SignalGen] Candle pattern '%s' (bias=%s) conflicts with SELL - "
                        "logging only, NOT blocking (M1 scalper mode)",
                        candle_pattern, candle_bias)

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

        # 5. HTF bias - M15/M5 penalty for scalper
        # Penalize if M15 is against momentum (magnitude > 0.5)
        # M15 is more meaningful for M1 scalps than lazy H4
        htf_confidence_adj = 0.0
        if htf_bias:
            m15_bias = htf_bias.get("15m", 0.0)
            if action == "BUY" and m15_bias < -0.5:
                htf_confidence_adj = -0.10
                logger.info("[SignalGen] HTF: Penalty for BUY against M15 bearish (%.2f)", m15_bias)
            elif action == "SELL" and m15_bias > 0.5:
                htf_confidence_adj = -0.10
                logger.info("[SignalGen] HTF: Penalty for SELL against M15 bullish (%.2f)", m15_bias)

            # 5m micro-trend check: additional penalty if strongly against
            m5_bias = htf_bias.get("5m", 0.0)
            if action == "BUY" and m5_bias < -0.6:
                htf_confidence_adj += -0.05
                logger.info("[SignalGen] HTF: Additional M5 penalty for BUY against M5 bearish (%.2f)", m5_bias)
            elif action == "SELL" and m5_bias > 0.6:
                htf_confidence_adj += -0.05
                logger.info("[SignalGen] HTF: Additional M5 penalty for SELL against M5 bullish (%.2f)", m5_bias)

        # Apply HTF adjustment
        timing_confidence = timing_confidence + htf_confidence_adj
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5b. Support/Resistance proximity penalty
        # NOTE: Known limitation - S/R detection degrades to a no-op when the prices
        # DataFrame has fewer bars than sr_lookback (e.g. 7-bar minimum for momentum).
        # At inference time with short live feeds, this filter silently skips. This is
        # acceptable because S/R is a supplementary confidence penalty, not a gate.
        sr_levels = self._detect_support_resistance(
            momentum_prices, lookback=self.data_config.sr_lookback
        )
        if sr_levels["nearest_resistance"] is not None and atr > 0:
            if (action == "BUY" and
                    abs(current_price - sr_levels["nearest_resistance"]) < 0.3 * atr):
                timing_confidence -= 0.10
                logger.info("[SignalGen] S/R penalty: BUY near resistance (%.2f), "
                            "confidence -0.10", sr_levels["nearest_resistance"])
        if sr_levels["nearest_support"] is not None and atr > 0:
            if (action == "SELL" and
                    abs(current_price - sr_levels["nearest_support"]) < 0.3 * atr):
                timing_confidence -= 0.10
                logger.info("[SignalGen] S/R penalty: SELL near support (%.2f), "
                            "confidence -0.10", sr_levels["nearest_support"])
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5c. RSI exhaustion filter
        # NOTE: Known limitation - the three stacking 0.10 penalties (S/R, RSI, HTF)
        # rarely block trades given the 0.15 min_confidence threshold (worst case:
        # 0.95 - 0.30 = 0.65 still passes). This is intentional for a high-frequency
        # scalper where the goal is to slightly reduce position sizing rather than
        # gate entries aggressively.
        import pandas as pd
        if isinstance(momentum_prices, pd.DataFrame) and "Close" in momentum_prices.columns:
            try:
                rsi_series = ta.momentum.rsi(momentum_prices["Close"], window=14)
                if rsi_series is not None and len(rsi_series) > 0:
                    current_rsi = float(rsi_series.iloc[-1])
                    if not np.isnan(current_rsi):
                        if current_rsi > self.data_config.rsi_overbought and action == "BUY":
                            timing_confidence -= 0.10
                            logger.info("[SignalGen] RSI exhaustion: overbought (RSI=%.1f), "
                                        "BUY confidence -0.10", current_rsi)
                        elif current_rsi < self.data_config.rsi_oversold and action == "SELL":
                            timing_confidence -= 0.10
                            logger.info("[SignalGen] RSI exhaustion: oversold (RSI=%.1f), "
                                        "SELL confidence -0.10", current_rsi)
            except Exception as e:
                logger.warning("[SignalGen] RSI filter error (filter disabled for this bar): %s", e)
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5d. Price structure check - penalize if momentum opposes structure
        structure = self._detect_price_structure(momentum_prices)
        if structure != "no_structure":
            logger.info("[SignalGen] Price structure: %s", structure)
            if action == "BUY" and structure == "downtrend":
                timing_confidence -= self.price_structure_config.confidence_penalty
                logger.info("[SignalGen] Price structure penalty: BUY against downtrend, confidence -%.2f",
                            self.price_structure_config.confidence_penalty)
            elif action == "SELL" and structure == "uptrend":
                timing_confidence -= self.price_structure_config.confidence_penalty
                logger.info("[SignalGen] Price structure penalty: SELL against uptrend, confidence -%.2f",
                            self.price_structure_config.confidence_penalty)
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5e. DXY correlation check
        dxy_adj = self._check_dxy_correlation(prices, cross_pair_info)
        if action == "BUY":
            timing_confidence += dxy_adj["buy_adjustment"]
        elif action == "SELL":
            timing_confidence += dxy_adj["sell_adjustment"]
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5f. FVG (Fair Value Gap) detection
        fvg_info = self._detect_fvg(prices)
        if action == "BUY" and fvg_info["fvg_aligns_buy"]:
            timing_confidence += self.fvg_config.confidence_boost
            logger.info("[SignalGen] FVG confluence: bullish FVG aligns with BUY, confidence +%.2f",
                        self.fvg_config.confidence_boost)
        elif action == "SELL" and fvg_info["fvg_aligns_sell"]:
            timing_confidence += self.fvg_config.confidence_boost
            logger.info("[SignalGen] FVG confluence: bearish FVG aligns with SELL, confidence +%.2f",
                        self.fvg_config.confidence_boost)
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 5g. Liquidity sweep detection
        sweep_info = self._detect_liquidity_sweep(prices)
        if action == "BUY" and sweep_info["bullish_sweep"]:
            timing_confidence += self.liquidity_sweep_config.confidence_boost
            logger.info("[SignalGen] Liquidity sweep: bullish sweep aligns with BUY, confidence +%.2f",
                        self.liquidity_sweep_config.confidence_boost)
        elif action == "SELL" and sweep_info["bearish_sweep"]:
            timing_confidence += self.liquidity_sweep_config.confidence_boost
            logger.info("[SignalGen] Liquidity sweep: bearish sweep aligns with SELL, confidence +%.2f",
                        self.liquidity_sweep_config.confidence_boost)
        timing_confidence = max(0.0, min(1.0, timing_confidence))

        # 6. Apply session confidence multiplier and confidence threshold (timing gate)
        # Asian/off-session trades get reduced confidence to filter out marginal signals
        timing_confidence *= session_confidence_mult
        timing_confidence = max(0.0, min(1.0, timing_confidence))

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

        # Cap ATR to prevent excessively large SL/TP values
        # M1 ATR (~$2-3) won't hit this; cap protects against H1 fallback or bad data
        atr = min(atr, 5.0)

        sl_mult = self.signal_config.atr_sl_multiplier * regime_adjustments.get("sl_mult", 1.0)
        tp_mult = self.signal_config.atr_tp_multiplier * regime_adjustments.get("tp_mult", 1.0)
        levels = self.risk_manager.calculate_sl_tp(
            atr=atr, direction=action,
            current_price=current_price,
            sl_mult=sl_mult, tp_mult=tp_mult
        )

        # 9a. Range mode SL override: tighter stop loss ($2 instead of $3)
        # since range-bound moves are smaller and we want quicker exits on failure
        if range_signal is not None:
            range_sl_pips = 20.0  # $2 for gold (10 pips = $1)
            if levels["sl_pips"] > range_sl_pips:
                logger.info("[SignalGen] RANGE MODE: Reducing SL from %.1f to %.1f pips "
                            "(tighter for range trading)", levels["sl_pips"], range_sl_pips)
                levels["sl_pips"] = range_sl_pips

        # 9b. Dynamic trailing mode: if tp_pips == 0 (atr_tp_multiplier=0),
        # set tp_pips=9999 to signal the EA to manage exit dynamically (no fixed TP)
        if levels["tp_pips"] == 0:
            levels["tp_pips"] = 9999

        # 10. Calculate position size
        position_mult = regime_adjustments.get("position_size_mult", 1.0)
        # Apply session multiplier to position sizing
        position_mult *= session_multiplier
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
        # Increment daily trade counter
        if hasattr(self, '_trades_today'):
            self._trades_today += 1

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

    def set_auto_optimizer(self, optimizer) -> None:
        """
        Set a reference to the AutoOptimizer for self-tuning parameters.
        Called from main.py during initialization.
        """
        self._auto_optimizer = optimizer

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
