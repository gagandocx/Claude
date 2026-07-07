"""
CCT Rectangle Strategy Implementation.

Full implementation of the CCT (Candle Continuity Theory) Rectangle Setup:
- Step 1: Direction Candle detection on 4H timeframe
- Step 2: Weakness detection on 15M timeframe  
- Step 3: Rectangle Entry on 1M timeframe

Based on "The Only 1 Minute Scalping Strategy You'll Ever Need"
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import config
from utils import (
    calculate_ema,
    detect_fair_value_gaps,
    identify_session_levels,
    calculate_swing_highs,
    calculate_swing_lows,
    is_level_in_fvg,
    is_at_session_extreme,
    get_next_key_level,
    calculate_rr_ratio,
)


@dataclass
class DirectionSignal:
    """Represents a 4H directional candle signal."""
    time: pd.Timestamp
    direction: str  # 'bullish' or 'bearish'
    candle_high: float
    candle_low: float
    candle_open: float
    candle_close: float
    next_candle_open: float  # Key reference level
    target: float  # Direction candle high (bullish) or low (bearish)


@dataclass
class WeaknessSignal:
    """Represents a confirmed weakness on the 15M timeframe."""
    time: pd.Timestamp
    direction: str  # Same as parent direction signal
    level_swept: float  # The swing level that was swept
    trigger_candle_close: float  # Close of the M15 trigger candle
    trigger_candle_extreme: float  # High or Low of the trigger candle (wick tip)
    rectangle_top: float  # Top of the rectangle
    rectangle_bottom: float  # Bottom of the rectangle
    in_fvg: bool  # Whether the level is inside a FVG
    at_session_extreme: bool  # Whether at session high/low


@dataclass
class TradeSetup:
    """Represents a complete trade setup ready for execution."""
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    rectangle_top: float
    rectangle_bottom: float
    direction_signal_time: pd.Timestamp
    weakness_signal_time: pd.Timestamp


@dataclass
class TradeResult:
    """Represents a completed trade with outcome."""
    setup: TradeSetup
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pips: float
    pnl_dollars: float
    outcome: str  # 'win', 'loss', 'breakeven'
    rr_achieved: float


class DirectionDetector:
    """
    Step 1: Identifies CCT directional candles on the 4H timeframe.
    
    BULLISH direction candle:
      - Sweeps previous candle's LOW (wick goes below prev low)
      - CLOSES ABOVE the previous candle's HIGH (engulfing)
      
    BEARISH direction candle:
      - Sweeps previous candle's HIGH (wick goes above prev high)
      - CLOSES BELOW the previous candle's LOW (engulfing)
    
    Additional filter: EMA alignment (price above 50/200 EMA = longs only)
    """
    
    def __init__(self, df_4h: pd.DataFrame):
        self.df = df_4h.copy()
        self.signals: List[DirectionSignal] = []
        
        # Calculate EMAs
        self.df["ema_fast"] = calculate_ema(self.df["Close"], config.EMA_FAST)
        self.df["ema_slow"] = calculate_ema(self.df["Close"], config.EMA_SLOW)
    
    def detect_all(self) -> List[DirectionSignal]:
        """
        Scan through all 4H candles and detect direction signals.
        
        Returns:
            List of DirectionSignal objects
        """
        self.signals = []
        
        for i in range(1, len(self.df) - 1):
            signal = self._check_direction_candle(i)
            if signal is not None:
                self.signals.append(signal)
        
        return self.signals
    
    def _check_direction_candle(self, idx: int) -> Optional[DirectionSignal]:
        """
        Check if candle at index is a valid CCT direction candle.
        
        Args:
            idx: Index position in the 4H DataFrame
        
        Returns:
            DirectionSignal if valid, None otherwise
        """
        curr = self.df.iloc[idx]
        prev = self.df.iloc[idx - 1]
        next_candle = self.df.iloc[idx + 1]
        
        tolerance = config.DIRECTION_SWEEP_TOLERANCE
        
        # Check for BULLISH direction candle
        # Condition: Sweeps previous LOW (wick below) AND closes above previous HIGH
        bullish = (
            curr["Low"] < prev["Low"] - tolerance and  # Sweeps prev low
            curr["Close"] > prev["High"]  # Closes above prev high
        )
        
        # Check for BEARISH direction candle
        # Condition: Sweeps previous HIGH (wick above) AND closes below previous LOW
        bearish = (
            curr["High"] > prev["High"] + tolerance and  # Sweeps prev high
            curr["Close"] < prev["Low"]  # Closes below prev low
        )
        
        if not bullish and not bearish:
            return None
        
        direction = "bullish" if bullish else "bearish"
        
# EMA Filter: Only take trades aligned with the trend
        # Strategy says: "Use 50 EMA or 200 EMA to lock in directional bias"
        # We use EMA 50 as the primary filter (more responsive for scalping)
        if config.USE_EMA_FILTER:
            ema_fast_val = curr["ema_fast"]
            
            if direction == "bullish":
                # Price must be above EMA for longs
                if curr["Close"] < ema_fast_val:
                    return None
            else:
                # Price must be below EMA for shorts
                if curr["Close"] > ema_fast_val:
                    return None
        
        # Continuation filter: ensure we are trading with the trend
        if config.CONTINUATION_ONLY:
            ema_fast_val = curr["ema_fast"]
            if direction == "bullish" and curr["Close"] < ema_fast_val:
                return None
            if direction == "bearish" and curr["Close"] > ema_fast_val:
                return None
        
        # Create the signal
        target = curr["High"] if direction == "bullish" else curr["Low"]
        
        signal = DirectionSignal(
            time=self.df.index[idx],
            direction=direction,
            candle_high=curr["High"],
            candle_low=curr["Low"],
            candle_open=curr["Open"],
            candle_close=curr["Close"],
            next_candle_open=next_candle["Open"],
            target=target,
        )
        
        return signal


class WeaknessDetector:
    """
    Step 2: Detects weakness on the 15M timeframe.
    
    After a 4H direction signal, look on 15M for:
    - A swing high/low that aligns with the direction
    - Price sweeps the level (takes liquidity) but FAILS to close beyond it
    - This wick rejection = WEAKNESS = trade opportunity
    
    Filters:
    - Level ideally sits inside a Fair Value Gap (FVG)
    - Level ideally at session extremes (Asia/London/NY highs or lows)
    - If price closes beyond the level = STRENGTH = no trade
    """
    
    def __init__(self, df_15m: pd.DataFrame):
        self.df = df_15m.copy()
        self.fvgs = detect_fair_value_gaps(self.df)
        self.session_levels = identify_session_levels(self.df)
        
        # Precompute swing highs and lows
        self.swing_highs = calculate_swing_highs(self.df)
        self.swing_lows = calculate_swing_lows(self.df)
    
    def find_weakness(
        self, 
        direction_signal: DirectionSignal,
    ) -> Optional[WeaknessSignal]:
        """
        Look for weakness on M15 after a direction signal.
        
        Args:
            direction_signal: The 4H direction signal to trade from
        
        Returns:
            WeaknessSignal if found, None otherwise
        """
        direction = direction_signal.direction
        signal_time = direction_signal.time
        
        # Find the M15 candles that fall within the next 4H candle period
        # Strategy: "The retracement/weakness happening in the VERY NEXT candle
        # after direction candle is highest probability" -- but we allow a wider
        # window to capture more setups during backtesting
        start_time = signal_time
        end_time = signal_time + pd.Timedelta(hours=12)  # Look up to 12 hours ahead
        
        mask = (self.df.index >= start_time) & (self.df.index <= end_time)
        relevant_15m = self.df[mask]
        
        if relevant_15m.empty:
            return None
        
        # Get the key levels to watch (swing highs/lows before the signal)
        lookback_start = signal_time - pd.Timedelta(hours=48)
        lookback_mask = (self.df.index >= lookback_start) & (self.df.index < start_time)
        lookback_data = self.df[lookback_mask]
        
        if lookback_data.empty:
            return None
        
        # Identify key levels based on direction
        key_levels = self._get_key_levels(lookback_data, direction)
        
        if not key_levels:
            # Use the next candle open as a fallback key level
            key_levels = [direction_signal.next_candle_open]
        
        # Also add the direction candle's next open as a key reference 
        # (per strategy: "Mark the opening price of the NEXT candle")
        if direction_signal.next_candle_open not in key_levels:
            key_levels.append(direction_signal.next_candle_open)
        
        # Look for weakness (wick rejection) at each key level
        for level in key_levels:
            weakness = self._check_weakness_at_level(
                relevant_15m, level, direction, direction_signal
            )
            if weakness is not None:
                return weakness
        
        return None
    
    def _get_key_levels(
        self, lookback_data: pd.DataFrame, direction: str
    ) -> List[float]:
        """
        Get key swing levels from the lookback period.
        
        For bullish direction: look for swing lows (buy setups form at swept lows)
        For bearish direction: look for swing highs (short setups form at swept highs)
        """
        levels = []
        
        lookback_swing_highs = calculate_swing_highs(lookback_data, lookback=3)
        lookback_swing_lows = calculate_swing_lows(lookback_data, lookback=3)
        
        if direction == "bullish":
            # Focus on lows in uptrend
            low_indices = lookback_data.index[lookback_swing_lows]
            for idx in low_indices:
                levels.append(lookback_data.loc[idx, "Low"])
        else:
            # Focus on highs in downtrend
            high_indices = lookback_data.index[lookback_swing_highs]
            for idx in high_indices:
                levels.append(lookback_data.loc[idx, "High"])
        
        # Sort: for bullish, lowest levels first; for bearish, highest first
        if direction == "bullish":
            levels.sort()
        else:
            levels.sort(reverse=True)
        
        return levels[:8]  # Return top 8 key levels
    
    def _check_weakness_at_level(
        self,
        relevant_15m: pd.DataFrame,
        level: float,
        direction: str,
        direction_signal: DirectionSignal,
    ) -> Optional[WeaknessSignal]:
        """
        Check if any M15 candle shows weakness (wick rejection) at a given level.
        
        WEAKNESS: Price sweeps the level but FAILS to close beyond it.
        STRENGTH (no trade): Price closes beyond the level.
        """
        sweep_min = config.SWEEP_MIN_PIPS
        
        for i in range(len(relevant_15m)):
            candle = relevant_15m.iloc[i]
            candle_time = relevant_15m.index[i]
            
            if direction == "bullish":
                # Looking for sweep of a LOW level (price goes below then closes above)
                swept = candle["Low"] < level - sweep_min
                closed_back = candle["Close"] > level
                
                if swept and closed_back:
                    # WEAKNESS confirmed - wick rejection below the level
                    trigger_close = candle["Close"]
                    trigger_extreme = candle["Low"]  # The wick tip (lowest point)
                    
                    # Rectangle: from close to extreme
                    rect_top = trigger_close
                    rect_bottom = trigger_extreme
                    
                    # Validate rectangle size
                    rect_size = rect_top - rect_bottom
                    if rect_size < config.MIN_RECTANGLE_SIZE_PIPS:
                        continue
                    
                    # Check filters
                    in_fvg = is_level_in_fvg(level, self.fvgs, "bullish")
                    at_session = is_at_session_extreme(
                        level, self.session_levels, candle_time
                    )
                    
                    # If requiring imbalance filter and it doesn't pass, skip
                    if config.REQUIRE_IMBALANCE_FILTER and not in_fvg:
                        continue
                    if config.REQUIRE_SESSION_EXTREME and not at_session:
                        continue
                    
                    return WeaknessSignal(
                        time=candle_time,
                        direction=direction,
                        level_swept=level,
                        trigger_candle_close=trigger_close,
                        trigger_candle_extreme=trigger_extreme,
                        rectangle_top=rect_top,
                        rectangle_bottom=rect_bottom,
                        in_fvg=in_fvg,
                        at_session_extreme=at_session,
                    )
                
                # Check for STRENGTH (no trade) - price closes below level
                # with strong displacement (not just a small close below)
                if (candle["Close"] < level - sweep_min * 3 and 
                    candle["Low"] < level - sweep_min * 3):
                    # Strong displacement through the level = strength = skip
                    break
            
            else:  # bearish
                # Looking for sweep of a HIGH level (price goes above then closes below)
                swept = candle["High"] > level + sweep_min
                closed_back = candle["Close"] < level
                
                if swept and closed_back:
                    # WEAKNESS confirmed - wick rejection above the level
                    trigger_close = candle["Close"]
                    trigger_extreme = candle["High"]  # The wick tip (highest point)
                    
                    # Rectangle: from close to extreme
                    rect_top = trigger_extreme
                    rect_bottom = trigger_close
                    
                    # Validate rectangle size
                    rect_size = rect_top - rect_bottom
                    if rect_size < config.MIN_RECTANGLE_SIZE_PIPS:
                        continue
                    
                    # Check filters
                    in_fvg = is_level_in_fvg(level, self.fvgs, "bearish")
                    at_session = is_at_session_extreme(
                        level, self.session_levels, candle_time
                    )
                    
                    if config.REQUIRE_IMBALANCE_FILTER and not in_fvg:
                        continue
                    if config.REQUIRE_SESSION_EXTREME and not at_session:
                        continue
                    
                    return WeaknessSignal(
                        time=candle_time,
                        direction=direction,
                        level_swept=level,
                        trigger_candle_close=trigger_close,
                        trigger_candle_extreme=trigger_extreme,
                        rectangle_top=rect_top,
                        rectangle_bottom=rect_bottom,
                        in_fvg=in_fvg,
                        at_session_extreme=at_session,
                    )
                
                # Check for STRENGTH - price closes above level
                # with strong displacement
                if (candle["Close"] > level + sweep_min * 3 and 
                    candle["High"] > level + sweep_min * 3):
                    break
        
        return None


class RectangleEntry:
    """
    Step 3: Rectangle Entry on the 1M timeframe.
    
    After confirming weakness on M15:
    - Draw rectangle from M15 trigger candle CLOSE to its EXTREME (wick tip)
    - On 1M, wait for a candle to CLOSE outside the rectangle
    - Entry: on the 1M close outside rectangle (above for longs, below for shorts)
    - Stop Loss: beyond the rectangle extreme
    - Take Profit: 3:1 RR minimum or next key M15 level
    
    Fallback: If 1M data is not available for the period, uses 15M candles
    for entry detection (simulating the M1 flip on larger timeframe).
    """
    
    def __init__(self, df_1m: pd.DataFrame, df_15m: pd.DataFrame):
        self.df_1m = df_1m
        self.df_15m = df_15m
    
    def find_entry(
        self, 
        weakness: WeaknessSignal,
    ) -> Optional[TradeSetup]:
        """
        Look for a 1M entry after a weakness signal is confirmed.
        Falls back to 15M if 1M data is not available for this time period.
        
        Args:
            weakness: The confirmed weakness signal from M15
        
        Returns:
            TradeSetup if entry found, None otherwise
        """
        direction = weakness.direction
        rect_top = weakness.rectangle_top
        rect_bottom = weakness.rectangle_bottom
        
        # Find entry candles after the weakness signal
        start_time = weakness.time
        
        # Try 1M first
        end_time_1m = start_time + pd.Timedelta(minutes=config.MAX_CANDLES_FOR_ENTRY)
        mask_1m = (self.df_1m.index > start_time) & (self.df_1m.index <= end_time_1m)
        entry_candles = self.df_1m[mask_1m]
        
        # If no 1M data available for this period, use 15M as fallback
        if entry_candles.empty:
            end_time_15m = start_time + pd.Timedelta(hours=4)
            mask_15m = (
                (self.df_15m.index > start_time) & 
                (self.df_15m.index <= end_time_15m)
            )
            entry_candles = self.df_15m[mask_15m]
        
        if entry_candles.empty:
            return None
        
        for i in range(len(entry_candles)):
            candle = entry_candles.iloc[i]
            candle_time = entry_candles.index[i]
            
            if direction == "bullish":
                # Entry: candle closes ABOVE the rectangle top
                if candle["Close"] > rect_top:
                    entry_price = candle["Close"]
                    stop_loss = rect_bottom - config.SPREAD  # SL beyond rectangle extreme
                    
                    # Calculate risk
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        continue
                    
                    # Take profit at target RR ratio
                    take_profit = entry_price + (risk * config.TARGET_RR_RATIO)
                    
                    # Check if there is a closer key level for TP
                    m15_idx = self.df_15m.index.get_indexer(
                        [candle_time], method="ffill"
                    )[0]
                    if m15_idx >= 0:
                        key_level = get_next_key_level(
                            self.df_15m, m15_idx, "bullish"
                        )
                        if key_level is not None:
                            level_rr = calculate_rr_ratio(
                                entry_price, stop_loss, key_level
                            )
                            if level_rr >= config.MIN_RR_RATIO:
                                take_profit = key_level
                    
                    rr = calculate_rr_ratio(entry_price, stop_loss, take_profit)
                    
                    if rr < config.MIN_RR_RATIO:
                        continue
                    
                    # Cap RR at maximum
                    if rr > config.MAX_RR_RATIO:
                        take_profit = entry_price + (risk * config.MAX_RR_RATIO)
                        rr = config.MAX_RR_RATIO
                    
                    return TradeSetup(
                        entry_time=candle_time,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        rr_ratio=rr,
                        rectangle_top=rect_top,
                        rectangle_bottom=rect_bottom,
                        direction_signal_time=weakness.time,
                        weakness_signal_time=weakness.time,
                    )
            
            else:  # bearish
                # Entry: candle closes BELOW the rectangle bottom
                if candle["Close"] < rect_bottom:
                    entry_price = candle["Close"]
                    stop_loss = rect_top + config.SPREAD  # SL beyond rectangle extreme
                    
                    # Calculate risk
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        continue
                    
                    # Take profit at target RR ratio
                    take_profit = entry_price - (risk * config.TARGET_RR_RATIO)
                    
                    # Check for key level TP
                    m15_idx = self.df_15m.index.get_indexer(
                        [candle_time], method="ffill"
                    )[0]
                    if m15_idx >= 0:
                        key_level = get_next_key_level(
                            self.df_15m, m15_idx, "bearish"
                        )
                        if key_level is not None:
                            level_rr = calculate_rr_ratio(
                                entry_price, stop_loss, key_level
                            )
                            if level_rr >= config.MIN_RR_RATIO:
                                take_profit = key_level
                    
                    rr = calculate_rr_ratio(entry_price, stop_loss, take_profit)
                    
                    if rr < config.MIN_RR_RATIO:
                        continue
                    
                    if rr > config.MAX_RR_RATIO:
                        take_profit = entry_price - (risk * config.MAX_RR_RATIO)
                        rr = config.MAX_RR_RATIO
                    
                    return TradeSetup(
                        entry_time=candle_time,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        rr_ratio=rr,
                        rectangle_top=rect_top,
                        rectangle_bottom=rect_bottom,
                        direction_signal_time=weakness.time,
                        weakness_signal_time=weakness.time,
                    )
        
        return None


class TradeChecklist:
    """
    Validates all required conditions before entering a trade.
    
    Required conditions:
    1. Direction Aligned - Price clearly above/below the 50 or 200 EMA
    2. Valid Structure - Clear structural breaks
    3. Continuation Focus - Lows in uptrend / Highs in downtrend
    4. Imbalance Filter - M15 level sits inside FVG or session extreme (high priority)
    5. Wick Rejection - M15 swept the level AND closed back inside
    6. Rectangle Drawn - From close to extreme of the M15 trigger candle
    """
    
    @staticmethod
    def validate(
        direction_signal: DirectionSignal,
        weakness_signal: WeaknessSignal,
        trade_setup: TradeSetup,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all checklist items.
        
        Returns:
            Tuple of (is_valid, list_of_failed_checks)
        """
        failed = []
        
        # 1. Direction Aligned (already checked in DirectionDetector via EMA)
        # This is inherently validated by the signal generation
        
        # 2. Valid Structure (direction candle is structural)
        if direction_signal is None:
            failed.append("No valid direction signal")
        
        # 3. Continuation Focus (validated in DirectionDetector)
        
        # 4. Imbalance Filter (high priority but not strictly required)
        # Tracked in weakness signal for scoring
        
        # 5. Wick Rejection (core requirement)
        if weakness_signal is None:
            failed.append("No wick rejection confirmed")
        
        # 6. Rectangle Drawn
        if trade_setup is None:
            failed.append("No valid rectangle entry")
        elif trade_setup.rectangle_top <= trade_setup.rectangle_bottom:
            failed.append("Invalid rectangle dimensions")
        
        # Validate RR ratio
        if trade_setup is not None and trade_setup.rr_ratio < config.MIN_RR_RATIO:
            failed.append(
                f"RR ratio {trade_setup.rr_ratio:.1f} below minimum {config.MIN_RR_RATIO}"
            )
        
        is_valid = len(failed) == 0
        return is_valid, failed


class CCTRectangleStrategy:
    """
    Main strategy class that orchestrates all components.
    Combines direction detection, weakness finding, and rectangle entry.
    """
    
    def __init__(
        self, 
        df_4h: pd.DataFrame, 
        df_15m: pd.DataFrame, 
        df_1m: pd.DataFrame
    ):
        self.df_4h = df_4h
        self.df_15m = df_15m
        self.df_1m = df_1m
        
        # Initialize components
        self.direction_detector = DirectionDetector(df_4h)
        self.weakness_detector = WeaknessDetector(df_15m)
        self.rectangle_entry = RectangleEntry(df_1m, df_15m)
        
        self.trade_setups: List[TradeSetup] = []
    
    def generate_signals(self) -> List[TradeSetup]:
        """
        Run the full strategy pipeline:
        1. Detect direction candles on 4H
        2. For each direction, find weakness on 15M
        3. For each weakness, find rectangle entry on 1M
        
        Returns:
            List of validated TradeSetup objects
        """
        self.trade_setups = []
        
        # Step 1: Get all direction signals
        direction_signals = self.direction_detector.detect_all()
        
        if not direction_signals:
            print("  No direction signals found on 4H timeframe.")
            return []
        
        print(f"  Found {len(direction_signals)} direction signals on 4H")
        
        # Step 2 & 3: For each direction signal, find weakness and entry
        for dir_signal in direction_signals:
            # Find weakness on M15
            weakness = self.weakness_detector.find_weakness(dir_signal)
            
            if weakness is None:
                continue
            
            # Find rectangle entry on M1
            entry = self.rectangle_entry.find_entry(weakness)
            
            if entry is None:
                continue
            
            # Update entry with direction signal time
            entry.direction_signal_time = dir_signal.time
            entry.weakness_signal_time = weakness.time
            
            # Validate through checklist
            is_valid, failures = TradeChecklist.validate(
                dir_signal, weakness, entry
            )
            
            if is_valid:
                self.trade_setups.append(entry)
        
        print(f"  Generated {len(self.trade_setups)} validated trade setups")
        return self.trade_setups
