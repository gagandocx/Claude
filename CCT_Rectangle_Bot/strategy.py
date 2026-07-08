"""
CCT Rectangle Strategy Implementation - AGGRESSIVE MODE.

Optimized for maximum trade frequency and profitability:
- Relaxed direction candle detection (partial sweeps allowed)
- Wider weakness detection window with multiple signals per direction
- More key levels and smaller rectangle thresholds
- Multiple concurrent trades supported

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
    strength: float  # Signal strength score (0-1)


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

    AGGRESSIVE MODE:
    - Full engulfing (sweep + close beyond) = high strength signal
    - Partial sweep (sweep + strong close into prev range) = medium strength signal
    - EMA filter optional (disabled by default for more signals)
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
        Includes both full engulfing and partial sweep patterns.

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
        Now supports partial sweeps for more signals.
        """
        curr = self.df.iloc[idx]
        prev = self.df.iloc[idx - 1]
        next_candle = self.df.iloc[idx + 1]

        tolerance = config.DIRECTION_SWEEP_TOLERANCE
        prev_range = prev["High"] - prev["Low"]

        # Prevent division by zero
        if prev_range <= 0:
            return None

        # --- BULLISH DETECTION ---
        # Full engulfing: sweeps prev low AND closes above prev high
        bullish_full = (
            curr["Low"] < prev["Low"] - tolerance and
            curr["Close"] > prev["High"]
        )

        # Partial sweep: sweeps prev low, closes strong but not above prev high
        bullish_partial = False
        bullish_strength = 0.0

        if not bullish_full and not config.REQUIRE_FULL_ENGULF:
            # Check if it sweeps the low
            sweeps_low = curr["Low"] < prev["Low"] - tolerance
            # Check if close is strong (at least X% into prev range from prev low)
            if sweeps_low and prev_range > 0:
                close_penetration = (curr["Close"] - prev["Low"]) / prev_range
                if close_penetration >= config.PARTIAL_ENGULF_RATIO:
                    bullish_partial = True
                    bullish_strength = min(close_penetration, 1.0) * 0.7

        if bullish_full:
            bullish_strength = 1.0

        # --- BEARISH DETECTION ---
        bearish_full = (
            curr["High"] > prev["High"] + tolerance and
            curr["Close"] < prev["Low"]
        )

        bearish_partial = False
        bearish_strength = 0.0

        if not bearish_full and not config.REQUIRE_FULL_ENGULF:
            sweeps_high = curr["High"] > prev["High"] + tolerance
            if sweeps_high and prev_range > 0:
                close_penetration = (prev["High"] - curr["Close"]) / prev_range
                if close_penetration >= config.PARTIAL_ENGULF_RATIO:
                    bearish_partial = True
                    bearish_strength = min(close_penetration, 1.0) * 0.7

        if bearish_full:
            bearish_strength = 1.0

        # Determine direction
        is_bullish = bullish_full or bullish_partial
        is_bearish = bearish_full or bearish_partial

        if not is_bullish and not is_bearish:
            return None

        # If both qualify, pick the stronger one
        if is_bullish and is_bearish:
            if bullish_strength >= bearish_strength:
                is_bearish = False
            else:
                is_bullish = False

        direction = "bullish" if is_bullish else "bearish"
        strength = bullish_strength if is_bullish else bearish_strength

        # EMA Filter (optional - disabled by default for max signals)
        if config.USE_EMA_FILTER:
            ema_fast_val = curr["ema_fast"]
            if direction == "bullish" and curr["Close"] < ema_fast_val:
                return None
            if direction == "bearish" and curr["Close"] > ema_fast_val:
                return None

        # Continuation filter (optional - disabled by default)
        if config.CONTINUATION_ONLY:
            ema_fast_val = curr["ema_fast"]
            if direction == "bullish" and curr["Close"] < ema_fast_val:
                return None
            if direction == "bearish" and curr["Close"] > ema_fast_val:
                return None

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
            strength=strength,
        )

        return signal


class WeaknessDetector:
    """
    Step 2: Detects weakness on the 15M timeframe.

    AGGRESSIVE MODE:
    - Wider search window (up to 24 hours after direction signal)
    - More key levels checked
    - Multiple weakness signals per direction signal
    - Reduced minimum sweep distance
    """

    def __init__(self, df_15m: pd.DataFrame):
        self.df = df_15m.copy()
        self.fvgs = detect_fair_value_gaps(self.df)
        self.session_levels = identify_session_levels(self.df)

        # Precompute swing highs and lows
        self.swing_highs = calculate_swing_highs(self.df)
        self.swing_lows = calculate_swing_lows(self.df)

    def find_all_weakness(
        self,
        direction_signal: DirectionSignal,
    ) -> List[WeaknessSignal]:
        """
        Look for ALL weakness signals on M15 after a direction signal.
        Returns multiple weakness signals (up to MAX_WEAKNESS_PER_DIRECTION).
        """
        direction = direction_signal.direction
        signal_time = direction_signal.time

        # Extended window for weakness detection
        start_time = signal_time
        window_hours = config.WEAKNESS_WINDOW_HOURS
        end_time = signal_time + pd.Timedelta(hours=window_hours)

        mask = (self.df.index >= start_time) & (self.df.index <= end_time)
        relevant_15m = self.df[mask]

        if relevant_15m.empty:
            return []

        # Get key levels (more levels with reduced lookback)
        lookback_start = signal_time - pd.Timedelta(hours=72)
        lookback_mask = (self.df.index >= lookback_start) & (self.df.index < start_time)
        lookback_data = self.df[lookback_mask]

        if lookback_data.empty:
            return []

        key_levels = self._get_key_levels(lookback_data, direction)

        if not key_levels:
            key_levels = [direction_signal.next_candle_open]

        # Also add direction candle's next open as key reference
        if direction_signal.next_candle_open not in key_levels:
            key_levels.append(direction_signal.next_candle_open)

        # Add midpoints and other computed levels
        if len(key_levels) >= 2:
            # Add midpoints between adjacent levels
            midpoints = []
            sorted_levels = sorted(key_levels)
            for i in range(len(sorted_levels) - 1):
                mid = (sorted_levels[i] + sorted_levels[i + 1]) / 2.0
                midpoints.append(mid)
            key_levels.extend(midpoints[:5])

        # Find multiple weakness signals
        weakness_signals = []
        used_times = set()

        for level in key_levels:
            if len(weakness_signals) >= config.MAX_WEAKNESS_PER_DIRECTION:
                break

            weakness = self._check_weakness_at_level(
                relevant_15m, level, direction, direction_signal, used_times
            )
            if weakness is not None:
                weakness_signals.append(weakness)
                used_times.add(weakness.time)

        return weakness_signals

    def find_weakness(
        self,
        direction_signal: DirectionSignal,
    ) -> Optional[WeaknessSignal]:
        """
        Backward-compatible: return first weakness signal found.
        """
        results = self.find_all_weakness(direction_signal)
        return results[0] if results else None

    def _get_key_levels(
        self, lookback_data: pd.DataFrame, direction: str
    ) -> List[float]:
        """
        Get key swing levels from the lookback period.
        AGGRESSIVE: reduced lookback = more swing points detected.
        """
        levels = []

        lookback_swing_highs = calculate_swing_highs(lookback_data, lookback=2)
        lookback_swing_lows = calculate_swing_lows(lookback_data, lookback=2)

        if direction == "bullish":
            low_indices = lookback_data.index[lookback_swing_lows]
            for idx in low_indices:
                levels.append(lookback_data.loc[idx, "Low"])
            # Also consider recent highs as possible retest levels
            high_indices = lookback_data.index[lookback_swing_highs]
            for idx in high_indices[-3:]:
                levels.append(lookback_data.loc[idx, "High"])
        else:
            high_indices = lookback_data.index[lookback_swing_highs]
            for idx in high_indices:
                levels.append(lookback_data.loc[idx, "High"])
            # Also consider recent lows as retest levels
            low_indices = lookback_data.index[lookback_swing_lows]
            for idx in low_indices[-3:]:
                levels.append(lookback_data.loc[idx, "Low"])

        # Sort: for bullish, closest to current price first
        if direction == "bullish":
            levels.sort(reverse=True)
        else:
            levels.sort()

        # Return more levels (up to 15)
        return levels[:15]

    def _check_weakness_at_level(
        self,
        relevant_15m: pd.DataFrame,
        level: float,
        direction: str,
        direction_signal: DirectionSignal,
        used_times: set,
    ) -> Optional[WeaknessSignal]:
        """
        Check if any M15 candle shows weakness at a given level.
        AGGRESSIVE: reduced sweep minimum, no strength-break early exit.
        """
        sweep_min = config.SWEEP_MIN_PIPS

        for i in range(len(relevant_15m)):
            candle = relevant_15m.iloc[i]
            candle_time = relevant_15m.index[i]

            # Skip if this time is already used by another weakness signal
            if candle_time in used_times:
                continue

            if direction == "bullish":
                # Sweep of a LOW level (price goes below then closes above)
                swept = candle["Low"] < level - sweep_min
                closed_back = candle["Close"] > level

                if swept and closed_back:
                    trigger_close = candle["Close"]
                    trigger_extreme = candle["Low"]

                    rect_top = trigger_close
                    rect_bottom = trigger_extreme

                    rect_size = rect_top - rect_bottom
                    if rect_size < config.MIN_RECTANGLE_SIZE_PIPS:
                        continue

                    in_fvg = is_level_in_fvg(level, self.fvgs, "bullish")
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

            else:  # bearish
                # Sweep of a HIGH level (price goes above then closes below)
                swept = candle["High"] > level + sweep_min
                closed_back = candle["Close"] < level

                if swept and closed_back:
                    trigger_close = candle["Close"]
                    trigger_extreme = candle["High"]

                    rect_top = trigger_extreme
                    rect_bottom = trigger_close

                    rect_size = rect_top - rect_bottom
                    if rect_size < config.MIN_RECTANGLE_SIZE_PIPS:
                        continue

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

        return None


class RectangleEntry:
    """
    Step 3: Rectangle Entry on the 1M timeframe.

    AGGRESSIVE MODE:
    - Wider entry window (up to 2 hours)
    - Smaller minimum rectangle size
    - Lower minimum RR ratio (2:1)
    """

    def __init__(self, df_1m: pd.DataFrame, df_15m: pd.DataFrame):
        self.df_1m = df_1m
        self.df_15m = df_15m

    def find_entry(
        self,
        weakness: WeaknessSignal,
    ) -> Optional[TradeSetup]:
        """
        Look for entry after a weakness signal.
        Falls back to 15M if 1M data not available.
        """
        direction = weakness.direction
        rect_top = weakness.rectangle_top
        rect_bottom = weakness.rectangle_bottom

        start_time = weakness.time

        # Try 1M first with wider window
        end_time_1m = start_time + pd.Timedelta(minutes=config.MAX_CANDLES_FOR_ENTRY)
        mask_1m = (self.df_1m.index > start_time) & (self.df_1m.index <= end_time_1m)
        entry_candles = self.df_1m[mask_1m]

        # Fallback to 15M with extended window
        if entry_candles.empty:
            end_time_15m = start_time + pd.Timedelta(hours=8)
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
                    stop_loss = rect_bottom - config.SPREAD

                    risk = entry_price - stop_loss
                    if risk <= 0:
                        continue

                    take_profit = entry_price + (risk * config.TARGET_RR_RATIO)

                    # Check for closer key level TP
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
                    stop_loss = rect_top + config.SPREAD

                    risk = stop_loss - entry_price
                    if risk <= 0:
                        continue

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
    AGGRESSIVE MODE: Relaxed validation (fewer filters).
    """

    @staticmethod
    def validate(
        direction_signal: DirectionSignal,
        weakness_signal: WeaknessSignal,
        trade_setup: TradeSetup,
    ) -> Tuple[bool, List[str]]:
        """
        Validate checklist items. Relaxed for aggressive trading.
        """
        failed = []

        if direction_signal is None:
            failed.append("No valid direction signal")

        if weakness_signal is None:
            failed.append("No wick rejection confirmed")

        if trade_setup is None:
            failed.append("No valid rectangle entry")
        elif trade_setup.rectangle_top <= trade_setup.rectangle_bottom:
            failed.append("Invalid rectangle dimensions")

        if trade_setup is not None and trade_setup.rr_ratio < config.MIN_RR_RATIO:
            failed.append(
                f"RR ratio {trade_setup.rr_ratio:.1f} below minimum {config.MIN_RR_RATIO}"
            )

        is_valid = len(failed) == 0
        return is_valid, failed


class CCTRectangleStrategy:
    """
    Main strategy class - AGGRESSIVE MODE.
    Generates maximum trade signals with relaxed filters and multiple
    weakness signals per direction candle.
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
        Run the full strategy pipeline with aggressive signal generation.
        Allows multiple weakness signals per direction and multiple trades.
        """
        self.trade_setups = []

        # Step 1: Get all direction signals
        direction_signals = self.direction_detector.detect_all()

        if not direction_signals:
            print("  No direction signals found on 4H timeframe.")
            return []

        print(f"  Found {len(direction_signals)} direction signals on 4H")

        # Step 2 & 3: For each direction signal, find multiple weakness/entry pairs
        for dir_signal in direction_signals:
            # Find ALL weakness signals (not just first)
            weaknesses = self.weakness_detector.find_all_weakness(dir_signal)

            if not weaknesses:
                continue

            for weakness in weaknesses:
                # Find rectangle entry for each weakness
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

        # Sort by entry time
        self.trade_setups.sort(key=lambda x: x.entry_time)

        # Deduplicate: remove trades with same entry time and direction
        # Also enforce minimum time gap between trades
        deduplicated = []
        min_gap_minutes = 30  # Minimum 30 minutes between entries
        for setup in self.trade_setups:
            if not deduplicated:
                deduplicated.append(setup)
                continue
            last = deduplicated[-1]
            time_gap = (setup.entry_time - last.entry_time).total_seconds() / 60
            # Allow same-time trades only if different direction
            if time_gap < min_gap_minutes and setup.direction == last.direction:
                continue
            # Skip exact duplicates (same time, same entry price)
            if (setup.entry_time == last.entry_time and
                    abs(setup.entry_price - last.entry_price) < 0.01):
                continue
            deduplicated.append(setup)

        self.trade_setups = deduplicated
        print(f"  After deduplication: {len(self.trade_setups)} unique trade setups")

        print(f"  Generated {len(self.trade_setups)} validated trade setups")
        return self.trade_setups
