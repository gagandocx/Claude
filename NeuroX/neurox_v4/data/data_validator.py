"""
=============================================================
  NeuroX - Live Data Validation Module

  Validates incoming market data to prevent garbage-in-garbage-out:
    - NaN / inf detection
    - Zero-volume bar detection
    - Price gap detection (> N x ATR)
    - Price sanity range checks (XAUUSD $500-$5000)
    - Tick data staleness detection (file mtime based)
    - Feature schema validation (expected feature count)

  Integration: called from main.py run_cycle() after data fetch.
  Non-critical issues log warnings; only critical failures skip cycle.
=============================================================
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DataValidatorConfig

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a data validation check."""

    def __init__(self):
        self.is_critical: bool = False
        self.warnings: list = []
        self.errors: list = []
        self.stats: Dict[str, float] = {}

    @property
    def is_valid(self) -> bool:
        """Data is valid if there are no critical errors."""
        return not self.is_critical

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_critical = True

    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append(f"ERRORS: {'; '.join(self.errors)}")
        if self.warnings:
            parts.append(f"WARNINGS: {'; '.join(self.warnings)}")
        return " | ".join(parts) if parts else "OK"


class DataValidator:
    """
    Live data validation for incoming market data.

    Performs sanity checks on OHLCV DataFrames and feature arrays
    to detect data quality issues before they corrupt model predictions.

    Usage:
        validator = DataValidator(DataValidatorConfig())
        result = validator.validate(df)
        if not result.is_valid:
            logger.error(f"Data validation failed: {result.summary()}")
            # skip cycle

        feature_result = validator.validate_features(feature_array)
    """

    def __init__(self, config: Optional[DataValidatorConfig] = None):
        self.config = config or DataValidatorConfig()
        self._last_valid_price: Optional[float] = None
        logger.info(
            "[DataValidator] Initialized. "
            f"Price range: ${self.config.min_price}-${self.config.max_price}, "
            f"Staleness: {self.config.staleness_seconds}s, "
            f"Expected features: {self.config.expected_feature_count}"
        )

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate an OHLCV DataFrame for data quality issues.

        Checks performed:
            1. NaN/inf detection (critical if >50%)
            2. Zero-volume bar detection (warning)
            3. Price gap detection (warning if gap > max_gap_atr_mult x ATR)
            4. Price sanity range check (critical if outside $500-$5000)

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns

        Returns:
            ValidationResult with warnings/errors and criticality flag
        """
        result = ValidationResult()

        if df is None or df.empty:
            result.add_error("DataFrame is empty or None")
            return result

        # 1. NaN / inf detection
        self._check_nan_inf(df, result)

        # 2. Zero-volume detection
        self._check_zero_volume(df, result)

        # 3. Price gap detection
        self._check_price_gaps(df, result)

        # 4. Price sanity range check
        self._check_price_sanity(df, result)

        return result

    def validate_features(self, features: np.ndarray) -> ValidationResult:
        """
        Validate feature array for schema compliance and data quality.

        Args:
            features: numpy array of shape (seq_length, n_features) or
                      (batch, seq_length, n_features)

        Returns:
            ValidationResult with schema validation results
        """
        result = ValidationResult()

        if features is None:
            result.add_error("Feature array is None")
            return result

        # Determine feature dimension
        if features.ndim == 2:
            n_features = features.shape[1]
        elif features.ndim == 3:
            n_features = features.shape[2]
        else:
            result.add_warning(
                f"Unexpected feature array dimensions: {features.ndim}D "
                f"(expected 2D or 3D)"
            )
            return result

        expected = self.config.expected_feature_count

        # Schema validation
        if n_features != expected:
            result.add_warning(
                f"Feature count mismatch: got {n_features}, expected {expected}. "
                f"Model may produce degraded predictions."
            )
            result.stats["feature_count"] = n_features
            result.stats["expected_feature_count"] = expected

        # NaN/inf check on feature array
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        total_elements = features.size

        if total_elements > 0:
            nan_pct = nan_count / total_elements
            result.stats["feature_nan_pct"] = nan_pct

            if nan_pct > self.config.max_nan_pct:
                result.add_error(
                    f"Feature array has {nan_pct:.1%} NaN values "
                    f"(threshold: {self.config.max_nan_pct:.1%})"
                )
            elif nan_count > 0:
                result.add_warning(
                    f"Feature array has {nan_count} NaN values ({nan_pct:.2%})"
                )

            if inf_count > 0:
                result.add_warning(
                    f"Feature array has {inf_count} inf values"
                )

        return result

    def is_tick_stale(self, tick_file: str,
                      max_age_seconds: Optional[int] = None) -> bool:
        """
        Check if tick data file is stale based on file modification time.

        Args:
            tick_file: Path to the tick data CSV file
            max_age_seconds: Maximum age in seconds before considered stale.
                           Defaults to config.staleness_seconds.

        Returns:
            True if tick data is stale (file too old or missing)
        """
        max_age = max_age_seconds or self.config.staleness_seconds

        if not os.path.exists(tick_file):
            return True

        try:
            mtime = os.path.getmtime(tick_file)
            age = time.time() - mtime
            if age > max_age:
                logger.debug(
                    f"[DataValidator] Tick data stale: "
                    f"age={age:.0f}s > threshold={max_age}s"
                )
                return True
            return False
        except OSError:
            return True

    # ── Internal validation methods ───────────────────────────────────────

    def _check_nan_inf(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for NaN and inf values in the DataFrame."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return

        total_cells = numeric_df.size
        nan_count = numeric_df.isna().sum().sum()
        inf_count = np.isinf(numeric_df.values).sum() if total_cells > 0 else 0

        nan_pct = nan_count / total_cells if total_cells > 0 else 0.0
        result.stats["nan_pct"] = nan_pct
        result.stats["nan_count"] = int(nan_count)

        # Critical: >50% NaN means data is unusable
        if nan_pct > 0.50:
            result.add_error(
                f"Data has {nan_pct:.1%} NaN values ({nan_count}/{total_cells}) "
                f"- too corrupted for reliable predictions"
            )
        elif nan_pct > self.config.max_nan_pct:
            result.add_warning(
                f"Data has {nan_pct:.1%} NaN values "
                f"(threshold: {self.config.max_nan_pct:.1%})"
            )

        if inf_count > 0:
            result.add_warning(f"Data has {inf_count} inf values")

    def _check_zero_volume(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for zero-volume bars (indicates data gaps or market closure)."""
        if "Volume" not in df.columns:
            return

        volume = df["Volume"]
        zero_count = (volume == 0).sum()
        total_bars = len(volume)

        if total_bars == 0:
            return

        zero_pct = zero_count / total_bars
        result.stats["zero_volume_pct"] = zero_pct
        result.stats["zero_volume_bars"] = int(zero_count)

        if zero_pct > 0.3:
            result.add_warning(
                f"High zero-volume rate: {zero_pct:.1%} of bars "
                f"({zero_count}/{total_bars}) - possible data gaps"
            )
        elif zero_count > 0:
            logger.debug(
                f"[DataValidator] {zero_count} zero-volume bars detected"
            )

    def _check_price_gaps(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for price gaps larger than N x ATR."""
        if "Close" not in df.columns or len(df) < 20:
            return

        close = df["Close"].values
        if len(close) < 20:
            return

        # Compute ATR for gap detection
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].values
            low = df["Low"].values
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]

            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - prev_close),
                    np.abs(low - prev_close)
                )
            )
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        else:
            # Fallback: use price changes as proxy
            changes = np.abs(np.diff(close))
            atr = np.mean(changes[-14:]) if len(changes) >= 14 else np.mean(changes)

        if atr <= 0:
            return

        # Detect gaps
        price_changes = np.abs(np.diff(close))
        gap_threshold = atr * self.config.max_gap_atr_mult
        large_gaps = np.where(price_changes > gap_threshold)[0]

        result.stats["atr"] = float(atr)
        result.stats["max_gap"] = float(np.max(price_changes)) if len(price_changes) > 0 else 0.0

        if len(large_gaps) > 0:
            max_gap_size = float(np.max(price_changes[large_gaps]))
            result.add_warning(
                f"Detected {len(large_gaps)} price gap(s) > {self.config.max_gap_atr_mult}x ATR. "
                f"Largest: ${max_gap_size:.2f} (ATR=${atr:.2f})"
            )

    def _check_price_sanity(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check if price is within sane range for XAUUSD."""
        if "Close" not in df.columns:
            return

        close = df["Close"].dropna()
        if close.empty:
            return

        latest_price = float(close.iloc[-1])
        min_price = float(close.min())
        max_price = float(close.max())

        result.stats["latest_price"] = latest_price
        result.stats["min_price"] = min_price
        result.stats["max_price"] = max_price

        # Track last valid price
        if self.config.min_price <= latest_price <= self.config.max_price:
            self._last_valid_price = latest_price

        # Critical: price completely outside sane range
        if latest_price < self.config.min_price or latest_price > self.config.max_price:
            result.add_error(
                f"Price ${latest_price:.2f} outside sane range "
                f"(${self.config.min_price}-${self.config.max_price}). "
                f"Possible bad data feed."
            )
        elif min_price < self.config.min_price * 0.9:
            result.add_warning(
                f"Historical price ${min_price:.2f} near lower sanity bound"
            )
