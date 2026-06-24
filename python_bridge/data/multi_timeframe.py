"""
=============================================================
  Python ML Bridge - Multi-Timeframe Data Pipeline
  Fetches OHLCV for M1/M5/M15/H1/H4 timeframes, computes
  features per timeframe using MarketDataFetcher, and produces
  a combined feature matrix for institutional-grade analysis.

  Professional Logic:
    - Higher timeframes (H4/H1) define the trend direction
    - Medium timeframes (M15) identify key levels and structure
    - Lower timeframes (M5/M1) time precise entries
    - Features are aligned to the lowest timeframe's timestamps
=============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MultiTimeframeConfig, DataConfig
from data.market_data import MarketDataFetcher


class MultiTimeframeDataFetcher:
    """
    Fetches and aligns data across multiple timeframes.

    Professional traders use multi-timeframe analysis (MTA) to:
    1. Confirm trend direction on higher timeframes
    2. Identify support/resistance on medium timeframes
    3. Time entries on lower timeframes

    This class fetches M1/M5/M15/H1/H4 data and produces aligned
    feature matrices for model consumption.
    """

    def __init__(self, config: Optional[MultiTimeframeConfig] = None,
                 data_config: Optional[DataConfig] = None):
        self.config = config or MultiTimeframeConfig()
        self.data_config = data_config or DataConfig()
        self.market_data = MarketDataFetcher(self.data_config)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None

    def fetch_timeframe(self, timeframe: str,
                        ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a specific timeframe.

        Args:
            timeframe: One of '1m', '5m', '15m', '1h', '4h'
            ticker: Yahoo Finance ticker (defaults to config)

        Returns:
            DataFrame with OHLCV columns
        """
        ticker = ticker or self.data_config.yfinance_ticker
        period = self.config.periods.get(timeframe, "60d")

        # yfinance doesn't support '4h' directly - fetch 1h and resample
        if timeframe == "4h":
            return self._fetch_and_resample_4h(ticker, period)

        try:
            data = yf.download(ticker, period=period, interval=timeframe,
                               progress=False)
            if data.empty:
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.dropna(inplace=True)
            self._cache[f"{ticker}_{timeframe}"] = data
            return data
        except Exception as e:
            print(f"[MultiTF] Error fetching {ticker} {timeframe}: {e}")
            cached = self._cache.get(f"{ticker}_{timeframe}")
            return cached if cached is not None else pd.DataFrame()

    def _fetch_and_resample_4h(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch 1h data and resample to 4h bars."""
        try:
            data = yf.download(ticker, period=period, interval="1h",
                               progress=False)
            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.dropna(inplace=True)

            # Resample to 4-hour bars
            resampled = data.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()

            self._cache[f"{ticker}_4h"] = resampled
            return resampled
        except Exception as e:
            print(f"[MultiTF] Error fetching/resampling 4h: {e}")
            cached = self._cache.get(f"{ticker}_4h")
            return cached if cached is not None else pd.DataFrame()

    def compute_timeframe_features(self, timeframe: str,
                                   df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features for a specific timeframe.

        Uses MarketDataFetcher.compute_features() and adds a prefix
        to distinguish features across timeframes.

        Args:
            timeframe: Timeframe identifier for column prefix
            df: OHLCV DataFrame

        Returns:
            DataFrame with prefixed feature columns
        """
        if df.empty or len(df) < self.config.min_bars:
            return pd.DataFrame()

        features = self.market_data.compute_features(df)
        if features.empty:
            return pd.DataFrame()

        # Prefix all columns with timeframe identifier
        prefix = f"tf_{timeframe.replace('m', 'M').replace('h', 'H')}_"
        features.columns = [f"{prefix}{col}" for col in features.columns]

        self._feature_cache[timeframe] = features
        return features

    def _align_features(self, features_dict: Dict[str, pd.DataFrame],
                        target_tf: str = "1h") -> pd.DataFrame:
        """
        Align features from multiple timeframes to a common index.

        Higher timeframe features are forward-filled to match the target
        timeframe's timestamps. This ensures each bar has the most recent
        HTF context available at that point in time.

        Args:
            features_dict: Dict mapping timeframe -> features DataFrame
            target_tf: Target timeframe to align to

        Returns:
            Combined DataFrame with all timeframe features aligned
        """
        if not features_dict:
            return pd.DataFrame()

        # Use target timeframe index as base
        if target_tf not in features_dict:
            # Fall back to first available
            target_tf = list(features_dict.keys())[0]

        base_index = features_dict[target_tf].index
        aligned_frames = []

        for tf, features in features_dict.items():
            if tf == target_tf:
                aligned_frames.append(features)
            else:
                # Reindex to target, forward-fill higher TF features
                reindexed = features.reindex(base_index, method="ffill")
                aligned_frames.append(reindexed)

        if not aligned_frames:
            return pd.DataFrame()

        combined = pd.concat(aligned_frames, axis=1)
        combined.dropna(inplace=True)
        return combined

    def fetch_all_timeframes(self, ticker: Optional[str] = None,
                             target_tf: str = "1h") -> pd.DataFrame:
        """
        Fetch data for all configured timeframes and produce a combined
        feature matrix aligned to the target timeframe.

        This is the primary entry point for the multi-timeframe pipeline.
        It fetches M1/M5/M15/H1/H4 data, computes 50+ features per
        timeframe, and concatenates them into a single feature array.

        Args:
            ticker: Yahoo Finance ticker (defaults to config)
            target_tf: Timeframe to align all features to

        Returns:
            DataFrame with combined features from all timeframes.
            Shape: (num_bars, num_features_per_tf * num_timeframes)
        """
        ticker = ticker or self.data_config.yfinance_ticker
        features_dict: Dict[str, pd.DataFrame] = {}

        for tf in self.config.timeframes:
            df = self.fetch_timeframe(tf, ticker)
            if df.empty:
                continue

            features = self.compute_timeframe_features(tf, df)
            if not features.empty:
                features_dict[tf] = features

        if not features_dict:
            return pd.DataFrame()

        # Align all timeframe features to target
        if self.config.align_to_lowest:
            combined = self._align_features(features_dict, target_tf)
        else:
            # Just concatenate matching indices
            combined = pd.concat(features_dict.values(), axis=1, join="inner")
            combined.dropna(inplace=True)

        self._last_fetch = datetime.now()
        return combined

    def get_htf_trend_bias(self, ticker: Optional[str] = None) -> Dict[str, float]:
        """
        Get higher timeframe trend bias for trade confirmation.

        M15 trend confirmation for M1 scalping. Uses 5m and 15m timeframes
        as the "higher timeframe" bias for scalp entries on M1.
        This method returns a trend score per HTF.

        Returns:
            Dict with timeframe -> trend_score (-1 to +1, bearish to bullish)
        """
        ticker = ticker or self.data_config.yfinance_ticker
        trend_bias = {}

        for tf in ["5m", "15m"]:
            df = self.fetch_timeframe(tf, ticker)
            if df.empty or len(df) < 50:
                trend_bias[tf] = 0.0
                continue

            features = self.market_data.compute_features(df)
            if features.empty:
                trend_bias[tf] = 0.0
                continue

            # Compute trend score from multiple signals
            score = 0.0
            n_signals = 0

            # EMA alignment (price above all EMAs = bullish)
            close = features["close"].iloc[-1] if "close" in features.columns else 0
            for period in [9, 21, 50, 200]:
                col = f"close_vs_ema_{period}"
                if col in features.columns:
                    val = features[col].iloc[-1]
                    score += 1.0 if val > 0 else -1.0
                    n_signals += 1

            # ADX direction (DI+ > DI- = bullish)
            if "adx_pos" in features.columns and "adx_neg" in features.columns:
                di_diff = features["adx_pos"].iloc[-1] - features["adx_neg"].iloc[-1]
                score += np.clip(di_diff / 20.0, -1, 1)
                n_signals += 1

            # MACD histogram direction
            if "macd_histogram" in features.columns:
                macd_hist = features["macd_histogram"].iloc[-1]
                score += np.clip(macd_hist * 100, -1, 1)
                n_signals += 1

            # Momentum (20-bar)
            if "momentum_20" in features.columns:
                mom = features["momentum_20"].iloc[-1]
                score += np.clip(mom * 20, -1, 1)
                n_signals += 1

            trend_bias[tf] = np.clip(score / max(n_signals, 1), -1.0, 1.0)

        return trend_bias

    def get_multi_tf_feature_array(self, seq_length: int = 64,
                                   ticker: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get the latest multi-timeframe feature sequence for live prediction.

        Returns:
            Array of shape (1, seq_length, total_features) or None
        """
        combined = self.fetch_all_timeframes(ticker)
        if combined.empty or len(combined) < seq_length:
            return None

        data = combined.values[-seq_length:].astype(np.float32)

        # Normalize to zero mean, unit variance
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0) + 1e-10
        data = (data - means) / stds

        # Replace any remaining NaN/inf with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data.reshape(1, seq_length, -1)
