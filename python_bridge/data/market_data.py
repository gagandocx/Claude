"""
=============================================================
  Python ML Bridge - Market Data Module
  Fetches OHLCV data using yfinance and computes 50+ technical
  features for model input including RSI, MACD, Bollinger Bands,
  ATR, ADX, EMAs, and derived features.
=============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DataConfig, MODEL_DIR


class MarketDataFetcher:
    """Fetches and processes market data with technical indicators."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self._cache = {}
        self._last_fetch = None

    def fetch_ohlcv(self, ticker: Optional[str] = None,
                    period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: Bar interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        ticker = ticker or self.config.yfinance_ticker
        try:
            data = yf.download(ticker, period=period, interval=interval,
                               progress=False)
            if data.empty:
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.dropna(inplace=True)
            self._cache[ticker] = data
            self._last_fetch = datetime.now()
            return data
        except Exception as e:
            print(f"[MarketData] Error fetching {ticker}: {e}")
            return self._cache.get(ticker, pd.DataFrame())

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 50+ technical indicators and derived features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all computed features (NaN rows dropped)
        """
        if df.empty or len(df) < 200:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # --- Price-based features ---
        features["close"] = df["Close"]
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        features["close_open_range"] = (df["Close"] - df["Open"]) / df["Close"]

        # --- Moving Averages ---
        for period in self.config.ema_periods:
            features[f"ema_{period}"] = (
                ta.trend.ema_indicator(df["Close"], window=period)
            )
            features[f"close_vs_ema_{period}"] = (
                (df["Close"] - features[f"ema_{period}"]) / df["Close"]
            )

        # --- RSI ---
        features["rsi"] = ta.momentum.rsi(df["Close"],
                                          window=self.config.rsi_period)
        features["rsi_norm"] = features["rsi"] / 100.0

        # --- MACD ---
        macd = ta.trend.MACD(df["Close"],
                             window_slow=self.config.macd_slow,
                             window_fast=self.config.macd_fast,
                             window_sign=self.config.macd_signal)
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_histogram"] = macd.macd_diff()

        # --- Bollinger Bands ---
        bb = ta.volatility.BollingerBands(df["Close"],
                                          window=self.config.bb_period,
                                          window_dev=self.config.bb_std)
        features["bb_upper"] = bb.bollinger_hband()
        features["bb_lower"] = bb.bollinger_lband()
        features["bb_width"] = (
            (features["bb_upper"] - features["bb_lower"]) / df["Close"]
        )
        features["bb_position"] = (
            (df["Close"] - features["bb_lower"]) /
            (features["bb_upper"] - features["bb_lower"] + 1e-10)
        )

        # --- ATR (Average True Range) ---
        features["atr"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"],
            window=self.config.atr_period
        )
        features["atr_pct"] = features["atr"] / df["Close"]

        # --- ADX (Average Directional Index) ---
        features["adx"] = ta.trend.adx(
            df["High"], df["Low"], df["Close"],
            window=self.config.adx_period
        )
        features["adx_pos"] = ta.trend.adx_pos(
            df["High"], df["Low"], df["Close"],
            window=self.config.adx_period
        )
        features["adx_neg"] = ta.trend.adx_neg(
            df["High"], df["Low"], df["Close"],
            window=self.config.adx_period
        )

        # --- Stochastic Oscillator ---
        stoch = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"]
        )
        features["stoch_k"] = stoch.stoch() / 100.0
        features["stoch_d"] = stoch.stoch_signal() / 100.0

        # --- Volume features ---
        features["volume"] = df["Volume"]
        features["volume_sma"] = df["Volume"].rolling(20).mean()
        features["volume_ratio"] = (
            df["Volume"] / (features["volume_sma"] + 1)
        )

        # --- Volatility features ---
        features["volatility_5"] = features["returns"].rolling(5).std()
        features["volatility_20"] = features["returns"].rolling(20).std()
        features["volatility_ratio"] = (
            features["volatility_5"] / (features["volatility_20"] + 1e-10)
        )

        # --- Momentum features ---
        features["momentum_5"] = df["Close"].pct_change(5)
        features["momentum_10"] = df["Close"].pct_change(10)
        features["momentum_20"] = df["Close"].pct_change(20)

        # --- Williams %R ---
        features["williams_r"] = ta.momentum.williams_r(
            df["High"], df["Low"], df["Close"]
        ) / 100.0

        # --- CCI (Commodity Channel Index) ---
        features["cci"] = ta.trend.cci(
            df["High"], df["Low"], df["Close"]
        ) / 200.0  # Normalize

        # --- OBV (On-Balance Volume) ---
        features["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        features["obv_norm"] = (
            features["obv"] / (features["obv"].rolling(20).std() + 1)
        )

        # --- Ichimoku ---
        ich = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        features["ichimoku_a"] = ich.ichimoku_a()
        features["ichimoku_b"] = ich.ichimoku_b()
        features["ichimoku_base"] = ich.ichimoku_base_line()

        # --- Candlestick pattern features ---
        features["body_size"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-10)
        features["upper_shadow"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / (df["High"] - df["Low"] + 1e-10)
        features["lower_shadow"] = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / (df["High"] - df["Low"] + 1e-10)

        # --- Time features ---
        if hasattr(df.index, 'hour'):
            features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            features["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Drop rows with NaN
        features.dropna(inplace=True)

        return features

    def prepare_model_input(self, features: pd.DataFrame,
                            seq_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for model input with labels.

        Args:
            features: DataFrame of computed features
            seq_length: Sequence length for time series input

        Returns:
            Tuple of (X, y) where X is (num_samples, seq_length, num_features)
            and y is (num_samples,) with labels 0=SELL, 1=HOLD, 2=BUY
        """
        if features.empty or len(features) < seq_length + 10:
            return np.array([]), np.array([])

        # Normalize features
        feature_cols = [c for c in features.columns
                        if c not in ["close", "volume", "obv",
                                     "ichimoku_a", "ichimoku_b", "ichimoku_base"]]
        data = features[feature_cols].values.astype(np.float32)

        # Normalize each feature to zero mean, unit variance
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0) + 1e-10
        data = (data - means) / stds

        # Save normalization stats for use during inference
        self._save_normalization_stats(feature_cols, means, stds)

        # Generate labels based on future returns
        close_prices = features["close"].values
        future_returns = np.zeros(len(close_prices))
        lookahead = 5  # 5-bar lookahead for label generation
        for i in range(len(close_prices) - lookahead):
            future_returns[i] = (
                (close_prices[i + lookahead] - close_prices[i]) / close_prices[i]
            )

        # Classify: BUY if return > threshold, SELL if < -threshold, else HOLD
        threshold = np.std(future_returns[future_returns != 0]) * 0.5
        labels = np.ones(len(future_returns), dtype=np.int64)  # HOLD = 1
        labels[future_returns > threshold] = 2   # BUY = 2
        labels[future_returns < -threshold] = 0  # SELL = 0

        # Create sequences
        X, y = [], []
        for i in range(len(data) - seq_length - lookahead):
            X.append(data[i:i + seq_length])
            y.append(labels[i + seq_length - 1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _save_normalization_stats(self, feature_cols, means, stds):
        """Save normalization statistics to JSON for inference use."""
        stats_path = os.path.join(MODEL_DIR, "normalization_stats.json")
        os.makedirs(MODEL_DIR, exist_ok=True)
        stats = {
            "feature_cols": list(feature_cols),
            "means": means.tolist(),
            "stds": stds.tolist(),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f)

    def _load_normalization_stats(self):
        """Load saved normalization statistics. Returns (means, stds) or None."""
        stats_path = os.path.join(MODEL_DIR, "normalization_stats.json")
        if not os.path.exists(stats_path):
            return None
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            means = np.array(stats["means"], dtype=np.float32)
            stds = np.array(stats["stds"], dtype=np.float32)
            return means, stds
        except Exception:
            return None

    def get_latest_features(self, seq_length: int = 64) -> Optional[np.ndarray]:
        """
        Get the most recent feature sequence for live prediction.
        Uses training normalization statistics if available to ensure
        consistent feature distributions between training and inference.

        Returns:
            Array of shape (1, seq_length, num_features) or None
        """
        df = self.fetch_ohlcv()
        if df.empty:
            return None

        features = self.compute_features(df)
        if features.empty or len(features) < seq_length:
            return None

        feature_cols = [c for c in features.columns
                        if c not in ["close", "volume", "obv",
                                     "ichimoku_a", "ichimoku_b", "ichimoku_base"]]
        data = features[feature_cols].values[-seq_length:].astype(np.float32)

        # Use saved training normalization stats if available
        saved_stats = self._load_normalization_stats()
        if saved_stats is not None:
            means, stds = saved_stats
            # Handle case where feature count may differ
            if len(means) == data.shape[1]:
                data = (data - means) / stds
            else:
                # Fallback to per-window normalization if feature count mismatch
                means = np.nanmean(data, axis=0)
                stds = np.nanstd(data, axis=0) + 1e-10
                data = (data - means) / stds
        else:
            # Fallback to per-window normalization if no saved stats
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0) + 1e-10
            data = (data - means) / stds

        return data.reshape(1, seq_length, -1)

    def get_current_atr(self) -> float:
        """Get the current ATR value for position sizing."""
        df = self.fetch_ohlcv()
        if df.empty:
            return 0.0
        atr = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"],
            window=self.config.atr_period
        )
        return float(atr.iloc[-1]) if not atr.empty else 0.0
