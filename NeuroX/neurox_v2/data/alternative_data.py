"""
=============================================================
  Python ML Bridge - Alternative Data Module
  Fetches VIX, DXY, Treasury Yields, Oil, and COT data.
  Computes cross-asset correlation signals for gold trading.
=============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DataConfig


class AlternativeDataFetcher:
    """Fetches alternative data feeds for cross-asset analysis."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_vix(self, period: str = "3mo") -> pd.Series:
        """Fetch VIX (fear index) data."""
        try:
            data = yf.download(self.config.vix_ticker, period=period,
                               interval="1d", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data["Close"]
        except Exception as e:
            print(f"[AltData] Error fetching VIX: {e}")
        return pd.Series(dtype=float)

    def fetch_dxy(self, period: str = "3mo") -> pd.Series:
        """Fetch US Dollar Index (DXY) data."""
        try:
            data = yf.download(self.config.dxy_ticker, period=period,
                               interval="1d", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data["Close"]
        except Exception as e:
            print(f"[AltData] Error fetching DXY: {e}")
        return pd.Series(dtype=float)

    def fetch_yields(self, period: str = "3mo") -> Dict[str, pd.Series]:
        """Fetch US Treasury yields (10Y and 2Y)."""
        yields = {}
        for name, ticker in [("10y", self.config.yield_10y_ticker),
                             ("2y", self.config.yield_2y_ticker)]:
            try:
                data = yf.download(ticker, period=period,
                                   interval="1d", progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    yields[name] = data["Close"]
            except Exception as e:
                print(f"[AltData] Error fetching {name} yield: {e}")
        return yields

    def fetch_oil(self, period: str = "3mo") -> pd.Series:
        """Fetch WTI Crude Oil price data."""
        try:
            data = yf.download(self.config.oil_ticker, period=period,
                               interval="1d", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data["Close"]
        except Exception as e:
            print(f"[AltData] Error fetching Oil: {e}")
        return pd.Series(dtype=float)

    def compute_correlations(self, gold_prices: pd.Series) -> Dict[str, float]:
        """
        Compute rolling correlations between gold and other assets.

        Args:
            gold_prices: Gold price series

        Returns:
            Dict with correlation values for each asset
        """
        correlations = {}
        window = 20  # 20-day rolling correlation

        dxy = self.fetch_dxy()
        if not dxy.empty and len(dxy) > window:
            # Gold typically negatively correlated with DXY
            aligned = pd.DataFrame({"gold": gold_prices, "dxy": dxy}).dropna()
            if len(aligned) > window:
                correlations["gold_dxy"] = float(
                    aligned["gold"].pct_change().corr(
                        aligned["dxy"].pct_change()
                    )
                )

        oil = self.fetch_oil()
        if not oil.empty and len(oil) > window:
            aligned = pd.DataFrame({"gold": gold_prices, "oil": oil}).dropna()
            if len(aligned) > window:
                correlations["gold_oil"] = float(
                    aligned["gold"].pct_change().corr(
                        aligned["oil"].pct_change()
                    )
                )

        vix = self.fetch_vix()
        if not vix.empty and len(vix) > window:
            # Gold typically positively correlated with VIX (safe haven)
            aligned = pd.DataFrame({"gold": gold_prices, "vix": vix}).dropna()
            if len(aligned) > window:
                correlations["gold_vix"] = float(
                    aligned["gold"].pct_change().corr(
                        aligned["vix"].pct_change()
                    )
                )

        return correlations

    def get_alternative_features(self) -> Dict[str, float]:
        """
        Get all alternative data features for model input.

        Returns:
            Dict with normalized alternative data signals
        """
        features = {}

        # VIX level and change
        vix = self.fetch_vix()
        if not vix.empty:
            features["vix_level"] = float(vix.iloc[-1])
            features["vix_change_5d"] = float(
                (vix.iloc[-1] - vix.iloc[-5]) / vix.iloc[-5]
            ) if len(vix) >= 5 else 0.0
            features["vix_high"] = 1.0 if vix.iloc[-1] > 25 else 0.0
            features["vix_extreme"] = 1.0 if vix.iloc[-1] > 35 else 0.0
        else:
            features["vix_level"] = 20.0
            features["vix_change_5d"] = 0.0
            features["vix_high"] = 0.0
            features["vix_extreme"] = 0.0

        # DXY level and change (inverse relationship with gold)
        dxy = self.fetch_dxy()
        if not dxy.empty:
            features["dxy_level"] = float(dxy.iloc[-1])
            features["dxy_change_5d"] = float(
                (dxy.iloc[-1] - dxy.iloc[-5]) / dxy.iloc[-5]
            ) if len(dxy) >= 5 else 0.0
            features["dxy_weakening"] = 1.0 if features["dxy_change_5d"] < -0.005 else 0.0
        else:
            features["dxy_level"] = 100.0
            features["dxy_change_5d"] = 0.0
            features["dxy_weakening"] = 0.0

        # Treasury yields
        yields = self.fetch_yields()
        if "10y" in yields and not yields["10y"].empty:
            features["yield_10y"] = float(yields["10y"].iloc[-1])
            features["yield_10y_change"] = float(
                yields["10y"].iloc[-1] - yields["10y"].iloc[-5]
            ) if len(yields["10y"]) >= 5 else 0.0
        else:
            features["yield_10y"] = 4.0
            features["yield_10y_change"] = 0.0

        # Yield curve (2Y-10Y spread if available)
        if "10y" in yields and "2y" in yields:
            if not yields["10y"].empty and not yields["2y"].empty:
                features["yield_curve"] = float(
                    yields["10y"].iloc[-1] - yields["2y"].iloc[-1]
                )
                features["yield_curve_inverted"] = (
                    1.0 if features["yield_curve"] < 0 else 0.0
                )
            else:
                features["yield_curve"] = 0.0
                features["yield_curve_inverted"] = 0.0
        else:
            features["yield_curve"] = 0.0
            features["yield_curve_inverted"] = 0.0

        # Oil prices
        oil = self.fetch_oil()
        if not oil.empty:
            features["oil_price"] = float(oil.iloc[-1])
            features["oil_change_5d"] = float(
                (oil.iloc[-1] - oil.iloc[-5]) / oil.iloc[-5]
            ) if len(oil) >= 5 else 0.0
        else:
            features["oil_price"] = 70.0
            features["oil_change_5d"] = 0.0

        return features

    def get_features_array(self) -> np.ndarray:
        """
        Get alternative data as numpy array for model input.

        Returns:
            Array of shape (n_features,) with normalized values
        """
        features = self.get_alternative_features()
        # Normalize key features
        arr = np.array([
            features.get("vix_level", 20.0) / 50.0,        # Normalize VIX
            features.get("vix_change_5d", 0.0),
            features.get("vix_high", 0.0),
            features.get("dxy_change_5d", 0.0),
            features.get("dxy_weakening", 0.0),
            features.get("yield_10y_change", 0.0),
            features.get("yield_curve", 0.0) / 2.0,
            features.get("yield_curve_inverted", 0.0),
            features.get("oil_change_5d", 0.0),
        ], dtype=np.float32)
        return arr
