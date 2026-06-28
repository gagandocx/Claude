"""
=============================================================
  Python ML Bridge v3 - Correlation Regime Detector (Tier 2)

  Uses cross-pair data (DXY, bonds/TNX, equities/SPY) to detect
  the current correlation regime state affecting gold direction.

  Regime states:
    - risk_on: Equities up, gold down/flat (risk appetite high)
    - risk_off: Equities down, gold up (flight to safety)
    - dollar_driven: Gold inversely tracks DXY strongly
    - decorrelated: Correlations broken down (unpredictable)

  Provides confidence adjustments based on signal direction
  alignment with the detected correlation regime.
=============================================================
"""

import os
import sys
import logging
from typing import Dict, Optional
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CorrelationRegimeConfig

logger = logging.getLogger(__name__)


class CorrelationRegimeDetector:
    """
    Detects cross-market correlation regime states for gold trading.

    Gold's relationship with DXY, bonds, and equities shifts over time.
    Sometimes gold moves purely inverse to USD (dollar_driven), sometimes
    it acts as a safe haven (risk_off), sometimes correlations break down
    entirely (decorrelated). Knowing the current regime allows proper
    confidence adjustment on signals.

    Usage:
        detector = CorrelationRegimeDetector()
        detector.update({'gold': [...], 'dxy': [...], 'tnx': [...], 'spy': [...]})
        regime = detector.get_regime()
        adj = detector.get_confidence_adjustment('BUY')
    """

    def __init__(self, config: Optional[CorrelationRegimeConfig] = None):
        self.config = config or CorrelationRegimeConfig()

        # Price history buffers for each asset
        self._gold_prices: deque = deque(maxlen=self.config.regime_lookback + 50)
        self._dxy_prices: deque = deque(maxlen=self.config.regime_lookback + 50)
        self._tnx_prices: deque = deque(maxlen=self.config.regime_lookback + 50)
        self._spy_prices: deque = deque(maxlen=self.config.regime_lookback + 50)

        # Current regime state
        self._current_regime: str = 'decorrelated'
        self._regime_confidence: float = 0.0
        self._last_rebalance_count: int = 0
        self._update_count: int = 0

        # Rolling correlations
        self._gold_dxy_corr: float = 0.0
        self._gold_tnx_corr: float = 0.0
        self._gold_spy_corr: float = 0.0

        logger.info("[CorrelationRegime] Initialized. lookback=%d, "
                    "dxy_threshold=%.2f, bond_threshold=%.2f, rebalance_interval=%d",
                    self.config.regime_lookback,
                    self.config.dxy_correlation_threshold,
                    self.config.bond_correlation_threshold,
                    self.config.rebalance_interval)

    def update(self, pair_data: Dict) -> None:
        """
        Update internal state with latest cross-pair price data.

        Args:
            pair_data: Dict with keys mapping asset names to price values or arrays.
                       Accepted keys: 'gold'/'xauusd', 'dxy', 'tnx'/'bonds', 'spy'/'equities'
                       Values can be: float (single price), list/array of recent prices
        """
        self._update_count += 1

        # Extract and store prices from pair_data
        self._store_prices(pair_data)

        # Rebalance (recompute regime) at configured interval
        bars_since_rebalance = self._update_count - self._last_rebalance_count
        if bars_since_rebalance >= self.config.rebalance_interval:
            self._compute_regime()
            self._last_rebalance_count = self._update_count

    def _store_prices(self, pair_data: Dict) -> None:
        """Store price data into internal buffers."""
        # Gold prices
        gold_val = pair_data.get('gold', pair_data.get('xauusd', None))
        if gold_val is not None:
            if isinstance(gold_val, (list, np.ndarray)):
                for p in gold_val:
                    self._gold_prices.append(float(p))
            else:
                self._gold_prices.append(float(gold_val))

        # DXY prices
        dxy_val = pair_data.get('dxy', None)
        if dxy_val is not None:
            if isinstance(dxy_val, (list, np.ndarray)):
                for p in dxy_val:
                    self._dxy_prices.append(float(p))
            else:
                self._dxy_prices.append(float(dxy_val))

        # TNX / Bond prices
        tnx_val = pair_data.get('tnx', pair_data.get('bonds', None))
        if tnx_val is not None:
            if isinstance(tnx_val, (list, np.ndarray)):
                for p in tnx_val:
                    self._tnx_prices.append(float(p))
            else:
                self._tnx_prices.append(float(tnx_val))

        # SPY / Equities prices
        spy_val = pair_data.get('spy', pair_data.get('equities', None))
        if spy_val is not None:
            if isinstance(spy_val, (list, np.ndarray)):
                for p in spy_val:
                    self._spy_prices.append(float(p))
            else:
                self._spy_prices.append(float(spy_val))

    def _compute_regime(self) -> None:
        """
        Compute the current correlation regime from price history.

        Uses rolling Pearson correlations between gold returns and
        DXY/TNX/SPY returns to classify the regime state.
        """
        lookback = self.config.regime_lookback
        min_data = max(20, lookback // 2)

        # Need sufficient data for all pairs to compute meaningful correlations
        has_dxy = len(self._dxy_prices) >= min_data
        has_tnx = len(self._tnx_prices) >= min_data
        has_spy = len(self._spy_prices) >= min_data
        has_gold = len(self._gold_prices) >= min_data

        if not has_gold:
            self._current_regime = 'decorrelated'
            self._regime_confidence = 0.0
            return

        # Compute returns for gold
        gold_arr = np.array(list(self._gold_prices)[-lookback:])
        gold_returns = np.diff(gold_arr) / (gold_arr[:-1] + 1e-10)

        # Compute correlations with each asset
        self._gold_dxy_corr = 0.0
        self._gold_tnx_corr = 0.0
        self._gold_spy_corr = 0.0

        if has_dxy:
            dxy_arr = np.array(list(self._dxy_prices)[-lookback:])
            if len(dxy_arr) >= len(gold_arr):
                dxy_arr = dxy_arr[-len(gold_arr):]
            else:
                gold_returns_trim = gold_returns[-(len(dxy_arr) - 1):]
                dxy_arr_for_ret = dxy_arr
                dxy_returns = np.diff(dxy_arr_for_ret) / (dxy_arr_for_ret[:-1] + 1e-10)
                self._gold_dxy_corr = self._safe_correlation(
                    gold_returns_trim if len(gold_returns_trim) == len(dxy_returns) else gold_returns[-len(dxy_returns):],
                    dxy_returns
                )
                dxy_arr = None  # skip below

            if dxy_arr is not None and len(dxy_arr) == len(gold_arr):
                dxy_returns = np.diff(dxy_arr) / (dxy_arr[:-1] + 1e-10)
                self._gold_dxy_corr = self._safe_correlation(gold_returns, dxy_returns)

        if has_tnx:
            tnx_arr = np.array(list(self._tnx_prices)[-lookback:])
            min_len = min(len(gold_arr), len(tnx_arr))
            gold_arr_trim = gold_arr[-min_len:]
            tnx_arr_trim = tnx_arr[-min_len:]
            gold_ret_trim = np.diff(gold_arr_trim) / (gold_arr_trim[:-1] + 1e-10)
            tnx_returns = np.diff(tnx_arr_trim) / (tnx_arr_trim[:-1] + 1e-10)
            self._gold_tnx_corr = self._safe_correlation(gold_ret_trim, tnx_returns)

        if has_spy:
            spy_arr = np.array(list(self._spy_prices)[-lookback:])
            min_len = min(len(gold_arr), len(spy_arr))
            gold_arr_trim = gold_arr[-min_len:]
            spy_arr_trim = spy_arr[-min_len:]
            gold_ret_trim = np.diff(gold_arr_trim) / (gold_arr_trim[:-1] + 1e-10)
            spy_returns = np.diff(spy_arr_trim) / (spy_arr_trim[:-1] + 1e-10)
            self._gold_spy_corr = self._safe_correlation(gold_ret_trim, spy_returns)

        # Classify regime based on correlation patterns
        self._classify_regime()

        logger.debug("[CorrelationRegime] Regime=%s (conf=%.2f) | "
                     "gold-dxy=%.3f, gold-tnx=%.3f, gold-spy=%.3f",
                     self._current_regime, self._regime_confidence,
                     self._gold_dxy_corr, self._gold_tnx_corr, self._gold_spy_corr)

    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation with safety checks."""
        if len(x) < 5 or len(y) < 5:
            return 0.0

        min_len = min(len(x), len(y))
        x = x[-min_len:]
        y = y[-min_len:]

        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0

        try:
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except (ValueError, FloatingPointError):
            return 0.0

    def _classify_regime(self) -> None:
        """
        Classify regime from computed correlations.

        Logic:
            - dollar_driven: gold-DXY strongly negative (< threshold, e.g., -0.5)
            - risk_off: gold-SPY strongly negative AND gold-TNX positive
              (gold up when stocks down, safety bid)
            - risk_on: gold-SPY positive AND gold going up less than stocks
              (risk appetite dominates, gold lags)
            - decorrelated: no strong correlation pattern detected
        """
        dxy_threshold = self.config.dxy_correlation_threshold  # e.g., -0.5
        bond_threshold = self.config.bond_correlation_threshold  # e.g., 0.3

        # Dollar-driven: strong inverse correlation with DXY
        if self._gold_dxy_corr < dxy_threshold:
            self._current_regime = 'dollar_driven'
            self._regime_confidence = min(abs(self._gold_dxy_corr), 1.0)
            return

        # Risk-off: gold negatively correlated with equities (safe haven)
        if self._gold_spy_corr < -0.3:
            self._current_regime = 'risk_off'
            self._regime_confidence = min(abs(self._gold_spy_corr), 1.0)
            return

        # Risk-on: gold positively correlated with equities (all assets up)
        if self._gold_spy_corr > 0.4:
            self._current_regime = 'risk_on'
            self._regime_confidence = min(self._gold_spy_corr, 1.0)
            return

        # Decorrelated: no strong pattern
        self._current_regime = 'decorrelated'
        # Confidence is based on how weak all correlations are
        max_abs_corr = max(abs(self._gold_dxy_corr),
                           abs(self._gold_tnx_corr),
                           abs(self._gold_spy_corr))
        self._regime_confidence = 1.0 - min(max_abs_corr, 1.0)

    def get_regime(self) -> str:
        """
        Get the current detected correlation regime.

        Returns:
            One of: 'risk_on', 'risk_off', 'dollar_driven', 'decorrelated'
        """
        return self._current_regime

    def get_regime_confidence(self) -> float:
        """
        Get confidence level of the current regime classification.

        Returns:
            Float in [0, 1]: how confident we are in the regime detection
        """
        return self._regime_confidence

    def get_confidence_adjustment(self, signal_direction: str) -> float:
        """
        Get confidence adjustment for a signal based on current correlation regime.

        The adjustment reflects how well the signal aligns with the detected
        regime. For example:
            - risk_off + BUY gold = positive (gold is a safe haven, buying aligns)
            - dollar_driven + strong DXY + BUY gold = negative (USD up = gold down)
            - decorrelated = no adjustment (can't trust correlations)

        Args:
            signal_direction: 'BUY' or 'SELL'

        Returns:
            Float: confidence adjustment to add to timing_confidence.
            Positive = boost, negative = penalty. Range: [-0.10, +0.05]
        """
        regime = self._current_regime
        confidence = self._regime_confidence

        # Scale adjustment by regime confidence (low confidence = smaller effect)
        scale = min(confidence, 1.0)

        if regime == 'decorrelated':
            # No reliable information - no adjustment
            return 0.0

        if regime == 'risk_off':
            # Safe haven mode: gold benefits from fear
            if signal_direction == 'BUY':
                # Buying gold in risk-off = aligned
                return 0.05 * scale
            else:
                # Selling gold in risk-off = against the flow
                return -0.05 * scale

        if regime == 'risk_on':
            # Risk appetite high: gold is less attractive
            if signal_direction == 'BUY':
                # Buying gold when risk is on = slight headwind
                return -0.03 * scale
            else:
                # Selling gold when risk is on = slight tailwind
                return 0.03 * scale

        if regime == 'dollar_driven':
            # Gold inversely tracking DXY
            # Check DXY direction from recent prices
            if len(self._dxy_prices) >= 5:
                dxy_recent = list(self._dxy_prices)[-5:]
                dxy_trend = dxy_recent[-1] - dxy_recent[0]

                if dxy_trend > 0:
                    # DXY rising (strong dollar)
                    if signal_direction == 'BUY':
                        # Buying gold against strong dollar = penalty
                        return -0.10 * scale
                    else:
                        # Selling gold with strong dollar = aligned
                        return 0.05 * scale
                elif dxy_trend < 0:
                    # DXY falling (weak dollar)
                    if signal_direction == 'BUY':
                        # Buying gold with weak dollar = aligned
                        return 0.05 * scale
                    else:
                        # Selling gold against weak dollar = penalty
                        return -0.10 * scale

            # DXY flat or insufficient data
            return 0.0

        return 0.0

    def compute_features(self) -> np.ndarray:
        """
        Compute correlation features as a numpy array for model input.

        Returns a fixed-size array of 6 features:
            [gold_dxy_corr, gold_tnx_corr, gold_spy_corr,
             regime_encoded, regime_confidence, dxy_momentum]

        Regime encoding:
            risk_on=0.0, risk_off=0.25, dollar_driven=0.5, decorrelated=0.75

        Returns:
            np.ndarray of shape (6,) with normalized correlation features.
        """
        regime_encoding = {
            'risk_on': 0.0,
            'risk_off': 0.25,
            'dollar_driven': 0.5,
            'decorrelated': 0.75,
        }

        # DXY momentum (recent 5-bar return)
        dxy_momentum = 0.0
        if len(self._dxy_prices) >= 5:
            dxy_recent = list(self._dxy_prices)[-5:]
            if dxy_recent[0] > 0:
                dxy_momentum = (dxy_recent[-1] - dxy_recent[0]) / dxy_recent[0]

        features = np.array([
            self._gold_dxy_corr,
            self._gold_tnx_corr,
            self._gold_spy_corr,
            regime_encoding.get(self._current_regime, 0.75),
            self._regime_confidence,
            dxy_momentum,
        ], dtype=np.float32)

        return features

    def get_state_info(self) -> Dict:
        """Get full state for logging/debugging."""
        return {
            'regime': self._current_regime,
            'confidence': self._regime_confidence,
            'gold_dxy_corr': self._gold_dxy_corr,
            'gold_tnx_corr': self._gold_tnx_corr,
            'gold_spy_corr': self._gold_spy_corr,
            'gold_buffer_size': len(self._gold_prices),
            'dxy_buffer_size': len(self._dxy_prices),
            'tnx_buffer_size': len(self._tnx_prices),
            'spy_buffer_size': len(self._spy_prices),
            'update_count': self._update_count,
        }
