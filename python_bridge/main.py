"""
=============================================================
  Python ML Bridge - Main Entry Point
  Runs continuous prediction loop:
    1. Fetch market data
    2. Compute features
    3. Run ensemble models
    4. Generate trade signal
    5. Write to bridge file for MT5
  Configurable interval (1min for M1, 5min for M5).
=============================================================
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    MainConfig, DataConfig, SignalConfig, RiskConfig,
    TransformerConfig, LSTMConfig, EnsembleConfig,
    MultiTimeframeConfig, NewsFilterConfig, MultiPairConfig,
    MODEL_DIR, LOG_DIR
)
from data.market_data import MarketDataFetcher
from data.sentiment import SentimentAnalyzer
from data.alternative_data import AlternativeDataFetcher
from data.multi_timeframe import MultiTimeframeDataFetcher
from data.news_calendar import NewsCalendarFilter
from models.ensemble import EnsembleManager
from strategies.signal_generator import SignalGenerator
from strategies.regime_detector import RegimeDetector
from strategies.risk_manager import RiskManager
from signals.bridge import MT5Bridge


# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logging(config: MainConfig) -> logging.Logger:
    """Configure logging for the bridge system."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        LOG_DIR, f"bridge_{datetime.now().strftime('%Y%m%d')}.log"
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("PythonBridge")


# ─────────────────────────────────────────────
#  MAIN BRIDGE CLASS
# ─────────────────────────────────────────────
class PythonMLBridge:
    """
    Main bridge system that orchestrates the entire ML pipeline.

    Lifecycle:
        1. Initialize all components
        2. Load model checkpoints (if available)
        3. Enter main loop:
           a. Fetch latest market data
           b. Compute features
           c. (Optional) Fetch sentiment and alt data
           d. Run ensemble prediction
           e. Generate signal through risk filters
           f. Write signal to bridge file
           g. Check for MT5 confirmations
           h. Sleep until next cycle
        4. Graceful shutdown on interrupt
    """

    def __init__(self, config: Optional[MainConfig] = None):
        self.config = config or MainConfig()
        self.logger = setup_logging(self.config)
        self._running = False

        # Initialize components
        self.data_fetcher = MarketDataFetcher()
        self.sentiment = SentimentAnalyzer() if self.config.enable_sentiment else None
        self.alt_data = AlternativeDataFetcher() if self.config.enable_alternative_data else None
        self.signal_generator = SignalGenerator()
        self.bridge = MT5Bridge()

        # Professional trading modules
        self.multi_tf = (
            MultiTimeframeDataFetcher()
            if self.config.enable_multi_timeframe else None
        )
        self.news_filter = (
            NewsCalendarFilter()
            if self.config.enable_news_filter else None
        )
        self.multi_pair_config = (
            MultiPairConfig()
            if self.config.enable_multi_pair else None
        )

        # Load models if checkpoints exist
        self._load_models()

        self.logger.info("Python ML Bridge initialized")
        self.logger.info(f"  Interval: {self.config.interval_seconds}s")
        self.logger.info(f"  Paper trading: {self.config.paper_trading}")
        self.logger.info(f"  Sentiment: {self.config.enable_sentiment}")
        self.logger.info(f"  Alt data: {self.config.enable_alternative_data}")
        self.logger.info(f"  Multi-timeframe: {self.config.enable_multi_timeframe}")
        self.logger.info(f"  News filter: {self.config.enable_news_filter}")
        self.logger.info(f"  Multi-pair: {self.config.enable_multi_pair}")

    def _load_models(self):
        """Load model checkpoints if available."""
        if os.path.exists(MODEL_DIR):
            try:
                self.signal_generator.ensemble.load_models(MODEL_DIR)
                self.logger.info(f"Loaded model checkpoints from {MODEL_DIR}")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoints: {e}")
        else:
            self.logger.info("No model checkpoints found, using untrained models")

    def run_cycle(self) -> dict:
        """
        Run a single prediction cycle.

        Professional trading logic:
        1. Check news calendar - skip if high-impact event window
        2. Fetch multi-timeframe data for trend confirmation
        3. Get cross-pair correlation features for context
        4. Run ensemble prediction with enriched features
        5. Gate signal through risk filters

        Returns:
            Dict with cycle results including signal generated
        """
        cycle_start = time.time()
        result = {"timestamp": datetime.now().isoformat(), "signal": None, "error": None}

        try:
            # 0. NEWS FILTER - Professional traders never trade through major events
            if self.news_filter:
                if self.news_filter.is_high_impact_window():
                    upcoming = self.news_filter.get_upcoming_events(hours_ahead=2)
                    event_info = upcoming[0]["title"] if upcoming else "Unknown"
                    self.logger.info(
                        f"NEWS FILTER: Skipping cycle - high impact event window "
                        f"({event_info}). No trades during NFP/FOMC/CPI."
                    )
                    result["error"] = f"News filter active: {event_info}"
                    result["news_filtered"] = True
                    return result

            # 1. Fetch market data
            df = self.data_fetcher.fetch_ohlcv(interval="1h", period="3mo")
            if df.empty:
                result["error"] = "No market data available"
                return result

            # 2. Compute features
            features_df = self.data_fetcher.compute_features(df)
            if features_df.empty:
                result["error"] = "Insufficient data for features"
                return result

            # 3. Get latest sequence for prediction
            seq_length = 64
            feature_input = self.data_fetcher.get_latest_features(seq_length)
            if feature_input is None:
                result["error"] = "Could not prepare model input"
                return result

            # 4. Get ATR and current price
            atr = self.data_fetcher.get_current_atr()
            current_price = float(df["Close"].iloc[-1])

            # 5. Get ADX for regime detection
            adx_series = None
            if "adx" in features_df.columns:
                adx_series = features_df["adx"]

            # 6. Get VIX level (optional)
            vix_level = None
            if self.alt_data:
                try:
                    alt_features = self.alt_data.get_alternative_features()
                    vix_level = alt_features.get("vix_level")
                except Exception as e:
                    self.logger.debug(f"Alt data fetch error: {e}")

            # 7. Multi-timeframe trend confirmation (professional HTF analysis)
            htf_bias = None
            if self.multi_tf:
                try:
                    htf_bias = self.multi_tf.get_htf_trend_bias()
                    if htf_bias:
                        self.logger.debug(
                            f"HTF Bias: H1={htf_bias.get('1h', 0):.2f} "
                            f"H4={htf_bias.get('4h', 0):.2f}"
                        )
                except Exception as e:
                    self.logger.debug(f"Multi-TF error: {e}")

            # 8. Cross-pair correlation analysis
            cross_pair_info = None
            if self.multi_pair_config:
                try:
                    pair_data = self.data_fetcher.fetch_multi_pair(
                        self.multi_pair_config, period="1mo", interval="1h"
                    )
                    if pair_data:
                        cross_features = self.data_fetcher.compute_cross_pair_features(
                            pair_data, self.multi_pair_config
                        )
                        if not cross_features.empty:
                            # Extract latest USD strength for logging
                            if "xpair_usd_strength" in cross_features.columns:
                                usd_str = cross_features["xpair_usd_strength"].iloc[-1]
                                cross_pair_info = {"usd_strength": float(usd_str)}
                                self.logger.debug(
                                    f"USD Strength: {usd_str:.4f} "
                                    f"({'strong' if usd_str > 0 else 'weak'} - "
                                    f"{'bearish' if usd_str > 0 else 'bullish'} for gold)"
                                )
                except Exception as e:
                    self.logger.debug(f"Multi-pair error: {e}")

            # 9. Generate signal
            signal = self.signal_generator.generate_signal(
                features=feature_input,
                prices=df,
                atr=atr,
                current_price=current_price,
                adx_series=adx_series,
                vix_level=vix_level
            )

            # 10. Write signal to bridge
            if signal.action != "HOLD":
                self.bridge.write_signal(signal)
                self.logger.info(
                    f"Signal: {signal.action} | Conf: {signal.confidence:.2f} | "
                    f"Lot: {signal.lot_size} | Regime: {signal.regime}"
                )
                if htf_bias:
                    self.logger.info(
                        f"  HTF Confirmation: H1={htf_bias.get('1h', 0):.2f} "
                        f"H4={htf_bias.get('4h', 0):.2f}"
                    )
                if cross_pair_info:
                    self.logger.info(
                        f"  USD Strength: {cross_pair_info.get('usd_strength', 0):.4f}"
                    )
            else:
                self.logger.debug("No signal (HOLD)")

            # 11. Write heartbeat
            self.bridge.write_heartbeat()

            # 12. Check confirmations from MT5
            confirmations = self.bridge.read_confirmations()
            if confirmations:
                for conf in confirmations:
                    self.logger.info(f"MT5 Confirmation: {conf}")
                self.bridge.clear_confirmations()

            result["signal"] = signal.to_dict()
            result["htf_bias"] = htf_bias
            result["cross_pair_info"] = cross_pair_info

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Cycle error: {e}", exc_info=True)

        cycle_time = time.time() - cycle_start
        self.logger.debug(f"Cycle completed in {cycle_time:.2f}s")
        result["cycle_time"] = cycle_time
        return result

    def run(self):
        """Run the main bridge loop (blocking)."""
        self._running = True
        self.logger.info("Starting main loop...")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        cycle_count = 0
        while self._running:
            cycle_count += 1
            self.logger.info(f"--- Cycle {cycle_count} ---")

            result = self.run_cycle()

            if result.get("error"):
                self.logger.warning(f"Cycle {cycle_count} error: {result['error']}")

            # Sleep until next cycle
            if self._running:
                time.sleep(self.config.interval_seconds)

        self.logger.info("Bridge stopped")

    def stop(self):
        """Stop the main loop."""
        self._running = False
        self.logger.info("Stop requested")

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    """Main entry point."""
    print("=" * 60)
    print("  Python ML Bridge for MetaTrader 5")
    print("  Deep Learning Trade Signal Generator")
    print("=" * 60)

    config = MainConfig()

    # Parse command line args
    if "--paper" in sys.argv:
        config.paper_trading = True
    if "--live" in sys.argv:
        config.paper_trading = False
    if "--interval" in sys.argv:
        idx = sys.argv.index("--interval")
        if idx + 1 < len(sys.argv):
            config.interval_seconds = int(sys.argv[idx + 1])

    bridge = PythonMLBridge(config)
    bridge.run()


if __name__ == "__main__":
    main()
