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
    MODEL_DIR, LOG_DIR
)
from data.market_data import MarketDataFetcher
from data.sentiment import SentimentAnalyzer
from data.alternative_data import AlternativeDataFetcher
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

        # Load models if checkpoints exist
        self._load_models()

        self.logger.info("Python ML Bridge initialized")
        self.logger.info(f"  Interval: {self.config.interval_seconds}s")
        self.logger.info(f"  Paper trading: {self.config.paper_trading}")
        self.logger.info(f"  Sentiment: {self.config.enable_sentiment}")
        self.logger.info(f"  Alt data: {self.config.enable_alternative_data}")

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

        Returns:
            Dict with cycle results including signal generated
        """
        cycle_start = time.time()
        result = {"timestamp": datetime.now().isoformat(), "signal": None, "error": None}

        try:
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

            # 7. Generate signal
            signal = self.signal_generator.generate_signal(
                features=feature_input,
                prices=df,
                atr=atr,
                current_price=current_price,
                adx_series=adx_series,
                vix_level=vix_level
            )

            # 8. Write signal to bridge
            if signal.action != "HOLD":
                self.bridge.write_signal(signal)
                self.logger.info(
                    f"Signal: {signal.action} | Conf: {signal.confidence:.2f} | "
                    f"Lot: {signal.lot_size} | Regime: {signal.regime}"
                )
            else:
                self.logger.debug("No signal (HOLD)")

            # 9. Write heartbeat
            self.bridge.write_heartbeat()

            # 10. Check confirmations from MT5
            confirmations = self.bridge.read_confirmations()
            if confirmations:
                for conf in confirmations:
                    self.logger.info(f"MT5 Confirmation: {conf}")
                self.bridge.clear_confirmations()

            result["signal"] = signal.to_dict()

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
