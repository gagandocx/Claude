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
import threading
from datetime import datetime
from typing import Optional

VERSION = "5.3.4"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    MainConfig, DataConfig, SignalConfig, RiskConfig,
    TransformerConfig, LSTMConfig, EnsembleConfig,
    MultiTimeframeConfig, NewsFilterConfig, MultiPairConfig,
    SmartExitConfig, RLConfig, RetrainConfig, DashboardConfig,
    AutoOptimizerConfig,
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
from strategies.smart_exits import SmartExitManager, ExitDecision
from strategies.auto_optimizer import AutoOptimizer
from signals.bridge import MT5Bridge
from training.auto_retrain import AutoRetrainer
from dashboard.performance_tracker import PerformanceTracker, TradeRecord
from dashboard.dashboard_renderer import DashboardRenderer


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
        # Pre-fetch news calendar in a background thread (non-blocking)
        if self.news_filter:
            self._news_refresh_thread: Optional[threading.Thread] = None
            self._start_news_refresh()
        self.multi_pair_config = (
            MultiPairConfig()
            if self.config.enable_multi_pair else None
        )

        # AI-driven exit management (RL agent + dynamic trailing stops)
        # Share a single RL agent between SignalGenerator and SmartExitManager
        # to accumulate experience in one replay buffer (avoids split learning)
        self.smart_exits = (
            SmartExitManager(SmartExitConfig(), rl_agent=self.signal_generator._rl_agent)
            if self.config.enable_smart_exits else None
        )

        # Weekend auto-retraining scheduler
        self.auto_retrainer = (
            AutoRetrainer(RetrainConfig())
            if self.config.enable_auto_retrain else None
        )

        # Self-tuning parameter optimizer
        self.auto_optimizer = (
            AutoOptimizer(AutoOptimizerConfig())
            if self.config.enable_auto_optimizer else None
        )

        # Live performance dashboard (prop desk style analytics)
        self.dashboard_config = DashboardConfig()
        self.performance_tracker = (
            PerformanceTracker(
                min_trades_for_stats=self.dashboard_config.min_trades_for_stats
            )
            if self.config.enable_dashboard else None
        )
        self.dashboard_renderer = (
            DashboardRenderer(
                self.performance_tracker,
                use_colors=self.dashboard_config.use_colors
            )
            if self.performance_tracker else None
        )
        self._dashboard_last_render = 0.0
        self._trades_since_render = 0

        # Rolling ATR baseline for post-news volatility comparison.
        # Stores recent ATR values from each signal cycle to compute a
        # "normal" baseline. The short ATR (5-period) is compared against
        # this rolling average to detect post-news volatility spikes.
        self._atr_history: list = []
        self._atr_history_max_size: int = 30  # ~30 cycles of history
        self._last_news_vol_check: float = 0.0  # timestamp of last vol check

        # Load models if checkpoints exist
        self._load_models()

        # Connect performance tracker to signal generator for trade feedback
        if self.performance_tracker:
            self.signal_generator.set_performance_tracker(self.performance_tracker)

        # Connect auto-optimizer to signal generator for parameter self-tuning
        if self.auto_optimizer:
            self.signal_generator.set_auto_optimizer(self.auto_optimizer)

        self.logger.info("Python ML Bridge initialized")
        self.logger.info(f"  Interval: {self.config.interval_seconds}s")
        self.logger.info(f"  Paper trading: {self.config.paper_trading}")
        self.logger.info(f"  Sentiment: {self.config.enable_sentiment}")
        self.logger.info(f"  Alt data: {self.config.enable_alternative_data}")
        self.logger.info(f"  Multi-timeframe: {self.config.enable_multi_timeframe}")
        self.logger.info(f"  News filter: {self.config.enable_news_filter}")
        self.logger.info(f"  Multi-pair: {self.config.enable_multi_pair}")
        self.logger.info(f"  Smart exits (RL): {self.config.enable_smart_exits}")
        self.logger.info(f"  Auto-retrain: {self.config.enable_auto_retrain}")
        self.logger.info(f"  Dashboard: {self.config.enable_dashboard}")
        self.logger.info(f"  Auto-optimizer: {self.config.enable_auto_optimizer}")

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

    def _start_news_refresh(self):
        """Start a background thread to refresh the news calendar.

        This avoids blocking the main signal loop with a 10s HTTP call.
        The calendar data is refreshed periodically in the background,
        and is_high_impact_window() reads from the already-loaded cache.
        """
        def _refresh():
            try:
                self.news_filter.fetch_calendar()
            except Exception as e:
                self.logger.debug(f"Background news refresh error: {e}")

        self._news_refresh_thread = threading.Thread(
            target=_refresh, daemon=True, name="NewsRefresh"
        )
        self._news_refresh_thread.start()

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
            # 0a. AUTO-RETRAIN CHECK - Professional firms retrain on weekends
            if self.auto_retrainer:
                if self.auto_retrainer.should_retrain():
                    self.logger.info(
                        "[AutoRetrain] Weekend retraining triggered. "
                        "Running training pipeline..."
                    )
                    retrain_result = self.auto_retrainer.run_retrain()
                    self.logger.info(
                        f"[AutoRetrain] Result: {retrain_result.get('reason', '')}"
                    )
                    if retrain_result.get("deployed"):
                        self._load_models()  # Reload newly trained models

            # 0. NEWS FILTER - Professional traders never trade through major events
            # Uses cached calendar data (refreshed in background thread) to avoid
            # blocking the signal loop with network calls.
            if self.news_filter:
                # Trigger background refresh if needed (non-blocking)
                if self.news_filter._needs_refresh():
                    if (self._news_refresh_thread is None
                            or not self._news_refresh_thread.is_alive()):
                        self._start_news_refresh()

                # Get ATR values for post-news volatility check.
                # current_atr: short-window (5-period) reflecting immediate volatility
                # normal_atr: rolling average of the 14-period ATR from recent cycles
                # This ensures a meaningful comparison: spiky short-term ATR vs calm baseline.
                current_atr = None
                normal_atr = None

                # Throttle ATR fetches using post_news_check_interval config
                now_ts = time.time()
                check_interval = self.news_filter.config.post_news_check_interval
                if now_ts - self._last_news_vol_check >= check_interval:
                    try:
                        current_atr = self.data_fetcher.get_short_atr(window=5)
                        baseline_atr = self.data_fetcher.get_current_atr()
                        self._last_news_vol_check = now_ts

                        # Update rolling ATR history for baseline calculation
                        if baseline_atr and baseline_atr > 0:
                            self._atr_history.append(baseline_atr)
                            if len(self._atr_history) > self._atr_history_max_size:
                                self._atr_history = self._atr_history[-self._atr_history_max_size:]

                        # Normal ATR is the rolling average of recent 14-period ATR values
                        if self._atr_history:
                            normal_atr = sum(self._atr_history) / len(self._atr_history)
                    except Exception:
                        # ATR fetch failed - leave both as None (fail-closed in should_block_trading)
                        current_atr = None
                        normal_atr = None

                news_status = self.news_filter.should_block_trading(
                    current_atr=current_atr,
                    normal_atr=normal_atr,
                )

                if news_status["blocked"]:
                    upcoming = self.news_filter.get_upcoming_events_cached(hours_ahead=2)
                    event_info = upcoming[0]["title"] if upcoming else "Unknown"
                    state = news_status["state"]

                    if state == "pre_news":
                        status_msg = f"Pre-news: paused - {event_info}"
                    elif state == "post_news_min_wait":
                        status_msg = f"Post-news: waiting minimum period - {event_info}"
                    elif state == "post_news_high_vol":
                        status_msg = f"Post-news: checking volatility... - {event_info}"
                    else:
                        status_msg = f"News filter active: {event_info}"

                    self.logger.info(
                        f"NEWS FILTER: Skipping cycle - {news_status['reason']}"
                    )
                    result["error"] = f"News filter: {news_status['reason']}"
                    result["news_filtered"] = True
                    self.bridge.write_status("NEWS", status_msg)
                    return result

                # Log if we just resumed after news
                if news_status["state"] == "post_news_safe":
                    self.logger.info(
                        f"NEWS FILTER: {news_status['reason']} - resuming trades"
                    )
                    self.bridge.write_status(
                        "OK", f"Post-news: low vol, resuming"
                    )

            # 1. Fetch market data
            df = self.data_fetcher.fetch_ohlcv(interval="1h", period="3mo")
            if df.empty:
                result["error"] = "No market data available"
                return result

            # 1b. Fetch M1 data for momentum direction and M1 ATR sizing
            # M1 gives 7-minute momentum (vs 7-hour from H1) - correct for scalping
            df_m1 = None
            try:
                df_m1 = self.data_fetcher.fetch_ohlcv(interval="1m", period="5d")
                if df_m1.empty:
                    self.logger.warning("[M1 Data] M1 fetch returned empty - falling back to H1 for momentum")
                    df_m1 = None
            except Exception as e:
                self.logger.warning(f"[M1 Data] M1 fetch error: {e} - falling back to H1 for momentum")
                df_m1 = None

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
            # Use M1 ATR for SL/TP sizing (~$2-3 for proper scalping stops)
            # Fall back to H1 ATR if M1 unavailable (with tighter $5 cap applied here)
            if df_m1 is not None and not df_m1.empty:
                # Compute ATR directly from the df_m1 we already fetched (no second network call)
                atr = self.data_fetcher.compute_atr_from_df(df_m1, window=14)
                if atr <= 0:
                    atr = self.data_fetcher.get_current_atr()
                    # H1 fallback: apply tighter cap to prevent outsized scalping stops
                    atr = min(atr, 5.0)
                # Use M1 close for most current price
                current_price = float(df_m1["Close"].iloc[-1])
            else:
                atr = self.data_fetcher.get_current_atr()
                # H1 fallback: apply tighter cap to prevent outsized scalping stops
                atr = min(atr, 5.0)
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
                            f"HTF Bias: M5={htf_bias.get('5m', 0):.2f} "
                            f"M15={htf_bias.get('15m', 0):.2f}"
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
                vix_level=vix_level,
                htf_bias=htf_bias,
                cross_pair_info=cross_pair_info,
                prices_m1=df_m1,
            )

            # 10. Write signal to bridge
            if signal.action != "HOLD":
                write_ok = self.bridge.write_signal(signal)
                if not write_ok:
                    self.logger.warning(
                        "Signal write failed, retrying once..."
                    )
                    time.sleep(0.05)
                    write_ok = self.bridge.write_signal(signal)
                    if not write_ok:
                        self.logger.warning(
                            "Signal write retry also failed. Signal may be stale on disk."
                        )
                self.logger.info(
                    f"Signal: {signal.action} | Conf: {signal.confidence:.2f} | "
                    f"Lot: {signal.lot_size} | Regime: {signal.regime}"
                )
                if htf_bias:
                    self.logger.info(
                        f"  HTF Confirmation: M5={htf_bias.get('5m', 0):.2f} "
                        f"M15={htf_bias.get('15m', 0):.2f}"
                    )
                if cross_pair_info:
                    self.logger.info(
                        f"  USD Strength: {cross_pair_info.get('usd_strength', 0):.4f}"
                    )
            else:
                self.logger.debug("No signal (HOLD)")

            # 10b. SMART EXITS - AI-driven position management for open positions
            if self.smart_exits:
                try:
                    # Update unrealized PnL for all open positions each cycle (Fix #4)
                    open_positions = self.signal_generator._open_positions
                    for ticket, pos_info in list(open_positions.items()):
                        direction = pos_info.get("direction", 1)
                        entry_price = pos_info.get("entry_price", current_price)
                        if direction == 1:  # LONG
                            unrealized = current_price - entry_price
                        else:  # SHORT
                            unrealized = entry_price - current_price
                        pos_info["current_price"] = current_price
                        pos_info["unrealized_pnl"] = unrealized
                        pos_info["atr"] = atr
                        # Track max favorable and adverse excursion
                        pos_info["max_favorable"] = max(
                            pos_info.get("max_favorable", 0.0), unrealized
                        )
                        pos_info["max_adverse"] = min(
                            pos_info.get("max_adverse", 0.0), unrealized
                        )

                    # Process exit signals from RL agent
                    exit_signals = []
                    for ticket, pos_info in list(open_positions.items()):
                        from models.rl_agent import PositionState
                        # Decay confidence over time
                        pos_info["hold_bars"] = pos_info.get("hold_bars", 0) + 1
                        pos_info["confidence"] = self.smart_exits.decay_confidence(
                            pos_info.get("initial_confidence", 0.5),
                            pos_info["hold_bars"]
                        )
                        position = PositionState(
                            direction=pos_info.get("direction", 1),
                            unrealized_pnl=pos_info.get("unrealized_pnl", 0.0),
                            unrealized_pnl_atr=pos_info.get("unrealized_pnl", 0.0) / max(pos_info.get("atr", 2.0), 0.01),
                            hold_bars=pos_info["hold_bars"],
                            entry_price=pos_info.get("entry_price", 0.0),
                            current_price=current_price,
                            atr=atr,
                            confidence=pos_info["confidence"],
                            initial_confidence=pos_info.get("initial_confidence", 0.5),
                            sl_distance_atr=pos_info.get("sl_distance_atr", 1.5),
                            tp_distance_atr=pos_info.get("tp_distance_atr", 2.5),
                            max_favorable=pos_info.get("max_favorable", 0.0),
                            max_adverse=pos_info.get("max_adverse", 0.0),
                            partial_closed_pct=pos_info.get("partial_closed_pct", 0.0),
                            regime_changed=pos_info.get("regime_changed", False),
                            ticket=ticket
                        )
                        decision = self.smart_exits.evaluate_exit(
                            position, market_features=feature_input
                        )
                        if decision.action != "HOLD":
                            exit_signals.append({
                                "ticket": ticket,
                                "action": decision.action,
                                "lot_pct": decision.lot_pct_to_close,
                                "new_sl": decision.new_sl_price,
                                "reason": decision.reason,
                            })
                            self.logger.info(
                                f"  Exit signal: {decision.action} ticket={ticket} "
                                f"reason={decision.reason}"
                            )
                    # Write exit signals to bridge
                    if exit_signals:
                        self.bridge.write_exit_signals(exit_signals)
                        # Fix #3: Update partial_closed_pct after writing
                        # CLOSE_PARTIAL signals so the smart exit manager knows
                        # how much has already been closed. Without this, repeated
                        # partial-close signals are emitted every bar.
                        for exit_sig in exit_signals:
                            if exit_sig["action"] == "CLOSE_PARTIAL":
                                ticket = exit_sig["ticket"]
                                if ticket in open_positions:
                                    open_positions[ticket]["partial_closed_pct"] = min(
                                        open_positions[ticket].get("partial_closed_pct", 0.0)
                                        + exit_sig["lot_pct"],
                                        1.0,
                                    )
                except Exception as e:
                    self.logger.debug(f"Smart exit processing error: {e}")

            # 11. Write heartbeat
            self.bridge.write_heartbeat()

            # Write detailed status for MT5 dashboard display
            if result.get("signal") and hasattr(result["signal"], "action") and result["signal"].action != "HOLD":
                sig = result["signal"]
                status_msg = f"SIGNAL: {sig.action} | Conf:{sig.confidence:.2f} | {sig.regime}"
                self.bridge.write_status("OK", status_msg)
            elif result.get("news_filtered"):
                pass  # Already written above
            elif result.get("error"):
                self.bridge.write_status("WARNING", result["error"][:150])
            else:
                # Show what the system is doing (momentum, session, range mode)
                self.bridge.write_status("OK", "Scanning - no signal this cycle")

            # 12. Check confirmations from MT5
            confirmations = self.bridge.read_confirmations()
            if confirmations:
                for conf in confirmations:
                    self.logger.info(f"MT5 Confirmation: {conf}")

                    # Fix #1: Register new positions when MT5 confirms an entry fill
                    if conf.get("type") == "open" or conf.get("type") == "fill":
                        trade_id = str(conf.get("ticket", ""))
                        direction = 1 if conf.get("direction", "BUY") == "BUY" else -1
                        entry_price = float(conf.get("entry_price", current_price))
                        confidence_val = float(conf.get("confidence", 0.5))
                        sl_pips_val = float(conf.get("sl_pips", 0.0))
                        tp_pips_val = float(conf.get("tp_pips", 0.0))
                        self.signal_generator.register_open_position(
                            trade_id=trade_id,
                            direction=direction,
                            entry_price=entry_price,
                            confidence=confidence_val,
                            atr=atr,
                            sl_pips=sl_pips_val,
                            tp_pips=tp_pips_val,
                        )
                        self.logger.info(
                            f"  Registered position {trade_id} for RL tracking "
                            f"(dir={direction}, entry={entry_price:.2f})"
                        )

                    # Handle trade closures: clean up positions, feed RL, record performance
                    if conf.get("type") == "close":
                        trade_id = str(conf.get("ticket", ""))
                        pnl = float(conf.get("pnl", 0.0))

                        # Fix #1: Call update_from_execution() to remove position
                        # from _open_positions and feed the RL agent its terminal
                        # reward. Without this, positions accumulate indefinitely
                        # and the DQN never learns trade outcomes.
                        self.signal_generator.update_from_execution(
                            trade_id=trade_id,
                            pnl=pnl,
                            predicted_action=0,
                            actual_outcome=0,
                        )

                        # Fix #6: Only feed trade closures to PerformanceTracker from
                        # MT5 confirmations (not from update_from_execution) to prevent
                        # double-counting. The signal_generator.update_from_execution()
                        # handles RL learning only, not performance tracking.
                        if self.performance_tracker:
                            trade_record = TradeRecord(
                                trade_id=trade_id,
                                entry_time=conf.get("open_time", datetime.now().isoformat()),
                                exit_time=conf.get("close_time", datetime.now().isoformat()),
                                direction=conf.get("direction", "BUY"),
                                pnl=pnl,
                                model=conf.get("model", "ensemble"),
                                regime=conf.get("regime", "ranging"),
                                entry_price=float(conf.get("entry_price", 0.0)),
                                exit_price=float(conf.get("exit_price", 0.0)),
                                lot_size=float(conf.get("lot_size", 0.0)),
                                confidence=float(conf.get("confidence", 0.0)),
                            )
                            self.performance_tracker.record_trade(trade_record)
                            self._trades_since_render += 1

                        # Feed trade to auto-optimizer for self-tuning
                        if self.auto_optimizer:
                            trade_context = {
                                "session": conf.get("session", "unknown"),
                                "confidence": float(conf.get("confidence", 0.0)),
                                "momentum_lookback": conf.get("momentum_lookback", 5),
                                "sl_distance": float(conf.get("sl_distance", 3.0)),
                                "result_pnl": pnl,
                                "direction": conf.get("direction", "BUY"),
                                "rsi_at_entry": float(conf.get("rsi_at_entry", 50.0)),
                                "trail_tier": conf.get("trail_tier", "medium"),
                                "cooldown_used": float(conf.get("cooldown_used", 2.0)),
                                "max_positions_at_entry": int(conf.get("max_positions_at_entry", 5)),
                                "entry_time": conf.get("open_time", datetime.now().isoformat()),
                                "exit_time": conf.get("close_time", datetime.now().isoformat()),
                            }
                            self.auto_optimizer.record_trade(trade_context)
                self.bridge.clear_confirmations()

            # 13. Periodic dashboard rendering
            if self.dashboard_renderer:
                now = time.time()
                should_render = (
                    now - self._dashboard_last_render
                    >= self.dashboard_config.console_refresh_seconds
                    and self.performance_tracker.total_trades > 0
                )
                if should_render:
                    if self.dashboard_config.enable_console:
                        dashboard_output = self.dashboard_renderer.render_console()
                        self.logger.info("\n" + dashboard_output)
                    if self.dashboard_config.enable_log_output:
                        self.dashboard_renderer.render_log()
                    self._dashboard_last_render = now

            result["signal"] = signal.to_dict()
            result["htf_bias"] = htf_bias
            result["cross_pair_info"] = cross_pair_info

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Cycle error: {e}", exc_info=True)
            # Write warning status for MT5 dashboard display
            self.bridge.write_status("WARNING", f"Cycle error: {str(e)[:150]}")

        cycle_time = time.time() - cycle_start
        self.logger.debug(f"Cycle completed in {cycle_time:.2f}s")
        result["cycle_time"] = cycle_time
        return result

    def run(self):
        """Run the main bridge loop (blocking)."""
        self._running = True
        self.logger.info(f"Python ML Bridge v{VERSION} starting...")
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
        """Stop the main loop and generate final reports."""
        self._running = False
        # Generate final HTML report on shutdown
        if (self.dashboard_renderer and self.dashboard_config.enable_html_report
                and self.performance_tracker.total_trades > 0):
            try:
                html_path = self.dashboard_config.html_output_path
                self.dashboard_renderer.render_html(html_path)
                self.logger.info(f"Final performance report saved to: {html_path}")
            except Exception as e:
                self.logger.warning(f"Could not generate final report: {e}")
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
    print(f"  Python ML Bridge for MetaTrader 5 v{VERSION}")
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
