"""
=============================================================
  NeuroX - Main Entry Point
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

VERSION = "7.5"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    MainConfig, DataConfig, SignalConfig, RiskConfig,
    TransformerConfig, LSTMConfig, EnsembleConfig,
    MultiTimeframeConfig, NewsFilterConfig, MultiPairConfig,
    SmartExitConfig, RLConfig, RetrainConfig, DashboardConfig,
    AutoOptimizerConfig, BrainConfig,
    PlattCalibrationConfig, EntryTimingConfig, SharpeWeightConfig,
    TickDataConfig, MicrostructureConfig,
    RegimeRoutingConfig, WalkForwardConfig, AdversarialFilterConfig,
    SpreadGateConfig, CorrelationRegimeConfig, AdaptiveThresholdConfig,
    DisagreementConfig, KellyConfig, MonteCarloConfig,
    DataValidatorConfig, PipelineConfig,
    AccountSyncConfig, SlippageTrackerConfig,
    FeatureMonitorConfig, ABTestConfig,
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
from strategies.trading_brain import TradingBrain
from strategies.confidence_calibrator import ConfidenceCalibrator
from strategies.entry_timing import EntryTimingManager
from signals.bridge import MT5Bridge
from training.auto_retrain import AutoRetrainer
from dashboard.performance_tracker import PerformanceTracker, TradeRecord
from dashboard.dashboard_renderer import DashboardRenderer
from data.tick_data import TickDataProcessor
from data.microstructure import MicrostructureAnalyzer
from data.spread_monitor import SpreadMonitor
from strategies.regime_router import RegimeModelRouter
from strategies.walk_forward import WalkForwardRetrainer
from strategies.adversarial_filter import AdversarialFilter
from strategies.correlation_regime import CorrelationRegimeDetector
from strategies.adaptive_threshold import AdaptiveConfidenceThreshold
from strategies.disagreement_signal import DisagreementSignal
from strategies.kelly_sizing import KellySizer
from strategies.monte_carlo import MonteCarloRiskSimulator
from data.data_validator import DataValidator
from data.pipeline import PipelineManager
from strategies.slippage_tracker import SlippageTracker
from strategies.feature_monitor import FeatureImportanceMonitor
from strategies.ab_testing import ABTestFramework


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

        # Trading Brain — fully autonomous professional trading intelligence
        self.brain = (
            TradingBrain(BrainConfig())
            if self.config.enable_brain else None
        )
        self._last_trade_date: str = ""   # for daily brain reset

        # v7.4 features: Platt scaling confidence calibration
        self.confidence_calibrator = (
            ConfidenceCalibrator(PlattCalibrationConfig())
            if self.config.enable_platt_calibration else None
        )

        # v7.4 features: Smart entry timing (micro-pullback)
        self.entry_timing = (
            EntryTimingManager(EntryTimingConfig())
            if self.config.enable_entry_timing else None
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

        # ── V3 Institutional Tier 1 Components ────────────────────────────────
        # Tick data processor (order flow features)
        self.tick_data_processor = (
            TickDataProcessor(TickDataConfig())
            if self.config.enable_tick_data else None
        )

        # Microstructure analyzer (tick-level features)
        self.microstructure_analyzer = (
            MicrostructureAnalyzer(MicrostructureConfig())
            if self.config.enable_microstructure else None
        )

        # Regime-specific model router
        self.regime_router = (
            RegimeModelRouter(RegimeRoutingConfig())
            if self.config.enable_regime_routing else None
        )

        # Walk-forward retrainer (weekly automated retraining)
        self.walk_forward_retrainer = (
            WalkForwardRetrainer(WalkForwardConfig())
            if self.config.enable_walk_forward else None
        )

        # Adversarial signal filter
        self.adversarial_filter = (
            AdversarialFilter(AdversarialFilterConfig())
            if self.config.enable_adversarial_filter else None
        )

        # ── V3 Institutional Tier 2 Components ────────────────────────────────
        # Spread monitor (EA-reported real-time spread gating)
        self.spread_monitor = (
            SpreadMonitor(SpreadGateConfig())
            if self.config.enable_spread_gate else None
        )

        # Correlation regime detector (DXY/bonds/equities state)
        self.correlation_regime = (
            CorrelationRegimeDetector(CorrelationRegimeConfig())
            if self.config.enable_correlation_regime else None
        )

        # Adaptive confidence threshold (self-tuning entry gate)
        self.adaptive_threshold = (
            AdaptiveConfidenceThreshold(AdaptiveThresholdConfig())
            if self.config.enable_adaptive_threshold else None
        )

        # ── V3 Institutional Tier 3 Components ────────────────────────────────
        # Multi-model disagreement signal (volatility prediction)
        self.disagreement_signal = (
            DisagreementSignal(DisagreementConfig())
            if self.config.enable_disagreement_signal else None
        )

        # Kelly criterion position sizing
        self.kelly_sizer = (
            KellySizer(KellyConfig())
            if self.config.enable_kelly_sizing else None
        )

        # Monte Carlo risk simulation
        self.monte_carlo_risk = (
            MonteCarloRiskSimulator(MonteCarloConfig(), BrainConfig())
            if self.config.enable_monte_carlo_risk else None
        )

        # ── V7.5 Data Quality & Pipeline Components ───────────────────────────
        # Live data validation (NaN, gaps, zero-volume, price sanity)
        self.data_validator = (
            DataValidator(DataValidatorConfig())
            if self.config.enable_data_validation else None
        )

        # Pipeline threading (overlap data fetch with model compute)
        # TODO: PipelineManager is ready but not yet wired into run_cycle().
        # Integrating it requires refactoring the fetch/compute flow to use
        # pipeline.submit_data_fetch() and pipeline.get_results() for overlapping
        # I/O with model inference. Currently data fetches remain serial.
        self.pipeline_manager = (
            PipelineManager(PipelineConfig())
            if self.config.enable_pipeline else None
        )

        # ── V7.5 Execution Quality & Account Sync ────────────────────────────
        # Slippage / execution quality tracker
        self.slippage_tracker = (
            SlippageTracker(SlippageTrackerConfig())
            if self.config.enable_slippage_tracker else None
        )

        # Account balance sync configuration
        self._account_sync_config = (
            AccountSyncConfig()
            if self.config.enable_account_sync else None
        )

        # ── V7.5 Feature Monitoring & Online Learning ─────────────────────
        # Feature importance monitor (detects degraded features)
        self.feature_monitor = (
            FeatureImportanceMonitor(FeatureMonitorConfig())
            if self.config.enable_feature_monitor else None
        )
        self._feature_monitor_log_interval: int = 100  # Log status every N cycles
        self._feature_monitor_cycle_count: int = 0

        # ── V7.5 A/B Testing & Equity Curve Trading ──────────────────────
        # A/B testing framework (rigorous parameter comparison)
        self.ab_testing = (
            ABTestFramework(ABTestConfig())
            if self.config.enable_ab_testing else None
        )

        # Equity curve trading is handled within the TradingBrain
        # (EquityCurveTrader class) - the flag controls whether it's active
        self._equity_curve_trading_enabled = self.config.enable_equity_curve_trading

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

        self.logger.info("NeuroX initialized")
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
        self.logger.info(f"  Trading Brain:  {self.config.enable_brain}")
        self.logger.info(f"  Platt calibration: {self.config.enable_platt_calibration}")
        self.logger.info(f"  Entry timing: {self.config.enable_entry_timing}")
        self.logger.info(f"  Sharpe weights: {self.config.enable_sharpe_weights}")
        self.logger.info(f"  Tick data: {self.config.enable_tick_data}")
        self.logger.info(f"  Regime routing: {self.config.enable_regime_routing}")
        self.logger.info(f"  Walk-forward: {self.config.enable_walk_forward}")
        self.logger.info(f"  Adversarial filter: {self.config.enable_adversarial_filter}")
        self.logger.info(f"  Spread gate: {self.config.enable_spread_gate}")
        self.logger.info(f"  Microstructure: {self.config.enable_microstructure}")
        self.logger.info(f"  Correlation regime: {self.config.enable_correlation_regime}")
        self.logger.info(f"  Adaptive threshold: {self.config.enable_adaptive_threshold}")
        self.logger.info(f"  Disagreement signal: {self.config.enable_disagreement_signal}")
        self.logger.info(f"  Kelly sizing: {self.config.enable_kelly_sizing}")
        self.logger.info(f"  Monte Carlo risk: {self.config.enable_monte_carlo_risk}")
        self.logger.info(f"  Data validation: {self.config.enable_data_validation}")
        self.logger.info(f"  Pipeline threading: {self.config.enable_pipeline}")
        self.logger.info(f"  Account sync: {self.config.enable_account_sync}")
        self.logger.info(f"  Slippage tracker: {self.config.enable_slippage_tracker}")
        self.logger.info(f"  Feature monitor: {self.config.enable_feature_monitor}")
        self.logger.info(f"  Online learning: {self.config.enable_online_learning}")
        self.logger.info(f"  A/B testing: {self.config.enable_ab_testing}")
        self.logger.info(f"  Equity curve trading: {self.config.enable_equity_curve_trading}")

        # Start background confirmation poller for instant position sync
        self._running = True  # Set before starting thread
        self._confirmation_thread = threading.Thread(
            target=self._poll_confirmations, daemon=True, name="ConfirmationPoller"
        )
        self._confirmation_thread.start()
        self.logger.info("  Confirmation poller: 10ms (instant sync)")

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

    def _poll_confirmations(self):
        """Background thread: polls confirmation file every 10ms for instant position sync."""
        last_mtime = 0.0
        last_heartbeat = 0.0
        while self._running:
            try:
                # Write heartbeat every 1s from poller thread (faster than main cycle)
                now = time.time()
                if now - last_heartbeat >= 1.0:
                    self.bridge.write_heartbeat()
                    last_heartbeat = now

                conf_path = self.bridge.confirmation_path
                if os.path.exists(conf_path):
                    mtime = os.path.getmtime(conf_path)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        confirmations = self.bridge.read_confirmations()
                        for conf in confirmations:
                            status = conf.get("status", "").strip()
                            if status == "CLOSED":
                                close_price = float(conf.get("open_price", 0))
                                ticket      = conf.get("ticket", "?")
                                close_time  = conf.get("timestamp", "")
                                lot_conf    = float(conf.get("lot_size", 0))
                                # Read actual profit from MT5 (accurate P&L in account currency)
                                mt5_profit  = conf.get("profit", "")
                                active = self.signal_generator._active_position
                                if active:
                                    entry_price = active.get("entry_price", close_price)
                                    direction   = active.get("direction", "BUY")
                                    lot         = lot_conf if lot_conf > 0 else active.get("lot_size", 0.01)
                                    entry_time  = active.get("entry_time", "")

                                    # Use MT5's actual profit if available (most accurate)
                                    if mt5_profit and mt5_profit.strip() and float(mt5_profit) != 0:
                                        pnl = float(mt5_profit)
                                    else:
                                        # Fallback: calculate from prices
                                        # For XAUUSD: profit = (close-entry) * lots * 100 (contract size)
                                        if direction == "BUY":
                                            pnl = (close_price - entry_price) * lot * 100
                                        else:
                                            pnl = (entry_price - close_price) * lot * 100

                                    won = pnl > 0
                                    result_str = "WIN" if won else "LOSS"
                                    # ── Full instant trade confirmation ──────────
                                    self.logger.info("=" * 55)
                                    self.logger.info(
                                        f"[TRADE CLOSED] #{ticket} | {result_str}"
                                    )
                                    self.logger.info(
                                        f"  Direction : {direction} {lot} lot"
                                    )
                                    self.logger.info(
                                        f"  Entry     : ${entry_price:.2f}  @  {entry_time}"
                                    )
                                    self.logger.info(
                                        f"  Exit      : ${close_price:.2f}  @  {close_time}"
                                    )
                                    self.logger.info(
                                        f"  P&L       : ${pnl:+.2f}  "
                                        f"{'(+' + str(round(pnl,2)) + ')' if won else '(' + str(round(pnl,2)) + ')'}"
                                    )
                                    if self.brain:
                                        self.logger.info(
                                            f"  Daily P&L : ${self.brain.daily_pnl + pnl:+.2f}"
                                        )
                                    self.logger.info("=" * 55)
                                    if self.auto_optimizer:
                                        trade_context = {
                                            "session": active.get("session", "unknown"),
                                            "confidence": active.get("confidence", 0.5),
                                            "momentum_lookback": active.get("momentum_lookback", 8),
                                            "sl_distance": active.get("sl_distance", 0.6),
                                            "result_pnl": pnl,
                                            "direction": direction,
                                            "rsi_at_entry": active.get("rsi_at_entry", 50.0),
                                        }
                                        self.auto_optimizer.record_trade(trade_context)
                                    # Brain learns from every closed trade
                                    if self.brain:
                                        self.brain.record_trade_closed(
                                            pnl    = pnl,
                                            won    = pnl > 0,
                                            regime = active.get("regime", "unknown"),
                                        )
                                    # Sync account balance from MT5 after close
                                    self._sync_account_balance()
                                    # v7.4: Feed Platt calibrator with trade outcome
                                    if self.confidence_calibrator:
                                        raw_conf = active.get("confidence", 0.5)
                                        trade_regime = active.get("regime", None)
                                        self.confidence_calibrator.record_outcome(
                                            raw_prob=raw_conf,
                                            actual_outcome=1 if won else 0,
                                            regime=trade_regime,
                                        )
                                        self.confidence_calibrator.save_state()
                                    # v7.5: Record true label for meta-learner accumulation
                                    if self.config.enable_online_learning or self.config.enable_feature_monitor:
                                        ensemble = self.signal_generator.ensemble
                                        # Map trade result to label:
                                        # BUY+win=2, SELL+win=0, loss=1(HOLD)
                                        if won:
                                            true_label = 2 if direction == "BUY" else 0
                                        else:
                                            true_label = 1  # HOLD would have been better
                                        # Record for meta-learner data accumulation
                                        ensemble.record_true_label(true_label)
                                        # Feed feature monitor
                                        if self.feature_monitor:
                                            feat_input = active.get("feature_input")
                                            pred_probs = active.get("prediction_probs")
                                            if feat_input is not None and pred_probs is not None:
                                                self.feature_monitor.record(
                                                    feat_input, pred_probs, won
                                                )
                                    # v7.4: Feed Sharpe-based model weighting
                                    # v7.5: Use marginal contribution analysis for attribution
                                    if self.config.enable_sharpe_weights and hasattr(self.signal_generator, 'ensemble'):
                                        ensemble = self.signal_generator.ensemble
                                        individual_preds = active.get("individual_preds", {})
                                        if individual_preds:
                                            # Use marginal contribution for attribution
                                            marginal = ensemble.compute_marginal_contributions(
                                                individual_preds, direction
                                            )
                                            if marginal:
                                                for mn, weight in marginal.items():
                                                    ensemble.update_pnl_attribution(mn, pnl * weight)
                                            else:
                                                # Fallback: attribute equally
                                                for mn in ensemble._model_names:
                                                    ensemble.update_pnl_attribution(mn, pnl / len(ensemble._model_names))
                                        else:
                                            # No individual preds stored (legacy position), attribute equally
                                            for mn in ensemble._model_names:
                                                ensemble.update_pnl_attribution(mn, pnl / len(ensemble._model_names))
                                        ensemble.sharpe_reweight()
                                    self.signal_generator.clear_active_position()
                                    self.signal_generator._estimated_close_pending = True
                                else:
                                    self.logger.info(
                                        f"[INSTANT SYNC] EA closed position #{ticket} "
                                        f"P&L=${float(mt5_profit) if mt5_profit and mt5_profit.strip() else 0:.2f} "
                                        f"(no active tracked)")
                            elif status == "FILLED":
                                # Update active position with actual entry price from MT5
                                actual_price = float(conf.get("open_price", 0))
                                ticket       = conf.get("ticket", "?")
                                lot_filled   = conf.get("lot_size", "?")
                                sl_price     = conf.get("sl", 0)
                                tp_price     = conf.get("tp", 0)
                                slippage_val = conf.get("slippage", "")
                                if self.signal_generator._active_position and actual_price > 0:
                                    self.signal_generator._active_position["entry_price"] = actual_price
                                    self.signal_generator._active_position["entry_time"]  = time.monotonic()
                                    self.signal_generator._active_position["entry_wall_time"] = time.time()
                                    direction = self.signal_generator._active_position.get("direction","BUY")
                                    # ── Instant entry confirmation ───────────────
                                    self.logger.info("=" * 55)
                                    self.logger.info(
                                        f"[TRADE OPENED] #{ticket} | {direction}"
                                    )
                                    self.logger.info(f"  Entry : ${actual_price:.2f}")
                                    if sl_price:
                                        self.logger.info(f"  SL    : ${float(sl_price):.2f}")
                                    if tp_price:
                                        self.logger.info(f"  TP    : ${float(tp_price):.2f}")
                                    if lot_filled:
                                        self.logger.info(f"  Lot   : {lot_filled}")
                                    # Track slippage from EA confirmation
                                    if slippage_val and slippage_val.strip():
                                        slip = float(slippage_val)
                                        self.logger.info(f"  Slip  : ${slip:.4f}")
                                        if self.slippage_tracker:
                                            result = self.slippage_tracker.record_fill(
                                                direction=direction,
                                                slippage=slip,
                                                fill_price=actual_price,
                                                ticket=ticket,
                                            )
                                            quality = result.get("quality_score", 1.0)
                                            if quality < 0.7:
                                                self.logger.warning(
                                                    f"[SlippageTracker] Fill quality degraded: "
                                                    f"{quality:.2f} (avg slip: ${result.get('avg_slippage', 0):.4f})"
                                                )
                                    self.logger.info("=" * 55)
                        self.bridge.clear_confirmations()
            except Exception as e:
                # Log the error so missed confirmations are visible (not silently swallowed)
                self.logger.debug(f"[ConfPoller] Error processing confirmation: {e}")
            time.sleep(0.01)  # 10ms polling — near-instant trade sync

    def _sync_account_balance(self):
        """
        Sync account balance from MT5 balance file.

        Reads python_bridge_balance.csv (written by EA after each trade close)
        and updates brain.account_balance and monte_carlo_risk configs to keep
        risk sizing accurate as the account grows/shrinks.
        """
        if not self._account_sync_config:
            return

        try:
            balance_data = self.bridge.read_balance(
                self._account_sync_config.balance_file
            )
            if balance_data and balance_data.get("balance", 0) > 0:
                new_balance = balance_data["balance"]
                old_balance = None

                # Update Brain config
                if self.brain:
                    old_balance = self.brain.config.account_balance
                    self.brain.config.account_balance = new_balance

                # Update Monte Carlo risk config
                if self.monte_carlo_risk and hasattr(self.monte_carlo_risk, '_brain_config'):
                    self.monte_carlo_risk._brain_config.account_balance = new_balance

                # Log the update
                if old_balance and abs(new_balance - old_balance) > 0.01:
                    self.logger.info(
                        f"[AccountSync] Balance updated: "
                        f"${old_balance:.2f} -> ${new_balance:.2f} "
                        f"(equity: ${balance_data.get('equity', 0):.2f})"
                    )
        except Exception as e:
            self.logger.debug(f"[AccountSync] Error: {e}")

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

            # 1-8: Data fetch and compute
            # 1. Fetch market data
            df = self.data_fetcher.fetch_ohlcv(interval="1h", period="3mo")
            if df.empty:
                result["error"] = "No market data available"
                return result

            # 1a. Validate fetched data (v7.5)
            if self.data_validator:
                val_result = self.data_validator.validate(df)
                if not val_result.is_valid:
                    self.logger.error(
                        f"[DataValidator] CRITICAL: {val_result.summary()}"
                    )
                    result["error"] = f"Data validation failed: {val_result.summary()}"
                    return result
                elif val_result.warnings:
                    for w in val_result.warnings:
                        self.logger.warning(f"[DataValidator] {w}")

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

            # 1c. Validate M1 data if available (v7.5)
            if self.data_validator and df_m1 is not None:
                m1_val = self.data_validator.validate(df_m1)
                if not m1_val.is_valid:
                    self.logger.warning(
                        f"[DataValidator] M1 data invalid: {m1_val.summary()} - "
                        f"falling back to H1"
                    )
                    df_m1 = None
                elif m1_val.warnings:
                    for w in m1_val.warnings:
                        self.logger.debug(f"[DataValidator M1] {w}")

            # 1d. Tick data staleness check (v7.5)
            if self.tick_data_processor and self.data_validator:
                if self.tick_data_processor.is_tick_stale():
                    self.logger.debug(
                        "[DataValidator] Tick data is stale - order flow "
                        "features may be outdated"
                    )

            # 2. Compute features
            features_df = self.data_fetcher.compute_features(df)
            if features_df.empty:
                result["error"] = "Insufficient data for features"
                return result

            # 3. Get latest sequence for prediction
            # Pass the already-fetched df so get_latest_features doesn't make
            # a second yfinance request with period=1y (was the source of the
            # extra GC=F rate-limit errors logged as period=1y).
            seq_length = 64
            feature_input = self.data_fetcher.get_latest_features(seq_length, df=df)
            if feature_input is None:
                result["error"] = "Could not prepare model input"
                return result

            # 3a. Validate feature schema (v7.5)
            if self.data_validator:
                feat_val = self.data_validator.validate_features(feature_input)
                if not feat_val.is_valid:
                    self.logger.error(
                        f"[DataValidator] Feature validation CRITICAL: {feat_val.summary()}"
                    )
                    result["error"] = f"Feature validation failed: {feat_val.summary()}"
                    return result
                elif feat_val.warnings:
                    for w in feat_val.warnings:
                        self.logger.warning(f"[DataValidator] {w}")

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
                spread_points=(
                    self.spread_monitor.get_current_spread()
                    if self.spread_monitor else None
                ),
            )

            # 9a. Store feature_input and prediction_probs on active position
            # for feature monitor and online learning feedback
            if (self.config.enable_feature_monitor or self.config.enable_online_learning):
                if self.signal_generator._active_position and signal.action != "HOLD":
                    self.signal_generator._active_position["feature_input"] = feature_input
                    pred_result = self.signal_generator.ensemble.predict(feature_input)
                    self.signal_generator._active_position["prediction_probs"] = (
                        pred_result["probabilities"]
                    )

            # 9b. Apply Platt scaling confidence calibration (v7.4)
            if self.confidence_calibrator and signal.action != "HOLD":
                raw_confidence = signal.confidence
                # v7.5: Use regime-conditional calibration
                calibrated = self.confidence_calibrator.calibrate_for_regime(
                    np.array([raw_confidence]),
                    regime=signal.regime,
                )[0]
                signal.confidence = float(calibrated)
                if abs(calibrated - raw_confidence) > 0.01:
                    self.logger.debug(
                        f"[PlattCal] Confidence: {raw_confidence:.3f} -> {calibrated:.3f} "
                        f"(regime: {signal.regime})"
                    )

            # 10. Brain evaluation — fully autonomous decision layer
            # The brain overrides lot size, SL, and TP; can veto the signal entirely

            # Brain pulse: log status every cycle so you can see it's always working
            if self.brain:
                t_status, t_session, t_mult = self.brain.timing.evaluate(self.brain.config)
                e_status, _, _, recent_pf   = self.brain.edge.evaluate(self.brain.config)
                dd_stage = self.brain.dd_recovery.get_stage(self.brain.total_drawdown)
                mt5_conn = self.bridge.get_mt5_connection_status()
                mt5_tag  = "[LIVE]" if mt5_conn["connected"] else "[OFFLINE]"
                self.logger.info(
                    f"[Brain] session={t_session}({t_status}) "
                    f"edge={e_status} PF={recent_pf:.2f} "
                    f"daily=${self.brain.daily_pnl:+.2f} "
                    f"dd={self.brain.total_drawdown*100:.1f}% "
                    f"stage={dd_stage['label']} "
                    f"signal={signal.action} | {mt5_tag} ({mt5_conn['status_str']})"
                )

            if self.brain and signal.action != "HOLD":
                # Gather recent OHLCV for regime detection
                closes = list(df_m1["Close"].tail(60)) if df_m1 is not None else list(df["Close"].tail(60))
                highs  = list(df_m1["High"].tail(60))  if df_m1 is not None else list(df["High"].tail(60))
                lows   = list(df_m1["Low"].tail(60))   if df_m1 is not None else list(df["Low"].tail(60))
                avg_atr = float(np.mean(self._atr_history)) if self._atr_history else atr
                atr_dollars = atr * signal.lot_size * 100  # approx $ per pip × lot

                # Daily brain reset at new trading day
                today = datetime.now().strftime("%Y-%m-%d")
                if today != self._last_trade_date:
                    self.brain.soft_reset()
                    self._last_trade_date = today

                decision = self.brain.evaluate(
                    signal        = signal,
                    closes        = closes,
                    highs         = highs,
                    lows          = lows,
                    atr           = atr,
                    avg_atr       = avg_atr,
                    spread_points = 10.0,       # approximate; EA reads live spread
                    tick_volume   = 100.0,       # placeholder
                    atr_dollars   = max(0.5, atr_dollars),
                )

                if not decision.should_trade:
                    # Brain vetoed — skip this signal
                    self.logger.info(f"[Brain] VETOED: {decision.reasoning.get('final','')}")
                    signal = type(signal)(
                        timestamp=signal.timestamp, symbol=signal.symbol, action="HOLD",
                        confidence=0.0, sl_pips=0, tp_pips=0,
                        lot_size=0.0, model_name=signal.model_name, regime=decision.regime,
                    )
                    # Clear ghost position — trade never reached MT5
                    self.signal_generator.clear_active_position()
                else:
                    # Brain approved — apply its sizing, SL, TP
                    signal.lot_size = decision.lot_size
                    if decision.sl_dollars > 0:
                        # Convert $ SL to pips (1 pip XAUUSD ≈ $1 per 0.01 lot)
                        pip_value = signal.lot_size * 100 if signal.lot_size > 0 else 0.01
                        signal.sl_pips = int(decision.sl_dollars / max(pip_value, 0.001) * 10)
                    if decision.tp_dollars > 0:
                        pip_value = signal.lot_size * 100 if signal.lot_size > 0 else 0.01
                        signal.tp_pips = int(decision.tp_dollars / max(pip_value, 0.001) * 10)
                    self.logger.info(decision.log_summary())

            # 10b. Entry timing - micro-pullback check (v7.4)
            if self.entry_timing and signal.action != "HOLD":
                # If no pending signal, register this one
                if not self.entry_timing.has_pending_signal:
                    # Capture individual_preds before active_position may be cleared
                    _individual_preds = {}
                    if self.signal_generator._active_position:
                        _individual_preds = self.signal_generator._active_position.get("individual_preds", {})
                    self.entry_timing.set_pending_signal(
                        action=signal.action,
                        price=current_price,
                        signal_details={
                            "confidence": signal.confidence,
                            "sl_pips": signal.sl_pips,
                            "tp_pips": signal.tp_pips,
                            "lot_size": signal.lot_size,
                            "model_name": signal.model_name,
                            "regime": signal.regime,
                            "symbol": signal.symbol,
                            "individual_preds": _individual_preds,
                        },
                        atr=atr,
                        avg_atr=float(np.mean(self._atr_history)) if self._atr_history else atr,
                    )
                # Evaluate whether to enter now
                entry_result = self.entry_timing.evaluate_entry(
                    action=signal.action,
                    current_price=current_price,
                )
                if not entry_result['should_enter']:
                    # Still waiting for pullback - skip bridge write this cycle
                    self.logger.debug(
                        f"[EntryTiming] {entry_result['reason']}"
                    )
                    # Don't write to bridge yet, but don't HOLD either
                    # (signal stays active, we just delay the execution)
                    signal = type(signal)(
                        timestamp=signal.timestamp, symbol=signal.symbol, action="HOLD",
                        confidence=signal.confidence, sl_pips=signal.sl_pips,
                        tp_pips=signal.tp_pips, lot_size=signal.lot_size,
                        model_name=signal.model_name, regime=signal.regime,
                    )
                    self.signal_generator.clear_active_position()
                else:
                    self.logger.info(
                        f"[EntryTiming] Entering: {entry_result['reason']}"
                    )
            elif self.entry_timing and signal.action == "HOLD":
                # If we have a pending signal but current cycle is HOLD,
                # keep checking the pending signal for pullback
                if self.entry_timing.has_pending_signal:
                    status = self.entry_timing.get_status()
                    entry_result = self.entry_timing.evaluate_entry(
                        action=status['action'],
                        current_price=current_price,
                    )
                    if entry_result['should_enter']:
                        # Reconstruct the signal from pending state and write to bridge
                        self.logger.info(
                            f"[EntryTiming] Pending signal triggered: {entry_result['reason']}"
                        )
                        details = status.get('signal_details') or {}
                        from strategies.signal_generator import TradeSignal
                        signal = TradeSignal(
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            symbol=details.get("symbol", "XAUUSD"),
                            action=status['action'],
                            confidence=details.get("confidence", 0.5),
                            sl_pips=details.get("sl_pips", 0),
                            tp_pips=details.get("tp_pips", 0),
                            lot_size=details.get("lot_size", 0.01),
                            model_name=details.get("model_name", "ensemble"),
                            regime=details.get("regime", "unknown"),
                        )
                        # Re-establish _active_position so confirmation poller
                        # can track P&L, Platt recording, and Sharpe attribution
                        self.signal_generator._position_id_counter += 1
                        self.signal_generator._active_position = {
                            "position_id": self.signal_generator._position_id_counter,
                            "direction": status['action'],
                            "entry_price": current_price,
                            "entry_time": time.monotonic(),
                            "entry_wall_time": time.time(),
                            "signal_context": {
                                "confidence": details.get("confidence", 0.5),
                                "session": details.get("regime", "unknown"),
                                "sl_distance": details.get("sl_pips", 0) * 0.1,
                            },
                            "individual_preds": details.get("individual_preds", {}),
                            "regime": details.get("regime", "unknown"),
                        }

            # 10c. Write signal to bridge
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

            # 12. Confirmations now handled by background _poll_confirmations thread
            # (100ms polling for instant position sync)

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

            # 14. Feature monitor periodic degradation check
            if self.feature_monitor:
                self._feature_monitor_cycle_count += 1
                if (self._feature_monitor_cycle_count
                        % self._feature_monitor_log_interval == 0):
                    if self.feature_monitor.has_sufficient_data:
                        fm_result = self.feature_monitor.check_degradation()
                        if fm_result.alert:
                            self.logger.warning(
                                f"[FeatureMonitor] {len(fm_result.degraded_features)} "
                                f"features degraded: {fm_result.degraded_features[:5]}"
                            )
                        else:
                            top = self.feature_monitor.get_top_features(3)
                            self.logger.debug(
                                f"[FeatureMonitor] OK - top features: {top}"
                            )

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
        self.logger.info(f"NeuroX v{VERSION} starting...")
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
    print(f"  NeuroX for MetaTrader 5 v{VERSION}")
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
