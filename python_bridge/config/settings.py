"""
=============================================================
  Python ML Bridge - Configuration Settings
  Central configuration for all model parameters, data sources,
  signal thresholds, and file paths for MT5 bridge communication.
=============================================================
"""

import os
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
#  FILE PATHS
# ─────────────────────────────────────────────
# MT5 Common Files folder (Windows path when running on trading machine)
MT5_COMMON_PATH = os.environ.get(
    "MT5_COMMON_PATH",
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                 "MetaQuotes", "Terminal", "Common", "Files")
)

# Signal file that Python writes and MT5 reads
SIGNAL_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_signal.csv")

# Execution confirmation file that MT5 writes and Python reads
CONFIRMATION_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_confirm.csv")

# Model checkpoint directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


# ─────────────────────────────────────────────
#  MODEL PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class TransformerConfig:
    """Transformer model hyperparameters."""
    input_features: int = 46        # Number of input features
    d_model: int = 256              # Model dimension
    n_heads: int = 8                # Number of attention heads
    n_layers: int = 4               # Number of encoder layers
    d_ff: int = 1024                # Feed-forward dimension
    dropout: float = 0.1            # Dropout rate
    seq_length: int = 64            # Input sequence length
    num_classes: int = 3            # BUY, SELL, HOLD
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15              # Early stopping patience


@dataclass
class LSTMConfig:
    """Bidirectional LSTM model hyperparameters."""
    input_features: int = 46        # Number of input features
    hidden_size: int = 128          # LSTM hidden units
    num_layers: int = 3             # Number of LSTM layers
    dropout: float = 0.3            # Dropout rate
    bidirectional: bool = True      # Bidirectional LSTM
    attention: bool = True          # Use attention mechanism
    seq_length: int = 64            # Input sequence length
    num_classes: int = 3            # BUY, SELL, HOLD
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15


@dataclass
class EnsembleConfig:
    """Ensemble model configuration."""
    transformer_weight: float = 0.40
    lstm_weight: float = 0.35
    gradient_boost_weight: float = 0.25
    meta_learner_features: int = 9  # 3 models x 3 classes
    min_agreement: float = 0.10     # Very low agreement threshold (bypassed in aggressive mode)
    dynamic_weights: bool = True    # Adjust weights based on recent accuracy
    weight_lookback: int = 50       # Number of recent predictions for weight calc


# ─────────────────────────────────────────────
#  DATA SOURCES
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    """Data source configuration."""
    symbol: str = "XAUUSD"
    yfinance_ticker: str = "GC=F"           # Gold futures on Yahoo Finance
    vix_ticker: str = "^VIX"                # VIX fear index
    dxy_ticker: str = "DX-Y.NYB"            # US Dollar Index
    oil_ticker: str = "CL=F"                # WTI Crude Oil
    yield_10y_ticker: str = "^TNX"          # 10-Year Treasury Yield
    yield_2y_ticker: str = "^IRX"           # 13-Week Treasury Bill

    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])

    # Data lookback
    lookback_days: int = 365
    update_interval_minutes: int = 1        # How often to fetch new data


# ─────────────────────────────────────────────
#  SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
@dataclass
class SentimentConfig:
    """Sentiment analysis configuration."""
    model_name: str = "ProsusAI/finbert"    # HuggingFace FinBERT model
    max_articles: int = 50                   # Max articles to analyze
    rolling_window: int = 24                 # Hours for rolling sentiment
    min_confidence: float = 0.6              # Minimum confidence threshold

    # RSS feed URLs for financial news
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
        "https://www.investing.com/rss/news_301.rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XAUUSD=X&region=US&lang=en-US",
    ])


# ─────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────
@dataclass
class SignalConfig:
    """Signal generation thresholds."""
    min_confidence: float = 0.15            # Very low threshold for aggressive trading
    strong_confidence: float = 0.40         # Strong signal threshold
    atr_sl_multiplier: float = 1.5          # SL = ATR * multiplier
    atr_tp_multiplier: float = 2.5          # TP = ATR * multiplier (1:1.67 R:R)
    max_signal_age_seconds: int = 300       # Signal expires after 5 minutes
    cooldown_seconds: int = 5               # Minimal cooldown for aggressive trading


# ─────────────────────────────────────────────
#  RISK MANAGEMENT
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.02        # 2% risk per trade
    max_daily_loss: float = 0.05            # 5% max daily loss
    max_drawdown: float = 0.10              # 10% max drawdown before halt
    max_correlation: float = 0.7            # Max correlation between open positions
    max_open_positions: int = 3             # Maximum simultaneous positions
    kelly_fraction: float = 0.25            # Quarter-Kelly for safety
    account_balance: float = 10000.0        # Default account balance
    min_lot_size: float = 0.01              # Minimum lot size
    max_lot_size: float = 1.0               # Maximum lot size

    # Time filters (UTC hours to avoid trading)
    no_trade_hours: List[int] = field(default_factory=lambda: [])
    # Days to avoid (0=Monday, 4=Friday afternoon)
    reduced_risk_days: List[int] = field(default_factory=lambda: [4])


# ─────────────────────────────────────────────
#  REGIME DETECTION
# ─────────────────────────────────────────────
@dataclass
class RegimeConfig:
    """Market regime detection parameters."""
    n_regimes: int = 4                      # trending, ranging, volatile, crash
    lookback_bars: int = 100                # Bars for regime detection
    volatility_threshold: float = 1.5       # Above avg = volatile
    trend_strength_threshold: float = 25.0  # ADX threshold for trending
    regime_names: List[str] = field(default_factory=lambda: [
        "trending", "ranging", "volatile", "crash"
    ])


# ─────────────────────────────────────────────
#  MULTI-TIMEFRAME ANALYSIS
# ─────────────────────────────────────────────
@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe data pipeline configuration.

    Professional traders analyze multiple timeframes to confirm
    trend direction, identify key support/resistance levels, and
    time entries precisely. This config enables M1/M5/M15/H1/H4
    analysis for institutional-grade signal quality.
    """
    timeframes: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h"
    ])
    # Period to fetch for each timeframe (yfinance limits differ per interval)
    periods: dict = field(default_factory=lambda: {
        "1m": "7d",       # yfinance max for 1m
        "5m": "60d",
        "15m": "60d",
        "1h": "2y",
        "4h": "2y",       # We'll use 1h and resample to 4h
    })
    # Feature aggregation across timeframes
    aggregate_method: str = "concat"  # concat | weighted_avg
    # Minimum bars required per timeframe for valid features
    min_bars: int = 200
    # Higher timeframe trend confirmation weight
    htf_trend_weight: float = 0.6
    # Lower timeframe entry precision weight
    ltf_entry_weight: float = 0.4
    # Enable alignment (forward-fill lower timeframe features to match)
    align_to_lowest: bool = True


# ─────────────────────────────────────────────
#  NEWS CALENDAR FILTER
# ─────────────────────────────────────────────
@dataclass
class NewsFilterConfig:
    """News calendar filter to avoid trading during high-impact events.

    Professional traders never hold positions through NFP, FOMC, or CPI
    releases. These events cause extreme volatility with unpredictable
    direction, making technical signals unreliable. This filter gates
    all trade entries during critical event windows.
    """
    high_impact_events: List[str] = field(default_factory=lambda: [
        "NFP", "FOMC", "CPI", "ECB", "BOE", "BOJ",
        "GDP", "Retail Sales", "PMI", "Interest Rate Decision",
        "Non-Farm Payrolls", "Consumer Price Index",
        "Federal Funds Rate", "ECB Interest Rate",
        "BOE Interest Rate", "BOJ Interest Rate"
    ])
    # Minutes to stop trading before event
    minutes_before: int = 30
    # Minutes to wait after event before resuming
    minutes_after: int = 30
    # Calendar data source URL (free investing.com RSS)
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    # Local cache path for fetched calendar data
    cache_file: str = "news_calendar_cache.json"
    # How often to refresh calendar (hours)
    refresh_interval_hours: int = 6
    # Enable strict mode (also avoid medium-impact USD events for gold)
    strict_mode: bool = True
    # Currencies to monitor (events affecting these currencies matter for XAUUSD)
    monitored_currencies: List[str] = field(default_factory=lambda: [
        "USD", "EUR", "GBP", "JPY", "CHF"
    ])


# ─────────────────────────────────────────────
#  MULTI-PAIR CORRELATION
# ─────────────────────────────────────────────
@dataclass
class MultiPairConfig:
    """Multi-pair support for cross-market correlation analysis.

    Gold is inversely correlated with USD strength. Professional traders
    monitor EURUSD, GBPUSD, USDJPY, DXY, and bond yields to gauge USD
    direction. Cross-pair momentum and divergence signals provide
    additional confirmation and early warnings for XAUUSD moves.
    """
    pairs: List[str] = field(default_factory=lambda: [
        "XAUUSD", "EURUSD", "GBPUSD", "USDJPY"
    ])
    # Yahoo Finance ticker mapping for each pair
    yfinance_tickers: dict = field(default_factory=lambda: {
        "XAUUSD": "GC=F",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "DXY": "DX-Y.NYB",
    })
    # Correlation rolling window (bars)
    correlation_window: int = 50
    # Cross-pair momentum lookback periods
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    # Relative strength comparison window
    relative_strength_window: int = 20
    # Expected correlations (for divergence detection)
    expected_correlations: dict = field(default_factory=lambda: {
        "XAUUSD_EURUSD": 0.6,    # Gold and EUR tend to move together (anti-USD)
        "XAUUSD_DXY": -0.8,      # Gold strongly inverse to USD
        "XAUUSD_USDJPY": -0.4,   # Gold inverse to USD/JPY
        "EURUSD_GBPUSD": 0.7,    # EUR and GBP correlated
    })
    # Divergence threshold (correlation deviation that signals opportunity)
    divergence_threshold: float = 0.3
    # Enable inter-market momentum as features
    enable_cross_features: bool = True


# ─────────────────────────────────────────────
#  REINFORCEMENT LEARNING (DQN)
# ─────────────────────────────────────────────
@dataclass
class RLConfig:
    """DQN Reinforcement Learning agent configuration.

    The RL agent learns position management like an experienced prop trader:
    when to hold, when to cut losses quickly, when to take partial profits,
    and when to let winners run with a tightened stop.
    """
    # Replay buffer
    replay_buffer_size: int = 10000          # Experience replay buffer capacity
    batch_size: int = 64                      # Mini-batch size for training
    # Q-learning parameters
    gamma: float = 0.99                       # Discount factor (value future rewards)
    epsilon_start: float = 1.0                # Initial exploration rate
    epsilon_end: float = 0.01                 # Minimum exploration rate
    epsilon_decay: float = 0.995              # Epsilon decay per training step
    learning_rate: float = 1e-3               # Adam learning rate
    target_update_freq: int = 100             # Steps between target network updates
    # Network architecture
    hidden_size: int = 256                    # Hidden layer size (3-layer MLP)
    state_size: int = 52                      # State dim: features + position info
    num_actions: int = 5                      # HOLD, CLOSE_FULL, CLOSE_25, CLOSE_50, TIGHTEN
    # Reward shaping (prop trader style)
    hold_penalty_per_bar: float = -0.001      # Small penalty for holding (encourages decisions)
    cut_loser_bonus: float = 0.5              # Bonus reward for cutting losing trades early
    max_hold_bars_penalty: int = 100          # Hard penalty threshold for holding too long
    winner_run_bonus: float = 0.3             # Bonus for letting winners run past 1R
    # Training
    min_replay_size: int = 256                # Minimum transitions before training starts
    tau: float = 0.005                        # Soft target network update rate


# ─────────────────────────────────────────────
#  SMART EXIT MANAGEMENT
# ─────────────────────────────────────────────
@dataclass
class SmartExitConfig:
    """Smart AI-driven exit management configuration.

    Implements dynamic trailing stops and partial closes like a professional
    prop trader. High-confidence signals get wide trailing stops (let profits
    run). Low/decaying confidence triggers tight stops (protect capital).
    Partial closes at key R-multiples lock in profits while maintaining
    upside exposure.
    """
    # Trailing stop parameters
    trailing_atr_mult_tight: float = 0.5      # Tight trail when confidence is low
    trailing_atr_mult_wide: float = 2.0       # Wide trail when confidence is high
    confidence_decay_threshold: float = 0.3   # Below this: use tight trail
    confidence_strong_threshold: float = 0.6  # Above this: use wide trail
    # Partial close rules (prop trader style)
    partial_close_at_2r: bool = True          # Close partial at 2R profit
    partial_close_pct: float = 0.5            # Close 50% at first target
    partial_close_at_3r: bool = True          # Close more at 3R
    partial_close_3r_pct: float = 0.25        # Close 25% more at 3R
    # Position management
    max_hold_bars: int = 100                  # Maximum bars to hold a position
    break_even_at_1r: bool = True             # Move stop to break-even at 1R profit
    # Time-based decay
    confidence_time_decay: float = 0.995      # Confidence decays per bar held
    min_confidence_to_hold: float = 0.15      # Below this: close immediately


# ─────────────────────────────────────────────
#  AUTO-RETRAINING SCHEDULER
# ─────────────────────────────────────────────
@dataclass
class RetrainConfig:
    """Weekend auto-retraining scheduler configuration.

    Professional trading firms retrain models on weekends when markets are
    closed. Walk-forward validation ensures new models genuinely improve
    before deployment. This prevents overfitting and adapts to changing
    market regimes automatically.
    """
    retrain_day: str = "Saturday"             # Day to trigger retraining
    min_days_between: int = 7                 # Minimum days between retrains
    min_improvement_pct: float = 1.0          # Deploy only if >1% better
    walk_forward_weeks: int = 2               # Walk-forward validation window
    max_retrain_attempts: int = 3             # Max retrain attempts per session
    save_retrain_history: bool = True         # Log retrain results to JSON
    retrain_log_file: str = "retrain_history.json"  # History log filename
    incorporate_trade_outcomes: bool = True   # Use trade results as training signal
    min_trades_for_retrain: int = 20          # Minimum trades before incorporating


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
@dataclass
class MainConfig:
    """Main loop configuration."""
    interval_seconds: int = 15              # Fast 15-second cycle for aggressive trading
    log_level: str = "INFO"
    enable_sentiment: bool = True
    enable_alternative_data: bool = True
    enable_regime_detection: bool = True
    enable_multi_timeframe: bool = True     # Multi-timeframe analysis
    enable_news_filter: bool = True         # News calendar event filter
    enable_multi_pair: bool = True          # Cross-pair correlation analysis
    enable_smart_exits: bool = True         # AI-driven exit management (RL agent)
    enable_auto_retrain: bool = True        # Weekend auto-retraining scheduler
    paper_trading: bool = True              # Paper trading mode by default
