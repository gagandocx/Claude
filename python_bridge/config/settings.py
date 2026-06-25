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
    batch_size_gpu: int = 256       # Larger batch size for GPU training
    epochs: int = 150
    patience: int = 20              # Early stopping patience
    label_smoothing: float = 0.1    # Reduces overconfident predictions


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
    batch_size_gpu: int = 256       # Larger batch size for GPU training
    epochs: int = 150
    patience: int = 20
    label_smoothing: float = 0.1    # Reduces overconfident predictions


@dataclass
class TCNConfig:
    """Temporal Convolutional Network hyperparameters."""
    input_features: int = 46        # Number of input features (matches Transformer/LSTM)
    n_filters: int = 64             # Conv channels throughout the network
    kernel_size: int = 3            # Dilated kernel size
    n_layers: int = 6               # Dilation: 1, 2, 4, 8, 16, 32 → covers 64-bar window
    dropout: float = 0.2
    num_classes: int = 3            # BUY, SELL, HOLD
    seq_length: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class XGBoostConfig:
    """LightGBM / XGBoost / HistGradientBoosting configuration."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 20
    num_leaves: int = 63            # LightGBM-specific (ignored by other backends)
    reg_alpha: float = 0.1          # L1 regularisation
    reg_lambda: float = 0.1         # L2 regularisation
    random_state: int = 42
    use_lightgbm: bool = True       # Prefer LightGBM if installed


@dataclass
class PatchTSTConfig:
    """PatchTST — patch-based transformer (Nie et al., MIT/IBM 2023)."""
    input_features: int = 46
    seq_length: int = 64
    patch_size: int = 8         # 64 / 8 = 8 patch tokens (vs 64 individual-bar tokens)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class TFTConfig:
    """Temporal Fusion Transformer (Lim et al., Google 2021)."""
    input_features: int = 46
    seq_length: int = 64
    d_model: int = 64           # TFT hidden size (VSN + GRN + LSTM + MHA all use d_model)
    num_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class NHiTSConfig:
    """N-HiTS — Neural Hierarchical Interpolation (Challu et al., Mila 2022)."""
    input_features: int = 46
    seq_length: int = 64
    hidden_size: int = 256
    block_output_size: int = 64     # Each block outputs this many features before fusion
    pool_sizes: list = field(default_factory=lambda: [8, 4, 2, 1])  # Macro → micro
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class CatBoostConfig:
    """CatBoost — ordered boosting (Prokhorenkova et al., Yandex 2018)."""
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    border_count: int = 128
    random_seed: int = 42
    early_stopping_rounds: int = 50
    verbose: bool = False


@dataclass
class ITransformerConfig:
    """iTransformer — inverted feature-space attention (Liu et al., ICLR 2024)."""
    input_features: int = 46
    seq_length: int = 64        # Used as the embedding dim per feature token
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class MambaConfig:
    """Mamba S6 — selective state space model (Gu & Dao, CMU/Princeton 2023)."""
    input_features: int = 46
    seq_length: int = 64
    d_model: int = 128          # Hidden dimension
    d_state: int = 16           # SSM state dimension
    d_conv: int = 4             # Local depthwise conv kernel
    expand: int = 2             # d_inner = expand × d_model
    n_layers: int = 4           # Number of Mamba blocks
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class DLinearConfig:
    """DLinear — decomposition linear model (Zeng et al., AAAI 2023)."""
    input_features: int = 46
    seq_length: int = 64
    kernel_size: int = 25       # Moving-average window for trend extraction
    hidden_size: int = 128      # MLP classifier hidden size
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64        # Simple model — larger batch is fine
    batch_size_gpu: int = 512
    epochs: int = 80
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class xLSTMConfig:
    """xLSTM mLSTM variant — matrix memory LSTM (Hochreiter et al., JKU 2024)."""
    input_features: int = 46
    seq_length: int = 64
    d_model: int = 128
    n_heads: int = 4            # head_dim = d_model // n_heads = 32
    n_layers: int = 4           # mLSTM blocks stacked
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 5e-5 # lower LR suits matrix memory updates
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 80
    patience: int = 12
    label_smoothing: float = 0.1


@dataclass
class TimesNetConfig:
    """TimesNet — 2D temporal variation via FFT periods (Wu et al., ICLR 2023)."""
    input_features: int = 46
    seq_length: int = 64
    d_model: int = 64
    top_k: int = 3              # top-k dominant periods discovered via FFT
    dropout: float = 0.1
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    batch_size_gpu: int = 256
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.1


@dataclass
class EnsembleConfig:
    """Ensemble model configuration — 14-model stack."""
    # Weights must sum to 1.0
    transformer_weight: float  = 0.07   # Global self-attention
    lstm_weight: float         = 0.06   # BiLSTM + attention
    tcn_weight: float          = 0.06   # Dilated temporal conv
    patch_tst_weight: float    = 0.10   # Patch-based SOTA 2023
    tft_weight: float          = 0.10   # Financial VSN+GRN
    nhits_weight: float        = 0.06   # Hierarchical macro→micro
    itransformer_weight: float = 0.10   # Feature-space attention 2024
    mamba_weight: float        = 0.08   # Selective state space 2023
    dlinear_weight: float      = 0.04   # Trend/residual decomposition
    xlstm_weight: float        = 0.11   # Matrix memory LSTM 2024 (NEW)
    timesnet_weight: float     = 0.09   # 2D temporal variation 2023 (NEW)
    gradient_boost_weight: float = 0.05 # sklearn HistGradBoost
    xgboost_weight: float      = 0.04   # LightGBM / XGBoost
    catboost_weight: float     = 0.04   # Ordered boosting
    meta_learner_features: int = 42     # 14 models × 3 classes
    min_agreement: float       = 0.10
    dynamic_weights: bool      = True
    weight_lookback: int       = 50


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

    # Multi-timeframe training parameters
    training_periods: list = field(default_factory=lambda: [
        {"period": "7d", "interval": "1m"},
        {"period": "60d", "interval": "15m"},
        {"period": "2y", "interval": "1h"},
    ])

    # Support/resistance detection
    sr_lookback: int = 100                  # Bars for S/R level detection

    # Momentum parameters
    momentum_lookback: int = 8              # 8-bar momentum lookback (optimal for M1 scalping)
    momentum_threshold: float = 0.60        # Min price move for momentum ($0.60 for gold - more entries)

    # RSI exhaustion filter thresholds
    rsi_overbought: int = 65               # RSI above this = overbought
    rsi_oversold: int = 35                 # RSI below this = oversold

    # ATR-based labeling threshold
    atr_label_threshold: float = 0.5       # Label BUY/SELL only for strong moves > 0.5*ATR (reduces noise)


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
    min_confidence: float = 0.25            # Higher bar for entries (quality over quantity)
    strong_confidence: float = 0.40         # Strong signal threshold
    atr_sl_multiplier: float = 0.2          # SL = ATR * 0.2 (with M1 ATR ~$3, gives ~$0.60 SL)
    atr_tp_multiplier: float = 0.0          # TP = 0 -> EA manages exit dynamically (no fixed TP)
    max_signal_age_seconds: int = 300       # Signal expires after 5 minutes
    cooldown_seconds: int = 2               # 2-cycle cooldown (~20 seconds) between signals
    max_hold_seconds: int = 300             # 5 minutes max hold for a single position
    max_hold_bars: int = 20                 # 20 M1 bars (20 min) max hold for position management
    max_positions: int = 1                  # Only 1 position at a time (Python-side enforcement)


# ─────────────────────────────────────────────
#  RISK MANAGEMENT
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.02        # 2% risk per trade
    max_daily_loss: float = 0.05            # 5% max daily loss (percentage)
    max_daily_loss_dollars: float = 50.0    # Absolute dollar drawdown cap per day
    max_drawdown: float = 0.10              # 10% max drawdown before halt
    max_correlation: float = 0.7            # Max correlation between open positions
    max_open_positions: int = 4             # Allow 4 concurrent positions for maximum opportunities
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
    time entries precisely. For M1 scalping, only 5m and 15m are
    used for HTF bias (faster reaction than H1/H4). The full list
    of timeframes is retained for training and feature extraction.
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
    # Minutes to stop trading before event (soft block with volatility check)
    minutes_before: int = 10
    # Minutes for hard block before event (no exceptions, no trading at all)
    hard_block_minutes: int = 2
    # Minutes to wait after event before resuming (max post-event window)
    minutes_after: int = 30
    # Post-news volatility check: how often to re-check (seconds)
    post_news_check_interval: int = 60
    # Post-news volatility threshold: ATR multiplier (resume if below this)
    post_news_volatility_threshold: float = 2.0
    # Post-news minimum wait after event time before checking volatility (minutes)
    post_news_min_wait: int = 2
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
#  LIVE PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────
@dataclass
class DashboardConfig:
    """Live performance dashboard configuration.

    Professional prop desks track every metric in real-time. This
    dashboard computes institutional-grade analytics: Sharpe ratio,
    Sortino ratio, profit factor, per-model alpha, and per-regime
    performance. Updated after every trade closure and rendered
    periodically to console, log, and HTML report.
    """
    update_interval_trades: int = 1          # Update after every N trade closures
    html_output_path: str = "dashboard/report.html"  # HTML report output path
    console_refresh_seconds: int = 60        # Console dashboard refresh interval
    track_per_model: bool = True             # Track per-model performance
    track_per_regime: bool = True            # Track per-regime performance
    min_trades_for_stats: int = 10           # Min trades before stats are meaningful
    enable_console: bool = True              # Show console dashboard
    enable_html_report: bool = True          # Generate HTML reports
    enable_log_output: bool = True           # Structured log output
    use_colors: bool = True                  # ANSI colors in console


# ─────────────────────────────────────────────
#  SESSION AWARENESS
# ─────────────────────────────────────────────
@dataclass
class SessionConfig:
    """Session awareness configuration for position sizing by trading session."""
    asian_start: int = 0                     # UTC hour Asian session starts
    asian_end: int = 8                       # UTC hour Asian session ends
    london_start: int = 8                    # UTC hour London session starts
    london_end: int = 16                     # UTC hour London session ends
    ny_start: int = 13                       # UTC hour New York session starts
    ny_end: int = 21                         # UTC hour New York session ends
    # Position sizing multipliers per session
    asian_multiplier: float = 1.0            # Full size - trade 24/7 no restrictions
    london_multiplier: float = 1.2           # Peak liquidity, full size
    ny_multiplier: float = 1.0              # Standard size
    overlap_multiplier: float = 1.2          # London/NY overlap, peak volatility


# ─────────────────────────────────────────────
#  SPREAD FILTER
# ─────────────────────────────────────────────
@dataclass
class SpreadFilterConfig:
    """Spread filter configuration to avoid trading in wide-spread conditions."""
    max_spread_multiplier: float = 2.0       # Block if spread > 2x average
    avg_spread_window: int = 20              # Number of bars for average spread


# ─────────────────────────────────────────────
#  WIN/LOSE STREAK DETECTION
# ─────────────────────────────────────────────
@dataclass
class StreakConfig:
    """Streak-based position sizing adjustment configuration."""
    lose_streak_reduce_threshold: int = 3    # 3 consecutive losses -> reduce
    severe_threshold: int = 5                # 5 consecutive losses -> severe reduce
    reduce_pct: float = 0.5                  # Reduce lot to 50% after 3 losses
    severe_reduce_pct: float = 0.25          # Reduce lot to 25% after 5 losses
    win_restore_threshold: int = 2           # 2 consecutive wins -> restore to 1.0x
    win_boost_threshold: int = 3             # 3 consecutive wins -> boost
    win_severe_threshold: int = 5            # 5 consecutive wins -> severe boost
    win_boost_pct: float = 1.25             # 1.25x lot after 3 wins
    win_severe_boost_pct: float = 1.5       # 1.5x lot after 5 wins


# ─────────────────────────────────────────────
#  ADAPTIVE MOMENTUM
# ─────────────────────────────────────────────
@dataclass
class AdaptiveMomentumConfig:
    """Adaptive momentum lookback configuration based on ATR."""
    high_atr_lookback: int = 3               # Short lookback when ATR is high
    low_atr_lookback: int = 7                # Long lookback when ATR is low
    atr_threshold_mult: float = 1.5          # ATR > 1.5x avg = high volatility
    atr_avg_period: int = 14                 # Period for average ATR calculation


# ─────────────────────────────────────────────
#  PRICE ACTION STRUCTURE
# ─────────────────────────────────────────────
@dataclass
class PriceStructureConfig:
    """Price action structure detection configuration."""
    swing_lookback: int = 20                 # Bars to analyze for structure
    confidence_penalty: float = 0.05         # Small penalty when momentum opposes structure


# ─────────────────────────────────────────────
#  FVG (FAIR VALUE GAP) DETECTION
# ─────────────────────────────────────────────
@dataclass
class FVGConfig:
    """Fair Value Gap detection configuration."""
    enabled: bool = True                     # Enable FVG detection
    confidence_boost: float = 0.05           # Confidence boost when FVG aligns


# ─────────────────────────────────────────────
#  LIQUIDITY SWEEP DETECTION
# ─────────────────────────────────────────────
@dataclass
class LiquiditySweepConfig:
    """Liquidity sweep (stop hunt) detection configuration."""
    lookback: int = 20                       # Bars to look back for swing levels
    min_recovery_pct: float = 0.5            # Min % recovery to confirm sweep
    confidence_boost: float = 0.10           # Confidence boost when sweep aligns


# ─────────────────────────────────────────────
#  AUTO-OPTIMIZER
# ─────────────────────────────────────────────
@dataclass
class AutoOptimizerConfig:
    """Auto-optimizer configuration for self-tuning parameter optimization.

    The auto-optimizer analyzes live trade results and gradually shifts
    trading parameters toward optimal values. It records every trade with
    full context, groups by parameter value, and shifts 10-20% toward
    the best-performing value each cycle.
    """
    enabled: bool = True
    optimize_frequency: int = 5             # Trades between optimization cycles (fast learning)
    min_trades_before_tuning: int = 5       # Min trades before first optimization
    shift_rate: float = 0.15                # 15% shift toward optimal per cycle
    rollback_threshold: float = 0.20        # Rollback if 20% worse after optimization
    state_file: str = "auto_optimizer_state.json"

    # Parameter ranges
    sl_range: tuple = (3.0, 10.0)           # SL distance in dollars (wider range)
    session_mult_range: tuple = (0.3, 1.5)  # Session multiplier range
    confidence_range: tuple = (0.10, 0.50)  # Min confidence threshold range
    momentum_range: tuple = (5, 10)         # Momentum lookback bars (wider range)
    rsi_ob_range: tuple = (65, 85)          # RSI overbought level range
    rsi_os_range: tuple = (15, 35)          # RSI oversold level range
    cooldown_range: tuple = (2, 120)       # Cooldown seconds range (minimum wait between trades)
    max_positions_range: tuple = (1, 3)     # Max concurrent positions range (conservative)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
@dataclass
class MainConfig:
    """Main loop configuration."""
    interval_seconds: int = 10              # Ultra-fast 10-second cycle for HF scalping
    log_level: str = "INFO"
    enable_sentiment: bool = True
    enable_alternative_data: bool = True
    enable_regime_detection: bool = True
    enable_multi_timeframe: bool = True     # Multi-timeframe analysis
    enable_news_filter: bool = False        # Disabled - trade 24/7 no restrictions
    enable_multi_pair: bool = True          # Cross-pair correlation analysis
    enable_smart_exits: bool = True         # AI-driven exit management (RL agent)
    enable_auto_retrain: bool = True        # Weekend auto-retraining scheduler
    enable_dashboard: bool = True           # Live performance dashboard
    enable_auto_optimizer: bool = True      # Self-tuning parameter optimizer
    paper_trading: bool = True              # Paper trading mode by default
