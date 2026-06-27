"""
=============================================================
  Python ML Bridge - Configuration Settings
  Central configuration for all model parameters, data sources,
  signal thresholds, and file paths for MT5 bridge communication.
=============================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


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

# Tick data file that MT5 writes and Python reads (order flow features)
TICK_DATA_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_tick_data.csv")

# Spread file that MT5 writes and Python reads (spread gating)
SPREAD_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_spread.csv")


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
class ChronosConfig:
    """Chronos — Amazon pre-trained T5 foundation model (2024)."""
    model_name: str     = "amazon/chronos-t5-tiny"  # 8 M params, fast on CPU
    input_features: int = 46
    seq_length: int     = 64
    d_model: int        = 64          # classification head hidden dim
    chronos_d_model: int = 512        # T5-tiny encoder hidden size (auto-detected at runtime)
    num_classes: int    = 3
    dropout: float      = 0.1
    learning_rate: float = 1e-4       # only the head trains; encoder is frozen
    weight_decay: float  = 1e-5
    batch_size: int      = 32
    batch_size_gpu: int  = 256
    epochs: int          = 60         # fewer epochs — head trains quickly
    patience: int        = 10
    label_smoothing: float = 0.1


@dataclass
class TimeMixerConfig:
    """TimeMixer — decomposable multiscale mixing (Wang et al., ICLR 2024)."""
    input_features: int  = 46
    seq_length: int      = 64
    hidden_size: int     = 256
    pool_sizes: list     = field(default_factory=lambda: [1, 2, 4, 8])
    decomp_kernel: int   = 25         # moving-average decomposition kernel
    num_classes: int     = 3
    dropout: float       = 0.1
    learning_rate: float = 1e-4
    weight_decay: float  = 1e-5
    batch_size: int      = 32
    batch_size_gpu: int  = 256
    epochs: int          = 100
    patience: int        = 15
    label_smoothing: float = 0.1


@dataclass
class SOFTSConfig:
    """SOFTS — Star aggregate O(N) series-core fusion (Han et al., NeurIPS 2024)."""
    input_features: int  = 46
    seq_length: int      = 64
    d_model: int         = 64
    n_layers: int        = 3          # number of STAR blocks
    num_classes: int     = 3
    dropout: float       = 0.1
    learning_rate: float = 1e-4
    weight_decay: float  = 1e-5
    batch_size: int      = 32
    batch_size_gpu: int  = 256
    epochs: int          = 100
    patience: int        = 15
    label_smoothing: float = 0.1


# ─────────────────────────────────────────────
#  PLATT SCALING CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────
# Anchor state file path to the neurox_v4 root directory
_NEUROX_V4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PlattCalibrationConfig:
    """Platt scaling confidence calibration configuration."""
    window_size: int = 200                   # Sliding window of (logit, outcome) pairs
    state_file: str = os.path.join(_NEUROX_V4_ROOT, "platt_calibration_state.json")  # Absolute path
    min_samples: int = 50                    # Minimum samples before calibration is active


# ─────────────────────────────────────────────
#  SMART ENTRY TIMING (MICRO-PULLBACK)
# ─────────────────────────────────────────────
@dataclass
class EntryTimingConfig:
    """Smart entry timing (micro-pullback) configuration."""
    enabled: bool = True                     # Enable/disable entry timing
    pullback_points: float = 0.30            # Pullback target in dollars (for gold)
    timeout_seconds: float = 10              # Max wait time before entering at market
    breakout_threshold_points: float = 1.0   # Enter immediately if price breaks away by this much
    adaptive_timeout: bool = True            # Adapt timeout based on ATR (volatility)
    adaptive_timeout_min: float = 3.0        # Minimum adaptive timeout (seconds)
    adaptive_timeout_max: float = 30.0       # Maximum adaptive timeout (seconds)


# ─────────────────────────────────────────────
#  SHARPE-RATIO-BASED MODEL WEIGHTS
# ─────────────────────────────────────────────
@dataclass
class SharpeWeightConfig:
    """Sharpe-ratio-based model weighting configuration."""
    enabled: bool = True                     # Enable Sharpe-based reweighting
    lookback_trades: int = 100               # Rolling window of trades per model
    min_trades_per_model: int = 20           # Min trades before Sharpe is computed for a model
    reweight_interval: int = 20              # Recompute weights every N trades
    min_weight_floor: float = 0.02           # Minimum weight per model (prevents zero allocation)


@dataclass
class EnsembleConfig:
    """Ensemble model configuration — 17-model stack."""
    # Weights must sum to 1.0
    transformer_weight: float  = 0.06   # Global self-attention
    lstm_weight: float         = 0.05   # BiLSTM + attention
    tcn_weight: float          = 0.05   # Dilated temporal conv
    patch_tst_weight: float    = 0.08   # Patch-based SOTA 2023
    tft_weight: float          = 0.08   # Financial VSN+GRN
    nhits_weight: float        = 0.05   # Hierarchical macro→micro
    itransformer_weight: float = 0.08   # Feature-space attention 2024
    mamba_weight: float        = 0.07   # Selective state space 2023
    dlinear_weight: float      = 0.03   # Trend/residual decomposition
    xlstm_weight: float        = 0.08   # Matrix memory LSTM 2024
    timesnet_weight: float     = 0.06   # 2D temporal variation 2023
    chronos_weight: float      = 0.09   # Pre-trained foundation model (NEW)
    timemixer_weight: float    = 0.07   # Multi-scale decomp mixing (NEW)
    softs_weight: float        = 0.05   # Star aggregate O(N) (NEW)
    gradient_boost_weight: float = 0.04 # sklearn HistGradBoost
    xgboost_weight: float      = 0.03   # LightGBM / XGBoost
    catboost_weight: float     = 0.03   # Ordered boosting
    meta_learner_features: int = 51     # 17 models x 3 classes
    min_agreement: float       = 0.10
    dynamic_weights: bool      = True
    weight_lookback: int       = 50
    # Online learning: lightweight gradient updates between weekly retrains
    online_learning_enabled: bool = True
    online_batch_size: int = 10          # Min labeled samples before update
    online_lr: float = 1e-4              # Learning rate for online head updates
    # Meta-learner data accumulation for auto-retrain
    accumulate_meta_data: bool = True
    meta_data_min_samples: int = 500     # Min samples before auto-retrain


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
    model_override_threshold: float = 0.60  # Models override momentum when prob > this


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
#  TRADING BRAIN
# ─────────────────────────────────────────────
@dataclass
class BrainConfig:
    """
    Trading Brain configuration — professional trader / hedge fund intelligence.
    Controls all 6 evaluation layers that filter and size every trade.
    """
    enabled: bool = True

    # ── Account ──────────────────────────────────────────────────────────
    account_balance: float = 10000.0
    base_lot:        float = 0.01
    min_lot:         float = 0.01
    max_lot:         float = 0.05

    # ── Edge tracker ─────────────────────────────────────────────────────
    lookback_trades: int   = 30          # trades to analyse for edge quality
    edge_hot_threshold:    float = 0.62  # win rate above this → HOT → size up
    edge_cold_threshold:   float = 0.42  # win rate below this → COLD → size down
    edge_broken_threshold: float = 0.30  # win rate below this → BROKEN → pause

    # ── Probe trade (BROKEN edge recovery) ────────────────────────────────
    probe_interval_seconds:    int = 300   # allow 1 probe trade every N seconds
    probe_min_broken_seconds:  int = 300   # edge must be BROKEN for N seconds before probing

    # ── Position sizing multipliers ───────────────────────────────────────
    lot_multiplier_hot:      float = 1.5   # edge is HOT
    lot_multiplier_cold:     float = 0.5   # edge is COLD
    lot_multiplier_volatile: float = 0.4   # volatile/crash regime

    # ── Risk circuit breakers ─────────────────────────────────────────────
    daily_loss_limit:         float = 50.0  # USD — stop for the day
    drawdown_reduce_threshold: float = 0.08  # 8%  drawdown → reduce to 50%
    drawdown_stop_threshold:   float = 0.15  # 15% drawdown → full stop
    consecutive_loss_reduce:   int   = 3     # reduce after 3 losses in a row
    consecutive_loss_stop:     int   = 6     # pause after 6 losses in a row

    # ── Equity curve filter ───────────────────────────────────────────────
    equity_curve_filter:  bool  = True   # pause if equity curve below MA
    equity_curve_ma_bars: int   = 10     # MA period for equity curve filter

    # ── Session multipliers ───────────────────────────────────────────────
    session_overlap_mult:    float = 1.3  # London / NY overlap  (peak)
    session_london_mult:     float = 1.2  # London session
    session_ny_mult:         float = 1.0  # New York session
    session_asian_mult:      float = 0.8  # Asian session
    session_off_hours_mult:  float = 0.6  # Off hours / late NY

    # ── Dynamic confidence threshold ──────────────────────────────────────
    base_min_confidence: float = 0.25  # base threshold (same as SignalConfig)

    # ── Logging ───────────────────────────────────────────────────────────
    status_report_interval: int = 50   # print brain status every N cycles

    # ── Market quality ────────────────────────────────────────────────────
    max_spread_atr_ratio: float = 0.35  # skip if spread > 35% of ATR
    min_tick_volume:      float = 10.0  # skip if volume suspiciously low
    min_atr_points:       float = 0.5   # skip if ATR near zero (stalled)

    # ── SL / TP limits ────────────────────────────────────────────────────
    min_sl_dollars: float = 0.80   # never set SL tighter than $0.80
    max_sl_dollars: float = 5.00   # never set SL wider than $5.00

    # ── VaR-based sizing ──────────────────────────────────────────────────
    risk_per_trade_pct: float = 0.005   # risk 0.5% of account per trade

    # ── Trade quality gate ────────────────────────────────────────────────
    min_trade_score: float = 30.0  # 0-100; below this → skip regardless
    min_win_probability: float = 0.52  # Bayesian posterior minimum to trade


# ─────────────────────────────────────────────
#  TICK DATA / ORDER FLOW (Tier 1)
# ─────────────────────────────────────────────
@dataclass
class TickDataConfig:
    """Tick data reader for order flow features (Tier 1).

    MT5 EA writes real-time tick data (bid, ask, volume, flags) to a CSV
    in Common Files. Python reads this to compute institutional order flow
    features: bid/ask imbalance, volume delta, and trade flow direction.
    """
    tick_file: str = TICK_DATA_FILE          # Path to tick data CSV
    poll_interval_ms: int = 100              # How often to poll for new ticks (ms)
    max_ticks: int = 5000                    # Max ticks to keep in memory
    # Features to compute from tick stream
    features: List[str] = field(default_factory=lambda: [
        "bid_ask_imbalance", "volume_delta", "trade_flow"
    ])


# ─────────────────────────────────────────────
#  MICROSTRUCTURE FEATURES (Tier 2)
# ─────────────────────────────────────────────
@dataclass
class MicrostructureConfig:
    """Microstructure features from tick stream (Tier 2).

    Computes institutional-grade microstructure signals: tick arrival rate,
    bid-ask bounce rate (how often price bounces off bid vs ask), and
    large-order detection from abnormal tick volume spikes.
    """
    tick_rate_window: int = 60               # Seconds window for tick rate calc
    large_order_threshold: float = 3.0       # Std devs above mean for large order
    bounce_rate_window: int = 30             # Ticks window for bounce rate calc


# ─────────────────────────────────────────────
#  REGIME-SPECIFIC MODEL ROUTING (Tier 1)
# ─────────────────────────────────────────────
@dataclass
class RegimeRoutingConfig:
    """Regime-specific model routing (Tier 1).

    Routes predictions to different model subsets based on detected market
    regime. Trending markets favor momentum models (TCN, LSTM), ranging
    markets favor mean-reversion models (DLinear, TimesNet), and volatile
    markets favor robust ensemble (Mamba, iTransformer).
    """
    trending_models: List[str] = field(default_factory=lambda: [
        "tcn", "lstm", "patch_tst", "tft", "mamba"
    ])
    ranging_models: List[str] = field(default_factory=lambda: [
        "dlinear", "timesnet", "nhits", "softs", "xlstm"
    ])
    volatile_models: List[str] = field(default_factory=lambda: [
        "mamba", "itransformer", "transformer", "chronos", "timemixer"
    ])
    routing_lookback: int = 50               # Bars to assess regime for routing


# ─────────────────────────────────────────────
#  WALK-FORWARD RETRAINING (Tier 1)
# ─────────────────────────────────────────────
@dataclass
class WalkForwardConfig:
    """Walk-forward retraining pipeline (Tier 1).

    Automated weekly retraining: saves training data, retrains models on
    latest market data, validates improvement via walk-forward before
    deploying new weights. Prevents overfitting and adapts to regime shifts.
    """
    retrain_interval_hours: int = 168        # 168h = 1 week
    validation_window_bars: int = 500        # Bars for walk-forward validation
    min_improvement_pct: float = 2.0         # Must improve >2% to deploy
    max_retrain_duration_min: int = 30       # Max minutes per retrain cycle
    data_save_path: str = "training_data/"   # Where to save training snapshots


# ─────────────────────────────────────────────
#  ADVERSARIAL SIGNAL FILTER (Tier 1)
# ─────────────────────────────────────────────
@dataclass
class AdversarialFilterConfig:
    """Adversarial signal filtering (Tier 1).

    Before trading, checks if recent similar signals (same direction,
    similar time, similar price level) won or lost. If last N similar
    signals had a high loss rate, skips the current signal. Prevents
    repeated losses in adverse conditions.
    """
    lookback_signals: int = 50               # How many past signals to compare
    similarity_threshold: float = 0.85       # Cosine similarity threshold
    min_similar_signals: int = 3             # Min similar signals to evaluate
    loss_rate_threshold: float = 0.6         # Skip if >60% of similar signals lost


# ─────────────────────────────────────────────
#  SPREAD/SLIPPAGE GATE (Tier 2)
# ─────────────────────────────────────────────
@dataclass
class SpreadGateConfig:
    """Spread/slippage-aware entry gate (Tier 2).

    EA writes current spread to a CSV file. Python reads it and only
    generates entry signals when spread is below a dynamic threshold
    (multiple of recent average spread). Prevents entries during
    high-spread periods (news, low liquidity).
    """
    spread_file: str = SPREAD_FILE           # Path to spread CSV from EA
    max_spread_multiplier: float = 1.5       # Enter only if spread < 1.5x average
    max_absolute_spread: float = 80.0        # Max absolute spread in points (gold typical max)
    update_interval_ms: int = 500            # How often EA updates spread file


# ─────────────────────────────────────────────
#  CORRELATION REGIME DETECTION (Tier 2)
# ─────────────────────────────────────────────
@dataclass
class CorrelationRegimeConfig:
    """Cross-market correlation regime detection (Tier 2).

    Monitors DXY/bonds/equities correlation state to detect regime
    changes that affect gold direction. When correlations break down
    (e.g., gold moves with USD instead of against), signals caution.
    """
    regime_lookback: int = 100               # Bars for correlation computation
    dxy_correlation_threshold: float = -0.5  # Expected XAUUSD-DXY correlation
    bond_correlation_threshold: float = 0.3  # Expected XAUUSD-Bond correlation
    rebalance_interval: int = 60             # Bars between regime reassessment


# ─────────────────────────────────────────────
#  ADAPTIVE CONFIDENCE THRESHOLD (Tier 2)
# ─────────────────────────────────────────────
@dataclass
class AdaptiveThresholdConfig:
    """Adaptive confidence threshold (Tier 2).

    Dynamically raises or lowers the model override threshold based on
    recent accuracy. When models are accurate, threshold drops (trade more).
    When models are inaccurate, threshold rises (trade less). Not a fixed
    value like the base 0.60.
    """
    lookback_trades: int = 30                # Trades to evaluate accuracy
    adjustment_rate: float = 0.02            # How much to shift per evaluation
    min_threshold: float = 0.15              # Floor for confidence threshold
    max_threshold: float = 0.55              # Ceiling for confidence threshold
    target_win_rate: float = 0.55            # Target win rate to maintain


# ─────────────────────────────────────────────
#  MULTI-MODEL DISAGREEMENT SIGNAL (Tier 3)
# ─────────────────────────────────────────────
@dataclass
class DisagreementConfig:
    """Multi-model disagreement as volatility signal (Tier 3).

    When models strongly disagree on direction, it predicts an upcoming
    volatility spike. Used to reduce position size or delay entry until
    models converge. Strong disagreement = uncertainty = smaller size.
    """
    strong_disagreement_threshold: float = 0.4   # Std dev of model outputs above this = disagreement
    volatility_scale_factor: float = 1.5         # Scale expected volatility by this during disagreement
    position_reduction_pct: float = 0.5          # Reduce position to 50% during disagreement


# ─────────────────────────────────────────────
#  KELLY CRITERION POSITION SIZING (Tier 3)
# ─────────────────────────────────────────────
@dataclass
class KellyConfig:
    """Kelly criterion position sizing (Tier 3).

    Full Kelly with fractional Kelly safety. Uses Brain's win rate and
    average win/loss ratio to compute mathematically optimal lot size.
    Fractional Kelly (25%) provides geometric growth with reduced variance.
    """
    full_kelly: bool = False                 # Use full Kelly (dangerous) or fractional
    kelly_fraction: float = 0.25             # 25% Kelly = safe geometric growth
    min_win_rate: float = 0.50               # Don't trade if win rate below this
    min_trades: int = 20                     # Min trades before Kelly is reliable
    max_kelly_lot: float = 0.10              # Hard cap on Kelly-computed lot size


# ─────────────────────────────────────────────
#  MONTE CARLO RISK SIMULATION (Tier 3)
# ─────────────────────────────────────────────
@dataclass
class MonteCarloConfig:
    """Monte Carlo risk simulation (Tier 3).

    Before each trade, simulates N scenarios given current drawdown,
    win/loss streak, and regime. Gates trades that have >X% probability
    of hitting the daily loss limit. Institutional-grade risk management.
    """
    num_simulations: int = 1000              # Number of Monte Carlo paths
    max_daily_loss_prob: float = 0.05        # Gate if >5% chance of hitting daily loss
    drawdown_threshold: float = 0.10         # Current drawdown that triggers simulation
    confidence_level: float = 0.95           # VaR confidence level
    serial_correlation: float = 0.15         # Loss clustering autocorrelation factor


# ─────────────────────────────────────────────
#  DATA VALIDATION
# ─────────────────────────────────────────────
@dataclass
class DataValidatorConfig:
    """Live data validation configuration.

    Validates incoming market data to prevent garbage-in-garbage-out.
    Checks for NaN/inf, zero-volume bars, price gaps, price sanity,
    tick staleness, and feature schema compliance.
    """
    max_nan_pct: float = 0.05                # Max allowed NaN percentage (5%)
    max_gap_atr_mult: float = 3.0            # Gap > 3x ATR triggers warning
    min_price: float = 500.0                 # Min sane price for XAUUSD
    max_price: float = 5000.0                # Max sane price for XAUUSD
    staleness_seconds: int = 120             # Tick data stale after 2 minutes
    expected_feature_count: int = 46         # Expected features per sample
    enable_validation: bool = True           # Master enable/disable


# ─────────────────────────────────────────────
#  ACCOUNT BALANCE SYNC
# ─────────────────────────────────────────────
# Balance file that MT5 writes after each trade close
BALANCE_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_balance.csv")


@dataclass
class AccountSyncConfig:
    """Live account balance sync from MT5 confirmations.

    The EA writes current balance and equity to python_bridge_balance.csv
    after each trade close. Python reads this file to keep risk sizing
    accurate as the account grows/shrinks.
    """
    sync_from_confirmations: bool = True     # Enable balance sync from confirmations
    balance_file: str = BALANCE_FILE         # Path to balance CSV in MT5 Common Files
    sync_on_close: bool = True               # Sync when CLOSED confirmation arrives
    fallback_balance: float = 10000.0        # Default if file not readable


# ─────────────────────────────────────────────
#  SLIPPAGE TRACKER
# ─────────────────────────────────────────────
@dataclass
class SlippageTrackerConfig:
    """Slippage/execution quality tracking configuration.

    Tracks fill quality by comparing requested price vs actual fill price
    reported in FILLED confirmations. Detects degrading broker execution
    quality over time.
    """
    window_size: int = 100                   # Rolling window of fills
    degradation_multiplier: float = 2.0      # Alert if slippage > 2x average
    min_fills_for_scoring: int = 5           # Min fills before scoring is active
    quality_warning_threshold: float = 0.7   # Log warning below this quality
    max_expected_slippage: float = 1.0       # Max expected slippage ($ for XAUUSD)


# ─────────────────────────────────────────────
#  PIPELINE ARCHITECTURE (THREADING)
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Pipeline architecture configuration for overlapping I/O and compute.

    Uses ThreadPoolExecutor to fetch data in parallel with model inference,
    reducing total cycle time by overlapping I/O-bound and CPU-bound work.
    """
    max_workers: int = 3                     # Thread pool size (data + sentiment + alt)
    fetch_timeout_seconds: float = 30.0      # Timeout per fetch task
    enable_pipeline: bool = True             # Master enable/disable


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE MONITOR
# ─────────────────────────────────────────────
@dataclass
class FeatureMonitorConfig:
    """Feature importance monitoring configuration.

    Tracks per-feature importance using permutation importance approximation.
    Alerts when a feature's importance drops below a threshold of its
    historical average, indicating that the feature has degraded or its
    data source has become unreliable.
    """
    importance_window: int = 200             # Rolling window of importance scores
    check_interval: int = 50                 # Predictions between importance checks
    degradation_threshold: float = 0.1       # Alert if importance < 10% of historical avg
    enabled: bool = True                     # Master enable/disable


# ─────────────────────────────────────────────
#  CORRELATION RISK
# ─────────────────────────────────────────────
@dataclass
class CorrelationRiskConfig:
    """Cross-position correlation risk tracking.

    Future-proof placeholder for multi-instrument trading.
    Currently only XAUUSD is traded with max 1 position,
    so this is a lightweight check that logs and returns safe.
    """
    enabled: bool = True                     # Enable correlation checking
    max_correlated_positions: int = 2        # Max positions on correlated instruments
    correlation_threshold: float = 0.70      # Threshold for 'correlated' instruments
    # Known correlated pairs (for future multi-instrument)
    correlated_groups: Dict = field(default_factory=lambda: {
        'gold_group': ['XAUUSD', 'XAGUSD', 'GDX'],
        'usd_group': ['EURUSD', 'GBPUSD', 'USDJPY'],
    })


# ─────────────────────────────────────────────
#  A/B TESTING FRAMEWORK
# ─────────────────────────────────────────────
@dataclass
class ABTestConfig:
    """A/B testing framework for parameter comparison.

    Randomly assigns trades to variant A (current) or variant B (candidate).
    After min_trades_per_variant, computes statistical significance using
    a z-test on win rates. Logs which variant is winning.
    """
    enabled: bool = False                    # Disabled by default (enable to test params)
    min_trades_per_variant: int = 30         # Min trades before significance testing
    significance_level: float = 0.05         # p-value threshold for significance
    test_name: str = "default"               # Name of current A/B test
    variant_a_label: str = "current"         # Label for control variant
    variant_b_label: str = "candidate"       # Label for test variant


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
@dataclass
class MainConfig:
    """Main loop configuration."""
    interval_seconds: int = 2               # 2-second cycle — instant signal re-evaluation
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
    enable_brain: bool = True               # Trading Brain — professional judgment layer
    enable_platt_calibration: bool = True   # Live confidence calibration (Platt scaling)
    enable_entry_timing: bool = True        # Smart entry timing (micro-pullback)
    enable_sharpe_weights: bool = True      # Sharpe-ratio-based model weighting
    paper_trading: bool = True              # Paper trading mode by default

    # V3 Institutional Feature Flags (Tier 1-3)
    enable_tick_data: bool = True            # Tier 1: Order flow / tick data features
    enable_regime_routing: bool = True       # Tier 1: Regime-specific model routing
    enable_walk_forward: bool = True         # Tier 1: Walk-forward retraining pipeline
    enable_adversarial_filter: bool = True   # Tier 1: Adversarial signal filtering
    enable_spread_gate: bool = True          # Tier 2: Spread/slippage-aware entry gate
    enable_microstructure: bool = True       # Tier 2: Microstructure features from ticks
    enable_correlation_regime: bool = True   # Tier 2: Cross-market correlation regime
    enable_adaptive_threshold: bool = True   # Tier 2: Adaptive confidence threshold
    enable_disagreement_signal: bool = True  # Tier 3: Multi-model disagreement signal
    enable_kelly_sizing: bool = True         # Tier 3: Kelly criterion position sizing
    enable_monte_carlo_risk: bool = True     # Tier 3: Monte Carlo risk simulation

    # V7.5 Data Quality & Pipeline
    enable_data_validation: bool = True      # Live data validation/sanity checking
    enable_pipeline: bool = True             # Pipeline threading (overlap fetch + compute)
    enable_account_sync: bool = True         # Live account balance sync from MT5
    enable_slippage_tracker: bool = True     # Slippage/execution quality tracking
    enable_feature_monitor: bool = True      # Feature importance monitoring
    enable_online_learning: bool = True      # Online learning adaptation between retrains
    enable_ab_testing: bool = False          # A/B testing framework for parameter comparison
    enable_equity_curve_trading: bool = True  # Equity curve meta-strategy for lot sizing
