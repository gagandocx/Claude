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
    input_features: int = 64        # Number of input features
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
    input_features: int = 64        # Number of input features
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
    min_agreement: float = 0.6      # Minimum model agreement for signal
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
    min_confidence: float = 0.65            # Minimum confidence to generate signal
    strong_confidence: float = 0.80         # Strong signal threshold
    atr_sl_multiplier: float = 1.5          # SL = ATR * multiplier
    atr_tp_multiplier: float = 2.5          # TP = ATR * multiplier (1:1.67 R:R)
    max_signal_age_seconds: int = 300       # Signal expires after 5 minutes
    cooldown_seconds: int = 60              # Minimum time between signals


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
    no_trade_hours: List[int] = field(default_factory=lambda: [0, 23])
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
#  MAIN LOOP
# ─────────────────────────────────────────────
@dataclass
class MainConfig:
    """Main loop configuration."""
    interval_seconds: int = 60              # 1-minute cycle for M1
    log_level: str = "INFO"
    enable_sentiment: bool = True
    enable_alternative_data: bool = True
    enable_regime_detection: bool = True
    paper_trading: bool = True              # Paper trading mode by default
