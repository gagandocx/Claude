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

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


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
    enable_regime_detection: bool = True
    enable_multi_timeframe: bool = True     # Multi-timeframe analysis
    enable_news_filter: bool = False        # Disabled - trade 24/7 no restrictions
    enable_multi_pair: bool = True          # Cross-pair correlation analysis
    enable_dashboard: bool = True           # Live performance dashboard
    enable_auto_optimizer: bool = True      # Self-tuning parameter optimizer
    paper_trading: bool = True              # Paper trading mode by default
