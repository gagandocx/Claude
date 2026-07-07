"""
Configuration for CCT Rectangle Trading Bot.
All strategy parameters, timeframes, session times, and risk settings.
"""

# =============================================================================
# SYMBOL & TIMEFRAME CONFIGURATION
# =============================================================================

# Trading symbol (yfinance format)
SYMBOL = "GC=F"  # Gold Futures (high liquidity, good volatility for CCT)

# Timeframes
TF_DIRECTION = "4h"       # Direction detection timeframe
TF_WEAKNESS = "15m"       # Weakness/rectangle detection timeframe
TF_ENTRY = "1m"           # Entry trigger timeframe

# Data fetch periods (limited by yfinance)
# 1m data: last 7 days only
# 15m data: last 60 days
# For backtesting we use 15m data resampled to 4h for longer history
DATA_PERIOD_15M = "60d"
DATA_PERIOD_1M = "7d"

# =============================================================================
# EMA CONFIGURATION (Foundation Rules)
# =============================================================================

EMA_FAST = 50              # Fast EMA for directional bias
EMA_SLOW = 200             # Slow EMA for strong trend confirmation
USE_EMA_FILTER = True      # Enable/disable EMA directional filter

# =============================================================================
# DIRECTION CANDLE PARAMETERS (4H)
# =============================================================================

# CCT Direction candle: Must sweep previous candle's high/low
# Bullish: sweeps prev low (wick below) AND closes above prev high
# Bearish: sweeps prev high (wick above) AND closes below prev low
DIRECTION_SWEEP_TOLERANCE = 0.0  # Tolerance for sweep detection (in price)

# =============================================================================
# WEAKNESS DETECTION PARAMETERS (15M)
# =============================================================================

# Swing detection lookback
SWING_LOOKBACK = 5         # Number of candles to look back for swing highs/lows

# Weakness = wick sweeps level but candle closes back inside
# How far beyond the level the wick must go to qualify as a sweep
SWEEP_MIN_PIPS = 0.50      # Minimum sweep distance (in price units, $0.50 for gold)

# Maximum candles to wait for weakness after direction signal
MAX_CANDLES_FOR_WEAKNESS = 16  # 16 x 15m = 4 hours (one full 4H candle period)

# FVG (Fair Value Gap) detection
FVG_MIN_SIZE_PIPS = 1.00   # Minimum FVG size to be considered valid ($1 for gold)

# =============================================================================
# RECTANGLE ENTRY PARAMETERS (1M)
# =============================================================================

# Rectangle is drawn from M15 trigger candle close to its extreme (wick tip)
# Entry: 1M candle closes outside the rectangle
# Stop loss: Beyond the rectangle extreme

# Maximum candles to wait for 1M entry after rectangle is drawn
MAX_CANDLES_FOR_ENTRY = 30   # 30 x 1m = 30 minutes

# Minimum rectangle size (prevents tiny rectangles with bad RR)
MIN_RECTANGLE_SIZE_PIPS = 0.50  # Minimum $0.50 rectangle for gold

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

RISK_PER_TRADE = 0.01      # Risk 1% of capital per trade
INITIAL_CAPITAL = 10000.0  # Starting capital for backtest
MIN_RR_RATIO = 3.0         # Minimum reward:risk ratio (3:1)
TARGET_RR_RATIO = 3.0      # Default target RR ratio
MAX_RR_RATIO = 5.0         # Maximum RR (clip TP at this level)

# =============================================================================
# SESSION TIMES (UTC)
# =============================================================================

# Session definitions in UTC hours
SESSIONS = {
    "asia": {"start": 0, "end": 8},       # 00:00 - 08:00 UTC
    "london": {"start": 7, "end": 16},     # 07:00 - 16:00 UTC
    "new_york": {"start": 12, "end": 21},  # 12:00 - 21:00 UTC
}

# =============================================================================
# TRADE FILTERS
# =============================================================================

# Only take trades where M15 weakness level sits inside FVG or at session extreme
REQUIRE_IMBALANCE_FILTER = False  # Set True for highest probability only
REQUIRE_SESSION_EXTREME = False   # Set True to only trade at session H/L

# Continuation only: trade exclusively with the trend
CONTINUATION_ONLY = True

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================

# Commission/spread in price units (approximate spread for Gold futures)
SPREAD = 0.30               # $0.30 spread for gold
COMMISSION_PER_TRADE = 0.0  # No additional commission

# Position sizing
LOT_SIZE = 100              # 1 lot = 100 oz for gold futures

# Pip multiplier: for gold 1 pip = $0.10, so multiply by 10 to get pips
# For forex pairs it would be 10000
PIP_MULTIPLIER = 10         # Multiply price difference by this to get pips
