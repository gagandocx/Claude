"""
Configuration for CCT Rectangle Trading Bot.
AGGRESSIVE MODE: Maximum compounding with quality signal detection.

Strategy: Keep original high-quality CCT detection (66%+ win rate)
while maximizing trade opportunities and using ultra-aggressive sizing.

With 60%+ win rate at 3:1 RR and 20% risk per trade compounding:
- Each win adds +60% to equity (3:1 * 20%)
- Each loss costs -20% of equity
- 10 wins and 5 losses compounded ~ 500%+ returns
"""

# =============================================================================
# SYMBOL & TIMEFRAME CONFIGURATION
# =============================================================================

SYMBOL = "GC=F"  # Gold Futures

# Timeframes
TF_DIRECTION = "4h"
TF_WEAKNESS = "15m"
TF_ENTRY = "1m"

# Data fetch periods
DATA_PERIOD_15M = "60d"
DATA_PERIOD_1M = "7d"

# =============================================================================
# EMA CONFIGURATION
# =============================================================================

EMA_FAST = 50
EMA_SLOW = 200
USE_EMA_FILTER = True       # KEEP ENABLED for quality

# =============================================================================
# DIRECTION CANDLE PARAMETERS (4H)
# =============================================================================

# Allow partial engulfing to find more direction signals while keeping quality
DIRECTION_SWEEP_TOLERANCE = 0.0
REQUIRE_FULL_ENGULF = True       # STRICT: full engulfing for quality
PARTIAL_ENGULF_RATIO = 0.65      # Not used when full engulf required

# =============================================================================
# WEAKNESS DETECTION PARAMETERS (15M)
# =============================================================================

SWING_LOOKBACK = 3               # More swing points
SWEEP_MIN_PIPS = 0.10            # Catch smaller sweeps
MAX_CANDLES_FOR_WEAKNESS = 64    # 16 hours window
WEAKNESS_WINDOW_HOURS = 24       # Full day lookforward
FVG_MIN_SIZE_PIPS = 0.50
MAX_WEAKNESS_PER_DIRECTION = 3   # Up to 3 weakness signals

# =============================================================================
# RECTANGLE ENTRY PARAMETERS (1M)
# =============================================================================

MAX_CANDLES_FOR_ENTRY = 120      # 2 hour window
MIN_RECTANGLE_SIZE_PIPS = 0.15   # Small rectangles OK

# =============================================================================
# RISK MANAGEMENT - ULTRA AGGRESSIVE
# =============================================================================

RISK_PER_TRADE = 0.25      # 25% risk per trade
INITIAL_CAPITAL = 10000.0
MIN_RR_RATIO = 3.0         # Keep 3:1 for quality
TARGET_RR_RATIO = 3.0
MAX_RR_RATIO = 12.0        # Allow very big runners with trailing stop

# Compounding
COMPOUNDING = True
LEVERAGE = 1

# Trailing stop - lets winners run far  
USE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_RR = 3.0  # Activate after 3R profit
TRAILING_STOP_DISTANCE_RR = 2.5    # Ultra wide trail (2.5R) for max trend capture

# Concurrent positions
MAX_CONCURRENT_TRADES = 2  # Conservative concurrent to protect capital

# =============================================================================
# SESSION TIMES (UTC)
# =============================================================================

SESSIONS = {
    "asia": {"start": 0, "end": 8},
    "london": {"start": 7, "end": 16},
    "new_york": {"start": 12, "end": 21},
}

# =============================================================================
# TRADE FILTERS
# =============================================================================

REQUIRE_IMBALANCE_FILTER = False
REQUIRE_SESSION_EXTREME = False
CONTINUATION_ONLY = True   # Trade with trend for quality

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================

SPREAD = 0.20
COMMISSION_PER_TRADE = 0.0
LOT_SIZE = 100
PIP_MULTIPLIER = 10
