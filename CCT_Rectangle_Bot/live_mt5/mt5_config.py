"""
Configuration for CCT Rectangle Bot - Live MT5 Trading.

CONSERVATIVE defaults for live trading. Do NOT use backtest settings here.
Adjust settings based on your account size and risk tolerance.

Credentials are loaded from environment variables first, with fallback to
the values defined below. Set these environment variables for security:
    MT5_ACCOUNT_NUMBER
    MT5_PASSWORD
    MT5_SERVER_NAME
    MT5_TERMINAL_PATH
"""

import os

# =============================================================================
# MT5 CONNECTION CREDENTIALS
# Prefer environment variables to avoid committing real credentials.
# =============================================================================

MT5_ACCOUNT = int(os.environ.get("MT5_ACCOUNT_NUMBER", "12345678"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "your_password")
MT5_SERVER = os.environ.get("MT5_SERVER_NAME", "YourBroker-Demo")
MT5_PATH = os.environ.get("MT5_TERMINAL_PATH", "")

# =============================================================================
# SESSION MODE
# =============================================================================

USE_EXISTING_SESSION = True  # True = attach to already-running MT5 terminal (no login call)
                             # False = full login with credentials (for automated/headless startup)

# =============================================================================
# TRADING MODE
# =============================================================================

DEMO_MODE = True  # ALWAYS start with True. Only set False after thorough demo testing.

# =============================================================================
# SYMBOL & TIMEFRAME CONFIGURATION
# =============================================================================

SYMBOL = "XAUUSD"         # Gold on MT5 (not GC=F which is yfinance format)
LOT_SIZE = 0.01           # Micro lot - conservative starting size
MAGIC_NUMBER = 20240101   # Unique identifier for this bot's orders

# Timeframes (MT5 timeframe constants will be used in mt5_data_feed.py)
TF_DIRECTION = "4h"
TF_WEAKNESS = "15m"
TF_ENTRY = "1m"

# Number of bars to fetch for each timeframe
BARS_4H = 200
BARS_15M = 500
BARS_1M = 500

# =============================================================================
# RISK MANAGEMENT - CONSERVATIVE FOR LIVE TRADING
# =============================================================================

RISK_PER_TRADE = 0.01     # 1% risk per trade (NOT 25% like backtest!)
INITIAL_CAPITAL = None     # Will use actual account balance from MT5

# RR settings (same as strategy)
MIN_RR_RATIO = 3.0
TARGET_RR_RATIO = 3.0
MAX_RR_RATIO = 12.0

# =============================================================================
# SAFETY LIMITS
# =============================================================================

MAX_DAILY_LOSS_PCT = 0.03       # Stop trading if daily loss exceeds 3%
MAX_TRADES_PER_DAY = 5          # Maximum trades allowed per day
MAX_DRAWDOWN_PCT = 0.05         # Stop trading if drawdown exceeds 5% from peak
MAX_CONCURRENT_TRADES = 1       # Only 1 position at a time for live

# =============================================================================
# TRAILING STOP
# =============================================================================

USE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_RR = 3.0   # Activate trailing stop after 3R profit
TRAILING_STOP_DISTANCE_RR = 2.5     # Trail at 2.5R distance

# =============================================================================
# TRADING HOURS (UTC) - Only trade during active sessions
# =============================================================================

TRADING_SESSIONS = {
    "london": {"start": 7, "end": 16},
    "new_york": {"start": 12, "end": 21},
}

# Set to True to also allow trading during Asia session
ALLOW_ASIA_SESSION = False
ASIA_SESSION = {"start": 0, "end": 8}

# =============================================================================
# POLLING & TIMING - SMART HYBRID POLLING
# =============================================================================

# Adaptive polling: IDLE = 1s (low CPU), ARMED = 100ms (fast reaction)
POLL_INTERVAL_IDLE_MS = 1000    # 1 second between checks when IDLE (low CPU)
POLL_INTERVAL_ARMED_MS = 100    # 100ms between checks when ARMED (fast response)
DISPLAY_UPDATE_MS = 1000        # Refresh display every 1 second
RECONNECT_MAX_RETRIES = 5       # Max reconnection attempts
RECONNECT_BASE_DELAY = 2        # Base delay for exponential backoff (seconds)
RECONNECT_MAX_DELAY = 60        # Maximum delay between reconnection attempts

# =============================================================================
# TICK MONITORING & REAL-TIME SETTINGS - SMART HYBRID
# =============================================================================

TICK_MONITORING_ENABLED = True         # Enable real-time tick monitoring
PRE_COMPUTE_SIGNALS = True             # Pre-compute direction+weakness between candle closes
MAX_EXECUTION_DELAY_MS = 200           # Target: signal-to-execution under 200ms (relaxed for hybrid)
TICK_STALE_THRESHOLD_SECONDS = 5       # Consider tick stale if older than this
PRE_STAGE_ORDERS = True                # Pre-build order request when ARMED for instant fire

# Pre-computation cache durations (seconds)
DIRECTION_CACHE_SECONDS = 14400        # 4H direction signal rarely changes (4 hours)
WEAKNESS_CACHE_SECONDS = 900           # 15M weakness changes every 15 minutes

# =============================================================================
# SPREAD & SLIPPAGE
# =============================================================================

MAX_SPREAD_POINTS = 50          # Max acceptable spread in points (skip trade if exceeded)
SLIPPAGE_POINTS = 20            # Maximum allowed slippage for order execution

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_DIR = "logs"
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per log file
LOG_BACKUP_COUNT = 5                # Keep 5 rotated log files
