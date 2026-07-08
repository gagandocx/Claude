"""
Trading Profiles for CCT Rectangle Bot.

Each profile is a dictionary of settings that override mt5_config.py defaults.
The bot reads the active profile from active_profile.txt on each trade decision,
allowing hot-switching without restarting.

Profiles:
    SAFE       - Conservative: 1% risk, strict limits, single position
    MODERATE   - Balanced: 5% risk, moderate limits, 2 concurrent trades
    AGGRESSIVE - High risk: 10% risk, relaxed limits, 3 concurrent trades
    ULTRA      - Maximum: 25% risk, minimal limits, Asia session enabled
"""

import os

# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

PROFILES = {
    "SAFE": {
        "display_name": "SAFE",
        "description": "Conservative - Low risk, strict limits, single position",
        "RISK_PER_TRADE": 0.01,          # 1% risk per trade
        "MIN_RR_RATIO": 3.0,
        "TARGET_RR_RATIO": 3.0,
        "MAX_CONCURRENT_TRADES": 1,
        "MAX_DAILY_LOSS_PCT": 0.03,      # 3% daily loss limit
        "MAX_TRADES_PER_DAY": 5,
        "MAX_DRAWDOWN_PCT": 0.05,
        "USE_TRAILING_STOP": True,
        "TRAILING_STOP_ACTIVATION_RR": 3.0,
        "TRAILING_STOP_DISTANCE_RR": 2.5,
        "ALLOW_ASIA_SESSION": False,
    },
    "MODERATE": {
        "display_name": "MODERATE",
        "description": "Balanced - Moderate risk, 2 concurrent trades",
        "RISK_PER_TRADE": 0.05,          # 5% risk per trade
        "MIN_RR_RATIO": 3.0,
        "TARGET_RR_RATIO": 3.0,
        "MAX_CONCURRENT_TRADES": 2,
        "MAX_DAILY_LOSS_PCT": 0.10,      # 10% daily loss limit
        "MAX_TRADES_PER_DAY": 10,
        "MAX_DRAWDOWN_PCT": 0.15,
        "USE_TRAILING_STOP": True,
        "TRAILING_STOP_ACTIVATION_RR": 3.0,
        "TRAILING_STOP_DISTANCE_RR": 2.5,
        "ALLOW_ASIA_SESSION": False,
    },
    "AGGRESSIVE": {
        "display_name": "AGGRESSIVE",
        "description": "High risk - 10% per trade, 3 concurrent, relaxed limits",
        "RISK_PER_TRADE": 0.10,          # 10% risk per trade
        "MIN_RR_RATIO": 2.0,
        "TARGET_RR_RATIO": 2.0,
        "MAX_CONCURRENT_TRADES": 3,
        "MAX_DAILY_LOSS_PCT": 0.25,      # 25% daily loss limit
        "MAX_TRADES_PER_DAY": 15,
        "MAX_DRAWDOWN_PCT": 0.30,
        "USE_TRAILING_STOP": True,
        "TRAILING_STOP_ACTIVATION_RR": 3.0,
        "TRAILING_STOP_DISTANCE_RR": 2.5,
        "ALLOW_ASIA_SESSION": False,
    },
    "ULTRA": {
        "display_name": "ULTRA AGGRESSIVE",
        "description": "Maximum risk - 25% per trade, Asia session enabled, wider trailing",
        "RISK_PER_TRADE": 0.25,          # 25% risk per trade
        "MIN_RR_RATIO": 2.0,
        "TARGET_RR_RATIO": 2.0,
        "MAX_CONCURRENT_TRADES": 3,
        "MAX_DAILY_LOSS_PCT": 0.50,      # 50% daily loss limit
        "MAX_TRADES_PER_DAY": 20,
        "MAX_DRAWDOWN_PCT": 0.60,
        "USE_TRAILING_STOP": True,
        "TRAILING_STOP_ACTIVATION_RR": 2.0,   # Activate earlier
        "TRAILING_STOP_DISTANCE_RR": 3.5,     # Wider trailing stop
        "ALLOW_ASIA_SESSION": True,            # Trade Asia session too
    },
}


def get_active_profile_name():
    """
    Read the active profile name from active_profile.txt.

    Returns the profile name string, or "SAFE" as default if file
    doesn't exist or contains an invalid profile name.
    """
    profile_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_profile.txt")
    try:
        with open(profile_file, "r") as f:
            name = f.read().strip().upper()
        if name in PROFILES:
            return name
    except (FileNotFoundError, IOError):
        pass
    return "SAFE"


def get_active_profile():
    """
    Get the active profile dictionary.

    Returns a tuple of (profile_name, profile_dict).
    """
    name = get_active_profile_name()
    return name, PROFILES[name]


def get_profile_display_name():
    """
    Get the display name of the active profile for terminal output.

    Returns string like "SAFE", "MODERATE", "AGGRESSIVE", or "ULTRA AGGRESSIVE".
    """
    name, profile = get_active_profile()
    return profile["display_name"]
