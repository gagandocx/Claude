"""
System Configuration Management
==================================
Centralized configuration for the adaptive multi-currency trading system.

Features:
    - SystemConfig dataclass with all tunable parameters and defaults
    - SymbolConfig dataclass per-symbol (session hours, spread, contract size)
    - Risk profiles: CONSERVATIVE, BALANCED, AGGRESSIVE
    - JSON persistence (load_config / save_config)
    - Default configs that work out of the box
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class SymbolSettings:
    """Per-symbol trading configuration."""
    symbol: str = "XAUUSD"
    pip_size: float = 0.01
    contract_size: float = 100.0
    typical_spread: float = 0.30
    min_lot: float = 0.01
    max_lot: float = 200.0
    lot_step: float = 0.01
    active: bool = True
    # Session hours (UTC) when trading is allowed
    active_hours: List[int] = field(default_factory=lambda: list(range(1, 22)))
    # Blackout hours (UTC) when no new trades are opened
    blackout_hours: List[int] = field(default_factory=lambda: [23, 0])
    # Symbol-specific overrides (optional)
    sl_mult_override: Optional[float] = None
    tp_mult_override: Optional[float] = None


@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection."""
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_strong_trend: float = 40.0
    adx_weak_threshold: float = 18.0
    vol_ratio_high: float = 1.4
    vol_ratio_low: float = 0.7
    vol_ratio_very_high: float = 2.0
    slope_strong: float = 0.15
    slope_weak: float = 0.05
    hurst_mean_revert: float = 0.4
    hurst_trending: float = 0.6
    ema_period: int = 20
    atr_period: int = 14
    vol_lookback: int = 50
    slope_lookback: int = 5
    hurst_window: int = 100


@dataclass
class StrategyConfig:
    """Configuration for strategy selection."""
    confidence_gate: float = 0.5
    low_confidence_size_mult: float = 0.5
    ensemble_mode: bool = False
    min_trades_for_learning: float = 5.0
    scorecard_alpha: float = 0.05


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    initial_equity: float = 1000.0
    leverage: float = 500.0
    risk_grow: float = 0.08
    risk_protect: float = 0.02
    max_risk_cap: float = 0.10
    min_risk_floor: float = 0.003
    dd_power: float = 12.0
    dd_power_adaptive: bool = True
    at_high_thresh: float = 0.01
    dd_halt: float = 0.20
    use_kelly: bool = True
    kelly_fraction: float = 0.5
    loss_boost: float = 1.0
    win_reduce: float = 0.7
    streak_threshold: int = 3
    correlation_penalty_factor: float = 0.3
    max_correlation_reduction: float = 0.5
    max_risk_per_symbol: float = 0.05
    max_risk_total: float = 0.15


@dataclass
class RiskManagementConfig:
    """Configuration for risk management."""
    max_risk_per_symbol: float = 0.05
    max_total_risk: float = 0.15
    max_concurrent_positions: int = 6
    max_correlation: float = 0.7
    correlation_reduction_factor: float = 0.5
    dd_halt_threshold: float = 0.20
    dd_reduce_threshold: float = 0.10
    dd_reduce_factor: float = 0.5
    daily_loss_limit: float = 0.05
    daily_loss_reset_hour: int = 0
    min_lot_size: float = 0.01


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    performance_alpha: float = 0.05
    transition_alpha: float = 0.02
    parameter_alpha: float = 0.05
    profiler_alpha: float = 0.02
    min_observations: int = 5


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_equity: float = 1000.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 1.0
    walk_forward_windows: int = 4
    walk_forward_train_ratio: float = 0.7
    min_bars_per_symbol: int = 200


@dataclass
class LiveTradingConfig:
    """Configuration for live trading."""
    loop_interval_seconds: float = 1.0
    bar_fetch_count: int = 500
    magic_number: int = 202501
    max_retries: int = 10
    base_retry_delay: float = 5.0
    max_retry_delay: float = 300.0
    state_save_interval: int = 10  # Save state every N bars
    log_level: str = "INFO"


@dataclass
class SystemConfig:
    """
    Master configuration for the entire adaptive trading system.

    Aggregates all sub-configurations into a single structure that can be
    serialized to/from JSON for persistence.
    """
    # System identification
    version: str = "1.0.0"
    name: str = "Adaptive Multi-Currency System"
    risk_profile: str = "balanced"  # "conservative", "balanced", "aggressive"

    # Sub-configurations
    regime_detection: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live_trading: LiveTradingConfig = field(default_factory=LiveTradingConfig)

    # Symbol configurations
    symbols: Dict[str, SymbolSettings] = field(default_factory=dict)

    def __post_init__(self):
        """Apply risk profile defaults if no symbols configured."""
        if not self.symbols:
            self.symbols = get_default_symbols()


# ============================================================================
# RISK PROFILES
# ============================================================================

def get_conservative_config() -> SystemConfig:
    """
    Conservative risk profile.

    Lower risk per trade, fewer concurrent positions, tighter drawdown limits.
    Suitable for capital preservation and steady growth.
    """
    config = SystemConfig(risk_profile="conservative")
    config.position_sizing.risk_grow = 0.05
    config.position_sizing.risk_protect = 0.01
    config.position_sizing.max_risk_cap = 0.07
    config.position_sizing.dd_halt = 0.15
    config.position_sizing.dd_power = 15.0
    config.position_sizing.loss_boost = 1.0  # No Martingale
    config.position_sizing.max_risk_per_symbol = 0.03
    config.position_sizing.max_risk_total = 0.08

    config.risk_management.max_concurrent_positions = 3
    config.risk_management.max_total_risk = 0.08
    config.risk_management.max_risk_per_symbol = 0.03
    config.risk_management.dd_halt_threshold = 0.15
    config.risk_management.dd_reduce_threshold = 0.07
    config.risk_management.daily_loss_limit = 0.03

    config.strategy.confidence_gate = 0.6
    config.strategy.low_confidence_size_mult = 0.3

    return config


def get_balanced_config() -> SystemConfig:
    """
    Balanced risk profile (default).

    Moderate risk with adaptive sizing. Good balance of growth and protection.
    """
    config = SystemConfig(risk_profile="balanced")
    # All defaults are balanced
    return config


def get_aggressive_config() -> SystemConfig:
    """
    Aggressive risk profile.

    Higher risk per trade, more concurrent positions, wider drawdown tolerance.
    Suitable for accounts seeking maximum growth with higher risk tolerance.
    """
    config = SystemConfig(risk_profile="aggressive")
    config.position_sizing.risk_grow = 0.12
    config.position_sizing.risk_protect = 0.03
    config.position_sizing.max_risk_cap = 0.15
    config.position_sizing.dd_halt = 0.25
    config.position_sizing.dd_power = 10.0
    config.position_sizing.loss_boost = 1.2  # Mild boost only
    config.position_sizing.max_risk_per_symbol = 0.06
    config.position_sizing.max_risk_total = 0.18
    config.position_sizing.kelly_fraction = 0.6

    config.risk_management.max_concurrent_positions = 8
    config.risk_management.max_total_risk = 0.18
    config.risk_management.max_risk_per_symbol = 0.06
    config.risk_management.dd_halt_threshold = 0.25
    config.risk_management.dd_reduce_threshold = 0.12
    config.risk_management.daily_loss_limit = 0.06

    config.strategy.confidence_gate = 0.4
    config.strategy.low_confidence_size_mult = 0.6
    config.strategy.ensemble_mode = True

    return config


RISK_PROFILES = {
    "conservative": get_conservative_config,
    "balanced": get_balanced_config,
    "aggressive": get_aggressive_config,
}


# ============================================================================
# DEFAULT SYMBOL CONFIGURATIONS
# ============================================================================

def get_default_symbols() -> Dict[str, SymbolSettings]:
    """Get default symbol configurations for common instruments."""
    return {
        "XAUUSD": SymbolSettings(
            symbol="XAUUSD",
            pip_size=0.01,
            contract_size=100.0,
            typical_spread=0.30,
            active_hours=list(range(1, 22)),
            blackout_hours=[23, 0],
        ),
        "EURUSD": SymbolSettings(
            symbol="EURUSD",
            pip_size=0.0001,
            contract_size=100000.0,
            typical_spread=0.00012,
            active_hours=list(range(7, 21)),
            blackout_hours=[23, 0, 1, 2, 3, 4, 5, 6],
        ),
        "GBPJPY": SymbolSettings(
            symbol="GBPJPY",
            pip_size=0.01,
            contract_size=100000.0,
            typical_spread=0.03,
            active_hours=list(range(3, 20)),
            blackout_hours=[23, 0, 1, 2],
        ),
        "USDJPY": SymbolSettings(
            symbol="USDJPY",
            pip_size=0.01,
            contract_size=100000.0,
            typical_spread=0.015,
            active_hours=list(range(1, 22)),
            blackout_hours=[23, 0],
        ),
        "NAS100": SymbolSettings(
            symbol="NAS100",
            pip_size=0.01,
            contract_size=1.0,
            typical_spread=1.5,
            min_lot=0.1,
            lot_step=0.1,
            active_hours=list(range(13, 21)),
            blackout_hours=list(range(0, 13)) + [22, 23],
        ),
        "BTCUSD": SymbolSettings(
            symbol="BTCUSD",
            pip_size=0.01,
            contract_size=1.0,
            typical_spread=30.0,
            active_hours=list(range(24)),  # 24/7
            blackout_hours=[],
        ),
    }


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_config(config: SystemConfig, path: Union[str, Path]):
    """
    Save system configuration to a JSON file.

    Parameters
    ----------
    config : SystemConfig
        Configuration to save.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _config_to_dict(config)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: Union[str, Path]) -> SystemConfig:
    """
    Load system configuration from a JSON file.

    Applies the saved risk profile first, then overlays any custom values.

    Parameters
    ----------
    path : str or Path
        Input JSON file path.

    Returns
    -------
    SystemConfig
        Loaded configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Start from risk profile if specified
    profile = data.get("risk_profile", "balanced")
    if profile in RISK_PROFILES:
        config = RISK_PROFILES[profile]()
    else:
        config = SystemConfig()

    # Overlay saved values
    _apply_dict_to_config(config, data)

    return config


def _config_to_dict(config: SystemConfig) -> dict:
    """Convert SystemConfig to a JSON-serializable dictionary."""
    data = {}
    data["version"] = config.version
    data["name"] = config.name
    data["risk_profile"] = config.risk_profile
    data["regime_detection"] = asdict(config.regime_detection)
    data["strategy"] = asdict(config.strategy)
    data["position_sizing"] = asdict(config.position_sizing)
    data["risk_management"] = asdict(config.risk_management)
    data["online_learning"] = asdict(config.online_learning)
    data["backtest"] = asdict(config.backtest)
    data["live_trading"] = asdict(config.live_trading)

    # Symbols
    symbols_dict = {}
    for sym_name, sym_cfg in config.symbols.items():
        symbols_dict[sym_name] = asdict(sym_cfg)
    data["symbols"] = symbols_dict

    return data


def _apply_dict_to_config(config: SystemConfig, data: dict):
    """Apply dictionary values to an existing config (overlay)."""
    if "version" in data:
        config.version = data["version"]
    if "name" in data:
        config.name = data["name"]

    # Apply sub-configs
    sub_configs = {
        "regime_detection": config.regime_detection,
        "strategy": config.strategy,
        "position_sizing": config.position_sizing,
        "risk_management": config.risk_management,
        "online_learning": config.online_learning,
        "backtest": config.backtest,
        "live_trading": config.live_trading,
    }

    for key, sub_cfg in sub_configs.items():
        if key in data and isinstance(data[key], dict):
            for field_name, value in data[key].items():
                if hasattr(sub_cfg, field_name):
                    setattr(sub_cfg, field_name, value)

    # Apply symbols
    if "symbols" in data and isinstance(data["symbols"], dict):
        config.symbols = {}
        for sym_name, sym_data in data["symbols"].items():
            sym_cfg = SymbolSettings(symbol=sym_name)
            for field_name, value in sym_data.items():
                if hasattr(sym_cfg, field_name):
                    setattr(sym_cfg, field_name, value)
            config.symbols[sym_name] = sym_cfg
