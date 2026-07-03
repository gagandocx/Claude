"""
Core engine modules for the adaptive multi-currency trading system.

Exports all major components for convenient import:
    from adaptive_system.core import RegimeDetector, MarketRegime, ...
"""

from .indicators import (
    compute_rsi,
    compute_atr,
    compute_ema,
    compute_sma,
    compute_bollinger_bands,
    compute_adx,
    compute_stochastic,
    compute_vwap,
    compute_keltner_channels,
    compute_hurst_exponent,
)

from .regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeHistory,
)

from .strategies import (
    AdaptiveBaseStrategy,
    TrendFollower,
    MeanReversion,
    BreakoutTrader,
    ScalpMomentum,
    FadeStrategy,
)

from .strategy_selector import (
    StrategySelector,
    StrategyScorecard,
)

from .position_sizer import (
    PositionSizer,
    SizingConfig,
    SizingResult,
)

from .online_learner import (
    ExponentialStats,
    StrategyPerformanceTracker,
    RegimeTransitionMatrix,
    ParameterAdapter,
    MarketProfiler,
)

from .risk_manager import (
    RiskManager,
    RiskConfig,
    TradeProposal,
    RiskDecision,
)

from .portfolio_manager import (
    PortfolioManager,
    SymbolConfig,
    PortfolioState,
)

__all__ = [
    # Indicators
    "compute_rsi",
    "compute_atr",
    "compute_ema",
    "compute_sma",
    "compute_bollinger_bands",
    "compute_adx",
    "compute_stochastic",
    "compute_vwap",
    "compute_keltner_channels",
    "compute_hurst_exponent",
    # Regime detection
    "MarketRegime",
    "RegimeDetector",
    "RegimeHistory",
    # Strategies
    "AdaptiveBaseStrategy",
    "TrendFollower",
    "MeanReversion",
    "BreakoutTrader",
    "ScalpMomentum",
    "FadeStrategy",
    # Strategy selection
    "StrategySelector",
    "StrategyScorecard",
    # Position sizing
    "PositionSizer",
    "SizingConfig",
    "SizingResult",
    # Online learning
    "ExponentialStats",
    "StrategyPerformanceTracker",
    "RegimeTransitionMatrix",
    "ParameterAdapter",
    "MarketProfiler",
    # Risk management
    "RiskManager",
    "RiskConfig",
    "TradeProposal",
    "RiskDecision",
    # Portfolio management
    "PortfolioManager",
    "SymbolConfig",
    "PortfolioState",
]
