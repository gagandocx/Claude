"""
Adaptive Multi-Currency Trading System
=======================================
An advanced regime-aware trading system that dynamically selects strategies,
adapts position sizing, and manages risk across multiple currency pairs.

Version: 1.0.0

Architecture:
    core/               - Core engine modules (regime detection, strategies, risk management)
    data_loader.py      - Multi-symbol data loading and synthetic data generation
    backtest_engine.py  - Multi-symbol adaptive backtester
    live_trader.py      - MT5 live trading with multi-symbol support
    config.py           - Configuration management with risk profiles
    run_demo.py         - Self-contained demo (no external data needed)
    run_backtest.py     - Production backtest with real data

Quick Start:
    python run_demo.py   # Run demo with synthetic data

Usage:
    from adaptive_system.core import (
        RegimeDetector, MarketRegime,
        StrategySelector,
        PositionSizer,
        RiskManager,
        PortfolioManager,
    )
"""

__version__ = "1.0.0"
__author__ = "Adaptive Trading System"
__description__ = "Multi-currency regime-aware adaptive trading system"
