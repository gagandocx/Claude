"""
=============================================================
  Python ML Bridge - Multi-Symbol Entry Point
  Runs the same ML pipeline as main.py but configured for
  ANY forex pair (EURUSD, GBPUSD, USDJPY, AUDUSD, etc.)
  instead of being locked to XAUUSD/Gold.

  Uses separate signal/status/heartbeat files so it can run
  alongside the original Gold-only bridge without conflicts.

  Usage:
    python main_multi.py --live --symbol EURUSD
    python main_multi.py --live --symbol GBPUSD
    python main_multi.py --paper --symbol USDJPY
    python main_multi.py --live --symbol AUDUSD --interval 60
=============================================================
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    MainConfig, DataConfig, MT5_COMMON_PATH, MODEL_DIR, LOG_DIR
)
from signals.bridge import MT5Bridge
from main import PythonMLBridge, setup_logging


# -----------------------------------------------
#  SYMBOL TO YAHOO FINANCE TICKER MAPPING
# -----------------------------------------------
SYMBOL_TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}


def get_yfinance_ticker(symbol: str) -> str:
    """Map a forex symbol to its Yahoo Finance ticker.

    Args:
        symbol: Forex pair like EURUSD, GBPUSD, etc.

    Returns:
        Yahoo Finance ticker string (e.g. EURUSD=X)
    """
    symbol_upper = symbol.upper()
    if symbol_upper in SYMBOL_TICKER_MAP:
        return SYMBOL_TICKER_MAP[symbol_upper]
    # Default pattern: append =X for unknown pairs
    return f"{symbol_upper}=X"


class MultiSymbolBridge(PythonMLBridge):
    """
    Multi-symbol version of the Python ML Bridge.

    Overrides the parent class to use:
    - A configurable symbol (default EURUSD instead of XAUUSD)
    - Separate file paths for signal/confirmation/heartbeat/status
      so it can run alongside the Gold-only bridge instance
    """

    def __init__(self, config=None, symbol="EURUSD"):
        # Override DataConfig before parent init
        self._target_symbol = symbol.upper()
        self._target_ticker = get_yfinance_ticker(self._target_symbol)

        # Monkey-patch DataConfig defaults for this symbol
        DataConfig.symbol = self._target_symbol
        DataConfig.yfinance_ticker = self._target_ticker

        # Call parent init
        super().__init__(config)

        # Replace the bridge with multi-symbol file paths
        signal_path = os.path.join(MT5_COMMON_PATH, "python_bridge_signal_multi.csv")
        confirm_path = os.path.join(MT5_COMMON_PATH, "python_bridge_confirm_multi.csv")
        exit_path = os.path.join(MT5_COMMON_PATH, "python_bridge_exit_multi.csv")
        status_path = os.path.join(MT5_COMMON_PATH, "python_bridge_status_multi.txt")
        heartbeat_path = os.path.join(MT5_COMMON_PATH, "python_bridge_heartbeat_multi.txt")

        self.bridge = MT5Bridge(
            signal_path=signal_path,
            confirmation_path=confirm_path,
            exit_signal_path=exit_path,
            status_path=status_path,
            heartbeat_path=heartbeat_path,
        )

        self.logger.info(f"Multi-Symbol Bridge configured for: {self._target_symbol}")
        self.logger.info(f"  Yahoo Finance ticker: {self._target_ticker}")
        self.logger.info(f"  Signal file: python_bridge_signal_multi.csv")
        self.logger.info(f"  Heartbeat file: python_bridge_heartbeat_multi.txt")


def parse_args():
    """Parse command-line arguments for multi-symbol bridge."""
    parser = argparse.ArgumentParser(
        description="Python ML Bridge - Multi-Symbol Version"
    )
    parser.add_argument(
        "--symbol", type=str, default="EURUSD",
        help="Forex pair to trade (e.g. EURUSD, GBPUSD, USDJPY, AUDUSD)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run in live trading mode (default is paper)"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run in paper trading mode"
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Cycle interval in seconds (default: 10)"
    )
    return parser.parse_args()


def main():
    """Main entry point for multi-symbol bridge."""
    args = parse_args()

    print("=" * 60)
    print("  Python ML Bridge - MULTI SYMBOL")
    print(f"  Trading: {args.symbol.upper()}")
    print(f"  Ticker:  {get_yfinance_ticker(args.symbol)}")
    print("=" * 60)

    config = MainConfig()

    if args.live:
        config.paper_trading = False
    elif args.paper:
        config.paper_trading = True

    if args.interval is not None:
        config.interval_seconds = args.interval

    bridge = MultiSymbolBridge(config=config, symbol=args.symbol)
    bridge.run()


if __name__ == "__main__":
    main()
