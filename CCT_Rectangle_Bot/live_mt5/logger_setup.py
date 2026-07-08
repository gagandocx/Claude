"""
Logging configuration for CCT Rectangle Bot - Live MT5 Trading.

Provides rotating file handler and console handler with configurable levels.
Log files are stored in the live_mt5/logs/ directory with timestamped names.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import mt5_config


def setup_logger(name: str = "cct_live_trader") -> logging.Logger:
    """
    Create and configure a logger with both file and console handlers.

    Args:
        name: Logger name (default: 'cct_live_trader')

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), mt5_config.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = f"cct_live_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # File handler - rotating, detailed logging
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=mt5_config.LOG_MAX_BYTES,
        backupCount=mt5_config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, mt5_config.LOG_LEVEL, logging.INFO))
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # Console handler - warnings and above for cleaner terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("CCT Rectangle Bot - Live MT5 Trader Started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Symbol: {mt5_config.SYMBOL}")
    logger.info(f"Demo Mode: {mt5_config.DEMO_MODE}")
    logger.info(f"Risk per trade: {mt5_config.RISK_PER_TRADE * 100:.1f}%")
    logger.info("=" * 60)

    return logger
