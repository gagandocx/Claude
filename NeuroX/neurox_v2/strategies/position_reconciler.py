"""
=============================================================
  Position Reconciliation on Startup

  Solves the "ghost HOLD" problem: if Python restarts while a
  trade is open, it would think there is no active position
  and refuse to generate new signals (or worse, open a second).

  On startup, Python reads a positions CSV file written by the EA
  every timer cycle. This file lists all open positions. Python
  reads it on init and sets _active_position accordingly.

  The EA writes: python_bridge_positions.csv
  Format: ticket,symbol,direction,lot_size,open_price,open_time,sl,tp
=============================================================
"""

import csv
import io
import logging
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Default path for positions CSV (in MT5 Common Files folder)
DEFAULT_POSITIONS_FILE = "python_bridge_positions.csv"

# Expected CSV columns
POSITIONS_HEADERS = [
    "ticket", "symbol", "direction", "lot_size",
    "open_price", "open_time", "sl", "tp"
]


class PositionReconciler:
    """
    Reads the EA's positions CSV on startup to reconcile Python's state
    with MT5's actual open positions.

    Usage:
        reconciler = PositionReconciler(common_files_path)
        position = reconciler.reconcile()
        if position:
            signal_generator._active_position = position
    """

    def __init__(self, common_files_path: str, symbol: str = "XAUUSD"):
        """
        Args:
            common_files_path: Path to MT5 Common Files directory
            symbol: The symbol we trade (filter positions to this symbol)
        """
        self.positions_file = os.path.join(
            common_files_path, DEFAULT_POSITIONS_FILE
        )
        self.symbol = symbol
        logger.info(
            f"[Reconciler] Initialized. Positions file: {self.positions_file}"
        )

    def reconcile(self) -> Optional[Dict]:
        """
        Read the positions CSV and return the active position (if any)
        matching our trading symbol.

        Returns:
            Dict with position info compatible with SignalGenerator._active_position:
            {
                "direction": "BUY" or "SELL",
                "entry_price": float,
                "entry_time": float (unix timestamp),
                "lot_size": float,
                "ticket": str,
                "sl": float,
                "tp": float,
            }
            or None if no open positions found.
        """
        positions = self._read_positions()
        if not positions:
            logger.info("[Reconciler] No open positions found - starting fresh")
            return None

        # Filter to our symbol
        our_positions = [
            p for p in positions
            if p.get("symbol", "").upper() == self.symbol.upper()
        ]

        if not our_positions:
            logger.info(
                f"[Reconciler] {len(positions)} positions found but none "
                f"for {self.symbol} - starting fresh"
            )
            return None

        # Take the most recent position (by open_time)
        # If multiple positions exist for our symbol, use the latest
        our_positions.sort(
            key=lambda p: p.get("open_time", ""), reverse=True
        )
        pos = our_positions[0]

        # Convert to internal format
        direction = pos.get("direction", "BUY").upper()
        try:
            entry_price = float(pos.get("open_price", 0))
            lot_size = float(pos.get("lot_size", 0.01))
            sl = float(pos.get("sl", 0))
            tp = float(pos.get("tp", 0))
        except (ValueError, TypeError):
            logger.warning(
                "[Reconciler] Could not parse position data: %s", pos
            )
            return None

        # Parse open_time - try multiple formats
        entry_time = self._parse_time(pos.get("open_time", ""))

        active_position = {
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "lot_size": lot_size,
            "ticket": pos.get("ticket", "unknown"),
            "sl": sl,
            "tp": tp,
            "reconciled": True,  # Flag indicating this was recovered on startup
        }

        logger.info(
            f"[Reconciler] RECOVERED POSITION: {direction} "
            f"{lot_size:.2f} lot @ ${entry_price:.2f} "
            f"(ticket #{pos.get('ticket', '?')})"
        )
        return active_position

    def _read_positions(self) -> List[Dict]:
        """
        Read the positions CSV file (with retry for file locking).

        Handles both UTF-8 and UTF-16LE (BOM) encoding that MT5 may write.

        Returns:
            List of position dicts, or empty list on failure.
        """
        if not os.path.exists(self.positions_file):
            logger.debug(
                "[Reconciler] Positions file not found: %s",
                self.positions_file
            )
            return []

        # Check file age - if older than 60 seconds, the EA might not be running
        try:
            file_age = time.time() - os.path.getmtime(self.positions_file)
            if file_age > 60:
                logger.warning(
                    "[Reconciler] Positions file is %.0fs old - EA may not "
                    "be running. Using data anyway for reconciliation.",
                    file_age
                )
        except OSError:
            pass

        # Copy to temp file to avoid lock conflicts with MT5
        tmp_path = None
        try:
            directory = os.path.dirname(self.positions_file) or "."
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="pos_read_", dir=directory
            )
            os.close(fd)

            copied = False
            for attempt in range(5):
                try:
                    shutil.copy2(self.positions_file, tmp_path)
                    copied = True
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        time.sleep(0.01)

            if not copied:
                logger.warning(
                    "[Reconciler] Could not copy positions file "
                    "(MT5 lock held)"
                )
                return []

            # Detect encoding
            with open(tmp_path, "rb") as f:
                raw = f.read(2)
            encoding = "utf-16-le" if raw == b"\xff\xfe" else "utf-8"

            with open(tmp_path, "r", encoding=encoding) as f:
                content = f.read()
            if content and content[0] == "\ufeff":
                content = content[1:]

            # Parse CSV
            reader = csv.DictReader(io.StringIO(content))
            positions = [dict(row) for row in reader]
            return positions

        except Exception as e:
            logger.warning(f"[Reconciler] Error reading positions: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _parse_time(self, time_str: str) -> float:
        """
        Parse a time string from MT5 into a Unix timestamp.
        Tries multiple formats that MT5 may use.

        Args:
            time_str: Time string from the CSV

        Returns:
            Unix timestamp (float), or current time if parsing fails.
        """
        if not time_str or not time_str.strip():
            return time.time()

        formats = [
            "%Y.%m.%d %H:%M:%S",   # MT5 default format
            "%Y-%m-%d %H:%M:%S",    # ISO-like
            "%Y.%m.%d %H:%M",       # Without seconds
            "%Y-%m-%dT%H:%M:%S",    # ISO 8601
        ]

        for fmt in formats:
            try:
                dt = __import__("datetime").datetime.strptime(
                    time_str.strip(), fmt
                )
                return dt.timestamp()
            except ValueError:
                continue

        # If all formats fail, return current time
        logger.debug(
            f"[Reconciler] Could not parse time '{time_str}', using now()"
        )
        return time.time()

    def get_positions_file_path(self) -> str:
        """Return the path to the positions file (for EA configuration)."""
        return self.positions_file
