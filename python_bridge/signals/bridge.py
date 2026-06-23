"""
=============================================================
  Python ML Bridge - MT5 File Bridge
  Writes trade signals to CSV for MT5 to read, and reads
  execution confirmations back from MT5.
  Signal format: timestamp,symbol,action,confidence,sl_pips,tp_pips,
                 lot_size,model_name,regime
=============================================================
"""

import os
import csv
import time
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SIGNAL_FILE, CONFIRMATION_FILE, MT5_COMMON_PATH
from strategies.signal_generator import TradeSignal


# CSV column headers
SIGNAL_HEADERS = [
    "timestamp", "symbol", "action", "confidence",
    "sl_pips", "tp_pips", "lot_size", "model_name", "regime"
]

# Exit signal file for position management commands
EXIT_SIGNAL_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_exit.csv")

EXIT_SIGNAL_HEADERS = [
    "timestamp", "ticket", "action", "lot_pct", "new_sl", "reason"
]

CONFIRMATION_HEADERS = [
    "timestamp", "ticket", "symbol", "action", "lot_size",
    "open_price", "sl", "tp", "status"
]

# Status file for communicating bridge state to MT5 dashboard
STATUS_FILE = os.path.join(MT5_COMMON_PATH, "python_bridge_status.txt")


class MT5Bridge:
    """
    File-based bridge between Python ML system and MetaTrader 5.

    Communication protocol:
        Python -> MT5: Writes signal CSV (python_bridge_signal.csv)
        MT5 -> Python: Writes confirmation CSV (python_bridge_confirm.csv)

    The CSV approach is chosen for reliability and simplicity:
        - No socket connections to maintain
        - Works across different machines via shared folders
        - Easy to debug (human-readable files)
        - MT5 file I/O is well-supported with FileOpen/FileRead
    """

    def __init__(self, signal_path: Optional[str] = None,
                 confirmation_path: Optional[str] = None,
                 exit_signal_path: Optional[str] = None,
                 status_path: Optional[str] = None,
                 heartbeat_path: Optional[str] = None):
        self.signal_path = signal_path or SIGNAL_FILE
        self.confirmation_path = confirmation_path or CONFIRMATION_FILE
        self.exit_signal_path = exit_signal_path or EXIT_SIGNAL_FILE
        self.status_path = status_path or STATUS_FILE
        # Heartbeat defaults to same directory as signal file
        if heartbeat_path:
            self.heartbeat_path = heartbeat_path
        else:
            self.heartbeat_path = os.path.join(
                os.path.dirname(self.signal_path), "python_bridge_heartbeat.txt"
            )
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.signal_path, self.confirmation_path,
                     self.exit_signal_path, self.status_path]:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def write_signal(self, signal: TradeSignal) -> bool:
        """
        Write a trade signal to the CSV file for MT5 to read.
        Uses atomic write (write to temp file, then rename) to prevent
        race conditions where MT5 reads a partially-written file.

        Includes retry logic for os.replace() to handle Windows file
        locking when MT5 has the signal file open for reading.

        Plain ASCII text with CRLF line endings - no BOM, no UTF-8.
        This ensures MQL5's FileOpen with FILE_ANSI reads it cleanly.

        Args:
            signal: TradeSignal object to write

        Returns:
            True if written successfully
        """
        logger = logging.getLogger("PythonBridge")
        try:
            directory = os.path.dirname(self.signal_path) or "."
            # Write to a temporary file in the same directory
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="signal_", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="ascii",
                               newline="") as f:
                    # Write header line with CRLF
                    header = ",".join(SIGNAL_HEADERS) + "\r\n"
                    f.write(header)
                    # Write data line with CRLF
                    data = (
                        f"{signal.timestamp},"
                        f"{signal.symbol},"
                        f"{signal.action},"
                        f"{signal.confidence:.4f},"
                        f"{signal.sl_pips:.1f},"
                        f"{signal.tp_pips:.1f},"
                        f"{signal.lot_size:.2f},"
                        f"{signal.model_name},"
                        f"{signal.regime}\r\n"
                    )
                    f.write(data)

                # Atomic rename with retry logic for Windows file locking.
                # On Windows, os.replace() can throw PermissionError if MT5
                # has the target file open (FileOpen with FILE_READ).
                max_retries = 3
                replace_succeeded = False
                for attempt in range(max_retries):
                    try:
                        os.replace(tmp_path, self.signal_path)
                        replace_succeeded = True
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"[Bridge] os.replace() attempt {attempt + 1}/{max_retries} "
                                f"failed: {e}. Retrying in 0.05s..."
                            )
                            time.sleep(0.05)
                        else:
                            logger.warning(
                                f"[Bridge] os.replace() failed after {max_retries} "
                                f"attempts: {e}. Falling back to direct write."
                            )

                # Fallback: if atomic replace failed after all retries,
                # try delete-then-rename, or direct write as last resort
                if not replace_succeeded:
                    try:
                        # Fallback 1: delete target then rename temp
                        if os.path.exists(self.signal_path):
                            os.unlink(self.signal_path)
                        os.rename(tmp_path, self.signal_path)
                        replace_succeeded = True
                        logger.warning(
                            "[Bridge] Fallback (unlink+rename) succeeded."
                        )
                    except (PermissionError, OSError) as e2:
                        # Retry os.rename once more after a short delay.
                        # The target was just unlinked, so a brief wait may
                        # allow the OS to release handles.
                        try:
                            time.sleep(0.05)
                            os.rename(tmp_path, self.signal_path)
                            replace_succeeded = True
                            logger.warning(
                                "[Bridge] Fallback (unlink+rename retry) "
                                "succeeded."
                            )
                        except (PermissionError, OSError) as e3:
                            logger.warning(
                                f"[Bridge] Fallback unlink+rename failed: "
                                f"{e2}, retry: {e3}. Using direct write."
                            )
                            # Clean up orphaned temp file before direct write
                            if os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except OSError:
                                    pass
                            # Fallback 2: direct write to target path
                            try:
                                with open(self.signal_path, "w",
                                          encoding="ascii",
                                          newline="") as f:
                                    header = (
                                        ",".join(SIGNAL_HEADERS) + "\r\n"
                                    )
                                    f.write(header)
                                    data = (
                                        f"{signal.timestamp},"
                                        f"{signal.symbol},"
                                        f"{signal.action},"
                                        f"{signal.confidence:.4f},"
                                        f"{signal.sl_pips:.1f},"
                                        f"{signal.tp_pips:.1f},"
                                        f"{signal.lot_size:.2f},"
                                        f"{signal.model_name},"
                                        f"{signal.regime}\r\n"
                                    )
                                    f.write(data)
                                replace_succeeded = True
                                logger.warning(
                                    "[Bridge] Direct write fallback "
                                    "succeeded."
                                )
                            except Exception as e4:
                                logger.error(
                                    f"[Bridge] All write methods "
                                    f"failed: {e4}"
                                )

            except Exception:
                # Clean up temp file on error
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            return replace_succeeded
        except Exception as e:
            print(f"[Bridge] Error writing signal: {e}")
            return False

    def read_signal(self) -> Optional[Dict]:
        """
        Read the current signal file (for testing/verification).

        Returns:
            Dict with signal data or None
        """
        if not os.path.exists(self.signal_path):
            return None

        try:
            with open(self.signal_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return dict(row)
        except Exception as e:
            print(f"[Bridge] Error reading signal: {e}")
        return None

    def read_confirmations(self) -> List[Dict]:
        """
        Read execution confirmations from MT5.

        Returns:
            List of confirmation dicts
        """
        if not os.path.exists(self.confirmation_path):
            return []

        confirmations = []
        try:
            with open(self.confirmation_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    confirmations.append(dict(row))
        except Exception as e:
            print(f"[Bridge] Error reading confirmations: {e}")

        return confirmations

    def clear_signal(self):
        """Clear the signal file after MT5 has read it."""
        try:
            if os.path.exists(self.signal_path):
                os.remove(self.signal_path)
        except Exception as e:
            print(f"[Bridge] Error clearing signal: {e}")

    def clear_confirmations(self):
        """Clear the confirmation file after Python has processed it."""
        try:
            if os.path.exists(self.confirmation_path):
                os.remove(self.confirmation_path)
        except Exception as e:
            print(f"[Bridge] Error clearing confirmations: {e}")

    def is_signal_fresh(self, max_age_seconds: int = 300) -> bool:
        """
        Check if the current signal file is fresh enough to act on.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            True if signal exists and is within max age
        """
        if not os.path.exists(self.signal_path):
            return False

        try:
            file_age = time.time() - os.path.getmtime(self.signal_path)
            return file_age < max_age_seconds
        except Exception:
            return False

    def write_heartbeat(self):
        """Write a heartbeat file to indicate the bridge is running."""
        try:
            with open(self.heartbeat_path, "w") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Python ML Bridge Active\n")
        except Exception as e:
            print(f"[Bridge] Error writing heartbeat: {e}")

    def write_status(self, status_type: str, message: str) -> bool:
        """
        Write a status message for the MT5 dashboard to display.

        The MT5 EA reads this file and shows the status on its
        graphical chart panel. This allows traders to see news warnings,
        errors, and running status without checking the Python terminal.

        Format: Plain ASCII, "type|message\\r\\n"
        Valid status_type values:
            - "OK"      : Normal operation (green on dashboard)
            - "NEWS"    : News event blocking trades (yellow/orange)
            - "WARNING" : Error or warning condition (red)
            - "ERROR"   : Critical error (red)

        Uses atomic write to prevent MT5 from reading a partial file.

        Args:
            status_type: One of "OK", "NEWS", "WARNING", "ERROR"
            message: Human-readable status message (ASCII only)

        Returns:
            True if written successfully
        """
        try:
            # Sanitize message: strip non-ASCII and pipe characters
            safe_message = "".join(
                c for c in message if 32 <= ord(c) < 127 and c != "|"
            ).strip()
            # Limit length to avoid display issues on MT5 chart
            if len(safe_message) > 200:
                safe_message = safe_message[:197] + "..."

            directory = os.path.dirname(self.status_path) or "."
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="status_", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="ascii", newline="") as f:
                    f.write(f"{status_type}|{safe_message}\r\n")
                os.replace(tmp_path, self.status_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            return True
        except Exception as e:
            print(f"[Bridge] Error writing status: {e}")
            return False

    def write_exit_signal(self, ticket: str, action: str,
                          lot_pct: float = 1.0, new_sl: float = 0.0,
                          reason: str = "") -> bool:
        """
        Write an exit/modification signal for an open position.
        MT5 EA reads this file to close, partial close, or modify SL.

        Uses atomic write to prevent race conditions.

        Args:
            ticket: Position ticket number
            action: CLOSE_FULL, CLOSE_PARTIAL, or MODIFY_SL
            lot_pct: Percentage of position to close (0.0-1.0)
            new_sl: New stop loss price (for MODIFY_SL)
            reason: Human-readable reason for the action

        Returns:
            True if written successfully
        """
        try:
            directory = os.path.dirname(self.exit_signal_path) or "."
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="exit_", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="ascii", newline="") as f:
                    header = ",".join(EXIT_SIGNAL_HEADERS) + "\r\n"
                    f.write(header)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data = (
                        f"{timestamp},"
                        f"{ticket},"
                        f"{action},"
                        f"{lot_pct:.2f},"
                        f"{new_sl:.5f},"
                        f"{reason}\r\n"
                    )
                    f.write(data)
                os.replace(tmp_path, self.exit_signal_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            return True
        except Exception as e:
            print(f"[Bridge] Error writing exit signal: {e}")
            return False

    def write_exit_signals(self, signals: List[Dict]) -> bool:
        """
        Write multiple exit signals at once (batch write).

        Args:
            signals: List of dicts with keys: ticket, action, lot_pct, new_sl, reason

        Returns:
            True if written successfully
        """
        if not signals:
            return True

        try:
            directory = os.path.dirname(self.exit_signal_path) or "."
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="exit_", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="ascii", newline="") as f:
                    header = ",".join(EXIT_SIGNAL_HEADERS) + "\r\n"
                    f.write(header)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for sig in signals:
                        data = (
                            f"{timestamp},"
                            f"{sig['ticket']},"
                            f"{sig['action']},"
                            f"{sig.get('lot_pct', 1.0):.2f},"
                            f"{sig.get('new_sl', 0.0):.5f},"
                            f"{sig.get('reason', '')}\r\n"
                        )
                        f.write(data)
                os.replace(tmp_path, self.exit_signal_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            return True
        except Exception as e:
            print(f"[Bridge] Error writing exit signals: {e}")
            return False

    def read_exit_signal(self) -> Optional[List[Dict]]:
        """Read exit signal file (for testing/verification)."""
        if not os.path.exists(self.exit_signal_path):
            return None
        try:
            signals = []
            with open(self.exit_signal_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    signals.append(dict(row))
            return signals if signals else None
        except Exception as e:
            print(f"[Bridge] Error reading exit signal: {e}")
            return None

    def clear_exit_signal(self):
        """Clear exit signal file after MT5 has processed it."""
        try:
            if os.path.exists(self.exit_signal_path):
                os.remove(self.exit_signal_path)
        except Exception as e:
            print(f"[Bridge] Error clearing exit signal: {e}")

    def get_bridge_status(self) -> Dict:
        """Get the current status of the bridge."""
        signal_exists = os.path.exists(self.signal_path)
        confirm_exists = os.path.exists(self.confirmation_path)

        status = {
            "signal_file_exists": signal_exists,
            "confirmation_file_exists": confirm_exists,
            "signal_path": self.signal_path,
            "confirmation_path": self.confirmation_path,
        }

        if signal_exists:
            status["signal_age_seconds"] = (
                time.time() - os.path.getmtime(self.signal_path)
            )
            status["signal_fresh"] = self.is_signal_fresh()

        return status
