"""
=============================================================
  Python ML Bridge - Bridge Communication Tests
  Tests for CSV signal writing/reading, signal format
  validation, and file bridge operations.
=============================================================
"""

import pytest
import os
import tempfile
import time
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.bridge import MT5Bridge, SIGNAL_HEADERS, CONFIRMATION_HEADERS
from strategies.signal_generator import TradeSignal


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def bridge(temp_dir):
    """Create bridge with temp file paths."""
    signal_path = os.path.join(temp_dir, "test_signal.csv")
    confirm_path = os.path.join(temp_dir, "test_confirm.csv")
    return MT5Bridge(signal_path=signal_path, confirmation_path=confirm_path)


@pytest.fixture
def sample_signal():
    """Create a sample trade signal."""
    return TradeSignal(
        timestamp="2024-01-15 14:30:00",
        symbol="XAUUSD",
        action="BUY",
        confidence=0.8532,
        sl_pips=150.5,
        tp_pips=251.0,
        lot_size=0.10,
        model_name="transformer",
        regime="trending"
    )


# ─────────────────────────────────────────────
#  SIGNAL WRITING TESTS
# ─────────────────────────────────────────────
class TestSignalWriting:
    """Tests for writing signals to CSV."""

    def test_write_signal_creates_file(self, bridge, sample_signal):
        """Test that writing a signal creates the CSV file."""
        result = bridge.write_signal(sample_signal)
        assert result is True
        assert os.path.exists(bridge.signal_path)

    def test_write_signal_content(self, bridge, sample_signal):
        """Test that written signal has correct content."""
        bridge.write_signal(sample_signal)
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["symbol"] == "XAUUSD"
        assert signal["action"] == "BUY"
        assert signal["model_name"] == "transformer"
        assert signal["regime"] == "trending"

    def test_write_signal_confidence_format(self, bridge, sample_signal):
        """Test confidence is written with 4 decimal places."""
        bridge.write_signal(sample_signal)
        signal = bridge.read_signal()
        assert signal["confidence"] == "0.8532"

    def test_write_signal_lot_format(self, bridge, sample_signal):
        """Test lot size is written with 2 decimal places."""
        bridge.write_signal(sample_signal)
        signal = bridge.read_signal()
        assert signal["lot_size"] == "0.10"

    def test_write_overwrites_previous(self, bridge, sample_signal):
        """Test that new signal overwrites the old one."""
        bridge.write_signal(sample_signal)

        new_signal = TradeSignal(
            timestamp="2024-01-15 14:31:00",
            symbol="XAUUSD",
            action="SELL",
            confidence=0.7200,
            sl_pips=120.0,
            tp_pips=200.0,
            lot_size=0.05,
            model_name="lstm",
            regime="volatile"
        )
        bridge.write_signal(new_signal)

        signal = bridge.read_signal()
        assert signal["action"] == "SELL"
        assert signal["model_name"] == "lstm"


# ─────────────────────────────────────────────
#  SIGNAL READING TESTS
# ─────────────────────────────────────────────
class TestSignalReading:
    """Tests for reading signals from CSV."""

    def test_read_nonexistent_file(self, bridge):
        """Test reading when no signal file exists."""
        signal = bridge.read_signal()
        assert signal is None

    def test_read_after_write(self, bridge, sample_signal):
        """Test read returns what was written."""
        bridge.write_signal(sample_signal)
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["timestamp"] == "2024-01-15 14:30:00"
        assert signal["symbol"] == "XAUUSD"
        assert signal["action"] == "BUY"

    def test_read_all_fields_present(self, bridge, sample_signal):
        """Test that all required fields are present in read signal."""
        bridge.write_signal(sample_signal)
        signal = bridge.read_signal()
        for header in SIGNAL_HEADERS:
            assert header in signal, f"Missing field: {header}"


# ─────────────────────────────────────────────
#  SIGNAL FRESHNESS TESTS
# ─────────────────────────────────────────────
class TestSignalFreshness:
    """Tests for signal age checking."""

    def test_fresh_signal(self, bridge, sample_signal):
        """Test that newly written signal is fresh."""
        bridge.write_signal(sample_signal)
        assert bridge.is_signal_fresh(max_age_seconds=60)

    def test_no_signal_not_fresh(self, bridge):
        """Test that missing file is not fresh."""
        assert not bridge.is_signal_fresh()

    def test_signal_expires(self, bridge, sample_signal):
        """Test that signal with very short max age expires."""
        bridge.write_signal(sample_signal)
        # With max_age=0, signal is immediately stale
        time.sleep(0.1)
        assert not bridge.is_signal_fresh(max_age_seconds=0)


# ─────────────────────────────────────────────
#  FILE OPERATIONS TESTS
# ─────────────────────────────────────────────
class TestFileOperations:
    """Tests for file management operations."""

    def test_clear_signal(self, bridge, sample_signal):
        """Test clearing the signal file."""
        bridge.write_signal(sample_signal)
        assert os.path.exists(bridge.signal_path)
        bridge.clear_signal()
        assert not os.path.exists(bridge.signal_path)

    def test_clear_nonexistent_signal(self, bridge):
        """Test clearing when no file exists (should not error)."""
        bridge.clear_signal()  # Should not raise

    def test_write_heartbeat(self, bridge):
        """Test heartbeat file creation."""
        bridge.write_heartbeat()
        heartbeat_path = os.path.join(
            os.path.dirname(bridge.signal_path),
            "python_bridge_heartbeat.txt"
        )
        assert os.path.exists(heartbeat_path)

    def test_bridge_status(self, bridge, sample_signal):
        """Test getting bridge status."""
        status = bridge.get_bridge_status()
        assert status["signal_file_exists"] is False

        bridge.write_signal(sample_signal)
        status = bridge.get_bridge_status()
        assert status["signal_file_exists"] is True
        assert status["signal_fresh"] is True


# ─────────────────────────────────────────────
#  CONFIRMATION READING TESTS
# ─────────────────────────────────────────────
class TestConfirmations:
    """Tests for reading MT5 execution confirmations."""

    def test_read_empty_confirmations(self, bridge):
        """Test reading when no confirmation file exists."""
        confirmations = bridge.read_confirmations()
        assert confirmations == []

    def test_read_confirmations(self, bridge):
        """Test reading confirmation CSV."""
        # Write a mock confirmation file
        import csv
        with open(bridge.confirmation_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CONFIRMATION_HEADERS)
            writer.writerow([
                "2024-01-15 14:30:05", "12345", "XAUUSD", "BUY",
                "0.10", "2050.50", "2043.00", "2062.75", "FILLED", "0.00"
            ])

        confirmations = bridge.read_confirmations()
        assert len(confirmations) == 1
        assert confirmations[0]["ticket"] == "12345"
        assert confirmations[0]["status"] == "FILLED"

    def test_clear_confirmations(self, bridge):
        """Test clearing confirmation file."""
        # Create file first
        with open(bridge.confirmation_path, "w") as f:
            f.write("test")
        assert os.path.exists(bridge.confirmation_path)

        bridge.clear_confirmations()
        assert not os.path.exists(bridge.confirmation_path)


# ─────────────────────────────────────────────
#  CSV FORMAT VALIDATION TESTS
# ─────────────────────────────────────────────
class TestCSVFormat:
    """Tests for CSV format compliance."""

    def test_signal_csv_has_header(self, bridge, sample_signal):
        """Test that signal CSV includes header row."""
        bridge.write_signal(sample_signal)
        with open(bridge.signal_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2  # Header + 1 data row
        header = lines[0].strip().split(",")
        assert header == SIGNAL_HEADERS

    def test_signal_csv_field_count(self, bridge, sample_signal):
        """Test that data row has correct number of fields."""
        bridge.write_signal(sample_signal)
        with open(bridge.signal_path, "r") as f:
            lines = f.readlines()
        data_fields = lines[1].strip().split(",")
        assert len(data_fields) == len(SIGNAL_HEADERS)

    def test_sell_signal_format(self, bridge):
        """Test SELL signal CSV format."""
        signal = TradeSignal(
            timestamp="2024-06-20 09:15:00",
            symbol="XAUUSD",
            action="SELL",
            confidence=0.7100,
            sl_pips=120.0,
            tp_pips=200.0,
            lot_size=0.05,
            model_name="gradient_boost",
            regime="ranging"
        )
        bridge.write_signal(signal)
        result = bridge.read_signal()
        assert result["action"] == "SELL"
        assert result["confidence"] == "0.7100"
        assert result["sl_pips"] == "120.0"
        assert result["tp_pips"] == "200.0"
        assert result["lot_size"] == "0.05"


# ─────────────────────────────────────────────
#  RETRY LOGIC TESTS
# ─────────────────────────────────────────────
class TestWriteSignalRetry:
    """Tests for write_signal() retry logic on PermissionError/OSError."""

    def test_retry_succeeds_on_second_attempt(self, bridge, sample_signal):
        """Test that write_signal retries and succeeds when os.replace fails once."""
        call_count = [0]
        original_replace = os.replace

        def mock_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("File in use by another process")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            result = bridge.write_signal(sample_signal)

        assert result is True
        assert call_count[0] == 2
        # Verify signal was actually written
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["action"] == "BUY"
        assert signal["symbol"] == "XAUUSD"

    def test_retry_succeeds_on_third_attempt(self, bridge, sample_signal):
        """Test that write_signal retries up to 3 times and succeeds on third."""
        call_count = [0]
        original_replace = os.replace

        def mock_replace(src, dst):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise PermissionError("File in use by another process")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            result = bridge.write_signal(sample_signal)

        assert result is True
        assert call_count[0] == 3
        # Verify signal was actually written
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["action"] == "BUY"

    def test_fallback_after_all_retries_fail(self, bridge, sample_signal):
        """Test fallback to direct write when all os.replace retries fail."""
        def mock_replace(src, dst):
            raise PermissionError("File permanently locked")

        with patch("os.replace", side_effect=mock_replace):
            result = bridge.write_signal(sample_signal)

        # Should succeed via fallback (unlink+rename or direct write)
        assert result is True
        # Verify signal was actually written via fallback
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["action"] == "BUY"
        assert signal["symbol"] == "XAUUSD"
        assert signal["confidence"] == "0.8532"

    def test_retry_handles_oserror(self, bridge, sample_signal):
        """Test that retry logic also catches OSError (not just PermissionError)."""
        call_count = [0]
        original_replace = os.replace

        def mock_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Filesystem busy")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            result = bridge.write_signal(sample_signal)

        assert result is True
        assert call_count[0] == 2

    def test_retry_logs_warnings(self, bridge, sample_signal, caplog):
        """Test that retry attempts are logged as warnings."""
        import logging
        call_count = [0]
        original_replace = os.replace

        def mock_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("File in use")
            return original_replace(src, dst)

        with caplog.at_level(logging.WARNING, logger="PythonBridge"):
            with patch("os.replace", side_effect=mock_replace):
                result = bridge.write_signal(sample_signal)

        assert result is True
        # Check that a warning was logged about the retry
        assert any("os.replace()" in record.message and "attempt" in record.message
                   for record in caplog.records)

    def test_no_retry_on_success(self, bridge, sample_signal):
        """Test that no retry happens when os.replace succeeds immediately."""
        call_count = [0]
        original_replace = os.replace

        def mock_replace(src, dst):
            call_count[0] += 1
            return original_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            result = bridge.write_signal(sample_signal)

        assert result is True
        assert call_count[0] == 1

    def test_fallback_direct_write_when_unlink_also_fails(self, bridge, sample_signal):
        """Test that direct write fallback is used when unlink+rename also fails."""
        replace_calls = [0]
        unlink_calls = [0]
        original_unlink = os.unlink

        def mock_replace(src, dst):
            raise PermissionError("File permanently locked")

        def mock_unlink(path):
            unlink_calls[0] += 1
            # Only raise for the signal file (the fallback unlink), not temp cleanup
            if "signal_" not in os.path.basename(path):
                raise PermissionError("Cannot delete locked file")
            return original_unlink(path)

        # First write a signal so the file exists (for fallback unlink to target)
        bridge.write_signal(sample_signal)

        with patch("os.replace", side_effect=mock_replace):
            with patch("os.unlink", side_effect=mock_unlink):
                result = bridge.write_signal(sample_signal)

        # Should still succeed via direct write fallback
        assert result is True
        signal = bridge.read_signal()
        assert signal is not None
        assert signal["action"] == "BUY"
