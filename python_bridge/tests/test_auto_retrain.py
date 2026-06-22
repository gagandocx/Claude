"""
=============================================================
  Python ML Bridge - Auto-Retraining Tests
  Tests for weekend retraining scheduler:
    - Day-of-week check logic
    - Minimum interval enforcement
    - Model performance comparison
    - Deploy-only-if-better gating
    - Trade outcome recording
    - Walk-forward validation
=============================================================
"""

import pytest
import os
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RetrainConfig, MODEL_DIR, LOG_DIR
from training.auto_retrain import AutoRetrainer


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def retrain_config():
    """Create test retrain config."""
    return RetrainConfig(
        retrain_day="Saturday",
        min_days_between=7,
        min_improvement_pct=1.0,
        walk_forward_weeks=2,
        max_retrain_attempts=3,
        save_retrain_history=False,  # Don't write files in tests
        retrain_log_file="test_retrain_history.json",
        incorporate_trade_outcomes=True,
        min_trades_for_retrain=5,
    )


@pytest.fixture
def retrainer(retrain_config):
    """Create AutoRetrainer with test config."""
    return AutoRetrainer(retrain_config)


@pytest.fixture
def saturday():
    """A datetime that falls on Saturday."""
    # Find next Saturday from a known date
    d = datetime(2024, 1, 6, 10, 0, 0)  # This is a Saturday
    assert d.strftime("%A") == "Saturday"
    return d


@pytest.fixture
def wednesday():
    """A datetime that falls on Wednesday."""
    d = datetime(2024, 1, 3, 10, 0, 0)  # This is a Wednesday
    assert d.strftime("%A") == "Wednesday"
    return d


# ─────────────────────────────────────────────
#  SHOULD_RETRAIN TESTS
# ─────────────────────────────────────────────
class TestShouldRetrain:
    """Tests for retraining eligibility logic."""

    def test_retrain_on_saturday(self, retrainer, saturday):
        """Test retraining is allowed on Saturday."""
        assert retrainer.should_retrain(saturday) is True

    def test_no_retrain_on_weekday(self, retrainer, wednesday):
        """Test retraining is not allowed on weekdays."""
        assert retrainer.should_retrain(wednesday) is False

    def test_no_retrain_too_soon(self, retrainer, saturday):
        """Test retraining blocked if done less than min_days ago."""
        # Pretend we retrained 3 days ago
        retrainer._last_retrain = saturday - timedelta(days=3)
        assert retrainer.should_retrain(saturday) is False

    def test_retrain_after_interval(self, retrainer, saturday):
        """Test retraining allowed after minimum interval."""
        # Last retrain was 8 days ago (more than min_days_between=7)
        retrainer._last_retrain = saturday - timedelta(days=8)
        assert retrainer.should_retrain(saturday) is True

    def test_retrain_exactly_at_interval(self, retrainer, saturday):
        """Test retraining allowed exactly at minimum interval."""
        retrainer._last_retrain = saturday - timedelta(days=7)
        assert retrainer.should_retrain(saturday) is True

    def test_no_retrain_max_attempts(self, retrainer, saturday):
        """Test retraining blocked after max attempts in session."""
        retrainer._retrain_attempts = 3  # Max is 3
        assert retrainer.should_retrain(saturday) is False

    def test_first_retrain_always_eligible(self, retrainer, saturday):
        """Test first retrain (no previous) is always eligible on retrain day."""
        assert retrainer._last_retrain is None
        assert retrainer.should_retrain(saturday) is True

    def test_custom_retrain_day(self, retrain_config):
        """Test custom retrain day (e.g., Sunday)."""
        retrain_config.retrain_day = "Sunday"
        retrainer = AutoRetrainer(retrain_config)
        sunday = datetime(2024, 1, 7, 10, 0, 0)  # Sunday
        assert sunday.strftime("%A") == "Sunday"
        assert retrainer.should_retrain(sunday) is True

        saturday = datetime(2024, 1, 6, 10, 0, 0)
        assert retrainer.should_retrain(saturday) is False


# ─────────────────────────────────────────────
#  RUN_RETRAIN TESTS
# ─────────────────────────────────────────────
class TestRunRetrain:
    """Tests for the retraining pipeline execution."""

    def test_retrain_deploys_better_model(self, retrainer):
        """Test that better model gets deployed."""
        call_count = [0]

        def mock_train():
            pass  # Simulate training

        def mock_evaluate():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.60  # Old model score
            else:
                return 0.65  # New model score (>1% better)

        result = retrainer.run_retrain(
            train_fn=mock_train,
            evaluate_fn=mock_evaluate
        )
        assert result["success"] is True
        assert result["deployed"] is True
        assert result["improvement_pct"] > 1.0

    def test_retrain_rejects_worse_model(self, retrainer):
        """Test that worse model is NOT deployed."""
        call_count = [0]

        def mock_train():
            pass

        def mock_evaluate():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.60  # Old model score
            else:
                return 0.59  # New model score (worse!)

        result = retrainer.run_retrain(
            train_fn=mock_train,
            evaluate_fn=mock_evaluate
        )
        assert result["success"] is True
        assert result["deployed"] is False
        assert result["improvement_pct"] < 0

    def test_retrain_rejects_marginal_improvement(self, retrainer):
        """Test that marginal improvement (<1%) is rejected."""
        call_count = [0]

        def mock_train():
            pass

        def mock_evaluate():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.60  # Old model
            else:
                return 0.604  # 0.67% improvement (below 1% threshold)

        result = retrainer.run_retrain(
            train_fn=mock_train,
            evaluate_fn=mock_evaluate
        )
        assert result["success"] is True
        assert result["deployed"] is False
        assert result["improvement_pct"] < 1.0

    def test_retrain_handles_training_failure(self, retrainer):
        """Test graceful handling of training errors."""
        def mock_train():
            raise RuntimeError("Training exploded")

        result = retrainer.run_retrain(train_fn=mock_train)
        assert result["success"] is False
        assert "Training failed" in result["reason"]

    def test_retrain_without_evaluate_fn(self, retrainer):
        """Test retraining without evaluation deploys by default."""
        def mock_train():
            pass

        result = retrainer.run_retrain(train_fn=mock_train)
        assert result["deployed"] is True
        assert "no previous model" in result["reason"].lower() or "first" in result["reason"].lower()

    def test_retrain_resets_attempts_on_success(self, retrainer):
        """Test that successful retrain resets attempt counter."""
        retrainer._retrain_attempts = 2

        def mock_train():
            pass

        retrainer.run_retrain(train_fn=mock_train)
        assert retrainer._retrain_attempts == 0

    def test_retrain_updates_last_retrain_time(self, retrainer):
        """Test that successful retrain updates timestamp."""
        assert retrainer._last_retrain is None

        def mock_train():
            pass

        retrainer.run_retrain(train_fn=mock_train)
        assert retrainer._last_retrain is not None
        # Should be recent (within last minute)
        assert (datetime.now() - retrainer._last_retrain).seconds < 60


# ─────────────────────────────────────────────
#  TRADE OUTCOME RECORDING TESTS
# ─────────────────────────────────────────────
class TestTradeOutcomes:
    """Tests for trade outcome recording and retrieval."""

    def test_record_trade_outcome(self, retrainer):
        """Test recording a trade outcome."""
        retrainer.record_trade_outcome(
            trade_id="T001",
            pnl=150.0,
            entry_time="2024-01-05 10:00:00",
            exit_time="2024-01-05 14:30:00",
            direction="BUY"
        )
        assert len(retrainer._trade_outcomes) == 1
        assert retrainer._trade_outcomes[0]["pnl"] == 150.0

    def test_get_outcomes_below_threshold(self, retrainer):
        """Test that outcomes are not returned below minimum threshold."""
        # Record fewer than min_trades_for_retrain (5)
        for i in range(3):
            retrainer.record_trade_outcome(
                f"T{i}", pnl=10.0 * i,
                entry_time="2024-01-01", exit_time="2024-01-01",
                direction="BUY"
            )
        outcomes = retrainer.get_trade_outcomes_for_training()
        assert outcomes == []  # Not enough trades

    def test_get_outcomes_above_threshold(self, retrainer):
        """Test outcomes returned when above threshold."""
        for i in range(6):
            retrainer.record_trade_outcome(
                f"T{i}", pnl=10.0 * i,
                entry_time="2024-01-01", exit_time="2024-01-01",
                direction="BUY"
            )
        outcomes = retrainer.get_trade_outcomes_for_training()
        assert len(outcomes) == 6

    def test_get_outcomes_clears_buffer(self, retrainer):
        """Test that getting outcomes clears the internal buffer."""
        for i in range(6):
            retrainer.record_trade_outcome(
                f"T{i}", pnl=10.0,
                entry_time="2024-01-01", exit_time="2024-01-01",
                direction="BUY"
            )
        retrainer.get_trade_outcomes_for_training()
        assert len(retrainer._trade_outcomes) == 0


# ─────────────────────────────────────────────
#  WALK-FORWARD VALIDATION TESTS
# ─────────────────────────────────────────────
class TestWalkForwardValidation:
    """Tests for walk-forward validation."""

    def test_walk_forward_with_perfect_model(self, retrainer):
        """Test walk-forward with a model that always predicts correctly."""
        # Create mock model that returns one-hot predictions
        class PerfectModel:
            def predict(self, x):
                import torch
                batch_size = x.shape[0]
                preds = torch.zeros(batch_size, 3)
                # Make it predict class 0 for all
                preds[:, 0] = 1.0
                return preds

        model = PerfectModel()
        data = np.random.randn(500, 64, 20)  # 500 bars of data
        labels = np.zeros(500, dtype=np.int64)  # All class 0

        accuracy = retrainer.walk_forward_validate(model, data, labels, weeks=1)
        assert accuracy == pytest.approx(1.0, abs=0.01)

    def test_walk_forward_insufficient_data(self, retrainer):
        """Test walk-forward returns 0 with insufficient data."""
        class DummyModel:
            def predict(self, x):
                import torch
                return torch.zeros(x.shape[0], 3)

        model = DummyModel()
        # Too few bars for 2 weeks (need 336, only have 100)
        data = np.random.randn(100, 64, 20)
        labels = np.zeros(100, dtype=np.int64)

        accuracy = retrainer.walk_forward_validate(model, data, labels)
        assert accuracy == 0.0

    def test_walk_forward_uses_recent_data(self, retrainer):
        """Test that walk-forward uses the most recent data."""
        class TrackingModel:
            def __init__(self):
                self.last_input_size = 0

            def predict(self, x):
                import torch
                self.last_input_size = x.shape[0]
                return torch.zeros(x.shape[0], 3)

        model = TrackingModel()
        data = np.random.randn(1000, 64, 20)
        labels = np.zeros(1000, dtype=np.int64)

        retrainer.walk_forward_validate(model, data, labels, weeks=1)
        # 1 week = 168 bars
        assert model.last_input_size == 168


# ─────────────────────────────────────────────
#  STATUS AND HISTORY TESTS
# ─────────────────────────────────────────────
class TestStatusAndHistory:
    """Tests for status reporting and history tracking."""

    def test_get_status(self, retrainer):
        """Test status dictionary contains expected keys."""
        status = retrainer.get_status()
        assert "last_retrain" in status
        assert "retrain_history_count" in status
        assert "trade_outcomes_pending" in status
        assert "retrain_attempts_this_session" in status
        assert "next_retrain_eligible" in status

    def test_history_accumulates(self, retrainer):
        """Test that retrain history grows with each attempt."""
        def mock_train():
            pass

        retrainer.run_retrain(train_fn=mock_train)
        assert len(retrainer._retrain_history) == 1

        retrainer._last_retrain = None  # Allow another
        retrainer.run_retrain(train_fn=mock_train)
        assert len(retrainer._retrain_history) == 2

    def test_save_and_load_history(self, retrain_config, tmp_path):
        """Test history persistence to JSON file."""
        retrain_config.save_retrain_history = True
        # Override LOG_DIR for test
        log_dir = str(tmp_path)

        with patch('training.auto_retrain.LOG_DIR', log_dir):
            retrainer = AutoRetrainer(retrain_config)

            def mock_train():
                pass

            retrainer.run_retrain(train_fn=mock_train)

            # Check file was created
            log_path = os.path.join(log_dir, retrain_config.retrain_log_file)
            assert os.path.exists(log_path)

            # Load and verify
            with open(log_path, "r") as f:
                data = json.load(f)
            assert "history" in data
            assert len(data["history"]) == 1
            assert "last_retrain" in data
