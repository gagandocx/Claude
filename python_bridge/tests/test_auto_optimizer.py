"""
=============================================================
  Tests for Auto-Optimizer Module
  Tests self-tuning parameter optimization, trade recording,
  optimization cycles, rollback, state persistence, and
  integration with signal_generator.
=============================================================
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.auto_optimizer import AutoOptimizer
from config.settings import AutoOptimizerConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def optimizer(temp_dir):
    """Create an AutoOptimizer with temp state directory."""
    config = AutoOptimizerConfig(
        optimize_frequency=5,  # Low threshold for testing
        min_trades_before_tuning=5,
    )
    return AutoOptimizer(config=config, state_dir=temp_dir)


@pytest.fixture
def optimizer_50(temp_dir):
    """Create an AutoOptimizer with default 50-trade threshold."""
    config = AutoOptimizerConfig()
    return AutoOptimizer(config=config, state_dir=temp_dir)


def _make_trade(pnl, session="london", confidence=0.5, sl_distance=3.0,
                momentum_lookback=5, rsi_at_entry=50.0, trail_tier="medium",
                cooldown_used=2.0, max_positions_at_entry=5, direction="BUY"):
    """Helper to create a trade context dict."""
    return {
        "session": session,
        "confidence": confidence,
        "momentum_lookback": momentum_lookback,
        "sl_distance": sl_distance,
        "result_pnl": pnl,
        "direction": direction,
        "rsi_at_entry": rsi_at_entry,
        "trail_tier": trail_tier,
        "cooldown_used": cooldown_used,
        "max_positions_at_entry": max_positions_at_entry,
        "entry_time": "2024-01-01T10:00:00",
        "exit_time": "2024-01-01T10:05:00",
    }


class TestAutoOptimizerInit:
    """Tests for initialization and default values."""

    def test_default_params(self, optimizer):
        """Test that default parameters are set correctly."""
        params = optimizer.get_current_params()
        assert params["sl_distance"] == 0.6
        assert params["min_confidence"] == 0.25
        assert params["momentum_lookback"] == 8
        assert params["rsi_overbought"] == 62
        assert params["rsi_oversold"] == 38
        assert params["cooldown_seconds"] == 2
        assert params["max_positions"] == 1

    def test_default_session_multipliers(self, optimizer):
        """Test that session multipliers have correct defaults."""
        params = optimizer.get_current_params()
        assert params["session_multipliers"]["asian"] == 1.0
        assert params["session_multipliers"]["london"] == 1.2
        assert params["session_multipliers"]["newyork"] == 1.0
        assert params["session_multipliers"]["overlap"] == 1.2
        assert params["session_multipliers"]["off_session"] == 1.0

    def test_default_trail_distances(self, optimizer):
        """Test that trail distances have correct defaults."""
        params = optimizer.get_current_params()
        assert params["trail_distances"]["tight"] == 0.5
        assert params["trail_distances"]["medium"] == 1.0
        assert params["trail_distances"]["wide"] == 2.0

    def test_is_enabled(self, optimizer):
        """Test that optimizer reports enabled status."""
        assert optimizer.is_enabled is True

    def test_disabled_optimizer(self, temp_dir):
        """Test disabled optimizer does not record or optimize."""
        config = AutoOptimizerConfig(enabled=False, optimize_frequency=5,
                                     min_trades_before_tuning=5)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        assert opt.is_enabled is False
        opt.record_trade(_make_trade(1.0))
        assert opt.trade_count == 0


class TestTradeRecording:
    """Tests for trade recording functionality."""

    def test_record_single_trade(self, optimizer):
        """Test recording a single trade."""
        optimizer.record_trade(_make_trade(1.5))
        assert optimizer.trade_count == 1

    def test_record_multiple_trades(self, optimizer):
        """Test recording multiple trades."""
        for i in range(10):
            optimizer.record_trade(_make_trade(float(i)))
        assert optimizer.trade_count == 10

    def test_trade_context_preserved(self, optimizer):
        """Test that trade context is preserved in history."""
        trade = _make_trade(2.5, session="asian", confidence=0.8)
        optimizer.record_trade(trade)
        assert optimizer._trades[0]["session"] == "asian"
        assert optimizer._trades[0]["confidence"] == 0.8
        assert optimizer._trades[0]["result_pnl"] == 2.5

    def test_trades_since_optimize_counter(self, optimizer):
        """Test that trades_since_optimize counter increments."""
        optimizer.record_trade(_make_trade(1.0))
        optimizer.record_trade(_make_trade(2.0))
        assert optimizer._trades_since_optimize == 2

    def test_record_estimated_trade(self, optimizer):
        """Test that record_estimated_trade records trade with estimated trail_tier."""
        trade = _make_trade(1.5, session="london", confidence=0.7)
        optimizer.record_estimated_trade(trade)
        assert optimizer.trade_count == 1
        assert optimizer._trades[0]["trail_tier"] == "estimated"
        assert optimizer._trades[0]["result_pnl"] == 1.5
        assert optimizer._trades[0]["session"] == "london"

    def test_record_estimated_trade_overrides_trail_tier(self, optimizer):
        """Test that record_estimated_trade always sets trail_tier to 'estimated'."""
        trade = _make_trade(2.0, trail_tier="tight")
        optimizer.record_estimated_trade(trade)
        assert optimizer._trades[0]["trail_tier"] == "estimated"


class TestOptimization:
    """Tests for the optimization logic."""

    def test_optimize_insufficient_trades(self, optimizer_50):
        """Test optimize returns insufficient_trades when not enough data."""
        for i in range(3):
            optimizer_50._trades.append(_make_trade(1.0))
        result = optimizer_50.optimize()
        assert result["status"] == "insufficient_trades"

    def test_optimize_triggers_automatically(self, optimizer):
        """Test that optimization triggers after optimize_frequency trades."""
        # Record 5 trades (the threshold)
        for i in range(5):
            optimizer.record_trade(_make_trade(1.0 if i % 2 == 0 else -0.5))
        # Should have triggered optimization
        assert optimizer.cycle_count == 1

    def test_optimize_resets_counter(self, optimizer):
        """Test that trades_since_optimize resets after optimization."""
        for i in range(5):
            optimizer.record_trade(_make_trade(1.0))
        assert optimizer._trades_since_optimize == 0

    def test_sl_distance_shifts_toward_winner(self, temp_dir):
        """Test SL distance shifts toward the value with highest win rate."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Add trades: SL=3.0 wins most, SL=7.0 loses most
        for _ in range(6):
            opt._trades.append(_make_trade(2.0, sl_distance=3.0))
        for _ in range(4):
            opt._trades.append(_make_trade(-1.0, sl_distance=7.0))

        result = opt.optimize()
        # SL should shift toward 3.0 from default 5.0
        assert opt.get_current_params()["sl_distance"] < 5.0

    def test_momentum_lookback_shifts(self, temp_dir):
        """Test momentum lookback shifts toward best performer."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Lookback 10 is best
        for _ in range(7):
            opt._trades.append(_make_trade(3.0, momentum_lookback=10))
        for _ in range(3):
            opt._trades.append(_make_trade(-1.0, momentum_lookback=5))

        opt.optimize()
        # Should shift from default 8 toward 10 (max 1 step = 9)
        assert opt.get_current_params()["momentum_lookback"] == 9

    def test_confidence_raises_when_low_conf_loses(self, temp_dir):
        """Test confidence threshold raises when low-confidence trades lose."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Low confidence trades lose (near current threshold 0.25)
        for _ in range(7):
            opt._trades.append(_make_trade(-1.0, confidence=0.22))
        # High confidence trades win
        for _ in range(3):
            opt._trades.append(_make_trade(2.0, confidence=0.50))

        opt.optimize()
        # Confidence should increase from 0.25
        assert opt.get_current_params()["min_confidence"] > 0.25

    def test_cooldown_reduces_when_fast_profitable(self, temp_dir):
        """Test cooldown reduces when fast entries are profitable."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        # Set cooldown to 60 so there's room to reduce
        opt._params["cooldown_seconds"] = 60

        # Fast entries (cooldown <= 60) are profitable
        for _ in range(8):
            opt._trades.append(_make_trade(1.5, cooldown_used=50.0))
        for _ in range(2):
            opt._trades.append(_make_trade(-0.5, cooldown_used=150.0))

        opt.optimize()
        # Cooldown was 60, fast entries profitable -> should reduce by 1
        assert opt.get_current_params()["cooldown_seconds"] == 59

    def test_max_positions_reduces_on_high_drawdown(self, temp_dir):
        """Test max positions reduces when drawdown is high."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        # Set positions to 3 so it can be reduced
        opt._params["max_positions"] = 3

        # Large losses creating high drawdown
        for _ in range(8):
            opt._trades.append(_make_trade(-3.0, max_positions_at_entry=3))
        for _ in range(2):
            opt._trades.append(_make_trade(1.0, max_positions_at_entry=3))

        opt.optimize()
        # Should reduce positions due to losses
        assert opt.get_current_params()["max_positions"] < 3

    def test_session_multipliers_increase_for_profitable_session(self, temp_dir):
        """Test session multiplier increases for profitable sessions."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # London session very profitable
        for _ in range(10):
            opt._trades.append(_make_trade(2.0, session="london"))

        opt.optimize()
        # London multiplier should increase from 1.2
        assert opt.get_current_params()["session_multipliers"]["london"] > 1.2

    def test_params_clamped_to_ranges(self, temp_dir):
        """Test parameters never exceed configured ranges."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        # Set momentum to max already
        opt._params["momentum_lookback"] = 10

        # All trades favor lookback 10
        for _ in range(10):
            opt._trades.append(_make_trade(5.0, momentum_lookback=10))

        opt.optimize()
        # Should not exceed max range (10)
        assert opt.get_current_params()["momentum_lookback"] <= 10

    def test_never_jumps_more_than_one_step(self, temp_dir):
        """Test that momentum lookback shifts at most 1 bar per cycle."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        # Current default is 8, best is 5
        for _ in range(10):
            opt._trades.append(_make_trade(3.0, momentum_lookback=5))

        opt.optimize()
        # Should shift from 8 to 7 (max 1 step)
        assert opt.get_current_params()["momentum_lookback"] == 7


class TestRollback:
    """Tests for rollback functionality."""

    def test_rollback_reverts_params(self, temp_dir):
        """Test that rollback restores previous parameters."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Record initial params
        initial_params = opt.get_current_params()

        # Force optimization
        for _ in range(5):
            opt._trades.append(_make_trade(2.0, sl_distance=2.0))
        opt.optimize()

        # Params changed
        assert opt.get_current_params() != initial_params

        # Rollback
        opt.rollback()
        assert opt.get_current_params() == initial_params

    def test_rollback_when_performance_drops(self, temp_dir):
        """Test automatic rollback when post-optimize performance drops 20%."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5,
                                     rollback_threshold=0.20)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Good trades before optimization (80% win rate)
        for _ in range(4):
            opt._trades.append(_make_trade(1.0))
        opt._trades.append(_make_trade(-0.5))

        # Record the pre-optimize SL value
        initial_sl = opt.get_current_params()["sl_distance"]

        # Force optimize (will set rollback_pending)
        opt.optimize()
        assert opt._rollback_pending is True

        # Now simulate bad post-optimize trades (0% win rate)
        for _ in range(5):
            opt._post_optimize_trades.append(_make_trade(-2.0))

        # Manually check rollback (in production this fires after optimize_frequency trades)
        opt._check_rollback()

        # Rollback should have triggered
        assert opt._rollback_pending is False
        # SL should be back to initial value
        assert opt.get_current_params()["sl_distance"] == initial_sl

    def test_no_rollback_when_performance_ok(self, temp_dir):
        """Test no rollback when post-optimize performance is acceptable."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5,
                                     rollback_threshold=0.20)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Trades before optimization (60% win rate)
        for _ in range(3):
            opt._trades.append(_make_trade(1.0))
        for _ in range(2):
            opt._trades.append(_make_trade(-0.5))

        opt.optimize()
        assert opt._rollback_pending is True

        # Simulate post-optimize trades with similar quality (60% WR)
        for _ in range(3):
            opt._post_optimize_trades.append(_make_trade(1.0))
        for _ in range(2):
            opt._post_optimize_trades.append(_make_trade(-0.5))

        # Manually check rollback
        opt._check_rollback()

        # Should not have rolled back (performance is acceptable)
        assert opt._rollback_pending is False
        # _prev_params should be cleared since performance was OK
        assert opt._prev_params is None

    def test_rollback_with_no_prev_params(self, optimizer):
        """Test rollback does nothing when no previous params exist."""
        optimizer._prev_params = None
        optimizer.rollback()  # Should not raise
        assert optimizer.get_current_params() is not None


class TestStatePersistence:
    """Tests for save/load state functionality."""

    def test_save_creates_file(self, optimizer, temp_dir):
        """Test that save_state creates a JSON file."""
        optimizer.record_trade(_make_trade(1.0))
        optimizer.save_state()
        state_file = os.path.join(temp_dir, "auto_optimizer_state.json")
        assert os.path.exists(state_file)

    def test_load_restores_params(self, temp_dir):
        """Test that load_state restores saved parameters."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt1 = AutoOptimizer(config=config, state_dir=temp_dir)
        opt1._params["sl_distance"] = 2.5
        opt1._params["min_confidence"] = 0.25
        opt1._cycle_count = 3
        opt1.save_state()

        # Create new optimizer, should load saved state
        opt2 = AutoOptimizer(config=config, state_dir=temp_dir)
        assert opt2.get_current_params()["sl_distance"] == 2.5
        assert opt2.get_current_params()["min_confidence"] == 0.25
        assert opt2.cycle_count == 3

    def test_load_restores_trades(self, temp_dir):
        """Test that trades are preserved across restarts."""
        config = AutoOptimizerConfig(optimize_frequency=50, min_trades_before_tuning=50)
        opt1 = AutoOptimizer(config=config, state_dir=temp_dir)
        for i in range(10):
            opt1.record_trade(_make_trade(float(i)))
        opt1.save_state()

        opt2 = AutoOptimizer(config=config, state_dir=temp_dir)
        assert opt2.trade_count == 10

    def test_load_nonexistent_file(self, temp_dir):
        """Test load_state handles missing file gracefully."""
        config = AutoOptimizerConfig()
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        result = opt.load_state()
        assert result is False

    def test_state_survives_restart(self, temp_dir):
        """Test full workflow: record, optimize, save, reload."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt1 = AutoOptimizer(config=config, state_dir=temp_dir)

        # Record trades and trigger optimization
        for _ in range(5):
            opt1.record_trade(_make_trade(2.0, sl_distance=2.0))

        assert opt1.cycle_count == 1
        params_after_opt = opt1.get_current_params()

        # Simulate restart
        opt2 = AutoOptimizer(config=config, state_dir=temp_dir)
        assert opt2.cycle_count == 1
        assert opt2.get_current_params() == params_after_opt


class TestRSIOptimization:
    """Tests for RSI overbought/oversold optimization."""

    def test_rsi_ob_raises_when_filter_too_loose(self, temp_dir):
        """Test RSI OB level raises when filtered trades still lose."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Trades near overbought that lose (RSI near current OB=62)
        for _ in range(10):
            opt._trades.append(_make_trade(-1.0, rsi_at_entry=60))

        opt.optimize()
        # OB should rise from 62
        assert opt.get_current_params()["rsi_overbought"] >= 62

    def test_rsi_os_lowers_when_filter_too_loose(self, temp_dir):
        """Test RSI OS level lowers when filtered trades still lose."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Trades near oversold that lose (RSI near current OS=38)
        for _ in range(10):
            opt._trades.append(_make_trade(-1.0, rsi_at_entry=40))

        opt.optimize()
        # OS should lower from 38
        assert opt.get_current_params()["rsi_oversold"] <= 38


class TestTrailDistanceOptimization:
    """Tests for trailing distance optimization."""

    def test_trail_widens_when_profitable(self, temp_dir):
        """Test trail distance widens for profitable tiers."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        for _ in range(10):
            opt._trades.append(_make_trade(3.0, trail_tier="wide"))

        opt.optimize()
        # Wide trail profitable -> widen further
        assert opt.get_current_params()["trail_distances"]["wide"] > 2.0

    def test_trail_tightens_when_losing(self, temp_dir):
        """Test trail distance tightens for losing tiers."""
        config = AutoOptimizerConfig(optimize_frequency=10, min_trades_before_tuning=10)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        for _ in range(10):
            opt._trades.append(_make_trade(-2.0, trail_tier="tight"))

        opt.optimize()
        # Tight trail losing -> tighten further
        assert opt.get_current_params()["trail_distances"]["tight"] < 0.5


class TestGetCurrentParams:
    """Tests for get_current_params method."""

    def test_returns_copy(self, optimizer):
        """Test get_current_params returns a copy, not reference."""
        params1 = optimizer.get_current_params()
        params1["sl_distance"] = 999.0
        params2 = optimizer.get_current_params()
        assert params2["sl_distance"] != 999.0

    def test_all_keys_present(self, optimizer):
        """Test that all expected keys are in the returned params."""
        params = optimizer.get_current_params()
        expected_keys = [
            "sl_distance", "session_multipliers", "min_confidence",
            "momentum_lookback", "rsi_overbought", "rsi_oversold",
            "trail_distances", "cooldown_seconds", "max_positions"
        ]
        for key in expected_keys:
            assert key in params, f"Missing key: {key}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_optimize_with_no_sl_data(self, temp_dir):
        """Test optimization gracefully handles trades without SL data."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        # Trades without sl_distance key
        for _ in range(5):
            trade = {"result_pnl": 1.0, "session": "london", "confidence": 0.5}
            opt._trades.append(trade)

        result = opt.optimize()
        assert result["status"] == "optimized"
        assert result["results"]["sl_distance"]["status"] == "no_data"

    def test_optimize_with_empty_trades(self, temp_dir):
        """Test optimize with empty trade list returns insufficient."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        result = opt.optimize()
        assert result["status"] == "insufficient_trades"

    def test_cycle_count_increments(self, temp_dir):
        """Test cycle count increments with each optimization."""
        config = AutoOptimizerConfig(optimize_frequency=5, min_trades_before_tuning=5)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)

        for _ in range(5):
            opt.record_trade(_make_trade(1.0))
        assert opt.cycle_count == 1

        for _ in range(5):
            opt.record_trade(_make_trade(1.0))
        assert opt.cycle_count == 2

    def test_disabled_optimize(self, temp_dir):
        """Test optimize returns disabled status when disabled."""
        config = AutoOptimizerConfig(enabled=False)
        opt = AutoOptimizer(config=config, state_dir=temp_dir)
        result = opt.optimize()
        assert result["status"] == "disabled"
