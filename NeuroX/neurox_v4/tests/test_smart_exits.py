"""
=============================================================
  Python ML Bridge - Smart Exit Manager Tests
  Tests for AI-driven exit management:
    - Dynamic trailing stop computation
    - Partial close logic at R-multiples
    - Confidence-based stop tightening
    - Rule-based overrides (max hold, min confidence)
    - ExitDecision output format
=============================================================
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SmartExitConfig, RLConfig
from strategies.smart_exits import SmartExitManager, ExitDecision
from models.rl_agent import PositionState, ExitAction


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def exit_config():
    """Create test exit config."""
    return SmartExitConfig(
        trailing_atr_mult_tight=0.5,
        trailing_atr_mult_wide=2.0,
        confidence_decay_threshold=0.3,
        confidence_strong_threshold=0.6,
        partial_close_at_2r=True,
        partial_close_pct=0.5,
        partial_close_at_3r=True,
        partial_close_3r_pct=0.25,
        max_hold_bars=100,
        break_even_at_1r=True,
        confidence_time_decay=0.995,
        min_confidence_to_hold=0.15,
    )


@pytest.fixture
def rl_config():
    """Create test RL config."""
    return RLConfig(
        replay_buffer_size=100,
        batch_size=8,
        hidden_size=32,
        state_size=20,
        num_actions=5,
        min_replay_size=16,
    )


@pytest.fixture
def manager(exit_config, rl_config):
    """Create SmartExitManager with test configs."""
    return SmartExitManager(config=exit_config, rl_config=rl_config)


@pytest.fixture
def profitable_long_position():
    """Long position at 2R profit."""
    return PositionState(
        direction=1,
        unrealized_pnl=6.0,
        unrealized_pnl_atr=2.0,
        hold_bars=20,
        entry_price=2050.0,
        current_price=2056.0,
        atr=3.0,
        confidence=0.7,
        initial_confidence=0.8,
        sl_distance_atr=1.5,
        tp_distance_atr=2.5,
        max_favorable=7.0,
        max_adverse=-1.0,
        partial_closed_pct=0.0,
        regime_changed=False,
        ticket="T001"
    )


@pytest.fixture
def losing_position():
    """Position with significant unrealized loss."""
    return PositionState(
        direction=1,
        unrealized_pnl=-8.0,
        unrealized_pnl_atr=-2.5,
        hold_bars=30,
        entry_price=2050.0,
        current_price=2042.0,
        atr=3.0,
        confidence=0.3,
        initial_confidence=0.7,
        sl_distance_atr=0.5,
        tp_distance_atr=4.0,
        max_favorable=1.0,
        max_adverse=-8.0,
        partial_closed_pct=0.0,
        regime_changed=False,
        ticket="T002"
    )


@pytest.fixture
def max_hold_position():
    """Position that has exceeded max hold time."""
    return PositionState(
        direction=-1,
        unrealized_pnl=2.0,
        unrealized_pnl_atr=0.7,
        hold_bars=110,  # Over max_hold_bars=100
        entry_price=2060.0,
        current_price=2058.0,
        atr=3.0,
        confidence=0.5,
        initial_confidence=0.6,
        sl_distance_atr=1.0,
        tp_distance_atr=2.0,
        max_favorable=4.0,
        max_adverse=-1.0,
        partial_closed_pct=0.0,
        regime_changed=False,
        ticket="T003"
    )


# ─────────────────────────────────────────────
#  TRAILING STOP TESTS
# ─────────────────────────────────────────────
class TestTrailingStop:
    """Tests for dynamic trailing stop computation."""

    def test_high_confidence_wide_trail(self, manager):
        """Test high confidence produces wide trailing stop."""
        sl = manager.compute_trailing_stop(
            entry_price=2050.0,
            current_price=2060.0,
            atr=3.0,
            direction=1,
            confidence=0.8  # High confidence
        )
        # Wide trail: 2.0 * ATR = 6.0, so SL = 2060 - 6 = 2054
        assert sl == pytest.approx(2054.0, abs=0.1)

    def test_low_confidence_tight_trail(self, manager):
        """Test low confidence produces tight trailing stop."""
        sl = manager.compute_trailing_stop(
            entry_price=2050.0,
            current_price=2060.0,
            atr=3.0,
            direction=1,
            confidence=0.2  # Low confidence
        )
        # Tight trail: 0.5 * ATR = 1.5, so SL = 2060 - 1.5 = 2058.5
        assert sl == pytest.approx(2058.5, abs=0.1)

    def test_medium_confidence_interpolated(self, manager):
        """Test medium confidence interpolates between tight and wide."""
        sl = manager.compute_trailing_stop(
            entry_price=2050.0,
            current_price=2060.0,
            atr=3.0,
            direction=1,
            confidence=0.45  # Between 0.3 and 0.6 thresholds
        )
        # Should be between tight (2058.5) and wide (2054.0)
        assert 2054.0 < sl < 2058.5

    def test_short_position_trailing_stop(self, manager):
        """Test trailing stop for short positions (stop above price)."""
        sl = manager.compute_trailing_stop(
            entry_price=2060.0,
            current_price=2050.0,
            atr=3.0,
            direction=-1,
            confidence=0.8
        )
        # Wide trail for short: SL = 2050 + 6.0 = 2056
        assert sl == pytest.approx(2056.0, abs=0.1)

    def test_trailing_stop_never_moves_backward_long(self, manager):
        """Test that trailing stop never moves below entry for winning long."""
        sl = manager.compute_trailing_stop(
            entry_price=2050.0,
            current_price=2055.0,  # In profit
            atr=3.0,
            direction=1,
            confidence=0.8  # Wide trail = 6.0
        )
        # Trail would put SL at 2055 - 6 = 2049, but entry is 2050
        # Should clamp to at least entry price
        assert sl >= 2050.0

    def test_trailing_stop_zero_atr_safety(self, manager):
        """Test safety fallback when ATR is zero."""
        sl = manager.compute_trailing_stop(
            entry_price=2050.0,
            current_price=2055.0,
            atr=0.0,  # Zero ATR (shouldn't happen but be safe)
            direction=1,
            confidence=0.5
        )
        # Should use fallback ATR of 1.0
        assert sl > 0


# ─────────────────────────────────────────────
#  PARTIAL CLOSE TESTS
# ─────────────────────────────────────────────
class TestPartialClose:
    """Tests for partial close logic."""

    def test_partial_close_at_2r(self, manager):
        """Test partial close triggered at 2R profit."""
        should_close, pct = manager.compute_partial_close(
            unrealized_pnl_atr=2.0,
            confidence=0.7,
            hold_bars=20,
            partial_closed_pct=0.0
        )
        assert should_close is True
        assert pct == pytest.approx(0.5, abs=0.01)  # 50% close

    def test_no_partial_close_below_2r(self, manager):
        """Test no partial close below 2R."""
        should_close, pct = manager.compute_partial_close(
            unrealized_pnl_atr=1.5,
            confidence=0.7,
            hold_bars=20,
            partial_closed_pct=0.0
        )
        assert should_close is False
        assert pct == 0.0

    def test_partial_close_at_3r(self, manager):
        """Test additional partial close at 3R."""
        should_close, pct = manager.compute_partial_close(
            unrealized_pnl_atr=3.0,
            confidence=0.7,
            hold_bars=30,
            partial_closed_pct=0.5  # Already closed 50% at 2R
        )
        assert should_close is True
        assert pct == pytest.approx(0.25, abs=0.01)  # 25% more

    def test_no_over_close(self, manager):
        """Test that partial close keeps at least 25% of position."""
        should_close, pct = manager.compute_partial_close(
            unrealized_pnl_atr=3.5,
            confidence=0.7,
            hold_bars=40,
            partial_closed_pct=0.75  # Already closed 75%
        )
        assert should_close is False  # Only 25% left, don't close more

    def test_confidence_decay_partial_close(self, manager):
        """Test partial close when confidence decays below threshold."""
        should_close, pct = manager.compute_partial_close(
            unrealized_pnl_atr=0.8,  # Small profit
            confidence=0.2,  # Below decay threshold (0.3)
            hold_bars=15,
            partial_closed_pct=0.0
        )
        assert should_close is True
        assert pct == 0.25


# ─────────────────────────────────────────────
#  EXIT EVALUATION TESTS (RULE-BASED)
# ─────────────────────────────────────────────
class TestExitEvaluation:
    """Tests for the main evaluate_exit function and rule overrides."""

    def test_max_hold_time_forces_close(self, manager, max_hold_position):
        """Test that exceeding max hold time forces full close."""
        decision = manager.evaluate_exit(max_hold_position)
        assert decision.action == "CLOSE_FULL"
        assert "Max hold time" in decision.reason

    def test_low_confidence_forces_close(self, manager):
        """Test that confidence below minimum forces close."""
        position = PositionState(
            direction=1, unrealized_pnl=1.0, unrealized_pnl_atr=0.3,
            hold_bars=10, entry_price=2050.0, current_price=2051.0,
            atr=3.0, confidence=0.1,  # Below min_confidence_to_hold=0.15
            initial_confidence=0.7, sl_distance_atr=1.5,
            tp_distance_atr=2.5, max_favorable=2.0, max_adverse=-1.0,
            partial_closed_pct=0.0, regime_changed=False, ticket="T004"
        )
        decision = manager.evaluate_exit(position)
        assert decision.action == "CLOSE_FULL"
        assert "Confidence too low" in decision.reason

    def test_catastrophic_loss_forces_close(self, manager, losing_position):
        """Test that loss > 2 ATR forces immediate close."""
        decision = manager.evaluate_exit(losing_position)
        assert decision.action == "CLOSE_FULL"
        assert "2 ATR" in decision.reason

    def test_2r_profit_triggers_partial(self, manager, profitable_long_position):
        """Test that reaching 2R profit triggers partial close rule."""
        decision = manager.evaluate_exit(profitable_long_position)
        # Should hit the 2R partial close rule
        assert decision.action == "CLOSE_PARTIAL"
        assert decision.lot_pct_to_close == pytest.approx(0.5, abs=0.01)
        assert "2R" in decision.reason

    def test_exit_decision_has_q_values(self, manager, profitable_long_position):
        """Test that exit decisions include Q-values for analysis."""
        decision = manager.evaluate_exit(profitable_long_position)
        assert decision.q_values is not None
        assert isinstance(decision.q_values, dict)
        assert len(decision.q_values) == 5  # 5 actions

    def test_normal_position_uses_rl(self, manager):
        """Test that normal position (no rule triggered) uses RL agent."""
        position = PositionState(
            direction=1, unrealized_pnl=1.0, unrealized_pnl_atr=0.5,
            hold_bars=5, entry_price=2050.0, current_price=2051.0,
            atr=3.0, confidence=0.6, initial_confidence=0.7,
            sl_distance_atr=1.5, tp_distance_atr=2.5,
            max_favorable=1.5, max_adverse=-0.5,
            partial_closed_pct=0.0, regime_changed=False, ticket="T005"
        )
        decision = manager.evaluate_exit(position)
        # Should get a valid decision (could be any action from RL)
        assert decision.action in ["HOLD", "CLOSE_FULL", "CLOSE_PARTIAL", "MODIFY_SL"]
        assert isinstance(decision.confidence, float)


# ─────────────────────────────────────────────
#  CONFIDENCE DECAY TESTS
# ─────────────────────────────────────────────
class TestConfidenceDecay:
    """Tests for confidence time decay."""

    def test_confidence_decays_over_time(self, manager):
        """Test confidence decreases with hold time."""
        initial = 0.8
        after_10 = manager.decay_confidence(initial, 10)
        after_50 = manager.decay_confidence(initial, 50)
        assert after_10 < initial
        assert after_50 < after_10

    def test_confidence_decay_formula(self, manager, exit_config):
        """Test confidence decay matches formula."""
        initial = 0.7
        bars = 20
        expected = initial * (exit_config.confidence_time_decay ** bars)
        result = manager.decay_confidence(initial, bars)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_zero_bars_no_decay(self, manager):
        """Test no decay when hold_bars is 0."""
        initial = 0.8
        result = manager.decay_confidence(initial, 0)
        assert result == pytest.approx(initial)


# ─────────────────────────────────────────────
#  FEED REWARD TESTS
# ─────────────────────────────────────────────
class TestFeedReward:
    """Tests for feeding trade outcomes to RL agent."""

    def test_feed_reward_stores_transition(self, manager):
        """Test that feeding reward stores transition in buffer."""
        position = PositionState(
            direction=1, unrealized_pnl=5.0, unrealized_pnl_atr=1.5,
            hold_bars=10, entry_price=2050.0, current_price=2055.0,
            atr=3.0, confidence=0.7, initial_confidence=0.8,
            sl_distance_atr=1.5, tp_distance_atr=2.5,
            max_favorable=6.0, max_adverse=-1.0,
            partial_closed_pct=0.0, regime_changed=False
        )
        next_state = np.zeros(manager.rl_agent.config.state_size, dtype=np.float32)

        initial_size = len(manager.rl_agent.replay_buffer)
        manager.feed_reward(position, ExitAction.CLOSE_FULL, 5.0, next_state, True)
        assert len(manager.rl_agent.replay_buffer) == initial_size + 1

    def test_get_stats(self, manager):
        """Test stats reporting."""
        stats = manager.get_stats()
        assert "decisions_made" in stats
        assert "positions_tracked" in stats
        assert "rl_stats" in stats
