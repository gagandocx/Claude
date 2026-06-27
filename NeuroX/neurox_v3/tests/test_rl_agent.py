"""
=============================================================
  Python ML Bridge - RL Agent Tests
  Tests for DQN reinforcement learning agent:
    - Q-network forward pass
    - Training step with TD-error
    - Epsilon-greedy exploration and decay
    - Replay buffer operations
    - Reward shaping logic
    - State construction from position info
=============================================================
"""

import pytest
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RLConfig
from models.rl_agent import (
    RLAgent, QNetwork, ReplayBuffer, ExitAction,
    ACTION_NAMES, PositionState
)


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def rl_config():
    """Create test RL config with small sizes for fast testing."""
    return RLConfig(
        replay_buffer_size=500,
        batch_size=16,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        learning_rate=1e-3,
        target_update_freq=10,
        hidden_size=64,
        state_size=20,
        num_actions=5,
        min_replay_size=32,
    )


@pytest.fixture
def agent(rl_config):
    """Create RL agent with test config."""
    return RLAgent(rl_config)


@pytest.fixture
def sample_position():
    """Create a sample position state."""
    return PositionState(
        direction=1,
        unrealized_pnl=5.0,
        unrealized_pnl_atr=1.5,
        hold_bars=10,
        entry_price=2050.0,
        current_price=2055.0,
        atr=3.5,
        confidence=0.7,
        initial_confidence=0.8,
        sl_distance_atr=1.5,
        tp_distance_atr=2.5,
        max_favorable=7.0,
        max_adverse=-2.0,
        partial_closed_pct=0.0,
        regime_changed=False,
        ticket="12345"
    )


@pytest.fixture
def filled_buffer(rl_config):
    """Create a replay buffer filled with random transitions."""
    buffer = ReplayBuffer(capacity=rl_config.replay_buffer_size)
    for _ in range(100):
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        action = np.random.randint(rl_config.num_actions)
        reward = np.random.randn()
        next_state = np.random.randn(rl_config.state_size).astype(np.float32)
        done = np.random.random() < 0.1
        buffer.push(state, action, reward, next_state, done)
    return buffer


# ─────────────────────────────────────────────
#  Q-NETWORK TESTS
# ─────────────────────────────────────────────
class TestQNetwork:
    """Tests for the Q-Network architecture."""

    def test_forward_pass_shape(self, rl_config):
        """Test Q-network produces correct output shape."""
        net = QNetwork(rl_config.state_size, rl_config.num_actions, rl_config.hidden_size)
        state = torch.randn(4, rl_config.state_size)
        output = net(state)
        assert output.shape == (4, rl_config.num_actions)

    def test_single_state_forward(self, rl_config):
        """Test Q-network works with single state input."""
        net = QNetwork(rl_config.state_size, rl_config.num_actions, rl_config.hidden_size)
        state = torch.randn(1, rl_config.state_size)
        output = net(state)
        assert output.shape == (1, rl_config.num_actions)

    def test_gradient_flow(self, rl_config):
        """Test gradients flow through Q-network."""
        net = QNetwork(rl_config.state_size, rl_config.num_actions, rl_config.hidden_size)
        state = torch.randn(4, rl_config.state_size)
        output = net(state)
        loss = output.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "Q-network should have non-zero gradients"

    def test_dueling_architecture(self, rl_config):
        """Test dueling DQN produces different Q-values per action."""
        net = QNetwork(rl_config.state_size, rl_config.num_actions, rl_config.hidden_size)
        state = torch.randn(1, rl_config.state_size)
        output = net(state)[0]
        # With random weights, Q-values should differ per action
        assert not torch.allclose(output, output[0].expand_as(output)), \
            "Dueling architecture should produce different Q-values per action"

    def test_deterministic_eval(self, rl_config):
        """Test Q-network is deterministic in eval mode."""
        net = QNetwork(rl_config.state_size, rl_config.num_actions, rl_config.hidden_size)
        net.eval()
        state = torch.randn(2, rl_config.state_size)
        with torch.no_grad():
            out1 = net(state)
            out2 = net(state)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ─────────────────────────────────────────────
#  REPLAY BUFFER TESTS
# ─────────────────────────────────────────────
class TestReplayBuffer:
    """Tests for experience replay buffer."""

    def test_push_and_length(self):
        """Test that pushing transitions increases buffer length."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        state = np.zeros(10, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)
        assert len(buffer) == 1

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        state = np.zeros(5, dtype=np.float32)
        for i in range(20):
            buffer.push(state, 0, float(i), state, False)
        assert len(buffer) == 10

    def test_sample_shape(self, rl_config):
        """Test that sample returns correct shapes."""
        buffer = ReplayBuffer(capacity=100)
        state_size = rl_config.state_size
        for _ in range(50):
            state = np.random.randn(state_size).astype(np.float32)
            buffer.push(state, 1, 0.5, state, False)

        states, actions, rewards, next_states, dones = buffer.sample(16)
        assert states.shape == (16, state_size)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, state_size)
        assert dones.shape == (16,)

    def test_sample_dtypes(self, rl_config):
        """Test that sampled data has correct dtypes."""
        buffer = ReplayBuffer(capacity=100)
        for _ in range(50):
            state = np.random.randn(rl_config.state_size).astype(np.float32)
            buffer.push(state, 2, -0.5, state, True)

        states, actions, rewards, next_states, dones = buffer.sample(8)
        assert states.dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32

    def test_is_ready(self):
        """Test is_ready property (needs 256 samples)."""
        buffer = ReplayBuffer(capacity=500)
        state = np.zeros(10, dtype=np.float32)
        assert not buffer.is_ready
        for _ in range(255):
            buffer.push(state, 0, 0.0, state, False)
        assert not buffer.is_ready
        buffer.push(state, 0, 0.0, state, False)
        assert buffer.is_ready

    def test_fifo_order(self):
        """Test that oldest transitions are dropped first."""
        buffer = ReplayBuffer(capacity=3)
        state = np.zeros(5, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)  # Will be dropped
        buffer.push(state, 1, 2.0, state, False)
        buffer.push(state, 2, 3.0, state, False)
        buffer.push(state, 3, 4.0, state, False)  # Drops first
        # Buffer should contain rewards 2.0, 3.0, 4.0
        rewards = [t[2] for t in buffer.buffer]
        assert 1.0 not in rewards
        assert 4.0 in rewards


# ─────────────────────────────────────────────
#  RL AGENT TESTS
# ─────────────────────────────────────────────
class TestRLAgent:
    """Tests for the DQN agent."""

    def test_agent_creation(self, agent, rl_config):
        """Test agent initializes correctly."""
        assert agent.epsilon == rl_config.epsilon_start
        assert agent.training_steps == 0
        assert len(agent.replay_buffer) == 0

    def test_state_from_position(self, agent, sample_position):
        """Test state vector construction from position."""
        state = agent.state_from_position(sample_position)
        assert state.shape == (agent.config.state_size,)
        assert state.dtype == np.float32
        # Direction should be first element
        assert state[0] == 1.0  # LONG

    def test_state_from_position_with_features(self, agent, sample_position):
        """Test state construction with market features appended."""
        market_features = np.random.randn(50).astype(np.float32)
        state = agent.state_from_position(sample_position, market_features)
        assert state.shape == (agent.config.state_size,)

    def test_select_action_exploration(self, agent, rl_config):
        """Test that epsilon=1.0 always explores (random actions)."""
        agent.epsilon = 1.0
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        actions = set()
        for _ in range(100):
            actions.add(agent.select_action(state, training=True))
        # With 100 random tries and 5 actions, we should hit multiple
        assert len(actions) > 1

    def test_select_action_exploitation(self, agent, rl_config):
        """Test that epsilon=0.0 always exploits (greedy)."""
        agent.epsilon = 0.0
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        action1 = agent.select_action(state, training=True)
        action2 = agent.select_action(state, training=True)
        # Should always pick same action when not exploring
        assert action1 == action2

    def test_select_action_inference(self, agent, rl_config):
        """Test that inference mode ignores epsilon."""
        agent.epsilon = 1.0  # Full exploration
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        # In inference mode (training=False), should be deterministic
        action1 = agent.select_action(state, training=False)
        action2 = agent.select_action(state, training=False)
        assert action1 == action2

    def test_train_step_returns_none_when_buffer_empty(self, agent):
        """Test train_step returns None when buffer has insufficient samples."""
        loss = agent.train_step()
        assert loss is None

    def test_train_step_updates_q_values(self, agent, rl_config):
        """Test that training step updates Q-network parameters."""
        # Fill replay buffer
        for _ in range(rl_config.min_replay_size + 10):
            state = np.random.randn(rl_config.state_size).astype(np.float32)
            action = np.random.randint(rl_config.num_actions)
            reward = np.random.randn()
            next_state = np.random.randn(rl_config.state_size).astype(np.float32)
            done = np.random.random() < 0.1
            agent.store_transition(state, action, reward, next_state, done)

        # Get Q-values before training
        test_state = np.random.randn(rl_config.state_size).astype(np.float32)
        q_before = agent.get_q_values(test_state).copy()

        # Train for several steps
        for _ in range(10):
            loss = agent.train_step()
            assert loss is not None
            assert loss >= 0

        # Q-values should have changed
        q_after = agent.get_q_values(test_state)
        assert not np.allclose(q_before, q_after), \
            "Q-values should change after training"

    def test_epsilon_decay(self, agent, rl_config):
        """Test epsilon decays during training."""
        # Fill buffer
        for _ in range(rl_config.min_replay_size + 10):
            state = np.random.randn(rl_config.state_size).astype(np.float32)
            agent.store_transition(state, 0, 0.0, state, False)

        initial_epsilon = agent.epsilon
        agent.train_step()
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon == initial_epsilon * rl_config.epsilon_decay

    def test_epsilon_never_below_minimum(self, agent, rl_config):
        """Test epsilon never goes below epsilon_end."""
        agent.epsilon = rl_config.epsilon_end + 0.001
        # Fill buffer
        for _ in range(rl_config.min_replay_size + 10):
            state = np.random.randn(rl_config.state_size).astype(np.float32)
            agent.store_transition(state, 0, 0.0, state, False)

        # Decay a lot
        for _ in range(100):
            agent.train_step()

        assert agent.epsilon >= rl_config.epsilon_end

    def test_store_transition(self, agent, rl_config):
        """Test storing transitions in replay buffer."""
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        next_state = np.random.randn(rl_config.state_size).astype(np.float32)
        agent.store_transition(state, 1, 0.5, next_state, False)
        assert len(agent.replay_buffer) == 1

    def test_get_q_values(self, agent, rl_config):
        """Test getting Q-values for analysis."""
        state = np.random.randn(rl_config.state_size).astype(np.float32)
        q_values = agent.get_q_values(state)
        assert q_values.shape == (rl_config.num_actions,)

    def test_get_stats(self, agent):
        """Test getting agent statistics."""
        stats = agent.get_stats()
        assert "training_steps" in stats
        assert "epsilon" in stats
        assert "total_reward" in stats
        assert "replay_buffer_size" in stats
        assert "buffer_ready" in stats


# ─────────────────────────────────────────────
#  REWARD SHAPING TESTS
# ─────────────────────────────────────────────
class TestRewardShaping:
    """Tests for the prop-trader-style reward function."""

    def test_positive_pnl_gives_positive_reward(self, agent, sample_position):
        """Test profitable trade gives positive reward."""
        reward = agent.compute_reward(
            sample_position, ExitAction.CLOSE_FULL,
            realized_pnl=10.0, trade_closed=True
        )
        assert reward > 0

    def test_negative_pnl_gives_negative_reward(self, agent, sample_position):
        """Test losing trade gives negative reward."""
        reward = agent.compute_reward(
            sample_position, ExitAction.CLOSE_FULL,
            realized_pnl=-10.0, trade_closed=True
        )
        assert reward < 0

    def test_cut_loser_bonus(self, agent):
        """Test early exit on loser gets bonus reward."""
        # Position held for only 5 bars (cut quickly)
        position = PositionState(
            direction=1, unrealized_pnl=-3.0, unrealized_pnl_atr=-1.0,
            hold_bars=5, entry_price=2050.0, current_price=2047.0,
            atr=3.0, confidence=0.4, initial_confidence=0.7,
            sl_distance_atr=1.5, tp_distance_atr=2.5,
            max_favorable=1.0, max_adverse=-3.0,
            partial_closed_pct=0.0, regime_changed=False
        )
        reward_quick = agent.compute_reward(
            position, ExitAction.CLOSE_FULL, realized_pnl=-3.0, trade_closed=True
        )

        # Same loss but held for 30 bars (too slow to cut)
        position.hold_bars = 30
        reward_slow = agent.compute_reward(
            position, ExitAction.CLOSE_FULL, realized_pnl=-3.0, trade_closed=True
        )

        # Quick cut should get better reward (less negative)
        assert reward_quick > reward_slow

    def test_hold_penalty(self, agent, sample_position):
        """Test that holding gives small penalty per bar."""
        reward = agent.compute_reward(
            sample_position, ExitAction.HOLD,
            realized_pnl=0.0, trade_closed=False
        )
        assert reward < 0  # Hold penalty is negative

    def test_winner_run_bonus(self, agent):
        """Test letting winners run past 1R gets bonus."""
        position = PositionState(
            direction=1, unrealized_pnl=5.0, unrealized_pnl_atr=2.0,
            hold_bars=15, entry_price=2050.0, current_price=2055.0,
            atr=2.5, confidence=0.7, initial_confidence=0.8,
            sl_distance_atr=1.5, tp_distance_atr=2.5,
            max_favorable=5.0, max_adverse=-1.0,
            partial_closed_pct=0.0, regime_changed=False
        )
        # Close with profit > ATR and held > 10 bars
        reward = agent.compute_reward(
            position, ExitAction.CLOSE_FULL,
            realized_pnl=5.0, trade_closed=True
        )
        # Should include winner run bonus
        base_reward = 5.0 / position.atr  # PnL/ATR
        assert reward > base_reward  # Bonus added

    def test_over_hold_penalty(self, agent):
        """Test penalty for holding past max bars."""
        position = PositionState(
            direction=1, unrealized_pnl=0.0, unrealized_pnl_atr=0.0,
            hold_bars=150,  # Way past max (100)
            entry_price=2050.0, current_price=2050.0,
            atr=3.0, confidence=0.3, initial_confidence=0.7,
            sl_distance_atr=1.5, tp_distance_atr=2.5,
            max_favorable=2.0, max_adverse=-2.0,
            partial_closed_pct=0.0, regime_changed=False
        )
        reward = agent.compute_reward(
            position, ExitAction.HOLD, realized_pnl=0.0, trade_closed=False
        )
        # Should be more negative than regular hold penalty
        assert reward < agent.config.hold_penalty_per_bar


# ─────────────────────────────────────────────
#  ACTION SPACE TESTS
# ─────────────────────────────────────────────
class TestActionSpace:
    """Tests for action definitions."""

    def test_action_count(self):
        """Test correct number of actions."""
        assert len(ExitAction) == 5

    def test_action_names_mapping(self):
        """Test all actions have names."""
        for action in ExitAction:
            assert action in ACTION_NAMES

    def test_action_values_sequential(self):
        """Test action values are 0-indexed sequential."""
        values = [a.value for a in ExitAction]
        assert values == [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────
#  SAVE/LOAD TESTS
# ─────────────────────────────────────────────
class TestSaveLoad:
    """Tests for agent checkpoint save/load."""

    def test_save_and_load(self, agent, rl_config, tmp_path):
        """Test that save and load preserves agent state."""
        # Modify agent state
        agent.epsilon = 0.5
        agent.training_steps = 42

        # Save
        agent.save(str(tmp_path))

        # Create new agent and load
        new_agent = RLAgent(rl_config)
        assert new_agent.epsilon != 0.5  # Different before load
        success = new_agent.load(str(tmp_path))
        assert success
        assert new_agent.epsilon == 0.5
        assert new_agent.training_steps == 42

    def test_load_nonexistent(self, agent, tmp_path):
        """Test loading from nonexistent path returns False."""
        success = agent.load(str(tmp_path / "nonexistent"))
        assert success is False
