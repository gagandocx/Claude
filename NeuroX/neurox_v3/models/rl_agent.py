"""
=============================================================
  Python ML Bridge - Deep Q-Network Reinforcement Learning Agent
  Learns position management from executed trades like an
  experienced prop trader. Makes exit decisions: hold, close,
  partial close, or tighten stop loss.

  Architecture:
    - State: market features + position info (direction, PnL, hold time)
    - Actions: HOLD, CLOSE_FULL, CLOSE_PARTIAL_25, CLOSE_PARTIAL_50, TIGHTEN_STOP
    - Reward: realized PnL/ATR with shaping for cutting losers and running winners
    - Training: Experience replay with target network (Double DQN)
=============================================================
"""

import os
import sys
import json
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RLConfig, MODEL_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  ACTION SPACE
# ─────────────────────────────────────────────
class ExitAction(IntEnum):
    """Exit actions the RL agent can take on open positions.

    A professional trader manages positions actively:
    - HOLD: Let the position run (only when conviction is high)
    - CLOSE_FULL: Cut the entire position (take profit or stop loss)
    - CLOSE_PARTIAL_25: Take 25% off the table (scale out)
    - CLOSE_PARTIAL_50: Take 50% off (lock profits, let rest run)
    - TIGHTEN_STOP: Move stop loss closer (protect gains without exiting)
    """
    HOLD = 0
    CLOSE_FULL = 1
    CLOSE_PARTIAL_25 = 2
    CLOSE_PARTIAL_50 = 3
    TIGHTEN_STOP = 4


ACTION_NAMES = {
    ExitAction.HOLD: "HOLD_POSITION",
    ExitAction.CLOSE_FULL: "CLOSE_FULL",
    ExitAction.CLOSE_PARTIAL_25: "CLOSE_PARTIAL_25",
    ExitAction.CLOSE_PARTIAL_50: "CLOSE_PARTIAL_50",
    ExitAction.TIGHTEN_STOP: "TIGHTEN_STOP",
}


# ─────────────────────────────────────────────
#  POSITION STATE
# ─────────────────────────────────────────────
@dataclass
class PositionState:
    """State representation for an open position.

    Encodes everything the RL agent needs to make exit decisions,
    similar to what a prop trader tracks on their position monitor.
    """
    direction: int              # 1 = LONG, -1 = SHORT
    unrealized_pnl: float       # Current unrealized PnL in price units
    unrealized_pnl_atr: float   # PnL normalized by ATR (R-multiple proxy)
    hold_bars: int              # How many bars the position has been open
    entry_price: float          # Original entry price
    current_price: float        # Current market price
    atr: float                  # Current ATR value
    confidence: float           # Current model confidence (may have decayed)
    initial_confidence: float   # Confidence at entry
    sl_distance_atr: float      # Current SL distance in ATR multiples
    tp_distance_atr: float      # Current TP distance in ATR multiples
    max_favorable: float        # Maximum favorable excursion (best unrealized)
    max_adverse: float          # Maximum adverse excursion (worst unrealized)
    partial_closed_pct: float   # Percentage already closed (0.0 to 0.75)
    regime_changed: bool        # Whether regime changed since entry
    ticket: str = ""            # Trade ticket ID for reference


# ─────────────────────────────────────────────
#  Q-NETWORK
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """Deep Q-Network for position management decisions.

    3-layer MLP with layer normalization and residual connection.
    Designed to approximate Q(state, action) for exit decisions.
    Uses Dueling DQN architecture: separate value and advantage streams
    for more stable learning.
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int = 256):
        super().__init__()
        self.state_size = state_size
        self.num_actions = num_actions

        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Dueling DQN: Value stream (how good is this state?)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Dueling DQN: Advantage stream (how much better is each action?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given state.

        Uses dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine using dueling formula
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ─────────────────────────────────────────────
#  EXPERIENCE REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    """Prioritized experience replay buffer.

    Stores (state, action, reward, next_state, done) transitions.
    Uses simple uniform sampling (can be upgraded to prioritized replay).
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= 256


# ─────────────────────────────────────────────
#  RL AGENT (DQN with Prop Trader Logic)
# ─────────────────────────────────────────────
class RLAgent:
    """Deep Q-Network agent for position management.

    Learns to manage positions like an experienced prop trader:
    - Cuts losers quickly (reward for early exit on losing trades)
    - Lets winners run (penalty for premature profit-taking)
    - Takes partial profits at key levels (2R, 3R)
    - Tightens stops as conviction wanes

    Uses Double DQN with soft target updates for stable learning.
    Epsilon-greedy exploration that decays as the agent gains experience.
    """

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()

        # Networks
        self.q_network = QNetwork(
            state_size=self.config.state_size,
            num_actions=self.config.num_actions,
            hidden_size=self.config.hidden_size
        )
        self.target_network = QNetwork(
            state_size=self.config.state_size,
            num_actions=self.config.num_actions,
            hidden_size=self.config.hidden_size
        )
        # Initialize target with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size
        )

        # State tracking
        self.epsilon = self.config.epsilon_start
        self.training_steps = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []

    def state_from_position(self, position: PositionState,
                            market_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert position info + market features into state vector.

        State components (professional trader's decision factors):
        - Position info: direction, PnL, hold time, distances
        - Market context: confidence, regime, volatility
        - Trade quality: max favorable/adverse excursion ratios
        - Remaining features from model (truncated/padded to state_size)
        """
        # Core position features (15 features)
        position_features = np.array([
            float(position.direction),
            position.unrealized_pnl_atr,
            min(position.hold_bars / 100.0, 2.0),  # Normalized hold time
            position.confidence,
            position.initial_confidence,
            position.confidence / max(position.initial_confidence, 0.01),  # Confidence ratio
            position.sl_distance_atr,
            position.tp_distance_atr,
            position.max_favorable / max(position.atr, 0.01),  # MFE in ATR
            position.max_adverse / max(position.atr, 0.01),    # MAE in ATR
            position.partial_closed_pct,
            1.0 if position.regime_changed else 0.0,
            position.unrealized_pnl_atr / max(abs(position.max_adverse / max(position.atr, 0.01)), 0.01),  # Recovery ratio
            1.0 if position.unrealized_pnl > 0 else 0.0,  # In profit flag
            min(position.hold_bars, 50) / 50.0,  # Short-term hold ratio
        ], dtype=np.float32)

        # Pad or combine with market features
        if market_features is not None:
            # Take first N features from market to fill remaining state
            remaining = self.config.state_size - len(position_features)
            if len(market_features.flatten()) >= remaining:
                market_slice = market_features.flatten()[:remaining]
            else:
                market_slice = np.zeros(remaining, dtype=np.float32)
                available = min(len(market_features.flatten()), remaining)
                market_slice[:available] = market_features.flatten()[:available]
            state = np.concatenate([position_features, market_slice])
        else:
            # Pad with zeros if no market features
            state = np.zeros(self.config.state_size, dtype=np.float32)
            state[:len(position_features)] = position_features

        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        During training: explores with probability epsilon.
        During inference: always picks the greedy action (best Q-value).
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.config.num_actions)

        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return int(q_values.argmax(dim=1).item())

    def compute_reward(self, position: PositionState, action: int,
                       realized_pnl: float = 0.0,
                       trade_closed: bool = False) -> float:
        """Compute shaped reward like a prop trader's P&L attribution.

        Reward components:
        1. Realized PnL normalized by ATR (main signal)
        2. Hold penalty (encourages active management)
        3. Cut-loser bonus (rewards discipline)
        4. Winner-run bonus (rewards patience with profits)
        5. Over-hold penalty (punishes passive holding)
        """
        reward = 0.0
        atr = max(position.atr, 0.01)

        if trade_closed:
            # Main reward: realized PnL normalized by ATR
            reward = realized_pnl / atr

            # Bonus for cutting losers early (prop trader discipline)
            if realized_pnl < 0 and position.hold_bars < 20:
                # Small loss cut quickly - reward the discipline
                reward += self.config.cut_loser_bonus * (1.0 - position.hold_bars / 20.0)

            # Bonus for letting winners run past 1R
            if realized_pnl > atr and position.hold_bars > 10:
                reward += self.config.winner_run_bonus

        else:
            # Per-step rewards (while position is still open)
            # Small hold penalty to encourage active decision-making
            reward += self.config.hold_penalty_per_bar

            # Penalize holding past maximum bars
            if position.hold_bars > self.config.max_hold_bars_penalty:
                over_hold = position.hold_bars - self.config.max_hold_bars_penalty
                reward -= 0.01 * over_hold

            # Reward for tightening stop when unrealized profit exists
            if action == ExitAction.TIGHTEN_STOP and position.unrealized_pnl > 0:
                reward += 0.05

            # Penalty for holding losing positions too long
            if position.unrealized_pnl_atr < -1.0 and action == ExitAction.HOLD:
                reward -= 0.1  # Should be cutting this loser

        return reward

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_reward += reward

    def train_step(self) -> Optional[float]:
        """Perform one training step (Double DQN update).

        Samples mini-batch from replay buffer, computes TD targets
        using target network, and updates Q-network.

        Returns:
            Loss value if training occurred, None otherwise.
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Current Q-values for taken actions
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # Compute Huber loss (more robust than MSE for RL)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network (soft update)
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self._soft_update_target()

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

        return loss.item()

    def _soft_update_target(self):
        """Soft update target network weights: target = tau*online + (1-tau)*target."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state (for analysis/debugging)."""
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return q_values.numpy()[0]

    def save(self, path: Optional[str] = None):
        """Save agent state (networks, optimizer, epsilon, training stats)."""
        save_dir = path or MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "rl_agent.pt")

        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "total_reward": self.total_reward,
            "config": {
                "state_size": self.config.state_size,
                "num_actions": self.config.num_actions,
                "hidden_size": self.config.hidden_size,
            }
        }
        torch.save(checkpoint, save_path)
        logger.info(f"[RLAgent] Saved checkpoint to {save_path}")

    def load(self, path: Optional[str] = None):
        """Load agent state from checkpoint."""
        load_dir = path or MODEL_DIR
        load_path = os.path.join(load_dir, "rl_agent.pt")

        if not os.path.exists(load_path):
            logger.warning(f"[RLAgent] No checkpoint found at {load_path}")
            return False

        checkpoint = torch.load(load_path, map_location="cpu")
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.total_reward = checkpoint.get("total_reward", 0.0)
        logger.info(f"[RLAgent] Loaded checkpoint (step={self.training_steps}, "
                    f"epsilon={self.epsilon:.4f})")
        return True

    def get_stats(self) -> Dict:
        """Get agent training statistics."""
        return {
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "replay_buffer_size": len(self.replay_buffer),
            "buffer_ready": self.replay_buffer.is_ready,
        }
