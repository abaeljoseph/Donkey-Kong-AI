"""
Person 3 — DQN Agent
Decides which action to take based on the CNN's feature vector.

Key components:
  QNetwork     — CNN backbone + linear head outputting Q-values per action
  ReplayBuffer — stores past transitions so training isn't correlated
  DQNAgent     — ties it together: ε-greedy selection, training, checkpointing
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cnn import DonkeyKongCNN


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.cnn  = DonkeyKongCNN()
        self.head = nn.Linear(self.cnn.output_dim, action_size)

    def forward(self, x):
        return self.head(self.cnn(x))


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        action_size,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        batch_size=32,
        target_update_freq=1000,
        buffer_capacity=20_000,
        device=None,
    ):
        self.action_size        = action_size
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.steps              = 0

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DQNAgent using device: {self.device}")

        self.policy_net = QNetwork(action_size).to(self.device)
        self.target_net = QNetwork(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss — more stable than MSE for DQN
        self.buffer    = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        # Called once per episode so exploration lasts across many episodes
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
            'steps':      self.steps,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
        self.steps   = ckpt['steps']
        print(f"Checkpoint loaded ← {path}  (step={self.steps}, ε={self.epsilon:.3f})")

    def fresh_buffer(self, epsilon=0.5):
        """Keep model weights but discard all replay experiences and reset epsilon.
        Use this when the reward function changes — the network's visual knowledge
        is still valid, but old transitions carry wrong reward values."""
        capacity = self.buffer.buffer.maxlen
        self.buffer  = ReplayBuffer(capacity)
        self.epsilon = epsilon
        print(f"Replay buffer cleared. ε reset to {epsilon:.2f} — retraining with new rewards.")
