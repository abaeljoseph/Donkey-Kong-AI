"""
Person 3 — DQN agent.
Implements epsilon-greedy action selection, Huber-loss training,
and periodic target network synchronisation.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import DQNNetwork
from training.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        obs_shape: tuple,
        device: torch.device,
        lr: float           = 1e-4,
        gamma: float        = 0.99,
        batch_size: int     = 32,
        buffer_capacity: int = 100_000,
        eps_start: float    = 1.0,
        eps_end: float      = 0.01,
        eps_decay_steps: int = 500_000,
        target_update_freq: int = 1_000,
        train_freq: int     = 4,
        min_buffer: int     = 10_000,
    ):
        self.num_actions    = num_actions
        self.device         = device
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.eps            = eps_start
        self.eps_end        = eps_end
        self.eps_decay      = (eps_start - eps_end) / eps_decay_steps
        self.target_update_freq = target_update_freq
        self.train_freq     = train_freq
        self.min_buffer     = min_buffer
        self.step_count     = 0

        self.online_net = DQNNetwork(num_actions).to(device)
        self.target_net = DQNNetwork(num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss
        self.buffer    = ReplayBuffer(buffer_capacity, obs_shape)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """Batch action selection for N parallel envs."""
        n = len(states)
        if random.random() < self.eps:
            return np.array([random.randrange(self.num_actions) for _ in range(n)])
        with torch.no_grad():
            s = torch.FloatTensor(states).to(self.device)
            return self.online_net(s).argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        loss = None
        if (len(self.buffer) >= self.min_buffer and
                self.step_count % self.train_freq == 0):
            loss = self._train_step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss

    def _train_step(self) -> float:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            targets    = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'step_count': self.step_count,
            'eps':        self.eps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.step_count = ckpt['step_count']
        self.eps        = ckpt['eps']
