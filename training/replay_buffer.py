"""Circular experience replay buffer backed by numpy arrays."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
