"""
Person 4 — Metrics tracker.
Collects per-episode statistics used for evaluation and plotting.
"""

import json
import os
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.episode_rewards  = []
        self.episode_steps    = []
        self.episode_deaths   = []
        self.episode_losses   = []
        self.episode_eps      = []

    def log_episode(self, reward, steps, deaths, loss, eps):
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_deaths.append(deaths)
        self.episode_losses.append(loss)
        self.episode_eps.append(eps)

    # ------------------------------------------------------------------

    def success_rate(self, window: int = 100) -> float:
        """Fraction of episodes in the last `window` where agent survived > 2000 steps."""
        recent = self.episode_steps[-window:]
        return sum(s > 2000 for s in recent) / len(recent) if recent else 0.0

    def collision_death_rate(self, window: int = 100) -> float:
        """Average deaths per episode over the last `window` episodes."""
        recent = self.episode_deaths[-window:]
        return sum(recent) / len(recent) if recent else 0.0

    def smoothed(self, data, window: int = 20):
        out = []
        for i in range(len(data)):
            lo = max(0, i - window + 1)
            out.append(np.mean(data[lo:i + 1]))
        return out

    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'rewards': self.episode_rewards,
                'steps':   self.episode_steps,
                'deaths':  self.episode_deaths,
                'losses':  self.episode_losses,
                'eps':     self.episode_eps,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'MetricsTracker':
        with open(path) as f:
            data = json.load(f)
        m = cls()
        m.episode_rewards = data['rewards']
        m.episode_steps   = data['steps']
        m.episode_deaths  = data['deaths']
        m.episode_losses  = data['losses']
        m.episode_eps     = data['eps']
        return m
