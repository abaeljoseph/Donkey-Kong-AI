"""
Person 4 — Evaluation & Metrics
Records per-episode stats, saves them to disk, and produces portfolio plots.
"""

import numpy as np
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self):
        self.episodes      = []
        self.rewards       = []
        self.losses        = []
        self.steps_list    = []
        self.success_flags = []  # 1 if survived > SURVIVAL_THRESHOLD steps
        self.death_flags   = []  # 1 if lives reached 0

    SURVIVAL_THRESHOLD = 300  # steps considered a "partial success"

    def record(self, episode, reward, loss, steps, lives_remaining):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.losses.append(loss)
        self.steps_list.append(steps)
        self.success_flags.append(int(steps > self.SURVIVAL_THRESHOLD))
        self.death_flags.append(int(lives_remaining == 0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path='metrics.npz'):
        np.savez(
            path,
            episodes=self.episodes,
            rewards=self.rewards,
            losses=self.losses,
            steps=self.steps_list,
            successes=self.success_flags,
            deaths=self.death_flags,
        )

    @classmethod
    def load(cls, path='metrics.npz'):
        tracker = cls()
        data = np.load(path)
        tracker.episodes      = data['episodes'].tolist()
        tracker.rewards       = data['rewards'].tolist()
        tracker.losses        = data['losses'].tolist()
        tracker.steps_list    = data['steps'].tolist()
        tracker.success_flags = data['successes'].tolist()
        tracker.death_flags   = data['deaths'].tolist()
        return tracker

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _moving_avg(data, window=20):
        if len(data) < window:
            return np.array([]), []
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        x_offset = window - 1
        return smoothed, x_offset

    def plot_all(self, save_path='training_metrics.png'):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Donkey Kong DQN — Training Metrics', fontsize=15, fontweight='bold')

        eps = np.array(self.episodes)

        # 1. Reward curve
        ax = axes[0, 0]
        ax.plot(eps, self.rewards, alpha=0.25, color='steelblue')
        sm, off = self._moving_avg(self.rewards)
        if len(sm):
            ax.plot(eps[off:], sm, color='steelblue', linewidth=2, label='MA-20')
            ax.legend()
        ax.set_title('Reward per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(alpha=0.3)

        # 2. Loss curve
        ax = axes[0, 1]
        ax.plot(eps, self.losses, alpha=0.25, color='tomato')
        sm, off = self._moving_avg(self.losses)
        if len(sm):
            ax.plot(eps[off:], sm, color='tomato', linewidth=2, label='MA-20')
            ax.legend()
        ax.set_title('Avg Loss per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.3)

        # 3. Steps per episode
        ax = axes[0, 2]
        ax.plot(eps, self.steps_list, alpha=0.25, color='mediumseagreen')
        sm, off = self._moving_avg(self.steps_list)
        if len(sm):
            ax.plot(eps[off:], sm, color='mediumseagreen', linewidth=2, label='MA-20')
            ax.axhline(self.SURVIVAL_THRESHOLD, color='gold', linestyle='--',
                       label=f'Survival threshold ({self.SURVIVAL_THRESHOLD})')
            ax.legend()
        ax.set_title('Steps per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.grid(alpha=0.3)

        # 4. Success rate (rolling 50)
        ax = axes[1, 0]
        window = 50
        sm, off = self._moving_avg(self.success_flags, window)
        if len(sm):
            ax.plot(eps[off:], sm * 100, color='gold', linewidth=2)
        ax.set_title(f'Success Rate (rolling {window}-ep)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

        # 5. Death rate (rolling 50)
        ax = axes[1, 1]
        sm, off = self._moving_avg(self.death_flags, window)
        if len(sm):
            ax.plot(eps[off:], sm * 100, color='salmon', linewidth=2)
        ax.set_title(f'Death Rate (rolling {window}-ep)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Death Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

        # 6. Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        n = min(100, len(self.rewards))
        summary = (
            f"Last {n} Episodes\n"
            f"{'─' * 24}\n"
            f"Avg Reward   : {np.mean(self.rewards[-n:]):>8.2f}\n"
            f"Max Reward   : {np.max(self.rewards[-n:]):>8.2f}\n"
            f"Avg Steps    : {np.mean(self.steps_list[-n:]):>8.0f}\n"
            f"Success Rate : {np.mean(self.success_flags[-n:])*100:>7.1f}%\n"
            f"Death Rate   : {np.mean(self.death_flags[-n:])*100:>7.1f}%\n"
            f"{'─' * 24}\n"
            f"Total Episodes: {len(self.episodes)}"
        )
        ax.text(0.05, 0.5, summary, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Metrics plot saved → {save_path}")


if __name__ == '__main__':
    tracker = MetricsTracker.load('metrics.npz')
    tracker.plot_all()
