"""
Person 4 — Evaluation plots.
Produces the five required metrics as a single figure saved to results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from evaluation.metrics import MetricsTracker


def plot_all_metrics(metrics: MetricsTracker, save_dir: str = 'results'):
    os.makedirs(save_dir, exist_ok=True)

    episodes = list(range(1, len(metrics.episode_rewards) + 1))
    smoothed_reward = metrics.smoothed(metrics.episode_rewards)
    smoothed_loss   = metrics.smoothed(
        [l for l in metrics.episode_losses if l > 0] or [0],
        window=20,
    )

    # Rolling success rate and death rate
    window = 50
    success_rates = []
    death_rates   = []
    for i in range(len(episodes)):
        lo = max(0, i - window + 1)
        steps_window  = metrics.episode_steps[lo:i + 1]
        deaths_window = metrics.episode_deaths[lo:i + 1]
        success_rates.append(np.mean(metrics.episode_successes[lo:i + 1]))
        death_rates.append(np.mean(deaths_window))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Donkey Kong DQN — Training Metrics', fontsize=15, fontweight='bold')

    # 1. Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, metrics.episode_rewards, alpha=0.3, color='steelblue', linewidth=0.8)
    ax.plot(episodes, smoothed_reward, color='steelblue', linewidth=2, label='Smoothed (w=20)')
    ax.set_title('Reward Curve')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Success rate
    ax = axes[0, 1]
    ax.plot(episodes, success_rates, color='green', linewidth=2)
    ax.set_title('Success Rate (reached Pauline)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 3. Collision / death rate
    ax = axes[1, 0]
    ax.plot(episodes, metrics.episode_deaths, alpha=0.3, color='crimson', linewidth=0.8)
    ax.plot(episodes, death_rates, color='crimson', linewidth=2, label=f'Rolling avg (w={window})')
    ax.set_title('Collision / Death Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Deaths per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Steps to goal (episode length)
    ax = axes[1, 1]
    smoothed_steps = metrics.smoothed(metrics.episode_steps)
    ax.plot(episodes, metrics.episode_steps, alpha=0.3, color='darkorange', linewidth=0.8)
    ax.plot(episodes, smoothed_steps, color='darkorange', linewidth=2, label='Smoothed (w=20)')
    ax.set_title('Steps per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Loss curve
    ax = axes[2, 0]
    loss_episodes = [i + 1 for i, l in enumerate(metrics.episode_losses) if l > 0]
    loss_values   = [l for l in metrics.episode_losses if l > 0]
    if loss_values:
        s_loss = metrics.smoothed(loss_values, window=20)
        ax.plot(loss_episodes, loss_values, alpha=0.3, color='purple', linewidth=0.8)
        ax.plot(loss_episodes, s_loss, color='purple', linewidth=2, label='Smoothed (w=20)')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Huber Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Epsilon decay
    ax = axes[2, 1]
    ax.plot(episodes, metrics.episode_eps, color='teal', linewidth=2)
    ax.set_title('Epsilon (Exploration Rate)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f'Saved metrics plot → {out_path}')

    _print_summary(metrics)


def _print_summary(metrics: MetricsTracker):
    n = len(metrics.episode_rewards)
    if n == 0:
        return
    last100 = slice(max(0, n - 100), n)
    print('\n========== Evaluation Summary ==========')
    print(f'  Total episodes       : {n}')
    print(f'  Avg reward (last 100): {np.mean(metrics.episode_rewards[last100]):.2f}')
    print(f'  Success rate (last 100): {metrics.success_rate(100):.1%}')
    print(f'  Avg deaths (last 100): {metrics.collision_death_rate(100):.2f}')
    print(f'  Avg steps  (last 100): {np.mean(metrics.episode_steps[last100]):.1f}')
    print('=========================================\n')
