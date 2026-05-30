"""
Training loop — ties Person 1 (env), Person 2 (CNN), Person 3 (DQN) together.

Usage:
    python train.py                  # train from scratch
    python train.py --resume checkpoints/model_ep500.pth

Google Colab setup (run once before importing):
    !pip install gym-retro torch opencv-python pybullet matplotlib
    !python -m retro.import DonkeyKong-Nes.nes
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DonkeyKongEnv
from models      import DQNAgent
from evaluation  import MetricsTracker


def train(num_episodes=1000, checkpoint_freq=100, checkpoint_dir='checkpoints',
          resume=None, render=False, fresh_buffer=False):

    os.makedirs(checkpoint_dir, exist_ok=True)

    env     = DonkeyKongEnv(render=render)
    agent   = DQNAgent(action_size=env.action_space_size)

    # Auto-resume from the latest checkpoint if one exists
    if resume is None:
        existing = sorted([
            f for f in os.listdir(checkpoint_dir)
            if f.startswith('model_ep') and f.endswith('.pth')
        ]) if os.path.isdir(checkpoint_dir) else []
        if existing:
            resume = os.path.join(checkpoint_dir, existing[-1])
            print(f"Auto-resuming from {resume}")

    start_episode = 1
    if resume and os.path.exists(resume):
        agent.load(resume)
        # Parse episode number from filename so the count continues correctly
        try:
            start_episode = int(resume.split('model_ep')[1].replace('.pth', '')) + 1
        except (IndexError, ValueError):
            start_episode = agent.steps + 1
        if fresh_buffer:
            agent.fresh_buffer(epsilon=0.50)

    metrics = MetricsTracker()
    if os.path.exists('metrics.npz'):
        metrics = MetricsTracker.load('metrics.npz')

    print(f"\nTraining for {num_episodes} episodes  |  "
          f"Actions: {env.action_space_size}  |  "
          f"State: {env.state_shape}\n")

    for episode in range(start_episode, start_episode + num_episodes):
        state        = env.reset()
        total_reward = 0.0
        total_loss   = 0.0
        loss_count   = 0
        steps        = 0
        done         = False
        info         = {}

        while not done:
            action                        = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            state        = next_state
            total_reward += reward
            steps        += 1

        agent.decay_epsilon()  # once per episode, not per step

        avg_loss       = total_loss / loss_count if loss_count else 0.0
        lives_remaining = info.get('lives', 0)

        metrics.record(episode, total_reward, avg_loss, steps, lives_remaining)

        if episode % 10 == 0:
            print(
                f"Ep {episode:5d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Loss: {avg_loss:.5f} | "
                f"Steps: {steps:5d} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Lives: {lives_remaining}"
            )

        if episode % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'model_ep{episode}.pth')
            agent.save(ckpt_path)
            metrics.save('metrics.npz')

    env.close()
    metrics.save('metrics.npz')
    metrics.plot_all('training_metrics.png')
    print("\nTraining complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',        type=int,  default=5000)
    parser.add_argument('--checkpoint-freq', type=int,  default=100)
    parser.add_argument('--checkpoint-dir',  type=str,  default='checkpoints')
    parser.add_argument('--resume',          type=str,  default=None)
    parser.add_argument('--render',          action='store_true')
    parser.add_argument('--fresh-buffer',    action='store_true',
                        help='Keep model weights but wipe replay buffer and reset epsilon to 0.5')
    args = parser.parse_args()

    train(
        num_episodes     = args.episodes,
        checkpoint_freq  = args.checkpoint_freq,
        checkpoint_dir   = args.checkpoint_dir,
        resume           = args.resume,
        render           = args.render,
        fresh_buffer     = args.fresh_buffer,
    )
