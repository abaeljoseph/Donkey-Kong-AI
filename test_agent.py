"""
Run a trained agent — no learning, no exploration, pure greedy policy.

Usage:
    python test_agent.py checkpoints/model_ep1000.pth
    python test_agent.py checkpoints/model_ep1000.pth --episodes 10 --render
"""

import argparse
import sys

import numpy as np

from environment import DonkeyKongEnv
from dqn_agent   import DQNAgent


def test(checkpoint_path, num_episodes=5, render=True):
    env   = DonkeyKongEnv(render=render)
    agent = DQNAgent(action_size=env.action_space_size)
    agent.load(checkpoint_path)

    # Disable all exploration — the agent always picks its best known action
    agent.epsilon = 0.0

    print(f"\nRunning {num_episodes} test episodes (ε=0 greedy policy)\n")

    results = []
    for episode in range(1, num_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        steps        = 0
        done         = False
        info         = {}

        while not done:
            action                        = agent.select_action(state)
            state, reward, done, info     = env.step(action)
            total_reward += reward
            steps        += 1

        lives   = info.get('lives', 0)
        score   = info.get('score', 0)
        survived = steps > env.action_space_size  # basic sanity check
        results.append({'reward': total_reward, 'steps': steps,
                        'lives': lives, 'score': score})

        status = "SURVIVED" if lives > 0 else "DIED"
        print(f"Episode {episode:3d}  {status}  |  "
              f"Reward: {total_reward:8.2f}  |  "
              f"Score: {score:6d}  |  "
              f"Steps: {steps:5d}  |  "
              f"Lives left: {lives}")

    print(f"\n{'─'*50}")
    print(f"Results over {num_episodes} episodes:")
    print(f"  Avg Reward  : {np.mean([r['reward'] for r in results]):.2f}")
    print(f"  Avg Score   : {np.mean([r['score']  for r in results]):.0f}")
    print(f"  Avg Steps   : {np.mean([r['steps']  for r in results]):.0f}")
    print(f"  Survival Rate: {np.mean([r['lives'] > 0 for r in results])*100:.1f}%")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',  type=str,
                        nargs='?', default='checkpoints/model_ep500.pth')
    parser.add_argument('--episodes',  type=int,  default=5)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()

    test(
        checkpoint_path = args.checkpoint,
        num_episodes    = args.episodes,
        render          = not args.no_render,
    )
