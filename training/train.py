"""
Training loop — vectorised multi-env DQN.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from environment.donkey_kong_env import DonkeyKongEnv, NUM_ACTIONS
from environment.pybullet_arm import RobotArm
from models.dqn_agent import DQNAgent
from training.vec_env import SubprocVecEnv
from evaluation.metrics import MetricsTracker


CUSTOM_INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'retro_data')


def _make_env_fn(render=False):
    path = os.path.abspath(CUSTOM_INTEGRATION_PATH)
    def _fn():
        return DonkeyKongEnv(render=render, custom_integration_path=path)
    return _fn


def train(
    num_episodes: int   = 1000,
    num_envs: int       = 4,
    render: bool        = False,
    arm_gui: bool       = False,
    load_model: str     = None,
    eval_only: bool     = False,
    save_dir: str       = 'saved_models',
    save_freq: int      = 100,
) -> MetricsTracker:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Parallel envs: {num_envs}')

    # Only render env 0
    env_fns = [_make_env_fn(render=render) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    arm     = RobotArm(gui=arm_gui)
    agent   = DQNAgent(
        num_actions=NUM_ACTIONS,
        obs_shape=(4, 84, 84),
        device=device,
    )
    metrics = MetricsTracker()

    if load_model and os.path.exists(load_model):
        agent.load(load_model)
        print(f'Loaded checkpoint: {load_model}')

    # Per-env episode accumulators
    ep_rewards = np.zeros(num_envs)
    ep_steps   = np.zeros(num_envs, dtype=int)
    ep_deaths  = np.zeros(num_envs, dtype=int)
    ep_losses  = [[] for _ in range(num_envs)]
    prev_lives = np.full(num_envs, 3)
    episode_count = 0

    states = vec_env.reset()   # (N, 4, 84, 84)

    try:
        pbar = tqdm(total=num_episodes, desc='Episodes')
        while episode_count < num_episodes:

            actions = agent.select_actions(states)   # (N,) — one batched forward pass
            arm.step(int(actions[0]))                 # arm mirrors env-0

            next_states, rewards, dones, infos = vec_env.step(actions)

            for i in range(num_envs):
                lives = infos[i].get('lives', prev_lives[i])
                if lives < prev_lives[i]:
                    ep_deaths[i] += 1
                prev_lives[i] = lives

                if not eval_only:
                    loss = agent.observe(states[i], int(actions[i]),
                                         float(rewards[i]), next_states[i], bool(dones[i]))
                    if loss is not None:
                        ep_losses[i].append(loss)

                ep_rewards[i] += rewards[i]
                ep_steps[i]   += 1

                if dones[i]:
                    avg_loss = float(np.mean(ep_losses[i])) if ep_losses[i] else 0.0
                    metrics.log_episode(
                        reward=float(ep_rewards[i]),
                        steps=int(ep_steps[i]),
                        deaths=int(ep_deaths[i]),
                        loss=avg_loss,
                        eps=agent.eps,
                    )
                    episode_count += 1
                    pbar.update(1)

                    ep_rewards[i] = 0.0
                    ep_steps[i]   = 0
                    ep_deaths[i]  = 0
                    ep_losses[i]  = []
                    prev_lives[i] = 3

                    if episode_count % save_freq == 0 and not eval_only:
                        path = os.path.join(save_dir, f'dqn_ep{episode_count}.pt')
                        agent.save(path)

                    if episode_count % 10 == 0:
                        avg_r = np.mean(metrics.episode_rewards[-10:])
                        pbar.set_postfix(avg_r=f'{avg_r:.1f}', eps=f'{agent.eps:.3f}')

                    if episode_count % 20 == 0:
                        from evaluation.plot_results import plot_all_metrics
                        plot_all_metrics(metrics)   # overwrites results/training_metrics.png

                    if episode_count >= num_episodes:
                        break

            states = next_states

        pbar.close()

    except KeyboardInterrupt:
        print('\nInterrupted — saving checkpoint...')
        if not eval_only and episode_count > 0:
            path = os.path.join(save_dir, f'dqn_ep{episode_count}_interrupted.pt')
            agent.save(path)
            print(f'Saved → {path}')

    finally:
        vec_env.close()
        arm.close()

    return metrics
