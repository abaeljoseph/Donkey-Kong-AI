"""
PPO training using stable-baselines3.
Uses SubprocVecEnv from SB3 for parallel data collection (more sample-efficient
than DQN and easier to parallelise with SB3's built-in vec env support).

Run via main.py:
    python main.py --algo ppo --timesteps 1000000
    python main.py --algo ppo --timesteps 500000 --render
    python main.py --algo ppo --timesteps 0 --load-model saved_models/ppo_final --eval-only
"""

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.gym_wrapper import DonkeyKongGymEnv
from environment.ghost_viewer import GhostViewerCallback
from environment.pybullet_arm import RobotArm
from evaluation.metrics import MetricsTracker


CUSTOM_INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'retro_data')


def _make_env_fn(render=False, rank=0, ghost_viewer=False):
    path = os.path.abspath(CUSTOM_INTEGRATION_PATH)
    def _fn():
        env = DonkeyKongGymEnv(
            render=(render and rank == 0),
            custom_integration_path=path,
            provide_frame=(ghost_viewer and rank == 0),   # full frame for background
            provide_detect=ghost_viewer,                  # detect frame from ALL envs
        )
        return Monitor(env)
    return _fn


class ArmMirrorCallback(BaseCallback):
    """Mirrors env-0 actions to the PyBullet robot arm each step."""

    def __init__(self, arm: RobotArm, verbose=0):
        super().__init__(verbose)
        self._arm = arm

    def _on_step(self) -> bool:
        action = self.locals['actions'][0]
        self._arm.step(int(action))
        return True


class MetricsCallback(BaseCallback):
    """Collects per-episode reward/steps/deaths into MetricsTracker."""

    def __init__(self, metrics: MetricsTracker, verbose=0):
        super().__init__(verbose)
        self._metrics = metrics

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            ep = info.get('episode')
            if ep is not None:
                self._metrics.log_episode(
                    reward=float(ep['r']),
                    steps=int(ep['l']),
                    deaths=0,
                    loss=0.0,
                    eps=0.0,
                )
        return True


def train_ppo(
    total_timesteps: int = 1_000_000,
    num_envs: int        = 4,
    render: bool         = False,
    arm_gui: bool        = False,
    load_model: str      = None,
    eval_only: bool      = False,
    save_dir: str        = 'saved_models',
    save_freq: int       = 50_000,
    ghost_viewer: bool   = True,
) -> MetricsTracker:

    os.makedirs(save_dir, exist_ok=True)

    env_fns = [_make_env_fn(render=render, rank=i, ghost_viewer=ghost_viewer) for i in range(num_envs)]
    # gym_wrapper already outputs (C,H,W) channels-first; normalize_images=False tells SB3
    vec_env = SubprocVecEnv(env_fns)

    arm     = RobotArm(gui=arm_gui)
    metrics = MetricsTracker()

    callbacks = [
        ArmMirrorCallback(arm),
        MetricsCallback(metrics),
        CheckpointCallback(
            save_freq=max(save_freq // num_envs, 1),
            save_path=save_dir,
            name_prefix='ppo',
        ),
    ]

    if ghost_viewer:
        callbacks.append(GhostViewerCallback(num_envs=num_envs))

    if load_model and os.path.exists(load_model + '.zip'):
        print(f'Loading PPO checkpoint: {load_model}')
        model = PPO.load(load_model, env=vec_env)
    else:
        model = PPO(
            policy='CnnPolicy',
            env=vec_env,
            policy_kwargs=dict(normalize_images=False),
            learning_rate=2.5e-4,
            n_steps=128,          # steps per env before each update
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,        # encourages exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, 'tb_logs'),
        )

    if not eval_only:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=load_model is None,
        )
        final_path = os.path.join(save_dir, 'ppo_final')
        model.save(final_path)
        print(f'Saved final model → {final_path}.zip')

    vec_env.close()
    arm.close()

    return metrics
