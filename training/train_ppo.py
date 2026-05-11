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
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


def linear_schedule(initial: float):
    def _fn(progress_remaining: float) -> float:
        return progress_remaining * initial
    return _fn

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
    use_arm: bool        = False,
    arm_gui: bool        = False,
    load_model: str      = None,
    eval_only: bool      = False,
    save_dir: str        = 'saved_models',
    save_freq: int       = 50_000,
    ghost_viewer: bool   = True,
) -> MetricsTracker:

    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    env_fns = [_make_env_fn(render=render, rank=i, ghost_viewer=ghost_viewer) for i in range(num_envs)]
    raw_env = SubprocVecEnv(env_fns)

    vecnorm_path = os.path.join(save_dir, 'vecnorm.pkl')
    if load_model and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, raw_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    arm     = RobotArm(gui=arm_gui) if use_arm else None
    metrics = MetricsTracker()

    callbacks = [
        MetricsCallback(metrics),
        CheckpointCallback(
            save_freq=max(save_freq // num_envs, 1),
            save_path=save_dir,
            name_prefix='ppo',
        ),
    ]

    if use_arm:
        callbacks.append(ArmMirrorCallback(arm))

    if ghost_viewer:
        callbacks.append(GhostViewerCallback(num_envs=num_envs))

    if load_model and os.path.exists(load_model + '.zip'):
        print(f'Loading PPO checkpoint: {load_model}')
        model = PPO.load(load_model, env=vec_env)
    else:
        model = PPO(
            policy='CnnPolicy',
            env=vec_env,
            device='cuda',
            policy_kwargs=dict(
                normalize_images=False,
                features_extractor_kwargs=dict(features_dim=512),
            ),
            learning_rate=linear_schedule(3e-4),
            n_steps=512,
            batch_size=2048,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, 'tb_logs'),
        )

    if eval_only:
        # NES Y thresholds for each platform (Y decreases going up)
        PLATFORMS = [
            (999, 195, 'Platform 1 (ground)'),
            (195, 160, 'Platform 2'),
            (160, 125, 'Platform 3'),
            (125,  90, 'Platform 4'),
            ( 90,  55, 'Platform 5'),
            ( 55,  30, 'Platform 6 (top)'),
            (  30,   0, 'WIN — reached Pauline!'),
        ]

        def platform_name(mario_y):
            for y_low, y_high, name in PLATFORMS:
                if mario_y < y_low:
                    return name
            return 'unknown'

        print('Running evaluation — press Ctrl+C to stop.')
        print(f'  {"ep":>4}  {"reward":>8}  {"best height":>20}  note')
        obs = vec_env.reset()
        ep_count, ep_reward, ep_best_y = 0, 0.0, 999
        best_ever_y, best_ever_ep = 999, 0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            mario_y = infos[0].get('_mario_y', 999)
            if mario_y < ep_best_y:
                ep_best_y = mario_y
            if dones[0]:
                ep_count += 1
                plat = platform_name(ep_best_y)
                note = '*** NEW BEST ***' if ep_best_y < best_ever_y else ''
                if ep_best_y < best_ever_y:
                    best_ever_y, best_ever_ep = ep_best_y, ep_count
                print(f'  {ep_count:4d}  {ep_reward:8.1f}  {plat:>20}  {note}')
                metrics.log_episode(reward=ep_reward, steps=0, deaths=0, loss=0.0, eps=0.0)
                ep_reward, ep_best_y = 0.0, 999

    else:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=load_model is None,
        )
        final_path = os.path.join(save_dir, 'ppo_final')
        model.save(final_path)
        vec_env.save(vecnorm_path)
        print(f'Saved final model → {final_path}.zip')

    vec_env.close()
    if arm:
        arm.close()

    return metrics
