"""
PPO training using stable-baselines3.
Uses SubprocVecEnv from SB3 for parallel data collection (more sample-efficient
than DQN and easier to parallelise with SB3's built-in vec env support).

Run via main.py:
    python main.py --algo ppo --timesteps 10000000 --no-ghost --num-envs 8
    python main.py --algo ppo --timesteps 10000000 --load-model saved_models/ppo_XXXXXX_steps --no-ghost --num-envs 8
    python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1
"""

import os
import torch

# PyTorch 2.6 changed torch.load default to weights_only=True, which blocks
# numpy globals in checkpoints saved by older stable-baselines3.
# Patch to restore weights_only=False for our own trusted checkpoints.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


def _best_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.gym_wrapper import DonkeyKongGymEnv
from environment.donkey_kong_env import MARIO_Y_START, auto_detect_all_ladders
from environment.ghost_viewer import GhostViewerCallback
from environment.pybullet_arm import RobotArm
from evaluation.metrics import MetricsTracker


CUSTOM_INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'retro_data')


def _make_env_fn(render=False, rank=0, ghost_viewer=False, use_checkpoints=False, force_highest=False):
    path = os.path.abspath(CUSTOM_INTEGRATION_PATH)
    def _fn():
        env = DonkeyKongGymEnv(
            render=(render and rank == 0),
            custom_integration_path=path,
            provide_frame=(ghost_viewer and rank == 0),
            provide_detect=ghost_viewer,
            use_checkpoints=use_checkpoints,
            force_highest=force_highest,
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


class HeightCallback(BaseCallback):
    """Logs Mario's height climbed per episode to TensorBoard, per env and as a mean."""

    def __init__(self, num_envs: int, verbose=0):
        super().__init__(verbose)
        self._num_envs   = num_envs
        self._ep_best_y  = [999] * num_envs
        self._ep_start_y = [None] * num_envs

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            mario_y = info.get('_mario_y')
            if mario_y is not None:
                if self._ep_start_y[i] is None:
                    self._ep_start_y[i] = mario_y
                if mario_y < self._ep_best_y[i]:
                    self._ep_best_y[i] = mario_y

            if done and self._ep_start_y[i] is not None:
                height_px = max(0, self._ep_start_y[i] - self._ep_best_y[i])
                self.logger.record_mean(f'height/env_{i:02d}', height_px)
                self.logger.record_mean('height/mean', height_px)
                self._ep_best_y[i]  = 999
                self._ep_start_y[i] = None

        return True


class MetricsCallback(BaseCallback):
    """Collects per-episode reward/steps/deaths/wins into MetricsTracker."""

    def __init__(self, metrics: MetricsTracker, num_envs: int, verbose=0):
        super().__init__(verbose)
        self._metrics    = metrics
        self._prev_lives = [None] * num_envs
        self._ep_deaths  = [0]    * num_envs

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for i, info in enumerate(infos):
            lives = info.get('lives')
            if lives is not None:
                prev = self._prev_lives[i]
                if prev is not None and lives < prev:
                    self._ep_deaths[i] += 1
                self._prev_lives[i] = lives

            ep = info.get('episode')
            if ep is not None:
                self._metrics.log_episode(
                    reward=float(ep['r']),
                    steps=int(ep['l']),
                    deaths=self._ep_deaths[i],
                    won=bool(info.get('_won', False)),
                    loss=0.0,
                    eps=0.0,
                )
                self._ep_deaths[i]  = 0
                self._prev_lives[i] = None
        return True


def train_ppo(
    total_timesteps: int  = 1_000_000,
    num_envs: int         = 4,
    render: bool          = False,
    use_arm: bool         = False,
    arm_gui: bool         = False,
    arm_renderer: str     = 'pybullet',
    use_real_arms: bool   = False,
    load_model: str       = None,
    eval_only: bool       = False,
    save_dir: str         = 'saved_models',
    save_freq: int        = 50_000,
    ghost_viewer: bool    = True,
    use_checkpoints: bool = False,
    force_highest: bool   = False,
) -> MetricsTracker:

    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    integration_path = os.path.abspath(CUSTOM_INTEGRATION_PATH)
    all_ladders  = auto_detect_all_ladders(integration_path)

    # Clear stale suspension flag from any previous run so envs don't start
    # suspended and fall back to ground floor on the very first episode.
    from environment.donkey_kong_env import _best_suspended_path
    stale_flag = _best_suspended_path()
    if os.path.exists(stale_flag):
        os.remove(stale_flag)
        print('[Checkpoint] Cleared stale suspension flag from previous run')

    env_fns = [_make_env_fn(render=render, rank=i, ghost_viewer=ghost_viewer,
                            use_checkpoints=use_checkpoints, force_highest=force_highest)
               for i in range(num_envs)]
    raw_env = SubprocVecEnv(env_fns)

    vecnorm_path = os.path.join(save_dir, 'vecnorm.pkl')
    if not eval_only and load_model and os.path.exists(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, raw_env)
            vec_env.training = True
        except AssertionError:
            raw_env.close()
            raise SystemExit(
                f'[Error] vecnorm.pkl obs shape does not match the current environment.\n'
                f'        The model was trained with a different number of observation channels.\n'
                f'        Delete saved_models/vecnorm.pkl if you want to start fresh reward normalisation.'
            )
    else:
        vec_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if use_real_arms:
        from environment.real_robot_arm import RealRobotArms
        arm = RealRobotArms()
    elif use_arm:
        arm = RobotArm(gui=arm_gui, software_viewer=(arm_gui and arm_renderer == 'software'))
    else:
        arm = None
    metrics = MetricsTracker()

    callbacks = [
        MetricsCallback(metrics, num_envs=num_envs),
        HeightCallback(num_envs=num_envs),
        CheckpointCallback(
            save_freq=max(save_freq // num_envs, 1),
            save_path=save_dir,
            name_prefix='ppo',
        ),
    ]

    if use_arm:
        callbacks.append(ArmMirrorCallback(arm))

    if ghost_viewer:
        callbacks.append(GhostViewerCallback(num_envs=num_envs,
                                              all_ladders=all_ladders))

    if load_model:
        if not os.path.exists(load_model + '.zip'):
            vec_env.close()
            raise SystemExit(f'[Error] Model not found: {load_model}.zip\n'
                             f'        Check the path and try again.')
        print(f'Loading PPO checkpoint: {load_model}')
        try:
            model = PPO.load(load_model, env=vec_env, device=_best_device())
        except (AssertionError, ValueError) as e:
            vec_env.close()
            raise SystemExit(
                f'[Error] Model is incompatible with the current environment.\n'
                f'        Likely cause: observation shape changed (e.g. OBS_CHANNELS).\n'
                f'        Details: {e}'
            )
        model.tensorboard_log = os.path.join(save_dir, 'tb_logs')
    else:
        model = PPO(
            policy='CnnPolicy',
            env=vec_env,
            device=_best_device(),
            policy_kwargs=dict(
                normalize_images=False,
                features_extractor_kwargs=dict(features_dim=512),
            ),
            learning_rate=3e-4,
            n_steps=512,
            batch_size=2048,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
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
        ep_deaths, ep_prev_lives, ep_won = 0, None, False
        best_ever_y, best_ever_ep = 999, 0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = vec_env.step(action)
            if arm is not None:
                raw_frame = infos[0].get('_raw_frame')
                if raw_frame is not None and hasattr(arm, 'set_game_frame'):
                    arm.set_game_frame(raw_frame)
                arm.step(int(action[0]))
            ep_reward += float(rewards[0])
            mario_y = infos[0].get('mario_y', 999)
            lives   = infos[0].get('lives')
            if mario_y < ep_best_y:
                ep_best_y = mario_y
            if infos[0].get('_won', False):
                ep_won = True
            if lives is not None and ep_prev_lives is not None and lives < ep_prev_lives:
                ep_deaths += 1
            ep_prev_lives = lives
            if dones[0]:
                ep_count += 1
                plat = platform_name(ep_best_y)
                note = '*** NEW BEST ***' if ep_best_y < best_ever_y else ''
                if ep_best_y < best_ever_y:
                    best_ever_y, best_ever_ep = ep_best_y, ep_count
                print(f'  {ep_count:4d}  {ep_reward:8.1f}  {plat:>20}  {note}')
                metrics.log_episode(reward=ep_reward, steps=0, deaths=ep_deaths,
                                    won=ep_won, loss=0.0, eps=0.0)
                ep_reward, ep_best_y = 0.0, 999
                ep_deaths, ep_prev_lives, ep_won = 0, None, False

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
