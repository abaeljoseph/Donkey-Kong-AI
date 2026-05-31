"""
Gymnasium-compatible wrapper around DonkeyKongEnv for stable-baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.donkey_kong_env import DonkeyKongEnv, NUM_ACTIONS, OBS_CHANNELS, FRAME_H, FRAME_W


class DonkeyKongGymEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, render=False, custom_integration_path=None,
                 provide_frame=False, provide_detect=False,
                 use_checkpoints=False, force_highest=False,
                 static_teal_mask=None, broken_zones=None,
                 game_name=None, game_state=None):
        super().__init__()
        self._env = DonkeyKongEnv(render=render, custom_integration_path=custom_integration_path,
                                  provide_frame=provide_frame, provide_detect=provide_detect,
                                  use_checkpoints=use_checkpoints, force_highest=force_highest,
                                  static_teal_mask=static_teal_mask, broken_zones=broken_zones,
                                  game_name=game_name, game_state=game_state)
        # Channels-first (C, H, W) — NatureCNN uses shape[0] as n_input_channels.
        # normalize_images=False is set in train_ppo so SB3 accepts float32 0-1.
        # Channel 5 is the teal/ladder binary mask so the CNN can explicitly see ladders.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_CHANNELS, FRAME_H, FRAME_W),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        return obs, reward, done, False, info

    def close(self):
        self._env.close()
