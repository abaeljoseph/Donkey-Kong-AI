"""
Person 1 — Environment wrapper.
Wraps gym-retro's Donkey Kong NES, preprocesses frames, stacks 4 frames,
maps a small discrete action set, and computes a shaped reward.
"""

import collections
import numpy as np
import cv2
import retro


# NES button layout in stable-retro: B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A
DISCRETE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],   # 0: NOOP
    [0, 0, 0, 0, 0, 0, 1, 0, 0],   # 1: LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],   # 2: RIGHT
    [0, 0, 0, 0, 1, 0, 0, 0, 0],   # 3: UP  (climb ladder)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],   # 4: DOWN (climb down)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],   # 5: JUMP (B)
    [1, 0, 0, 0, 0, 0, 1, 0, 0],   # 6: JUMP + LEFT
    [1, 0, 0, 0, 0, 0, 0, 1, 0],   # 7: JUMP + RIGHT
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)

FRAME_H     = 84
FRAME_W     = 84
FRAME_STACK = 4
FRAME_SKIP  = 4      # hold each action for N frames (4x speedup)
MAX_STEPS   = 500    # decision steps per episode (= 2000 emulator frames, ~33s game time)

# Donkey Kong NES level 1: Mario starts near Y=176, princess is near Y=22.
# Y decreases as Mario climbs (NES screen origin is top-left).
MARIO_Y_START = 176
MARIO_Y_WIN   = 30   # reaching this Y or lower = level complete

# Ghost viewer frame settings
DETECT_W          = 160   # detection frame width  (half NES res, keeps Mario ≥3 skin px)
DETECT_H          = 150   # detection frame height
FRAME_SEND_EVERY  = 5     # only send colour frames every N steps (reduces pipe traffic)


class DonkeyKongEnv:
    """
    Wraps the retro Donkey Kong NES environment.
    Starts from the official 1Player.GameA save state (level 1, gameplay).
    Returns stacked grayscale frames as observations.
    """

    def __init__(self, render=False, custom_integration_path=None,
                 provide_frame=False, provide_detect=False):
        if custom_integration_path:
            retro.data.Integrations.add_custom_path(custom_integration_path)
            inttype = retro.data.Integrations.CUSTOM_ONLY
        else:
            inttype = retro.data.Integrations.DEFAULT

        self.env = retro.make(
            game='DonkeyKong-Nes',
            state='1Player.GameA',
            inttype=inttype,
            render_mode='human' if render else None,
        )

        self.action_space_n = NUM_ACTIONS
        self.observation_shape = (FRAME_STACK, FRAME_H, FRAME_W)

        self._provide_frame  = provide_frame
        self._provide_detect = provide_detect
        self._frames        = collections.deque(maxlen=FRAME_STACK)
        self._prev_lives    = 3
        self._prev_mario_y  = MARIO_Y_START
        self._step_count    = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        self._prev_lives   = 3
        self._prev_mario_y = MARIO_Y_START
        self._step_count   = 0

        frame = self._preprocess(obs)
        for _ in range(FRAME_STACK):
            self._frames.append(frame)

        return self._get_state()

    def step(self, action_idx):
        buttons = DISCRETE_ACTIONS[action_idx]

        raw_obs, obs, info = None, None, {}
        for _ in range(FRAME_SKIP):
            result = self.env.step(buttons)
            raw_obs, _, terminated, truncated, info = result
        obs = raw_obs

        self._step_count += 1
        self._frames.append(self._preprocess(obs))

        lives    = info.get('lives',    self._prev_lives)
        mario_y  = info.get('mario_y',  self._prev_mario_y)
        gameover = info.get('gameover', 0)

        reward, won = self._compute_reward(lives, mario_y, gameover)

        done = bool(gameover) or won or self._step_count >= MAX_STEPS

        self._prev_lives   = lives
        self._prev_mario_y = mario_y

        # Throttled colour frames for ghost viewer (avoids flooding the pipe)
        if self._step_count % FRAME_SEND_EVERY == 0:
            if self._provide_frame:
                info['_raw_frame'] = obs   # full res for background display (env 0)
            if self._provide_detect:
                info['_detect_frame'] = cv2.resize(obs, (DETECT_W, DETECT_H))

        return self._get_state(), reward, done, info

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _get_state(self):
        return np.array(self._frames, dtype=np.float32)

    def _compute_reward(self, lives, mario_y, gameover):
        reward = 0.0

        # Height gained: Y decreases as Mario climbs (NES origin = top-left)
        dy = self._prev_mario_y - mario_y   # positive = climbed upward
        reward += dy * 0.5

        # Win: reached the top of the level
        won = mario_y <= MARIO_Y_WIN
        if won:
            reward += 100.0

        # Time penalty — every step costs something, so faster = higher reward
        reward -= 0.1

        # Death penalty
        if lives < self._prev_lives:
            reward -= 15.0

        # Game over penalty
        if gameover:
            reward -= 30.0

        return reward, won
