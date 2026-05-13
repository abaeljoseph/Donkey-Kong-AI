"""
Person 1 — Environment wrapper.
Wraps gym-retro's Donkey Kong NES, preprocesses frames, stacks 4 frames,
maps a small discrete action set, and computes a shaped reward.
"""

import collections
import random
import numpy as np
import cv2
import retro


# NES button layout in stable-retro: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# In NES Donkey Kong, A button (index 8) is jump — B does nothing.
DISCRETE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],   # 0: NOOP
    [0, 0, 0, 0, 0, 0, 1, 0, 0],   # 1: LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],   # 2: RIGHT
    [0, 0, 0, 0, 1, 0, 0, 0, 0],   # 3: UP  (climb ladder)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],   # 4: DOWN (climb down)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],   # 5: JUMP (A)
    [0, 0, 0, 0, 0, 0, 1, 0, 1],   # 6: JUMP + LEFT  (A + LEFT)
    [0, 0, 0, 0, 0, 0, 0, 1, 1],   # 7: JUMP + RIGHT (A + RIGHT)
]
NUM_ACTIONS  = len(DISCRETE_ACTIONS)
JUMP_ACTIONS = frozenset({5, 6, 7})   # JUMP, JUMP+LEFT, JUMP+RIGHT

FRAME_H      = 84
FRAME_W      = 84
FRAME_STACK  = 4
OBS_CHANNELS = 7    # 4 grayscale + teal/ladder + barrel + fire
FRAME_SKIP   = 8    # hold each action for N frames
MAX_STEPS    = 1200 # decision steps per episode (randomised ±200 per env to stagger resets)

# Donkey Kong NES level 1: Mario starts near Y=176, princess is near Y=22.
# Y decreases as Mario climbs (NES screen origin is top-left).
MARIO_Y_START = 176
MARIO_Y_WIN   = 30   # reaching this Y or lower = level complete

TOTAL_HEIGHT = MARIO_Y_START - MARIO_Y_WIN   # ~146 NES pixels, full climb distance

# NES Y thresholds where a platform checkpoint is saved (Y decreases going up).
# One slot per platform — saved the first time Mario safely crosses each boundary.
PLATFORM_THRESHOLDS = [175, 140, 105, 70]   # platforms 2, 3, 4, 5
# Reset sampling weights: [ground, plat2, plat3, plat4, plat5]
CHECKPOINT_WEIGHTS  = [0.20, 0.15, 0.20, 0.25, 0.20]

# Ghost viewer frame settings
DETECT_W          = 160   # detection frame width  (half NES res, keeps Mario ≥3 skin px)
DETECT_H          = 150   # detection frame height
FRAME_SEND_EVERY  = 5     # full-colour background frame (larger — keep throttled)
DETECT_SEND_EVERY = 2     # detect frame (small — send more often to catch mid-climb)


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
        self.observation_shape = (OBS_CHANNELS, FRAME_H, FRAME_W)

        self._provide_frame  = provide_frame
        self._provide_detect = provide_detect
        self._frames        = collections.deque(maxlen=FRAME_STACK)
        self._prev_lives    = 3
        self._prev_mario_y  = MARIO_Y_START
        self._step_count    = 0
        self._last_teal     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_barrel   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_fire     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._max_steps     = MAX_STEPS
        self._reset_mario_y    = MARIO_Y_START
        self._best_y           = MARIO_Y_START   # best height reached (lower Y = higher up)
        self._platform_states  = [None] * len(PLATFORM_THRESHOLDS)   # checkpoint per platform

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        # Sample a checkpoint to start from — weighted toward higher platforms so the
        # agent drills every transition, not just the hardest one it barely reached.
        pool    = [(None, CHECKPOINT_WEIGHTS[0])]   # ground / fresh start always available
        for state, weight in zip(self._platform_states, CHECKPOINT_WEIGHTS[1:]):
            if state is not None:
                pool.append((state, weight))
        states  = [s for s, _ in pool]
        weights = [w for _, w in pool]
        total   = sum(weights)
        chosen  = random.choices(states, [w / total for w in weights], k=1)[0]
        if chosen is not None:
            self.env.em.set_state(chosen)
            result = self.env.step([0, 0, 0, 0, 0, 0, 0, 0, 0])   # NOOP to get obs
            obs = result[0]

        self._prev_lives   = 3
        self._prev_mario_y = MARIO_Y_START
        self._step_count   = 0
        self._max_steps    = MAX_STEPS + random.randint(-200, 200)

        h, w      = obs.shape[:2]
        hsv       = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        gray      = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small     = cv2.resize(obs, (DETECT_W, DETECT_H))
        small_hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sh, sw    = small.shape[:2]

        # Detect actual starting Y so the first step doesn't get a spurious height penalty
        # from the mismatch between hardcoded MARIO_Y_START and true detected position (~210).
        self._prev_mario_y  = self._detect_mario_y(small_hsv, sh, sw)
        self._reset_mario_y = self._prev_mario_y

        frame = self._preprocess_gray(gray)
        self._last_teal   = self._extract_teal(hsv)
        self._last_barrel = self._extract_barrel(hsv, h, w)
        self._last_fire   = self._extract_fire(hsv, h, w)
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

        # Precompute colour conversions once — shared by all detection and extraction methods.
        # Previously each method did its own cvtColor; this eliminates 6 redundant conversions
        # and 1 redundant resize per step.
        h, w      = obs.shape[:2]
        hsv       = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        gray      = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small     = cv2.resize(obs, (DETECT_W, DETECT_H))
        small_hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sh, sw    = small.shape[:2]

        self._last_teal   = self._extract_teal(hsv)
        self._last_barrel = self._extract_barrel(hsv, h, w)
        self._last_fire   = self._extract_fire(hsv, h, w)
        self._frames.append(self._preprocess_gray(gray))

        lives    = info.get('lives',    self._prev_lives)
        gameover = info.get('gameover', 0)
        mario_y  = self._detect_mario_y(small_hsv, sh, sw)

        barrel_penalty = self._barrel_proximity_penalty(hsv, mario_y, h)
        fire_penalty   = self._fire_proximity_penalty(hsv, mario_y, h)
        dy_this_step   = self._prev_mario_y - mario_y   # positive = moved up this step
        ladder_bonus   = self._ladder_climbing_bonus(hsv, action_idx, mario_y, h, dy_this_step)

        # Jump is only useful for dodging — suppress height reward and penalise
        # pointless jumping when no barrel or fire is nearby.
        danger_nearby = barrel_penalty < -0.01 or fire_penalty < -0.01

        # New height record — save checkpoint only if obstacles are far enough away
        # that Mario has time to react when this state is loaded. get_state() is only
        # called here (infrequent) not every step.
        if mario_y < self._best_y and lives >= self._prev_lives and not gameover:
            self._best_y = mario_y
            if self._is_safe_to_checkpoint(hsv, mario_y, h):
                state = None
                for i, threshold in enumerate(PLATFORM_THRESHOLDS):
                    if mario_y < threshold and self._platform_states[i] is None:
                        if state is None:
                            state = self.env.em.get_state()   # call get_state once
                        self._platform_states[i] = state
        height_mult   = 1.0 if (action_idx not in JUMP_ACTIONS or danger_nearby) else 0.0

        reward, won = self._compute_reward(lives, mario_y, gameover, height_mult)
        reward += barrel_penalty + fire_penalty + ladder_bonus
        if action_idx in JUMP_ACTIONS and not danger_nearby:
            reward -= 0.15   # active penalty for pointless jumping

        done = bool(gameover) or won or self._step_count >= self._max_steps

        self._prev_lives   = lives
        self._prev_mario_y = mario_y
        info['_mario_y']   = mario_y

        # Full-res color frame for ghost viewer background — throttled to reduce pipe load
        if self._provide_frame and self._step_count % FRAME_SEND_EVERY == 0:
            info['_raw_frame'] = obs

        # Reuse the already-resized small frame — no second resize needed
        if self._provide_detect and self._step_count % DETECT_SEND_EVERY == 0:
            info['_detect_frame'] = small

        return self._get_state(), reward, done, info

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_mario_y(self, small_hsv, sh, sw):
        """
        Detect Mario's Y position from a precomputed small HSV frame.
        Falls back to previous value if detection fails.
        Uses skin (low-S) + blue overalls + red hat all adjacent — same as ghost viewer.
        """
        skin = cv2.inRange(small_hsv, np.array([  0,  45, 215]), np.array([ 12, 110, 255]))
        blue = cv2.inRange(small_hsv, np.array([115, 230, 130]), np.array([125, 255, 210]))
        red  = cv2.inRange(small_hsv, np.array([  0, 200, 180]), np.array([ 12, 255, 240]))

        # Exclude HUD and DK zone
        hud_cut = int(sh * 0.106)
        skin[:hud_cut, :] = 0; blue[:hud_cut, :] = 0; red[:hud_cut, :] = 0
        dy0, dy1 = int(sh * 0.078), int(sh * 0.254)
        dx1 = int(sw * 0.322)
        skin[dy0:dy1, :dx1] = 0; blue[dy0:dy1, :dx1] = 0; red[dy0:dy1, :dx1] = 0

        k = np.ones((14, 14), np.uint8)
        region = cv2.bitwise_and(cv2.dilate(skin, k),
                 cv2.bitwise_and(cv2.dilate(blue, k), cv2.dilate(red, k)))
        pixels = cv2.bitwise_and(skin, region)

        ys, _ = np.where(pixels > 0)
        if len(ys) < 2:
            return self._prev_mario_y   # keep last known position

        # Convert detect-frame y back to NES screen y (0-240)
        mario_y_nes = int(np.median(ys) * 224 / sh)
        return mario_y_nes

    def _ladder_climbing_bonus(self, hsv, action_idx, mario_y, h, dy):
        """
        +3.0 bonus for actively climbing an intact ladder.
        Requires pressing UP, moving upward, and teal pixels present — AND
        checks that teal exists above Mario in a 40px window so broken ladder
        sections (no teal above the gap) give no bonus.
        Mario can still physically use broken ladders to dodge obstacles,
        he just gets no climbing reward for them.
        """
        hud_offset = int(h * 0.1)
        region_hsv = hsv[hud_offset:, :]
        teal = cv2.inRange(region_hsv, np.array([83, 220, 190]), np.array([95, 255, 255]))

        if action_idx == 3 and dy > 0 and cv2.countNonZero(teal) > 80:
            mario_y_px = int(mario_y * h / 224) - hud_offset
            # Check only 2–20px above Mario — working ladder rails are immediately
            # adjacent, broken ladder teal starts above the gap further up.
            above_top  = max(0, mario_y_px - 20)
            above_bot  = max(0, mario_y_px - 2)
            teal_above = cv2.countNonZero(teal[above_top:above_bot, :]) if above_bot > above_top else 0
            if teal_above >= 5:
                return 3.0   # intact ladder immediately above — reward climbing
        return 0.0

    def _barrel_proximity_penalty(self, hsv, mario_y, h):
        """
        Return a negative reward if a barrel is within ~15 NES pixels vertically
        of Mario. Barrels detected by their orange/amber body colour (H=15-35).
        Uses the raw RGB frame so no extra conversion needed.
        """
        w = hsv.shape[1]
        # Barrel orange body: sampled H=15,S=197,V=248
        orange = cv2.inRange(hsv,
                             np.array([10, 160, 220]),
                             np.array([25, 215, 255]))
        # Exclude DK zone (stacked barrels at top-left are not in play)
        dk_y0 = int(h * 0.078);  dk_y1 = int(h * 0.254)
        dk_x1 = int(w * 0.322)
        orange[dk_y0:dk_y1, :dk_x1] = 0

        ys, _ = np.where(orange > 0)
        if len(ys) == 0:
            return 0.0
        # Convert pixel y → NES y coords (frame is 240×224)
        barrel_nes_ys = ys * 224.0 / h
        # Soft repulsive gradient: closer barrel = larger penalty
        dists = np.abs(barrel_nes_ys - mario_y)
        closest = dists.min() if len(dists) else 999
        if closest < 15:
            proximity_factor = 1.0 - (closest / 15.0)   # 1.0 when touching, 0.0 at edge
            return -0.2 * proximity_factor
        return 0.0

    def _fire_proximity_penalty(self, hsv, mario_y, h):
        """
        Penalty when a fire/fireball is within ~20 NES pixels of Mario.
        Fire cream/low-saturation orange: H=12-25, S=50-130 (unique vs barrels S=160+).
        Tune HSV range using: python tools/debug_detection.py — click a fire pixel.
        """
        w = hsv.shape[1]
        # Fire cream/low-saturation orange — sampled HSV=(18,82,248)
        fire = cv2.inRange(hsv,
                           np.array([ 12,  50, 220]),
                           np.array([ 25, 130, 255]))
        dk_y0 = int(h * 0.078);  dk_y1 = int(h * 0.254)
        dk_x1 = int(w * 0.322)
        fire[dk_y0:dk_y1, :dk_x1] = 0

        ys, _ = np.where(fire > 0)
        if len(ys) == 0:
            return 0.0
        fire_nes_ys  = ys * 224.0 / h
        dists        = np.abs(fire_nes_ys - mario_y)
        closest      = dists.min()
        if closest < 20:
            return -0.2 * (1.0 - closest / 20.0)
        return 0.0

    def _preprocess_gray(self, gray):
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _extract_teal(self, hsv):
        """Binary ladder channel: 1.0 where teal ladder pixels are."""
        mask = cv2.inRange(hsv, np.array([83, 220, 190]), np.array([95, 255, 255]))
        resized = cv2.resize(mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.float32)

    def _extract_barrel(self, hsv, h, w):
        """Binary barrel channel: 1.0 where barrel orange pixels are."""
        mask = cv2.inRange(hsv, np.array([10, 160, 220]), np.array([25, 215, 255]))
        # Exclude DK zone (stacked barrels at top-left are not in play)
        dk_y0 = int(h * 0.078); dk_y1 = int(h * 0.254); dk_x1 = int(w * 0.322)
        mask[dk_y0:dk_y1, :dk_x1] = 0
        resized = cv2.resize(mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.float32)

    def _extract_fire(self, hsv, h, w):
        """Binary fire channel: 1.0 where fire/fireball pixels are."""
        mask = cv2.inRange(hsv, np.array([12, 50, 220]), np.array([25, 130, 255]))
        dk_y0 = int(h * 0.078); dk_y1 = int(h * 0.254); dk_x1 = int(w * 0.322)
        mask[dk_y0:dk_y1, :dk_x1] = 0
        resized = cv2.resize(mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.float32)

    def _get_state(self):
        gray_stack = np.array(self._frames, dtype=np.float32)   # (4, H, W)
        teal   = self._last_teal  [np.newaxis]                   # (1, H, W) ladders
        barrel = self._last_barrel[np.newaxis]                   # (1, H, W) barrels
        fire   = self._last_fire  [np.newaxis]                   # (1, H, W) fires
        return np.concatenate([gray_stack, teal, barrel, fire], axis=0)  # (7, H, W)

    def _is_safe_to_checkpoint(self, hsv, mario_y, h):
        """
        Returns True if no barrel or fire is within a safe distance of Mario.
        Uses a larger radius than the penalty threshold so Mario has reaction time
        when the checkpoint is loaded. Only called when a new height record is set.
        """
        w = hsv.shape[1]
        dk_y0 = int(h * 0.078); dk_y1 = int(h * 0.254); dk_x1 = int(w * 0.322)

        orange = cv2.inRange(hsv, np.array([10, 160, 220]), np.array([25, 215, 255]))
        orange[dk_y0:dk_y1, :dk_x1] = 0
        ys, _ = np.where(orange > 0)
        if len(ys) > 0:
            if np.abs(ys * 224.0 / h - mario_y).min() < 40:
                return False

        fire = cv2.inRange(hsv, np.array([12, 50, 220]), np.array([25, 130, 255]))
        fire[dk_y0:dk_y1, :dk_x1] = 0
        ys, _ = np.where(fire > 0)
        if len(ys) > 0:
            if np.abs(ys * 224.0 / h - mario_y).min() < 50:
                return False

        return True

    def _compute_reward(self, lives, mario_y, gameover, height_mult=1.0):
        reward = 0.0

        dy = self._prev_mario_y - mario_y   # positive = climbed upward

        # Height reward scales with how high Mario already is:
        # near bottom each pixel = 3pts, near top each pixel = 10pts
        # height_mult = 0 suppresses this for pointless jumps (no danger nearby)
        height_progress = max(0.0, min(1.0, (self._reset_mario_y - mario_y) / TOTAL_HEIGHT))
        climb_scale = 3.0 + height_progress * 7.0
        reward += dy * climb_scale * height_mult

        # Win — bonus scales with how fast Mario finished
        won = mario_y <= MARIO_Y_WIN
        if won:
            reward += 500.0 + (self._max_steps - self._step_count) * 0.5

        # Time penalty — discourages idling and hesitation
        reward -= 0.1

        # Death: small penalty — risk-taking to climb should be acceptable
        if lives < self._prev_lives:
            reward -= 3.0

        if gameover:
            reward -= 5.0

        return reward, won
