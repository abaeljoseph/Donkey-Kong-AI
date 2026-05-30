"""
Person 1 — Environment wrapper.
Wraps gym-retro's Donkey Kong NES, preprocesses frames, stacks 4 frames,
maps a small discrete action set, and computes a shaped reward.
"""

import collections
import os
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

# Broken ladder zones (x0, y0, x1, y1) in NES coords.
# Generated from debug_detection.py — press L to list all ladder coords, then identify pairs
# at the same X that are separated by a small gap. Update these if the ROM integration changes.
BROKEN_LADDER_ZONES = [
    (185,  95, 198, 130),   # right-side pair (L8 + L11)
    ( 74, 128,  87, 160),   # left-centre pair (L12 + L15)
    ( 91, 193, 104, 225),   # lower-left pair  (L18 + L20)
    (100,  60, 112,  95),   # upper-centre pair
]

# Directory where platform game-states are persisted to disk.
# All subprocesses share the same files so one env's discovery benefits the rest.
GAMESTATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'gamestates')


def _gamestate_path(i: int) -> str:
    return os.path.join(GAMESTATE_DIR, f'platform_{i}.bin')

def _best_gamestate_path() -> str:
    return os.path.join(GAMESTATE_DIR, 'platform_best.bin')

def _best_suspended_path() -> str:
    return os.path.join(GAMESTATE_DIR, 'best_suspended.flag')


def detect_broken_ladder_zones(hsv, h, w):
    """
    Broken ladder = two small teal stubs vertically separated by a small gap.
    Uses connected-component pairs instead of column scan — more robust to H-sprite gaps.
    Returns list of (x0, y0, x1, y1) covering both stubs, in NES coords (0-256, 0-224).
    """
    teal = cv2.inRange(hsv, np.array([75, 60, 80]), np.array([110, 255, 255]))
    teal[:int(h * 0.1),  :]              = 0   # exclude HUD
    teal[int(h * 0.85):, :int(w * 0.20)] = 0   # exclude oil-can corner

    MIN_SEG   = 3    # stub height at least 3 px
    MAX_SEG   = 20   # stub height at most 20 px (full ladders are 25-40 px)
    MIN_GAP   = 4    # vertical gap at least 4 px
    MAX_GAP   = 25   # vertical gap at most 25 px
    MIN_X_OVL = 3    # stubs must share at least 3 px of horizontal range

    n, _, stats, _ = cv2.connectedComponentsWithStats(teal, connectivity=8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        comps.append((
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        ))

    zones = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ax, ay, aw, ah = comps[i]
            bx, by, bw, bh = comps[j]
            if ay > by:   # ensure a is above b
                ax, ay, aw, ah, bx, by, bw, bh = bx, by, bw, bh, ax, ay, aw, ah
            if not (MIN_SEG <= ah <= MAX_SEG and MIN_SEG <= bh <= MAX_SEG):
                continue
            gap = by - (ay + ah)
            if not (MIN_GAP <= gap <= MAX_GAP):
                continue
            if min(ax + aw, bx + bw) - max(ax, bx) < MIN_X_OVL:
                continue
            zones.append((
                int(min(ax, bx)),
                int(ay),
                int(max(ax+aw, bx+bw)),
                int(by + bh),
            ))
    return zones

def auto_detect_all_ladders(custom_integration_path=None):
    """
    Capture one clean frame and return all ladder segment positions as NES coords.
    Returns list of (nes_x0, nes_y0, nes_x1, nes_y1).
    """
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make('DonkeyKong-Nes', state='1Player.GameA', inttype=inttype)
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    env.close()

    h, w = obs.shape[:2]
    hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    teal = cv2.inRange(hsv, np.array([75, 60, 80]), np.array([110, 255, 255]))
    teal[:int(h * 0.1), :]              = 0
    teal[int(h * 0.85):, :int(w * 0.20)] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(teal, connectivity=8)
    ladders = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 10:
            lx = stats[i, cv2.CC_STAT_LEFT]
            ly = stats[i, cv2.CC_STAT_TOP]
            lw = stats[i, cv2.CC_STAT_WIDTH]
            lh = stats[i, cv2.CC_STAT_HEIGHT]
            ladders.append((
                int(lx * 256 / w), int(ly * 224 / h),
                int((lx + lw) * 256 / w), int((ly + lh) * 224 / h),
            ))
    print(f'[Ladders] Detected {len(ladders)} ladder segments')
    return ladders


def auto_detect_broken_zones(custom_integration_path=None):
    """
    Spin up a throwaway retro env, grab one frame, run detect_broken_ladder_zones,
    and convert the result to NES coords. Called once in the main process before
    SubprocVecEnv is created so subprocesses never need to re-run it.
    Falls back to BROKEN_LADDER_ZONES if nothing is detected.
    """
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make('DonkeyKong-Nes', state='1Player.GameA', inttype=inttype)
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    env.close()

    h, w = obs.shape[:2]
    hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    raw  = detect_broken_ladder_zones(hsv, h, w)

    if not raw:
        print('[BrokenLadder] Auto-detection found no broken zones — using hardcoded fallback')
        return list(BROKEN_LADDER_ZONES)

    zones = [
        (int(z[0] * 256 / w), int(z[1] * 224 / h),
         int(z[2] * 256 / w), int(z[3] * 224 / h))
        for z in raw
    ]
    print(f'[BrokenLadder] Auto-detected {len(zones)} broken zones: {zones}')
    return zones


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
                 provide_frame=False, provide_detect=False,
                 use_checkpoints=False, force_highest=False,
                 broken_zones=None):
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

        self._use_checkpoints = use_checkpoints
        self._force_highest   = force_highest
        self._broken_zones    = broken_zones if broken_zones is not None else list(BROKEN_LADDER_ZONES)
        self._provide_frame   = provide_frame
        self._provide_detect = provide_detect
        self._frames        = collections.deque(maxlen=FRAME_STACK)
        self._prev_lives    = 3
        self._prev_mario_y  = MARIO_Y_START
        self._prev_mario_x  = 128
        self._step_count    = 0
        self._last_teal     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_barrel   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_fire     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._max_steps        = MAX_STEPS
        self._reset_mario_y    = MARIO_Y_START
        self._best_y           = MARIO_Y_START
        self._ep_best_y        = MARIO_Y_START
        os.makedirs(GAMESTATE_DIR, exist_ok=True)
        self._platform_states        = []
        self._checkpoint_fail_streak = [0] * len(PLATFORM_THRESHOLDS)
        self._last_checkpoint_idx    = -1
        for i in range(len(PLATFORM_THRESHOLDS)):
            path = _gamestate_path(i)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self._platform_states.append(f.read())
            else:
                self._platform_states.append(None)
        self._best_ever_state       = None
        self._best_ever_suspended   = False   # True → fall back to lower checkpoint
        self._best_ever_fail_streak = 0
        self._start_stage           = 0       # stage index at episode start (level 1 = 0)
        best_path = _best_gamestate_path()
        if os.path.exists(best_path):
            with open(best_path, 'rb') as f:
                self._best_ever_state = f.read()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        # Pick up any platform states saved to disk by other envs since this env last reset.
        for i in range(len(PLATFORM_THRESHOLDS)):
            if self._platform_states[i] is None:
                path = _gamestate_path(i)
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self._platform_states[i] = f.read()
        # Pick up best-ever state saved by any subprocess since last reset.
        if self._best_ever_state is None:
            best_path = _best_gamestate_path()
            if os.path.exists(best_path):
                with open(best_path, 'rb') as f:
                    self._best_ever_state = f.read()
        # Pick up suspension written by any other subprocess.
        if not self._best_ever_suspended and os.path.exists(_best_suspended_path()):
            self._best_ever_suspended = True

        # Sample a checkpoint to start from.
        if not self._use_checkpoints:
            chosen_idx, chosen = -1, None
        elif self._force_highest:
            # Use the best-ever file when available and not suspended.
            # Suspension kicks in after 3 instant deaths — falls back to the highest
            # threshold checkpoint so Mario can rebuild a safer high point.
            if self._best_ever_state is not None and not self._best_ever_suspended:
                chosen_idx, chosen = -1, self._best_ever_state
            else:
                chosen_idx, chosen = -1, None
                for i in range(len(PLATFORM_THRESHOLDS) - 1, -1, -1):
                    if self._platform_states[i] is not None:
                        chosen_idx, chosen = i, self._platform_states[i]
                        break
        else:
            pool = [(-1, None, CHECKPOINT_WEIGHTS[0])]
            for i, (state, weight) in enumerate(zip(self._platform_states, CHECKPOINT_WEIGHTS[1:])):
                if state is not None:
                    pool.append((i, state, weight))
            total      = sum(w for _, _, w in pool)
            chosen_pos = random.choices(range(len(pool)), [w / total for _, _, w in pool], k=1)[0]
            chosen_idx, chosen, _ = pool[chosen_pos]
        self._last_checkpoint_idx = chosen_idx
        actual_lives = 3
        _noop = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if chosen is not None:
            self.env.em.set_state(chosen)
            for _ in range(4):
                result = self.env.step(_noop)
            obs, _, _, _, noop_info = result
            actual_lives = noop_info.get('lives', 3)
        else:
            # 1Player.GameA starts with a ~150-frame intro animation where no input
            # is accepted. Advance past it so the policy has control from frame 1.
            for _ in range(200):
                result = self.env.step(_noop)
            obs, _, _, _, noop_info = result
            actual_lives = noop_info.get('lives', 3)

        self._start_stage  = noop_info.get('stage', 0)
        self._prev_lives   = actual_lives
        self._prev_mario_y = MARIO_Y_START
        self._step_count   = 0
        self._max_steps    = MAX_STEPS + random.randint(-200, 200)

        h, w      = obs.shape[:2]
        hsv       = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        gray      = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small     = cv2.resize(obs, (DETECT_W, DETECT_H))
        small_hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sh, sw    = small.shape[:2]

        self._prev_mario_x, self._prev_mario_y = self._detect_mario(small_hsv, sh, sw)
        self._reset_mario_y = self._prev_mario_y
        self._ep_best_y     = self._prev_mario_y

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
        stage    = info.get('stage',    self._start_stage)
        level    = info.get('level',    0)
        level_changed = (stage != self._start_stage)
        mario_x, mario_y = self._detect_mario(small_hsv, sh, sw)

        barrel_penalty, barrel_at_level = self._barrel_proximity_penalty(hsv, mario_y, h)
        fire_penalty                    = self._fire_proximity_penalty(hsv, mario_y, h)
        oil_can_penalty                 = self._oil_can_penalty(mario_x, mario_y)
        dy_this_step                    = self._prev_mario_y - mario_y
        is_new_ep_height                = mario_y < self._ep_best_y
        self._ep_best_y                 = min(self._ep_best_y, mario_y)
        ladder_bonus                    = self._ladder_climbing_bonus(hsv, action_idx, mario_y, h, dy_this_step, is_new_ep_height)
        ladder_approach                 = self._intact_ladder_proximity_bonus(hsv, mario_x, mario_y, h, w)
        in_broken_zone = any(
            z[0] <= mario_x <= z[2] and z[1] <= mario_y <= z[3]
            for z in self._broken_zones
        )
        broken_ladder_penalty = -50.0 if (action_idx == 3 and in_broken_zone) else 0.0

        # Only count barrels that are at Mario's level (not above on a higher platform)
        # as jump-relevant danger. Barrels above still apply their proximity penalty
        # but should not unlock the height-reward unsuppression for jumping.
        danger_nearby = barrel_at_level or fire_penalty < -0.01

        # New height record — save checkpoint only when standing on a platform (not a ladder).
        # _best_y is only advanced when we actually save, so if Mario is on a ladder at the
        # threshold height, we keep retrying until he's standing on the platform proper.
        if self._use_checkpoints and mario_y < self._best_y and mario_y > MARIO_Y_WIN and lives >= self._prev_lives and not gameover:
            cx0 = int(max(0,   mario_x - 20) * w / 256)
            cx1 = max(cx0 + 1, int(min(256, mario_x + 20) * w / 256))
            fy0 = int(mario_y * h / 224)
            fy1 = max(fy0 + 1, min(int((mario_y + 16) * h / 224), h))
            on_ladder = cv2.countNonZero(cv2.inRange(
                hsv[fy0:fy1, cx0:cx1],
                np.array([83, 220, 190]), np.array([95, 255, 255]))) >= 2
            if not on_ladder:
                self._best_y = mario_y   # advance only on a successful save
                state = None
                for i, threshold in enumerate(PLATFORM_THRESHOLDS):
                    if mario_y < threshold and self._platform_states[i] is None:
                        if state is None:
                            state = self.env.em.get_state()
                        self._platform_states[i] = state
                        path = _gamestate_path(i)
                        if not os.path.exists(path):
                            with open(path, 'wb') as f:
                                f.write(state)
                # Always update the best-ever file so --force-highest stays current.
                # Also clear any suspension — Mario found a new safe high point.
                if state is None:
                    state = self.env.em.get_state()
                self._best_ever_state       = state
                self._best_ever_suspended   = False
                self._best_ever_fail_streak = 0
                with open(_best_gamestate_path(), 'wb') as f:
                    f.write(state)
                # Remove suspension flag so all subprocesses resume best-ever spawning.
                flag = _best_suspended_path()
                if os.path.exists(flag):
                    os.remove(flag)

        # --force-highest: after 3 quick deaths suspend best-ever spawn and fall
        # back to the nearest lower threshold checkpoint until a new high is found.
        if self._use_checkpoints and self._force_highest and lives < self._prev_lives:
            if self._step_count <= 15:   # ~2 seconds at FRAME_SKIP=8, 60fps
                self._best_ever_fail_streak += 1
                if self._best_ever_fail_streak >= 3:
                    self._best_ever_suspended   = True
                    self._best_ever_fail_streak = 0
                    # Write flag so all other subprocesses suspend on their next reset.
                    with open(_best_suspended_path(), 'w') as f:
                        f.write('1')
            else:
                self._best_ever_fail_streak = 0

        # Track early deaths — 3 consecutive instant deaths → delete that checkpoint.
        # Disabled in FORCE mode so the agent keeps retrying the same spot.
        if self._use_checkpoints and not self._force_highest and lives < self._prev_lives and self._last_checkpoint_idx >= 0:
            idx = self._last_checkpoint_idx
            if self._step_count <= 30:
                self._checkpoint_fail_streak[idx] += 1
                if self._checkpoint_fail_streak[idx] >= 3:
                    self._platform_states[idx] = None
                    path = _gamestate_path(idx)
                    if os.path.exists(path):
                        os.remove(path)
                    self._checkpoint_fail_streak[idx] = 0
                    self._last_checkpoint_idx = -1
            else:
                self._checkpoint_fail_streak[idx] = 0

        height_mult = 1.0 if (action_idx not in JUMP_ACTIONS or danger_nearby) else 0.0

        reward, won = self._compute_reward(lives, mario_y, gameover, height_mult)
        won = won or level_changed   # stage register changed → Mario beat level 1
        reward += barrel_penalty + fire_penalty + oil_can_penalty + ladder_bonus + ladder_approach + broken_ladder_penalty
        if action_idx in JUMP_ACTIONS and not danger_nearby:
            reward -= 0.15

        done = bool(gameover) or won or self._step_count >= self._max_steps or lives < self._prev_lives

        self._prev_lives   = lives
        self._prev_mario_y = mario_y
        self._prev_mario_x = mario_x
        info['_mario_y']      = mario_y
        info['_stage']        = stage
        info['_level']        = level
        info['_step_reward']  = reward
        info['_won']          = won

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

    def _detect_mario(self, small_hsv, sh, sw):
        """
        Detect Mario's (x, y) NES position from a precomputed small HSV frame.
        Falls back to previous values if detection fails.
        """
        skin = cv2.inRange(small_hsv, np.array([  0,  45, 215]), np.array([ 12, 110, 255]))
        blue = cv2.inRange(small_hsv, np.array([115, 230, 130]), np.array([125, 255, 210]))
        red  = cv2.inRange(small_hsv, np.array([  0, 200, 180]), np.array([ 12, 255, 240]))

        hud_cut = int(sh * 0.106)
        skin[:hud_cut, :] = 0; blue[:hud_cut, :] = 0; red[:hud_cut, :] = 0
        dy0, dy1 = int(sh * 0.078), int(sh * 0.254)
        dx1 = int(sw * 0.322)
        skin[dy0:dy1, :dx1] = 0; blue[dy0:dy1, :dx1] = 0; red[dy0:dy1, :dx1] = 0

        k = np.ones((14, 14), np.uint8)
        region = cv2.bitwise_and(cv2.dilate(skin, k),
                 cv2.bitwise_and(cv2.dilate(blue, k), cv2.dilate(red, k)))
        pixels = cv2.bitwise_and(skin, region)

        ys, xs = np.where(pixels > 0)
        if len(ys) < 2:
            return self._prev_mario_x, self._prev_mario_y
        mario_y_nes = int(np.median(ys) * 224 / sh)
        mario_x_nes = int(np.median(xs) * 256 / sw)
        return mario_x_nes, mario_y_nes

    def _ladder_climbing_bonus(self, hsv, action_idx, mario_y, h, dy, is_new_ep_height):
        """
        +3.0 bonus for actively climbing an intact ladder to a new episode height.
        Gated on is_new_ep_height so Mario cannot farm by oscillating up-down on
        the same ladder section — the bonus only fires at each new personal best.
        """
        if not is_new_ep_height:
            return 0.0
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
                return 3.0
        return 0.0

    def _barrel_proximity_penalty(self, hsv, mario_y, h):
        """
        Return (penalty, at_level) where penalty is negative when a barrel is within
        ~15 NES pixels vertically, and at_level is True only when the closest barrel
        is at or below Mario's Y (same platform or falling toward him).
        Barrels on a platform above Mario still incur the penalty but are NOT flagged
        as at_level — they should not unlock the jump reward unsuppression.
        """
        w = hsv.shape[1]
        orange = cv2.inRange(hsv, np.array([10, 160, 220]), np.array([25, 215, 255]))
        dk_y0 = int(h * 0.078);  dk_y1 = int(h * 0.254)
        dk_x1 = int(w * 0.322)
        orange[dk_y0:dk_y1, :dk_x1] = 0

        ys, _ = np.where(orange > 0)
        if len(ys) == 0:
            return 0.0, False
        barrel_nes_ys = ys * 224.0 / h
        dists = np.abs(barrel_nes_ys - mario_y)
        closest_idx = int(dists.argmin())
        closest = float(dists[closest_idx])
        if closest < 15:
            proximity_factor = 1.0 - (closest / 15.0)
            # at_level: barrel is at Mario's Y or within 8px below him (same platform lane)
            at_level = float(barrel_nes_ys[closest_idx]) >= mario_y - 8
            return -0.2 * proximity_factor, at_level
        return 0.0, False

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

    def _intact_ladder_proximity_bonus(self, hsv, mario_x, mario_y, h, w):
        """
        Small reward for being near an intact (non-broken) ladder.
        Broken zone pixels are masked out so proximity to broken sections gives nothing.
        Up to +0.3 when touching, linearly falling to 0 at 20 NES pixels distance.
        """
        teal = cv2.inRange(hsv, np.array([75, 60, 80]), np.array([110, 255, 255]))
        teal[:int(h * 0.1), :] = 0   # exclude HUD

        # Mask out broken zones (NES coords → frame pixel coords)
        for z in self._broken_zones:
            py0 = max(0, int(z[1] * h / 224))
            py1 = min(h, int(z[3] * h / 224))
            px0 = max(0, int(z[0] * w / 256))
            px1 = min(w, int(z[2] * w / 256))
            if py1 > py0 and px1 > px0:
                teal[py0:py1, px0:px1] = 0

        ys, xs = np.where(teal > 0)
        if len(ys) == 0:
            return 0.0

        ladder_nes_ys = ys * 224.0 / h
        ladder_nes_xs = xs * 256.0 / w
        dists   = np.sqrt((ladder_nes_xs - mario_x) ** 2 + (ladder_nes_ys - mario_y) ** 2)
        closest = dists.min()
        PROXIMITY   = 20
        ON_LADDER   = 5   # within 5 NES px = already on the ladder, no approach bonus
        if closest < ON_LADDER:
            return 0.0
        if closest < PROXIMITY:
            return 0.3 * (1.0 - closest / PROXIMITY)
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

    def _oil_can_penalty(self, mario_x, mario_y):
        """Penalty for approaching the oil can (bottom-left, fixed NES position). Kills on contact."""
        OIL_X, OIL_Y = 28, 200
        dist = ((mario_x - OIL_X) ** 2 + (mario_y - OIL_Y) ** 2) ** 0.5
        if dist < 40:
            return -1.0 * (1.0 - dist / 40.0)   # up to -1.0 at contact
        return 0.0

    def _compute_reward(self, lives, mario_y, gameover, height_mult=1.0):
        reward = 0.0

        dy = self._prev_mario_y - mario_y   # positive = climbed upward

        # Height reward scales with how high Mario already is:
        # near bottom each pixel = 3pts, near top each pixel = 10pts
        # height_mult = 0 suppresses this for pointless jumps (no danger nearby)
        height_progress = max(0.0, min(1.0, (MARIO_Y_START - mario_y) / TOTAL_HEIGHT))
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
