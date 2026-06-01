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
OBS_CHANNELS = 7    # 4 grayscale + teal/ladder + barrel + fire  (set to 8 to add hammer — requires retraining from scratch)
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
    env = retro.make('DonkeyKong-Nes', state='1Player.GameA', inttype=inttype, render_mode=None)
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


def precompute_stage_maps(custom_integration_path=None):
    """
    Spin up one throwaway env, capture Stage 1's opening frame, and return:
      teal_mask    — (84, 84) float32 binary mask of ladder pixels (ready to drop into channel 4)
      broken_zones — list of (x0, y0, x1, y1) broken-ladder zones in NES coords

    Call this ONCE in the main process before SubprocVecEnv is created.
    Pass both values to each DonkeyKongEnv so reset() never has to re-scan —
    the ladder layout and broken zones are identical every episode.
    """
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make('DonkeyKong-Nes', state='1Player.GameA', inttype=inttype, render_mode=None)
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    env.close()

    h, w = obs.shape[:2]
    hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    # Teal mask — same HSV range and resize as _extract_teal
    raw_mask = cv2.inRange(hsv, np.array([83, 220, 190]), np.array([95, 255, 255]))
    resized  = cv2.resize(raw_mask, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
    teal_mask = (resized > 0).astype(np.float32)

    # Broken zones
    raw = detect_broken_ladder_zones(hsv, h, w)
    if raw:
        broken_zones = [
            (int(z[0] * 256 / w), int(z[1] * 224 / h),
             int(z[2] * 256 / w), int(z[3] * 224 / h))
            for z in raw
        ]
    else:
        broken_zones = list(BROKEN_LADDER_ZONES)

    print(f'[StageMaps] Teal pixels: {int(teal_mask.sum())}, '
          f'broken zones: {len(broken_zones)}: {broken_zones}')
    return teal_mask, broken_zones


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
                 static_teal_mask=None, broken_zones=None,
                 frame_send_every=None):
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

        self._use_checkpoints   = use_checkpoints
        self._force_highest     = force_highest
        # Static maps computed once at startup — if provided, reset() skips HSV scanning entirely.
        self._static_teal_mask  = static_teal_mask
        self._static_broken_zones = broken_zones  # None means detect dynamically at reset
        self._provide_frame   = provide_frame
        self._provide_detect = provide_detect
        # How often to attach the full-colour frame for the arm cabinet / ghost viewer.
        # 1 = every decision (smooth cabinet during eval); default keeps training light.
        self._frame_send_every = frame_send_every or FRAME_SEND_EVERY
        self._frames        = collections.deque(maxlen=FRAME_STACK)
        self._prev_lives    = 3
        self._prev_mario_y  = MARIO_Y_START
        self._prev_mario_x  = 128
        self._step_count    = 0
        self._last_teal     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_barrel   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_fire     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_hammer   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
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
        _noop = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if chosen is not None:
            self.env.em.set_state(chosen)
            for _ in range(4):
                result = self.env.step(_noop)
            obs, _, _, _, noop_info = result
        else:
            noop_info = result[4] if len(result) > 4 else {}
            obs = result[0]

        self._start_stage  = noop_info.get('stage', 0)
        self._prev_lives   = noop_info.get('lives', 3)
        self._prev_mario_y = MARIO_Y_START
        self._step_count   = 0
        self._max_steps    = MAX_STEPS + random.randint(-200, 200)

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        ram  = np.frombuffer(self.env.get_ram(), dtype=np.uint8)

        self._prev_mario_x, self._prev_mario_y = self._detect_mario_oam(ram)
        self._reset_mario_y = self._prev_mario_y
        self._ep_best_y     = self._prev_mario_y

        if self._static_teal_mask is not None:
            # Static maps precomputed at startup — no HSV scan needed.
            self._last_teal    = self._static_teal_mask
            self._broken_zones = self._static_broken_zones
        else:
            # Dynamic scan fallback (debug tool, or future multi-stage support).
            h, w = obs.shape[:2]
            hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
            detected_raw = detect_broken_ladder_zones(hsv, h, w)
            if detected_raw:
                self._broken_zones = [
                    (int(z[0] * 256 / w), int(z[1] * 224 / h),
                     int(z[2] * 256 / w), int(z[3] * 224 / h))
                    for z in detected_raw
                ]
            else:
                self._broken_zones = list(BROKEN_LADDER_ZONES)
            self._last_teal = self._extract_teal(hsv)

        frame = self._preprocess_gray(gray)
        self._last_barrel = self._extract_barrel_oam(ram)
        self._last_fire   = self._extract_fire_oam(ram)
        self._last_hammer = self._extract_hammer_oam(ram)
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

        gray  = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(obs, (DETECT_W, DETECT_H))
        ram   = np.frombuffer(self.env.get_ram(), dtype=np.uint8)

        self._last_barrel = self._extract_barrel_oam(ram)
        self._last_fire   = self._extract_fire_oam(ram)
        self._last_hammer = self._extract_hammer_oam(ram)
        self._frames.append(self._preprocess_gray(gray))

        lives    = info.get('lives',    self._prev_lives)
        gameover = info.get('gameover', 0)
        stage    = info.get('stage',    self._start_stage)
        level    = info.get('level',    0)
        mario_x, mario_y = self._detect_mario_oam(ram)

        barrel_penalty, barrel_at_level = self._barrel_proximity_penalty_oam(mario_y, ram)
        fire_penalty                    = self._fire_proximity_penalty_oam(mario_y, ram)
        dy_this_step                    = self._prev_mario_y - mario_y
        old_ep_best_y                   = self._ep_best_y
        mario_cx = mario_x + 8   # OAM X is left edge of 16px sprite; use centre for zone test
        mario_cy = mario_y + 8
        in_broken_zone = any(
            z[0] <= mario_cx <= z[2] and z[1] <= mario_cy <= z[3]
            for z in self._broken_zones
        )

        self._ep_best_y = min(self._ep_best_y, mario_y)
        if in_broken_zone and action_idx == 3:
            # Pressing UP through a broken ladder: suppress the active climbing bonus
            # but still credit height gained — climbing through is still possible but
            # always less rewarding than an intact ladder (no active bonus).
            ladder_bonus          = 0.3   # passive only
            broken_ladder_penalty = -2.0
        else:
            ladder_bonus          = self._ladder_climbing_bonus(action_idx, dy_this_step)
            broken_ladder_penalty = 0.0
        height_ref_y = old_ep_best_y

        # New height record — save checkpoint only when standing on a platform (not a ladder).
        # _best_y is only advanced when we actually save, so if Mario is on a ladder at the
        # threshold height, we keep retrying until he's standing on the platform proper.
        if self._use_checkpoints and mario_y < self._best_y and mario_y > MARIO_Y_WIN and lives >= self._prev_lives and not gameover:
            tx0 = max(0, int((mario_x - 20) * FRAME_W / 256))
            tx1 = min(FRAME_W, int((mario_x + 20) * FRAME_W / 256))
            ty0 = max(0, int(mario_y * FRAME_H / 224))
            ty1 = min(FRAME_H, int((mario_y + 16) * FRAME_H / 224) + 1)
            on_ladder = np.any(self._last_teal[ty0:ty1, tx0:tx1] > 0)
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

        reward, won = self._compute_reward(lives, mario_y, gameover, height_ref_y)
        reward += barrel_penalty + fire_penalty + ladder_bonus + broken_ladder_penalty

        done = bool(gameover) or won or self._step_count >= self._max_steps

        # Reset height tracker when a life is lost so each new life gets the same
        # reward signal as the first — otherwise later lives have no height headroom
        # and the model learns to play differently depending on which life it's on.
        if lives < self._prev_lives:
            self._ep_best_y = MARIO_Y_START

        self._prev_lives   = lives
        self._prev_mario_y = mario_y
        self._prev_mario_x = mario_x
        info['_mario_y']      = mario_y
        info['_stage']        = stage
        info['_level']        = level
        info['_step_reward']  = reward
        info['_won']          = won
        info['_broken_zones'] = self._broken_zones

        # Full-res color frame for ghost viewer background — throttled to reduce pipe load
        if self._provide_frame and self._step_count % self._frame_send_every == 0:
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

    def _detect_mario_oam(self, ram):
        """Detect Mario's NES position from OAM shadow (slots 0-3). Falls back to previous."""
        xs, ys = [], []
        for s in range(4):
            sy = int(ram[512 + s * 4])
            sx = int(ram[512 + s * 4 + 3])
            if sy < 0xEF and sx > 0:
                ys.append(sy); xs.append(sx)
        if xs:
            mario_y = int(min(ys) * 224 / 240)
            mario_x = int(min(xs))
            return mario_x, mario_y
        return self._prev_mario_x, self._prev_mario_y

    def _barrel_proximity_penalty_oam(self, mario_y, ram):
        """Penalty when a barrel (OAM slots 12-51, groups of 4) is within 15 NES px."""
        barrel_ys = []
        for g in range(10):
            sy = int(ram[512 + (12 + g * 4) * 4])
            if sy < 0xEF:
                barrel_ys.append(int(sy * 224 / 240))
        if not barrel_ys:
            return 0.0, False
        dists = np.abs(np.array(barrel_ys, dtype=float) - mario_y)
        closest_idx = int(dists.argmin())
        closest = float(dists[closest_idx])
        if closest < 15:
            proximity_factor = 1.0 - (closest / 15.0)
            at_level = float(barrel_ys[closest_idx]) >= mario_y - 8
            return -0.2 * proximity_factor, at_level
        return 0.0, False

    def _fire_proximity_penalty_oam(self, mario_y, ram):
        """Penalty when fire (OAM slots 4-7 or 8-11) is within 20 NES px of Mario."""
        fire_ys = []
        for base in (4, 8):
            sy = int(ram[512 + base * 4])
            if sy < 0xEF:
                fire_ys.append(int(sy * 224 / 240))
        if not fire_ys:
            return 0.0
        closest = float(min(abs(fy - mario_y) for fy in fire_ys))
        if closest < 20:
            return -0.2 * (1.0 - closest / 20.0)
        return 0.0

    def _ladder_climbing_bonus(self, action_idx, dy):
        """
        +0.5 for actively climbing (UP + moving up + teal visible).
        Small enough that cycling up-down gives only ~+0.10/step over idling, which the
        height reward (+3 to +10 per new pixel) completely dominates — no farming incentive.
        +0.3 passive when teal is visible — keeps the AI near ladders.
        """
        teal_count = int(np.count_nonzero(self._last_teal))
        if action_idx == 3 and dy > 0 and teal_count > 10:
            return 0.5
        if teal_count > 10:
            return 0.3
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

    def _extract_barrel_oam(self, ram):
        """Barrel channel: paint sprite footprint for each active barrel (OAM slots 12-51)."""
        canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for g in range(10):
            xs, ys = [], []
            for s in range(4):
                slot = 12 + g * 4 + s
                sy = int(ram[512 + slot * 4])
                sx = int(ram[512 + slot * 4 + 3])
                if sy < 0xEF and sx > 0:
                    ys.append(sy); xs.append(sx)
            if not xs:
                continue
            px = int(min(xs) * FRAME_W / 256)
            py = int(min(ys) * FRAME_H / 240)
            r = 4
            canvas[max(0, py - r):min(FRAME_H, py + r),
                   max(0, px - r):min(FRAME_W, px + r)] = 1.0
        return canvas

    def _extract_fire_oam(self, ram):
        """Fire channel: paint sprite footprint for each active fire group (OAM slots 4-11)."""
        canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for base in (4, 8):
            xs, ys = [], []
            for s in range(4):
                slot = base + s
                sy = int(ram[512 + slot * 4])
                sx = int(ram[512 + slot * 4 + 3])
                if sy < 0xEF and sx > 0:
                    ys.append(sy); xs.append(sx)
            if not xs:
                continue
            px = int(min(xs) * FRAME_W / 256)
            py = int(min(ys) * FRAME_H / 240)
            r = 4
            canvas[max(0, py - r):min(FRAME_H, py + r),
                   max(0, px - r):min(FRAME_W, px + r)] = 1.0
        return canvas

    def _extract_hammer_oam(self, ram):
        """Hammer channel: paint sprite footprint for each active hammer (OAM slots 52-55)."""
        canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for base in (52, 54):
            xs, ys = [], []
            for s in range(2):
                slot = base + s
                sy = int(ram[512 + slot * 4])
                sx = int(ram[512 + slot * 4 + 3])
                if sy < 0xEF and sx > 0:
                    ys.append(sy); xs.append(sx)
            if not xs:
                continue
            px = int(min(xs) * FRAME_W / 256)
            py = int(min(ys) * FRAME_H / 240)
            r = 4
            canvas[max(0, py - r):min(FRAME_H, py + r),
                   max(0, px - r):min(FRAME_W, px + r)] = 1.0
        return canvas

    def _get_state(self):
        gray_stack = np.array(self._frames, dtype=np.float32)   # (4, H, W)
        teal   = self._last_teal  [np.newaxis]                   # (1, H, W) ladders
        barrel = self._last_barrel[np.newaxis]                   # (1, H, W) barrels
        fire   = self._last_fire  [np.newaxis]                   # (1, H, W) fires
        return np.concatenate([gray_stack, teal, barrel, fire], axis=0)  # (7, H, W)
        # To enable hammer channel (8 channels), add hammer[np.newaxis] and set OBS_CHANNELS=8

    def _oil_can_penalty(self, mario_x, mario_y):
        """Penalty for approaching the oil can (bottom-left, fixed NES position). Kills on contact.
        Uses horizontal distance only — oil can is a ground hazard so jumping up doesn't escape it."""
        OIL_X = 28
        OIL_RADIUS = 60   # widened from 40 — zone now starts at x≈88, well before the oil can
        OIL_MAX_Y = 210   # only penalise when on or near the ground floor
        if mario_y < OIL_MAX_Y:
            x_dist = abs(mario_x - OIL_X)
            if x_dist < OIL_RADIUS:
                return -3.0 * (1.0 - x_dist / OIL_RADIUS)
        return 0.0

    def _compute_reward(self, lives, mario_y, gameover, old_ep_best_y):
        reward = 0.0

        # Progress-only height reward: only pay for NEW height reached this step.
        # Revisiting old ground gives nothing, so farming the same ladder segment
        # is unprofitable — every idle step costs the time penalty with no offset.
        new_height_px = max(0.0, old_ep_best_y - mario_y)
        height_progress = max(0.0, min(1.0, (MARIO_Y_START - mario_y) / TOTAL_HEIGHT))
        climb_scale = 3.0 + height_progress * 7.0
        reward += new_height_px * climb_scale

        # Win — bonus scales with how fast Mario finished
        won = mario_y <= MARIO_Y_WIN
        if won:
            reward += 500.0 + (self._max_steps - self._step_count) * 0.5

        # Time penalty — discourages idling and hesitation
        reward -= 0.05

        # Death: small penalty — risk-taking to climb should be acceptable
        if lives < self._prev_lives:
            reward -= 3.0

        if gameover:
            reward -= 5.0

        return reward, won
