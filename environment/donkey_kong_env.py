"""
Person 1 — Environment wrapper.
Wraps gym-retro's Donkey Kong NES, preprocesses frames, stacks 4 frames,
maps a small discrete action set, and computes a shaped reward.
"""

import collections
import json
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
MAX_STEPS    = 2000 # decision steps per episode (randomised ±200); increased for multi-stage runs

GAME_NAME  = 'DonkeyKongOriginalEdition-Nes'
GAME_STATE = '1Player.GameA'

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
GAMESTATE_DIR    = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'gamestates')
STAGE_STARTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'stage_starts')

# Stage 4 rivet positions — detected by colour from the opening frame and cached.
RIVET_POSITIONS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'saved_models', 'stage_starts', 'stage_4_rivets.json')

# Stage 4 rivet colour (small orange studs on girders, much smaller area than barrels).
STAGE4_RIVET_HSV_LO   = np.array([10, 120, 120])
STAGE4_RIVET_HSV_HI   = np.array([30, 255, 255])
_STAGE4_RIVET_CHECK_R = 5   # pixel radius of the window used to detect rivet disappearance

# Per-stage cache versions.  Bump a stage's value to force only that stage to rescan.
# Stage 3 is at 5: forces rescan after the MORPH_CLOSE kernel was widened to bridge
#   the fireball gap.  Stage 4 is at 5: forces rescan after the secondary colour pass
#   exclusion and aspect-ratio fixes.
_LADDER_CACHE_VERSIONS = {1: 2, 2: 3, 3: 5, 4: 5}

# Per-stage ladder position cache.  Scanned ONCE on first encounter, then loaded from
# disk every subsequent run.  Delete the file to force a fresh scan.
def _ladder_cache_path(stage: int) -> str:
    return os.path.join(
        os.path.dirname(__file__), '..', 'saved_models', 'stage_starts',
        f'stage_{stage}_ladders.json')


def _gamestate_path(i: int) -> str:
    return os.path.join(GAMESTATE_DIR, f'platform_{i}.bin')

def _best_gamestate_path() -> str:
    return os.path.join(GAMESTATE_DIR, 'platform_best.bin')

def _best_suspended_path() -> str:
    return os.path.join(GAMESTATE_DIR, 'best_suspended.flag')

def _stage_start_path(stage: int) -> str:
    return os.path.join(STAGE_STARTS_DIR, f'stage_{stage}_start.bin')


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
                int(min(ax, bx)) + 1,
                int(ay),
                int(max(ax+aw, bx+bw)) + 1,
                int(by + bh),
            ))
    return zones

def auto_detect_all_ladders(custom_integration_path=None, game_name=None, game_state=None):
    """
    Capture one clean frame and return all ladder segment positions as NES coords.
    Returns list of (nes_x0, nes_y0, nes_x1, nes_y1).
    """
    game_name  = game_name  or GAME_NAME
    game_state = game_state or GAME_STATE
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make(game_name, state=game_state, inttype=inttype, render_mode=None)
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


def precompute_stage_maps(custom_integration_path=None, game_name=None, game_state=None):
    """
    Spin up one throwaway env, capture Stage 1's opening frame, and return:
      teal_mask    — (84, 84) float32 binary mask of ladder pixels (ready to drop into channel 4)
      broken_zones — list of (x0, y0, x1, y1) broken-ladder zones in NES coords

    Call this ONCE in the main process before SubprocVecEnv is created.
    Pass both values to each DonkeyKongEnv so reset() never has to re-scan —
    the ladder layout and broken zones are identical every episode.
    """
    game_name  = game_name  or GAME_NAME
    game_state = game_state or GAME_STATE
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make(game_name, state=game_state, inttype=inttype, render_mode=None)
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


def auto_detect_broken_zones(custom_integration_path=None, game_name=None, game_state=None):
    """
    Spin up a throwaway retro env, grab one frame, run detect_broken_ladder_zones,
    and convert the result to NES coords. Called once in the main process before
    SubprocVecEnv is created so subprocesses never need to re-run it.
    Falls back to BROKEN_LADDER_ZONES if nothing is detected.
    """
    game_name  = game_name  or GAME_NAME
    game_state = game_state or GAME_STATE
    if custom_integration_path:
        retro.data.Integrations.add_custom_path(custom_integration_path)
        inttype = retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype = retro.data.Integrations.DEFAULT
    env = retro.make(game_name, state=game_state, inttype=inttype)
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
                 game_name=None, game_state=None,
                 start_stage: int = 1):
        _game_name  = game_name  or GAME_NAME
        _game_state = game_state or GAME_STATE
        # 'none' sentinel = boot ROM from scratch without loading a save state.
        # Used when no valid state file exists for the ROM (e.g. OE before recording one).
        if _game_state == 'none':
            _game_state = retro.State.NONE
        if custom_integration_path:
            retro.data.Integrations.add_custom_path(custom_integration_path)
            inttype = retro.data.Integrations.CUSTOM_ONLY
        else:
            inttype = retro.data.Integrations.DEFAULT

        self.env = retro.make(
            game=_game_name,
            state=_game_state,
            inttype=inttype,
            render_mode='human' if render else None,
        )

        self.action_space_n = NUM_ACTIONS
        self.observation_shape = (OBS_CHANNELS, FRAME_H, FRAME_W)

        self._start_stage_override = start_stage
        # Checkpoints are stage-1 platform saves — irrelevant when starting later stages.
        self._use_checkpoints   = use_checkpoints and (start_stage == 1)
        self._force_highest     = force_highest and (start_stage == 1)
        # Static maps computed once at startup — if provided, reset() skips HSV scanning entirely.
        self._static_teal_mask  = static_teal_mask
        self._static_broken_zones = broken_zones  # None means detect dynamically at reset
        self._provide_frame   = provide_frame
        self._provide_detect = provide_detect
        self._frames        = collections.deque(maxlen=FRAME_STACK)
        self._prev_lives    = 3
        self._prev_mario_y  = MARIO_Y_START
        self._prev_mario_x  = 128
        self._step_count    = 0
        self._last_teal       = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._ladder_rects    = []   # (x0,y0,x1,y1) in screen pixels, updated once per stage
        self._ladder_src_w    = 240  # obs frame width when ladder rects were scanned
        self._ladder_src_h    = 224  # obs frame height when ladder rects were scanned
        self._ladder_template = None  # grayscale tile patch extracted from stage 1 ladder
        self._rescan_pending  = False # True after stage change; fires when Mario spawns at bottom
        self._rivet_positions   = []   # (x,y) frame pixel coords of rivets, detected at stage 4 start
        self._rivet_popped      = set() # VISUAL indices into _rivet_positions that have been collected
        self._rivet_game_popped = set() # GAME indices (0-7, = 0xC1+i) already assigned to visual slots
        self._ep_stages_cleared = set() # stages cleared this episode (prevents double bonus)
        self._last_barrel   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_fire     = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._last_hammer   = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self._max_steps        = MAX_STEPS
        self._reset_mario_y    = MARIO_Y_START
        self._best_y           = MARIO_Y_START
        self._ep_best_y          = MARIO_Y_START
        self._ep_cumulative_best = 0
        self._prev_ladder_dx  = None   # for ladder attraction shaping
        self._prev_rivet_dist = None   # for rivet attraction shaping
        self._prev_fire_penalty   = 0.0  # fire proximity from last step (death cause)
        self._prev_barrel_penalty = 0.0  # barrel proximity from last step (death cause)
        self._stage_before_clear = 1   # stage value captured before RAM update each step
        self._stage_clear_cooldown = 0 # steps remaining before stage_clear can fire again
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

        # Override: load a recorded stage-start state so training begins mid-game.
        if self._start_stage_override > 1:
            _ss_path = _stage_start_path(self._start_stage_override)
            if os.path.exists(_ss_path):
                with open(_ss_path, 'rb') as _sf:
                    self.env.em.set_state(_sf.read())
                for _ in range(4):
                    result = self.env.step(_noop)
                obs, _, _, _, noop_info = result
            else:
                print(f'[Warning] Stage {self._start_stage_override} start state not found: {_ss_path}')
                print(f'          Play to stage {self._start_stage_override} in debug_env.py first to record it.')

        self._start_stage   = noop_info.get('stage', 0)
        self._current_stage = self._start_stage
        # Fallback: env.reset() returns (obs, info) — 2 values, not 5 — so noop_info
        # is {} when no checkpoint was loaded, leaving stage = 0. Read from RAM instead.
        if self._current_stage == 0:
            _ram = np.frombuffer(self.env.get_ram(), dtype=np.uint8)
            self._current_stage = int(_ram[83]) or 1
            self._start_stage   = self._current_stage
        # Ensure stage is correct when using an override (RAM may briefly lag).
        if self._start_stage_override > 1:
            self._current_stage = self._start_stage_override
            self._start_stage   = self._start_stage_override
        self._prev_lives    = noop_info.get('lives', 3)
        self._prev_mario_y = MARIO_Y_START
        self._step_count   = 0
        self._max_steps    = MAX_STEPS + random.randint(-200, 200)
        # Pre-populate cleared stages and cumulative height for direct stage starts
        # so the mountain reward correctly reflects total game progress.
        self._ep_stages_cleared    = set(range(1, self._start_stage_override))
        self._rescan_pending       = False
        self._ep_cumulative_best   = (self._start_stage_override - 1) * TOTAL_HEIGHT
        self._stage_before_clear   = self._start_stage_override
        self._stage_clear_cooldown = 0
        self._prev_ladder_dx  = None
        self._prev_rivet_dist = None
        self._prev_fire_penalty   = 0.0
        self._prev_barrel_penalty = 0.0

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        ram  = np.frombuffer(self.env.get_ram(), dtype=np.uint8)

        self._prev_mario_x, self._prev_mario_y = self._detect_mario_oam(ram)
        self._reset_mario_y = self._prev_mario_y
        self._ep_best_y     = self._prev_mario_y

        if self._static_teal_mask is not None and self._start_stage_override == 1:
            # Static maps precomputed at startup — no HSV scan needed (stage 1 only).
            self._last_teal    = self._static_teal_mask
            self._broken_zones = self._static_broken_zones
            # Ladder rects are needed for the zone-based on_ladder reward check.
            # _rescan_stage_maps loads from the JSON cache (fast after first run) but
            # also overwrites _last_teal — restore the static mask afterward so the
            # observation channel stays identical to what the model was trained on.
            if not self._ladder_rects:
                self._rescan_stage_maps(obs)
                self._last_teal    = self._static_teal_mask
                self._broken_zones = self._static_broken_zones
        else:
            # Dynamic scan: used for stages 2-4 or when no static mask is provided.
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
            # Build ladder display rects for stage 1
            self._rescan_stage_maps(obs)

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

        # Capture stage BEFORE updating from RAM — used for stage_clear attribution.
        self._stage_before_clear = self._current_stage

        # On stage change, arm the pending flag — don't rescan yet because the transition
        # animation plays before the new layout is on-screen.  Fire only when Mario
        # appears at the bottom of the stage (OAM Y ≥ 150), which means he has spawned
        # in and the new stage background is fully drawn.
        if stage != self._current_stage and stage != 0:
            self._current_stage = stage
            self._rescan_pending = True

        if self._stage_clear_cooldown > 0:
            self._stage_clear_cooldown -= 1

        if self._rescan_pending:
            _oam_y = int(ram[512])   # OAM slot 0 Y; 0xFF/0xEF+ = off-screen
            if 150 <= _oam_y < 0xEF:
                self._rescan_pending = False
                self._rescan_stage_maps(raw_obs)

        mario_x, mario_y = self._detect_mario_oam(ram)

        # Stage 4: track rivet collection via the per-rivet state bytes 0xC1–0xC8.
        # Each byte is 0=intact, 1=collected.  When byte i flips 0→1, find the
        # nearest uncollected visual position and assign it.  When bytes flip back
        # to 0 (stage restart / game-over), clear everything.
        # Rivets persist through deaths in DK NES — no death reset needed.
        _new_rivet_pops = 0
        if self._current_stage == 4 and self._rivet_positions:
            _any_reset = False
            for _gi in range(8):
                _byte = int(ram[0xC1 + _gi])
                if _byte == 1 and _gi not in self._rivet_game_popped:
                    self._rivet_game_popped.add(_gi)
                    if _gi < len(self._rivet_positions):
                        self._rivet_popped.add(_gi)
                        _new_rivet_pops += 1
                elif _byte == 0 and _gi in self._rivet_game_popped:
                    _any_reset = True
            if _any_reset:
                # Bytes reverted to 0 → stage restarted from scratch
                self._rivet_popped      = set()
                self._rivet_game_popped = set()
                _new_rivet_pops         = 0
                self._prev_rivet_dist   = None

        barrel_penalty, barrel_at_level = self._barrel_proximity_penalty_oam(mario_y, ram)
        dy_this_step                    = self._prev_mario_y - mario_y
        old_ep_best_y                   = self._ep_best_y
        mario_cx = mario_x + 8   # OAM X is left edge of 16px sprite; use centre for zone test
        mario_cy = mario_y + 8
        in_broken_zone = any(
            z[0] <= mario_cx <= z[2] and z[1] <= mario_cy <= z[3]
            for z in self._broken_zones
        )

        # Zone-based ladder check — OAM x converted to frame pixel space for rect comparison.
        # No HSV/colour used: position comes from RAM, rects from cached stage scan.
        _lcx = mario_cx * self._ladder_src_w / 256   # OAM x (0-255) → frame x
        _lcy = mario_cy                               # mario_y already in 0-224 = frame y
        on_ladder = any(
            r[0] <= _lcx <= r[2] and r[1] <= _lcy <= r[3]
            for r in self._ladder_rects
        )
        fire_penalty = self._fire_proximity_penalty_oam(mario_y, mario_cx, ram, on_ladder)

        self._ep_best_y = min(self._ep_best_y, mario_y)
        if in_broken_zone and action_idx == 3:
            # Pressing UP through a broken ladder: suppress the active climbing bonus
            # but still credit height gained — climbing through is still possible but
            # always less rewarding than an intact ladder (no active bonus).
            ladder_bonus          = 0.3   # passive only
            broken_ladder_penalty = -2.0
        else:
            ladder_bonus          = self._ladder_climbing_bonus(action_idx, dy_this_step, on_ladder)
            broken_ladder_penalty = 0.0
        height_ref_y = old_ep_best_y

        # New height record — save checkpoint only when standing on a platform (not a ladder).
        # _best_y is only advanced when we actually save, so if Mario is on a ladder at the
        # threshold height, we keep retrying until he's standing on the platform proper.
        if self._use_checkpoints and mario_y < self._best_y and mario_y > MARIO_Y_WIN and lives >= self._prev_lives and not gameover:
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

        reward, won, stage_clear = self._compute_reward(lives, mario_y, gameover, height_ref_y)
        reward += barrel_penalty + fire_penalty + ladder_bonus + broken_ladder_penalty
        reward += _new_rivet_pops * 40.0
        reward += self._ladder_attraction_reward(mario_cx, mario_y, on_ladder)
        reward += self._rivet_attraction_reward(mario_cx, mario_y)

        # Death-cause attribution: use proximity from the PREVIOUS step (this step's
        # OAM may already reflect post-death positions). Fire deaths get an extra hit
        # because fireballs actively chase and the death is always avoidable.
        if lives < self._prev_lives:
            if self._prev_fire_penalty < -0.3:
                reward -= 4.0   # died to fire/fireball
            elif self._prev_barrel_penalty < -0.1:
                reward -= 1.5   # died to barrel

        self._prev_fire_penalty   = fire_penalty
        self._prev_barrel_penalty = barrel_penalty

        # Stage clear: mark this stage done, reset height tracker so the next stage
        # gets its own full height reward, then let the episode continue naturally
        # through the game's built-in stage transition animation.
        if stage_clear:
            self._ep_stages_cleared.add(self._stage_before_clear)
            self._ep_best_y = MARIO_Y_START
            self._stage_clear_cooldown = 30  # suppress false clears during transition animation
            self._prev_ladder_dx  = None     # new stage has different ladder layout

        # Level cycle: game has looped back to stage 1 on a harder difficulty round.
        # Treat this as a full-game win and end the episode.
        game_cycled = (level > 0 and self._current_stage == 1 and
                       len(self._ep_stages_cleared) > 0)
        if game_cycled:
            won    = True
            reward += 2000.0   # massive bonus for completing the full loop

        done = bool(gameover) or won or self._step_count >= self._max_steps

        # Reset height tracker when a life is lost so each new life gets the same
        # reward signal as the first — otherwise later lives have no height headroom
        # and the model learns to play differently depending on which life it's on.
        # _ep_cumulative_best (not _ep_best_y) drives the actual height reward, so
        # both must be reset. Keep the stage-cleared base so stage-2+ deaths don't
        # reset back to 0 — only the within-stage climb resets.
        if lives < self._prev_lives:
            self._ep_best_y = MARIO_Y_START
            _stages_done = len([s for s in self._ep_stages_cleared if s < self._current_stage])
            self._ep_cumulative_best = _stages_done * TOTAL_HEIGHT
            self._prev_ladder_dx  = None
            self._prev_rivet_dist = None
            # Rivets reset on death in stage 4.  No explicit reset needed here because
            # step() already detects it: when 0xC1-C8 bytes revert from 1→0 after a
            # death, _any_reset fires and clears both popped sets.

        self._prev_lives   = lives
        self._prev_mario_y = mario_y
        self._prev_mario_x = mario_x
        info['_mario_y']      = mario_y
        info['_mario_x']      = mario_x
        info['_stage']            = stage
        info['_stage_just_cleared'] = self._stage_before_clear if stage_clear else 0
        info['_level']        = level
        info['_step_reward']     = reward
        info['_won']             = won
        info['_stage_clear']     = stage_clear
        info['_stages_cleared']     = len(self._ep_stages_cleared)
        info['_cumulative_height']  = self._ep_cumulative_best
        info['_broken_zones'] = self._broken_zones
        info['_ladder_rects'] = self._ladder_rects   # frame pixel coords

        # OAM sprite positions for ghost viewer — barrels (slots 12-51) and fires (4-11)
        _barrel_pts = []
        for _g in range(10):
            _sy = int(ram[512 + (12 + _g * 4) * 4])
            _sx = int(ram[512 + (12 + _g * 4) * 4 + 3])
            if _sy < 0xEF and _sx > 0:
                _barrel_pts.append((_sx, _sy))   # raw OAM coords: x=0-255, y=0-240
        _fire_pts = []
        for _base in (4, 8):
            _sy = int(ram[512 + _base * 4])
            _sx = int(ram[512 + _base * 4 + 3])
            if _sy < 0xEF and _sx > 0:
                _fire_pts.append((_sx, _sy))     # raw OAM coords: x=0-255, y=0-240
        info['_barrel_pts']      = _barrel_pts   # (oam_x 0-255, nes_y 0-224)
        info['_fire_pts']        = _fire_pts
        info['_rivet_positions'] = list(self._rivet_positions)  # frame pixel coords
        info['_rivet_popped']    = list(self._rivet_popped)     # indices of collected rivets

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

    def _fire_proximity_penalty_oam(self, mario_y, mario_cx_oam, ram, on_ladder):
        """Penalty when fire (OAM slots 4-11) is within 35 NES px of Mario.
        Fireballs are far more dangerous than barrels — penalty is much stronger.
        Extra penalty when fire is directly above Mario on a ladder (can't dodge)."""
        fire_pts = []
        for base in (4, 8):
            ys, xs = [], []
            for s in range(4):
                slot = base + s
                sy = int(ram[512 + slot * 4])
                sx = int(ram[512 + slot * 4 + 3])
                if sy < 0xEF and sx > 0:
                    ys.append(int(sy * 224 / 240)); xs.append(sx)
            if ys:
                fire_pts.append((min(ys), min(xs)))
        if not fire_pts:
            return 0.0

        RADIUS = 35
        penalty = 0.0
        for fy, fx in fire_pts:
            dy = abs(fy - mario_y)
            dx = abs(fx - mario_cx_oam)
            if dy < RADIUS:
                prox = 1.0 - dy / RADIUS
                penalty -= 1.0 * prox   # base penalty (5× old value)
                # Extra penalty when fire is above and Mario is on a ladder — sitting
                # under fire on a ladder is the single most avoidable death in DK.
                if on_ladder and fy < mario_y and dy < 25 and dx < 20:
                    penalty -= 1.5 * prox
        return max(penalty, -3.0)

    def _ladder_climbing_bonus(self, action_idx, dy, on_ladder: bool):
        """
        +0.5 for actively climbing (UP + moving up + inside a ladder zone).
        +0.3 passive when inside a ladder zone — keeps the AI near ladders.
        Zone check is OAM-position vs cached ladder rects — no HSV/colour used.
        """
        if action_idx == 3 and dy > 0 and on_ladder:
            return 0.5
        if on_ladder:
            return 0.3
        return 0.0

    def _ladder_attraction_reward(self, mario_cx_oam, mario_y, on_ladder):
        """Potential-based shaping: reward for moving horizontally toward the nearest
        upward ladder reachable from Mario's current platform level.
        Zero when already on a ladder. Not used on stage 4 (no height objective there)."""
        if on_ladder or not self._ladder_rects or self._current_stage == 4:
            return 0.0
        if not self._ladder_src_w:
            return 0.0

        mario_fx = mario_cx_oam * self._ladder_src_w / 256

        # Ladders whose bottom (y1/r[3]) is within REACH px of Mario's Y and whose
        # top (y0/r[1]) is above Mario — these are entries to the next platform up.
        REACH = 30
        candidates = [
            r for r in self._ladder_rects
            if r[3] >= mario_y - REACH and r[1] < mario_y - 5
        ]
        if not candidates:
            return 0.0

        best_dx = min(abs((r[0] + r[2]) / 2.0 - mario_fx) for r in candidates)
        prev = self._prev_ladder_dx
        self._prev_ladder_dx = best_dx
        if prev is None:
            return 0.0
        return (prev - best_dx) * 0.2

    def _rivet_attraction_reward(self, mario_cx_oam, mario_y):
        """Potential-based shaping: reward for moving toward the nearest uncollected
        rivet on stage 4. Target resets automatically when a rivet is collected."""
        if self._current_stage != 4 or not self._rivet_positions:
            return 0.0

        uncollected = [(x, y) for i, (x, y) in enumerate(self._rivet_positions)
                       if i not in self._rivet_popped]
        if not uncollected:
            return 0.0

        mario_fx  = mario_cx_oam * self._ladder_src_w / 256
        best_dist = min(((x - mario_fx) ** 2 + (y - mario_y) ** 2) ** 0.5
                        for x, y in uncollected)

        prev = self._prev_rivet_dist
        self._prev_rivet_dist = best_dist
        if prev is None:
            return 0.0
        return (prev - best_dist) * 0.3

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

    def _rescan_stage_maps(self, obs):
        """Load cached ladder positions (or scan once and cache) for the current stage.

        Cache files live in saved_models/stage_starts/stage_N_ladders.json.
        Delete a file to force a fresh scan for that stage.
        The 4→1 level-rollover bug is fixed automatically: stage 1's cache is written
        at game start, so the rollover just loads from file instead of rescanning.
        """
        stage      = self._current_stage
        cache_path = _ladder_cache_path(stage)
        h, w       = obs.shape[:2]   # needed by _teal_from_rects regardless of cache path

        _cache_ok = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as _f:
                    _data = json.load(_f)
                _expected_ver = _LADDER_CACHE_VERSIONS.get(stage, max(_LADDER_CACHE_VERSIONS.values()))
                if _data.get('version', 1) < _expected_ver:
                    print(f'[Stage {stage}] Cache outdated (v{_data.get("version",1)} < {_expected_ver}) — rescanning')
                    os.remove(cache_path)
                else:
                    rects = [tuple(r) for r in _data.get('rects', [])]
                    if stage == 1 and 'broken_zones' in _data:
                        self._broken_zones = [tuple(z) for z in _data['broken_zones']]
                    else:
                        self._broken_zones = [] if stage != 1 else list(BROKEN_LADDER_ZONES)
                    _cache_ok = True
            except (json.JSONDecodeError, KeyError, ValueError) as _e:
                print(f'[Stage {stage}] Cache file corrupt ({_e}) — deleting and rescanning')
                os.remove(cache_path)

        if not _cache_ok:
            # ── Cache miss: detect from frame and save ────────────────────────
            hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

            # Broken zones (stage 1 only)
            if stage == 1:
                _raw_bz = detect_broken_ladder_zones(hsv, h, w)
                self._broken_zones = (
                    [(int(z[0]*256/w), int(z[1]*224/h),
                      int(z[2]*256/w), int(z[3]*224/h)) for z in _raw_bz]
                    if _raw_bz else list(BROKEN_LADDER_ZONES)
                )
            else:
                self._broken_zones = []

            # Stage 4: detect rivets BEFORE ladders so we can clip ladder rects that
            # overlap rivet positions.  Short "top stub" ladders sit at the same x
            # columns as the adjacent rivets — clipping is the only reliable fix.
            _s4_rivets_for_clip = []
            if stage == 4:
                if os.path.exists(RIVET_POSITIONS_PATH):
                    with open(RIVET_POSITIONS_PATH) as _rf:
                        _s4_rivets_for_clip = [tuple(p) for p in json.load(_rf)]
                else:
                    _s4_rivets_for_clip = self._scan_rivets_from_frame(obs)
                    if _s4_rivets_for_clip:
                        os.makedirs(os.path.dirname(RIVET_POSITIONS_PATH), exist_ok=True)
                        with open(RIVET_POSITIONS_PATH, 'w') as _rf:
                            json.dump([[int(x), int(y)] for x, y in _s4_rivets_for_clip], _rf)

            rects = self._detect_ladders_color(gray, hsv, h, w, stage)

            if stage == 4 and _s4_rivets_for_clip:
                rects = DonkeyKongEnv._clip_stage4_ladders(rects, _s4_rivets_for_clip)

            if rects:
                # Convert numpy int32 → Python int so JSON can serialise them
                _save = {'version': _LADDER_CACHE_VERSIONS.get(stage, max(_LADDER_CACHE_VERSIONS.values())),
                         'rects': [[int(v) for v in r] for r in rects]}
                if stage == 1:
                    _save['broken_zones'] = [[int(v) for v in z]
                                             for z in self._broken_zones]
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as _f:
                    json.dump(_save, _f)
                print(f'[Stage {stage}] Detected {len(rects)} ladder rects → saved to cache')
            else:
                print(f'[Stage {stage}] Detection returned 0 rects — running diagnostic')
                self._print_color_diagnostic(hsv, h, w, stage)

        # ── Stage 4 rivets ────────────────────────────────────────────────────
        if stage == 4:
            self._rivet_popped      = set()
            self._rivet_game_popped = set()
            # Pre-populate from save state: if any 0xC1-C8 bytes are already 1,
            # mark those game indices as popped and assign visual index = game index
            # (best possible without a live Mario position to guide nearest-match).
            _s4_ram = np.frombuffer(self.env.get_ram(), dtype=np.uint8)
            for _gi in range(8):
                if int(_s4_ram[0xC1 + _gi]) == 1:
                    self._rivet_game_popped.add(_gi)
                    if _gi < len(self._rivet_positions):
                        self._rivet_popped.add(_gi)
            if os.path.exists(RIVET_POSITIONS_PATH):
                with open(RIVET_POSITIONS_PATH) as _f:
                    self._rivet_positions = [tuple(p) for p in json.load(_f)]
                print(f'[Stage 4] {len(self._rivet_positions)} rivet positions loaded from cache')
            else:
                self._rivet_positions = self._scan_rivets_from_frame(obs)
                if self._rivet_positions:
                    os.makedirs(os.path.dirname(RIVET_POSITIONS_PATH), exist_ok=True)
                    with open(RIVET_POSITIONS_PATH, 'w') as _f:
                        json.dump([[int(x), int(y)] for x, y in self._rivet_positions], _f)
                    print(f'[Stage 4] Saved {len(self._rivet_positions)} rivet positions')
                else:
                    print('[Stage 4] No rivets detected from frame — delete cache to retry')
        else:
            self._rivet_positions = []
            self._rivet_popped    = set()

        self._ladder_rects  = rects
        self._ladder_src_w  = w
        self._ladder_src_h  = h
        self._last_teal     = self._teal_from_rects(rects, h, w)

    @staticmethod
    def _detect_ladders_color(gray, hsv, h, w, stage):
        """
        Colour-based ladder detection.  Called only on cache miss (first encounter).

        Stage 1   — teal HSV [75,60,80]→[110,255,255] (original, working).
        Stage 2   — white: HSV S<65, V>130 + horizontal dilation to merge rails,
                    vertical close to bridge rung gaps; falls back to rung-counting.
        Stage 3   — teal with wider S floor (30 vs 60) + morphological close to
                    capture the full bottom section of each ladder.
        Stage 4   — rung-counting (Sobel-Y, colour-independent) as primary;
                    yellow/amber HSV [12,80,80]→[45,255,255] as fallback.

        Returns a list of (x0,y0,x1,y1) rects in the frame's pixel space.
        """
        if stage == 1:
            mask = cv2.inRange(hsv, np.array([75, 60, 80]), np.array([110, 255, 255]))
            mask[:int(h * 0.10), :]              = 0
            mask[int(h * 0.85):, :int(w * 0.20)] = 0
            return DonkeyKongEnv._rects_from_mask(mask)

        elif stage == 2:
            # White ladders: low saturation, high value.
            # No bottom exclusion — stage 2 has no oil-can; rely on aspect ratio only.
            raw = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([179, 65, 255]))
            raw[:int(h * 0.10), :] = 0
            # Two-pass approach: dilate into a connectivity mask to merge the two side rails
            # and bridge rung gaps, but compute TIGHT bounding boxes from the original raw
            # mask so returned rects hug actual pixels rather than the inflated envelope.
            conn = cv2.dilate(raw, np.ones((1, 8), np.uint8))
            conn = cv2.morphologyEx(conn, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
            n_labels, labels = cv2.connectedComponents(conn, connectivity=8)
            rects = []
            for lbl in range(1, n_labels):
                in_comp = (labels == lbl) & (raw > 0)
                ys, xs = np.where(in_comp)
                if len(ys) < 10:
                    continue
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
                bh, bw = y1 - y0, x1 - x0
                if bh > 12 and bw < 45 and bh > 1.5 * bw:
                    rects.append((x0, y0, x1, y1))
            if not rects:
                rects = DonkeyKongEnv._detect_ladders_by_rungs(gray, h, w)
            return rects

        elif stage == 3:
            # Teal ladders — lower S floor catches dim rungs near the ladder bottom.
            # Close BEFORE applying exclusions: MORPH_CLOSE = dilate+erode, and if the
            # HUD zone is zeroed first, the erode step eats the top pixels of any ladder
            # that touches the boundary. Closing on the full mask then clearing the HUD
            # preserves those top pixels.
            # Kernel height 17 (8px reach each side) bridges a ~15px fireball gap so
            # the full ladder rect is captured even if a fireball obscures part of it
            # at scan time.
            mask = cv2.inRange(hsv, np.array([75, 30, 60]), np.array([110, 255, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 3), np.uint8))
            mask[:int(h * 0.10), :]              = 0
            mask[int(h * 0.85):, :int(w * 0.20)] = 0
            return DonkeyKongEnv._rects_from_mask(mask)

        elif stage == 4:
            # Primary: rung-counting (colour-independent, separates ladders from girders).
            rects = DonkeyKongEnv._detect_ladders_by_rungs(gray, h, w)
            # Trim 2 pixels from each horizontal edge so ladder boxes don't overlap
            # adjacent rivet positions.
            rects = [(x0 + 4, y0 + 2, x1 - 4, y1 - 2) for x0, y0, x1, y1 in rects
                     if x1 - x0 > 10 and y1 - y0 > 6]
            # Secondary: colour-based pass to pick up short sections (< 20 px tall)
            # that the rung detector's height filter discards.
            # No bottom-left exclusion — stage 4 has no oil can there, and the
            # bottom-left stub ladder sits in that area.
            # Aspect ratio ≥ 2.0 (height must be at least 2× width) to prevent
            # orange rivet blobs (roughly square) from being mistaken for ladders.
            col_mask = cv2.inRange(hsv, np.array([12, 80, 80]), np.array([45, 255, 255]))
            col_mask[:int(h * 0.10), :] = 0  # HUD only
            col_mask = cv2.dilate(col_mask, np.ones((2, 3), np.uint8))
            for cr in DonkeyKongEnv._rects_from_mask(col_mask):
                bh = cr[3] - cr[1];  bw = cr[2] - cr[0]
                if bh < 8 or bw >= 40 or bh < 2.0 * bw:
                    continue
                cx = (cr[0] + cr[2]) / 2;  cy = (cr[1] + cr[3]) / 2
                if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                    continue   # already covered by rung detector
                rects.append(cr)
            return rects

        return []

    @staticmethod
    def _rects_from_mask(mask):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rects = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 15:
                continue
            lx = stats[i, cv2.CC_STAT_LEFT]
            ly = stats[i, cv2.CC_STAT_TOP]
            lw = stats[i, cv2.CC_STAT_WIDTH]
            lh = stats[i, cv2.CC_STAT_HEIGHT]
            rects.append((lx, ly, lx + lw, ly + lh))
        return rects

    @staticmethod
    def _detect_ladders_by_rungs(gray, h, w):
        """
        Find ladder bounding boxes by counting horizontal Sobel edges per column.

        Ladders have many short rung edges stacked vertically; platform girders have only
        1-2 long horizontal edges per column.  Columns with >= MIN_RUNG_ROWS horizontal
        edges that span a narrow x-range are collected as a ladder segment.
        Works for any ladder colour (white, yellow, teal, etc.).
        """
        sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edges   = (np.abs(sobel_y) > 12).astype(np.uint8)
        edges[:int(h * 0.10), :]              = 0   # exclude HUD
        edges[int(h * 0.85):, :int(w * 0.20)] = 0   # exclude oil-can corner

        col_density = edges.sum(axis=0)   # edge-row count per column

        rects    = []
        in_group = False
        x0       = 0
        for x in range(w + 1):
            active = x < w and col_density[x] >= 5
            if active and not in_group:
                in_group = True;  x0 = x
            elif not active and in_group:
                in_group = False;  x1 = x
                if x1 - x0 >= 35:          # too wide — platform girder, not a ladder
                    continue
                col_edges = edges[:, x0:x1]
                full_row  = edges.sum(axis=1).astype(np.float32)
                col_sum   = col_edges.sum(axis=1).astype(np.float32)
                # Keep only rows where ≥30 % of the frame-wide edges fall inside
                # this column group — platform girders span the full frame width so
                # their rows have a very low in-column fraction and get filtered out.
                mask_rows = np.where(
                    (col_sum > 0) & (col_sum / np.where(full_row > 0, full_row, 1) >= 0.30)
                )[0]
                rows = mask_rows
                if len(rows) < 5 or rows[-1] - rows[0] < 20:
                    continue               # too few rung rows or too short
                rects.append((x0, int(rows[0]), x1, int(rows[-1])))
        return rects

    @staticmethod
    def _print_color_diagnostic(hsv, h, w, stage):
        """
        Called when ladder detection returns 0 rects.
        Reports both white/grey pixel counts AND the dominant saturated hues so the
        developer can see the actual ladder colour and adjust the HSV range.
        Note: white pixels have S<30 so they do NOT appear in the hue histogram —
        check the 'white pixels' line separately.
        """
        roi = hsv[int(h * 0.12):int(h * 0.85), int(w * 0.05):int(w * 0.95)]
        bright = roi[:, :, 2] > 60

        # White/near-white pixels: bright AND low saturation
        white_px = int(np.sum(bright & (roi[:, :, 1] < 40)))
        print(f'[Stage {stage} DIAG] No ladders found.')
        print(f'  White/grey pixels (S<40, V>60): {white_px}  '
              f'{"← ladder candidate" if white_px > 200 else "← very few, ladders not white"}')

        # Saturated hues
        sat  = roi[:, :, 1] > 40
        hues = roi[:, :, 0][bright & sat]
        if hues.size > 0:
            counts = np.bincount(hues.astype(np.int32), minlength=180)
            top    = sorted(enumerate(counts), key=lambda x: -x[1])[:6]
            print('  Top saturated hue bins (H in OpenCV 0-179 = 0-358°):')
            names  = {range(0,10):'red/orange', range(10,25):'orange', range(25,35):'yellow',
                      range(35,75):'green',      range(75,100):'teal',  range(100,130):'blue',
                      range(130,160):'purple',    range(160,180):'red'}
            for hval, cnt in top:
                nm = next((v for k, v in names.items() if hval in k), '?')
                print(f'    H={hval:3d} ({nm}): {cnt} px')
        print('  Fix: for white ladders use gray>155; '
              'for coloured ladders use inRange([H-5,80,80],[H+5,255,255])')

    def _detect_ladders_by_template(self, gray, h, w):
        """Find ladder tiles using normalised cross-correlation — colour-independent."""
        tmpl = self._ladder_template
        res  = cv2.matchTemplate(gray.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
        match_mask = (res >= 0.72).astype(np.uint8)
        # Dilate to merge adjacent tile hits into one segment
        match_mask = cv2.dilate(match_mask, np.ones((8, 4), np.uint8))
        match_mask[:int(h * 0.10), :]              = 0
        match_mask[int(h * 0.85):, :int(w * 0.20)] = 0
        return self._rects_from_mask(match_mask)

    @staticmethod
    def _clip_stage4_ladders(rects, rivet_positions, margin=3):
        """Shrink any stage 4 ladder rect whose x-range contains a rivet position.

        Horizontal clip: pushes the nearer x-edge past the rivet so the ladder
        channel doesn't mark rivet columns as climbable.

        Vertical clip: rivets sit at platform level and produce concentrated
        Sobel edges that score 100 % column-fraction, bypassing the row filter.
        If a rivet falls inside the outer 25 % of the ladder rect's height, push
        the nearer y-edge past it so the rect stops at the actual rung area.
        """
        VERT_M = 3
        out = []
        for x0, y0, x1, y1 in rects:
            lx0, lx1 = x0, x1
            ly0, ly1 = y0, y1
            for rx, ry in rivet_positions:
                if not (x0 <= rx <= x1):   # check against original x-span
                    continue
                # Horizontal clip
                if ly0 <= ry <= ly1:
                    cx = (lx0 + lx1) / 2
                    if rx < cx:
                        lx0 = int(rx) + margin
                    else:
                        lx1 = int(rx) - margin
                # Vertical clip — only when rivet is in the outer 25 % of height
                rect_h = ly1 - ly0
                if rect_h > 0:
                    if ry < ly0 + rect_h * 0.25:
                        ly0 = max(ly0, int(ry) + VERT_M)
                    elif ry > ly1 - rect_h * 0.25:
                        ly1 = min(ly1, int(ry) - VERT_M)
            if lx1 - lx0 > 4 and ly1 - ly0 > 15:
                out.append((int(lx0), int(ly0), int(lx1), int(ly1)))
        return out

    @staticmethod
    def _teal_from_rects(rects, src_h, src_w):
        """Build the 84×84 ladder channel from pre-computed bounding boxes."""
        canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        for (x0, y0, x1, y1) in rects:
            # Convert from source screen coords to 84×84 canvas
            cx0 = int(x0 * FRAME_W / src_w);  cx1 = int(x1 * FRAME_W / src_w)
            cy0 = int(y0 * FRAME_H / src_h);  cy1 = int(y1 * FRAME_H / src_h)
            canvas[cy0:cy1, cx0:cx1] = 1.0
        return canvas

    def _scan_rivets_from_frame(self, obs):
        """Detect stage 4 rivet positions from the opening frame.
        Rivets are small orange studs (3-50 px area) on the girders — much smaller
        than rolling barrels (100-500 px), so area filter cleanly separates them.
        Returns list of (x, y) in frame pixel coords, capped at 8.
        """
        h, w = obs.shape[:2]
        hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, STAGE4_RIVET_HSV_LO, STAGE4_RIVET_HSV_HI)
        # Stage 4: DK + Pauline sit at the TOP CENTRE of the stage (unlike stage 1
        # where DK is upper-left).  Exclude the top 22% full-width to clear both the
        # HUD strip and the DK/Pauline sprite area above the first platform.
        mask[:int(h * 0.22), :] = 0
        mask[int(h * 0.85):, :int(w * 0.20)] = 0  # oil-can corner
        mask[:, :int(w * 0.10)] = 0   # left-border ladder-platform junction
        mask[:, int(w * 0.90):] = 0   # right-border ladder-platform junction
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            rh   = stats[i, cv2.CC_STAT_HEIGHT]
            rw   = stats[i, cv2.CC_STAT_WIDTH]
            if 3 <= area <= 50 and rh <= 12 and rw <= 12:
                # Use bounding-box centre — more stable than centroid for irregular blobs
                cx = int(stats[i, cv2.CC_STAT_LEFT] + rw / 2)
                cy = int(stats[i, cv2.CC_STAT_TOP]  + rh / 2)
                candidates.append((area, cx, cy))
        # Sort to match RAM order: bottom row first, left-to-right within each row.
        # Row-bucketing groups rivets within ROW_TOL px vertically so left-to-right
        # ordering within a row isn't disrupted by slight Y variance between rivets
        # on the same girder. RAM byte 0xC1+i maps directly to _rivet_positions[i].
        candidates.sort(key=lambda t: -t[2])   # bottommost (highest Y) first
        ROW_TOL = 15
        rows = []
        for cand in candidates[:8]:
            _, cx, cy = cand
            placed = False
            for row in rows:
                if abs(row[0][2] - cy) <= ROW_TOL:
                    row.append(cand)
                    placed = True
                    break
            if not placed:
                rows.append([cand])
        rows.sort(key=lambda row: -max(t[2] for t in row))   # bottom row first
        for row in rows:
            row.sort(key=lambda t: t[1])                       # left-to-right within each row
        rivets = [(cx, cy) for row in rows for _, cx, cy in row][:8]
        print(f'[Stage 4] Frame scan found {len(candidates)} candidates → kept {len(rivets)}')
        return rivets

    def _check_rivet_pops_from_frame(self, obs):
        """Mark rivet positions where orange pixels have disappeared as popped."""
        h, w = obs.shape[:2]
        hsv  = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        r    = _STAGE4_RIVET_CHECK_R
        for ri, (rx, ry) in enumerate(self._rivet_positions):
            if ri in self._rivet_popped:
                continue
            y0 = max(0, ry - r);  y1 = min(h, ry + r + 1)
            x0 = max(0, rx - r);  x1 = min(w, rx + r + 1)
            orange = cv2.inRange(hsv[y0:y1, x0:x1], STAGE4_RIVET_HSV_LO, STAGE4_RIVET_HSV_HI)
            if int(orange.sum()) < 255 * 4:   # < 4 orange pixels = rivet is gone
                self._rivet_popped.add(ri)

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

        # Mountain height reward: stages 1-3 treated as a continuous climb.
        # Cumulative height = stages_cleared * TOTAL_HEIGHT + current_stage_height,
        # so height_progress and climb_scale naturally increase across stages.
        # Stage 4 uses rivet rewards instead — no height reward there.
        if self._current_stage < 4:
            _stages_done  = len([s for s in self._ep_stages_cleared if s < self._current_stage])
            _cur_h        = max(0.0, MARIO_Y_START - mario_y)
            _cumulative_h = _stages_done * TOTAL_HEIGHT + _cur_h
            new_height_px = max(0.0, _cumulative_h - self._ep_cumulative_best)
            self._ep_cumulative_best = max(self._ep_cumulative_best, _cumulative_h)
            # Stage 1 keeps its original scale (height_progress 0→1 within the stage).
            # Later stages inherit a higher baseline so each pixel is worth more,
            # making progression through the mountain genuinely more valuable.
            height_progress = min(1.0, _cur_h / TOTAL_HEIGHT)
            stage_base      = _stages_done / 3.0
            climb_scale     = 3.0 + (stage_base + height_progress * (1.0 - stage_base)) * 7.0
            reward += new_height_px * climb_scale

        # stage_clear: a non-final stage was beaten — give bonus but keep episode alive.
        # won: the full game is beaten (stage 4 rivets all collected) — end episode.
        if self._current_stage == 4:
            won         = len(self._rivet_popped) >= 8
            stage_clear = False
        else:
            stage_clear = (mario_y <= MARIO_Y_WIN and
                           self._stage_before_clear not in self._ep_stages_cleared and
                           self._stage_clear_cooldown == 0)
            won = False   # game not finished until stage 4

        speed_bonus = (self._max_steps - self._step_count) * 0.5
        if won:
            reward += 1000.0 + speed_bonus   # full game clear
        elif stage_clear:
            reward += 500.0 + speed_bonus    # stage clear, episode continues

        # Time penalty — discourages idling and hesitation
        reward -= 0.05

        # Death: small penalty — risk-taking to climb should be acceptable
        if lives < self._prev_lives:
            reward -= 3.0

        if gameover:
            reward -= 5.0

        return reward, won, stage_clear
