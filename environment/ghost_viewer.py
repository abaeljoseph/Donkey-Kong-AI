"""
Ghost viewer:
  Background  : actual full-res NES color frame from env 0
  Ghost dots  : Mario detected by the blue of his overalls in the color frame
                (env 0) or via motion + size filtering in the grayscale obs
                (all other envs). Barrels are brown, platforms are dark red,
                ladders are teal — only Mario's overalls are that mid-blue.
"""

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Window is 3× NES resolution
_NES_W, _NES_H = 256, 240
_SCALE          = 3
_WIN_W          = _NES_W * _SCALE   # 768
_WIN_H          = _NES_H * _SCALE   # 720

_OBS_W = _OBS_H = 84   # preprocessed frame size

_MARIO_Y_WIN = 30       # win-line in NES coords

_COLORS = [
    (255,  80,  80), ( 80, 255,  80), ( 80, 130, 255), (255, 255,  80),
    (255,  80, 255), ( 80, 255, 255), (255, 165,   0), (200, 200, 200),
    (255, 140, 180), (140, 255, 140), (140, 140, 255), (200, 200,   0),
    (  0, 200, 200), (200,   0, 200), (255, 200, 100), (100, 200, 255),
]

# ── Mario detection ────────────────────────────────────────────────────────────

def _find_mario_color(frame_rgb):
    """
    Detect Mario purely by his skin colour HSV=(5, 74, 248).

    S=74 is uniquely low in this game — every other object (barrels, platforms
    H=168, ladders H=88, DK) has S=255.  Capping S<110 isolates Mario's skin
    from everything else with zero false positives.

    Pauline at the top of the level also has skin, so we exclude the top 20%.
    """
    if frame_rgb is None:
        return None

    cutoff = frame_rgb.shape[0] // 5   # top 20 % = DK + Pauline + score HUD

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Exact sampled value: H=5, S=74, V=248
    # Tight S cap (<110) is the key — nothing else in DK NES has such low S
    skin = cv2.inRange(hsv,
                       np.array([ 0,  45, 220]),
                       np.array([12, 110, 255]))
    skin[:cutoff] = 0

    ys, xs = np.where(skin > 0)
    if len(xs) < 3:
        return None

    return (int(np.median(xs)), int(np.median(ys)))


def _find_mario_motion(obs_stack):
    """
    Fallback: detect Mario from motion between grayscale obs frames.
    Excludes top 25 % (DK + score), keeps only small blobs (Mario < barrels).
    obs_stack: (4, 84, 84) float32.
    Returns (win_x, win_y) in window pixel coords, or None.
    """
    motion = np.abs(obs_stack[-1].astype(np.float32) - obs_stack[-2].astype(np.float32))

    # Ignore top area (DK / score display) and bottom strip (floor noise)
    motion[:int(_OBS_H * 0.25), :] = 0

    threshold = 0.15
    mask = motion > threshold
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None

    # Prefer small clusters (Mario ~2-3 px in 84-px obs; barrels are larger)
    # Simple: take the centroid of pixels in the lower half first
    lower = ys > _OBS_H * 0.4
    if np.any(lower):
        ys, xs = ys[lower], xs[lower]

    cx = float(np.median(xs))
    cy = float(np.median(ys))
    return (int(cx * _WIN_W / _OBS_W), int(cy * _WIN_H / _OBS_H))


# ── Callback ───────────────────────────────────────────────────────────────────

class GhostViewerCallback(BaseCallback):

    def __init__(self, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self._num_envs   = num_envs
        default_pos      = (_WIN_W // 2, int(0.73 * _WIN_H))  # approx start (bottom)
        self._pos        = [default_pos] * num_envs
        self._ep_rewards = [0.0] * num_envs
        self._best_env   = 0
        self._pygame     = None

    def _on_training_start(self) -> None:
        try:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((_WIN_W, _WIN_H))
            pygame.display.set_caption('Donkey Kong — Ghost Viewer')
            self._font_sm  = pygame.font.SysFont(None, 16)
            self._font_hud = pygame.font.SysFont(None, 24)
            self._clock    = pygame.time.Clock()
        except Exception as e:
            print(f'[GhostViewer] pygame init failed ({e}), viewer disabled.')
            self._pygame = None

    def _on_step(self) -> bool:
        if self._pygame is None:
            return True

        pg      = self._pygame
        infos   = self.locals.get('infos',   [])
        rewards = self.locals.get('rewards', [])
        new_obs = self.locals.get('new_obs')   # (N, 4, 84, 84) float32

        # Cumulative reward tracking
        for i, info in enumerate(infos):
            if i < len(rewards):
                self._ep_rewards[i] += float(rewards[i])
            if info.get('episode'):
                self._ep_rewards[i] = 0.0

        self._best_env = int(np.argmax(self._ep_rewards))

        # ── Background: full-res NES color frame from env 0 ─────────────
        raw_frame = infos[0].get('_raw_frame') if infos else None

        if raw_frame is not None:
            h, w = raw_frame.shape[:2]
            # pygame surfarray wants (W, H, 3): transpose axes 0 and 1
            surf   = pg.surfarray.make_surface(raw_frame.transpose(1, 0, 2))
            scaled = pg.transform.scale(surf, (_WIN_W, _WIN_H))
            self._screen.blit(scaled, (0, 0))
            frame_w, frame_h = w, h
        elif new_obs is not None:
            frame_u8  = (np.clip(new_obs[0, -1], 0, 1) * 255).astype(np.uint8)
            frame_rgb = np.stack([frame_u8] * 3, axis=2)
            surf      = pg.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
            scaled    = pg.transform.scale(surf, (_WIN_W, _WIN_H))
            self._screen.blit(scaled, (0, 0))
            frame_w, frame_h = _OBS_W, _OBS_H
        else:
            self._screen.fill((15, 15, 30))
            frame_w, frame_h = _NES_W, _NES_H

        # ── Ghost positions ──────────────────────────────────────────────

        # All envs: colour detection from their _detect_frame (160×150, throttled).
        # Env 0 also uses _raw_frame as higher-quality fallback for its own detection.
        for i in range(min(self._num_envs, len(infos))):
            info_i = infos[i]

            # Choose the best available frame for this env
            if i == 0 and raw_frame is not None:
                det = raw_frame          # full-res for env 0
                fw, fh = frame_w, frame_h
            else:
                det = info_i.get('_detect_frame')   # 160×150 for all envs
                if det is not None:
                    fh, fw = det.shape[:2]
                else:
                    det, fw, fh = None, 1, 1

            if det is not None:
                pos = _find_mario_color(det)
                if pos is not None:
                    self._pos[i] = (int(pos[0] * _WIN_W / fw),
                                    int(pos[1] * _WIN_H / fh))

        # ── Win line ─────────────────────────────────────────────────────
        win_y_px = int(_MARIO_Y_WIN * _WIN_H / _NES_H)
        pg.draw.line(self._screen, (0, 220, 0),
                     (0, win_y_px), (_WIN_W, win_y_px), 2)

        # ── Ghost dots ───────────────────────────────────────────────────
        for i, (wx, wy) in enumerate(self._pos):
            color   = _COLORS[i % len(_COLORS)]
            is_best = (i == self._best_env)
            r       = 8 if is_best else 5
            if is_best:
                pg.draw.circle(self._screen, (255, 255, 255), (wx, wy), r + 3, 2)
            pg.draw.circle(self._screen, color, (wx, wy), r)
            tag = self._font_sm.render(f'E{i}{"★" if is_best else ""}', True, color)
            self._screen.blit(tag, (wx + r + 2, wy - 7))

        # ── HUD ──────────────────────────────────────────────────────────
        txt = f'steps: {self.num_timesteps:,}   best: E{self._best_env}'
        self._screen.blit(self._font_hud.render(txt, True, (0, 0, 0)),       (7, 7))
        self._screen.blit(self._font_hud.render(txt, True, (230, 230, 230)), (6, 6))

        pg.display.flip()
        self._clock.tick(30)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._pygame = None
                pg.quit()

        return True

    def _on_training_end(self) -> None:
        if self._pygame:
            self._pygame.quit()
