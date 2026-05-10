"""
Ghost viewer:
  Background  : actual full-res NES color frame from env 0
  Ghost dots  : Mario detected by skin + blue overalls + red hat all adjacent
  Barrel dots : rolling barrels detected by their orange/amber body colour
  DK zone     : fixed exclusion rectangle so DK is never mistaken for Mario
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

_MARIO_Y_WIN   = 30    # win-line in NES coords
_MARIO_Y_START = 176   # Mario's Y at episode start (bottom)

_COLORS = [
    (255,  80,  80), ( 80, 255,  80), ( 80, 130, 255), (255, 255,  80),
    (255,  80, 255), ( 80, 255, 255), (255, 165,   0), (200, 200, 200),
    (255, 140, 180), (140, 255, 140), (140, 140, 255), (200, 200,   0),
    (  0, 200, 200), (200,   0, 200), (255, 200, 100), (100, 200, 255),
]

# ── Exclusion zones — tune with tools/debug_detection.py ──────────────────────
# Run:  python tools/debug_detection.py
# Press H to draw HUD zone, D to draw DK zone, P to print values, then paste here.
_HUD_Y          = 0.106  # top strip: score / bonus / lives display (full width)
_DK_X0, _DK_X1 = 0.000, 0.322   # DK horizontal range
_DK_Y0, _DK_Y1 = 0.078, 0.254   # DK vertical range


def _exclusion_mask(h, w):
    """Return a uint8 mask (255 = excluded) for HUD and DK zone."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:int(h * _HUD_Y), :] = 255                             # HUD strip
    mask[int(h * _DK_Y0):int(h * _DK_Y1),
         int(w * _DK_X0):int(w * _DK_X1)] = 255                # DK body
    return mask


# ── Mario detection ────────────────────────────────────────────────────────────

def _find_mario_color(frame_rgb):
    """
    Detect Mario by requiring skin, blue overalls, AND red hat/shirt all adjacent.

    Sampled pixels on Mario's sprite:
      Skin (47,195): HSV=(  5,  74, 248)  — uniquely low S
      Blue (46,194): HSV=(120, 255, 168)  — pure blue, not teal ladders (H≈88)
      Red  (47,192): HSV=(  6, 255, 216)  — saturated red, unlike skin (S=74)

    DK has skin but no adjacent blue/red — and is blocked by the exclusion zone.
    Barrels have red+blue but no adjacent skin.
    """
    if frame_rgb is None:
        return None

    h, w = frame_rgb.shape[:2]
    excl = _exclusion_mask(h, w)

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    skin = cv2.inRange(hsv, np.array([  0,  45, 215]), np.array([ 12, 110, 255]))
    blue = cv2.inRange(hsv, np.array([115, 230, 130]), np.array([125, 255, 210]))
    red  = cv2.inRange(hsv, np.array([  0, 200, 180]), np.array([ 12, 255, 240]))

    skin[excl > 0] = 0
    blue[excl > 0] = 0
    red [excl > 0] = 0

    k         = np.ones((14, 14), np.uint8)
    skin_near = cv2.dilate(skin, k)
    blue_near = cv2.dilate(blue, k)
    red_near  = cv2.dilate(red,  k)

    mario_region = cv2.bitwise_and(skin_near, cv2.bitwise_and(blue_near, red_near))
    mario_pixels = cv2.bitwise_and(skin, mario_region)

    ys, xs = np.where(mario_pixels > 0)
    if len(xs) < 3:
        return None

    return (int(np.median(xs)), int(np.median(ys)))


# ── Barrel detection ───────────────────────────────────────────────────────────

def _find_barrels_color(frame_rgb):
    """
    Detect rolling barrels by their orange/amber body colour.
    Returns list of (x, y) positions in frame pixel coords.

    Barrel orange body: H≈20-30, S=180+, V=100-225.
    Mario has no orange; ladders/platforms/DK have no orange.
    """
    if frame_rgb is None:
        return []

    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Barrel orange body: sampled H=15,S=197,V=248
    orange = cv2.inRange(hsv,
                         np.array([10, 160, 220]),
                         np.array([25, 215, 255]))

    # Exclude HUD strip and DK zone (stacked barrel pile sits there)
    orange[:int(h * _HUD_Y), :] = 0
    orange[int(h * _DK_Y0):int(h * _DK_Y1),
           int(w * _DK_X0):int(w * _DK_X1)] = 0

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(orange, connectivity=8)
    barrels = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 15 < area < 800:   # filter noise and large non-barrel blobs
            cx, cy = centroids[i]
            barrels.append((int(cx), int(cy)))
    return barrels


# ── Callback ───────────────────────────────────────────────────────────────────

class GhostViewerCallback(BaseCallback):

    def __init__(self, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self._num_envs      = num_envs
        default_pos         = (_WIN_W // 2, int(0.73 * _WIN_H))
        self._pos           = [default_pos] * num_envs
        self._ep_rewards    = [0.0] * num_envs
        self._best_env      = 0
        self._pygame        = None
        self._cached_bg     = None    # last received full-color frame from env 0

        # Height chart state
        self._show_chart    = True    # toggle with C key
        self._ep_heights    = []      # best height (px climbed from start) per finished episode
        self._ep_start_y    = [None] * num_envs   # first detected mario_y of each episode
        self._ep_cur_best   = [None] * num_envs   # lowest mario_y seen this episode (lower=higher)
        self._cur_height_px = [0]    * num_envs   # live pixels climbed this step

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

        # Cumulative reward tracking + episode-end height recording
        for i, info in enumerate(infos):
            if i < len(rewards):
                self._ep_rewards[i] += float(rewards[i])
            if info.get('episode'):
                # Record best height reached this episode
                start = self._ep_start_y[i]
                best  = self._ep_cur_best[i]
                if start is not None and best is not None:
                    self._ep_heights.append(max(0, start - best))
                self._ep_start_y[i]   = None
                self._ep_cur_best[i]  = None
                self._cur_height_px[i] = 0
                self._ep_rewards[i]   = 0.0

        self._best_env = int(np.argmax(self._ep_rewards))

        # ── Background: full-res NES color frame from env 0 ─────────────
        raw_frame = infos[0].get('_raw_frame') if infos else None
        if raw_frame is not None:
            self._cached_bg = raw_frame

        if self._cached_bg is not None:
            h, w = self._cached_bg.shape[:2]
            surf   = pg.surfarray.make_surface(self._cached_bg.transpose(1, 0, 2))
            scaled = pg.transform.scale(surf, (_WIN_W, _WIN_H))
            self._screen.blit(scaled, (0, 0))
            frame_w, frame_h = w, h
        else:
            self._screen.fill((15, 15, 30))
            frame_w, frame_h = _NES_W, _NES_H

        # ── Exclusion zone overlays ──────────────────────────────────────
        hud_h_px = int(_HUD_Y * _WIN_H)
        hud_surf = pg.Surface((_WIN_W, hud_h_px), pg.SRCALPHA)
        hud_surf.fill((255, 50, 50, 80))
        self._screen.blit(hud_surf, (0, 0))
        pg.draw.line(self._screen, (255, 80, 80), (0, hud_h_px), (_WIN_W, hud_h_px), 1)
        self._screen.blit(self._font_sm.render('HUD', True, (255, 120, 120)), (4, 2))

        dk_x0_px = int(_DK_X0 * _WIN_W)
        dk_y0_px = int(_DK_Y0 * _WIN_H)
        dk_w_px  = int((_DK_X1 - _DK_X0) * _WIN_W)
        dk_h_px  = int((_DK_Y1 - _DK_Y0) * _WIN_H)
        dk_surf  = pg.Surface((dk_w_px, dk_h_px), pg.SRCALPHA)
        dk_surf.fill((255, 50, 50, 80))
        self._screen.blit(dk_surf, (dk_x0_px, dk_y0_px))
        pg.draw.rect(self._screen, (255, 80, 80), (dk_x0_px, dk_y0_px, dk_w_px, dk_h_px), 2)
        self._screen.blit(self._font_sm.render('DK ZONE', True, (255, 120, 120)),
                          (dk_x0_px + 4, dk_y0_px + 4))

        # ── Barrel dots (from env 0 cached frame) ───────────────────────
        if self._cached_bg is not None:
            for bx, by in _find_barrels_color(self._cached_bg):
                bx_win = int(bx * _WIN_W / frame_w)
                by_win = int(by * _WIN_H / frame_h)
                pg.draw.circle(self._screen, (255, 140, 0), (bx_win, by_win), 7)
                pg.draw.circle(self._screen, (0, 0, 0), (bx_win, by_win), 7, 2)
                self._screen.blit(self._font_sm.render('B', True, (0, 0, 0)),
                                  (bx_win - 4, by_win - 6))

        # ── Ghost positions ──────────────────────────────────────────────
        for i in range(min(self._num_envs, len(infos))):
            info_i = infos[i]

            if i == 0 and raw_frame is not None:
                det = raw_frame
                fw, fh = frame_w, frame_h
            else:
                det = info_i.get('_detect_frame')
                if det is not None:
                    fh, fw = det.shape[:2]
                else:
                    det, fw, fh = None, 1, 1

            if det is not None:
                pos = _find_mario_color(det)
                if pos is not None:
                    self._pos[i] = (int(pos[0] * _WIN_W / fw),
                                    int(pos[1] * _WIN_H / fh))
                    mario_nes_y = int(pos[1] * _NES_H / fh)
                    # Calibrate start of episode on first successful detection
                    if self._ep_start_y[i] is None:
                        self._ep_start_y[i]  = mario_nes_y
                        self._ep_cur_best[i] = mario_nes_y
                    else:
                        self._ep_cur_best[i] = min(self._ep_cur_best[i], mario_nes_y)
                    self._cur_height_px[i] = max(0, self._ep_start_y[i] - mario_nes_y)

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
        txt = f'steps: {self.num_timesteps:,}   best: E{self._best_env}   [C] chart {"ON" if self._show_chart else "OFF"}'
        self._screen.blit(self._font_hud.render(txt, True, (0, 0, 0)),       (7, 7))
        self._screen.blit(self._font_hud.render(txt, True, (230, 230, 230)), (6, 6))

        # ── Height chart ─────────────────────────────────────────────────
        if self._show_chart and len(self._ep_heights) >= 2:
            self._draw_height_chart(pg)

        pg.display.flip()
        self._clock.tick(0)   # uncapped — run as fast as training allows

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._pygame = None
                pg.quit()
            elif event.type == pg.KEYDOWN and event.key == pg.K_c:
                self._show_chart = not self._show_chart

        return True

    def _draw_height_chart(self, pg):
        """
        Left panel : historical line — best height per finished episode (all envs combined).
        Right panel: live bars — current height of each env this episode.
        """
        CHART_W, CHART_H = 420, 150
        CHART_X, CHART_Y = _WIN_W - CHART_W - 10, _WIN_H - CHART_H - 10
        TOTAL_HEIGHT     = _MARIO_Y_START - _MARIO_Y_WIN   # ~146 NES px
        SPLIT            = 260   # pixels wide for the history panel

        # ── Background ───────────────────────────────────────────────────
        panel = pg.Surface((CHART_W, CHART_H), pg.SRCALPHA)
        panel.fill((10, 10, 30, 210))
        self._screen.blit(panel, (CHART_X, CHART_Y))
        pg.draw.rect(self._screen, (80, 80, 160), (CHART_X, CHART_Y, CHART_W, CHART_H), 1)
        # Divider between history and live bars
        pg.draw.line(self._screen, (80, 80, 160),
                     (CHART_X + SPLIT, CHART_Y), (CHART_X + SPLIT, CHART_Y + CHART_H), 1)

        # ── History panel (left) ─────────────────────────────────────────
        hist_x0  = CHART_X + 28
        hist_y0  = CHART_Y + 20
        hist_h   = CHART_H - 30
        hist_w   = SPLIT - 36

        self._screen.blit(self._font_sm.render(f'Best height/ep ({len(self._ep_heights)} eps)', True, (180, 180, 255)),
                          (CHART_X + 4, CHART_Y + 4))
        self._screen.blit(self._font_sm.render(f'{TOTAL_HEIGHT}px', True, (100, 255, 100)), (CHART_X + 2, hist_y0))
        self._screen.blit(self._font_sm.render('0px',              True, (200, 100, 100)), (CHART_X + 2, hist_y0 + hist_h - 12))

        # Win line
        pg.draw.line(self._screen, (0, 180, 0),
                     (hist_x0, hist_y0), (hist_x0 + hist_w, hist_y0), 1)

        data = self._ep_heights[-hist_w:]
        n = len(data)
        if n >= 2:
            points = []
            for idx, h in enumerate(data):
                frac = min(h / TOTAL_HEIGHT, 1.0)
                cx   = hist_x0 + int(idx * hist_w / (n - 1))
                cy   = hist_y0 + hist_h - int(frac * hist_h)
                points.append((cx, cy))
            pg.draw.lines(self._screen, (100, 200, 255), False, points, 2)
            lbl = self._font_sm.render(f'{int(data[-1])}px', True, (100, 200, 255))
            self._screen.blit(lbl, (points[-1][0] + 2, points[-1][1] - 8))

        # ── Live bars panel (right) ──────────────────────────────────────
        live_x0  = CHART_X + SPLIT + 6
        live_y0  = CHART_Y + 20
        live_h   = CHART_H - 30
        live_w   = CHART_W - SPLIT - 12
        n_envs   = self._num_envs
        bar_w    = max(1, (live_w - (n_envs - 1) * 2) // n_envs)

        self._screen.blit(self._font_sm.render(f'Live  (0-{TOTAL_HEIGHT}px)', True, (180, 180, 255)),
                          (live_x0, CHART_Y + 4))
        # Win line
        pg.draw.line(self._screen, (0, 180, 0),
                     (live_x0, live_y0), (live_x0 + live_w, live_y0), 1)
        # Bot line
        pg.draw.line(self._screen, (80, 40, 40),
                     (live_x0, live_y0 + live_h), (live_x0 + live_w, live_y0 + live_h), 1)

        for i in range(n_envs):
            bx      = live_x0 + i * (bar_w + 2)
            height  = self._cur_height_px[i] if i < len(self._cur_height_px) else 0
            frac    = max(0.0, min(height / TOTAL_HEIGHT, 1.0))
            bar_h   = max(1, int(frac * live_h))
            color   = _COLORS[i % len(_COLORS)]
            pg.draw.rect(self._screen, color,
                         (bx, live_y0 + live_h - bar_h, bar_w, bar_h))
            # Pixel label above bar if climbing
            if height > 2:
                lbl = self._font_sm.render(f'{int(height)}', True, color)
                self._screen.blit(lbl, (bx, live_y0 + live_h - bar_h - 12))
            lbl = self._font_sm.render(f'E{i}', True, color)
            self._screen.blit(lbl, (bx, live_y0 + live_h + 2))

    def _on_training_end(self) -> None:
        if self._pygame:
            self._pygame.quit()
