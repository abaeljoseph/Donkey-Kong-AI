"""
Ghost viewer:
  Background  : actual full-res NES color frame from env 0
  Ghost dots  : Mario detected by skin + blue overalls + red hat all adjacent
  Barrel dots : rolling barrels detected by their orange/amber body colour
  DK zone     : fixed exclusion rectangle so DK is never mistaken for Mario
"""

import collections
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Window is 3× NES resolution
_NES_W, _NES_H = 240, 224
_SCALE          = 3
_WIN_W          = _NES_W * _SCALE   # 720
_WIN_H          = _NES_H * _SCALE   # 672

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


# ── Oil can detection ─────────────────────────────────────────────────────────

def _find_oil_can(frame_rgb):
    """
    Detect the oil can in the bottom-left corner by its blue colour.
    Sampled pixel HSV=(116,220,232). Returns (x, y, w, h) bounding box or None.
    """
    if frame_rgb is None:
        return None
    h, w = frame_rgb.shape[:2]
    hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([110, 190, 200]), np.array([125, 255, 255]))
    # Only look in the bottom-left quarter
    mask = np.zeros_like(blue)
    mask[int(h * 0.75):, :int(w * 0.25)] = 255
    blue = cv2.bitwise_and(blue, mask)
    ys, xs = np.where(blue > 0)
    if len(xs) < 10:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return (x0, y0, x1 - x0, y1 - y0)



# ── Fire detection ─────────────────────────────────────────────────────────────

def _find_fires_color(frame_rgb):
    """
    Detect fires/fireballs by their bright red colour.
    Fires in DK NES are brighter red than barrels (H≈0-8 vs barrel H≈10-25).
    Returns list of (x, y) positions in frame pixel coords.

    Tune with: python tools/debug_detection.py — click a fire pixel and read HSV.
    """
    if frame_rgb is None:
        return []

    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Fire cream/low-saturation orange — sampled HSV=(18,82,248)
    # Barrels are S=160+, Mario red is S=200+, fire sits at S=50-130
    fire = cv2.inRange(hsv,
                       np.array([ 12,  50, 220]),
                       np.array([ 25, 130, 255]))

    # Exclude HUD and DK zone
    fire[:int(h * _HUD_Y), :] = 0
    fire[int(h * _DK_Y0):int(h * _DK_Y1),
         int(w * _DK_X0):int(w * _DK_X1)] = 0

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(fire, connectivity=8)
    fires = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 8 < area < 400:   # fires are smaller than barrels
            cx, cy = centroids[i]
            fires.append((int(cx), int(cy)))
    return fires


# ── Callback ───────────────────────────────────────────────────────────────────

class GhostViewerCallback(BaseCallback):

    def __init__(self, num_envs: int, all_ladders=None, broken_zones=None, verbose: int = 0):
        super().__init__(verbose)
        self._num_envs      = num_envs
        default_pos         = (_WIN_W // 2, int(0.73 * _WIN_H))
        self._pos           = [default_pos] * num_envs
        self._ep_rewards    = [0.0] * num_envs
        self._step_rewards  = [0.0] * num_envs
        self._reward_scale  = 10.0
        self._best_env      = 0
        self._pygame        = None
        self._cached_bg     = None

        # Height chart state
        self._show_chart    = True
        self._ep_heights    = []
        self._ep_start_y    = [None] * num_envs
        self._ep_cur_best   = [None] * num_envs
        self._cur_height_px = [0]    * num_envs

        # Reward history chart state — keeps last 300 step rewards per env
        self._show_reward_chart = True
        self._reward_history    = [collections.deque(maxlen=300) for _ in range(num_envs)]

        # Static level data from startup detection (NES coords)
        self._all_ladders  = all_ladders  or []   # (nes_x0, nes_y0, nes_x1, nes_y1)
        self._broken_zones = broken_zones or []   # broken penalty zones in NES coords

        # Overlay toggles
        self._show_ladders = True    # ladder labels + broken zone rectangles
        self._show_zones   = True    # HUD + DK exclusion zone overlays
        self._show_dangers = True    # barrel + fire dots

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
            print(f'[GhostViewer] window open ({_WIN_W}×{_WIN_H}) — check your taskbar if you can\'t see it', flush=True)
        except Exception as e:
            print(f'[GhostViewer] FAILED TO OPEN — {type(e).__name__}: {e}', flush=True)
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
            sr = info.get('_step_reward')
            if sr is not None and i < self._num_envs:
                sr = float(sr)
                self._step_rewards[i] = sr
                self._reward_history[i].append(sr)
                if abs(sr) > self._reward_scale:
                    self._reward_scale = abs(sr)
            if info.get('episode'):
                best = self._ep_cur_best[i]
                if best is not None:
                    self._ep_heights.append(max(0, _MARIO_Y_START - best))
                self._ep_start_y[i]    = None
                self._ep_cur_best[i]   = None
                self._cur_height_px[i] = 0
                self._ep_rewards[i]    = 0.0
                self._step_rewards[i]  = 0.0

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

        # ── Exclusion zone overlays (HUD + DK zone) ─────────────────────
        if self._show_zones:
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

        # ── Oil can ──────────────────────────────────────────────────────
        oil = _find_oil_can(self._cached_bg) if self._cached_bg is not None else None
        if oil:
            ox, oy, ow, oh = oil
            ox_w = int(ox * _WIN_W / frame_w);  oy_w = int(oy * _WIN_H / frame_h)
            ow_w = max(int(ow * _WIN_W / frame_w), 6)
            oh_w = max(int(oh * _WIN_H / frame_h), 6)
            oil_surf = pg.Surface((ow_w + 10, oh_w + 10), pg.SRCALPHA)
            oil_surf.fill((255, 30, 30, 100))
            self._screen.blit(oil_surf, (ox_w - 5, oy_w - 5))
            pg.draw.rect(self._screen, (255, 30, 30), (ox_w - 5, oy_w - 5, ow_w + 10, oh_w + 10), 2)
            self._screen.blit(self._font_sm.render('OIL ☠', True, (255, 80, 80)), (ox_w, oy_w - 14))

        # ── Ladders + broken zones (static reference coords from startup) ─
        if self._show_ladders and self._all_ladders:
            for idx, (nx0, ny0, nx1, ny1) in enumerate(self._all_ladders):
                cx_nes = (nx0 + nx1) // 2
                cy_nes = (ny0 + ny1) // 2
                is_broken = any(
                    z[0] <= cx_nes <= z[2] and z[1] <= cy_nes <= z[3]
                    for z in self._broken_zones
                )
                color  = (255, 60, 60) if is_broken else (0, 220, 220)
                wx0    = int(nx0 * _WIN_W / 256)
                wy0    = int(ny0 * _WIN_H / 224)
                ww     = max(int((nx1 - nx0) * _WIN_W / 256), 4)
                wh     = max(int((ny1 - ny0) * _WIN_H / 224), 4)
                pg.draw.rect(self._screen, color, (wx0, wy0, ww, wh), 2)
                prefix = 'BROKEN' if is_broken else f'L{idx}'
                label  = f'{prefix} ({nx0},{ny0})-({nx1},{ny1})'
                self._screen.blit(self._font_sm.render(label, True, color),
                                  (wx0, max(wy0 - 13, 0)))
            # Broken penalty zone rectangles (orange fill)
            for z in self._broken_zones:
                wx0 = int(z[0] * _WIN_W / 256);  wy0 = int(z[1] * _WIN_H / 224)
                wx1 = int(z[2] * _WIN_W / 256);  wy1 = int(z[3] * _WIN_H / 224)
                zsurf = pg.Surface((wx1 - wx0, wy1 - wy0), pg.SRCALPHA)
                zsurf.fill((255, 120, 0, 40))
                self._screen.blit(zsurf, (wx0, wy0))
                pg.draw.rect(self._screen, (255, 120, 0), (wx0, wy0, wx1 - wx0, wy1 - wy0), 1)

        # ── Barrel + fire dots ───────────────────────────────────────────
        if self._show_dangers and self._cached_bg is not None:
            for bx, by in _find_barrels_color(self._cached_bg):
                bx_win = int(bx * _WIN_W / frame_w)
                by_win = int(by * _WIN_H / frame_h)
                pg.draw.circle(self._screen, (255, 140, 0), (bx_win, by_win), 7)
                pg.draw.circle(self._screen, (0, 0, 0), (bx_win, by_win), 7, 2)
                self._screen.blit(self._font_sm.render('B', True, (0, 0, 0)),
                                  (bx_win - 4, by_win - 6))
            for fx, fy in _find_fires_color(self._cached_bg):
                fx_win = int(fx * _WIN_W / frame_w)
                fy_win = int(fy * _WIN_H / frame_h)
                pg.draw.circle(self._screen, (255, 30, 30), (fx_win, fy_win), 6)
                pg.draw.circle(self._screen, (255, 255, 0), (fx_win, fy_win), 6, 2)
                self._screen.blit(self._font_sm.render('F', True, (255, 255, 0)),
                                  (fx_win - 4, fy_win - 6))

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
            sr = self._step_rewards[i] if i < len(self._step_rewards) else 0.0
            sr_col = (80, 255, 80) if sr > 0.1 else (255, 80, 80) if sr < -0.1 else (180, 180, 180)
            self._screen.blit(self._font_sm.render(f'{sr:+.1f}', True, sr_col), (wx + r + 2, wy + 5))

        # ── HUD ──────────────────────────────────────────────────────────
        _STAGE_LABEL = {1: 'Barrels', 3: 'Elevator', 4: 'Rivets'}
        _stage = infos[0].get('_stage', '?') if infos else '?'
        _level = infos[0].get('_level', '?') if infos else '?'
        _stage_str = _STAGE_LABEL.get(_stage, f'stage={_stage}')
        _lv_str = (_level + 1) if isinstance(_level, int) else _level
        txt1 = f'steps: {self.num_timesteps:,}   best: E{self._best_env}   Lv{_lv_str} {_stage_str}'
        txt2 = (f'[C] height {"ON" if self._show_chart else "OFF"}'
                f'  [R] rewards {"ON" if self._show_reward_chart else "OFF"}'
                f'  [L] ladders {"ON" if self._show_ladders else "OFF"}'
                f'  [Z] HUD/DK {"ON" if self._show_zones else "OFF"}'
                f'  [D] dangers {"ON" if self._show_dangers else "OFF"}')
        self._screen.blit(self._font_hud.render(txt1, True, (0, 0, 0)),       (7,  7))
        self._screen.blit(self._font_hud.render(txt1, True, (230, 230, 230)), (6,  6))
        self._screen.blit(self._font_sm.render(txt2,  True, (0, 0, 0)),       (7, 27))
        self._screen.blit(self._font_sm.render(txt2,  True, (200, 200, 200)), (6, 26))

        # ── Height chart ─────────────────────────────────────────────────
        if self._show_chart and len(self._ep_heights) >= 2:
            self._draw_height_chart(pg)

        # ── Reward history chart ─────────────────────────────────────────
        if self._show_reward_chart:
            self._draw_reward_chart(pg)
        pg.display.flip()
        self._clock.tick(0)   # uncapped — run as fast as training allows

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._pygame = None
                pg.quit()
            elif event.type == pg.KEYDOWN and event.key == pg.K_c:
                self._show_chart = not self._show_chart
            elif event.type == pg.KEYDOWN and event.key == pg.K_r:
                self._show_reward_chart = not self._show_reward_chart
            elif event.type == pg.KEYDOWN and event.key == pg.K_l:
                self._show_ladders = not self._show_ladders
            elif event.type == pg.KEYDOWN and event.key == pg.K_z:
                self._show_zones = not self._show_zones
            elif event.type == pg.KEYDOWN and event.key == pg.K_d:
                self._show_dangers = not self._show_dangers

        return True

    def _draw_reward_chart(self, pg):
        """
        Bottom-left panel: step reward history for each env as separate coloured lines.
        Useful for spotting negative spikes (e.g. broken-ladder penalty of -50).
        Toggle with R key.
        """
        CHART_W, CHART_H = 278, 150
        CHART_X          = 4
        CHART_Y          = _WIN_H - CHART_H - 10

        panel = pg.Surface((CHART_W, CHART_H), pg.SRCALPHA)
        panel.fill((10, 10, 30, 210))
        self._screen.blit(panel, (CHART_X, CHART_Y))
        pg.draw.rect(self._screen, (80, 80, 160), (CHART_X, CHART_Y, CHART_W, CHART_H), 1)
        self._screen.blit(self._font_sm.render('Step reward history [R]', True, (180, 180, 255)),
                          (CHART_X + 4, CHART_Y + 3))

        # Plot area
        plot_x0 = CHART_X + 26
        plot_y0 = CHART_Y + 16
        plot_w  = CHART_W - 30
        plot_h  = CHART_H - 24

        # Dynamic y scale across all histories
        all_vals = [v for hist in self._reward_history for v in hist]
        if not all_vals:
            return
        y_min = min(all_vals)
        y_max = max(all_vals)
        if y_max == y_min:
            y_min -= 1.0; y_max += 1.0

        def to_screen_y(v):
            frac = (v - y_min) / (y_max - y_min)
            return max(plot_y0, min(plot_y0 + plot_h, plot_y0 + plot_h - int(frac * plot_h)))

        # Zero line
        zero_y = to_screen_y(0.0)
        pg.draw.line(self._screen, (120, 120, 120),
                     (plot_x0, zero_y), (plot_x0 + plot_w, zero_y), 1)

        # Y-axis labels
        self._screen.blit(self._font_sm.render(f'{y_max:+.0f}', True, (100, 255, 100)),
                          (CHART_X + 1, plot_y0))
        self._screen.blit(self._font_sm.render('0', True, (160, 160, 160)),
                          (CHART_X + 1, zero_y - 6))
        self._screen.blit(self._font_sm.render(f'{y_min:+.0f}', True, (255, 100, 100)),
                          (CHART_X + 1, plot_y0 + plot_h - 10))

        # One coloured line per env
        for i in range(self._num_envs):
            hist = list(self._reward_history[i])
            if len(hist) < 2:
                continue
            color = _COLORS[i % len(_COLORS)]
            n = len(hist)
            points = [
                (plot_x0 + int(idx * plot_w / max(n - 1, 1)), to_screen_y(v))
                for idx, v in enumerate(hist)
            ]
            pg.draw.lines(self._screen, color, False, points, 1)
            # Current value label at the right edge
            cur = hist[-1]
            cur_col = (80, 255, 80) if cur >= 0 else (255, 80, 80)
            self._screen.blit(
                self._font_sm.render(f'E{i} {cur:+.1f}', True, cur_col),
                (plot_x0 + plot_w + 2, to_screen_y(cur) - 5),
            )

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
