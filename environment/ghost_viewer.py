"""
Ghost viewer — training render that matches the debug tool visual style.

Background : actual full-res NES colour frame from env 0.
Sprites    : positions read from OAM info (same source as debug_env.py).
Ladders    : per-step rects from env._ladder_rects (stage-aware, frame pixel coords).
Overlays   : broken zones (stage 1), win line, ghost dots, reward + height charts.
"""

import collections
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_NES_W, _NES_H = 240, 224
_SCALE          = 3
_WIN_W          = _NES_W * _SCALE   # 720
_WIN_H          = _NES_H * _SCALE   # 672

_MARIO_Y_WIN   = 30
_MARIO_Y_START = 176

_COLORS = [
    (255,  80,  80), ( 80, 255,  80), ( 80, 130, 255), (255, 255,  80),
    (255,  80, 255), ( 80, 255, 255), (255, 165,   0), (200, 200, 200),
    (255, 140, 180), (140, 255, 140), (140, 140, 255), (200, 200,   0),
    (  0, 200, 200), (200,   0, 200), (255, 200, 100), (100, 200, 255),
]

_STAGE_LABEL = {1: 'Barrels', 2: 'Cement', 3: 'Elevator', 4: 'Rivets'}


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
        self._cached_ladder_rects = []   # frame pixel coords, updated from env 0 info
        self._cached_frame_w      = _NES_W
        self._cached_frame_h      = _NES_H

        # Height chart state
        self._show_chart    = True
        self._ep_heights    = []
        self._ep_start_y    = [None] * num_envs
        self._ep_cur_best   = [None] * num_envs
        self._cur_height_px = [0]    * num_envs

        # Reward history chart state
        self._show_reward_chart = True
        self._reward_history    = [collections.deque(maxlen=300) for _ in range(num_envs)]

        # Broken zones (stage 1 only) — synced each step from env 0 info
        self._broken_zones = broken_zones or []

        # Overlay toggles
        self._show_ladders = True
        self._show_dangers = True

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
            print(f'[GhostViewer] window open ({_WIN_W}×{_WIN_H})', flush=True)
        except Exception as e:
            print(f'[GhostViewer] FAILED TO OPEN — {type(e).__name__}: {e}', flush=True)
            self._pygame = None

    def _on_step(self) -> bool:
        if self._pygame is None:
            return True

        pg      = self._pygame
        infos   = self.locals.get('infos',   [])
        rewards = self.locals.get('rewards', [])

        # ── Per-env reward + height tracking ────────────────────────────────
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
                best  = self._ep_cur_best[i]
                start = self._ep_start_y[i]
                if best is not None and start is not None:
                    self._ep_heights.append(max(0, start - best))
                self._ep_start_y[i]    = None
                self._ep_cur_best[i]   = None
                self._cur_height_px[i] = 0
                self._ep_rewards[i]    = 0.0
                self._step_rewards[i]  = 0.0

        self._best_env = int(np.argmax(self._ep_rewards))

        # Sync broken zones and ladder rects from env 0
        if infos:
            bz = infos[0].get('_broken_zones')
            if bz is not None:
                self._broken_zones = bz
            lr = infos[0].get('_ladder_rects')
            if lr is not None:
                self._cached_ladder_rects = lr

        # ── Background frame ─────────────────────────────────────────────────
        raw_frame = infos[0].get('_raw_frame') if infos else None
        if raw_frame is not None:
            self._cached_bg      = raw_frame
            self._cached_frame_h = raw_frame.shape[0]
            self._cached_frame_w = raw_frame.shape[1]

        if self._cached_bg is not None:
            surf   = pg.surfarray.make_surface(self._cached_bg.transpose(1, 0, 2))
            scaled = pg.transform.scale(surf, (_WIN_W, _WIN_H))
            self._screen.blit(scaled, (0, 0))
        else:
            self._screen.fill((15, 15, 30))

        fw = self._cached_frame_w
        fh = self._cached_frame_h

        # ── Ladder rects (frame pixel coords → screen) ───────────────────────
        if self._show_ladders:
            for x0, y0, x1, y1 in self._cached_ladder_rects:
                wx0 = int(x0 * _WIN_W / fw)
                wy0 = int(y0 * _WIN_H / fh)
                ww  = max(int((x1 - x0) * _WIN_W / fw), 4)
                wh  = max(int((y1 - y0) * _WIN_H / fh), 4)
                pg.draw.rect(self._screen, (0, 220, 220), (wx0, wy0, ww, wh), 2)
            # Broken penalty zones (stage 1 only) — NES coords
            for z in self._broken_zones:
                wx0 = int(z[0] * _WIN_W / 256);  wy0 = int(z[1] * _WIN_H / 224)
                wx1 = int(z[2] * _WIN_W / 256);  wy1 = int(z[3] * _WIN_H / 224)
                if wx1 > wx0 and wy1 > wy0:
                    zsurf = pg.Surface((wx1 - wx0, wy1 - wy0), pg.SRCALPHA)
                    zsurf.fill((255, 120, 0, 40))
                    self._screen.blit(zsurf, (wx0, wy0))
                    pg.draw.rect(self._screen, (255, 120, 0),
                                 (wx0, wy0, wx1 - wx0, wy1 - wy0), 1)

        # ── Barrels + fires from OAM (env 0 only — matches background frame) ─
        # OAM coords: x=0-255, y=0-240 (raw hardware values).
        # Scale matches debug_env.py: x / 240, (y + 2) / 224.
        if self._show_dangers and infos:
            for bx, by in infos[0].get('_barrel_pts', []):
                bx_w = int(bx * _WIN_W / 240)
                by_w = int((by + 2) * _WIN_H / 224)
                pg.draw.circle(self._screen, (255, 140, 0), (bx_w, by_w), 7)
                pg.draw.circle(self._screen, (0, 0, 0),     (bx_w, by_w), 7, 2)
                self._screen.blit(self._font_sm.render('B', True, (0, 0, 0)),
                                  (bx_w - 4, by_w - 6))
            for fx, fy in infos[0].get('_fire_pts', []):
                fx_w = int(fx * _WIN_W / 240)
                fy_w = int((fy + 2) * _WIN_H / 224)
                pg.draw.circle(self._screen, (255, 30, 30),  (fx_w, fy_w), 6)
                pg.draw.circle(self._screen, (255, 255, 0),  (fx_w, fy_w), 6, 2)
                self._screen.blit(self._font_sm.render('F', True, (255, 255, 0)),
                                  (fx_w - 4, fy_w - 6))

        # ── Rivets (stage 4) — frame pixel coords → screen ──────────────────
        if infos:
            _riv_pos    = infos[0].get('_rivet_positions', [])
            _riv_popped = set(infos[0].get('_rivet_popped', []))
            for _ri, (_rx, _ry) in enumerate(_riv_pos):
                _popped = _ri in _riv_popped
                _rcol   = (80, 80, 80) if _popped else (255, 210, 30)
                _rpx    = int(_rx * _WIN_W / fw)
                _rpy    = int(_ry * _WIN_H / fh)
                pg.draw.rect(self._screen, _rcol, (_rpx - 8, _rpy - 8, 16, 16), 2)
                _lbl = 'REMOVED' if _popped else f'R{_ri}'
                self._screen.blit(self._font_sm.render(_lbl, True, _rcol),
                                  (_rpx + 10, _rpy - 6))

        # ── Ghost positions from OAM info (all envs) ─────────────────────────
        for i in range(min(self._num_envs, len(infos))):
            info_i      = infos[i]
            mario_y_nes = info_i.get('_mario_y')
            mario_x_nes = info_i.get('_mario_x')

            if mario_y_nes is not None:
                if self._ep_start_y[i] is None:
                    self._ep_start_y[i]  = mario_y_nes
                    self._ep_cur_best[i] = mario_y_nes
                else:
                    self._ep_cur_best[i] = min(self._ep_cur_best[i], mario_y_nes)
                self._cur_height_px[i] = max(0, self._ep_start_y[i] - mario_y_nes)

            if mario_x_nes is not None and mario_y_nes is not None:
                # mario_y_nes is pre-scaled (raw_oam_y * 224/240); recover raw then
                # apply the same formula as debug_env.py: x/240, (y+2)/224.
                raw_oam_y = mario_y_nes * 240 / 224
                self._pos[i] = (int(mario_x_nes * _WIN_W / 240),
                                int((raw_oam_y + 2) * _WIN_H / 224))

        # ── Win line ─────────────────────────────────────────────────────────
        win_y_px = int(_MARIO_Y_WIN * _WIN_H / _NES_H)
        pg.draw.line(self._screen, (0, 220, 0), (0, win_y_px), (_WIN_W, win_y_px), 2)

        # ── Ghost dots ───────────────────────────────────────────────────────
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
            self._screen.blit(self._font_sm.render(f'{sr:+.1f}', True, sr_col),
                              (wx + r + 2, wy + 5))

        # ── HUD ──────────────────────────────────────────────────────────────
        _stage  = infos[0].get('_stage', '?') if infos else '?'
        _level  = infos[0].get('_level', '?') if infos else '?'
        _stage_str = _STAGE_LABEL.get(_stage, f'stage={_stage}')
        _lv_str    = (_level + 1) if isinstance(_level, int) else _level
        txt1 = f'steps: {self.num_timesteps:,}   best: E{self._best_env}   Lv{_lv_str} {_stage_str}'
        txt2 = (f'[C] height {"ON" if self._show_chart else "OFF"}'
                f'  [R] rewards {"ON" if self._show_reward_chart else "OFF"}'
                f'  [L] ladders {"ON" if self._show_ladders else "OFF"}'
                f'  [D] dangers {"ON" if self._show_dangers else "OFF"}')
        self._screen.blit(self._font_hud.render(txt1, True, (0, 0, 0)),       (7,  7))
        self._screen.blit(self._font_hud.render(txt1, True, (230, 230, 230)), (6,  6))
        self._screen.blit(self._font_sm.render(txt2,  True, (0, 0, 0)),       (7, 27))
        self._screen.blit(self._font_sm.render(txt2,  True, (200, 200, 200)), (6, 26))

        # ── Charts ───────────────────────────────────────────────────────────
        if self._show_chart and len(self._ep_heights) >= 2:
            self._draw_height_chart(pg)
        if self._show_reward_chart:
            self._draw_reward_chart(pg)

        pg.display.flip()
        self._clock.tick(0)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._pygame = None
                pg.quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_c:
                    self._show_chart = not self._show_chart
                elif event.key == pg.K_r:
                    self._show_reward_chart = not self._show_reward_chart
                elif event.key == pg.K_l:
                    self._show_ladders = not self._show_ladders
                elif event.key == pg.K_d:
                    self._show_dangers = not self._show_dangers

        return True

    def _draw_reward_chart(self, pg):
        CHART_W, CHART_H = 278, 150
        CHART_X          = 4
        CHART_Y          = _WIN_H - CHART_H - 10

        panel = pg.Surface((CHART_W, CHART_H), pg.SRCALPHA)
        panel.fill((10, 10, 30, 210))
        self._screen.blit(panel, (CHART_X, CHART_Y))
        pg.draw.rect(self._screen, (80, 80, 160), (CHART_X, CHART_Y, CHART_W, CHART_H), 1)
        self._screen.blit(self._font_sm.render('Step reward history [R]', True, (180, 180, 255)),
                          (CHART_X + 4, CHART_Y + 3))

        plot_x0 = CHART_X + 26
        plot_y0 = CHART_Y + 16
        plot_w  = CHART_W - 30
        plot_h  = CHART_H - 24

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

        zero_y = to_screen_y(0.0)
        pg.draw.line(self._screen, (120, 120, 120),
                     (plot_x0, zero_y), (plot_x0 + plot_w, zero_y), 1)
        self._screen.blit(self._font_sm.render(f'{y_max:+.0f}', True, (100, 255, 100)),
                          (CHART_X + 1, plot_y0))
        self._screen.blit(self._font_sm.render('0', True, (160, 160, 160)),
                          (CHART_X + 1, zero_y - 6))
        self._screen.blit(self._font_sm.render(f'{y_min:+.0f}', True, (255, 100, 100)),
                          (CHART_X + 1, plot_y0 + plot_h - 10))

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
            cur     = hist[-1]
            cur_col = (80, 255, 80) if cur >= 0 else (255, 80, 80)
            self._screen.blit(
                self._font_sm.render(f'E{i} {cur:+.1f}', True, cur_col),
                (plot_x0 + plot_w + 2, to_screen_y(cur) - 5),
            )

    def _draw_height_chart(self, pg):
        CHART_W, CHART_H = 420, 150
        CHART_X, CHART_Y = _WIN_W - CHART_W - 10, _WIN_H - CHART_H - 10
        TOTAL_HEIGHT     = _MARIO_Y_START - _MARIO_Y_WIN
        SPLIT            = 260

        panel = pg.Surface((CHART_W, CHART_H), pg.SRCALPHA)
        panel.fill((10, 10, 30, 210))
        self._screen.blit(panel, (CHART_X, CHART_Y))
        pg.draw.rect(self._screen, (80, 80, 160), (CHART_X, CHART_Y, CHART_W, CHART_H), 1)
        pg.draw.line(self._screen, (80, 80, 160),
                     (CHART_X + SPLIT, CHART_Y), (CHART_X + SPLIT, CHART_Y + CHART_H), 1)

        hist_x0 = CHART_X + 28
        hist_y0 = CHART_Y + 20
        hist_h  = CHART_H - 30
        hist_w  = SPLIT - 36

        self._screen.blit(
            self._font_sm.render(f'Best height/ep ({len(self._ep_heights)} eps)', True, (180, 180, 255)),
            (CHART_X + 4, CHART_Y + 4))
        self._screen.blit(self._font_sm.render(f'{TOTAL_HEIGHT}px', True, (100, 255, 100)),
                          (CHART_X + 2, hist_y0))
        self._screen.blit(self._font_sm.render('0px', True, (200, 100, 100)),
                          (CHART_X + 2, hist_y0 + hist_h - 12))

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

        live_x0 = CHART_X + SPLIT + 6
        live_y0 = CHART_Y + 20
        live_h  = CHART_H - 30
        live_w  = CHART_W - SPLIT - 12
        n_envs  = self._num_envs
        bar_w   = max(1, (live_w - (n_envs - 1) * 2) // n_envs)

        self._screen.blit(
            self._font_sm.render(f'Live  (0-{TOTAL_HEIGHT}px)', True, (180, 180, 255)),
            (live_x0, CHART_Y + 4))
        pg.draw.line(self._screen, (0, 180, 0),
                     (live_x0, live_y0), (live_x0 + live_w, live_y0), 1)
        pg.draw.line(self._screen, (80, 40, 40),
                     (live_x0, live_y0 + live_h), (live_x0 + live_w, live_y0 + live_h), 1)

        for i in range(n_envs):
            bx     = live_x0 + i * (bar_w + 2)
            height = self._cur_height_px[i] if i < len(self._cur_height_px) else 0
            frac   = max(0.0, min(height / TOTAL_HEIGHT, 1.0))
            bar_h  = max(1, int(frac * live_h))
            color  = _COLORS[i % len(_COLORS)]
            pg.draw.rect(self._screen, color,
                         (bx, live_y0 + live_h - bar_h, bar_w, bar_h))
            if height > 2:
                lbl = self._font_sm.render(f'{int(height)}', True, color)
                self._screen.blit(lbl, (bx, live_y0 + live_h - bar_h - 12))
            lbl = self._font_sm.render(f'E{i}', True, color)
            self._screen.blit(lbl, (bx, live_y0 + live_h + 2))

    def _on_training_end(self) -> None:
        if self._pygame:
            self._pygame.quit()
