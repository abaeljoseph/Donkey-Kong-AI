"""
Interactive detection debugger — no training needed.

Controls:
  H          : next drag sets the HUD exclusion strip
  D          : next drag sets the DK exclusion zone
  Click-drag : draw the selected zone rectangle
  P          : print current zone fractions to console (paste into ghost_viewer.py)
  F          : advance 60 more emulator frames and recapture
  R          : recapture from the start state
  Z          : toggle overlays on/off (turn off to click true game colors)
  Q          : quit
  Click      : print RGB + HSV of that pixel (status bar shows live value under cursor)

Detections shown:
  Red overlay  = excluded zones (HUD strip + DK zone)
  Orange dot B = barrel
  Green dot  M = Mario

Run:  python tools/debug_detection.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import retro
import pygame

INTEGRATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retro_data'))
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

SCALE    = 3
NES_W, NES_H = 256, 240
WIN_W, WIN_H = NES_W * SCALE, NES_H * SCALE
BTN_NONE  = [0] * 9
BTN_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0]   # walk right
BTN_JUMP  = [0, 0, 0, 0, 0, 0, 0, 1, 1]   # jump right (to clear barrels)

# Current zone values — edit these defaults or drag to set new ones
hud_zone  = [0.0, 0.0, 1.0, 0.106]             # [x0, y0, x1, y1] as fractions
dk_zone   = [0.000, 0.078, 0.322, 0.254]        # drag to adjust


def capture_frame(advance=180):
    env = retro.make(
        game='DonkeyKong-Nes',
        state='1Player.GameA',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    obs, _ = env.reset()
    info = {}
    for i in range(advance):
        obs, _, _, _, info = env.step(BTN_RIGHT)
    env.close()
    return obs, info   # (H, W, 3) RGB uint8, info dict with RAM values


def advance_existing(obs_initial, extra_frames):
    """Step the emulator extra_frames more from where it left off — approximated
    by re-launching and advancing the total count."""
    return None   # we'll track total advance in main()


def zone_to_px(zone, w, h):
    x0 = int(zone[0] * w);  y0 = int(zone[1] * h)
    x1 = int(zone[2] * w);  y1 = int(zone[3] * h)
    return x0, y0, x1, y1


def build_excl_mask(frame_h, frame_w):
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    hx0,hy0,hx1,hy1 = zone_to_px(hud_zone, frame_w, frame_h)
    mask[hy0:hy1, hx0:hx1] = 255
    dx0,dy0,dx1,dy1 = zone_to_px(dk_zone, frame_w, frame_h)
    mask[dy0:dy1, dx0:dx1] = 255
    return mask


def find_mario(frame_rgb):
    h, w = frame_rgb.shape[:2]
    excl = build_excl_mask(h, w)
    hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    skin = cv2.inRange(hsv, np.array([  0,  45, 215]), np.array([ 12, 110, 255]))
    blue = cv2.inRange(hsv, np.array([115, 230, 130]), np.array([125, 255, 210]))
    red  = cv2.inRange(hsv, np.array([  0, 200, 180]), np.array([ 12, 255, 240]))
    for m in (skin, blue, red):
        m[excl > 0] = 0
    k = np.ones((14, 14), np.uint8)
    region = cv2.bitwise_and(cv2.dilate(skin, k),
             cv2.bitwise_and(cv2.dilate(blue, k), cv2.dilate(red, k)))
    pixels = cv2.bitwise_and(skin, region)
    ys, xs = np.where(pixels > 0)
    if len(xs) < 3:
        return None
    return (int(np.median(xs)), int(np.median(ys)))


def find_barrels(frame_rgb):
    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    # Barrel orange body: sampled H=15,S=197,V=248
    orange = cv2.inRange(hsv, np.array([10, 160, 220]), np.array([25, 215, 255]))
    # Exclude HUD and DK zone
    hy1 = int(hud_zone[3] * h)
    orange[:hy1, :] = 0
    orange[int(dk_zone[1] * h):int(dk_zone[3] * h),
           int(dk_zone[0] * w):int(dk_zone[2] * w)] = 0
    n, _, stats, centroids = cv2.connectedComponentsWithStats(orange, connectivity=8)
    barrels = []
    for i in range(1, n):
        if 15 < stats[i, cv2.CC_STAT_AREA] < 800:
            barrels.append((int(centroids[i][0]), int(centroids[i][1])))
    return barrels


def find_fires(frame_rgb):
    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    fire = cv2.inRange(hsv, np.array([12, 50, 220]), np.array([25, 130, 255]))
    hy1 = int(hud_zone[3] * h)
    fire[:hy1, :] = 0
    fire[int(dk_zone[1] * h):int(dk_zone[3] * h),
         int(dk_zone[0] * w):int(dk_zone[2] * w)] = 0
    n, _, stats, centroids = cv2.connectedComponentsWithStats(fire, connectivity=8)
    fires = []
    for i in range(1, n):
        if 8 < stats[i, cv2.CC_STAT_AREA] < 400:
            fires.append((int(centroids[i][0]), int(centroids[i][1])))
    return fires


def draw_zone_overlay(screen, zone, color_rgba, label, font):
    x0, y0, x1, y1 = zone_to_px(zone, WIN_W, WIN_H)
    s = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
    s.fill(color_rgba)
    screen.blit(s, (x0, y0))
    pygame.draw.rect(screen, color_rgba[:3], (x0, y0, x1-x0, y1-y0), 2)
    screen.blit(font.render(label, True, color_rgba[:3]), (x0 + 4, y0 + 4))


def draw_scene(screen, frame_rgb, font, drag_mode, drag_rect, show_overlays, mouse_pos, ram_info=None):
    h, w = frame_rgb.shape[:2]
    scale_x = WIN_W / w
    scale_y = WIN_H / h

    # Background
    surf   = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, (WIN_W, WIN_H))
    screen.blit(scaled, (0, 0))

    if show_overlays:
        # Exclusion zones
        draw_zone_overlay(screen, hud_zone, (255, 60, 60, 100), 'HUD (excluded — score/bonus)', font)
        draw_zone_overlay(screen, dk_zone,  (255, 60, 60, 100), 'DK ZONE (excluded)', font)

        # Live drag preview
        if drag_rect:
            rx0, ry0, rx1, ry1 = drag_rect
            c = (0, 160, 255) if drag_mode == 'D' else (255, 200, 0)
            pygame.draw.rect(screen, c, (min(rx0,rx1), min(ry0,ry1),
                                         abs(rx1-rx0), abs(ry1-ry0)), 2)

        # Barrel dots
        barrels = find_barrels(frame_rgb)
        for bx, by in barrels:
            bx_w = int(bx * WIN_W / w);  by_w = int(by * WIN_H / h)
            pygame.draw.circle(screen, (255, 140, 0), (bx_w, by_w), 10)
            pygame.draw.circle(screen, (0, 0, 0), (bx_w, by_w), 10, 2)
            screen.blit(font.render('B', True, (0, 0, 0)), (bx_w - 5, by_w - 7))
            screen.blit(font.render(f'({bx},{by})', True, (255, 200, 0)), (bx_w + 12, by_w - 7))

        # Fire dots
        fires = find_fires(frame_rgb)
        for fx, fy in fires:
            fx_w = int(fx * WIN_W / w);  fy_w = int(fy * WIN_H / h)
            pygame.draw.circle(screen, (255, 30, 30), (fx_w, fy_w), 10)
            pygame.draw.circle(screen, (255, 255, 0), (fx_w, fy_w), 10, 2)
            screen.blit(font.render('F', True, (255, 255, 0)), (fx_w - 5, fy_w - 7))

        # Mario dot
        mario = find_mario(frame_rgb)
        if mario:
            mx_w = int(mario[0] * WIN_W / w);  my_w = int(mario[1] * WIN_H / h)
            pygame.draw.circle(screen, (0, 255, 80), (mx_w, my_w), 12)
            pygame.draw.circle(screen, (255,255,255), (mx_w, my_w), 12, 2)
            screen.blit(font.render('M', True, (0, 0, 0)), (mx_w - 5, my_w - 7))
            screen.blit(font.render(f'({mario[0]},{mario[1]})', True, (0, 255, 80)), (mx_w + 14, my_w - 7))
    else:
        barrels = find_barrels(frame_rgb)
        fires   = find_fires(frame_rgb)
        mario   = find_mario(frame_rgb)

    # Crosshair on cursor pixel (snapped to NES grid)
    if mouse_pos:
        cpx = min(int(mouse_pos[0] / scale_x), w - 1)
        cpy = min(int(mouse_pos[1] / scale_y), h - 1)
        snap_x = int((cpx + 0.5) * scale_x)
        snap_y = int((cpy + 0.5) * scale_y)
        pygame.draw.line(screen, (255,255,0), (snap_x - 6, snap_y), (snap_x + 6, snap_y), 1)
        pygame.draw.line(screen, (255,255,0), (snap_x, snap_y - 6), (snap_x, snap_y + 6), 1)
    else:
        cpx = cpy = 0

    # Live pixel value under cursor
    cursor_str = ''
    if mouse_pos:
        r, g, b = frame_rgb[cpy, cpx]
        hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0,0]
        cursor_str = f'  |  ({cpx},{cpy}) RGB=({r},{g},{b}) HSV=({hsv[0]},{hsv[1]},{hsv[2]})'

    color_y   = mario[1] if mario else '?'
    color_x   = mario[0] if mario else '?'
    mario_str = f'Mario: ({color_x},{color_y})  lives={ram_info.get("lives","?") if ram_info else "?"}'

    overlay_str = '[Z] overlays ON' if show_overlays else '[Z] overlays OFF — click raw pixels'
    mode_str    = f'[{drag_mode}] drag to draw' if drag_mode else 'H=HUD  D=DK  P=print  F=+60fr  R=restart  Q=quit'
    status      = f'Barrels:{len(barrels)}  Fires:{len(fires)}  {mario_str}  {overlay_str}  {mode_str}{cursor_str}'
    screen.blit(font.render(status, True, (0,0,0)),       (7, WIN_H - 19))
    screen.blit(font.render(status, True, (220,220,220)), (6, WIN_H - 20))

    pygame.display.flip()


def print_zones():
    print('\n── Zone values (paste into ghost_viewer.py) ──')
    print(f'_HUD_Y  = {hud_zone[3]:.3f}')
    print(f'_DK_X0  = {dk_zone[0]:.3f}')
    print(f'_DK_X1  = {dk_zone[2]:.3f}')
    print(f'_DK_Y0  = {dk_zone[1]:.3f}')
    print(f'_DK_Y1  = {dk_zone[3]:.3f}\n')


def main():
    global hud_zone, dk_zone

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Detection Debug — H/D=zone  drag=draw  P=print  F=+60fr  R=restart  Q=quit')
    font  = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    total_advance = 180
    print(f'Capturing frame (advance={total_advance} frames)...')
    frame, ram_info = capture_frame(total_advance)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    print(f'Frame: {frame.shape[1]}×{frame.shape[0]}  (scale_x={WIN_W/frame.shape[1]:.3f} scale_y={WIN_H/frame.shape[0]:.3f})')
    print(f'RAM values: mario_y={ram_info.get("mario_y","?")}  mario_x={ram_info.get("mario_x","?")}  lives={ram_info.get("lives","?")}  gameover={ram_info.get("gameover","?")}')

    drag_mode     = None    # 'H' or 'D'
    drag_start    = None
    drag_rect     = None
    dragging      = False
    show_overlays = True
    mouse_pos     = None

    running = True
    while running:
        draw_scene(screen, frame, font, drag_mode, drag_rect, show_overlays, mouse_pos, ram_info)
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                if dragging:
                    x0, y0 = drag_start
                    x1, y1 = event.pos
                    drag_rect = (x0, y0, x1, y1)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_z:
                    show_overlays = not show_overlays
                elif event.key == pygame.K_h:
                    drag_mode = 'H'
                    print('Mode: draw HUD zone (drag a strip across the top)')
                elif event.key == pygame.K_d:
                    drag_mode = 'D'
                    print('Mode: draw DK zone (drag a box around DK)')
                elif event.key == pygame.K_p:
                    print_zones()
                elif event.key == pygame.K_f:
                    total_advance += 60
                    frame, ram_info = capture_frame(total_advance)
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    mario = find_mario(frame)
                    color_y = mario[1] if mario else '?'
                    color_x = mario[0] if mario else '?'
                    print(f'frame={total_advance:5d}  RAM=({ram_info.get("mario_x","?"):>3},{ram_info.get("mario_y","?"):>3})  COLOR=({color_x},{color_y})  lives={ram_info.get("lives","?")}')
                elif event.key == pygame.K_r:
                    total_advance = 180
                    print('Recapturing from start...')
                    frame, ram_info = capture_frame(total_advance)
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    print(f'RAM values: mario_y={ram_info.get("mario_y","?")}  mario_x={ram_info.get("mario_x","?")}  lives={ram_info.get("lives","?")}')

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if drag_mode:
                    drag_start = event.pos
                    dragging   = True
                else:
                    mx, my = event.pos
                    fh, fw = frame.shape[:2]
                    px = min(int(mx * fw / WIN_W), fw - 1)
                    py = min(int(my * fh / WIN_H), fh - 1)
                    r, g, b = frame[py, px]
                    hv, sv, vv = frame_hsv[py, px]
                    print(f'  pixel ({px:3d},{py:3d})  RGB=({r:3d},{g:3d},{b:3d})  HSV=({hv:3d},{sv:3d},{vv:3d})')

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                dragging = False
                x0, y0 = drag_start
                x1, y1 = event.pos
                # Normalise so x0<x1, y0<y1
                fx0 = min(x0, x1) / WIN_W;  fx1 = max(x0, x1) / WIN_W
                fy0 = min(y0, y1) / WIN_H;  fy1 = max(y0, y1) / WIN_H
                if drag_mode == 'H':
                    hud_zone = [0.0, 0.0, 1.0, fy1]   # HUD always spans full width
                    print(f'HUD zone set: y up to {fy1:.3f}')
                elif drag_mode == 'D':
                    dk_zone = [fx0, fy0, fx1, fy1]
                    print(f'DK zone set: x={fx0:.3f}-{fx1:.3f}  y={fy0:.3f}-{fy1:.3f}')
                drag_mode  = None
                drag_rect  = None
                drag_start = None

    pygame.quit()
    print_zones()


if __name__ == '__main__':
    main()
