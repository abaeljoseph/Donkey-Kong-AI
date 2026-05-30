"""
Interactive broken-ladder reward tester + bonus-timer analyser.

Play normally — the tool silently records RAM every step.
When you quit (Q / Escape), it analyses the recording and reports
which RAM address is the bonus/time-remaining counter.

Controls (pygame window must be focused):
  Arrow keys   : move / climb ladders
  Space        : jump
  Q / Escape   : quit
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pygame
from environment.donkey_kong_env import DonkeyKongEnv, BROKEN_LADDER_ZONES

INTEGRATION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'retro_data'))

ACTION_NAMES = ['NOOP','LEFT','RIGHT','UP','DOWN','JUMP','JUMP+L','JUMP+R']

SCALE    = 3
NES_W, NES_H = 256, 240

KNOWN_ADDRS = {85: 'lives', 78: 'gameover', 36: 'score',
               67: 'mario_x', 46: 'mario_y', 83: 'stage', 84: 'level'}


def make_display_frame(obs_rgb, mario_y, mario_x, broken_zones):
    frame = obs_rgb.copy()
    h, w  = frame.shape[:2]
    for z in broken_zones:
        x0 = int(z[0] * w / 256); y0 = int(z[1] * h / 224)
        x1 = int(z[2] * w / 256); y1 = int(z[3] * h / 224)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 60, 60), 1)
    return frame


def analyse(ram_log, lives_log):
    """
    Analyse recorded RAM to find the bonus/time-remaining counter.

    A bonus timer is monotonically non-increasing within each life
    (never goes up mid-life), may reset to a higher value on respawn,
    and has a meaningful total drop across the session.
    """
    if len(ram_log) < 10:
        print('[analyse] Not enough data recorded.')
        return

    snaps = np.stack(ram_log)          # (N, ram_size)
    lives = np.array(lives_log)        # (N,)
    N     = snaps.shape[0]
    RAM   = min(snaps.shape[1], 2048)  # true NES RAM only

    # Find life boundaries (indices where lives dropped)
    death_idxs = list(np.where(np.diff(lives) < 0)[0] + 1)
    boundaries = [0] + death_idxs + [N]
    segments   = [(boundaries[i], boundaries[i+1])
                  for i in range(len(boundaries) - 1)
                  if boundaries[i+1] - boundaries[i] > 5]

    print(f'\n[analyse] {N} snapshots,  {len(segments)} life segment(s),  '
          f'deaths at steps {death_idxs}')

    def score_addr(vals_per_seg):
        """Return (score, seg_data) or None if address is not timer-like."""
        seg_data = []
        for vals in vals_per_seg:
            if len(vals) < 8:
                continue
            diffs = np.diff(vals)
            n_inc = int(np.sum(diffs > 0))
            n_dec = int(np.sum(diffs < 0))
            total_drop = int(vals[0]) - int(vals.min())
            # Allow at most 2 upward blips per life (noise tolerance)
            if n_inc > 2:
                return None
            # Must decrement at least 3 separate times
            if n_dec < 3:
                continue
            # Must start positive
            if vals[0] == 0 or total_drop < 3:
                continue
            seg_data.append((int(vals[0]), int(vals[-1]), total_drop, n_dec, n_inc))
        if not seg_data:
            return None
        # Score: mean_drop penalised by upward noise
        total_inc = sum(d[4] for d in seg_data)
        mean_drop  = np.mean([d[2] for d in seg_data])
        score = mean_drop / (1 + total_inc)
        return score, seg_data

    scores = {}   # addr → (score, seg_data, label)

    # --- single-byte scan (all RAM) ---
    for addr in range(min(snaps.shape[1], 10240)):
        if addr in KNOWN_ADDRS:
            continue
        segs = [snaps[s:e, addr].astype(int) for s, e in segments]
        result = score_addr(segs)
        if result:
            sc, sd = result
            scores[f'{addr}'] = (sc, sd, f'addr {addr} (0x{addr:03X})'  )

    # --- 2-byte BCD scan (first 2048 bytes) ---
    def bcd2(hi, lo):
        hh, hl = hi >> 4, hi & 0xF
        lh, ll = lo >> 4, lo & 0xF
        if hh > 9 or hl > 9 or lh > 9 or ll > 9:
            return None
        return hh * 1000 + hl * 100 + lh * 10 + ll

    for addr in range(min(snaps.shape[1] - 1, 2047)):
        hi_segs = [snaps[s:e, addr    ].astype(int) for s, e in segments]
        lo_segs = [snaps[s:e, addr + 1].astype(int) for s, e in segments]
        bcd_segs = []
        for hi_v, lo_v in zip(hi_segs, lo_segs):
            bcd_v = np.array([bcd2(h, l) for h, l in zip(hi_v, lo_v)])
            if None in bcd_v:
                bcd_segs = []
                break
            bcd_segs.append(bcd_v.astype(float))
        if not bcd_segs:
            continue
        result = score_addr(bcd_segs)
        if result:
            sc, sd = result
            key = f'bcd_{addr}'
            if sc > scores.get(key, (0,))[0]:
                scores[key] = (sc, sd, f'BCD2 addr {addr}+{addr+1} (0x{addr:03X})')

    if not scores:
        print('[analyse] No clear bonus-timer candidate found.')
        print('          Try playing longer (at least one full life without')
        print('          the bonus timer resetting more than once per life).')
        return

    ranked = sorted(scores.values(), key=lambda v: -v[0])

    print('\n=== Bonus timer candidates (best first) ===')
    print(f'  {"Label":<30}  {"Score":>7}  Life segments (start→end, drop, ndec, ninc)')
    for sc, segs, label in ranked[:15]:
        seg_str = '  |  '.join(
            f'{s}→{e} d={d} ↓{n} ↑{u}' for s, e, d, n, u in segs)
        print(f'  {label:<30}  {sc:7.1f}  {seg_str}')

    best = ranked[0]
    sc, segs, label = best
    print(f'\n>>> Most likely bonus timer: {label}')
    print(f'    Starting values per life: {[s[0] for s in segs]}')
    # Extract address for data.json suggestion
    import re
    m = re.search(r'addr (\d+)', label)
    if m:
        a = int(m.group(1))
        typ = '"|u1"' if 'BCD' not in label else '">d2"'
        print(f'    Add to data.json:')
        print(f'    "bonus": {{ "address": {a}, "type": {typ} }}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', type=int, default=None,
                        help='RAM address to display live on screen (e.g. --watch 68)')
    args = parser.parse_args()
    watch_addr = args.watch

    pygame.init()
    win_w, win_h = NES_W * SCALE, NES_H * SCALE
    PANEL_W = 300
    screen = pygame.display.set_mode((win_w + PANEL_W, win_h))
    pygame.display.set_caption('Debug — Arrow keys / Space / Q  (RAM recorded, analysed on quit)')
    font       = pygame.font.SysFont('Consolas', 13)
    font_bold  = pygame.font.SysFont('Consolas', 13, bold=True)
    clock = pygame.time.Clock()

    env = DonkeyKongEnv(
        render=False,
        custom_integration_path=INTEGRATION_PATH,
        provide_frame=False,
        provide_detect=False,
        use_checkpoints=False,
    )
    obs = env.reset()

    cum_reward = 0.0
    step_count = 0
    ram_log    = []   # list of np.uint8 arrays
    lives_log  = []   # parallel list of int lives values

    print('\n=== Broken Ladder + Bonus Timer Finder ===')
    print('Play normally. RAM is recorded silently.')
    print('Quit with Q/Escape — analysis prints automatically.\n')
    print(f'{"Step":>5}  {"Action":<10}  {"dy":>5}  {"Reward":>8}  {"Cumul":>9}  Note')

    running = True
    while running:
        action_idx = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        keys = pygame.key.get_pressed()
        jump = keys[pygame.K_SPACE]
        if jump and keys[pygame.K_LEFT]:
            action_idx = 6
        elif jump and keys[pygame.K_RIGHT]:
            action_idx = 7
        elif jump:
            action_idx = 5
        elif keys[pygame.K_UP]:
            action_idx = 3
        elif keys[pygame.K_DOWN]:
            action_idx = 4
        elif keys[pygame.K_LEFT]:
            action_idx = 1
        elif keys[pygame.K_RIGHT]:
            action_idx = 2

        obs, reward, done, info = env.step(action_idx)
        cum_reward += reward
        step_count += 1

        mario_y = info.get('_mario_y', 0)
        mario_x = getattr(env, '_prev_mario_x', 128)
        dy      = getattr(env, '_prev_mario_y', mario_y) - mario_y

        # Record RAM every step
        try:
            ram = np.frombuffer(env.env.get_ram(), dtype=np.uint8).copy()
            ram_log.append(ram)
            lives_log.append(int(ram[85]))
        except Exception:
            pass

        mario_cx = mario_x + 8
        mario_cy = mario_y + 8
        in_broken = any(
            z[0] <= mario_cx <= z[2] and z[1] <= mario_cy <= z[3]
            for z in env._broken_zones
        )
        note = '<<< IN BROKEN ZONE' if in_broken else ''
        if done:
            note += '  [EPISODE END]'

        if action_idx != 0 or step_count % 10 == 0:
            print(f'{step_count:5d}  {ACTION_NAMES[action_idx]:<10}  {dy:+5.1f}  '
                  f'{reward:+8.2f}  {cum_reward:+9.2f}  {note}')

        # ── Game frame ────────────────────────────────────────────────
        try:
            raw = env.env.em.get_screen()
        except Exception:
            raw = np.zeros((NES_H, NES_W, 3), dtype=np.uint8)

        display = make_display_frame(raw, mario_y, mario_x, env._broken_zones)
        surf = pygame.surfarray.make_surface(display.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (win_w, win_h)), (0, 0))

        # ── OAM sprite constants ───────────────────────────────────────
        OAM_BASE  = 512
        NES_VIS_H = 224

        def xhair(px, py, col, sz=8, rad=5):
            pygame.draw.circle(screen, col, (px, py), rad, 2)
            pygame.draw.line(screen, col, (px - sz, py), (px + sz, py), 1)
            pygame.draw.line(screen, col, (px, py - sz), (px, py + sz), 1)

        def group_center(slots):
            xs, ys = [], []
            for s in slots:
                sy = int(ram_log[-1][OAM_BASE + s * 4])
                sx = int(ram_log[-1][OAM_BASE + s * 4 + 3])
                if sy < 0xEF and sx > 0:
                    xs.append(sx); ys.append(sy)
            if not xs:
                return None
            return min(xs), min(ys)

        def to_px(obj_x, obj_y, x_off=0, y_off=2):
            px = int((obj_x + x_off) * win_w / 240)
            py = int((obj_y + y_off) * win_h / NES_VIS_H)
            return px, py

        n_barrels = 0
        n_fire    = 0
        barrel_positions = []
        fire_positions   = []

        if len(ram_log) > 0:
            # Mario — pink crosshair + green dot
            mc = group_center([0, 1, 2, 3])
            if mc:
                xhair(*to_px(*mc), (255, 0, 200), sz=10, rad=6)
                pygame.draw.circle(screen, (0, 255, 80), to_px(*mc), 4)

            # Fire (slots 4-7, 8-11) — orange
            for slots in ([4, 5, 6, 7], [8, 9, 10, 11]):
                fc = group_center(slots)
                if fc:
                    xhair(*to_px(*fc, y_off=3), (255, 140, 0))
                    fire_positions.append(fc)
                    n_fire += 1

            # Barrels (slots 12-51, groups of 4) — red
            for g in range(10):
                base = 12 + g * 4
                bc = group_center([base, base+1, base+2, base+3])
                if bc:
                    xhair(*to_px(*bc, y_off=1), (255, 80, 80))
                    barrel_positions.append(bc)
                    n_barrels += 1

            # Hammers (slots 52-53, 54-55) — yellow
            for slots in ([52, 53], [54, 55]):
                hc = group_center(slots)
                if hc:
                    xhair(*to_px(*hc, x_off=-4, y_off=-2), (255, 255, 0), sz=6, rad=4)

        # ── Side panel ────────────────────────────────────────────────
        pygame.draw.rect(screen, (18, 18, 28), (win_w, 0, PANEL_W, win_h))
        pygame.draw.line(screen, (60, 60, 80), (win_w, 0), (win_w, win_h), 2)

        py = 10
        def pline(label, value, col=(190, 210, 255), bold=False):
            nonlocal py
            f = font_bold if bold else font
            txt = f'{label:<13}{value}'
            screen.blit(f.render(txt, True, col), (win_w + 10, py))
            py += 17

        def psep(title=''):
            nonlocal py
            py += 4
            if title:
                screen.blit(font_bold.render(title, True, (120, 140, 180)), (win_w + 10, py))
                py += 16
            pygame.draw.line(screen, (50, 55, 75), (win_w + 6, py), (win_w + PANEL_W - 6, py), 1)
            py += 6

        psep('GAME STATE')
        pline('Score',   info.get('score',   '?'))
        pline('Lives',   info.get('lives',    '?'))
        pline('Stage',   info.get('stage',    '?'))
        _lv = info.get('level', None)
        pline('Level',   (_lv + 1) if isinstance(_lv, int) else '?')
        pline('Bonus',   info.get('bonus',    '?'), col=(255, 220, 50))
        pline('Gameover',info.get('gameover', '?'))

        psep('MARIO')
        pline('X  (OAM)',  mario_x)
        pline('Y  (OAM→NES)', mario_y)
        in_zone_col = (255, 80, 80) if in_broken else (100, 200, 100)
        pline('Broken zone', 'YES' if in_broken else 'no', col=in_zone_col, bold=in_broken)

        psep('HAZARDS')
        pline('Barrels',  n_barrels, col=(255, 100, 100))
        for i, (bx, by) in enumerate(barrel_positions):
            pline(f'  [{i}]', f'x={bx} y={by}', col=(200, 80, 80))
        pline('Fire',     n_fire,    col=(255, 160, 60))
        for i, (fx, fy) in enumerate(fire_positions):
            pline(f'  [{i}]', f'x={fx} y={fy}', col=(200, 120, 40))

        psep('EPISODE')
        pline('Step',     step_count)
        pline('Reward',   f'{reward:+.3f}')
        pline('Cumul',    f'{cum_reward:+.2f}')
        pline('Action',   ACTION_NAMES[action_idx])

        psep('BROKEN ZONES')
        for i, z in enumerate(env._broken_zones):
            pline(f'  z{i}', f'({z[0]},{z[1]})-({z[2]},{z[3]})', col=(160, 160, 160))

        pygame.display.flip()
        clock.tick(15)

        if done:
            print(f'  → Episode ended. Resetting...\n')
            obs        = env.reset()
            cum_reward = 0.0
            step_count = 0

    env.close()
    pygame.quit()

    # Post-session analysis
    analyse(ram_log, lives_log)


if __name__ == '__main__':
    main()
