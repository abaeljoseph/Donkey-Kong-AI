"""
Donkey Kong NES — interactive debug tool.

Shows live sprite positions, ladder/rivet/broken-zone detection overlays,
OAM slot numbers, and a RAM watch mode that prints any address which was
stable for ≥ 2 s but just changed (good for finding rivet/state addresses).
Also records RAM silently and analyses it for bonus-timer candidates on exit.

Controls (pygame window must be focused):
  Arrow keys     : move / climb
  Space          : jump
  1–4            : jump to stage (stage must have been visited once to unlock 2-4)
  P              : pause / unpause
  D              : delete + force-rescan ladder cache for current stage
  N              : toggle raw OAM slot-number overlay
  W              : toggle RAM watch mode (prints stable-then-changed addresses)
  Q / Escape     : quit + run bonus-timer analysis
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pygame
import json
from environment.donkey_kong_env import (DonkeyKongEnv, BROKEN_LADDER_ZONES,
                                         RIVET_POSITIONS_PATH, _ladder_cache_path)

INTEGRATION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'retro_data'))

STAGE_START_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'stage_starts'))
os.makedirs(STAGE_START_DIR, exist_ok=True)

def _stage_state_path(stage: int) -> str:
    return os.path.join(STAGE_START_DIR, f'stage_{stage}_start.bin')

SCALE    = 3
NES_W, NES_H = 256, 240

KNOWN_ADDRS = {85: 'lives', 78: 'gameover', 36: 'score',
               67: 'mario_x', 46: 'mario_y', 83: 'stage', 84: 'level'}

# RAM watch: print any work-RAM address that was stable for ≥ this many steps
# then suddenly changed.  120 ≈ 2 s at 60 fps (single-step tool runs at 60 fps).
RAM_WATCH_STABLE = 120
WORK_RAM         = 2048


def make_display_frame(obs_rgb, mario_y, mario_x, broken_zones, stage=1, ladder_rects=None):
    frame = obs_rgb.copy()
    h, w  = frame.shape[:2]
    for (x0, y0, x1, y1) in (ladder_rects or []):
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 220), 1)
    if stage == 1:
        for z in broken_zones:
            bx0 = int(z[0] * w / 256); by0 = int(z[1] * h / 224)
            bx1 = int(z[2] * w / 256); by1 = int(z[3] * h / 224)
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), (255, 60, 60), 1)
    return frame


def analyse(ram_log, lives_log):
    """Analyse recorded RAM to find the bonus/time-remaining counter."""
    if len(ram_log) < 10:
        print('[analyse] Not enough data recorded.')
        return

    snaps = np.stack(ram_log)
    lives = np.array(lives_log)
    N     = snaps.shape[0]
    RAM   = min(snaps.shape[1], 2048)

    death_idxs = list(np.where(np.diff(lives) < 0)[0] + 1)
    boundaries = [0] + death_idxs + [N]
    segments   = [(boundaries[i], boundaries[i+1])
                  for i in range(len(boundaries) - 1)
                  if boundaries[i+1] - boundaries[i] > 5]

    print(f'\n[analyse] {N} snapshots,  {len(segments)} life segment(s),  '
          f'deaths at steps {death_idxs}')

    def score_addr(vals_per_seg):
        seg_data = []
        for vals in vals_per_seg:
            if len(vals) < 8:
                continue
            diffs = np.diff(vals)
            n_inc = int(np.sum(diffs > 0))
            n_dec = int(np.sum(diffs < 0))
            total_drop = int(vals[0]) - int(vals.min())
            if n_inc > 2:
                return None
            if n_dec < 3:
                continue
            if vals[0] == 0 or total_drop < 3:
                continue
            seg_data.append((int(vals[0]), int(vals[-1]), total_drop, n_dec, n_inc))
        if not seg_data:
            return None
        total_inc = sum(d[4] for d in seg_data)
        mean_drop = np.mean([d[2] for d in seg_data])
        return mean_drop / (1 + total_inc), seg_data

    scores = {}

    for addr in range(min(snaps.shape[1], 10240)):
        if addr in KNOWN_ADDRS:
            continue
        segs   = [snaps[s:e, addr].astype(int) for s, e in segments]
        result = score_addr(segs)
        if result:
            sc, sd = result
            scores[f'{addr}'] = (sc, sd, f'addr {addr} (0x{addr:03X})')

    def bcd2(hi, lo):
        hh, hl = hi >> 4, hi & 0xF
        lh, ll = lo >> 4, lo & 0xF
        if hh > 9 or hl > 9 or lh > 9 or ll > 9:
            return None
        return hh * 1000 + hl * 100 + lh * 10 + ll

    for addr in range(min(snaps.shape[1] - 1, 2047)):
        hi_segs  = [snaps[s:e, addr    ].astype(int) for s, e in segments]
        lo_segs  = [snaps[s:e, addr + 1].astype(int) for s, e in segments]
        bcd_segs = []
        for hi_v, lo_v in zip(hi_segs, lo_segs):
            bcd_v = np.array([bcd2(h, l) for h, l in zip(hi_v, lo_v)])
            if None in bcd_v:
                bcd_segs = []; break
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
    import re
    m = re.search(r'addr (\d+)', label)
    if m:
        a   = int(m.group(1))
        typ = '"|u1"' if 'BCD' not in label else '">d2"'
        print(f'    Add to data.json:')
        print(f'    "bonus": {{ "address": {a}, "type": {typ} }}')


def main():
    parser = argparse.ArgumentParser(description='Donkey Kong NES debug tool')
    parser.add_argument('--watch', type=int, default=None,
                        help='RAM address to display live on screen (e.g. --watch 68)')
    parser.add_argument('--no-auto-reset', action='store_true',
                        help='Do not auto-reset on episode end')
    args       = parser.parse_args()
    watch_addr = args.watch
    auto_reset = not args.no_auto_reset

    pygame.init()
    win_w, win_h = NES_W * SCALE, NES_H * SCALE
    PANEL_W = 300
    screen = pygame.display.set_mode((win_w + PANEL_W * 2, win_h))
    pygame.display.set_caption(
        'DK Debug Tool — Arrows/Space  W=RAM watch  P=pause  N=slots  D=rescan  1-4=stage  Q=quit')
    font      = pygame.font.SysFont('Consolas', 13)
    font_bold = pygame.font.SysFont('Consolas', 13, bold=True)
    clock     = pygame.time.Clock()

    env = DonkeyKongEnv(
        render=False,
        custom_integration_path=INTEGRATION_PATH,
        provide_frame=False,
        provide_detect=False,
        use_checkpoints=False,
    )
    obs = env.reset()

    cum_reward     = 0.0
    step_count     = 0
    ram_log        = []
    lives_log      = []
    ram            = np.frombuffer(env.env.get_ram(), dtype=np.uint8).copy()
    obs_raw        = np.zeros((224, 240, 3), dtype=np.uint8)
    prev_stage     = int(ram[83])
    cur_stage      = prev_stage
    rescan_pending = False
    goto_stage     = 0
    paused         = False
    show_slots     = False

    # ── RAM watch state ───────────────────────────────────────────────────────
    ram_watch        = False                              # toggled with W
    _rw_prev         = None                               # previous RAM snapshot
    _rw_last_changed = np.zeros(WORK_RAM, dtype=np.int32) # step# of last change per addr
    _rw_frame        = 0                                  # monotonic step counter

    print('\n=== DK Debug Tool ===')
    print('  Arrows/Space  : play')
    print('  1-4           : jump to stage')
    print('  W             : toggle RAM watch (prints stable→changed addresses)')
    print('  P             : pause   N : OAM slots   D : rescan ladders   Q : quit')
    print()

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n:
                    show_slots = not show_slots
                elif event.key == pygame.K_w:
                    ram_watch = not ram_watch
                    if ram_watch:
                        # Reset change tracker so we only report changes from NOW
                        _rw_prev         = ram[:WORK_RAM].copy()
                        _rw_last_changed[:] = _rw_frame
                        print('[RAM Watch] ON — will print stable (≥2 s) addresses that change.')
                        print('            Tip: walk over a rivet / collect an item / change state,')
                        print('            then read which addresses flip in the console.')
                    else:
                        print('[RAM Watch] OFF')
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    goto_stage = event.key - pygame.K_0
                elif event.key == pygame.K_d:
                    _cp = _ladder_cache_path(cur_stage)
                    if os.path.exists(_cp):
                        os.remove(_cp)
                        print(f'[Cache] Deleted stage {cur_stage} ladder cache.')
                    env._rescan_stage_maps(obs_raw)
                    print(f'[Cache] Forced rescan for stage {cur_stage}.')

        # ── Stage jump ────────────────────────────────────────────────────────
        if goto_stage > 0:
            target     = goto_stage
            goto_stage = 0
            path       = _stage_state_path(target)
            if target == 1:
                obs        = env.reset()
                ram        = np.frombuffer(env.env.get_ram(), dtype=np.uint8).copy()
                cur_stage  = int(ram[83])
                prev_stage = cur_stage
                rescan_pending = False
                step_count = 0;  ram_log.clear();  lives_log.clear()
                print(f'[Jump] Stage 1 — reset to default start.')
            elif os.path.exists(path):
                with open(path, 'rb') as f:
                    state_data = f.read()
                env.env.em.set_state(state_data)
                _noop = np.zeros(9, dtype=np.uint8)
                for _ in range(4):
                    result = env.env.step(_noop)
                obs_raw    = result[0]
                ram        = np.frombuffer(env.env.get_ram(), dtype=np.uint8).copy()
                env._current_stage = target
                env._rescan_stage_maps(obs_raw)
                cur_stage  = target
                prev_stage = target
                rescan_pending = False
                step_count = 0;  ram_log.clear();  lives_log.clear()
                print(f'[Jump] Stage {target} loaded.')
            else:
                print(f'[Jump] No saved state for stage {target} — '
                      f'play to that stage first.')

        keys = pygame.key.get_pressed()
        jump = keys[pygame.K_SPACE]
        btns = np.array([
            0, 0,
            1 if keys[pygame.K_BACKSPACE] else 0,
            1 if keys[pygame.K_s]         else 0,
            1 if keys[pygame.K_UP]        else 0,
            1 if keys[pygame.K_DOWN]      else 0,
            1 if keys[pygame.K_LEFT]      else 0,
            1 if keys[pygame.K_RIGHT]     else 0,
            1 if jump                     else 0,
        ], dtype=np.uint8)

        if not paused:
            result   = env.env.step(btns)
            obs_raw  = result[0] if isinstance(result, tuple) else result
            ram      = np.frombuffer(env.env.get_ram(), dtype=np.uint8).copy()
            step_count += 1

        gameover = int(ram[78])
        done     = (gameover == 1 and int(ram[85]) == 0)

        cur_stage = int(ram[83])
        if cur_stage != prev_stage and cur_stage != 0:
            prev_stage         = cur_stage
            env._current_stage = cur_stage
            rescan_pending     = True

        if rescan_pending:
            _oam_y = int(ram[512])
            if 150 <= _oam_y < 0xEF:
                rescan_pending = False
                env._rescan_stage_maps(obs_raw)
                if cur_stage >= 2 and int(ram[84]) == 0:
                    _sp = _stage_state_path(cur_stage)
                    with open(_sp, 'wb') as _f:
                        _f.write(env.env.em.get_state())
                    print(f'[Stage {cur_stage}] Start state saved → press {cur_stage} to jump here.')

        mario_y = int(ram[512])
        mario_x = int(ram[515])

        # ── Record RAM ────────────────────────────────────────────────────────
        ram_log.append(ram)
        lives_log.append(int(ram[85]))

        # Rivet death reset is handled automatically in step() via the 0xC1-C8 bytes
        # reverting to 0 — no explicit life-count reset needed here.

        # ── RAM watch ─────────────────────────────────────────────────────────
        if ram_watch and _rw_prev is not None and not paused:
            cur_w = ram[:WORK_RAM]
            changed = np.where(cur_w != _rw_prev)[0]
            for addr in changed:
                stable = _rw_frame - _rw_last_changed[addr]
                if stable >= RAM_WATCH_STABLE:
                    known = KNOWN_ADDRS.get(addr, '')
                    tag   = f'  [{known}]' if known else ''
                    print(f'[RAM Watch] {addr:4d} (0x{addr:03X}):  '
                          f'{_rw_prev[addr]:3d} → {cur_w[addr]:3d}'
                          f'  stable {stable} steps ({stable/60:.1f}s){tag}')
                _rw_last_changed[addr] = _rw_frame
            _rw_prev = cur_w.copy()
        elif ram_watch and _rw_prev is None and not paused:
            _rw_prev = ram[:WORK_RAM].copy()

        if not paused:
            _rw_frame += 1

        # ── Draw game frame ───────────────────────────────────────────────────
        mario_cx = mario_x + 8;  mario_cy = mario_y + 8
        in_broken = any(
            z[0] <= mario_cx <= z[2] and z[1] <= mario_cy <= z[3]
            for z in env._broken_zones
        )

        try:
            raw = env.env.em.get_screen()
        except Exception:
            raw = np.zeros((NES_H, NES_W, 3), dtype=np.uint8)

        display = make_display_frame(raw, mario_y, mario_x, env._broken_zones,
                                     stage=cur_stage, ladder_rects=env._ladder_rects)
        surf = pygame.surfarray.make_surface(display.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (win_w, win_h)), (0, 0))

        # ── OAM helpers ───────────────────────────────────────────────────────
        OAM_BASE  = 512
        NES_VIS_H = 224

        def bcd(data, n_bytes):
            val = 0
            for b in data[:n_bytes]:
                val = val * 100 + (int(b) >> 4) * 10 + (int(b) & 0xF)
            return val

        def xhair(px, py, col, label='', sz=8, rad=5):
            pygame.draw.circle(screen, col, (px, py), rad, 2)
            pygame.draw.line(screen, col, (px - sz, py), (px + sz, py), 1)
            pygame.draw.line(screen, col, (px, py - sz), (px, py + sz), 1)
            if label:
                screen.blit(font.render(label, True, col), (px + rad + 2, py - 7))

        def group_center(slots):
            xs, ys = [], []
            for s in slots:
                sy = int(ram[OAM_BASE + s * 4])
                sx = int(ram[OAM_BASE + s * 4 + 3])
                if sy < 0xEF and sx > 0:
                    xs.append(sx); ys.append(sy)
            return (min(xs), min(ys)) if xs else None

        def to_px(obj_x, obj_y, x_off=0, y_off=2):
            return (int((obj_x + x_off) * win_w / 240),
                    int((obj_y + y_off) * win_h / NES_VIS_H))

        STAGE_GROUP_NAMES = {
            1: { 'fire': ['FIREBALL0', 'FIREBALL1'],
                 'groups': {**{12+g*4: f'BARREL{g}' for g in range(10)},
                             48: 'SCORE', 56: 'FLAME',
                             52: 'HAMMER0', 54: 'HAMMER1'} },
            2: { 'fire': ['FIREBALL0', 'FIREBALL1'],
                 'groups': {**{12+g*4: f'PIE{g}' for g in range(7)},
                             40: 'PLATFORM0', 44: 'PLATFORM1',
                             48: 'SCORE', 56: 'FLAME',
                             52: 'HAMMER0',  54: 'HAMMER1'} },
            3: { 'fire': ['FIREBALL0', 'FIREBALL1'],
                 'groups': {12: 'ELEVATOR0', 14: 'ELEVATOR1', 16: 'ELEVATOR2',
                             18: 'ELEVATOR3', 20: 'ELEVATOR4', 22: 'ELEVATOR5',
                             24: 'SPRING0',  26: 'SPRING1',   28: 'SPRING2',  30: 'SPRING3',
                             48: 'SCORE', 52: 'HAMMER0', 54: 'HAMMER1'} },
            4: { 'fire': ['FIREBALL0', 'FIREBALL1'],
                 'groups': {12: 'FIREBALL2', 16: 'FIREBALL3',
                             48: 'SCORE', 52: 'HAMMER0', 54: 'HAMMER1'} },
        }
        stage_names    = STAGE_GROUP_NAMES.get(cur_stage, STAGE_GROUP_NAMES[1])
        detected_sprites = []

        # Mario
        mc = group_center([0, 1, 2, 3])
        if mc:
            xhair(*to_px(*mc), (255, 0, 200), label='MARIO', sz=10, rad=6)
            pygame.draw.circle(screen, (0, 255, 80), to_px(*mc), 4)

        # Fire slots 4-11
        fire_labels = stage_names['fire']
        for i, slots in enumerate(([4, 5, 6, 7], [8, 9, 10, 11])):
            fc = group_center(slots)
            if fc:
                lbl = fire_labels[i] if i < len(fire_labels) else f'FIRE{i}'
                xhair(*to_px(*fc, y_off=3), (255, 140, 0), label=lbl)
                detected_sprites.append((lbl, fc[0], fc[1]))

        # Slots 12-63
        # Base slots 56 (flame) and 60 (Pauline) only meaningful in stages 1-2.
        _suppress_bases = {56, 60} if cur_stage >= 3 else {60}
        slot_groups  = stage_names['groups']
        drawn_groups = set()
        for slot in range(12, 64):
            sy = int(ram[OAM_BASE + slot * 4])
            sx = int(ram[OAM_BASE + slot * 4 + 3])
            if sy >= 0xF0 or sx == 0:
                continue
            if (cur_stage == 3 and 12 <= slot <= 31) or (52 <= slot <= 55):
                base_slot = (slot // 2) * 2
            else:
                base_slot = (slot // 4) * 4
            if base_slot in _suppress_bases:
                continue
            lbl = slot_groups.get(base_slot, f'S{slot}')
            col = (255, 80, 80)   if any(x in lbl for x in ('BARREL', 'PIE', 'FIREBALL2', 'FIREBALL3')) else \
                  (0, 200, 255)   if 'ELEVATOR' in lbl else \
                  (255, 200, 0)   if any(x in lbl for x in ('SPRING', 'HAMMER')) else \
                  (180, 255, 100) if 'PLATFORM' in lbl else \
                  (255, 180, 255) if 'PAULINE' in lbl else \
                  (255, 100, 200) if 'HEART' in lbl else \
                  (255, 120, 0)   if 'FLAME' in lbl else \
                  (200, 200, 200)
            if base_slot not in drawn_groups:
                xhair(*to_px(sx, sy, y_off=1), col, label=lbl, sz=6, rad=4)
                drawn_groups.add(base_slot)
                detected_sprites.append((lbl, sx, sy))

        # Oil barrel flame spawner
        if len(ram) > 359:
            ob_y = int(ram[356]);  ob_x = int(ram[359])
            if ob_y < 0xEF and ob_x > 0:
                xhair(*to_px(ob_x, ob_y, y_off=1), (255, 80, 0), label='OIL BARREL', sz=8, rad=5)
                detected_sprites.append(('OIL BARREL', ob_x, ob_y))

        # Stage 4 rivets — direct RAM byte → visual index mapping.
        # RAM byte i (0xC1+i) == 1 means _rivet_positions[i] was collected.
        # (the debug tool calls env.env.step() directly so step() never runs here)
        if cur_stage == 4 and env._rivet_positions:
            _riv_reset = False
            for _gi in range(8):
                _rb = int(ram[0xC1 + _gi])
                if _rb == 1 and _gi not in env._rivet_game_popped:
                    env._rivet_game_popped.add(_gi)
                    if _gi < len(env._rivet_positions):
                        env._rivet_popped.add(_gi)
                elif _rb == 0 and _gi in env._rivet_game_popped:
                    _riv_reset = True
            if _riv_reset:
                env._rivet_popped      = set()
                env._rivet_game_popped = set()
            _fh, _fw = obs_raw.shape[:2]
            for _ri, (_rx, _ry) in enumerate(env._rivet_positions):
                _popped = _ri in env._rivet_popped
                _col    = (80, 80, 80) if _popped else (255, 210, 30)
                _rpx    = int(_rx * win_w / _fw)
                _rpy    = int(_ry * win_h / _fh)
                pygame.draw.rect(screen, _col, (_rpx - 8, _rpy - 8, 16, 16), 2)
                _lbl = 'REMOVED' if _popped else f'RIVET{_ri}'
                screen.blit(font.render(_lbl, True, _col), (_rpx + 10, _rpy - 6))
                detected_sprites.append((_lbl, _rx, _ry))

        # OAM slot-number overlay
        if show_slots:
            slot_font = pygame.font.SysFont('Consolas', 10)
            for s in range(64):
                sy = int(ram[OAM_BASE + s * 4])
                sx = int(ram[OAM_BASE + s * 4 + 3])
                if sy >= 0xF0 or sx == 0:
                    continue
                px2, py2 = to_px(sx, sy, y_off=1)
                pygame.draw.circle(screen, (255, 255, 0), (px2, py2), 3)
                screen.blit(slot_font.render(str(s), True, (255, 255, 0)), (px2 + 4, py2 - 5))

        # Pause overlay
        if paused:
            pause_surf = pygame.font.SysFont('Consolas', 28, bold=True).render(
                'PAUSED  (P to resume)', True, (255, 220, 0))
            screen.blit(pause_surf, (10, win_h // 2 - 18))

        # ── Side panel ────────────────────────────────────────────────────────
        pygame.draw.rect(screen, (18, 18, 28), (win_w, 0, PANEL_W * 2, win_h))
        pygame.draw.line(screen, (60, 60, 80), (win_w, 0), (win_w, win_h), 2)
        pygame.draw.line(screen, (60, 60, 80), (win_w + PANEL_W, 0), (win_w + PANEL_W, win_h), 2)

        py    = 10
        col_x = win_w   # x origin of the current panel column (win_w or win_w+PANEL_W)

        def _col_advance():
            """Switch to the second panel column; returns False if already in col 2."""
            nonlocal py, col_x
            if col_x < win_w + PANEL_W:
                col_x = win_w + PANEL_W
                py    = 10
                return True
            return False

        def pline(label, value, col=(190, 210, 255), bold=False):
            nonlocal py
            if py + 17 > win_h:
                if not _col_advance():
                    return
            f = font_bold if bold else font
            screen.blit(f.render(f'{label:<13}{value}', True, col), (col_x + 10, py))
            py += 17

        def psep(title=''):
            nonlocal py
            py += 4
            if py + 22 > win_h:
                if not _col_advance():
                    return
            if title:
                screen.blit(font_bold.render(title, True, (120, 140, 180)), (col_x + 10, py))
                py += 16
            pygame.draw.line(screen, (50, 55, 75), (col_x + 6, py), (col_x + PANEL_W - 6, py), 1)
            py += 6

        psep('GAME STATE')
        pline('Score',    bcd(ram[36:40], 4))
        pline('Lives',    int(ram[85]))
        pline('Stage',    int(ram[83]))
        pline('Level',    int(ram[84]) + 1)
        pline('Bonus',    bcd(ram[46:48], 2), col=(255, 220, 50))
        pline('Gameover', int(ram[78]))

        if watch_addr is not None:
            psep('WATCH ADDR')
            pline(f'  [{watch_addr}]', int(ram[watch_addr]) if watch_addr < len(ram) else '??',
                  col=(0, 255, 180))

        psep('MARIO')
        pline('X  (OAM)',      mario_x)
        pline('Y  (OAM)',      mario_y)
        in_zone_col = (255, 80, 80) if in_broken else (100, 200, 100)
        pline('Broken zone', 'YES' if in_broken else 'no', col=in_zone_col, bold=in_broken)

        psep('SPRITES')
        for name, sx, sy in detected_sprites:
            col = (255, 80, 80)   if any(x in name for x in ('BARREL', 'PIE')) else \
                  (255, 140, 0)   if 'FIREBALL' in name or 'FIRE' in name else \
                  (0, 200, 255)   if 'ELEVATOR' in name else \
                  (255, 200, 0)   if any(x in name for x in ('SPRING', 'HAMMER')) else \
                  (180, 255, 100) if 'PLATFORM' in name else \
                  (255, 180, 255) if 'PAULINE' in name else \
                  (255, 100, 200) if 'HEART' in name else \
                  (255, 120, 0)   if 'FLAME' in name else \
                  (190, 210, 255)
            pline(f'{name:<12}', f'x={sx:3d} y={sy:3d}', col=col)

        psep('EPISODE')
        pline('Step', step_count)

        if cur_stage == 4:
            psep('RIVETS')
            _nk       = len(env._rivet_positions)
            _np       = len(env._rivet_popped)
            _ram_sum  = sum(int(ram[0xC1 + i]) for i in range(8))
            _rb_str   = ''.join(str(int(ram[0xC1 + i])) for i in range(8))
            _match    = (_ram_sum == _np)
            pline('Detected', f'{_nk}/8',
                  col=(0, 200, 100) if _nk == 8 else (255, 210, 30))
            pline('Removed',  f'{_np}/{_nk} (RAM:{_ram_sum})',
                  col=(0, 200, 100) if _match else (255, 80, 80))
            pline('0xC1-C8', _rb_str, col=(160, 200, 160))

        psep('LADDERS')
        if env._ladder_rects:
            _fh2, _fw2 = obs_raw.shape[:2]
            for _li, (_lx0, _ly0, _lx1, _ly1) in enumerate(env._ladder_rects):
                _nx0 = int(_lx0 * 256 / _fw2); _ny0 = int(_ly0 * 224 / _fh2)
                _nx1 = int(_lx1 * 256 / _fw2); _ny1 = int(_ly1 * 224 / _fh2)
                pline(f'  L{_li}', f'({_nx0},{_ny0})-({_nx1},{_ny1})',
                      col=(0, 220, 220))
        else:
            pline('  (none)', '', col=(90, 90, 110))

        # RAM watch status
        psep('RAM WATCH  [W]')
        if ram_watch:
            pline('Status', 'ON', col=(0, 255, 180), bold=True)
            pline('Stable', f'≥{RAM_WATCH_STABLE} steps', col=(140, 200, 140))
            pline('', '→ see console', col=(100, 130, 100))
        else:
            pline('Status', 'off', col=(90, 90, 110))
            pline('', 'W to enable', col=(70, 70, 90))

        psep('STAGE JUMP  (keys 1-4)')
        _snames = {1: 'Barrels', 2: 'Cement', 3: 'Elevator', 4: 'Rivets'}
        for _s in range(1, 5):
            if _s == 1:
                pline(f'  [1] {_snames[1]}', 'ready', col=(0, 200, 100))
            else:
                _sp = _stage_state_path(_s)
                col = (0, 200, 100) if os.path.exists(_sp) else (90, 90, 110)
                lbl = 'ready' if os.path.exists(_sp) else 'play to unlock'
                pline(f'  [{_s}] {_snames[_s]}', lbl, col=col)

        psep('BROKEN ZONES')
        for i, z in enumerate(env._broken_zones):
            pline(f'  z{i}', f'({z[0]},{z[1]})-({z[2]},{z[3]})', col=(160, 160, 160))

        pygame.display.flip()
        clock.tick(60)

        if done and auto_reset:
            obs        = env.reset()
            cum_reward = 0.0
            step_count = 0
        elif done:
            print('  → Episode ended (auto-reset disabled — press Q to quit)')

    env.close()
    pygame.quit()
    analyse(ram_log, lives_log)


if __name__ == '__main__':
    main()
