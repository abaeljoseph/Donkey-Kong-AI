"""
Records a 1Player.GameA.state for Donkey Kong Original Edition.

Finds the NES RAM region inside the emulator state blob by cross-referencing
two known states (before/after attract-mode starts), then patches the attract-
mode flag (RAM[88]) to 0 and saves a clean player-mode starting state.

Run once from the project root:
    python tools/record_oe_state.py
"""

import os, gzip, numpy as np, retro

INTEGRATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retro_data'))
GAME_NAME = 'DonkeyKongOriginalEdition-Nes'
OUT_PATH  = os.path.join(INTEGRATION_PATH, GAME_NAME, '1Player.GameA.state')
NOOP      = np.zeros(9, dtype=np.uint8)

retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

def make_env():
    e = retro.make(GAME_NAME, state=retro.State.NONE,
                   inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
    e.reset()
    return e

# ── Step 1: find the frame attract mode kicks in ─────────────────────────────
print('Step 1: locating attract-mode transition frame...')
env = make_env()
prev_ram = env.get_ram().copy()
attract_frame = None
for frame in range(800):
    env.step(NOOP)
    ram = env.get_ram()
    if int(ram[78]) == 0 and int(prev_ram[78]) == 1:
        attract_frame = frame
        print(f'  Attract mode starts at frame {attract_frame}')
        break
    prev_ram = ram.copy()
env.close()

if attract_frame is None:
    raise SystemExit('ERROR: attract mode never detected')

# ── Step 2: capture two states to locate RAM inside state blob ───────────────
print('Step 2: locating NES RAM inside emulator state bytes...')
env = make_env()
for _ in range(attract_frame - 2):       # one frame before transition
    env.step(NOOP)

state_A   = bytearray(env.em.get_state())
ram_A     = env.get_ram().copy()         # gameover=1, attract_flag=0, stage=0

env.step(NOOP)
env.step(NOOP)

state_B   = bytearray(env.em.get_state())
ram_B     = env.get_ram().copy()         # gameover=0, attract_flag=1, stage=1
env.close()

# Find byte offset R such that state_B[R + addr] == ram_B[addr].
# Use the unique 7-byte sequence at RAM[24:31] in attract-mode state:
# [46, 42, 118, 69, 157, 22, 44] — these values appear exactly at that spot.
pattern = bytes([int(ram_B[i]) for i in range(24, 31)])
print(f'  Searching for pattern at RAM[24:31]: {list(pattern)}')

ram_base = None
for i in range(len(state_B) - 31):
    if bytes(state_B[i:i+7]) == pattern:
        R = i - 24
        if R >= 0:
            # Verify against several other known attract-mode RAM values
            checks = [(67, int(ram_B[67])), (78, int(ram_B[78])),
                      (83, int(ram_B[83])), (88, int(ram_B[88]))]
            if all(R + addr < len(state_B) and state_B[R + addr] == val
                   for addr, val in checks):
                ram_base = R
                print(f'  RAM base offset in state: {ram_base}')
                break

if ram_base is None:
    # Fallback: search for mario_x=117 at RAM[67] and cross-check
    print('  Pattern not found — trying fallback search on mario_x=117...')
    for i in range(len(state_B)):
        if state_B[i] == 117:
            R = i - 67
            if R >= 0 and R + 2048 <= len(state_B):
                checks = [(78, 0), (83, 1), (88, 1)]
                if all(state_B[R + addr] == val for addr, val in checks):
                    ram_base = R
                    print(f'  RAM base offset (fallback): {ram_base}')
                    break

if ram_base is None:
    # Last resort: dump all positions where state bytes differ between A and B
    diffs = [i for i in range(min(len(state_A), len(state_B))) if state_A[i] != state_B[i]]
    print(f'  State A→B differs at {len(diffs)} positions: {diffs[:30]}')
    raise SystemExit('ERROR: could not locate RAM — see diff positions above')

# ── Step 3: boot to attract mode, patch flag, save ───────────────────────────
print('Step 3: patching attract-mode flag and saving state...')
env = make_env()
for _ in range(attract_frame + 10):     # a few frames into attract mode so game is fully initialised
    env.step(NOOP)

state = bytearray(env.em.get_state())
ram   = env.get_ram()
print(f'  Before patch: gameover={ram[78]}  attract={ram[88]}  lives={ram[85]}  stage={ram[83]}')

# Patch attract-mode flag off and set correct starting lives
state[ram_base + 88] = 0   # RAM[88]: attract flag  0=player, 1=attract
state[ram_base + 85] = 3   # RAM[85]: lives          3 = 3 lives

env.em.set_state(bytes(state))
for _ in range(30):
    env.step(NOOP)

ram = env.get_ram()
print(f'  After patch:  gameover={ram[78]}  attract={ram[88]}  lives={ram[85]}  stage={ram[83]}')
env.close()

# Re-get state after settling frames
env = make_env()
for _ in range(attract_frame + 10):
    env.step(NOOP)
state = bytearray(env.em.get_state())
state[ram_base + 88] = 0
state[ram_base + 85] = 3
env.em.set_state(bytes(state))
for _ in range(30):
    env.step(NOOP)
final_state = env.em.get_state()
env.close()

with gzip.open(OUT_PATH, 'wb') as f:
    f.write(final_state)
print(f'State saved → {OUT_PATH}')

# ── Verify ────────────────────────────────────────────────────────────────────
print('\nVerifying...')
env2 = retro.make(GAME_NAME, state='1Player.GameA',
                  inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
r = env2.reset()
ram2 = env2.get_ram()
print(f'  gameover={ram2[78]}  attract={ram2[88]}  lives={ram2[85]}  stage={ram2[83]}')
env2.close()
print('Done — run: python tools/test_broken_ladder.py --game dkoe')
