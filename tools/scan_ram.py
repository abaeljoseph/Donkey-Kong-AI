"""
RAM address scanner for Donkey Kong NES.
Runs the game, presses RIGHT for a few frames, then compares RAM snapshots
to find which addresses correlate with Mario's X position.
Run:  python tools/scan_ram.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import retro

INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'retro_data')
retro.data.Integrations.add_custom_path(os.path.abspath(INTEGRATION_PATH))

env = retro.make(
    game='DonkeyKong-Nes',
    state='1Player.GameA',
    inttype=retro.data.Integrations.CUSTOM_ONLY,
)
env.reset()

BTN_NONE  = [0]*9
BTN_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0]
BTN_LEFT  = [0, 0, 0, 0, 0, 0, 1, 0, 0]
BTN_JUMP  = [1, 0, 0, 0, 0, 0, 0, 0, 0]

def get_ram(env):
    return np.array(env.get_ram(), dtype=np.uint8)

# Warm up — let the game fully initialise
for _ in range(30):
    env.step(BTN_NONE)

snap_before = get_ram(env)

# Move RIGHT for 30 frames
for _ in range(30):
    env.step(BTN_RIGHT)

snap_after = get_ram(env)

changed = np.where(snap_before != snap_after)[0]
print(f'\nAddresses that changed while moving RIGHT ({len(changed)} total):')
for addr in changed:
    print(f'  addr {addr:4d} (0x{addr:04X}):  {snap_before[addr]:3d} → {snap_after[addr]:3d}  (delta {int(snap_after[addr]) - int(snap_before[addr]):+d})')

# Now move LEFT and see what decreases
snap_mid = get_ram(env)
for _ in range(30):
    env.step(BTN_LEFT)
snap_left = get_ram(env)

print('\nAddresses that DECREASED while moving LEFT (likely X position):')
for addr in changed:
    if snap_left[addr] < snap_mid[addr]:
        print(f'  addr {addr:4d} (0x{addr:04X}):  {snap_mid[addr]:3d} → {snap_left[addr]:3d}')

# Jump and check Y
snap_before_jump = get_ram(env)
for _ in range(20):
    env.step(BTN_JUMP)
snap_jump = get_ram(env)

print('\nAddresses that changed during JUMP (likely Y position or jump flag):')
changed_jump = np.where(snap_before_jump != snap_jump)[0]
for addr in changed_jump[:20]:
    print(f'  addr {addr:4d} (0x{addr:04X}):  {snap_before_jump[addr]:3d} → {snap_jump[addr]:3d}')

env.close()
print('\nDone. Add promising addresses to retro_data/DonkeyKong-Nes/data.json')
