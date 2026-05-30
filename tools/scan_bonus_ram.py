"""
RAM scanner to find the bonus/time remaining counter in NES Donkey Kong.

Runs the game headlessly for ~600 steps (with FRAME_SKIP=1 so we get full
NES-frame resolution), takes RAM snapshots every 60 frames (≈1 second), and
reports addresses whose value decreases monotonically — the bonus timer is
the only RAM location that does this reliably throughout a full run.

Run with:
    python tools/scan_bonus_ram.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

INTEGRATION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'retro_data'))

SNAPSHOT_INTERVAL = 60   # NES frames between snapshots
N_SNAPSHOTS       = 12   # collect 12 snapshots ≈ 12 seconds of gameplay
NOOP              = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def main():
    import retro  # noqa

    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
    env = retro.make(
        'DonkeyKong-Nes',
        state='1Player.GameA',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    env.reset()
    ram_size = len(env.get_ram())
    print(f'NES RAM size: {ram_size} bytes')

    snapshots = []   # list of np.uint8 arrays

    # Warm up — skip first 30 frames (title/flash)
    for _ in range(30):
        env.step(NOOP)

    # Collect snapshots every SNAPSHOT_INTERVAL frames
    frame = 0
    while len(snapshots) < N_SNAPSHOTS:
        env.step(NOOP)
        frame += 1
        if frame % SNAPSHOT_INTERVAL == 0:
            ram = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
            snapshots.append(ram)
            print(f'  snapshot {len(snapshots)}/{N_SNAPSHOTS}  (frame {frame})')

    env.close()

    snapshots = np.stack(snapshots)   # (N_SNAPSHOTS, ram_size)

    print('\n--- Single-byte candidates (monotonically decreasing, non-zero start) ---')
    print(f'{"Addr":>6}  {"Type":>8}  {"Values across snapshots"}')

    for addr in range(ram_size):
        vals = snapshots[:, addr].astype(int)
        if vals[0] == 0:
            continue
        diffs = np.diff(vals)
        # Strictly non-increasing with at least one actual decrease
        if np.all(diffs <= 0) and np.any(diffs < 0):
            print(f'  {addr:4d} (0x{addr:03X})  uint8  {vals.tolist()}')

    # Also check 16-bit big-endian values
    print('\n--- 16-bit big-endian candidates ---')
    for addr in range(ram_size - 1):
        vals = (snapshots[:, addr].astype(int) << 8) | snapshots[:, addr + 1].astype(int)
        if vals[0] == 0:
            continue
        diffs = np.diff(vals)
        if np.all(diffs <= 0) and np.any(diffs < 0) and vals[0] < 32000:
            print(f'  {addr:4d} (0x{addr:03X})  uint16be  {vals.tolist()}')

    # BCD 2-byte: interpret each nibble as decimal digit
    print('\n--- BCD 2-byte candidates (values make sense as 4-digit decimals) ---')
    def bcd2(hi, lo):
        return (hi >> 4) * 1000 + (hi & 0xF) * 100 + (lo >> 4) * 10 + (lo & 0xF)

    for addr in range(ram_size - 1):
        vals = [bcd2(int(snapshots[i, addr]), int(snapshots[i, addr + 1]))
                for i in range(N_SNAPSHOTS)]
        if vals[0] == 0:
            continue
        # Valid BCD: each nibble in 0-9
        valid = all(
            all((int(snapshots[i, addr + b]) >> 4) <= 9 and (int(snapshots[i, addr + b]) & 0xF) <= 9
                for b in range(2))
            for i in range(N_SNAPSHOTS)
        )
        if not valid:
            continue
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs) and vals[0] <= 9999:
            print(f'  {addr:4d} (0x{addr:03X})  BCD2  {vals}')


if __name__ == '__main__':
    main()
