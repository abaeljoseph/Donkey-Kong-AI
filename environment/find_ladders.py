"""
Runs random episodes and logs every frame where Mario moves upward.
After enough episodes the X values cluster around each real ladder position.

Usage:
    python find_ladders.py --episodes 200
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.environment import DonkeyKongEnv

def find_ladders(num_episodes=200):
    env = DonkeyKongEnv(render=False)
    climb_log = []   # list of (mario_x, mario_y) whenever upward movement detected

    prev_x = None
    prev_y = None

    for ep in range(1, num_episodes + 1):
        env.reset()
        prev_x = None
        prev_y = None
        done = False

        while not done:
            action = np.random.randint(0, env.action_space_size)
            _, _, done, _ = env.step(action)

            try:
                ram = env.env.get_ram()
                mx = int(ram[0x0043])
                my = int(ram[0x0044])

                if prev_y is not None:
                    dy = prev_y - my   # positive = moved up
                    if 0 < dy <= 15:
                        climb_log.append((mx, my))

                prev_x = mx
                prev_y = my
            except Exception:
                pass

        if ep % 20 == 0:
            print(f"Episode {ep}/{num_episodes}  |  climb samples so far: {len(climb_log)}")

    env.close()

    if not climb_log:
        print("No climbing detected — try more episodes.")
        return

    xs = np.array([x for x, y in climb_log])
    ys = np.array([y for x, y in climb_log])

    print("\n--- Raw X distribution when Mario climbs ---")
    hist, edges = np.histogram(xs, bins=30)
    for i, count in enumerate(hist):
        if count > 0:
            x_lo = int(edges[i])
            x_hi = int(edges[i + 1])
            bar  = "#" * (count // max(1, max(hist) // 40))
            print(f"  X {x_lo:3d}-{x_hi:3d} | {bar} ({count})")

    # Simple peak detection — find X ranges with high climb density
    print("\n--- Likely ladder X zones (high climb density) ---")
    threshold = max(hist) * 0.20   # zones with at least 20% of peak activity
    zones = []
    in_zone = False
    for i, count in enumerate(hist):
        x_lo = int(edges[i])
        x_hi = int(edges[i + 1])
        if count >= threshold:
            if not in_zone:
                zone_start = x_lo
                in_zone = True
            zone_end = x_hi
        else:
            if in_zone:
                zones.append((zone_start, zone_end))
                in_zone = False
    if in_zone:
        zones.append((zone_start, zone_end))

    for i, (lo, hi) in enumerate(zones, 1):
        mask = (xs >= lo) & (xs <= hi)
        y_in_zone = ys[mask]
        y_range = f"Y {int(y_in_zone.min())}–{int(y_in_zone.max())}" if len(y_in_zone) else ""
        print(f"  Ladder {i}: X {lo}–{hi}   {y_range}   ({mask.sum()} samples)")

    print("\nCopy these zones into environment.py as UNBROKEN_LADDERS.")
    print("Ignore any zone near X 60-90 — that's the broken/oil-barrel ladder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()
    find_ladders(args.episodes)
