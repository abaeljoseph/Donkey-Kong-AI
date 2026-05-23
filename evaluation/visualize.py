"""
Visualize how the AI perceives the game world.

Overlays on the game window:
  GREEN boxes  — ladders (checkpoints, climb these)
  RED boxes    — barrels and fireballs (danger, avoid)
  YELLOW box   — Mario with movement arrow

Usage:
    python visualize.py                               # random actions
    python visualize.py checkpoints/model_ep500.pth   # trained agent
"""

import sys
import numpy as np
import cv2
import torch

from environment import DonkeyKongEnv
from dqn_agent import DQNAgent

ACTION_NAMES = ['NOOP', 'RIGHT', 'LEFT', 'JUMP', 'R+JUMP', 'L+JUMP',
                'UP', 'UP+R', 'UP+L', 'DOWN']

# Checkpoint Y positions in the 448-pixel display height.
# Each entry is (display_y, label). Lower Y = higher on screen = closer to princess.
# These align with the 4 platform floors Mario must climb through.
CHECKPOINTS = [
    (370, 'CP1  Floor 2'),
    (285, 'CP2  Floor 3'),
    (200, 'CP3  Floor 4'),
    (115, 'CP4  Floor 5'),
]

# HSV color ranges for NES Donkey Kong palette
# Ladders are bright cyan — tightly bounded so blue oil barrel (darker blue) is excluded
LADDER_LOWER  = np.array([85,  180, 160])
LADDER_UPPER  = np.array([97,  255, 255])

# Rolling barrels (orange-brown) + animated spinning barrel sprites
BARREL_LOWER  = np.array([5,  60, 80])
BARREL_UPPER  = np.array([30, 255, 255])

# Fireballs — bright saturated red only (platforms are dark/unsaturated, excluded)
FIRE_LOWER1   = np.array([0,   200, 200])
FIRE_UPPER1   = np.array([10,  255, 255])
FIRE_LOWER2   = np.array([170, 200, 200])
FIRE_UPPER2   = np.array([180, 255, 255])

# Spinning barrel sprite — small bright blue only
SPIN_LOWER    = np.array([100, 180, 180])
SPIN_UPPER    = np.array([125, 255, 255])

# Princess (Pauline) — pink/magenta dress, top quarter of screen only
PRINCESS_LOWER = np.array([140, 80, 150])
PRINCESS_UPPER = np.array([175, 255, 255])

# Mario's red is pure bright red — tighter than DK's brownish red
MARIO_BLUE_L  = np.array([105, 150, 100])  # Mario's blue overalls — unique, no overlap with fire
MARIO_BLUE_U  = np.array([125, 255, 220])


def detect_objects(frame_rgb, last_mario_box=None):
    """
    Detect ladders, danger objects, and Mario by color.
    Returns dict of bounding boxes for each category.
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # --- Ladders ---
    ladder_mask = cv2.inRange(hsv, LADDER_LOWER, LADDER_UPPER)
    ladder_mask = cv2.dilate(ladder_mask, np.ones((3, 3), np.uint8), iterations=1)

    # --- Barrels (rolling orange-brown circles DK throws) ---
    barrel_mask = cv2.inRange(hsv, BARREL_LOWER, BARREL_UPPER)

    # --- Fireballs ---
    fire_mask = cv2.bitwise_or(
        cv2.inRange(hsv, FIRE_LOWER1, FIRE_UPPER1),
        cv2.inRange(hsv, FIRE_LOWER2, FIRE_UPPER2),
    )

    # --- Spinning barrel / enemy sprites (blue body) ---
    spin_mask = cv2.inRange(hsv, SPIN_LOWER, SPIN_UPPER)

    danger_mask = cv2.bitwise_or(cv2.bitwise_or(barrel_mask, fire_mask), spin_mask)

    # --- Princess detection — top 25% of screen only ---
    princess_mask = cv2.inRange(hsv, PRINCESS_LOWER, PRINCESS_UPPER)
    princess_cutoff = int(frame_rgb.shape[0] * 0.25)
    princess_mask[princess_cutoff:, :] = 0  # ignore anything below top quarter

    # --- Mario detection via blue overalls ---
    mario_mask = cv2.inRange(hsv, MARIO_BLUE_L, MARIO_BLUE_U)
    top_cutoff = int(frame_rgb.shape[0] * 0.30)
    mario_mask[:top_cutoff, :] = 0

    mario_contours, _ = cv2.findContours(mario_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    found_mario = None
    for c in mario_contours:
        if 8 < cv2.contourArea(c) < 300:
            found_mario = cv2.boundingRect(c)
            break

    # Use last known position as fallback during jump (sprite changes, blue shrinks)
    mario_box = found_mario if found_mario else last_mario_box

    # Blank Mario's region from danger mask — extra upward pad covers jump arc
    if mario_box:
        x, y, w, h = mario_box
        pad_up   = 40  # Mario moves up during a jump; generous pad
        pad_side = 22
        pad_down = 18
        danger_mask[max(0, y - pad_up):y + h + pad_down,
                    max(0, x - pad_side):x + w + pad_side] = 0

    # Secondary blanking via Mario's red hat — only applied when we already
    # know where Mario is (mario_box confirmed). Blanks only the small region
    # directly above the known Mario position, so fireballs elsewhere are safe.
    if mario_box:
        mx, my, mw, mh = mario_box
        hat_region = hsv[max(0, my - 20):my + mh, max(0, mx - 8):mx + mw + 8]
        if hat_region.size > 0:
            hat_mask = cv2.bitwise_or(
                cv2.inRange(hat_region, np.array([0,  150, 150]), np.array([8,  255, 255])),
                cv2.inRange(hat_region, np.array([172, 150, 150]), np.array([180, 255, 255])),
            )
            y1 = max(0, my - 20)
            x1 = max(0, mx - 8)
            danger_mask[y1:my + mh, x1:mx + mw + 8] = np.where(
                hat_mask > 0, 0, danger_mask[y1:my + mh, x1:mx + mw + 8]
            )

    def get_boxes(mask, min_area=40, max_area=99999, min_aspect=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            # min_aspect = min height/width ratio — ladders are tall, oil barrel is squat
            if min_aspect and (h / max(w, 1)) < min_aspect:
                continue
            boxes.append((x, y, w, h))
        return boxes

    return {
        'ladders':    get_boxes(ladder_mask,   min_area=80,  min_aspect=1.5),
        'danger':     get_boxes(danger_mask,   min_area=25,  max_area=3000),
        'mario':      get_boxes(mario_mask,    min_area=8,   max_area=300),
        'princess':   get_boxes(princess_mask, min_area=15,  max_area=500),
        'mario_box':  mario_box,
    }


def draw_overlays(frame_rgb, objects, mario_prev, action_name, q_values,
                  reached_checkpoints=None):
    """Draw all overlays onto the frame."""
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (480, 448))

    sx = 480 / frame_rgb.shape[1]
    sy = 448 / frame_rgb.shape[0]

    def scale(box):
        x, y, w, h = box
        return int(x*sx), int(y*sy), int(w*sx), int(h*sy)

    # --- Princess — gold/pink box + GOAL label ---
    for box in objects.get('princess', []):
        x, y, w, h = scale(box)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 220), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 100, 255), 2)
        cv2.putText(frame, 'GOAL', (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    # --- Ladders — green boxes + CLIMB label ---
    for box in objects['ladders']:
        x, y, w, h = scale(box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
        cv2.putText(frame, 'CLIMB', (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)

    # --- Danger — red boxes + DANGER label ---
    for box in objects['danger']:
        x, y, w, h = scale(box)
        # Red filled semi-transparent danger zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'DANGER', (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # --- Mario — cyan box + movement arrow ---
    # Prefer live blob detection; fall back to last_mario_box during jump
    mario_center = None
    if objects['mario']:
        all_x = [b[0] for b in objects['mario']]
        all_y = [b[1] for b in objects['mario']]
        all_r = [b[0]+b[2] for b in objects['mario']]
        all_b = [b[1]+b[3] for b in objects['mario']]
        draw_box = (min(all_x), min(all_y),
                    max(all_r) - min(all_x), max(all_b) - min(all_y))
        label_color = (0, 220, 220)
    elif objects['mario_box']:
        draw_box = objects['mario_box']
        label_color = (0, 160, 160)  # dimmed to show it's a fallback
    else:
        draw_box = None
        label_color = (0, 220, 220)

    if draw_box:
        x, y, w, h = scale(draw_box)
        x -= 4; y -= 4; w += 8; h += 8

        cv2.rectangle(frame, (x, y), (x+w, y+h), label_color, 2)
        cv2.putText(frame, 'MARIO', (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

        mario_center = (x + w//2, y + h//2)

        # Movement arrow from previous position
        if mario_prev is not None:
            dx = mario_center[0] - mario_prev[0]
            dy = mario_center[1] - mario_prev[1]
            dist = max(abs(dx) + abs(dy), 1)
            if dist > 3:
                arrow_end = (mario_center[0] + dx*3, mario_center[1] + dy*3)
                cv2.arrowedLine(frame, mario_center, arrow_end,
                                (0, 255, 255), 2, tipLength=0.4)
                direction = ''
                if dx > 2:   direction = '→ Moving RIGHT (good!)'
                elif dx < -2: direction = '← Moving LEFT'
                if dy < -2:   direction += ' ↑ Climbing!'
                if direction:
                    cv2.putText(frame, direction, (8, 440),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # --- Chosen action + Q-bar ---
    cv2.putText(frame, f'Action: {action_name}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if q_values is not None:
        q_min, q_max = q_values.min(), q_values.max()
        q_range = max(q_max - q_min, 1e-6)
        bx = frame.shape[1] - 155
        cv2.putText(frame, 'Q-VALUES', (bx, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for i, (q, name) in enumerate(zip(q_values, ACTION_NAMES)):
            y     = 26 + i * 17
            blen  = int(((q - q_min) / q_range) * 130)
            color = (0, 255, 0) if i == np.argmax(q_values) else (70, 70, 70)
            cv2.rectangle(frame, (bx, y), (bx + blen, y + 13), color, -1)
            cv2.putText(frame, f'{name:<6}{q:+.1f}', (bx - 8, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # --- Checkpoint lines ---
    if reached_checkpoints is None:
        reached_checkpoints = set()
    for cp_y, cp_label in CHECKPOINTS:
        done_cp = cp_label in reached_checkpoints
        color   = (0, 255, 80) if done_cp else (0, 165, 255)  # green=done, orange=pending
        # Dashed line: draw segments across the width
        dash_len, gap_len = 12, 6
        x = 0
        while x < frame.shape[1] - 155:  # stop before Q-value panel
            cv2.line(frame, (x, cp_y), (min(x + dash_len, frame.shape[1] - 155), cp_y),
                     color, 1)
            x += dash_len + gap_len
        tick = '✓' if done_cp else '○'
        cv2.putText(frame, f'{tick} {cp_label}', (4, cp_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return frame, mario_center


def build_ai_view(frame_stack):
    """4x84x84 float stack → side-by-side grayscale panels."""
    frames = (frame_stack * 255).astype(np.uint8)
    scale  = 4
    panels = []
    labels = ['Oldest', 'Old', 'Recent', 'NOW']
    for i in range(4):
        p = cv2.resize(frames[i], (84*scale, 84*scale),
                       interpolation=cv2.INTER_NEAREST)
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        cv2.putText(p, labels[i], (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(p)
    div = np.full((84*scale, 2, 3), (0, 255, 0), dtype=np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, div, p], axis=1)
    return row


def run(checkpoint_path=None, episodes=10):
    env   = DonkeyKongEnv(render=False, rgb_array=True)
    agent = DQNAgent(action_size=env.action_space_size)

    if checkpoint_path:
        agent.load(checkpoint_path)
        agent.epsilon = 0.05
        print(f"Loaded: {checkpoint_path}")
    else:
        agent.epsilon = 1.0
        print("No checkpoint — random actions")

    cv2.namedWindow('Donkey Kong — Tracked',   cv2.WINDOW_NORMAL)
    cv2.namedWindow('AI Vision (CNN input)',    cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Donkey Kong — Tracked',  480, 448)
    cv2.resizeWindow('AI Vision (CNN input)',   84*4*4 + 6, 84*4)
    print("Press Q to quit\n")

    for episode in range(1, episodes + 1):
        state             = env.reset()
        done              = False
        total_rew         = 0.0
        steps             = 0
        mario_prev        = None
        last_mario_box    = None
        reached_cps       = set()
        cp_dwell          = {lbl: 0 for _, lbl in CHECKPOINTS}  # frames above each line

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.policy_net(state_t).cpu().numpy()[0]

            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_rew += reward
            steps     += 1

            raw = env.get_raw_frame()
            if raw is not None and isinstance(raw, np.ndarray):
                objects = detect_objects(raw, last_mario_box)
                last_mario_box = objects['mario_box'] or last_mario_box

                # Only mark checkpoint after Mario has been above the line for 8 frames
                # — prevents a jump arc from falsely triggering it
                if mario_prev is not None:
                    for cp_y, cp_label in CHECKPOINTS:
                        if cp_label in reached_cps:
                            continue
                        if mario_prev[1] <= cp_y:
                            cp_dwell[cp_label] += 1
                            if cp_dwell[cp_label] >= 8:
                                reached_cps.add(cp_label)
                        else:
                            cp_dwell[cp_label] = 0  # reset if Mario drops back down

                tracked, mario_prev = draw_overlays(
                    raw, objects, mario_prev, ACTION_NAMES[action], q_vals,
                    reached_checkpoints=reached_cps)
                cv2.imshow('Donkey Kong — Tracked', tracked)

            cv2.imshow('AI Vision (CNN input)', build_ai_view(state))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return

        print(f"Ep {episode:3d} | Reward: {total_rew:7.2f} | "
              f"Steps: {steps:5d} | Lives: {info.get('lives', 0)}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run(checkpoint_path=ckpt)
