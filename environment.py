"""
Person 1 — Environment
Wraps gym-retro, preprocesses frames, defines the reward function,
and runs a PyBullet robot arm that mirrors every action the agent takes.
"""

import numpy as np
import cv2
from collections import deque

import stable_retro as retro
import pybullet as p
import pybullet_data


GAME_NAME = 'DonkeyKong-Nes-v0'

# Unbroken ladder zones discovered via find_ladders.py (400 random episodes).
# Each tuple: (name, x_min, x_max).  Broken ladder never produces upward Y
# movement so it does not appear in the data.
UNBROKEN_LADDERS = [
    ('ladder_main',  99, 112),   # correct floor-1→2 ladder (spans all floors, Y 0-254)
    ('ladder_left',  45,  72),   # upper-floor left ladders  (floors 2-4, Y 0-45)
    ('ladder_right', 117, 135),  # upper-floor right ladders (floors 3-4, Y 0-35)
]

def _on_ladder(mario_x):
    return any(lo <= mario_x <= hi for _, lo, hi in UNBROKEN_LADDERS)

# Discrete action set: each entry is a 9-element NES button array
# NES layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: RIGHT
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: LEFT
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: JUMP (A button)
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 4: RIGHT + JUMP
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # 5: LEFT + JUMP
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: UP (climb ladder)
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 7: UP + RIGHT (approach & climb)
    [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 8: UP + LEFT
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 9: DOWN (descend ladder)
]

# Maps each action index to (shoulder_angle, elbow_angle) in radians
ARM_ANGLES = {
    0: (0.0,  0.0),   # NOOP
    1: (0.5,  0.2),   # RIGHT
    2: (-0.5, 0.2),   # LEFT
    3: (0.0, -0.6),   # JUMP
    4: (0.5, -0.4),   # RIGHT+JUMP
    5: (-0.5, -0.4),  # LEFT+JUMP
    6: (0.0,  0.5),   # UP
    7: (0.5,  0.5),   # UP+RIGHT
    8: (-0.5, 0.5),   # UP+LEFT
    9: (0.0, -0.3),   # DOWN
}


class DonkeyKongEnv:
    def __init__(self, render=False, frame_skip=4, rgb_array=False):
        self.render_mode = render
        self.rgb_array = rgb_array
        self.frame_skip = frame_skip
        self._init_retro()
        self._init_pybullet()
        self.frame_stack = deque(maxlen=4)
        self.prev_score = 0
        self.prev_lives = 3
        self.prev_mario_x = None
        self.prev_mario_y = None
        self.best_mario_y = None  # lowest Y reached on correct ladder (lower Y = higher floor)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def get_raw_frame(self):
        try:
            return self.env.render()
        except Exception:
            return None

    def _init_retro(self):
        if self.rgb_array:
            render_mode = 'rgb_array'
        elif self.render_mode:
            render_mode = 'human'
        else:
            render_mode = None
        try:
            self.env = retro.make(game=GAME_NAME, render_mode=render_mode)
        except TypeError:
            # older retro builds don't accept render_mode
            self.env = retro.make(game=GAME_NAME)
        except Exception:
            list_fn = getattr(retro.data, 'list_games', None) or retro.list_games
            available = list_fn()
            dk_games = [g for g in available if 'donkey' in g.lower() or 'kong' in g.lower()]
            raise RuntimeError(
                f"Could not load '{GAME_NAME}'. "
                f"Run: python -m retro.import DonkeyKong-Nes.nes\n"
                f"Donkey Kong entries found: {dk_games}"
            )

    def _init_pybullet(self):
        mode = p.GUI if self.render_mode else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self._build_arm()

    def _build_arm(self):
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        link1_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.3)
        link2_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.25)

        base_vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                         rgbaColor=[0.4, 0.4, 0.4, 1])
        link1_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.3,
                                         rgbaColor=[0.8, 0.2, 0.2, 1])
        link2_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.25,
                                         rgbaColor=[0.2, 0.8, 0.2, 1])

        self.arm = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.05],
            linkMasses=[1.0, 0.5],
            linkCollisionShapeIndices=[link1_col, link2_col],
            linkVisualShapeIndices=[link1_vis, link2_vis],
            linkPositions=[[0, 0, 0.1], [0, 0, 0.15]],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0.15], [0, 0, 0.125]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 1],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 1, 0], [0, 1, 0]],
        )
        for joint in range(2):
            p.setJointMotorControl2(self.arm, joint, p.POSITION_CONTROL,
                                    targetPosition=0, force=50)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _stacked_state(self):
        return np.array(list(self.frame_stack), dtype=np.float32)  # (4, 84, 84)

    # ------------------------------------------------------------------
    # PyBullet arm
    # ------------------------------------------------------------------

    def _move_arm(self, action_idx):
        shoulder, elbow = ARM_ANGLES.get(action_idx, (0.0, 0.0))
        p.setJointMotorControl2(self.arm, 0, p.POSITION_CONTROL,
                                targetPosition=shoulder, force=50)
        p.setJointMotorControl2(self.arm, 1, p.POSITION_CONTROL,
                                targetPosition=elbow, force=50)
        p.stepSimulation()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, info, done, action_idx=0):
        reward = 0.0

        # Score reward — verified correct via stable-retro data.json.
        # Level completion gives far more score than barrel jumping, so the
        # agent will naturally prefer climbing once it discovers that path.
        score = info.get('score', self.prev_score)
        delta_score = score - self.prev_score
        if delta_score > 0:
            reward += delta_score * 0.3
        self.prev_score = score

        # Death penalty — verified correct via stable-retro data.json.
        lives = info.get('lives', self.prev_lives)
        if lives < self.prev_lives:
            reward -= 15.0
        self.prev_lives = lives

        # Tiny survival bonus — kept small so score and climbing rewards
        # dominate Q-values instead of all actions converging to the same value.
        reward += 0.001


        # Ladder climbing reward — uses zones discovered by find_ladders.py.
        # Only fires on unbroken ladders; broken ladder never produces upward
        # Y movement so it is naturally excluded.
        try:
            ram = self.env.get_ram()
            mario_x = int(ram[0x0043])
            mario_y = int(ram[0x0044])

            # Reward rightward, penalise leftward — oil barrel is to the left
            if self.prev_mario_x is not None:
                dx = mario_x - self.prev_mario_x
                if dx > 0:
                    reward += dx * 0.1
                elif dx < 0:
                    reward -= abs(dx) * 0.15

            # Climbing reward — active on every labeled unbroken ladder
            if _on_ladder(mario_x) and self.prev_mario_y is not None:
                dy = self.prev_mario_y - mario_y  # positive = moved up
                if 0 < dy <= 15:
                    reward += dy * 5.0

                    # One-time bonus for reaching a new personal-best height —
                    # scales with how much higher Mario has gone, so pushing to
                    # each new floor is worth far more than retreating.
                    if self.best_mario_y is None or mario_y < self.best_mario_y:
                        if self.best_mario_y is not None:
                            reward += (self.best_mario_y - mario_y) * 4.0
                        self.best_mario_y = mario_y

            self.prev_mario_x = mario_x
            self.prev_mario_y = mario_y
        except Exception:
            pass

        return reward

    def print_ram_positions(self):
        """Call this during a test run to find Mario's real RAM addresses."""
        try:
            ram = self.env.get_ram()
            print(f"  0x0060={ram[0x0060]:3d}  0x0062={ram[0x0062]:3d}  "
                  f"0x0043={ram[0x0043]:3d}  0x0044={ram[0x0044]:3d}  "
                  f"0x0074={ram[0x0074]:3d}  0x0075={ram[0x0075]:3d}")
        except Exception as e:
            print(f"RAM read failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        result = self.env.reset()
        # gymnasium returns (obs, info), old gym returns obs directly
        obs = result[0] if isinstance(result, tuple) else result
        frame = self._preprocess(obs)
        for _ in range(4):
            self.frame_stack.append(frame)
        self.prev_score = 0
        self.prev_lives = 3
        self.prev_mario_x = None
        self.prev_mario_y = None
        self.best_mario_y = None
        return self._stacked_state()

    def step(self, action_idx):
        retro_action = np.array(ACTIONS[action_idx], dtype=np.int8)
        total_reward = 0.0
        done = False
        info = {}

        # Repeat the action for frame_skip frames and accumulate reward.
        # The agent makes 4x fewer decisions, so training is ~4x faster.
        for _ in range(self.frame_skip):
            result = self.env.step(retro_action)
            if len(result) == 5:
                obs, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, _, done, info = result
            total_reward += self._compute_reward(info, done, action_idx)
            if done:
                break

        self.frame_stack.append(self._preprocess(obs))
        self._move_arm(action_idx)

        return self._stacked_state(), total_reward, done, info

    def close(self):
        self.env.close()
        p.disconnect(self.physics_client)

    @property
    def action_space_size(self):
        return len(ACTIONS)

    @property
    def state_shape(self):
        return (4, 84, 84)
