"""
Real UR3 dual-arm controller via ur_rtde.

JoystickUR3  — UR3 that grips and tilts the joystick for directional input.
ButtonUR3    — UR3 that presses the jump button whenever the AI fires action 5/6/7.

Both expose the same step(action_idx) / close() interface as the PyBullet arm
so the rest of the codebase needs no changes.

Install:
    pip install ur-rtde

Calibrate (run once, robots must be powered on and unlocked):
    python tools/calibrate_arms.py

Then run with:
    python main.py --algo ppo --eval-only --load-model <ckpt> --real-arms --num-envs 1

Action index mapping (from donkey_kong_env.py):
  0 NOOP  1 LEFT  2 RIGHT  3 UP  4 DOWN
  5 JUMP  6 JUMP+LEFT  7 JUMP+RIGHT
"""

import json
import os
import time

try:
    import rtde_control
    import rtde_receive
    _RTDE_OK = True
except ImportError:
    _RTDE_OK = False

# Default config path — override by passing config_path to RealRobotArms
_DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_config.json')

_JUMP_ACTIONS = frozenset({5, 6, 7})

# Safe defaults — tune once calibrated
_MOVE_SPEED = 0.05   # m/s   — 5 cm/s (conservative for safety)
_MOVE_ACCEL = 0.10   # m/s²

# Joystick deflection from centre (metres)
_JOYSTICK_DEFLECT = 0.015   # 15 mm in X or Y

# How far the finger travels down to press the button (metres)
_BUTTON_PRESS_DEPTH = 0.020   # 20 mm


# Maps each action to (dx, dy) offset from joystick centre pose
_JOYSTICK_OFFSETS = {
    0: ( 0.0,                0.0),               # NOOP
    1: (-_JOYSTICK_DEFLECT,  0.0),               # LEFT
    2: ( _JOYSTICK_DEFLECT,  0.0),               # RIGHT
    3: ( 0.0,                _JOYSTICK_DEFLECT), # UP
    4: ( 0.0,               -_JOYSTICK_DEFLECT), # DOWN
    5: ( 0.0,                0.0),               # JUMP        — no direction
    6: (-_JOYSTICK_DEFLECT,  0.0),               # JUMP+LEFT
    7: ( _JOYSTICK_DEFLECT,  0.0),               # JUMP+RIGHT
}


def _require_rtde():
    if not _RTDE_OK:
        raise RuntimeError(
            'ur_rtde is not installed.\n'
            'Run:  pip install ur-rtde\n'
            'Docs: https://sdurobotics.gitlab.io/ur_rtde/'
        )


# ---------------------------------------------------------------------------
# JoystickUR3
# ---------------------------------------------------------------------------

class JoystickUR3:
    """
    UR3 that grips the joystick shaft and tilts it to match the AI's
    directional output.  Home pose = joystick perfectly centred.

    X axis = left/right, Y axis = forward/back (up/down on joystick).
    Only sends a new move command when the action changes, so the robot
    isn't flooded with identical targets between action changes.
    """

    def __init__(self, ip: str, home_pose: list, move_speed=_MOVE_SPEED, move_accel=_MOVE_ACCEL):
        """
        ip        — UR3 IP address, e.g. '192.168.1.2'
        home_pose — TCP pose with joystick centred:
                    [x, y, z, rx, ry, rz]  (metres / axis-angle radians)
        """
        _require_rtde()
        self.home        = list(home_pose)
        self._speed      = move_speed
        self._accel      = move_accel
        self._last_action = None

        print(f'[JoystickUR3] Connecting to {ip} …')
        self.ctrl = rtde_control.RTDEControlInterface(ip)
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        print(f'[JoystickUR3] Connected.  Moving to home …')
        self._go(self.home, async_move=False)

    # ------------------------------------------------------------------

    def step(self, action_idx: int):
        if action_idx == self._last_action:
            return
        self._last_action = action_idx

        dx, dy = _JOYSTICK_OFFSETS.get(action_idx, (0.0, 0.0))
        target = self.home.copy()
        target[0] += dx
        target[1] += dy
        self._go(target, async_move=True)

    def close(self):
        print('[JoystickUR3] Returning to home and disconnecting …')
        self._go(self.home, async_move=False)
        self.ctrl.stopScript()
        self.ctrl.disconnect()
        self.recv.disconnect()

    # ------------------------------------------------------------------

    def _go(self, pose: list, async_move: bool):
        self.ctrl.moveL(pose, self._speed, self._accel, asynchronous=async_move)


# ---------------------------------------------------------------------------
# ButtonUR3
# ---------------------------------------------------------------------------

class ButtonUR3:
    """
    UR3 that presses the jump button.

    Rest pose = TCP hovering just above the button surface.
    Press = TCP moves straight down by _BUTTON_PRESS_DEPTH metres.
    The arm only moves when the pressed/released state actually changes.
    """

    def __init__(self, ip: str, rest_pose: list, move_speed=_MOVE_SPEED, move_accel=_MOVE_ACCEL):
        """
        ip        — UR3 IP address, e.g. '192.168.1.3'
        rest_pose — TCP pose with finger just above the button surface:
                    [x, y, z, rx, ry, rz]  (metres / axis-angle radians)
        """
        _require_rtde()
        self.rest    = list(rest_pose)
        self._speed  = move_speed
        self._accel  = move_accel
        self._pressed = None   # None forces an initial move on first step()

        # Pre-compute the press pose — just Z shifted down
        self._press_pose = list(rest_pose)
        self._press_pose[2] -= _BUTTON_PRESS_DEPTH

        print(f'[ButtonUR3] Connecting to {ip} …')
        self.ctrl = rtde_control.RTDEControlInterface(ip)
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        print(f'[ButtonUR3] Connected.  Moving to rest position …')
        self._go(self.rest, async_move=False)
        self._pressed = False

    # ------------------------------------------------------------------

    def step(self, action_idx: int):
        pressing = action_idx in _JUMP_ACTIONS
        if pressing == self._pressed:
            return
        if pressing:
            self._go(self._press_pose, async_move=True)
        else:
            self._go(self.rest, async_move=True)
        self._pressed = pressing

    def close(self):
        print('[ButtonUR3] Releasing button and disconnecting …')
        self._go(self.rest, async_move=False)
        self.ctrl.stopScript()
        self.ctrl.disconnect()
        self.recv.disconnect()

    # ------------------------------------------------------------------

    def _go(self, pose: list, async_move: bool):
        self.ctrl.moveL(pose, self._speed, self._accel, asynchronous=async_move)


# ---------------------------------------------------------------------------
# RealRobotArms  — public interface
# ---------------------------------------------------------------------------

class RealRobotArms:
    """
    Combines both UR3 arms behind the same step() / close() interface
    as the PyBullet RobotArm.  Reads IPs and calibrated poses from
    robot_config.json (generated by tools/calibrate_arms.py).

    Usage:
        arms = RealRobotArms()         # reads robot_config.json
        arms.step(action_idx)          # called every env step
        arms.close()                   # park both arms and disconnect
    """

    def __init__(self, config_path: str = _DEFAULT_CFG):
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f'robot_config.json not found at {config_path}.\n'
                f'Run tools/calibrate_arms.py first to generate it.'
            )

        with open(config_path) as f:
            cfg = json.load(f)

        self.joystick = JoystickUR3(
            ip        = cfg['joystick_arm']['ip'],
            home_pose = cfg['joystick_arm']['home_pose'],
        )
        self.button = ButtonUR3(
            ip        = cfg['button_arm']['ip'],
            rest_pose = cfg['button_arm']['rest_pose'],
        )

    def step(self, action_idx: int):
        self.joystick.step(action_idx)
        self.button.step(action_idx)

    def close(self):
        self.joystick.close()
        self.button.close()
