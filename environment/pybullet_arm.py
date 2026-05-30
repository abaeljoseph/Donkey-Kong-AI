"""
Dual KUKA IIWA arm simulation driven by the AI's action outputs.

Two 7-DOF KUKA IIWA arms share one PyBullet physics client:
  Left arm  (blue)   — grips and tilts a joystick for directional input.
  Right arm (orange) — presses a button for jump actions.

Inverse kinematics is solved with joint limits and a natural rest pose so
both arms stay in a clean "elbow-up, reaching forward" configuration.

Public interface (unchanged):
    arm = RobotArm(gui=True)
    arm.step(action_idx)
    arm.close()

Action indices:
  0 NOOP  1 LEFT  2 RIGHT  3 UP  4 DOWN
  5 JUMP  6 JUMP+LEFT  7 JUMP+RIGHT
"""

import math
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------

# How far the joystick arm deflects the EEF from centre (metres)
JOYSTICK_DEFLECT   = 0.035

# How far the button arm presses down (metres)
BUTTON_PRESS_DEPTH = 0.032

# How much the joystick shaft visually tilts (radians, ~17°)
_JOYSTICK_TILT = 0.30
# Half-length of the joystick shaft (metres)
_SHAFT_HALF = 0.065

# Arm base positions (floor-mounted, side by side)
JOYSTICK_ARM_BASE = (-0.50, 0.0, 0.0)
BUTTON_ARM_BASE   = ( 0.50, 0.0, 0.0)

# EEF target positions in world space
# Both arms reach forward (+Y) and slightly up (+Z)
JOYSTICK_CENTER = (-0.12, 0.42, 0.50)
BUTTON_REST     = ( 0.12, 0.42, 0.50)

# ---------------------------------------------------------------------------
# IK parameters — key to natural-looking arm poses
# ---------------------------------------------------------------------------

# Rest pose: arm angled forward with elbow bent upward.
# IK prefers solutions close to this, giving a clean elbow-up posture.
_REST_POSE = [0.0, 0.45, 0.0, -1.2, 0.0, 0.85, 0.0]

# KUKA IIWA joint limits (radians)
_LOWER = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
_UPPER = [ 2.96,  2.09,  2.96,  2.09,  2.96,  2.09,  3.05]
_RANGE = [ u - l for u, l in zip(_UPPER, _LOWER)]

# EEF points slightly forward and down (more natural than straight down)
_EEF_ORI = p.getQuaternionFromEuler([math.pi * 0.85, 0.0, 0.0])

_KUKA_EEF_LINK = 6
_KUKA_JOINTS   = 7
_JOINT_FORCE   = 300

_JUMP_ACTIONS = frozenset({5, 6, 7})


# ---------------------------------------------------------------------------
# KukaArm — base class
# ---------------------------------------------------------------------------

class KukaArm:
    """Loads a KUKA IIWA with IK-driven end-effector control."""

    def __init__(self, client: int, base_pos: tuple, base_orn: tuple = (0, 0, 0, 1),
                 colour: list = None):
        self.client = client
        self.arm = p.loadURDF(
            'kuka_iiwa/model.urdf',
            basePosition=base_pos,
            baseOrientation=base_orn,
            useFixedBase=True,
            physicsClientId=client,
        )
        if colour:
            for link in range(-1, _KUKA_JOINTS):
                p.changeVisualShape(self.arm, link, rgbaColor=colour,
                                    physicsClientId=client)

        # Initialise joints to rest pose
        for i, angle in enumerate(_REST_POSE):
            p.resetJointState(self.arm, i, angle, physicsClientId=client)
            p.setJointMotorControl2(self.arm, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=_JOINT_FORCE,
                                    physicsClientId=client)
        self._last_target = None

    def move_to(self, target_pos: tuple, target_ori: tuple = None):
        """IK-drive the EEF to target_pos. Only re-solves when target changes."""
        if target_pos == self._last_target:
            return
        self._last_target = target_pos

        angles = p.calculateInverseKinematics(
            self.arm,
            _KUKA_EEF_LINK,
            target_pos,
            targetOrientation=target_ori or _EEF_ORI,
            lowerLimits=_LOWER,
            upperLimits=_UPPER,
            jointRanges=_RANGE,
            restPoses=_REST_POSE,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        for i in range(_KUKA_JOINTS):
            p.setJointMotorControl2(
                self.arm, i, p.POSITION_CONTROL,
                targetPosition=angles[i],
                force=_JOINT_FORCE,
                maxVelocity=1.5,
                physicsClientId=self.client,
            )


# ---------------------------------------------------------------------------
# JoystickKukaArm
# ---------------------------------------------------------------------------

class JoystickKukaArm(KukaArm):
    """Blue KUKA IIWA that grips and tilts a joystick."""

    def __init__(self, client: int):
        super().__init__(
            client,
            base_pos=JOYSTICK_ARM_BASE,
            colour=[0.20, 0.48, 0.90, 1.0],
        )
        cx, cy, cz = JOYSTICK_CENTER
        d = JOYSTICK_DEFLECT
        self._targets = {
            0: (cx,     cy,     cz),
            1: (cx - d, cy,     cz),
            2: (cx + d, cy,     cz),
            3: (cx,     cy + d, cz),
            4: (cx,     cy - d, cz),
            5: (cx,     cy,     cz),
            6: (cx - d, cy,     cz),
            7: (cx + d, cy,     cz),
        }
        self._build_joystick()
        self.move_to(JOYSTICK_CENTER)

    def step(self, action_idx: int):
        self.move_to(self._targets.get(action_idx, JOYSTICK_CENTER))
        self._tilt_joystick(action_idx)

    # ------------------------------------------------------------------
    # Joystick tilt helpers

    def _tilt_joystick(self, action_idx: int):
        """Physically tilt the shaft and knob to match the action direction."""
        cx, cy, cz = JOYSTICK_CENTER
        # Pivot = top of base plate
        pivot = (cx, cy, cz - 0.158)
        shaft_pos, shaft_ori, knob_pos = self._shaft_transform(action_idx, pivot)
        p.resetBasePositionAndOrientation(
            self._shaft, shaft_pos, shaft_ori, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self._knob, knob_pos, (0, 0, 0, 1), physicsClientId=self.client)

    @staticmethod
    def _shaft_transform(action_idx, pivot):
        """Return (shaft_centre, shaft_ori_quat, knob_pos) for a given action."""
        px, py, pz = pivot
        L = _SHAFT_HALF
        T = _JOYSTICK_TILT
        s, c = math.sin(T), math.cos(T)

        if action_idx in (1, 6):          # LEFT  — tilt toward -X
            shaft = (px - s*L, py,      pz + c*L)
            ori   = p.getQuaternionFromEuler([0, -T, 0])
            knob  = (px - s*2*L, py,    pz + c*2*L)
        elif action_idx in (2, 7):        # RIGHT — tilt toward +X
            shaft = (px + s*L, py,      pz + c*L)
            ori   = p.getQuaternionFromEuler([0,  T, 0])
            knob  = (px + s*2*L, py,    pz + c*2*L)
        elif action_idx == 3:             # UP    — tilt toward +Y
            shaft = (px, py + s*L,      pz + c*L)
            ori   = p.getQuaternionFromEuler([-T, 0, 0])
            knob  = (px, py + s*2*L,    pz + c*2*L)
        elif action_idx == 4:             # DOWN  — tilt toward -Y
            shaft = (px, py - s*L,      pz + c*L)
            ori   = p.getQuaternionFromEuler([ T, 0, 0])
            knob  = (px, py - s*2*L,    pz + c*2*L)
        else:                             # NOOP / JUMP — centred
            shaft = (px,       py,      pz + L)
            ori   = (0, 0, 0, 1)
            knob  = (px,       py,      pz + 2*L)

        return shaft, ori, knob

    def _build_joystick(self):
        c  = self.client
        cx, cy, cz = JOYSTICK_CENTER
        pivot_z = cz - 0.158

        # Fixed base plate
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.012], physicsClientId=c),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.012],
                rgbaColor=[0.12, 0.12, 0.12, 1], physicsClientId=c),
            basePosition=[cx, cy, pivot_z - 0.012],
            physicsClientId=c,
        )
        # Moveable shaft (separate body so we can tilt it)
        self._shaft = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.011, height=_SHAFT_HALF * 2, physicsClientId=c),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.011, length=_SHAFT_HALF * 2,
                rgbaColor=[0.22, 0.22, 0.22, 1], physicsClientId=c),
            basePosition=[cx, cy, pivot_z + _SHAFT_HALF],
            physicsClientId=c,
        )
        # Moveable ball-top handle
        self._knob = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.020, physicsClientId=c),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, radius=0.020,
                rgbaColor=[0.05, 0.05, 0.05, 1], physicsClientId=c),
            basePosition=[cx, cy, pivot_z + _SHAFT_HALF * 2],
            physicsClientId=c,
        )


# ---------------------------------------------------------------------------
# ButtonKukaArm
# ---------------------------------------------------------------------------

class ButtonKukaArm(KukaArm):
    """Orange KUKA IIWA that presses a button on jump actions."""

    def __init__(self, client: int):
        super().__init__(
            client,
            base_pos=BUTTON_ARM_BASE,
            colour=[0.92, 0.42, 0.12, 1.0],
        )
        bx, by, bz = BUTTON_REST
        self._rest_pos  = BUTTON_REST
        self._press_pos = (bx, by, bz - BUTTON_PRESS_DEPTH)
        self._pressed   = False
        self._btn_body  = self._build_button()
        self.move_to(self._rest_pos)

    def step(self, action_idx: int):
        pressing = action_idx in _JUMP_ACTIONS
        self.move_to(self._press_pos if pressing else self._rest_pos)
        if pressing != self._pressed:
            # Move button cap down and turn red; lift back up and turn yellow
            bx, by, bz = BUTTON_REST
            cap_z   = bz - 0.038 - (BUTTON_PRESS_DEPTH if pressing else 0)
            colour  = [0.9, 0.08, 0.08, 1] if pressing else [0.85, 0.78, 0.0, 1]
            p.resetBasePositionAndOrientation(
                self._btn_cap, (bx, by, cap_z), (0, 0, 0, 1),
                physicsClientId=self.client)
            p.changeVisualShape(self._btn_cap, -1, rgbaColor=colour,
                                physicsClientId=self.client)
            self._pressed = pressing

    def _build_button(self):
        c = self.client
        bx, by, bz = BUTTON_REST

        # Fixed base plate
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.055, 0.055, 0.012], physicsClientId=c),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.055, 0.055, 0.012],
                rgbaColor=[0.15, 0.15, 0.15, 1], physicsClientId=c),
            basePosition=[bx, by, bz - 0.058],
            physicsClientId=c,
        )
        # Moveable button cap (separate body so it can slide down on press)
        self._btn_cap = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.026, height=0.020, physicsClientId=c),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.026, length=0.020,
                rgbaColor=[0.85, 0.78, 0.0, 1], physicsClientId=c),
            basePosition=[bx, by, bz - 0.038],
            physicsClientId=c,
        )


# ---------------------------------------------------------------------------
# RobotArm — public interface (unchanged)
# ---------------------------------------------------------------------------

class RobotArm:
    """
    Dual KUKA IIWA controller.
    Left  (blue)   — joystick arm, deflects in XY to match directional input.
    Right (orange) — button arm, presses down on jump actions (5, 6, 7).
    """

    def __init__(self, gui: bool = False):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF('plane.urdf', physicsClientId=self.client)

        self.joystick_arm = JoystickKukaArm(self.client)
        self.button_arm   = ButtonKukaArm(self.client)

        if gui:
            # Camera looking straight at both arms from the front
            p.resetDebugVisualizerCamera(
                cameraDistance=1.6,
                cameraYaw=0,
                cameraPitch=-20,
                cameraTargetPosition=[0.0, 0.35, 0.35],
                physicsClientId=self.client,
            )

    def step(self, action_idx: int):
        self.joystick_arm.step(action_idx)
        self.button_arm.step(action_idx)
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)

    def close(self):
        p.disconnect(self.client)
