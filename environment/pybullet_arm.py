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
import time
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
_SOFTWARE_VIEW_W = 640
_SOFTWARE_VIEW_H = 426
_SOFTWARE_VIEW_FPS = 10.0

COIN_PICKUP_POS = (0.28, 0.31, 0.42)
COIN_SLOT_POS = (0.08, 0.462, 0.255)
COIN_SLOT_APPROACH = (0.08, 0.35, 0.32)
COIN_SLOT_INSERT = (0.08, 0.445, 0.285)
COIN_HALF_INSERTED = (0.08, 0.438, 0.261)
_COIN_ORI = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Arcade cabinet helpers
# ---------------------------------------------------------------------------

def _static_box(client, half_extents, position, colour, orientation=(0, 0, 0, 1)):
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=colour,
            physicsClientId=client),
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=client,
    )


def _static_cylinder(client, radius, height, position, colour, orientation=(0, 0, 0, 1)):
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=colour,
            physicsClientId=client),
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=client,
    )


def _build_arcade_console(client: int):
    """Build a static arcade cabinet around the controls."""
    black = [0.02, 0.02, 0.025, 1]
    dark_blue = [0.04, 0.10, 0.26, 1]
    blue = [0.08, 0.22, 0.55, 1]
    red = [0.86, 0.06, 0.08, 1]
    yellow = [0.95, 0.78, 0.06, 1]
    teal = [0.00, 0.75, 0.85, 1]
    grey = [0.22, 0.22, 0.24, 1]

    # Upright cabinet shell behind the controls.
    _static_box(client, [0.46, 0.16, 0.76], [0.0, 0.68, 0.76], dark_blue)
    _static_box(client, [0.50, 0.035, 0.08], [0.0, 0.50, 1.34], red)
    _static_box(client, [0.42, 0.012, 0.19], [0.0, 0.49, 0.95], black)

    # Simple Donkey Kong-style screen art: blue girders and a yellow player marker.
    _static_box(client, [0.34, 0.006, 0.010], [0.0, 0.478, 0.88], blue)
    _static_box(client, [0.34, 0.006, 0.010], [0.0, 0.478, 0.96], blue)
    _static_box(client, [0.34, 0.006, 0.010], [0.0, 0.478, 1.04], blue)
    _static_box(client, [0.012, 0.006, 0.075], [-0.16, 0.476, 0.92], teal)
    _static_box(client, [0.012, 0.006, 0.075], [0.12, 0.476, 1.00], teal)
    _static_box(client, [0.018, 0.006, 0.018], [-0.04, 0.474, 1.065], red)
    _static_box(client, [0.014, 0.006, 0.024], [-0.22, 0.474, 0.905], yellow)

    # Control deck. The existing joystick and button sit on/above this panel.
    _static_box(client, [0.44, 0.18, 0.035], [0.0, 0.40, 0.285], black)
    _static_box(client, [0.40, 0.14, 0.018], [0.0, 0.40, 0.335], grey)
    _static_box(client, [0.39, 0.012, 0.020], [0.0, 0.23, 0.33], red)

    # Raised mount for the jump button, because the button cap is taller than
    # the joystick collar in the current arm target layout.
    bx, by, bz = BUTTON_REST
    _static_cylinder(client, 0.060, 0.080, [bx, by, bz - 0.108], black)
    _static_cylinder(client, 0.045, 0.012, [bx, by, bz - 0.064], yellow)

    # Joystick collar and compact visual labels.
    jx, jy, jz = JOYSTICK_CENTER
    _static_cylinder(client, 0.075, 0.010, [jx, jy, jz - 0.166], black)
    _static_box(client, [0.090, 0.008, 0.006], [jx, jy - 0.095, jz - 0.135], teal)
    _static_box(client, [0.075, 0.008, 0.006], [bx, by - 0.095, bz - 0.035], red)

    # Lower cabinet details.
    _static_box(client, [0.30, 0.012, 0.10], [0.0, 0.485, 0.22], black)
    _static_box(client, [0.045, 0.008, 0.020], [-0.06, 0.47, 0.25], grey)
    _static_box(client, [0.074, 0.008, 0.038], [COIN_SLOT_POS[0], 0.468, COIN_SLOT_POS[2]], yellow)
    _static_box(client, [0.058, 0.006, 0.028], [COIN_SLOT_POS[0], 0.462, COIN_SLOT_POS[2]], grey)
    _static_box(client, [0.040, 0.005, 0.005], [COIN_SLOT_POS[0], 0.456, COIN_SLOT_POS[2] + 0.006], black)
    _static_box(client, [0.060, 0.006, 0.006], [COIN_SLOT_POS[0], 0.459, COIN_SLOT_POS[2] + 0.044], red)
    _static_box(client, [0.012, 0.006, 0.028], [COIN_SLOT_POS[0] - 0.052, 0.458, COIN_SLOT_POS[2]], teal)
    _static_box(client, [0.012, 0.006, 0.028], [COIN_SLOT_POS[0] + 0.052, 0.458, COIN_SLOT_POS[2]], teal)


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

    def __init__(self, gui: bool = False, software_viewer: bool = False):
        self._software_viewer = software_viewer
        self._pygame = None
        self._screen = None
        self._clock = None
        self._view_w = _SOFTWARE_VIEW_W if software_viewer else 960
        self._view_h = _SOFTWARE_VIEW_H if software_viewer else 640
        self._max_render_dim = 960
        self._render_interval = 1.0 / _SOFTWARE_VIEW_FPS
        self._last_render_time = 0.0
        self._sim_steps = 3 if software_viewer else 12
        self._cam_target = [0.0, 0.43, 0.68]
        self._cam_distance = 2.20
        self._cam_yaw = 0.0
        self._cam_pitch = -18.0
        self._coin_body = None
        self._viewer_font = None
        self._viewer_font_big = None
        self._screen_state = 'INSERT COIN'
        self._game_frame = None
        self._closed = False

        mode = p.DIRECT if software_viewer else (p.GUI if gui else p.DIRECT)
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF('plane.urdf', physicsClientId=self.client)
        _build_arcade_console(self.client)

        self.joystick_arm = JoystickKukaArm(self.client)
        self.button_arm   = ButtonKukaArm(self.client)
        self._coin_body = self._build_coin()

        if gui and not software_viewer:
            # Camera looking straight at both arms from the front
            p.resetDebugVisualizerCamera(
                cameraDistance=2.20,
                cameraYaw=0,
                cameraPitch=-18,
                cameraTargetPosition=[0.0, 0.43, 0.68],
                physicsClientId=self.client,
            )
        elif software_viewer:
            self._init_software_viewer()
        if gui:
            self._run_startup_sequence()

    def _build_coin(self):
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.020, height=0.004,
                physicsClientId=self.client),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.020, length=0.004,
                rgbaColor=[0.95, 0.72, 0.12, 1.0], physicsClientId=self.client),
            basePosition=COIN_PICKUP_POS,
            baseOrientation=_COIN_ORI,
            physicsClientId=self.client,
        )

    def _set_coin_position(self, pos, visible=True):
        target = pos if visible else (0.0, 0.0, -1.0)
        p.resetBasePositionAndOrientation(
            self._coin_body, target, _COIN_ORI, physicsClientId=self.client)

    def set_game_frame(self, frame):
        self._game_frame = frame

    @staticmethod
    def _lerp(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    def _move_startup_segment(self, start, end, frames, coin_mode='hidden'):
        for frame in range(max(frames, 1)):
            if self._closed:
                return
            t = (frame + 1) / float(max(frames, 1))
            pos = self._lerp(start, end, t)
            self.joystick_arm.step(0)
            self.button_arm.move_to(pos)

            if coin_mode == 'table':
                self._set_coin_position(COIN_PICKUP_POS)
            elif coin_mode == 'hand':
                self._set_coin_position((pos[0], pos[1] + 0.025, pos[2] - 0.025))
            elif coin_mode == 'slot':
                self._set_coin_position((COIN_SLOT_POS[0], COIN_SLOT_POS[1] - 0.004, COIN_SLOT_POS[2]))
            else:
                self._set_coin_position(COIN_PICKUP_POS, visible=False)

            for _ in range(self._sim_steps):
                p.stepSimulation(physicsClientId=self.client)
            self._draw_software_viewer(force=True)
            if not self._software_viewer:
                time.sleep(1.0 / 60.0)

    def _run_startup_sequence(self):
        print('[RobotArm] Startup sequence: inserting coin')
        self._screen_state = 'INSERT COIN'
        self._set_coin_position(COIN_PICKUP_POS)
        self._draw_software_viewer(force=True)
        self._move_startup_segment(BUTTON_REST, COIN_PICKUP_POS, 10, coin_mode='table')
        self._move_startup_segment(COIN_PICKUP_POS, COIN_SLOT_APPROACH, 14, coin_mode='hand')
        self._move_startup_segment(COIN_SLOT_APPROACH, COIN_SLOT_INSERT, 8, coin_mode='hand')
        self._screen_state = 'LOADING GAME'
        self._move_startup_segment(COIN_SLOT_INSERT, COIN_SLOT_APPROACH, 6, coin_mode='slot')
        self._move_startup_segment(COIN_SLOT_APPROACH, BUTTON_REST, 12, coin_mode='hidden')
        self._set_coin_position(COIN_HALF_INSERTED)

    def _init_software_viewer(self):
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((self._view_w, self._view_h), pygame.RESIZABLE)
        pygame.display.set_caption('Robot Arms - Software Viewer')
        self._viewer_font = pygame.font.SysFont('Consolas', 16)
        self._viewer_font_big = pygame.font.SysFont('Consolas', 24, bold=True)

    def _draw_software_viewer(self, force=False):
        if self._pygame is None:
            return

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return
            if event.type == self._pygame.VIDEORESIZE:
                self._view_w = max(360, event.w)
                self._view_h = max(240, event.h)
                self._screen = self._pygame.display.set_mode(
                    (self._view_w, self._view_h), self._pygame.RESIZABLE)

        self._update_software_camera()

        now = time.perf_counter()
        if not force and now - self._last_render_time < self._render_interval:
            return
        self._last_render_time = now

        render_w, render_h = self._software_render_size()
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._cam_target,
            distance=self._cam_distance,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=55,
            aspect=float(render_w) / float(render_h),
            nearVal=0.01,
            farVal=10.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            render_w,
            render_h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )

        import numpy as np

        rgb = np.asarray(rgba, dtype=np.uint8).reshape(render_h, render_w, 4)[:, :, :3]
        surf = self._pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if render_w != self._view_w or render_h != self._view_h:
            surf = self._pygame.transform.smoothscale(surf, (self._view_w, self._view_h))
        self._screen.blit(surf, (0, 0))
        self._draw_cabinet_screen_overlay(view, proj, render_w, render_h)
        self._draw_viewer_overlay()
        self._pygame.display.flip()

    def _software_render_size(self):
        longest = max(self._view_w, self._view_h)
        if longest <= self._max_render_dim:
            return self._view_w, self._view_h
        scale = self._max_render_dim / float(longest)
        return max(320, int(self._view_w * scale)), max(213, int(self._view_h * scale))

    def _project_world_to_viewer(self, pos, view, proj, render_w, render_h):
        import numpy as np

        view_m = np.asarray(view, dtype=float).reshape((4, 4), order='F')
        proj_m = np.asarray(proj, dtype=float).reshape((4, 4), order='F')
        point = np.asarray([pos[0], pos[1], pos[2], 1.0], dtype=float)
        clip = proj_m @ view_m @ point
        if abs(clip[3]) < 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None

        render_x = (ndc[0] * 0.5 + 0.5) * render_w
        render_y = (1.0 - (ndc[1] * 0.5 + 0.5)) * render_h
        return (
            int(render_x * self._view_w / float(render_w)),
            int(render_y * self._view_h / float(render_h)),
        )

    def _draw_cabinet_screen_overlay(self, view, proj, render_w, render_h):
        y = 0.468
        corners = [
            (-0.38, y, 1.12),
            (0.38, y, 1.12),
            (0.38, y, 0.78),
            (-0.38, y, 0.78),
        ]
        points = [
            self._project_world_to_viewer(c, view, proj, render_w, render_h)
            for c in corners
        ]
        if any(pnt is None for pnt in points):
            return

        pg = self._pygame
        pg.draw.polygon(self._screen, (3, 4, 8), points)
        pg.draw.polygon(self._screen, (210, 40, 45), points, 2)

        xs = [pnt[0] for pnt in points]
        ys = [pnt[1] for pnt in points]
        rect = pg.Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if rect.w < 70 or rect.h < 45:
            return
        self._draw_mock_arcade_screen(rect)

    def _draw_viewer_overlay(self):
        if self._viewer_font is None:
            return

        pg = self._pygame
        panel = pg.Rect(12, self._view_h - 64, 210, 52)
        pg.draw.rect(self._screen, (6, 8, 12), panel)
        pg.draw.rect(self._screen, (80, 100, 130), panel, 2)

        title = self._viewer_font.render('ARCADE STATUS', True, (160, 210, 255))
        self._screen.blit(title, (24, panel.y + 10))

        state = self._viewer_font.render(self._screen_state, True, (255, 225, 80))
        self._screen.blit(state, (24, panel.y + 32))

    def _draw_mock_arcade_screen(self, screen_rect):
        pg = self._pygame
        pg.draw.rect(self._screen, (5, 5, 8), screen_rect)
        pg.draw.rect(self._screen, (180, 30, 35), screen_rect, 2)

        if self._screen_state == 'INSERT COIN':
            msg = self._viewer_font_big.render('INSERT', True, (255, 225, 80))
            coin = self._viewer_font_big.render('COIN', True, (255, 225, 80))
            self._screen.blit(msg, msg.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.42)))
            self._screen.blit(coin, coin.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.64)))
            return

        if self._screen_state == 'LOADING GAME':
            msg = self._viewer_font.render('LOADING GAME', True, (90, 230, 255))
            self._screen.blit(msg, msg.get_rect(center=(screen_rect.centerx, screen_rect.centery - 10)))
            pg.draw.rect(self._screen, (90, 230, 255),
                         (screen_rect.x + int(screen_rect.w * 0.22),
                          screen_rect.y + int(screen_rect.h * 0.62),
                          int(screen_rect.w * 0.56),
                          max(4, int(screen_rect.h * 0.05))))
            return

        if self._game_frame is not None:
            import numpy as np

            frame = np.asarray(self._game_frame)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = np.ascontiguousarray(frame[:, :, :3])
                surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
                surf = pg.transform.smoothscale(surf, (screen_rect.w, screen_rect.h))
                self._screen.blit(surf, screen_rect)
                pg.draw.rect(self._screen, (180, 30, 35), screen_rect, 2)
                return

        girder_col = (40, 95, 220)
        ladder_col = (0, 210, 220)
        for frac in (0.30, 0.52, 0.74):
            gy = int(screen_rect.y + screen_rect.h * frac)
            pg.draw.line(self._screen, girder_col,
                         (screen_rect.x + int(screen_rect.w * 0.10), gy),
                         (screen_rect.x + int(screen_rect.w * 0.90), gy + int(screen_rect.h * 0.06)),
                         max(2, int(screen_rect.h * 0.035)))
        pg.draw.line(self._screen, ladder_col,
                     (screen_rect.x + int(screen_rect.w * 0.28), screen_rect.y + int(screen_rect.h * 0.30)),
                     (screen_rect.x + int(screen_rect.w * 0.28), screen_rect.y + int(screen_rect.h * 0.75)),
                     max(2, int(screen_rect.h * 0.025)))
        pg.draw.line(self._screen, ladder_col,
                     (screen_rect.x + int(screen_rect.w * 0.63), screen_rect.y + int(screen_rect.h * 0.34)),
                     (screen_rect.x + int(screen_rect.w * 0.63), screen_rect.y + int(screen_rect.h * 0.78)),
                     max(2, int(screen_rect.h * 0.025)))
        pg.draw.circle(self._screen, (210, 60, 45),
                       (screen_rect.x + int(screen_rect.w * 0.78), screen_rect.y + int(screen_rect.h * 0.33)),
                       max(4, int(screen_rect.h * 0.07)))
        pg.draw.rect(self._screen, (255, 210, 55),
                     (screen_rect.x + int(screen_rect.w * 0.25),
                      screen_rect.y + int(screen_rect.h * 0.70),
                      max(5, int(screen_rect.w * 0.05)),
                      max(8, int(screen_rect.h * 0.13))))

    def _update_software_camera(self):
        keys = self._pygame.key.get_pressed()
        dt = max(time.perf_counter() - self._last_render_time, 0.016)
        orbit_speed = 80.0 * dt
        pan_speed = 0.45 * dt
        zoom_speed = 1.4 * dt

        if keys[self._pygame.K_LEFT]:
            self._cam_yaw -= orbit_speed
        if keys[self._pygame.K_RIGHT]:
            self._cam_yaw += orbit_speed
        if keys[self._pygame.K_UP]:
            self._cam_pitch = min(self._cam_pitch + orbit_speed, 20.0)
        if keys[self._pygame.K_DOWN]:
            self._cam_pitch = max(self._cam_pitch - orbit_speed, -80.0)

        if keys[self._pygame.K_w]:
            self._cam_distance = max(0.7, self._cam_distance - zoom_speed)
        if keys[self._pygame.K_s]:
            self._cam_distance = min(4.0, self._cam_distance + zoom_speed)

        yaw_rad = math.radians(self._cam_yaw)
        right_x = math.cos(yaw_rad)
        right_y = math.sin(yaw_rad)
        fwd_x = -math.sin(yaw_rad)
        fwd_y = math.cos(yaw_rad)

        if keys[self._pygame.K_a]:
            self._cam_target[0] -= right_x * pan_speed
            self._cam_target[1] -= right_y * pan_speed
        if keys[self._pygame.K_d]:
            self._cam_target[0] += right_x * pan_speed
            self._cam_target[1] += right_y * pan_speed
        if keys[self._pygame.K_q]:
            self._cam_target[0] -= fwd_x * pan_speed
            self._cam_target[1] -= fwd_y * pan_speed
        if keys[self._pygame.K_e]:
            self._cam_target[0] += fwd_x * pan_speed
            self._cam_target[1] += fwd_y * pan_speed
        if keys[self._pygame.K_PAGEUP]:
            self._cam_target[2] += pan_speed
        if keys[self._pygame.K_PAGEDOWN]:
            self._cam_target[2] -= pan_speed

        if keys[self._pygame.K_r]:
            self._cam_target = [0.0, 0.43, 0.68]
            self._cam_distance = 2.20
            self._cam_yaw = 0.0
            self._cam_pitch = -18.0

    def step(self, action_idx: int):
        if self._closed:
            return
        if self._screen_state == 'LOADING GAME':
            self._screen_state = 'GAME RUNNING'
        self.joystick_arm.step(action_idx)
        self.button_arm.step(action_idx)
        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self.client)
        self._draw_software_viewer()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
        if p.isConnected(self.client):
            p.disconnect(self.client)
