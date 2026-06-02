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

# How far the button arm presses down (metres).  Kept shallow so the gripper
# only dips onto the cap and never looks like it sinks through the button.
BUTTON_PRESS_DEPTH = 0.016
# Extra height the button arm hovers above the cap (metres), applied to both
# the rest and pressed poses so the gripper sits clearly above the button.
BUTTON_HOVER_LIFT = 0.016

# How much the joystick shaft visually tilts (radians, ~17°)
_JOYSTICK_TILT = 0.30
# Half-length of the joystick shaft (metres)
_SHAFT_HALF = 0.065

# Low-pass factor for the joystick deflection, applied per decision step.
# The eval policy samples stochastically, so raw actions flicker between
# directions; filtering makes the stick follow the dominant direction and
# stops single-frame flickers from snapping it to a full tilt. Lower = smoother
# (more lag), higher = snappier (more jitter). 0..1.
_JOYSTICK_SMOOTH = 0.30
# Tiny residual deflection below this magnitude snaps to centre.
_JOYSTICK_DEADZONE = 0.06

# Action -> target (dx, dy) joystick deflection in [-1, 1].
_ACTION_DEFLECT = {
    0: ( 0.0,  0.0),  # NOOP
    1: (-1.0,  0.0),  # LEFT
    2: ( 1.0,  0.0),  # RIGHT
    3: ( 0.0,  1.0),  # UP
    4: ( 0.0, -1.0),  # DOWN
    5: ( 0.0,  0.0),  # JUMP
    6: (-1.0,  0.0),  # JUMP + LEFT
    7: ( 1.0,  0.0),  # JUMP + RIGHT
}

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
_SOFTWARE_VIEW_FPS = 24.0
# Cap the internal (CPU-rendered) camera image, then upscale to the window.
# TINY_RENDERER cost scales with pixel count: 640x426≈90ms, 512x340≈58ms,
# 384x255≈33ms. 512 keeps good quality while still fitting the ~66ms budget
# at 2.0x playback (the speed this looks best at).
_SOFTWARE_MAX_RENDER_DIM = 512

# ---------------------------------------------------------------------------
# Cabinet keep-out zones
# ---------------------------------------------------------------------------
# Axis-aligned solid volumes (world space) the robot end-effector must NEVER
# enter, expressed as (xmin, xmax, ymin, ymax, zmin, zmax).  The control deck
# and lower cabinet body fill Y 0.18–0.90 up to Z ≈ 0.42.  The joystick/button
# the arms actually operate sit ABOVE this (Z ≈ 0.46–0.52), so normal play is
# unaffected — only the low coin reach is constrained.  Any commanded target
# that lands inside a zone is pushed OUT toward the camera (−Y), so the arm
# goes around the front of the cabinet instead of plunging down into it.
_CABINET_KEEPOUT = [
    (-0.50, 0.50, 0.18, 0.90, 0.00, 0.42),
]
# How far in front of a zone (−Y) to park a clamped target.
_KEEPOUT_MARGIN = 0.04


def _clamp_outside_cabinet(pos):
    """Pull an EEF target out of any cabinet keep-out zone, toward the camera."""
    x, y, z = pos
    for xmin, xmax, ymin, ymax, zmin, zmax in _CABINET_KEEPOUT:
        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
            y = ymin - _KEEPOUT_MARGIN
    return (x, y, z)


# Coin choreography — a simple rectilinear pick-and-place that cannot clip the
# cabinet:  grab → LIFT straight up → traverse HORIZONTALLY (high, above the
# whole cabinet) to over the slot → lower straight DOWN to slot height → push
# the coin forward into the front-face slot → straight back up and home.  Every
# placement EEF target keeps Y < 0.18 (in front of the cabinet) so the down/up
# legs never enter the keep-out body.
COIN_LIFT_Z        = 0.55                   # traverse height, above the cabinet
COIN_PICKUP_POS    = (0.30, 0.10, 0.40)    # floating tray, in front-right
COIN_PICKUP_LIFT   = (0.30, 0.10, COIN_LIFT_Z)   # straight up from the tray
COIN_OVER_SLOT     = (0.0, 0.13, COIN_LIFT_Z)    # high, directly above the slot
COIN_SLOT_POS      = (0.0, 0.205, 0.30)    # coin door on the front face
COIN_SLOT_INSERT   = (0.0, 0.13, 0.30)     # in front of the slot face (Y < 0.18)
COIN_SLOT_APPROACH = COIN_OVER_SLOT        # retained alias for the over-slot point
COIN_HALF_INSERTED = (0.0, 0.19, 0.30)
_COIN_ORI = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Simple two-finger gripper (visual, glued to each arm's end-effector)
# ---------------------------------------------------------------------------

# Half-distance from the EEF centre to each finger. set_gripper(0)=closed, (1)=open.
GRIP_CLOSED_GAP = 0.024   # straddles the ~0.040 coin / joystick knob
GRIP_OPEN_GAP   = 0.046
GRIP_RATE       = 0.0045  # metres per update — gives a smooth open/close
_PALM_HALF      = [0.032, 0.024, 0.010]
_BRIDGE_HALF    = [GRIP_OPEN_GAP, 0.020, 0.010]  # body bar the fingers slide along
_FINGER_HALF    = [0.007, 0.013, 0.026]
_PALM_DZ        = -0.014  # palm sits just below the flange
_BRIDGE_DZ      = -0.028  # body/spacer between palm and the two fingers
_FINGER_DZ      = -0.044  # fingers hang below the body
_GRIP_TIP_DZ    = -0.066  # where a held object sits between the fingertips
_GRIP_ORI       = (0.0, 0.0, 0.0, 1.0)


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
    _static_box(client, [0.42, 0.012, 0.27], [0.0, 0.49, 0.95], black)

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
    # Coin door on the FRONT face of the cabinet, facing the camera (−Y side).
    # Layered front-to-back: the black slot is frontmost so the coin visibly
    # goes into the front of the arcade.
    cdx, cdz = COIN_SLOT_POS[0], COIN_SLOT_POS[2]
    _static_box(client, [0.058, 0.024, 0.052], [cdx, 0.236, cdz], yellow)         # door body (set into front)
    _static_box(client, [0.042, 0.008, 0.040], [cdx, 0.205, cdz], grey)           # bezel
    _static_box(client, [0.006, 0.007, 0.024], [cdx, 0.190, cdz + 0.004], black)  # vertical coin slot
    _static_box(client, [0.048, 0.007, 0.006], [cdx, 0.192, cdz + 0.042], red)    # label strip
    _static_box(client, [0.010, 0.007, 0.030], [cdx - 0.050, 0.197, cdz], teal)   # side accents
    _static_box(client, [0.010, 0.007, 0.030], [cdx + 0.050, 0.197, cdz], teal)


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

        # Two-finger gripper, visually attached to the end-effector flange.
        self._grip        = GRIP_CLOSED_GAP   # current half-gap (animated)
        self._grip_target = GRIP_CLOSED_GAP
        self._eef_world   = base_pos
        self._build_gripper()
        self.update_gripper()

    def _build_gripper(self):
        c = self.client
        body_col   = [0.16, 0.16, 0.18, 1.0]   # palm + spacer body (dark)
        finger_col = [0.62, 0.64, 0.68, 1.0]   # sliding fingers (metallic grey)

        def _visual_box(half, colour):
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half, rgbaColor=colour,
                    physicsClientId=c),
                basePosition=[0.0, 0.0, -1.0],
                physicsClientId=c,
            )

        self._palm     = _visual_box(_PALM_HALF, body_col)
        self._bridge   = _visual_box(_BRIDGE_HALF, body_col)   # boxed spacer between grips
        self._finger_l = _visual_box(_FINGER_HALF, finger_col)
        self._finger_r = _visual_box(_FINGER_HALF, finger_col)

    def set_gripper(self, opening: float):
        """opening: 0.0 = fully closed (grasping), 1.0 = fully open."""
        opening = max(0.0, min(1.0, opening))
        self._grip_target = GRIP_CLOSED_GAP + opening * (GRIP_OPEN_GAP - GRIP_CLOSED_GAP)

    def gripper_tip(self):
        """World point sitting between the fingertips (for placing a held coin)."""
        ex, ey, ez = self._eef_world
        return (ex, ey, ez + _GRIP_TIP_DZ)

    def update_gripper(self):
        """Animate the gripper toward its target and glue it to the live EEF pose."""
        delta = self._grip_target - self._grip
        self._grip += max(-GRIP_RATE, min(GRIP_RATE, delta))

        state = p.getLinkState(self.arm, _KUKA_EEF_LINK, physicsClientId=self.client)
        ex, ey, ez = state[4]
        self._eef_world = (ex, ey, ez)
        p.resetBasePositionAndOrientation(
            self._palm, (ex, ey, ez + _PALM_DZ), _GRIP_ORI, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self._bridge, (ex, ey, ez + _BRIDGE_DZ), _GRIP_ORI, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self._finger_l, (ex - self._grip, ey, ez + _FINGER_DZ), _GRIP_ORI,
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self._finger_r, (ex + self._grip, ey, ez + _FINGER_DZ), _GRIP_ORI,
            physicsClientId=self.client)

    def move_to(self, target_pos: tuple, target_ori: tuple = None):
        """IK-drive the EEF to target_pos. Only re-solves when target changes.

        Targets are clamped out of the cabinet keep-out zones first, so the arm
        is never asked to drive its end-effector into the arcade body.
        """
        target_pos = _clamp_outside_cabinet(tuple(target_pos))
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
        # Smoothed deflection state (dx, dy) in [-1, 1]; filtered each step.
        self._defl = [0.0, 0.0]
        self._build_joystick()
        self.move_to(JOYSTICK_CENTER)
        self._tilt_joystick(0.0, 0.0)

    def step(self, action_idx: int):
        tx, ty = _ACTION_DEFLECT.get(action_idx, (0.0, 0.0))
        a = _JOYSTICK_SMOOTH
        self._defl[0] += a * (tx - self._defl[0])
        self._defl[1] += a * (ty - self._defl[1])
        # Settle to dead-centre when the residual deflection is negligible.
        if abs(self._defl[0]) < _JOYSTICK_DEADZONE and tx == 0.0:
            self._defl[0] = 0.0
        if abs(self._defl[1]) < _JOYSTICK_DEADZONE and ty == 0.0:
            self._defl[1] = 0.0

        dx, dy = self._defl
        cx, cy, cz = JOYSTICK_CENTER
        self.move_to((cx + dx * JOYSTICK_DEFLECT,
                      cy + dy * JOYSTICK_DEFLECT,
                      cz))
        self._tilt_joystick(dx, dy)

    # ------------------------------------------------------------------
    # Joystick tilt helpers

    def _tilt_joystick(self, dx: float, dy: float):
        """Tilt the shaft/knob continuously toward the (dx, dy) deflection."""
        cx, cy, cz = JOYSTICK_CENTER
        px, py, pz = (cx, cy, cz - 0.158)   # pivot = top of base plate
        L = _SHAFT_HALF
        ex = -_JOYSTICK_TILT * dy           # tilt about X for forward/back
        ey =  _JOYSTICK_TILT * dx           # tilt about Y for left/right

        # Unit direction of the shaft after the [ex, ey, 0] rotation.
        dvec = (
            math.sin(ey),
            -math.cos(ey) * math.sin(ex),
            math.cos(ey) * math.cos(ex),
        )
        shaft = (px + dvec[0] * L,     py + dvec[1] * L,     pz + dvec[2] * L)
        knob  = (px + dvec[0] * 2 * L, py + dvec[1] * 2 * L, pz + dvec[2] * 2 * L)
        ori   = p.getQuaternionFromEuler([ex, ey, 0.0])

        p.resetBasePositionAndOrientation(
            self._shaft, shaft, ori, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self._knob, knob, (0, 0, 0, 1), physicsClientId=self.client)

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
        self._rest_pos  = (bx, by, bz + BUTTON_HOVER_LIFT)
        self._press_pos = (bx, by, bz + BUTTON_HOVER_LIFT - BUTTON_PRESS_DEPTH)
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
        self._max_render_dim = _SOFTWARE_MAX_RENDER_DIM if software_viewer else 960
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
        self._disable_arm_collisions()

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
        # NOTE: the opening coin-insert is NOT run here. Animating during
        # construction (before the model/env finish loading) froze the very
        # first frame. The eval loop calls run_coin_insert() once everything
        # is ready, so the first game and post-GAME-OVER restarts behave the
        # same way. The cabinet simply starts on the 'INSERT COIN' screen.

    def _disable_arm_collisions(self):
        """Both arms are driven kinematically (position control), so remove them
        from collision entirely. This guarantees the robots never collide with
        each other or the arcade cabinet — no interpenetration jitter or fighting
        against the static console geometry."""
        for body in (self.joystick_arm.arm, self.button_arm.arm):
            njoints = p.getNumJoints(body, physicsClientId=self.client)
            for link in range(-1, njoints):
                p.setCollisionFilterGroupMask(
                    body, link, collisionFilterGroup=0, collisionFilterMask=0,
                    physicsClientId=self.client)

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
        # The button arm's gripper closes only while it is carrying the coin
        # ('hand'); it stays open while reaching for or releasing it.
        grip_open = 0.0 if coin_mode == 'hand' else 1.0
        for frame in range(max(frames, 1)):
            if self._closed:
                return
            t = (frame + 1) / float(max(frames, 1))
            pos = self._lerp(start, end, t)
            self.joystick_arm.step(0)
            self.button_arm.move_to(pos)
            self.button_arm.set_gripper(grip_open)

            for _ in range(self._sim_steps):
                p.stepSimulation(physicsClientId=self.client)
            self.joystick_arm.update_gripper()
            self.button_arm.update_gripper()

            if coin_mode == 'table':
                self._set_coin_position(COIN_PICKUP_POS)
            elif coin_mode == 'hand':
                # Glue the coin to the actual fingertip gap so it reads as held.
                self._set_coin_position(self.button_arm.gripper_tip())
            elif coin_mode == 'slot':
                self._set_coin_position((COIN_SLOT_POS[0], COIN_SLOT_POS[1] - 0.004, COIN_SLOT_POS[2]))
            else:
                self._set_coin_position(COIN_PICKUP_POS, visible=False)

            self._draw_software_viewer(force=True)
            if not self._software_viewer:
                time.sleep(1.0 / 60.0)

    def _hold_screen(self, state, frames=40):
        """Hold a screen state (e.g. GAME OVER) with both arms parked at rest."""
        self._screen_state = state
        for _ in range(max(frames, 1)):
            if self._closed:
                return
            self.joystick_arm.step(0)
            self.button_arm.step(0)
            for _ in range(self._sim_steps):
                p.stepSimulation(physicsClientId=self.client)
            self.joystick_arm.update_gripper()
            self.button_arm.update_gripper()
            self._draw_software_viewer(force=True)
            if not self._software_viewer:
                time.sleep(1.0 / 60.0)

    def _drop_coin_into_slot(self, frames=10):
        """From outside the cabinet, push the coin INWARD (+Y) into the slot on
        the front face, then it disappears inside."""
        start = (COIN_SLOT_POS[0], COIN_SLOT_INSERT[1], COIN_SLOT_INSERT[2])
        end   = (COIN_SLOT_POS[0], COIN_SLOT_POS[1] + 0.04, COIN_SLOT_POS[2])
        for f in range(max(frames, 1)):
            if self._closed:
                return
            t = (f + 1) / float(max(frames, 1))
            self.button_arm.set_gripper(1.0)   # open to release the coin
            self._set_coin_position(self._lerp(start, end, t))
            for _ in range(self._sim_steps):
                p.stepSimulation(physicsClientId=self.client)
            self.joystick_arm.update_gripper()
            self.button_arm.update_gripper()
            self._draw_software_viewer(force=True)
            if not self._software_viewer:
                time.sleep(1.0 / 60.0)
        self._set_coin_position(COIN_PICKUP_POS, visible=False)

    def run_coin_insert(self, outcome=None):
        """
        Play the coin-insertion animation, then leave the screen on LOADING GAME
        so the next step() flips it to GAME RUNNING.  Replayable: call it again
        after a round ends to physically start a fresh game.

        outcome:
            None    — first startup, no end-of-round screen.
            'loss'  — Mario died before clearing the stage; show GAME OVER.
            'win'   — Mario cleared the stage/map; show WINNER!.
        """
        if self._closed:
            return
        if outcome == 'win':
            print('[RobotArm] Stage cleared — WINNER! Re-inserting coin')
            self._hold_screen('WINNER', frames=48)
        elif outcome == 'loss':
            print('[RobotArm] Game over — re-inserting coin')
            self._hold_screen('GAME OVER', frames=48)
        else:
            print('[RobotArm] Startup sequence: inserting coin')
        self._screen_state = 'INSERT COIN'
        self._set_coin_position(COIN_PICKUP_POS)
        self._draw_software_viewer(force=True)
        # Grab the coin off the tray.
        self._move_startup_segment(BUTTON_REST, COIN_PICKUP_POS, 10, coin_mode='table')
        # 1) LIFT straight up off the tray.
        self._move_startup_segment(COIN_PICKUP_POS, COIN_PICKUP_LIFT, 8, coin_mode='hand')
        # 2) Traverse HORIZONTALLY (high above the cabinet) to over the slot.
        self._move_startup_segment(COIN_PICKUP_LIFT, COIN_OVER_SLOT, 12, coin_mode='hand')
        # 3) Lower straight DOWN to slot height, still in front of the cabinet.
        self._move_startup_segment(COIN_OVER_SLOT, COIN_SLOT_INSERT, 10, coin_mode='hand')
        self._screen_state = 'LOADING GAME'
        # 4) Push the coin forward into the front-face slot.
        self._drop_coin_into_slot()
        # 5) Retract straight UP, then horizontally back home — never into the body.
        self._move_startup_segment(COIN_SLOT_INSERT, COIN_OVER_SLOT, 8, coin_mode='hidden')
        self._move_startup_segment(COIN_OVER_SLOT, BUTTON_REST, 12, coin_mode='hidden')

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
        # Screen quad sized to the real Donkey Kong arcade ratio (256x224 ≈ 8:7),
        # so the game frame is shown without horizontal stretching.
        y = 0.468
        z_top, z_bot = 1.20, 0.70          # height 0.50 (enlarged), centred at 0.95
        half_w = 0.5 * (z_top - z_bot) * (256.0 / 224.0)   # = 0.286
        corners = [
            (-half_w, y, z_top),
            (half_w, y, z_top),
            (half_w, y, z_bot),
            (-half_w, y, z_bot),
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

        if self._screen_state == 'GAME OVER':
            msg = self._viewer_font_big.render('GAME', True, (255, 80, 80))
            over = self._viewer_font_big.render('OVER', True, (255, 80, 80))
            self._screen.blit(msg, msg.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.42)))
            self._screen.blit(over, over.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.64)))
            return

        if self._screen_state == 'WINNER':
            msg = self._viewer_font_big.render('WINNER', True, (90, 240, 120))
            bang = self._viewer_font_big.render('!', True, (255, 225, 80))
            self._screen.blit(msg, msg.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.42)))
            self._screen.blit(bang, bang.get_rect(center=(screen_rect.centerx, screen_rect.y + screen_rect.h * 0.64)))
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
        self.begin_action(action_idx)
        self.tick()

    def begin_action(self, action_idx: int):
        """Set the arm/joystick/button targets for one decision (no sim/render)."""
        if self._closed:
            return
        if self._screen_state in ('LOADING GAME', 'INSERT COIN'):
            self._screen_state = 'GAME RUNNING'
        self.joystick_arm.step(action_idx)
        self.button_arm.step(action_idx)
        # During play both hands stay closed: one grips the stick, one rests
        # as a fist on the button.
        self.joystick_arm.set_gripper(0.0)
        self.button_arm.set_gripper(0.0)

    def tick(self, force_draw: bool = False):
        """Advance physics one slice and render once. Call several times per
        decision to animate the robot smoothly at real-time speed."""
        if self._closed:
            return
        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self.client)
        self.joystick_arm.update_gripper()
        self.button_arm.update_gripper()
        self._draw_software_viewer(force=force_draw)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
        if p.isConnected(self.client):
            p.disconnect(self.client)
