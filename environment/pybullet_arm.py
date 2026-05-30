"""
Person 1 — PyBullet robot arm.
A simple 2-DOF arm whose joints mirror the DQN agent's joystick actions.
Run in DIRECT mode (headless) for training; switch to GUI for demos.
"""

import pybullet as p
import pybullet_data


# Maps discrete action index → (base rotation, shoulder elevation) in radians
_ACTION_JOINTS = {
    0: ( 0.00,  0.00),   # NOOP       — rest position
    1: (-0.70,  0.00),   # LEFT       — base turns left
    2: ( 0.70,  0.00),   # RIGHT      — base turns right
    3: ( 0.00,  0.70),   # UP         — shoulder raises
    4: ( 0.00, -0.30),   # DOWN       — shoulder lowers
    5: ( 0.00,  1.10),   # JUMP       — shoulder extends up
    6: (-0.70,  1.10),   # JUMP+LEFT
    7: ( 0.70,  1.10),   # JUMP+RIGHT
}


class RobotArm:
    """
    Two-link planar arm in PyBullet.
    Joint 0 — base, revolves around Z (left/right).
    Joint 1 — shoulder, revolves around Y (up/down).
    """

    def __init__(self, gui=False):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF('plane.urdf', physicsClientId=self.client)
        self.arm = self._build_arm()

    # ------------------------------------------------------------------

    def step(self, action_idx):
        base_target, shoulder_target = _ACTION_JOINTS.get(action_idx, (0.0, 0.0))
        p.setJointMotorControl2(
            self.arm, 0, p.POSITION_CONTROL,
            targetPosition=base_target, force=500,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(
            self.arm, 1, p.POSITION_CONTROL,
            targetPosition=shoulder_target, force=500,
            physicsClientId=self.client,
        )
        for _ in range(8):
            p.stepSimulation(physicsClientId=self.client)

    def close(self):
        p.disconnect(self.client)

    # ------------------------------------------------------------------

    def _build_arm(self):
        base_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.10, height=0.20,
            physicsClientId=self.client,
        )
        base_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.10, length=0.20,
            rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self.client,
        )
        link1_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.30],
            physicsClientId=self.client,
        )
        link1_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.30],
            rgbaColor=[0.2, 0.6, 0.8, 1],
            physicsClientId=self.client,
        )
        link2_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.25],
            physicsClientId=self.client,
        )
        link2_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.25],
            rgbaColor=[0.8, 0.4, 0.2, 1],
            physicsClientId=self.client,
        )

        arm = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.10],
            linkMasses=[1.0, 0.5],
            linkCollisionShapeIndices=[link1_col, link2_col],
            linkVisualShapeIndices=[link1_vis, link2_vis],
            linkPositions=[[0, 0, 0.50], [0, 0, 0.55]],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 1],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1], [0, 1, 0]],
            physicsClientId=self.client,
        )

        for i in range(2):
            p.setJointMotorControl2(
                arm, i, p.POSITION_CONTROL,
                targetPosition=0, force=500,
                physicsClientId=self.client,
            )
        return arm
