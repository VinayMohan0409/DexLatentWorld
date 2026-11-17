"""Tests of the controller settings: the per-axis arm gains, the finger controller choice and its torque law."""
from types import SimpleNamespace

import numpy as np

from lawm.sim.dexwm.control import controller as C


class Base:
    """The class a finger controller derives from."""

    def run_controller(self):
        """Count the call of the original law."""
        self.original = getattr(self, "original", 0) + 1


class Fingers(Base):
    """A finger controller with the attributes the PD law reads."""

    def __init__(self, goal, pos, vel):
        """Set the goal, the state and the torque limits."""
        self.goal_qpos, self.joint_pos, self.joint_vel = goal, pos, vel
        self.actuator_min, self.actuator_max = np.full(16, -0.35), np.full(16, 0.35)

    def update(self):
        """Nothing to refresh."""


def env_with(parts):
    """The part of an environment the finger controller lookup walks."""
    return SimpleNamespace(robots=[SimpleNamespace(composite_controller=SimpleNamespace(part_controllers=parts))])


def test_arm_gains_per_axis():
    """Stiffness 300 is doubled on z and x10 on the rotations; damping is 2 sqrt(stiffness) x ratio, doubled on z; interpolation is off."""
    ctrl = SimpleNamespace(interpolator_pos="p", interpolator_ori="o")
    C.set_arm_gains(ctrl, 300.0, 1.0)
    assert np.allclose(ctrl.kp, [300, 300, 600, 3000, 3000, 3000])
    assert np.allclose(ctrl.kd, [2 * np.sqrt(300), 2 * np.sqrt(300), 4 * np.sqrt(600), 2 * np.sqrt(3000), 2 * np.sqrt(3000), 2 * np.sqrt(3000)])
    assert ctrl.interpolator_pos is None and ctrl.interpolator_ori is None
    C.set_arm_gains(ctrl, 150.0, 2.0)
    assert np.allclose(ctrl.kd[0], 2 * np.sqrt(150) * 2.0)


def test_finger_controller_is_the_right_gripper_part():
    """The part with 'right' and 'grip' in its name wins; without one, any gripper part; the arm part is never chosen."""
    a, b, c = object(), object(), object()
    assert C.finger_controller(env_with({"right": a, "left_gripper": b, "right_gripper": c})) is c
    assert C.finger_controller(env_with({"right": a, "left_gripper": b})) is b


def test_finger_torque_law_and_clipping():
    """tau = kp (goal - angle) - kd velocity, clipped to +-0.35, and the original controller's run_controller is called after it."""
    fingers = Fingers(np.full(16, 0.1), np.zeros(16), np.zeros(16))
    C.set_finger_pd(env_with({"right_gripper": fingers}), 2.0, 0.05)
    assert np.allclose(fingers.run_controller(), 0.2) and fingers.original == 1
    fingers.joint_vel = np.full(16, 1.0)
    assert np.allclose(fingers.run_controller(), 0.2 - 0.05)
    fingers.goal_qpos = np.full(16, 5.0)
    assert np.allclose(fingers.run_controller(), 0.35)


def test_only_that_controller_is_patched():
    """Another instance of the same class keeps the class's own law."""
    one, other = Fingers(np.ones(16), np.zeros(16), np.zeros(16)), Fingers(np.ones(16), np.zeros(16), np.zeros(16))
    C.set_finger_pd(env_with({"right_gripper": one}), 2.0, 0.05)
    other.run_controller()
    assert "run_controller" in one.__dict__ and "run_controller" not in other.__dict__ and other.original == 1
