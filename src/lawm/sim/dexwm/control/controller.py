"""The controller settings of the DexWM hand: the arm's stiffness and damping per axis, and a position-and-velocity torque law for the finger joints."""
from __future__ import annotations

import types

import numpy as np


def set_arm_gains(ctrl, kp, damping):
    """Stiffness kp (twice on z, ten times on the rotations) and damping = damping ratio x critical (twice on z) on the arm's OSC controller, without interpolation."""
    stiffness = np.full(6, float(kp))
    stiffness[2] *= 2
    stiffness[3:6] *= 10
    derivative = 2 * np.sqrt(stiffness) * damping
    derivative[2] *= 2
    ctrl.kp, ctrl.kd = stiffness, derivative
    ctrl.interpolator_pos = None
    ctrl.interpolator_ori = None


def finger_controller(env):
    """The part controller of the right hand's fingers."""
    parts = env.robots[0].composite_controller.part_controllers
    keys = [n for n in parts if "right" in n and "grip" in n.lower()] or [n for n in parts if "grip" in n.lower()]
    return parts[keys[0]]


def set_finger_pd(env, kp, kd):
    """Replace the finger controller's torque law by kp (goal - angle) - kd velocity, clipped to the actuator limits."""
    gc = finger_controller(env)

    def run(self):
        """One control step of the finger joints; returns the torques."""
        self.update()
        tau = kp * (np.asarray(self.goal_qpos) - self.joint_pos) - kd * self.joint_vel
        self.ctrl = np.clip(tau, self.actuator_min, self.actuator_max)
        super(type(self), self).run_controller()
        return self.ctrl
    gc.run_controller = types.MethodType(run, gc)
