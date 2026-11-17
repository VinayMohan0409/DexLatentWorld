"""Scripted end-effector driving with the real controller, used to set up a grasp before the planner takes over."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

EEF_TO_PAD_Z = 0.029  # the fingertip pad midpoint is 29 mm above the end-effector site
LIFT_OK_MM = 30.0     # a grasp holds if the lifted object rises more than this


def drive_to(env, obs, target_pos, grip_action, n_steps, target_quat=None, rot_gain=2.0):
    """Drive the end effector towards target_pos for n_steps with clip(10 * error, -1, 1) into OSC; target_quat (xyzw) is tracked by a quaternion proportional loop."""
    for _ in range(n_steps):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[0:3] = np.clip((target_pos - np.array(obs["robot0_eef_pos"])) * 10.0, -1.0, 1.0)
        if target_quat is not None and env.action_dim >= 6:
            error = Rotation.from_quat(target_quat) * Rotation.from_quat(np.array(obs["robot0_eef_quat"])).inv()
            action[3:6] = np.clip(error.as_rotvec() * rot_gain, -1.0, 1.0)
        action[-1] = grip_action
        obs, _, _, _ = env.step(action)
    return obs


def grasp_and_lift(env, obs, body_name, above_pos, grasp_pos, target_quat=None):
    """Open and go to above_pos, descend to grasp_pos, close, lift 50 mm; returns (obs, rise of the object in mm)."""
    bid = env.sim.model.body_name2id(body_name)
    obs = drive_to(env, obs, above_pos, -1.0, 120, target_quat)
    obs = drive_to(env, obs, grasp_pos, -1.0, 120, target_quat)
    obs = drive_to(env, obs, grasp_pos, 1.0, 80, target_quat)
    before = env.sim.data.body_xpos[bid][2]
    obs = drive_to(env, obs, grasp_pos + np.array([0.0, 0.0, 0.05]), 1.0, 100, target_quat)
    return obs, (env.sim.data.body_xpos[bid][2] - before) * 1000
