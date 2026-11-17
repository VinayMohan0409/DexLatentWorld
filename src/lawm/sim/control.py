"""The closed loop of the gripper tasks: execute a predicted end-effector displacement with a proportional controller into OSC, pick the executed step, stop on stall, switch approach and push."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def execute_displacement(env, obs, disp, n_steps, grip_action, action_dim, gain=10.0, on_step=None, rot_delta=None, rot_gain=2.0):
    """Move the end effector by disp over n_steps sub-steps: each step commands clip((sub_target - position) * gain, -1, 1) into OSC (scaled by its output_max), plus a quaternion proportional loop when rot_delta is given."""
    start = np.array(obs["robot0_eef_pos"], dtype=np.float64)
    target = start + disp
    target_rot = None
    if action_dim >= 6 and rot_delta is not None and np.linalg.norm(rot_delta) > 1e-8:
        target_rot = Rotation.from_rotvec(rot_delta) * Rotation.from_quat(np.array(obs["robot0_eef_quat"]))
    for step in range(n_steps):
        sub_target = start + (step + 1) / n_steps * (target - start)
        action = np.zeros(action_dim, dtype=np.float32)
        action[0:3] = np.clip((sub_target - np.array(obs["robot0_eef_pos"])) * gain, -1.0, 1.0)
        if target_rot is not None:
            action[3:6] = np.clip((target_rot * Rotation.from_quat(np.array(obs["robot0_eef_quat"])).inv()).as_rotvec() * rot_gain, -1.0, 1.0)
        if action_dim >= 7:
            action[-1] = grip_action
        obs, _, _, _ = env.step(action)
        if on_step is not None:
            on_step(step, obs)
    return obs


def select_step_delta(plan, step_index, mask):
    """Displacement to execute: the mean over the mask rows of one horizon step of the saved prediction (-1 the last step, 0 standard receding-horizon control)."""
    return plan["all_step_deltas"][step_index][mask].mean(axis=0)


class StallDetector:
    """True once the tracked value has failed to improve on the PREVIOUS value by more than tol for patience updates in a row (it compares with the last value, not the best)."""

    def __init__(self, patience, tol=0.5, mode="decreasing"):
        """Set the patience, the tolerance and whether lower or higher is better."""
        self.patience, self.tol, self.mode = patience, tol, mode
        self.prev = np.inf if mode == "decreasing" else -np.inf
        self.count = 0

    def update(self, value):
        """Record a value; returns True if the patience is used up."""
        improved = value < self.prev - self.tol if self.mode == "decreasing" else value > self.prev + self.tol
        self.count = 0 if improved else self.count + 1
        self.prev = value
        return self.count >= self.patience
