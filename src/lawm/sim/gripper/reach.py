"""Reach a commanded gripper position in closed loop: no object cost and the gripper held open, the simplest test of the planner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lawm.sim.control import execute_displacement, select_step_delta
from lawm.sim.costs import SEG_HAND, goal_cost_mm
from lawm.sim.gripper import clouds

EEF_SITE = "gripper0_right_grip_site"
FINGER_JOINTS = ("gripper0_right_finger_joint", "gripper0_right_left_inner_finger_joint", "gripper0_right_left_inner_knuckle_joint", "gripper0_right_right_outer_knuckle_joint", "gripper0_right_right_inner_finger_joint", "gripper0_right_right_inner_knuckle_joint")
OPEN_QPOS = (0.012, 0.065, 0.065, -0.012, 0.065, 0.065)
GRIP_OPEN = -1.0


@dataclass(frozen=True)
class ReachSettings:
    """The Reach runs: distance, the direction of each seed (degrees in the table plane), replans, sub-steps per replan, the horizon step executed, the largest step and the stop distance."""

    distance_mm: float
    angles_deg: tuple
    replans: int
    steps_per_replan: int
    execute_step_index: int
    max_step_mm: float
    converge_mm: float


def hand_cost(coords0, goal, segment, weights):
    """(cost, goal mask): the horizon-weighted distance of the predicted hand points to their goal, in mm."""
    mask = segment == SEG_HAND
    return (lambda deltas: goal_cost_mm(deltas, coords0, goal, mask, weights)), mask


def run_trial(env, planner, seed, settings):
    """One Reach trial on a newly built environment, from the arm's nominal pose with the fingers open; returns the result record."""
    obs = env.reset()
    sim, s = env.sim, settings
    robot = env.robots[0]
    sim.data.qpos[robot._ref_joint_pos_indexes] = robot.init_qpos
    sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    for joint, value in zip(FINGER_JOINTS, OPEN_QPOS):
        sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id(joint)]] = value
    sim.forward()

    def eef():
        """The end effector's position now."""
        return np.array(sim.data.get_site_xpos(EEF_SITE))

    start = eef().copy()
    angle = np.radians(s.angles_deg[seed])
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    target = start + direction / np.linalg.norm(direction) * (s.distance_mm / 1000.0)
    scene = clouds.reach_scene(sim)
    planner.reset(seed)
    records, converged = [], False
    for replan in range(s.replans):
        points, segment = scene(sim)
        hand = segment == SEG_HAND
        now = eef().copy()
        goal = points.copy()
        goal[hand] = points[hand] + (target - now)[None, :].astype(points.dtype)
        plan = planner.plan(points, segment, goal, hand_cost)
        disp = select_step_delta(plan, s.execute_step_index, hand)
        size = float(np.linalg.norm(disp) * 1000)
        if s.max_step_mm and size > s.max_step_mm:
            disp = disp * (s.max_step_mm / 1000.0 / np.linalg.norm(disp))
        obs = execute_displacement(env, obs, disp, s.steps_per_replan, GRIP_OPEN, env.action_dim, rot_delta=plan["disp_rot"])
        error = float(np.linalg.norm(target - eef()) * 1000)
        records.append(dict(replan=replan, err_mm=error, disp_mm=size, moved_mm=float(np.linalg.norm(eef() - now) * 1000), pred_err_mm=plan["err_mm"], stayed=bool(plan["stayed"])))
        converged = error < s.converge_mm
        if converged:
            break
    return dict(seed=seed, setup_ok=True, angle_deg=float(s.angles_deg[seed]), start_eef=start.tolist(), target_eef=target.tolist(), final_error_mm=records[-1]["err_mm"],
                converged=converged, real_success=converged, replans=len(records), records=records)
