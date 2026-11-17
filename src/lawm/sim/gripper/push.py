"""Push an object across the table with the wiping gripper: an approach and push loop that re-approaches whenever the gripper is out of position behind the object."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from lawm.sim.control import StallDetector, execute_displacement, select_step_delta
from lawm.sim.costs import SEG_HAND, SEG_OBJECT, eef_floor_penalty_mm, goal_cost_mm, maniskill_push_cost_mm
from lawm.sim.gripper import clouds
from lawm.sim.gripper.push_env import PUSH_OBJECTS

TABLE_HALF_EXTENT_XY = 0.4
TABLE_EDGE_MARGIN_M = 0.06
REACH_RADIUS_M = 0.80
EEF_SITE = "gripper0_right_grip_site"
PLATE_TO_EEF_Z = 0.0149  # the plate's centre sits this far above the end-effector site
PLATE_HALF_DEPTH_M = 0.025
APPROACH_CLEARANCE_M = 0.015
PUSH_DIR = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class PushSettings:
    """One push task: the object and distance; the replan budgets, stops and horizon steps of both modes; the search of both modes; and the geometry of the approach and the floor."""

    object: str
    distance_mm: float
    approach_offset_mm: float
    behind_offset_mm: float
    reposition_threshold_mm: float
    replans: int
    approach_replans: int
    stall_patience: int
    steps_per_replan: int
    execute_step_index: int
    approach_execute_step_index: int
    max_step_mm: float
    converge_mm: float
    push_pop: int
    approach_pop: int
    push_iters: int
    push_std_decay: float
    w_floor: float
    floor_ramp_depth_mm: float
    scripted_approach: bool
    scripted_approach_steps: int


def approach_cost(w_floor, min_eef_z, cur_eef_z, ramp_depth_mm):
    """A cost builder for the approach mode: the hand's distance to its goal plus the floor barrier."""
    def build(coords0, goal, segment, weights):
        """(cost, goal mask) of this replan."""
        hand = segment == SEG_HAND
        return (lambda deltas: goal_cost_mm(deltas, coords0, goal, hand, weights) + w_floor * eef_floor_penalty_mm(deltas, hand, cur_eef_z, min_eef_z, ramp_depth_mm=ramp_depth_mm)), hand
    return build


def push_cost(push_dir, offset_m):
    """A cost builder for the push mode: the ManiSkill form, scored on the object's goal."""
    def build(coords0, goal, segment, weights):
        """(cost, goal mask) of this replan."""
        hand, obj = segment == SEG_HAND, segment == SEG_OBJECT
        return (lambda deltas: maniskill_push_cost_mm(deltas, coords0, goal, hand, obj, weights, push_dir, offset_m)), obj
    return build


def max_travel_mm(pos_xy, direction_xy, base_xy):
    """The tightest of the table-edge and robot-reach limits (mm) along a unit direction from pos_xy."""
    bound = TABLE_HALF_EXTENT_XY - TABLE_EDGE_MARGIN_M
    to_table = np.inf
    for pos, d in zip(pos_xy, direction_xy):
        if abs(d) >= 1e-9:
            to_table = min(to_table, ((bound if d > 0 else -bound) - pos) / d)
    rel = pos_xy - base_xy
    b, c = 2.0 * float(np.dot(direction_xy, rel)), float(np.dot(rel, rel)) - REACH_RADIUS_M ** 2
    disc = b * b - 4.0 * c
    to_reach = (-b + np.sqrt(disc)) / 2.0 if (disc >= 0.0 and c < 0.0) else 0.0
    return max(min(to_table, to_reach) * 1000.0, 0.0)


def place_flush(env, object_kind, body):
    """Set up the reset scene: stable contacts on the object, no velocity on the arm or the object, and the object resting flush with no yaw at its sampled x and y."""
    sim = env.sim
    sim.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    sim.model.geom_condim[sim.model.geom_name2id("cube_g0")] = 6
    sim.data.qvel[env.robots[0]._ref_joint_vel_indexes] = 0.0
    sim.forward()
    joint = sim.model.body_jntadr[body]
    dof, qpos = sim.model.jnt_dofadr[joint], sim.model.jnt_qposadr[joint]
    sim.data.qpos[qpos + 2] = env.table_offset[2] + PUSH_OBJECTS[object_kind]["size"][1]
    sim.data.qpos[qpos + 3:qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    sim.data.qvel[dof:dof + 6] = 0.0
    sim.forward()
    return qpos


def run_trial(env, planner, seed, settings, scene_builder=clouds.push_scene):
    """One push trial: seeded reset, flush placement, optional scripted approach, then the approach and push loop; returns the result record."""
    np.random.seed(seed)
    obs = env.reset()
    sim, s = env.sim, settings
    body = sim.model.body_name2id("cube_main")
    qpos = place_flush(env, s.object, body)
    obj_pos0 = sim.data.body_xpos[body].copy()
    base_xy = sim.data.body_xpos[sim.model.body_name2id("robot0_link0")][:2].copy()
    room = max_travel_mm(obj_pos0[:2], PUSH_DIR[:2], base_xy)
    if room < s.distance_mm:
        sim.data.qpos[qpos:qpos + 2] = obj_pos0[:2] - PUSH_DIR[:2] * (s.distance_mm - room) / 1000.0
        sim.forward()
        obj_pos0 = sim.data.body_xpos[body].copy()
    distance_mm = min(s.distance_mm, max_travel_mm(obj_pos0[:2], PUSH_DIR[:2], base_xy))
    target = obj_pos0 + PUSH_DIR * distance_mm / 1000.0
    offset_m = max(s.approach_offset_mm / 1000.0, PUSH_OBJECTS[s.object]["size"][0] + PLATE_HALF_DEPTH_M + APPROACH_CLEARANCE_M)
    hand_goal = target - PUSH_DIR * offset_m
    min_eef_z = obj_pos0[2] - PLATE_TO_EEF_Z
    scene = scene_builder(sim, seed)
    planner.reset(seed)

    def eef():
        """The end effector's position now."""
        return np.array(sim.data.get_site_xpos(EEF_SITE))

    if s.scripted_approach:
        approach = obj_pos0 - PUSH_DIR * offset_m
        approach[2] = min_eef_z
        obs = execute_displacement(env, obs, (approach - eef()).astype(np.float32), s.scripted_approach_steps, 0.0, env.action_dim)
    stall = StallDetector(s.stall_patience, tol=0.5)
    records, converged, clamped = [], False, 0
    for replan in range(s.approach_replans + s.replans):
        points, segment = scene(sim)
        hand = segment == SEG_HAND
        obj_now, eef_now = sim.data.body_xpos[body].copy(), eef()
        approach = obj_now - PUSH_DIR * offset_m
        approach[2] = obj_now[2] - PLATE_TO_EEF_Z
        to_approach = float(np.linalg.norm(eef_now - approach) * 1000)
        goal = points.copy()
        if to_approach > s.reposition_threshold_mm:
            mode, step_index = "approach", s.approach_execute_step_index
            goal[hand] = points[hand] + (approach - eef_now)[None, :]
            plan = planner.plan(points, segment, goal, approach_cost(s.w_floor, min_eef_z, float(eef_now[2]), s.floor_ramp_depth_mm), pop=s.approach_pop)
        else:
            mode, step_index = "push", s.execute_step_index
            obj = segment == SEG_OBJECT
            goal[obj] = points[obj] + (target - obj_now)[None, :]
            goal[hand] = points[hand] + (hand_goal - points[hand].mean(0))[None, :]
            plan = planner.plan(points, segment, goal, push_cost(PUSH_DIR, s.behind_offset_mm / 1000.0), pop=s.push_pop, iters=s.push_iters, std_decay=s.push_std_decay)
        disp = select_step_delta(plan, step_index, hand)
        size = float(np.linalg.norm(disp) * 1000)
        if s.max_step_mm and size > s.max_step_mm:
            disp, size = disp * (s.max_step_mm / size), s.max_step_mm
        if eef_now[2] + disp[2] < min_eef_z:
            disp = disp.copy()
            disp[2] = min_eef_z - eef_now[2]
            clamped += 1
        obs = execute_displacement(env, obs, disp, s.steps_per_replan, 0.0, env.action_dim)
        error = float(np.linalg.norm(target - sim.data.body_xpos[body]) * 1000)
        records.append(dict(replan=replan, mode=mode, eef_to_approach_mm=to_approach, obj_err_mm=error, disp_mm=size, pred_err_mm=plan["err_mm"], stayed=bool(plan["stayed"])))
        converged = error < s.converge_mm
        if converged or stall.update(error):
            break
    return dict(seed=seed, setup_ok=True, obj_pos0=obj_pos0.tolist(), target_pos=target.tolist(), final_obj_err_mm=float(np.linalg.norm(target - sim.data.body_xpos[body]) * 1000),
                converged=converged, real_success=converged, replans=len(records), n_floor_clamped=clamped, n_approach_mode_replans=sum(r["mode"] == "approach" for r in records),
                n_push_mode_replans=sum(r["mode"] == "push" for r in records), records=records)
