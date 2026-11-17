"""Stack, grasped start: cubeA is picked up by a scripted real grasp (verified by a lift test), then the planner carries it onto cubeB."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lawm.sim.gripper import clouds, drive, loop, stage as stage_lib

PAD_HALF_HEIGHT = 0.0375  # half the height of the fingertip pad box


@dataclass(frozen=True)
class StackSettings:
    """The Stack runs: replans, sub-steps per replan, the horizon step executed (-1 the last), the largest step, the stop distance and the stall patience."""

    replans: int
    steps_per_replan: int
    execute_step_index: int
    max_step_mm: float
    converge_mm: float
    stall_patience: int


def setup_grasp(env):
    """Reset and grasp cubeA with the real controller; returns (obs, rise of cubeA in mm). The caller seeds numpy right before this call."""
    obs = env.reset()
    model = env.sim.model
    pos = env.sim.data.body_xpos[model.body_name2id("cubeA_main")].copy()
    half = model.geom_size[model.geom_name2id("cubeA_g0")][2]
    table = model.geom_name2id("table_collision")
    table_top = env.sim.data.geom_xpos[table][2] + model.geom_size[table][2]
    pad_center_z = table_top + 2 * half / 3.0 + PAD_HALF_HEIGHT
    return drive.grasp_and_lift(env, obs, "cubeA_main", pos + np.array([0.0, 0.0, 0.12]), np.array([pos[0], pos[1], pad_center_z - drive.EEF_TO_PAD_Z]))


def check_stack_success(env):
    """(success, detail): cubeA lifted above the table, within 50 mm of cubeB in x and y, and touching it."""
    a, b = env.sim.data.body_xpos[env.cubeA_body_id], env.sim.data.body_xpos[env.cubeB_body_id]
    lifted = a[2] > env.table_offset[2] + 0.04
    aligned = bool(lifted and np.linalg.norm(a[:2] - b[:2]) < 0.05)
    touching = bool(env.check_contact(env.cubeA, env.cubeB))
    return aligned and touching, dict(lifted=bool(lifted), aligned=aligned, touching=touching)


def run_trial(env, planner, seed, settings):
    """One Stack trial: seeded reset and grasp, then the planner carries cubeA to cubeB's top; returns the result record."""
    np.random.seed(seed)
    obs, rise_mm = setup_grasp(env)
    if rise_mm <= drive.LIFT_OK_MM:
        return dict(seed=seed, setup_ok=False, grasp_lift_mm=float(rise_mm))
    model, data = env.sim.model, env.sim.data
    a_body, b_body = model.body_name2id("cubeA_main"), model.body_name2id("cubeB_main")
    offset = np.array([0.0, 0.0, model.geom_size[model.geom_name2id("cubeA_g0")][2] + model.geom_size[model.geom_name2id("cubeB_g0")][2]])

    def target():
        """Where cubeA's centre should end up: on top of cubeB."""
        return data.body_xpos[b_body].copy() + offset

    a0, b0 = data.body_xpos[a_body].copy(), data.body_xpos[b_body].copy()
    scene = clouds.stack_scene(env.sim)
    planner.reset(seed)
    stage = stage_lib.Stage("stack", stage_lib.object_translated_to(target, lambda: data.body_xpos[a_body]), stage_lib.body_pos_error("cubeA_main", target),
                            settings.converge_mm, settings.replans, settings.steps_per_replan, settings.stall_patience, grip_action=1.0)
    obs, info = loop.run_stage(env, obs, planner, scene, stage, stage_lib.object_goal_cost, settings.execute_step_index, settings.max_step_mm)
    success, detail = check_stack_success(env)
    final = data.body_xpos[a_body].copy()
    return dict(seed=seed, setup_ok=True, grasp_lift_mm=float(rise_mm), cubeA_pos0=a0.tolist(), cubeB_pos0=b0.tolist(), final_err_mm=float(np.linalg.norm(target() - final) * 1000),
                err_lateral_mm=float(np.linalg.norm((target() - final)[:2]) * 1000), err_vertical_mm=float((final - target())[2] * 1000), converged=info["converged"],
                replans=info["replans"], real_success=bool(success), real_success_detail=detail, records=info["records"])
