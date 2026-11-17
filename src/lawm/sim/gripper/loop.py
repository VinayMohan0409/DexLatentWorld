"""The closed loop of one stage in one process: observe the scene cloud, plan, execute the hand's displacement, stop on convergence or stall."""
from __future__ import annotations

import numpy as np

from lawm.sim.control import StallDetector, execute_displacement, select_step_delta
from lawm.sim.costs import SEG_HAND


def run_stage(env, obs, planner, scene, stage, cost_builder, execute_step_index, max_step_mm, on_step=None):
    """Replan, execute, re-observe until the true error is under stage.converge_mm or stalls; returns (obs, {"converged", "replans", "records"})."""
    stall = StallDetector(stage.stall_patience, tol=0.5)
    records, converged = [], False
    for replan in range(stage.max_replans):
        points, segment = scene(env.sim)
        goal = stage.goal(points, segment)
        plan = planner.plan(points, segment, goal, cost_builder)
        disp = select_step_delta(plan, execute_step_index, segment == SEG_HAND)
        if max_step_mm and np.linalg.norm(disp) * 1000 > max_step_mm:
            disp = disp * (max_step_mm / 1000.0 / np.linalg.norm(disp))
        before = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        obs = execute_displacement(env, obs, disp, stage.steps_per_replan, stage.grip_action, env.action_dim, on_step=on_step)
        achieved = (np.array(obs["robot0_eef_pos"], dtype=np.float64) - before) * 1000
        true_err = stage.true_error_mm(env)
        records.append(dict(replan=replan, pred_err_mm=plan["err_mm"], true_err_mm=true_err, disp_mm=(disp * 1000).tolist(), achieved_mm=achieved.tolist(),
                            tracking_err_mm=float(np.linalg.norm(achieved - disp * 1000)), stayed=bool(plan["stayed"])))
        converged = true_err < stage.converge_mm
        if converged or stall.update(true_err):
            break
    return obs, dict(converged=converged, replans=len(records), records=records)
