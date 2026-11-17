"""PickPlaceMilk, grasped start: the milk carton is picked up by a scripted real grasp, then the planner lifts it clear of the bin walls, carries it over and lowers it into the target slot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from lawm.sim.gripper import clouds, drive, loop, stage as stage_lib


@dataclass(frozen=True)
class MilkSettings:
    """The Milk runs: replans of the transfer and lowering stages (the lift gets a quarter, at least 5), sub-steps per replan, the horizon step executed, the largest step, the stop distance of the lowering stage, the stall patience and the grasp yaw policy."""

    replans: int
    steps_per_replan: int
    execute_step_index: int
    max_step_mm: float
    converge_mm: float
    stall_patience: int
    grasp_yaw: str
    lift_converge_mm: float
    transfer_converge_mm: float


def carton_target_quat(obs, yaw, policy):
    """Orientation the gripper tracks while grasping: none (policy "none") or the carton's yaw (policy "full")."""
    if policy == "none":
        return None
    return (Rotation.from_rotvec([0, 0, yaw]) * Rotation.from_quat(np.array(obs["robot0_eef_quat"]))).as_quat()


def setup_grasp(env, policy="none"):
    """Reset and grasp the carton with the real controller; returns (obs, rise of the carton in mm). The caller seeds numpy right before this call."""
    obs = env.reset()
    bid = env.sim.model.body_name2id("Milk_main")
    pos = env.sim.data.body_xpos[bid].copy()
    w, x, y, z = env.sim.data.body_xquat[bid]
    quat = carton_target_quat(obs, Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2], policy)
    return drive.grasp_and_lift(env, obs, "Milk_main", pos + np.array([0.0, 0.0, 0.15]), pos + np.array([0.0, 0.0, -drive.EEF_TO_PAD_Z]), quat)


def wall_top_z(sim, body_name):
    """Highest world z reached by any geom of the body (bin walls)."""
    model, data = sim.model, sim.data
    bid = model.body_name2id(body_name)
    return float(max(data.geom_xpos[g][2] + model.geom_size[g][2] for g in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid])))


def run_trial(env, planner, seed, settings):
    """One Milk trial: seeded reset and grasp, then lift, transfer and lower with one planner (its random stream and warm start continue across the three stages); returns the result record."""
    np.random.seed(seed)
    obs, rise_mm = setup_grasp(env, settings.grasp_yaw)
    if rise_mm <= drive.LIFT_OK_MM:
        return dict(seed=seed, setup_ok=False, grasp_lift_mm=float(rise_mm))
    model, data = env.sim.model, env.sim.data
    bid = model.body_name2id("Milk_main")
    pos0 = data.body_xpos[bid].copy()
    half_h = model.geom_size[model.geom_name2id("Milk_g0")][2]
    target = env.target_bin_placements[env.object_to_id["milk"]].copy()
    target[2] += half_h
    clear_z = max(wall_top_z(env.sim, "bin1"), wall_top_z(env.sim, "bin2")) + half_h + 0.02
    scene = clouds.milk_scene(env.sim)
    planner.reset(seed)
    infos = {}
    for name, goal_pos, converge, replans in (("lift", np.array([pos0[0], pos0[1], clear_z]), settings.lift_converge_mm, max(5, settings.replans // 4)),
                                              ("transfer", np.array([target[0], target[1], clear_z]), settings.transfer_converge_mm, settings.replans),
                                              ("lower", target, settings.converge_mm, settings.replans)):
        stage = stage_lib.Stage(name, stage_lib.object_translated_to(lambda g=goal_pos: g, lambda: data.body_xpos[bid]), stage_lib.body_pos_error("Milk_main", lambda g=goal_pos: g),
                                converge, replans, settings.steps_per_replan, settings.stall_patience, grip_action=1.0)
        obs, infos[name] = loop.run_stage(env, obs, planner, scene, stage, stage_lib.object_goal_cost, settings.execute_step_index, settings.max_step_mm)
    return dict(seed=seed, setup_ok=True, grasp_lift_mm=float(rise_mm), milk_pos0=pos0.tolist(), target_pos=target.tolist(), clear_z=float(clear_z),
                final_obj_err_mm=float(np.linalg.norm(target - data.body_xpos[bid]) * 1000), lift_converged=infos["lift"]["converged"], transfer_converged=infos["transfer"]["converged"],
                converged=infos["lower"]["converged"], replans=sum(i["replans"] for i in infos.values()), real_success=bool(infos["lower"]["converged"]),
                records={name: info["records"] for name, info in infos.items()})
