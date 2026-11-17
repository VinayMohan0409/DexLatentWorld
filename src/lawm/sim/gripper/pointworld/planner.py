"""The PointWorld planner of the gripper tasks: cross-entropy search over spline knots of the gripper's displacement, the same cost and executor as the latent-action planner, and the best candidate as the plan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from lawm.sim.cem import cem_iterate
from lawm.sim.costs import SEG_HAND, SEG_OBJECT, SEG_TABLE, step_weights
from lawm.sim.gripper.pointworld import observation, rollout

SPLINE_DEGREE = 3
TASK_OBSERVATION = {"stack": (("cubeA_g0", "cubeA_g0_vis"), "cubeA_main"), "milk": (("Milk_g0", "Milk_g0_vis", "Milk_g0_visual"), "Milk_main"),
                    "push_puck": (("cube_g0", "cube_g0_vis", "cube_g0_visual"), "cube_main"), "push_cube": (("cube_g0", "cube_g0_vis", "cube_g0_visual"), "cube_main")}
ROBOTIQ_BODIES = ("gripper0_right_right_gripper", "gripper0_right_left_outer_knuckle", "gripper0_right_left_inner_finger", "gripper0_right_left_inner_knuckle",
                  "gripper0_right_right_outer_knuckle", "gripper0_right_right_inner_finger", "gripper0_right_right_inner_knuckle")
PLATE_BODIES = ("gripper0_right_wiping_gripper",)


@dataclass(frozen=True)
class PointWorldSettings:
    """The search (candidates, iterations, elite share, spread floor and start, and decay), the trajectory (knots, metres per unit, largest displacement), the model's batch and the cost's chunk, and whether the known normal errors of the original runs are off."""

    pop: int
    iters: int
    elite_frac: float
    min_std: float
    sigma_max: float
    std_decay: float
    n_knots: int
    action_scale: float
    max_reach: float
    batch: int
    cost_chunk: int
    outward_normals: bool


def spline_basis(horizon, n_knots, degree=SPLINE_DEGREE):
    """(horizon + 1, n_knots) design matrix of a clamped B-spline."""
    knots = np.concatenate([np.zeros(degree + 1), np.linspace(0, 1, n_knots - degree + 1)[1:-1], np.ones(degree + 1)])
    u = np.linspace(0.0, 1.0, horizon + 1)
    basis = np.zeros((horizon + 1, n_knots), dtype=np.float32)
    for i in range(n_knots):
        c = np.zeros(n_knots)
        c[i] = 1.0
        basis[:, i] = BSpline(knots, c, degree, extrapolate=False)(u)
    return np.nan_to_num(basis, nan=0.0)


def knots_to_trajectory(knots, basis, action_scale, max_reach):
    """(K, n_knots, 3) knots to (K, horizon, 3) cumulative displacement in metres: anchored at zero, row i is after step i + 1, and clamped in length."""
    trajectory = np.einsum("tk,bkd->btd", basis, knots) * action_scale
    trajectory = (trajectory - trajectory[:, 0:1, :])[:, 1:, :]
    norm = np.linalg.norm(trajectory, axis=-1, keepdims=True)
    return (trajectory * np.minimum(1.0, max_reach / np.maximum(norm, 1e-9))).astype(np.float32)


def predicted_deltas(trajectory, object_disp, scene_disp, segment):
    """(K, H, N, 3) displacements on the driver's rows: hand rows take the trajectory, object and table rows the predicted means."""
    out = np.zeros((trajectory.shape[0], trajectory.shape[1], len(segment), 3), dtype=np.float32)
    out[:, :, segment == SEG_HAND, :] = trajectory[:, :, None, :]
    out[:, :, segment == SEG_OBJECT, :] = object_disp[:, :, None, :]
    out[:, :, segment == SEG_TABLE, :] = scene_disp[:, :, None, :]
    return out


class PointWorldPlanner:
    """The same interface as LawmPlanner; bind(env, task) names the environment to observe, and every replan captures its own observation."""

    def __init__(self, model, model_args, device, pipeline, collate, settings, seed_torch=None):
        """Keep the model, PointWorld's release functions and the settings."""
        self.model, self.args, self.device, self.pipeline, self.collate, self.settings, self.seed_torch = model, model_args, device, pipeline, collate, settings, seed_torch
        self.basis = spline_basis(rollout.H_CHUNK, settings.n_knots)
        self.weights = step_weights(rollout.H_CHUNK, rollout.H_CHUNK - 1)
        self.rng, self.env, self.task, self.observer = np.random.default_rng(0), None, None, None

    def bind(self, env, task):
        """The environment and task of the next trial."""
        self.env, self.task = env, task

    def reset(self, seed):
        """Start a trial, after the environment's reset: a new random stream and an observer of this scene."""
        self.rng = np.random.default_rng(int(seed))
        if self.seed_torch is not None:
            self.seed_torch(int(seed))
        geoms, body = TASK_OBSERVATION[self.task]
        bodies = PLATE_BODIES if self.task.startswith("push") else ROBOTIQ_BODIES
        self.observer = observation.Observer(self.env, geoms, body, bodies, outward_normals=self.settings.outward_normals)

    def search(self, predict, cost, segment, pop, iters, std_decay):
        """(best trajectory (H, 3), its cost) of a cross-entropy search over the knots, started at zero."""
        s = self.settings

        def decode(candidates):
            """Knots to predicted displacements on the driver's rows."""
            trajectories = knots_to_trajectory(candidates, self.basis, s.action_scale, s.max_reach)
            return predicted_deltas(trajectories, *predict(trajectories), segment)

        def chunked_cost(deltas):
            """The cost over chunks of candidates."""
            return np.concatenate([cost(deltas[lo:lo + s.cost_chunk]) for lo in range(0, len(deltas), s.cost_chunk)])

        mean, std = np.zeros((s.n_knots, 3), dtype=np.float32), np.full((s.n_knots, 3), s.sigma_max, dtype=np.float32)
        _, best_cost, best = cem_iterate(decode, chunked_cost, mean, std, pop, iters, s.elite_frac, s.min_std, self.rng, std_decay=std_decay, return_best=True)
        return knots_to_trajectory(best[None], self.basis, s.action_scale, s.max_reach)[0], best_cost

    def plan(self, coords0, segment, goal, cost_builder, pop=None, iters=None, std_decay=None):
        """Plan one replan: {"all_step_deltas" (10, N, 3), "disp_rot" (zero: this planner is translation only), "err_mm", "best_cost_mm", "stayed" (always False)}."""
        s = self.settings
        obs, _ = rollout.without_robot(self.observer.capture(self.env))
        predictor = rollout.Rollout(self.model, self.args, obs, self.device, self.pipeline, self.collate, s.batch)
        cost, goal_mask = cost_builder(coords0, goal, segment, self.weights)
        best, best_cost = self.search(predictor.decode, cost, segment, pop or s.pop, iters or s.iters, s.std_decay if std_decay is None else std_decay)
        deltas = predicted_deltas(best[None], *predictor.decode(best[None]), segment)[0]
        err_mm = float(np.linalg.norm((coords0 + deltas[-1])[goal_mask] - goal[goal_mask], axis=-1).mean() * 1000)
        return dict(all_step_deltas=deltas, disp_rot=np.zeros(3, dtype=np.float32), err_mm=err_mm, best_cost_mm=best_cost, stayed=False)
