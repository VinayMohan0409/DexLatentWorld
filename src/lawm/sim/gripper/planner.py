"""The latent-action planner of the gripper tasks in one process: a cross-entropy search over latents decoded by the model, a hold candidate, and the plan to execute."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.spatial.transform import Rotation

from lawm.sim.cem import cem_iterate
from lawm.sim.costs import SEG_HAND, step_weights, terminal_weights

STAY_MARGIN_MM = 0.5  # the hold candidate must beat the search by more than this


@dataclass(frozen=True)
class PlannerSettings:
    """The search: candidates per iteration, iterations, elite share and spread floor, the extra spread decay, whether the previous mean seeds the next replan, and the horizon step the cost scores (None = the last)."""

    pop: int
    iters: int
    elite_frac: float
    min_std: float
    std_decay: float
    warm_start: bool
    weight_step_index: int | None


def kabsch_rotvec(points, moved):
    """The rotation vector that best aligns points onto moved, row for row."""
    u, _, vt = np.linalg.svd((points - points.mean(axis=0)).T @ (moved - moved.mean(axis=0)))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return Rotation.from_matrix(vt.T @ np.diag([1.0, 1.0, float(d)]) @ u.T).as_rotvec()


class LawmPlanner:
    """Plans every replan of a trial; model.prior = (z_mean, z_std) and model.decoder(coords, segment) -> decode(z) as for the DexWM planner. reset(seed) starts a trial: a new random stream and no warm start."""

    def __init__(self, model, settings):
        """Keep the model adapter and the search settings."""
        self.model, self.settings = model, settings
        self.rng, self.warm_start = np.random.default_rng(0), None

    def reset(self, seed):
        """Start a trial: the random stream of the search from the trial seed, no warm start."""
        self.rng, self.warm_start = np.random.default_rng(int(seed)), None

    def plan(self, coords0, segment, goal, cost_builder, pop=None, iters=None, std_decay=None):
        """Plan one replan: {"all_step_deltas" (n_steps, N, 3), "disp_rot", "err_mm", "best_cost_mm", "stayed"}; cost_builder(coords0, goal, segment, weights) -> (cost, goal mask)."""
        s = self.settings
        z_mean, z_std = self.model.prior
        decode = self.model.decoder(coords0, segment)
        n_steps = z_mean.shape[0]
        weights = terminal_weights(n_steps) if s.weight_step_index is None else step_weights(n_steps, s.weight_step_index)
        cost, goal_mask = cost_builder(coords0, goal, segment, weights)
        start = self.warm_start if (self.warm_start is not None and s.warm_start) else z_mean
        mean, best_cost = cem_iterate(decode, cost, start.astype(np.float32).copy(), np.maximum(z_std, s.min_std).astype(np.float32), pop or s.pop, iters or s.iters,
                                      s.elite_frac, s.min_std, self.rng, std_decay=s.std_decay if std_decay is None else std_decay)
        self.warm_start = mean.copy()
        deltas = decode(mean[None])[0]
        cost_zero = float(cost(np.zeros((1, *deltas.shape), dtype=np.float32))[0])
        stayed = cost_zero < best_cost - STAY_MARGIN_MM
        if stayed:
            deltas = np.zeros_like(deltas)
        hand = segment == SEG_HAND
        disp_rot = kabsch_rotvec(coords0[hand], (coords0 + deltas[-1])[hand])
        err_mm = float(np.linalg.norm((coords0 + deltas[-1])[goal_mask] - goal[goal_mask], axis=-1).mean() * 1000)
        return dict(all_step_deltas=deltas, disp_rot=disp_rot, err_mm=err_mm, best_cost_mm=best_cost, cost_zero_mm=cost_zero, stayed=stayed)
