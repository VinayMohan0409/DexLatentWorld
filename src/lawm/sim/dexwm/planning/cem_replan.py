"""One replan of the DexWM planner: a cross-entropy search over latent actions around the prior, the lowest-cost sampled candidate as the plan, and the hold candidate; the decoder is passed in."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lawm.sim.cem import cem_iterate
from lawm.sim.costs import goal_cost_mm, terminal_weights


@dataclass(frozen=True)
class CemSettings:
    """The search budget of one replan: candidates per iteration, iterations, the elite share, the floor of the sampling spread and whether holding still competes."""

    pop: int
    iters: int
    elite_frac: float
    min_std: float
    hold_candidate: bool


@dataclass
class Plan:
    """The plan of one replan: cumulative displacements (n_steps, N, 3) from the current cloud, their cost, how they were chosen ('best_sample' or 'hold'), the best sampled cost and the predicted final hand error (mm)."""

    delta: np.ndarray
    cost: float
    executed: str
    best_sample_cost: float
    final_hand_error_mm: float


def replan_rng(seed, replan):
    """The random generator of one replan of one episode: different for every (seed, replan) pair, the same when repeated."""
    return np.random.default_rng((seed, replan))


def plan_replan(decode, cost, z_mean, z_std, settings, rng):
    """CEM from the prior with decode(z) -> (n, n_steps, N, 3) and the replan's PlanCost; the plan is the lowest-cost candidate sampled (no extra decode), and the zero-displacement candidate replaces it only when strictly cheaper."""
    best = {}

    def scored(deltas):
        """The candidates' costs; keeps the prediction of the cheapest seen so far."""
        costs = cost(deltas)
        i = int(np.argmin(costs))
        if costs[i] < best.get("cost", np.inf):
            best.update(cost=float(costs[i]), delta=deltas[i].copy())
        return costs

    mean = np.array(z_mean, dtype=np.float32)
    std = np.maximum(np.asarray(z_std, dtype=np.float32), settings.min_std)
    _, best_sample_cost = cem_iterate(decode, scored, mean, std, settings.pop, settings.iters, settings.elite_frac, settings.min_std, rng)
    delta, plan_cost, executed = best["delta"], best["cost"], "best_sample"
    if settings.hold_candidate:
        hold = np.zeros((1, *delta.shape), dtype=np.float32)
        hold_cost = float(cost(hold)[0])
        if hold_cost < plan_cost:
            delta, plan_cost, executed = hold[0], hold_cost, "hold"
    final = float(goal_cost_mm(delta[None], cost.coords0, cost.goal, cost.hand_mask, terminal_weights(len(delta)))[0])
    return Plan(delta, plan_cost, executed, best_sample_cost, final)
