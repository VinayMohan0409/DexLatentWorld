"""The cross-entropy method of the planners: sample candidates around a mean, decode and score them, refit mean and spread from the elites."""
from __future__ import annotations

import numpy as np


def cem_iterate(decode, cost, mean, std, pop, iters, elite_frac, min_std, rng, std_decay=1.0, return_best=False):
    """One CEM solve: decode(candidates) gives predicted displacements, cost(deltas) a (pop,) cost; each iteration refits the mean (elite mean) and std (elite std times std_decay, floored at min_std); returns (final mean, lowest cost seen[, lowest-cost candidate]); the executed plan is the final mean, never itself scored."""
    n_elite = max(2, int(pop * elite_frac))
    best_cost, best = np.inf, None
    for _ in range(iters):
        candidates = mean[None] + rng.standard_normal((pop, *mean.shape)).astype(np.float32) * std[None]
        costs = cost(decode(candidates))
        order = np.argsort(costs)
        elite = candidates[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0) * std_decay, min_std)
        if float(costs[order[0]]) < best_cost:
            best, best_cost = candidates[order[0]].copy(), float(costs[order[0]])
    return (mean.astype(np.float32), best_cost, best) if return_best else (mean.astype(np.float32), best_cost)
