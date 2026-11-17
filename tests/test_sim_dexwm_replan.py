"""Tests of one replan: the search budget and decode calls, the elite-mean plan, the hold candidate (strictly cheaper wins), the random generators and the absence of state between replans."""
import sim_settings as cfg
import numpy as np
import pytest

from lawm.sim.dexwm.planning import plan_costs as C
from lawm.sim.dexwm.planning import cem_replan as R

STEPS, QUERIES, Z_DIM = 5, 4, 8
PRIOR_MEAN = np.zeros((STEPS, QUERIES, Z_DIM), dtype=np.float32)
PRIOR_STD = np.full((STEPS, QUERIES, Z_DIM), 0.5, dtype=np.float32)


def scene(goal_is_start=False):
    """6 hand points, 4 object points and 2 table points; the goal moves the hand 0.1 m along x unless goal_is_start."""
    coords0 = np.concatenate([np.random.default_rng(0).standard_normal((6, 3)) * 0.02, np.full((4, 3), 0.3), np.full((2, 3), 1.0)])
    segment = np.array([0] * 6 + [1] * 4 + [2] * 2)
    goal = coords0.copy()
    if not goal_is_start:
        goal[:6, 0] += 0.1
    return coords0, goal, segment


def recorder(direction, still=False):
    """A stand-in decoder (all points move along direction by a sigmoid of the latent sum) and the list of batch sizes it was called with."""
    calls = []

    def decode(z):
        """Latents (n, STEPS, QUERIES, Z_DIM) to displacements (n, STEPS, N, 3)."""
        calls.append(len(z))
        progress = np.zeros(len(z)) if still else 1.0 / (1.0 + np.exp(-z.reshape(len(z), -1).sum(axis=1) * 0.1))
        return (progress[:, None, None, None] * ((np.arange(STEPS) + 1) / STEPS)[None, :, None, None] * direction[None, None]).astype(np.float32)
    return decode, calls


def plan(goal_is_start=False, settings=None, seed=0, still=False, weights=None):
    """One replan on the small scene; returns (plan, decode calls)."""
    coords0, goal, segment = scene(goal_is_start)
    decode, calls = recorder(np.full(coords0.shape, 0.05) if goal_is_start else goal - coords0, still)
    cost = C.PlanCost(coords0, goal, segment, None, weights or C.CostWeights())
    result = R.plan_replan(decode, cost, PRIOR_MEAN, PRIOR_STD, settings or cfg.cem(pop=16, iters=3), np.random.default_rng(seed))
    return result, calls


def test_the_decoder_sees_pop_candidates_per_iteration_and_nothing_more():
    """iters calls with pop candidates; the plan reuses the best sampled prediction, so there is no extra decode."""
    _, calls = plan(settings=cfg.cem(pop=16, iters=3))
    assert calls == [16, 16, 16]


def test_more_iterations_never_make_the_best_sampled_cost_worse():
    """The first iteration is the same draw, so the best cost over more iterations is at most that of one."""
    one, _ = plan(settings=cfg.cem(pop=16, iters=1))
    many, _ = plan(settings=cfg.cem(pop=16, iters=6))
    assert many.best_sample_cost <= one.best_sample_cost


def test_the_plan_moves_towards_the_goal_and_beats_holding_when_the_goal_is_away():
    """With a reachable goal the best sampled candidate is chosen, its cost is the best sampled cost, and its predicted final hand error is below the start-to-goal distance."""
    result, _ = plan(settings=cfg.cem(pop=32, iters=4))
    assert result.executed == "best_sample" and result.delta.shape == (STEPS, 12, 3) and result.cost == result.best_sample_cost
    assert result.final_hand_error_mm < 100.0


def test_holding_wins_when_the_goal_is_the_start_and_is_zero_motion():
    """Every motion the decoder offers moves the hand away from its goal, so no motion (cost 0) wins and the plan is zeros."""
    result, _ = plan(goal_is_start=True)
    assert result.executed == "hold" and result.cost == 0.0 and not result.delta.any() and result.final_hand_error_mm == 0.0


def test_holding_must_be_strictly_cheaper_and_can_be_switched_off():
    """A decoder that never moves ties with holding, so the best sample is kept; with hold_candidate False holding never replaces the plan."""
    tied, _ = plan(still=True)
    assert tied.executed == "best_sample"
    off, _ = plan(goal_is_start=True, settings=cfg.cem(pop=16, iters=3, hold_candidate=False))
    assert off.executed == "best_sample"


def test_the_same_generator_gives_the_same_plan_and_another_gives_another():
    """Equal seeds repeat the plan exactly; the search depends on the draws."""
    a, _ = plan(seed=1)
    b, _ = plan(seed=1)
    c, _ = plan(seed=2)
    assert np.array_equal(a.delta, b.delta) and a.cost == b.cost
    assert not np.array_equal(a.delta, c.delta)


def test_replan_generators_differ_by_seed_and_replan_and_repeat():
    """A generator is fixed by the pair (episode seed, replan index)."""
    draw = lambda seed, replan: R.replan_rng(seed, replan).standard_normal(4)  # noqa: E731
    assert np.array_equal(draw(3, 5), draw(3, 5))
    assert not np.array_equal(draw(3, 5), draw(3, 6)) and not np.array_equal(draw(3, 5), draw(4, 5))
    assert not np.array_equal(draw(1, 2), draw(2, 1))


def test_the_prior_is_not_modified_and_its_spread_has_a_floor():
    """The inputs are untouched; a zero prior spread still gives different candidates because of min_std."""
    coords0, goal, segment = scene()
    seen = []
    decode, _ = recorder(goal - coords0)

    def watching(z):
        """Remember the latents of the first batch."""
        seen.append(z.copy())
        return decode(z)
    mean, std = PRIOR_MEAN.copy(), np.zeros_like(PRIOR_STD)
    R.plan_replan(watching, C.PlanCost(coords0, goal, segment, None, C.CostWeights()), mean, std, cfg.cem(pop=16, iters=2), np.random.default_rng(0))
    assert np.array_equal(mean, PRIOR_MEAN) and not std.any()
    assert seen[0].std(axis=0).min() > 0.0


def test_every_replan_starts_from_the_prior_and_keeps_no_state():
    """Planning twice with fresh generators of the same seed gives the same plan, whatever happened in between."""
    first, _ = plan(seed=7)
    plan(seed=8)
    again, _ = plan(seed=7)
    assert np.array_equal(first.delta, again.delta)


def test_the_predicted_final_hand_error_is_the_mean_hand_point_distance_at_the_last_step():
    """The reported error equals the L2 distance of the hand points to their goal after the last step, in mm."""
    coords0, goal, segment = scene()
    result, _ = plan(settings=cfg.cem(pop=16, iters=3))
    final = coords0[:6] + result.delta[-1][:6]
    assert result.final_hand_error_mm == pytest.approx(np.linalg.norm(final - goal[:6], axis=1).mean() * 1000.0, rel=1e-5)
