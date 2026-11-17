"""Tests of the task planner: the gate and weights of each task, the executed hand points, the budgets, and the repeatability of a replan."""
import sim_settings as cfg
import numpy as np
import pytest

from lawm.sim.dexwm.planning import plan_costs as C
from lawm.sim.dexwm.planning import task_planner as P
from lawm.sim.dexwm.planning import phase_gates as S

STEPS, QUERIES, Z_DIM = 5, 4, 8
PRIOR = (np.zeros((STEPS, QUERIES, Z_DIM), dtype=np.float32), np.full((STEPS, QUERIES, Z_DIM), 0.5, dtype=np.float32))
SEGMENT = np.array([0] * 6 + [1] * 4 + [2] * 2)


def scene():
    """The cloud (6 hand, 4 object, 2 table points) and a goal that moves the hand 0.1 m along x."""
    coords = np.concatenate([np.random.default_rng(0).standard_normal((6, 3)) * 0.02, np.full((4, 3), 0.3), np.full((2, 3), 1.0)])
    goal = coords.copy()
    goal[:6, 0] += 0.1
    return coords, goal


def decode(z):
    """A stand-in decoder: every point moves along x by a sigmoid of the latent sum, later steps further."""
    progress = 1.0 / (1.0 + np.exp(-z.reshape(len(z), -1).sum(axis=1) * 0.1))
    out = np.zeros((len(z), STEPS, len(SEGMENT), 3), dtype=np.float32)
    out[..., 0] = (progress[:, None, None] * ((np.arange(STEPS) + 1) / STEPS)[None, :, None]).astype(np.float32) * 0.05
    return out


class Model:
    """A model adapter stand-in: the prior, and a decoder per observed cloud (recording the clouds it was asked for)."""

    prior = PRIOR

    def __init__(self):
        """No clouds seen yet."""
        self.seen = []

    def decoder(self, coords, segment):
        """The stand-in decode function for this cloud."""
        self.seen.append((coords, segment))
        return decode


def planner(task, seed=0, settings=None):
    """A planner of the task on the small scene."""
    coords, goal = scene()
    return P.TaskPlanner(task, goal, SEGMENT, None, Model(), seed, settings or cfg.cem(pop=16, iters=2), cfg.weights(task), cfg.gates()), coords


def test_gate_follows_the_task():
    """Grasp has the hover gate, place the secure gate, reach none."""
    assert isinstance(planner("grasp")[0].gate, S.HoverGate) and isinstance(planner("place")[0].gate, S.SecureGate) and planner("reach")[0].gate is None


def test_budgets_per_task():
    """Reach and grasp search 64 x 4, place 128 x 6."""
    assert (cfg.cem("reach").pop, cfg.cem("reach").iters) == (64, 4) and cfg.cem("grasp") == cfg.cem("reach")
    assert (cfg.cem("place").pop, cfg.cem("place").iters) == (128, 6)


def test_executed_points_are_the_first_step_of_the_plan():
    """The returned hand points are the observed hand plus the first predicted step, float32, hand points only."""
    p, coords = planner("reach")
    hand, plan = p.plan(coords, 0)
    assert hand.shape == (6, 3) and hand.dtype == np.float32
    assert np.allclose(hand, coords[:6] + plan.delta[0][:6], atol=1e-6)


def test_decoder_is_built_from_the_observed_cloud_of_each_replan():
    """Each replan asks the model for a decoder of the cloud observed then, with the episode's segment labels."""
    p, coords = planner("reach")
    p.plan(coords, 0)
    later = coords + 0.01
    p.plan(later, 1)
    assert np.array_equal(p.model.seen[0][0], coords) and np.array_equal(p.model.seen[1][0], later) and np.array_equal(p.model.seen[1][1], SEGMENT)


def test_same_seed_and_replan_repeat_and_other_replans_differ():
    """A replan is exactly repeatable; the next replan draws other noise."""
    a, coords = planner("grasp", seed=3)
    b, _ = planner("grasp", seed=3)
    assert np.array_equal(a.plan(coords, 0)[0], b.plan(coords, 0)[0])
    assert not np.array_equal(a.plan(coords, 0)[1].delta, a.plan(coords, 1)[1].delta)
    c, _ = planner("grasp", seed=4)
    assert not np.array_equal(a.plan(coords, 0)[1].delta, c.plan(coords, 0)[1].delta)


def test_weights_and_goal_used_by_each_task(monkeypatch):
    """Reach uses fingertip 2.0 and object still 1.0 on the real goal; grasp uses the raised goal until the hover ends; place uses the weights of its current stage."""
    seen = []
    real = P.PlanCost
    monkeypatch.setattr(P, "PlanCost", lambda *a: seen.append(a) or real(*a))
    p, coords = planner("reach")
    p.plan(coords, 0)
    assert seen[-1][4] == cfg.weights("reach") and np.array_equal(seen[-1][1], p.goal)
    g, _ = planner("grasp")
    g.plan(coords, 0)
    assert seen[-1][4] == cfg.weights("grasp") and seen[-1][1][0, 2] == pytest.approx(p.goal[0, 2] + 0.2)
    g.gate.active = False
    g.plan(coords, 1)
    assert np.array_equal(seen[-1][1], g.goal)
    q, _ = planner("place")
    q.plan(coords, 0)
    assert seen[-1][4] == cfg.weights("place")[0] 
    q.gate.stage = 2
    q.plan(coords, 1)
    assert seen[-1][4] == cfg.weights("place")[1]


def test_update_reports_a_phase_change_only_with_a_gate():
    """Reach never changes phase; grasp does once the observed hand is at the hover goal."""
    p, coords = planner("reach")
    assert p.update(coords, 0) is False
    g, _ = planner("grasp")
    assert g.update(coords, 0) is False
    near = coords.copy()
    near[:6] = g.gate.hover_goal[:6]
    assert g.update(near, 1) is True and g.gate.active is False


def test_given_weights_are_used(monkeypatch):
    """Weights passed in are used for reach, and for place as the weights of its two stages."""
    seen = []
    real = P.PlanCost
    monkeypatch.setattr(P, "PlanCost", lambda *a: seen.append(a) or real(*a))
    coords, goal = scene()
    mine = C.CostWeights(fingertip=5.0)
    P.TaskPlanner("reach", goal, SEGMENT, None, Model(), 0, cfg.cem(pop=8, iters=1), mine, cfg.gates()).plan(coords, 0)
    assert seen[-1][4] is mine
    stages = (C.CostWeights(hand_scale=0.0, grasp_quality=1.0), C.CostWeights(hand_scale=0.0, object_goal=1.0))
    placer = P.TaskPlanner("place", goal, SEGMENT, None, Model(), 0, cfg.cem(pop=8, iters=1), stages, cfg.gates())
    placer.plan(coords, 0)
    assert seen[-1][4] is stages[0]
    placer.gate.stage = 2
    placer.plan(coords, 1)
    assert seen[-1][4] is stages[1]
