"""Tests of the shared planner pieces: horizon weights, goal / floor / ManiSkill costs against hand-computed values, CEM on a toy problem, the execute loop on a fake plant, stall and mode switching."""
import numpy as np
import pytest

from lawm.sim import cem, control, costs


def test_horizon_weights_sum_to_one_and_keep_half_on_the_last_step():
    """horizon_weights(5) is [.05, .1, .15, .2, .5]."""
    assert np.allclose(costs.horizon_weights(5), [0.05, 0.1, 0.15, 0.2, 0.5]) and np.isclose(costs.horizon_weights(5).sum(), 1.0)
    assert np.array_equal(costs.terminal_weights(3), [0, 0, 1]) and np.array_equal(costs.step_weights(3, 0), [1, 0, 0])


def test_goal_cost_is_the_mean_point_distance_in_millimetres():
    """Two points each 3-4-5 mm... scaled: displacements of (0.03, 0.04, 0) from the goal give 50 mm; masks and weights select rows and steps."""
    coords0 = np.zeros((3, 3))
    goal = np.zeros((3, 3))
    deltas = np.zeros((2, 2, 3, 3))
    deltas[1, 1, :2] = [0.03, 0.04, 0.0]  # candidate 1, step 1, rows 0 and 1
    mask = np.array([True, True, False])
    assert np.allclose(costs.goal_cost_mm(deltas, coords0, goal, mask, costs.terminal_weights(2)), [0.0, 50.0])
    assert np.allclose(costs.goal_cost_mm(deltas, coords0, goal, mask, np.array([0.0, 0.5])), [0.0, 25.0])
    assert np.allclose(costs.goal_cost_mm(deltas, coords0, goal, np.array([False, False, True]), costs.terminal_weights(2)), [0.0, 0.0])


def test_floor_penalty_shape():
    """The barrier is 0 above the floor, a squared ramp up to 10 mm below it, and flat 5000 beyond; the worst step counts."""
    hand = np.array([True])
    d = np.zeros((4, 2, 1, 3))
    d[:, 1, 0, 2] = [0.0, -0.005, -0.010, -0.020]  # hand centroid dz at the last step, from a hand starting exactly at the floor
    assert np.allclose(costs.eef_floor_penalty_mm(d, hand, cur_eef_z=1.0, min_eef_z=1.0), [0.0, 200.0, 5000.0, 5000.0])


def push_scene():
    """A hand block just behind a unit-ish object along +x, a goal 0.1 m ahead, and the push pose."""
    coords0 = np.array([[-0.2, 0, 0], [-0.19, 0, 0], [0.0, 0, 0.0], [0.04, 0, 0.0], [5.0, 5.0, 0.0]])  # hand, hand, object, object, table
    goal = coords0.copy()
    goal[2:4, 0] += 0.1
    return coords0, goal, np.array([1, 1, 0, 0, 0], bool), np.array([0, 0, 1, 1, 0], bool), np.array([0, 0, 0, 0, 1], bool)


def test_maniskill_push_cost_rewards_reaching_then_placing():
    """With the leading hand face on the push pose and the object on the goal the cost is -(1 + 1 + 1) = -3; with the hand far it is about -(reach only)."""
    coords0, goal, hand, obj, static = push_scene()
    push_dir = np.array([1.0, 0, 0])
    half = 0.02
    offset = 0.015
    push_pose_x = 0.02 - (half + offset)  # object centre x = 0.02
    d = np.zeros((2, 1, 5, 3))
    d[0, 0, :2, 0] = push_pose_x - (-0.19) - 0.0  # leading face (the hand point at -0.19, top 20% of the projection) onto the push pose
    d[0, 0, 2:4, 0] = 0.1  # object onto the goal
    c = costs.maniskill_push_cost_mm(d, coords0, goal, hand, obj, np.array([1.0]), push_dir, offset, leading_face_frac=0.5)
    assert np.isclose(c[0], -3.0, atol=1e-6)
    assert c[1] > -1.0  # candidate 1 does nothing: hand 0.1+ m from the push pose, no place or height reward


def test_cem_finds_the_minimum_of_a_toy_cost_and_is_seeded():
    """On a quadratic bowl the final mean is near the minimum; the same seed gives the same result; the std never falls below min_std."""
    target = np.array([0.5, -0.3, 0.2])
    decode = lambda c: c.reshape(len(c), 1, 1, 3)  # noqa: E731
    cost = lambda d: np.linalg.norm(d[:, 0, 0] - target, axis=-1)  # noqa: E731
    run = lambda seed: cem.cem_iterate(decode, cost, np.zeros(3, np.float32), np.ones(3, np.float32), 64, 6, 0.2, 0.05, np.random.default_rng(seed))
    mean, best = run(0)
    assert np.linalg.norm(mean - target) < 0.15 and best < 0.3 and np.array_equal(run(0)[0], mean) and not np.array_equal(run(1)[0], mean)


def test_cem_returns_the_best_sample_on_request():
    """With return_best the lowest-cost candidate seen comes back and its cost equals the reported best."""
    target = np.array([1.0, 0, 0])
    decode = lambda c: c.reshape(len(c), 1, 1, 3)  # noqa: E731
    cost = lambda d: np.linalg.norm(d[:, 0, 0] - target, axis=-1)  # noqa: E731
    mean, best_cost, best = cem.cem_iterate(decode, cost, np.zeros(3, np.float32), np.ones(3, np.float32), 32, 3, 0.25, 0.05, np.random.default_rng(0), return_best=True)
    assert np.isclose(np.linalg.norm(best - target), best_cost)


class FakePlant:
    """An end effector that moves by 0.05 m times the commanded action each step (OSC's output_max), with no lag."""

    def __init__(self):
        """Start at the origin with the identity orientation."""
        self.pos = np.zeros(3)
        self.steps = 0

    def obs(self):
        """The observation dict the loop reads."""
        return {"robot0_eef_pos": self.pos.copy(), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}

    def step(self, action):
        """Apply the position channels."""
        self.pos = self.pos + 0.05 * action[:3]
        self.steps += 1
        return self.obs(), 0.0, False, {}


def test_execute_displacement_tracks_the_ramp_with_a_small_lag():
    """A proportional loop lags a ramp by a fixed fraction: after 25 sub-steps the end effector has covered 95-100 % of the displacement, in its direction, and on_step is called every step."""
    plant, seen = FakePlant(), []
    disp = np.array([0.06, -0.02, 0.01])
    obs = control.execute_displacement(plant, plant.obs(), disp, 25, 1.0, 7, on_step=lambda i, o: seen.append(i))
    moved = obs["robot0_eef_pos"]
    assert 0.95 < moved @ disp / (disp @ disp) < 1.0 and np.linalg.norm(moved - disp) < 0.05 * np.linalg.norm(disp) and plant.steps == 25 and seen == list(range(25))


def test_execute_displacement_clips_the_command():
    """A distant sub-target saturates the action at 1, so one step moves at most 0.05 m."""
    plant = FakePlant()
    control.execute_displacement(plant, plant.obs(), np.array([2.0, 0.0, 0.0]), 1, 0.0, 6)
    assert np.isclose(plant.pos[0], 0.05)


def test_select_step_delta_averages_the_mask_rows_of_one_step():
    """The last step's mean displacement over the hand rows is returned."""
    deltas = np.zeros((3, 4, 3))
    deltas[2, :2] = [[0.1, 0, 0], [0.3, 0, 0]]
    assert np.allclose(control.select_step_delta({"all_step_deltas": deltas}, -1, np.array([1, 1, 0, 0], bool)), [0.2, 0, 0])


def test_stall_detector_compares_with_the_previous_value():
    """Improvements smaller than tol count as no progress; patience in a row triggers; a real improvement resets."""
    s = control.StallDetector(patience=3, tol=0.5)
    assert [s.update(v) for v in (10.0, 9.0, 8.9, 8.8, 8.7)] == [False, False, False, False, True]
    s = control.StallDetector(patience=2, tol=0.5)
    assert [s.update(v) for v in (10.0, 10.0, 8.0, 8.0, 8.0)] == [False, False, False, False, True]
