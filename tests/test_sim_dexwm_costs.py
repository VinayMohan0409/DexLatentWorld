"""Tests of the DexWM planner costs: known values for each term, rigid-motion invariance, the antipodal grasp penalty, the weighted sum and the finger mask."""
import numpy as np
import pytest

from lawm.sim.costs import horizon_weights, terminal_weights
from lawm.sim.dexwm.planning import plan_costs as C

STEPS = 5
LAST = terminal_weights(STEPS)


def deltas(n_points, pop=1):
    """Zero displacements (pop, STEPS, n_points, 3)."""
    return np.zeros((pop, STEPS, n_points, 3))


def test_hand_cost_is_the_mean_point_distance_in_mm_at_the_weighted_steps():
    """A hand 10 mm from its goal costs 10 at the last step; a weight on step 1 only reads step 1."""
    coords0 = np.zeros((3, 3))
    d = deltas(3)
    d[0, :, :, 0] = 0.010
    d[0, 0, :, 0] = 0.004
    assert C.hand_goal_cost_mm(d, coords0, coords0, np.ones(3, bool), LAST)[0] == pytest.approx(10.0)
    first = np.array([1.0, 0, 0, 0, 0])
    assert C.hand_goal_cost_mm(d, coords0, coords0, np.ones(3, bool), first)[0] == pytest.approx(4.0)


def test_point_weights_pull_the_mean_towards_the_heavy_points():
    """Errors 10 and 30 mm with point weights 1 and 3 give (10 + 3 * 30) / 4 = 25."""
    coords0 = np.zeros((2, 3))
    d = deltas(2)
    d[0, -1, 0, 0], d[0, -1, 1, 0] = 0.010, 0.030
    cost = C.hand_goal_cost_mm(d, coords0, coords0, np.ones(2, bool), LAST, point_weights=np.array([1.0, 3.0], dtype=np.float32))
    assert cost[0] == pytest.approx(25.0)
    assert C.hand_goal_cost_mm(d, coords0, coords0, np.ones(2, bool), LAST)[0] == pytest.approx(20.0)


def test_object_still_is_zero_without_motion_and_the_mean_displacement_otherwise():
    """Moving all object points 5 mm costs 5; a still object costs 0."""
    d = deltas(4, pop=2)
    d[1, -1, :, 2] = 0.005
    cost = C.object_still_cost_mm(d, np.ones(4, bool), LAST)
    assert cost[0] == 0.0 and cost[1] == pytest.approx(5.0)


def test_object_goal_is_the_centroid_distance_in_mm():
    """An object centroid 20 mm from the goal centroid costs 20, and 0 at the goal."""
    start, goal = np.zeros(3), np.array([0.0, 0.0, 0.020])
    d = deltas(5, pop=2)
    d[1, -1, :, 2] = 0.020
    cost = C.object_goal_cost_mm(d, np.ones(5, bool), LAST, goal, start)
    assert cost[0] == pytest.approx(20.0) and cost[1] == pytest.approx(0.0)


def rigid(rng):
    """A random rotation and translation."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q)), rng.standard_normal(3) * 0.05


def test_kabsch_recovers_a_rigid_transform():
    """Rotations and translations of a point set are recovered exactly, also for a batch."""
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((30, 3))
    moved = []
    for _ in range(4):
        r, t = rigid(rng)
        moved.append(ref @ r.T + t)
    r_hat, t_hat = C.kabsch_batched(ref, np.array(moved))
    assert np.allclose(np.einsum("pij,nj->pni", r_hat, ref) + t_hat[:, None, :], np.array(moved), atol=1e-10)
    assert np.allclose(np.linalg.det(r_hat), 1.0)


def test_rigid_attachment_is_zero_when_hand_and_object_move_together_and_the_object_shift_when_the_hand_stays():
    """The hand riding rigidly on the object costs 0; an object moved 10 mm with the hand staying costs 10."""
    rng = np.random.default_rng(1)
    obj, hand = rng.standard_normal((40, 3)) * 0.03, rng.standard_normal((20, 3)) * 0.03 + 0.05
    coords0 = np.concatenate([hand, obj])
    hand_mask = np.arange(60) < 20
    r, t = rigid(rng)
    d = deltas(60, pop=2)
    d[0, -1] = coords0 @ r.T + t - coords0
    d[1, -1, 20:, 0] = 0.010
    cost = C.rigid_attachment_cost_mm(hand, obj, d, hand_mask, ~hand_mask, LAST)
    assert cost[0] == pytest.approx(0.0, abs=1e-6) and cost[1] == pytest.approx(10.0, abs=1e-6)


def test_grasp_tight_is_the_finger_distance_to_the_object_centre():
    """Fingers 30 mm from the object centre cost 30; only the selected finger rows count."""
    obj_c = np.zeros(3)
    fingers = np.array([[0.03, 0, 0], [-0.03, 0, 0], [0.5, 0.5, 0.5]])
    coords0 = np.concatenate([fingers, np.zeros((2, 3))])
    hand_mask = np.arange(5) < 3
    select = np.array([True, True, False])
    d = deltas(5)
    cost = C.grasp_tight_cost_mm(d, hand_mask, ~hand_mask, LAST, obj_c, coords0[hand_mask][select], select)
    assert cost[0] == pytest.approx(30.0)
    everything = C.grasp_tight_cost_mm(d, hand_mask, ~hand_mask, LAST, obj_c, coords0[hand_mask], None)
    assert everything[0] > cost[0]


def grasp_scene(finger_positions):
    """An object of points at (+-0.02, 0, 0) and the given hand points, with no displacement."""
    obj = np.array([[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]] * 25)
    hand = np.asarray(finger_positions, dtype=float)
    return hand, obj, deltas(len(hand) + 50), np.arange(len(hand) + 50) < len(hand)


def test_grasp_penalty_is_zero_for_two_opposite_contacts_and_one_otherwise():
    """Two touching points on opposite sides of the object score 0; one touching point, or none, score 1."""
    for fingers, expected in (([[0.05, 0, 0], [-0.05, 0, 0]], 0.0), ([[0.05, 0, 0], [0.5, 0, 0]], 1.0), ([[0.5, 0, 0], [-0.5, 0, 0]], 1.0)):
        hand, obj, d, hand_mask = grasp_scene(fingers)
        assert C.grasp_quality_penalty(hand, obj, d, hand_mask, ~hand_mask)[0] == pytest.approx(expected, abs=1e-6)


def test_grasp_penalty_is_one_minus_the_cosine_of_the_best_opposition():
    """Contacts pointing at the object along (-0.03, 0, 0) and (0.02, -0.03, 0) are 124 degrees apart: penalty 1 - 0.5547."""
    hand, obj, d, hand_mask = grasp_scene([[0.05, 0, 0], [-0.04, 0.03, 0]])
    expected = 1.0 - 0.0006 / (0.03 * np.hypot(0.02, 0.03))
    assert C.grasp_quality_penalty(hand, obj, d, hand_mask, ~hand_mask)[0] == pytest.approx(expected, abs=1e-6)


def test_grasp_penalty_is_one_for_contacts_on_the_same_side():
    """Two touching points side by side point the same way, so nothing is opposed."""
    hand, obj, d, hand_mask = grasp_scene([[0.05, 0.01, 0], [0.05, -0.01, 0]])
    assert C.grasp_quality_penalty(hand, obj, d, hand_mask, ~hand_mask)[0] == pytest.approx(1.0)


def test_grasp_penalty_ignores_the_points_outside_the_finger_selection():
    """With no point selected there is no contact."""
    hand, obj, d, hand_mask = grasp_scene([[0.05, 0, 0], [-0.05, 0, 0]])
    assert C.grasp_quality_penalty(hand, obj, d, hand_mask, ~hand_mask, finger_select=np.array([False, False]))[0] == 1.0


def plan_inputs():
    """A small scene: 6 hand points (4 on fingers), 8 object points, 5 table points."""
    rng = np.random.default_rng(3)
    coords0 = np.concatenate([rng.standard_normal((6, 3)) * 0.04, rng.standard_normal((8, 3)) * 0.03 + 0.02, rng.standard_normal((5, 3))])
    segment = np.array([0] * 6 + [1] * 8 + [2] * 5)
    goal = coords0 + 0.05
    finger_mask = np.array([True, True, True, True, False, False])
    d = rng.standard_normal((4, STEPS, 19, 3)) * 0.01
    return coords0, goal, segment, finger_mask, d


def test_plan_cost_with_only_the_hand_term_equals_the_scaled_hand_cost():
    """Extra terms at weight 0 add nothing and hand_scale multiplies the hand term."""
    coords0, goal, segment, finger_mask, d = plan_inputs()
    horizon = horizon_weights(STEPS)
    base = C.hand_goal_cost_mm(d, coords0, goal, segment == 0, horizon)
    assert np.array_equal(C.PlanCost(coords0, goal, segment, finger_mask, C.CostWeights())(d), base)
    assert np.allclose(C.PlanCost(coords0, goal, segment, finger_mask, C.CostWeights(hand_scale=0.5))(d), 0.5 * base)
    assert np.all(C.PlanCost(coords0, goal, segment, finger_mask, C.CostWeights(hand_scale=0.0))(d) == 0.0)


def test_plan_cost_is_the_weighted_sum_of_its_terms_and_follows_changed_weights():
    """The total equals the hand cost plus each term times its weight, and editing the weights changes the next call."""
    coords0, goal, segment, finger_mask, d = plan_inputs()
    horizon = horizon_weights(STEPS)
    hand_mask, obj_mask = segment == 0, segment == 1
    weights = C.CostWeights(object_still=2.0, object_goal=3.0)
    cost = C.PlanCost(coords0, goal, segment, finger_mask, weights)
    expected = (C.hand_goal_cost_mm(d, coords0, goal, hand_mask, horizon) + 2.0 * C.object_still_cost_mm(d, obj_mask, horizon)
                + 3.0 * C.object_goal_cost_mm(d, obj_mask, horizon, goal[obj_mask].mean(axis=0), coords0[obj_mask].mean(axis=0)))
    assert np.allclose(cost(d), expected)
    weights.object_goal = 0.0
    assert not np.allclose(cost(d), expected)
    weights.fingertip = 2.0
    assert not np.allclose(cost(d), C.PlanCost(coords0, goal, segment, None, weights)(d))


def test_the_hold_candidate_costs_the_distance_from_the_start_to_the_goal():
    """Zero displacement scores the start-to-goal distance, which is what the planner compares the sampled candidates with."""
    coords0, goal, segment, finger_mask, _ = plan_inputs()
    cost = C.PlanCost(coords0, goal, segment, finger_mask, C.CostWeights())(np.zeros((1, STEPS, 19, 3)))
    assert cost[0] == pytest.approx(np.linalg.norm(goal[segment == 0] - coords0[segment == 0], axis=1).mean() * 1000.0)


XML = """
<mujoco>
  <worldbody>
    <body name="gripper0_right_hand" pos="0 0 1">
      <geom name="palm" type="box" size="0.05 0.05 0.01"/>
      <body name="gripper0_right_link_0.0" pos="0.1 0 0">
        <geom name="finger" type="box" size="0.01 0.01 0.03"/>
      </body>
    </body>
    <body name="other" pos="1 0 0"><geom name="table" type="box" size="0.3 0.3 0.01"/></body>
  </worldbody>
</mujoco>
"""


def test_finger_mask_marks_points_of_finger_bodies_and_not_the_palm():
    """Points on the root body of the hand are palm; points on any other hand body are fingers."""
    mujoco = pytest.importorskip("mujoco")
    from lawm.sim.dexwm.scene.hand_surface import hand_finger_mask
    model = mujoco.MjModel.from_xml_string(XML)
    geom = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(model.ngeom)}
    owner = np.array([geom["palm"], geom["finger"], geom["finger"], geom["palm"]])
    assert hand_finger_mask(model, owner).tolist() == [False, True, True, False]
