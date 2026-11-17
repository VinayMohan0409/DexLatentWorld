"""Tests of the Push and Reach tasks and the PointWorld planner's pieces: the push room limit, the cost builders, the spline trajectories and the rows of the plan."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
from lawm.sim.costs import SEG_HAND, SEG_OBJECT, SEG_TABLE  # noqa: E402
from lawm.sim.gripper import push, reach, runner  # noqa: E402
from lawm.sim.gripper.pointworld import planner as pw, rollout  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stage3" / "gripper.json"
SEGMENT = np.r_[np.full(4, SEG_HAND), np.full(3, SEG_OBJECT), np.full(2, SEG_TABLE)].astype(np.int8)


def test_push_room_is_limited_by_the_table_edge_and_the_reach():
    """Along +x the object stops 60 mm before the table edge, or at the arm's reach when that is nearer."""
    assert push.max_travel_mm(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-0.56, 0.0])) == pytest.approx(240.0)
    assert push.max_travel_mm(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-0.70, 0.0])) == pytest.approx(100.0)
    assert push.max_travel_mm(np.array([0.3, 0.0]), np.array([1.0, 0.0]), np.array([-0.6, 0.0])) == 0.0


def test_approach_cost_scores_the_hand_and_the_floor():
    """The approach cost is zero at the hand's goal above the floor, grows with the hand's distance and with going under the floor."""
    coords = np.zeros((9, 3), np.float32)
    goal = coords.copy()
    cost, mask = push.approach_cost(1.0, 0.0, 1.0, 10.0)(coords, goal, SEGMENT, np.array([1.0]))
    assert mask.tolist() == (SEGMENT == SEG_HAND).tolist() and cost(np.zeros((1, 1, 9, 3)))[0] == 0.0
    far = np.zeros((1, 1, 9, 3))
    far[..., 0] = 0.01
    assert cost(far)[0] == pytest.approx(10.0)
    low = np.zeros((1, 1, 9, 3))
    low[:, :, SEGMENT == SEG_HAND, 2] = -2.0
    assert cost(low)[0] > 2000.0


def test_push_cost_is_goal_masked_on_the_object():
    """The push builder reports the object rows as its goal, and reaching the push pose lowers the cost."""
    coords = np.zeros((9, 3), np.float32)
    coords[SEGMENT == SEG_OBJECT, 0] = [-0.01, 0.0, 0.01]
    coords[SEGMENT == SEG_HAND, 0] = -0.3
    goal = coords.copy()
    goal[SEGMENT == SEG_OBJECT, 0] += 0.2
    cost, mask = push.push_cost(push.PUSH_DIR, 0.015)(coords, goal, SEGMENT, np.array([1.0]))
    assert mask.tolist() == (SEGMENT == SEG_OBJECT).tolist()
    near = np.zeros((1, 1, 9, 3))
    near[:, :, SEGMENT == SEG_HAND, 0] = 0.275
    assert cost(near)[0] < cost(np.zeros((1, 1, 9, 3)))[0]


def test_config_holds_the_recorded_settings():
    """Puck and cube keep their recorded settings, the four final tasks are listed, and every section loads."""
    config = runner.load_config(CONFIG)
    assert config.tasks == ("stack", "milk", "push_puck", "push_cube")
    assert (config.push_puck.execute_step_index, config.push_puck.approach_replans, config.push_puck.replans, config.push_puck.scripted_approach) == (1, 50, 15, False)
    assert (config.push_cube.execute_step_index, config.push_cube.approach_replans, config.push_cube.replans, config.push_cube.scripted_approach) == (0, 100, 50, True)
    assert config.reach.angles_deg == (0.0, 72.0, 144.0, 216.0, 288.0) and isinstance(config.reach, reach.ReachSettings)
    assert (config.pointworld.pop, config.pointworld.iters, config.pointworld.batch, config.pointworld.outward_normals) == (64, 4, 32, False)
    assert json.loads(CONFIG.read_text())["planner"]["elite_frac"] == 0.1


def test_spline_trajectory_starts_moving_at_the_first_row_and_is_clamped():
    """Row 0 is the displacement after step 1, not the zero anchor; a large knot is clamped to the largest reach."""
    basis = pw.spline_basis(10, 4)
    knots = np.zeros((1, 4, 3), np.float32)
    knots[0, 1:, 0] = 1.0
    trajectory = pw.knots_to_trajectory(knots, basis, 0.3, 0.35)
    assert trajectory.shape == (1, 10, 3) and trajectory[0, 0, 0] > 0
    assert np.linalg.norm(pw.knots_to_trajectory(knots * 100, basis, 0.3, 0.35), axis=-1).max() <= 0.35 + 1e-6


def test_plan_rows_follow_the_segments():
    """Hand rows take the trajectory, object and table rows their predicted means."""
    trajectory = np.full((1, 2, 3), 0.1, np.float32)
    deltas = pw.predicted_deltas(trajectory, np.full((1, 2, 3), 0.2, np.float32), np.full((1, 2, 3), 0.3, np.float32), SEGMENT)
    assert deltas.shape == (1, 2, 9, 3)
    assert np.allclose(deltas[0, 0, SEGMENT == SEG_HAND], 0.1) and np.allclose(deltas[0, 0, SEGMENT == SEG_OBJECT], 0.2) and np.allclose(deltas[0, 0, SEGMENT == SEG_TABLE], 0.3)


def test_object_marker_survives_scaling_and_the_robot_is_removed():
    """Painted object rows are found on either colour scale, and robot rows leave the observation."""
    colors = np.random.default_rng(0).integers(0, 200, (6, 3)).astype(np.uint8)
    mask = np.array([1, 1, 0, 0, 0, 0], bool)
    painted = rollout.marked_colors(colors, mask)
    assert rollout.recover_object_mask(painted).tolist() == mask.tolist() and rollout.recover_object_mask(painted / 255.0).tolist() == mask.tolist()
    obs = dict(scene_pts=np.zeros((6, 3)), scene_colors=colors, scene_normals=np.zeros((6, 3)), scene_obj_mask=mask, scene_is_robot=~mask & np.array([0, 0, 1, 1, 0, 0], bool))
    kept, removed = rollout.without_robot(obs)
    assert removed == 2 and len(kept["scene_pts"]) == 4
