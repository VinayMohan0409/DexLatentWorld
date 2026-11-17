"""Tests of the gripper-task pieces without a model: goal and cost of a stage, the planner on a stand-in model, the closed loop on a toy robot, the scene cloud on the real Stack environment."""
import sim_settings as cfg
import types

import numpy as np
import pytest

from lawm.sim.costs import SEG_HAND, SEG_OBJECT, SEG_TABLE
from lawm.sim.gripper import loop, planner, stage

STEPS, N = 5, 12
SEGMENT = np.array([SEG_HAND] * 4 + [SEG_OBJECT] * 4 + [SEG_TABLE] * 4)
POINTS = np.random.default_rng(0).normal(size=(N, 3)).astype(np.float32)


def test_goal_moves_only_the_object_rows():
    """The goal cloud shifts the object rows by (target - current) and leaves hand and table rows where they are."""
    goal = stage.object_translated_to(lambda: np.array([1.0, 0, 0]), lambda: np.array([0.25, 0.1, 0]))(POINTS, SEGMENT)
    assert np.allclose(goal[SEGMENT == SEG_OBJECT] - POINTS[SEGMENT == SEG_OBJECT], [0.75, -0.1, 0]) and np.array_equal(goal[SEGMENT != SEG_OBJECT], POINTS[SEGMENT != SEG_OBJECT])


def test_object_cost_is_zero_at_the_goal_and_counts_only_the_object():
    """A prediction that puts the object rows on the goal costs 0 whatever the hand does; one mm of error on the last step costs about 1 mm."""
    goal = POINTS.copy()
    goal[SEGMENT == SEG_OBJECT] += [0.1, 0, 0]
    cost, mask = stage.object_goal_cost(POINTS, goal, SEGMENT, np.eye(STEPS, dtype=np.float32)[-1])
    exact = np.zeros((1, STEPS, N, 3), np.float32)
    exact[:, :, SEGMENT == SEG_OBJECT] = [0.1, 0, 0]
    exact[:, :, SEGMENT == SEG_HAND] = [5.0, 5.0, 5.0]
    off = exact.copy()
    off[:, -1, SEGMENT == SEG_OBJECT, 0] += 0.001
    assert mask.sum() == 4 and cost(exact)[0] == pytest.approx(0, abs=1e-4) and cost(off)[0] == pytest.approx(1.0, abs=1e-3)


class StandInModel:
    """A model adapter stand-in: the decoder moves hand and object rows by 2 cm plus a sigmoid of the latent sum over 10, the same on every step, so a candidate with a larger latent sum goes further and none stands still."""

    prior = (np.zeros((STEPS, 4, 8), np.float32), np.full((STEPS, 4, 8), 0.5, np.float32))

    def decoder(self, coords, segment):
        """decode(z) for the cloud: (n, STEPS, N, 3) displacements."""
        def decode(z):
            """Rows move along x by 0.02 + sigmoid(sum z) / 10 metres on every step."""
            move = 0.02 + 1.0 / (1.0 + np.exp(-z.reshape(len(z), -1).sum(axis=1))) / 10.0
            out = np.zeros((len(z), STEPS, len(coords), 3), np.float32)
            out[:, :, segment == SEG_OBJECT, 0] = move[:, None, None]
            out[:, :, segment == SEG_HAND, 0] = move[:, None, None]
            return out
        return decode


def goal_for(shift):
    """A goal cloud: the object rows moved by shift."""
    goal = POINTS.copy()
    goal[SEGMENT == SEG_OBJECT] += shift
    return goal


def test_planner_moves_towards_a_reachable_goal_and_repeats_with_the_same_seed():
    """The search finds a plan that brings the object near a goal 5 cm away, and the same trial seed gives the same plan."""
    p = planner.LawmPlanner(StandInModel(), cfg.gripper_planner(pop=64, iters=4))
    p.reset(3)
    a = p.plan(POINTS, SEGMENT, goal_for([0.05, 0, 0]), stage.object_goal_cost)
    p.reset(3)
    b = p.plan(POINTS, SEGMENT, goal_for([0.05, 0, 0]), stage.object_goal_cost)
    assert np.array_equal(a["all_step_deltas"], b["all_step_deltas"]) and a["err_mm"] < 15 and not a["stayed"]
    assert a["all_step_deltas"].shape == (STEPS, N, 3)


def test_planner_holds_when_staying_is_cheapest():
    """When the goal is where the object already is, the hold candidate wins and the plan is all zeros."""
    p = planner.LawmPlanner(StandInModel(), cfg.gripper_planner())
    p.reset(0)
    plan = p.plan(POINTS, SEGMENT, goal_for([0.0, 0, 0]), stage.object_goal_cost)
    assert plan["stayed"] and not plan["all_step_deltas"].any() and plan["err_mm"] == pytest.approx(0, abs=1e-4)


def test_warm_start_is_used_and_reset_clears_it():
    """A stored mean seeds the next search when warm_start is on; reset() drops it; with warm_start off it is never used."""
    for warm, expected in ((True, 1.0), (False, 0.0)):
        seen = []

        class Recording(StandInModel):
            """Records the mean of the first latent batch of each replan."""

            def decoder(self, coords, segment):
                """The stand-in decoder, remembering the first batch."""
                inner, first = super().decoder(coords, segment), []

                def decode(z):
                    """Decode, noting the batch mean once per replan."""
                    if not first:
                        first.append(1)
                        seen.append(float(z.mean()))
                    return inner(z)
                return decode

        p = planner.LawmPlanner(Recording(), cfg.gripper_planner(warm_start=warm))
        p.reset(1)
        p.warm_start = np.full((STEPS, 4, 8), 1.0, np.float32)
        p.plan(POINTS, SEGMENT, goal_for([0.08, 0, 0]), stage.object_goal_cost)
        assert seen[0] == pytest.approx(expected, abs=0.1)
    p.reset(1)
    assert p.warm_start is None


class ToyEnv:
    """A robot that moves its end effector by 0.05 * action per step; the object is the end effector."""

    action_dim = 7

    def __init__(self):
        """Start at the origin."""
        self.pos = np.zeros(3)
        self.steps = 0
        self.sim = None

    def step(self, action):
        """Move and return the observation tuple of robosuite."""
        self.pos = self.pos + 0.05 * np.asarray(action[:3], dtype=np.float64)
        self.steps += 1
        return {"robot0_eef_pos": self.pos.copy(), "robot0_eef_quat": np.array([0, 0, 0, 1.0])}, 0.0, False, {}


class ToyScene:
    """The cloud of the toy robot: three hand rows, one object row, one table row, all at the end effector."""

    def __init__(self, env):
        """Keep the environment."""
        self.env = env

    def __call__(self, sim):
        """Points at the current end effector."""
        return np.tile(self.env.pos, (5, 1)).astype(np.float32), np.array([0, 0, 0, 1, 2])


class ToyPlanner:
    """Always plans the displacement towards the target (so the loop's clipping and stopping are what is tested)."""

    def __init__(self, env, target):
        """Keep the environment and the target."""
        self.env, self.target = env, np.asarray(target)

    def plan(self, points, segment, goal, cost_builder):
        """Cumulative displacements that are all the vector to the target."""
        return dict(all_step_deltas=np.tile(self.target - self.env.pos, (STEPS, 5, 1)).astype(np.float32), err_mm=0.0, stayed=False)


def toy_stage(env, target, **kwargs):
    """A stage whose error is the distance of the toy robot to the target."""
    return stage.Stage("toy", lambda points, segment: points, lambda e: float(np.linalg.norm(env.pos - target) * 1000), **kwargs)


def test_loop_converges_clips_each_step_and_reports_records():
    """Each replan moves at most max_step_mm; the loop stops as soon as the true error is under the threshold; a record per replan."""
    env, target = ToyEnv(), np.array([0.25, 0.0, 0.0])
    obs, info = loop.run_stage(env, {"robot0_eef_pos": env.pos.copy(), "robot0_eef_quat": np.array([0, 0, 0, 1.0])}, ToyPlanner(env, target), ToyScene(env),
                               toy_stage(env, target, converge_mm=20, max_replans=30), stage.object_goal_cost, -1, 60.0)
    assert info["converged"] and info["replans"] == len(info["records"]) and 4 <= info["replans"] <= 10
    assert max(np.linalg.norm(r["disp_mm"]) for r in info["records"]) <= 60.0 + 1e-3
    assert info["records"][-1]["true_err_mm"] < 20


def test_loop_stops_when_the_error_stalls():
    """A planner that never moves the robot ends the loop after stall_patience + 1 replans, not converged."""
    env, target = ToyEnv(), np.array([0.5, 0.0, 0.0])
    obs0 = {"robot0_eef_pos": env.pos.copy(), "robot0_eef_quat": np.array([0, 0, 0, 1.0])}
    stuck = types.SimpleNamespace(plan=lambda points, segment, goal, cost_builder: dict(all_step_deltas=np.zeros((STEPS, 5, 3), np.float32), err_mm=0.0, stayed=True))
    obs, info = loop.run_stage(env, obs0, stuck, ToyScene(env), toy_stage(env, target, max_replans=30, stall_patience=4), stage.object_goal_cost, -1, 60.0)
    assert not info["converged"] and info["replans"] == 5


def test_scene_cloud_of_the_real_stack_environment():
    """Rows are hand 2048, object 400, table 2048 with 110 points on each fingertip pad; calls repeat, the hand follows the arm and the table stays."""
    pytest.importorskip("robosuite")
    from lawm.sim.gripper import clouds, envs
    env = envs.build_env("Stack")
    env.reset()
    scene = clouds.stack_scene(env.sim)
    a, segment = scene(env.sim)
    names = [env.sim.model.geom_id2name(int(g)) or "" for g in scene.hand.owner]
    pads = [sum(n.endswith(side + "_fingertip_visual") for n in names) for side in ("left", "right")]
    assert a.shape == (4496, 3) and [int((segment == s).sum()) for s in (SEG_HAND, SEG_OBJECT, SEG_TABLE)] == [2048, 400, 2048] and pads == [110, 110]
    assert np.array_equal(a, scene(env.sim)[0])
    for _ in range(10):
        env.step(np.array([1.0, 0, 0, 0, 0, 0, -1.0]))
    b = scene(env.sim)[0]
    assert np.abs(b[segment == SEG_HAND] - a[segment == SEG_HAND]).max() > 0.01 and np.array_equal(b[segment == SEG_TABLE], a[segment == SEG_TABLE])
    env.close()


def test_runner_runs_stack_and_resumes(tmp_path):
    """A run with the stand-in model writes one result file per seed, summarises them, skips finished seeds without opening the environment, and refuses a changed config."""
    pytest.importorskip("robosuite")
    import json
    from dataclasses import replace
    from pathlib import Path

    from lawm.sim.gripper import envs, runner
    config = runner.load_config(Path(__file__).resolve().parents[1] / "configs" / "stage3" / "gripper.json")
    config = replace(config, tasks=("stack",), seeds=(0,), stack=replace(config.stack, replans=3))
    p = planner.LawmPlanner(StandInModel(), config.planner)
    table = runner.run_all(config, p, tmp_path)
    assert table["stack"]["trials"] == 1 and table["stack"]["setup_ok"] == 1 and (tmp_path / "stack_seed0.json").exists()
    assert json.loads((tmp_path / "stack_seed0.json").read_text())["replans"] <= 3
    real = envs.build_env
    envs.build_env = lambda *a, **k: (_ for _ in ()).throw(AssertionError("environment opened for a finished trial"))
    try:
        runner.run_all(config, p, tmp_path)
    finally:
        envs.build_env = real
    with pytest.raises(AssertionError):
        runner.run_all(replace(config, copies=32), p, tmp_path)


def test_config_file_loads_and_milk_horizon_follows_the_original_formula():
    """The shipped config loads; Milk's horizon is 1500 for 20 replans of 25 steps, the other tasks keep the default."""
    pytest.importorskip("robosuite")
    from pathlib import Path

    from lawm.sim.gripper import runner
    config = runner.load_config(Path(__file__).resolve().parents[1] / "configs" / "stage3" / "gripper.json")
    assert config.copies == 64 and runner.horizon("milk", config.milk) == 1500 and runner.horizon("stack", config.stack) is None
