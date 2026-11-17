"""Tests of the episode loop: the success counter across replans, each stop reason and its order, the check at the end, the records, and the call order with the planner."""
import sim_settings as cfg
import numpy as np
import pytest

pytest.importorskip("mujoco")
from lawm.sim.dexwm.control.executor import ExecSettings  # noqa: E402
from lawm.sim.dexwm.env.observation import Observation  # noqa: E402
from lawm.sim.dexwm import closed_loop as L  # noqa: E402


class Env:
    """A simulator stand-in: the success flag and the task distance (m) after each control step come from functions of the step number."""

    def __init__(self, flag=lambda i: False, distance=lambda i: 1.0 - 0.001 * i):
        """Keep the scripts and count the steps and observations."""
        self.flag, self.dist, self.steps, self.observations = flag, distance, 0, 0

    def observe(self):
        """A fixed observation."""
        self.observations += 1
        return Observation(np.zeros((4, 3)), np.eye(4), np.zeros(16), np.zeros(3), np.eye(3))

    def step(self, action):
        """Count the step and return its scripted flag."""
        self.steps += 1
        return self.flag(self.steps)

    def distance(self):
        """The scripted distance after the steps so far."""
        return self.dist(self.steps)


class Planner:
    """A planner stand-in recording its calls and optionally reporting a phase change."""

    def __init__(self):
        """No calls yet."""
        self.calls = []

    def plan(self, coords, replan):
        """Record the replan and return fixed points and a plan record."""
        self.calls.append(("plan", replan))
        return np.zeros((4, 3), dtype=np.float32), type("Plan", (), dict(executed="best_sample", cost=1.0, final_hand_error_mm=2.0))()

    def update(self, coords, replan):
        """Record the update."""
        self.calls.append(("update", replan))
        return replan == 1


class Kinematics:
    """A fit stand-in: identity wrist, zero fingers."""

    def fit(self, predicted, q_init):
        """The fixed hand state."""
        return np.eye(3), np.zeros(3), np.zeros(16), {}


def run(env, planner=None, **loop):
    """Run an episode with 5 control steps per replan."""
    return L.run_episode(env, planner or Planner(), Kinematics(), cfg.exec_settings(raw_steps=5), cfg.loop(**loop))


def test_success_after_ten_steps_in_a_row_across_replans():
    """With the check true from the first step, the episode stops at the 10th step, which is the end of replan 2, with success."""
    env = Env(flag=lambda i: True)
    result = run(env)
    assert (result.success, result.stop_reason, result.n_replans, env.steps) == (True, "success", 2, 10)


def test_success_in_the_middle_of_a_burst_stops_the_burst():
    """If the 10th step falls inside a burst of 5, the remaining steps of the burst are not taken."""
    env = Env(flag=lambda i: i >= 3)
    result = run(env)
    assert (result.success, env.steps, result.n_replans) == (True, 12, 3)


def test_a_failed_step_resets_the_count_and_it_counts_across_replans():
    """9 good steps, one bad, then 10 good ones: success only at step 20."""
    env = Env(flag=lambda i: i != 10)
    result = run(env)
    assert result.success and env.steps == 20


def test_no_success_and_check_at_the_end():
    """Short runs of good steps never reach 10; the cap ends the episode and the last step's check is reported."""
    result = run(Env(flag=lambda i: i % 10 != 5), max_replans=4)
    assert (result.success, result.stop_reason, result.n_replans) == (False, "max_replans", 4)
    assert result.holds_at_end is True
    assert run(Env(flag=lambda i: False), max_replans=2).holds_at_end is False


def test_converged_stop_is_not_a_success():
    """Reaching the convergence distance ends the episode without success."""
    result = run(Env(distance=lambda i: 0.005))
    assert (result.success, result.stop_reason, result.n_replans) == (False, "converged", 1)


def test_stall_stop_after_the_patience():
    """A distance that never improves ends the episode at the replan where the patience is used up (the first value always counts as an improvement)."""
    result = run(Env(distance=lambda i: 0.5), stall_patience=3)
    assert (result.stop_reason, result.n_replans) == ("stalled", 4)


def test_success_comes_before_converged_and_stalled():
    """When the 10th good step is also the first one within the convergence distance, the reason is success; a distance that converged earlier ends the episode first, without success."""
    result = run(Env(flag=lambda i: True, distance=lambda i: 0.5 if i < 10 else 0.001))
    assert result.stop_reason == "success" and result.success
    early = run(Env(flag=lambda i: True, distance=lambda i: 0.001))
    assert (early.stop_reason, early.success, early.n_replans) == ("converged", False, 1)


def test_records_and_phase_changes():
    """One record per replan with the error in mm; the phase change reported by the planner is stored; no update after the last replan."""
    planner = Planner()
    result = run(Env(distance=lambda i: 1.0 - 0.01 * i), planner, max_replans=3)
    assert [r["replan"] for r in result.records] == [0, 1, 2]
    assert result.records[0]["error_mm"] == pytest.approx(950.0) and result.records[0]["commanded_mm"] == pytest.approx(0.0)
    assert [r.get("phase_changed") for r in result.records] == [False, True, False]
    assert planner.calls == [("plan", 0), ("update", 0), ("plan", 1), ("update", 1), ("plan", 2), ("update", 2)]


def test_one_observation_per_replan_and_none_after_a_stop():
    """The cloud is observed once at the start and once after each executed replan that does not end the episode."""
    env = Env(distance=lambda i: 0.5)
    run(env, max_replans=3)
    assert env.observations == 4
    env = Env(flag=lambda i: True)
    run(env)
    assert env.observations == 2


def test_no_stall_stop_when_the_patience_is_none():
    """With stall_patience = None a distance that never improves runs to the replan cap."""
    result = run(Env(distance=lambda i: 0.5), stall_patience=None, max_replans=30)
    assert (result.stop_reason, result.n_replans) == ("max_replans", 30)
