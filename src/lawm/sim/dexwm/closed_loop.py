"""The closed loop of one DexWM episode: plan, fit the hand, execute, observe, count the success steps, and stop on success, convergence, stall or the replan cap."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lawm.sim.control import StallDetector
from lawm.sim.dexwm.control.executor import Executor


@dataclass(frozen=True)
class LoopSettings:
    """When an episode ends: the replan cap, the task distance (mm) that counts as converged, the stall patience (None = no stall stop) and tolerance (mm), and the consecutive successful steps that count as success."""

    max_replans: int
    converge_mm: float
    stall_patience: int | None
    stall_tol_mm: float
    sustain_steps: int


@dataclass
class EpisodeResult:
    """How an episode ended: success (the check held for sustain_steps steps in a row at some point), whether it holds at the last step, the reason it stopped, the replans used and one record per replan."""

    success: bool
    holds_at_end: bool
    stop_reason: str
    n_replans: int
    records: list = field(default_factory=list)


def run_episode(env, planner, kinematics, exec_settings, settings):
    """Run one episode on env (observe() -> Observation, step(action) -> success flag, distance() -> task distance in m); the stops, in order: sustained success, converged, stalled, replan cap."""
    obs = env.observe()
    executor = Executor(exec_settings, Rotation.from_matrix(obs.ref_rot).as_rotvec())
    stall = None if settings.stall_patience is None else StallDetector(settings.stall_patience, settings.stall_tol_mm)
    streak, ok, records = 0, False, []
    for replan in range(settings.max_replans):
        hand, plan = planner.plan(obs.coords, replan)
        root_rot, root_pos, q, _ = kinematics.fit(hand.astype(np.float64), obs.q)
        action, commanded_mm = executor.command((root_rot, root_pos, q), obs.root_pose, obs.ref_pos, obs.ref_rot)
        for _ in range(exec_settings.raw_steps):
            ok = bool(env.step(action))
            streak = streak + 1 if ok else 0
            if streak >= settings.sustain_steps:
                break
        error_mm = float(env.distance() * 1000.0)
        records.append(dict(replan=replan, error_mm=error_mm, commanded_mm=commanded_mm, executed=plan.executed, plan_cost=plan.cost, predicted_error_mm=plan.final_hand_error_mm, streak=streak))
        reason = "success" if streak >= settings.sustain_steps else "converged" if error_mm < settings.converge_mm else "stalled" if stall and stall.update(error_mm) else None
        if reason:
            return EpisodeResult(streak >= settings.sustain_steps, ok, reason, replan + 1, records)
        obs = env.observe()
        records[-1]["phase_changed"] = planner.update(obs.coords, replan)
    return EpisodeResult(False, ok, "max_replans", settings.max_replans, records)
