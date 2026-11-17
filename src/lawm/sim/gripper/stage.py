"""A stage of a gripper task: where the moved object must go, how the planner scores a candidate, when the stage is over."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lawm.sim.costs import SEG_OBJECT, goal_cost_mm


@dataclass
class Stage:
    """goal(points, segment) -> goal cloud with the same rows; true_error_mm(env) -> mm (used only to stop, never to plan); the gripper command is fixed."""

    name: str
    goal: Callable
    true_error_mm: Callable
    converge_mm: float = 20.0
    max_replans: int = 15
    steps_per_replan: int = 25
    stall_patience: int = 6
    grip_action: float = 1.0


def object_goal_cost(coords0, goal, segment, weights):
    """(cost, goal mask): the horizon-weighted distance of the predicted object points to their goal, in mm."""
    mask = segment == SEG_OBJECT
    return (lambda pred_deltas: goal_cost_mm(pred_deltas, coords0, goal, mask, weights)), mask


def object_translated_to(target_pos_fn, current_pos_fn):
    """Goal cloud: the object rows moved so the object lands on the target, every other row where it is."""
    def build(points, segment):
        """The goal cloud for the points and segment labels of this replan."""
        goal = points.copy()
        obj = segment == SEG_OBJECT
        goal[obj] = points[obj] + (np.asarray(target_pos_fn()) - np.asarray(current_pos_fn()))[None, :]
        return goal
    return build


def body_pos_error(sim_body, target_pos_fn):
    """Probe: distance in mm from a body's position to the target."""
    def probe(env):
        """The distance for the environment's current state."""
        bid = env.sim.model.body_name2id(sim_body)
        return float(np.linalg.norm(env.sim.data.body_xpos[bid] - np.asarray(target_pos_fn())) * 1000)
    return probe
