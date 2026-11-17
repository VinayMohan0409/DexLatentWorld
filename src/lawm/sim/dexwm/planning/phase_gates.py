"""The two subgoal gates of the DexWM planner: hover above the goal before a grasp, and secure the grip before the place motion; both read the observed clouds, not the model's prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from lawm.sim.costs import SEG_HAND, SEG_OBJECT

@dataclass(frozen=True)
class HoverSettings:
    """The hover before a grasp: how far the hand goal is raised (mm), the observed distance that ends the hover (mm) and the replan that ends it anyway."""

    margin_mm: float
    proximity_mm: float
    replan_cap: int


@dataclass(frozen=True)
class SecureSettings:
    """The grip check before a place: a hand point closer than contact_m to the object touches it, two contacts oppose within half_angle_deg, and min_contact points on sustain_replans replans in a row (or replan_cap) end the stage."""

    contact_m: float
    half_angle_deg: float
    min_contact: int
    sustain_replans: int
    replan_cap: int


@dataclass(frozen=True)
class GateSettings:
    """The settings of both gates."""

    hover: HoverSettings
    secure: SecureSettings


@dataclass
class GraspContact:
    """The observed contact between the hand and the object: how many hand points touch, the best opposition of two contacts (1 = facing each other) and whether it passes the angle."""

    n_contact: int
    best_opposition: float | None
    has_antipodal_pair: bool


def grasp_contact(hand, obj, settings):
    """Contact between observed hand points and object points; the opposition of two contacts is -cos of the angle between their directions towards the object."""
    distance, nearest = cKDTree(obj).query(hand)
    touching = distance < settings.contact_m
    n_contact = int(touching.sum())
    if n_contact < 2:
        return GraspContact(n_contact, None, False)
    to_obj = obj[nearest[touching]] - hand[touching]
    unit = to_obj / (np.linalg.norm(to_obj, axis=1, keepdims=True) + 1e-9)
    opposition = -(unit @ unit.T)
    np.fill_diagonal(opposition, -1.0)
    best = float(opposition.max())
    return GraspContact(n_contact, best, bool(best > np.cos(np.radians(settings.half_angle_deg))))


class HoverGate:
    """The raised hand goal of a grasp: active until the observed hand is close to it, then the real goal."""

    def __init__(self, goal, segment, settings):
        """Keep the real goal and build the goal with the hand points raised."""
        self.real_goal, self.proximity_mm, self.replan_cap = goal, settings.proximity_mm, settings.replan_cap
        self.hover_goal = goal.copy()
        self.hover_goal[segment == SEG_HAND, 2] += settings.margin_mm / 1000.0
        self.hand_mask = segment == SEG_HAND
        self.active = True
        self.distance_mm = float("nan")

    @property
    def goal(self):
        """The goal the planner should use now."""
        return self.hover_goal if self.active else self.real_goal

    def update(self, coords, segment, replan):
        """After the replan with index replan has been executed: measure the observed hand centre against the hover goal; True at the moment the real goal takes over."""
        if not self.active:
            return False
        self.distance_mm = float(np.linalg.norm(coords[segment == SEG_HAND].mean(axis=0) - self.hover_goal[self.hand_mask].mean(axis=0)) * 1000.0)
        if self.distance_mm <= self.proximity_mm or replan >= self.replan_cap:
            self.active = False
            return True
        return False


class SecureGate:
    """The two stages of a place: tighten the grip first, then carry the object; the switch follows the observed contact."""

    def __init__(self, settings, stage_weights):
        """Start in stage 1 with no secure replans counted; stage_weights = (stage 1, stage 2) cost weights."""
        self.settings, self.stage_weights = settings, stage_weights
        self.min_contact, self.sustain_replans, self.replan_cap = settings.min_contact, settings.sustain_replans, settings.replan_cap
        self.stage, self.streak = 1, 0
        self.contact = None

    @property
    def weights(self):
        """The cost weights of the current stage."""
        return self.stage_weights[self.stage - 1]

    def update(self, coords, segment, replan):
        """After the replan with index replan has been executed: count the replans in which the grip looks secure; True at the moment stage 2 starts."""
        if self.stage == 2:
            return False
        self.contact = grasp_contact(coords[segment == SEG_HAND], coords[segment == SEG_OBJECT], self.settings)
        secure = self.contact.has_antipodal_pair and self.contact.n_contact >= self.min_contact
        self.streak = self.streak + 1 if secure else 0
        if self.streak >= self.sustain_replans or replan >= self.replan_cap:
            self.stage = 2
            return True
        return False
