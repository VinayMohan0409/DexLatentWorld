"""Executor for the DexWM hand: the fitted wrist pose and finger angles become the controller's absolute action, with the rotation smoothed across replans and the translation limited."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lawm.sim.dexwm.control.hand_fit import hand_state_to_action

ACTION_DIM = 47  # [0:3] wrist position, [3:6] wrist rotation vector, [15:31] finger angles, the rest unused


@dataclass(frozen=True)
class ExecSettings:
    """How a replan is executed: the longest wrist translation per replan (mm), the share of the way the rotation moves from the last target to the new one, and the identical control steps per replan."""

    max_step_mm: float
    ori_alpha: float
    raw_steps: int


class Executor:
    """Turns one fitted hand state per replan into the controller's action, remembering the last target rotation."""

    def __init__(self, settings, start_rotvec):
        """Start the rotation smoothing from the controller's rotation at the start of the episode."""
        self.settings = settings
        self.last_rotvec = np.array(start_rotvec, dtype=float)

    def command(self, hand_state, root_now, ref_pos, ref_rot):
        """The (47,) float32 action and the unclipped wrist translation (mm) for hand_state = (root rotation, root position, 16 angles), the observed root pose (4, 4) and the controller's reference pose."""
        root_rot, root_pos, q = hand_state
        pos, rotvec, fingers = hand_state_to_action(root_rot, root_pos, q, root_now[:3, :3], root_now[:3, 3], ref_pos, ref_rot)
        keys = Rotation.concatenate([Rotation.from_rotvec(self.last_rotvec), Rotation.from_rotvec(rotvec)])
        rotvec = Slerp([0, 1], keys)([self.settings.ori_alpha])[0].as_rotvec()
        self.last_rotvec = rotvec.copy()
        move = pos - ref_pos
        move_mm = float(np.linalg.norm(move) * 1000.0)
        if move_mm > self.settings.max_step_mm:
            pos = ref_pos + move * (self.settings.max_step_mm / move_mm)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0:3], action[3:6], action[15:31] = pos, rotvec, fingers
        return action, move_mm
