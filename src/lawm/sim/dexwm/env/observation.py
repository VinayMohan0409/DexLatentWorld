"""What the episode loop reads from the simulator after each replan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """The scene cloud (N, 3), the hand root pose (4, 4), the 16 finger angles and the controller's reference position (3,) and rotation matrix (3, 3)."""

    coords: np.ndarray
    root_pose: np.ndarray
    q: np.ndarray
    ref_pos: np.ndarray
    ref_rot: np.ndarray
