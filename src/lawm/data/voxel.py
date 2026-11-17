"""Reduce a cloud to one point per voxel, deciding on frame 0 so that every kept row can be followed through all frames."""
from __future__ import annotations

import numpy as np


def voxel_select(coord0: np.ndarray, size: float, seed=None) -> np.ndarray:
    """Ascending rows of frame 0 kept at one point per voxel of size metres: the first of each voxel, or a random one (a function of seed only) if seed is given."""
    grid = np.floor((coord0 - coord0.min(axis=0)) / size).astype(np.int64)
    _, first, inverse, counts = np.unique(grid, axis=0, return_index=True, return_inverse=True, return_counts=True)
    if seed is None:
        return np.sort(first)
    by_voxel = np.argsort(inverse.reshape(-1), kind="stable")  # rows grouped by voxel, in row order within a voxel
    starts = np.cumsum(counts) - counts
    return np.sort(by_voxel[starts + np.random.default_rng(seed).integers(0, counts)])
