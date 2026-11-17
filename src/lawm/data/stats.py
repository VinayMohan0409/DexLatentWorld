"""Normalisation statistics: mean and variance of every point's displacement from frame 0 at each step."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np

from lawm.data.dataset import WindowDataset, load_window
from lawm.data.sequences import Sequence

VAR_FLOOR = 0.001 ** 2  # a 1 mm standard deviation, so no step or axis is divided by ~0
_SEQUENCES: Dict[tuple, Sequence] = {}  # (dataset folder, sequence name, points per entity) -> sequence, per worker process


def window_moments(job):
    """One window's mean and mean of squares of the displacement from frame 0, (n_steps, 3), over all points; uses no labels."""
    ds_dir, name, f, e, gap, n_steps, points, voxel = job
    if (ds_dir, name, points) not in _SEQUENCES:
        _SEQUENCES[ds_dir, name, points] = Sequence(Path(ds_dir) / name, points)
    (sample,) = load_window(_SEQUENCES[ds_dir, name, points], f, [e], gap, n_steps, voxel)
    coords = sample["coords"].numpy()
    delta = (coords[1:] - coords[0:1]).astype(np.float64)
    return delta.mean(axis=1), (delta ** 2).mean(axis=1), delta.shape[1]


def compute_norm_stats(dataset: WindowDataset, windows: int = 4000, seed: int = 0, workers: int = 1) -> Dict[str, np.ndarray]:
    """Equal-weight mean and variance over a seeded sample of windows, one embodiment each; any failure raises."""
    rows = np.arange(len(dataset))
    if len(rows) > windows:
        rows = np.random.default_rng(seed).choice(len(dataset), size=windows, replace=False)
    jobs = []
    for j, row in enumerate(rows):
        e = int(np.random.default_rng(seed * 1_000_003 + j).choice(np.flatnonzero(dataset.valid[row])))
        jobs.append((str(dataset.ds_dir), str(dataset.seq[row]), int(dataset.f[row]), e, dataset.gap, dataset.n_steps, dataset.points_per_entity, dataset.voxel_size))
    if workers == 1:
        results = map(window_moments, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(window_moments, jobs, chunksize=16))  # map keeps the order, so the sums are reproducible
    mean_sum = sq_sum = 0.0
    points = 0
    for mean_t, sq_t, n in results:  # summed in window order
        mean_sum, sq_sum, points = mean_sum + mean_t, sq_sum + sq_t, points + n
    mean = mean_sum / len(jobs)
    var = np.maximum(sq_sum / len(jobs) - mean ** 2, 0.0)
    n_floored = int((var < VAR_FLOOR).sum())
    var = np.maximum(var, VAR_FLOOR)
    if not (np.isfinite(mean).all() and np.isfinite(var).all()):
        raise ValueError("normalisation statistics contain NaN or Inf")
    return dict(per_step_mean=mean.astype(np.float32), per_step_var=var.astype(np.float32), windows=len(jobs), windows_requested=windows,
                seed=seed, n_steps=dataset.n_steps, gap=dataset.gap, points_per_entity=dataset.points_per_entity, voxel_size=dataset.voxel_size, total_points=points,
                n_floored=n_floored)


def save_norm_stats(path, stats: Dict, fingerprint: str, split: str) -> None:
    """Write the statistics with what they were computed from: the dataset fingerprint (see lawm.data.index) and the split name."""
    np.savez(path, fingerprint=fingerprint, split=split, **stats)


def load_norm_stats(path, fingerprint: str, split: str, n_steps: int, gap: int, windows: int, seed: int,
                    points_per_entity: int = 0, voxel_size: float = 0.0) -> Dict[str, np.ndarray]:
    """Read statistics and refuse them unless they were computed from this dataset and split with these settings."""
    with np.load(path) as z:
        stats = {k: z[k] for k in z.files}
    stats.setdefault("points_per_entity", np.array(0))  # files written before the options kept every stored point and every point of a cloud
    stats.setdefault("voxel_size", np.array(0.0))
    expected = dict(fingerprint=fingerprint, split=split, n_steps=n_steps, gap=gap, windows_requested=windows, seed=seed,
                    points_per_entity=points_per_entity, voxel_size=voxel_size)
    wrong = {k: (stats[k].item(), v) for k, v in expected.items() if stats[k].item() != v}
    if wrong:
        raise ValueError(f"{path}: statistics do not match this run (stored, expected): {wrong}; delete the file to recompute")
    return stats
