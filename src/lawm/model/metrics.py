"""Mean endpoint error in mm on sets of points, kept as sums and counts so batching and sharding do not change it; segment labels only name the sets."""
from __future__ import annotations

import numpy as np
import torch

SEG_STATIC = (2, 3)  # table, distractors (segment ids of the compact point-cloud files)
NAMES = ("mover_5mm", "mover_2mm", "dynamic", "static", "overall")


def step_sums(delta_pred, delta_gt, segment):
    """Per set in NAMES, (summed error in mm, point count) as (len(NAMES), 2) float64, from delta_pred and delta_gt (N, 1, 3) and segment (N,)."""
    err = ((delta_pred - delta_gt).norm(dim=-1).reshape(-1) * 1000).double()
    moved = delta_gt.norm(dim=-1).reshape(-1)
    static = (segment == SEG_STATIC[0]) | (segment == SEG_STATIC[1])
    masks = (moved >= 0.005, moved >= 0.002, ~static, static, torch.ones_like(static))
    return torch.stack([torch.stack([err[m].sum(), m.sum().double()]) for m in masks])


def summarise(totals):
    """From summed totals (n_steps, len(NAMES), 2) return per_step mean mm errors, count and mean over steps; empty sets give NaN and are skipped in the mean."""
    totals = np.asarray(totals, dtype=np.float64)
    count = totals[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_step = np.where(count > 0, totals[..., 0] / count, np.nan)
    mean = np.array([np.nan if np.isnan(per_step[:, i]).all() else np.nanmean(per_step[:, i]) for i in range(len(NAMES))])
    return dict(per_step=per_step, count=count, mean=mean)
