"""The two training losses: a motion-weighted reconstruction loss and a cross-embodiment contrastive loss on the latents."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HUBER_DELTA = 5.0  # in normalised units, where it behaves like squared error for almost every point (PointWorld's default)
MOVE_TAU = 0.005   # metres: displacement at which a point counts as half moving
MOVE_KAPPA = 1000  # steepness of that gate, per metre


def movement_weighted_huber_loss(delta_norm_pred, delta_gt, offset, out_mean, out_std,
                                 huber_delta=HUBER_DELTA, tau=MOVE_TAU, kappa=MOVE_KAPPA):
    """Huber loss on normalised displacements (N, 1, 3), gated by sigmoid(kappa * (|d| - tau)), weights summing to 1 per cloud, averaged over clouds."""
    delta_norm_gt = ((delta_gt - out_mean) / out_std).float()
    err = nn.HuberLoss(delta=huber_delta, reduction="none")(delta_norm_pred.float(), delta_norm_gt).mean(-1).reshape(-1)
    m = torch.sigmoid(kappa * (delta_gt.norm(dim=-1) - tau)).reshape(-1)
    B = offset.shape[0]
    cid = torch.searchsorted(offset.to(torch.long), torch.arange(delta_gt.shape[0], device=offset.device), right=True)
    denom = torch.zeros(B, device=m.device).scatter_add_(0, cid, m).clamp_min(1e-6)
    per_cloud = torch.zeros(B, device=m.device).scatter_add_(0, cid, m / denom[cid] * err)
    return per_cloud.sum() / B


def embodiment_info_nce_loss(z, group_idx, tau=0.1):
    """Supervised contrastive loss on z (B, n_queries, z_dim): rows sharing group_idx (B,) are positives, all others negatives; scalar, 0 if no row has a positive."""
    B = z.shape[0]
    zf = F.normalize(z.reshape(B, -1).float(), dim=-1)
    sim = zf @ zf.t() / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    positive = (group_idx[:, None] == group_idx[None, :]) & ~eye
    log_denom = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=-1)
    n_pos = positive.sum(dim=-1)
    per_anchor = -((sim * positive).sum(dim=-1) / n_pos.clamp_min(1) - log_denom)[n_pos > 0]
    return per_anchor.mean() if per_anchor.numel() else z.sum() * 0.0
