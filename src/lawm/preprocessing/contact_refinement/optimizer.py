"""Gated geometric optimiser (no learned network) that cleans tracked MANO hands against the object they manipulate."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh

from lawm.preprocessing.contact_refinement.sequence import SequenceData

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefineConfig:
    """Hyperparameters of the refinement run used for the released data."""

    lam_temporal: float = 10.0
    n_iter: int = 300
    lr: float = 0.01
    lam_anchor: float = 1e-2
    lam_accel: float = 1e-2
    lam_pen: float = 4.0
    soft_tissue_allowance_m: float = 0.002  # overlap tolerated before penetration is penalised
    early_stop_patience: int = 30
    early_stop_tol: float = 1e-4
    dist_near_m: float = 0.008              # gate is 1 when the hand is this close to the object
    dist_far_m: float = 0.02                # ... and 0 when it is this far
    contact_near_m: float = 0.008
    contact_far_m: float = 0.02
    obj_sample_n: int = 512                 # farthest-point object vertices used as contact targets


def make_mano_layer(mano_root: Path, side: str, device: str):
    """Build a manopth MANO layer (full 45-dim axis-angle pose, flat hand mean) on the given device."""
    if not hasattr(inspect, "getargspec"):  # chumpy (loads the MANO pickles) still calls it; removed in Python 3.11
        inspect.getargspec = inspect.getfullargspec
    from manopth.manolayer import ManoLayer

    return ManoLayer(mano_root=str(mano_root), side=side, use_pca=False, ncomps=45, flat_hand_mean=True).to(device)


def smoothstep(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Return 1.0 near `low`, 0.0 near `high`, smooth in between."""
    t = ((high - x) / (high - low)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def second_diff(x: torch.Tensor) -> torch.Tensor:
    """Second finite difference along the first (time) axis."""
    return x[2:] - 2.0 * x[1:-1] + x[:-2]


def build_batch(seq: SequenceData, device: str) -> Dict[str, torch.Tensor]:
    """Move a sequence to tensors and compute per-frame object vertex normals from the posed meshes."""
    T = len(seq.frame_ids)
    obj_normals_np = np.stack([
        trimesh.Trimesh(vertices=seq.obj_verts[t], faces=seq.obj_faces, process=False).vertex_normals
        for t in range(T)
    ])
    return {
        "hand_rot0": torch.tensor(seq.hand_rot, dtype=torch.float32, device=device),
        "hand_pose0": torch.tensor(seq.hand_pose, dtype=torch.float32, device=device),
        "hand_mTc_world": torch.tensor(seq.hand_mTc_world, dtype=torch.float32, device=device),
        "hand_beta": torch.tensor(np.tile(seq.hand_beta, (T, 1)), dtype=torch.float32, device=device),
        "obj_verts": torch.tensor(seq.obj_verts, dtype=torch.float32, device=device),
        "obj_normals": torch.tensor(np.ascontiguousarray(obj_normals_np), dtype=torch.float32, device=device),
    }


def mano_forward(mano_model, hand_rot, hand_pose, hand_beta, mTc_world) -> torch.Tensor:
    """Return world-frame MANO vertices (metres); mTc_world places camera-frame MANO output into the world."""
    pose_full = torch.cat([hand_rot, hand_pose], dim=1)
    verts_cam, _ = mano_model(pose_full, hand_beta)
    verts_cam = verts_cam / 1000.0  # manopth outputs millimetres
    R = mTc_world[:, :3, :3]
    t = mTc_world[:, :3, 3:4]
    verts_world = torch.bmm(R, verts_cam.transpose(1, 2)) + t
    return verts_world.transpose(1, 2)


def subsample_object(obj_verts: torch.Tensor, n: int, seed: int = 0):
    """Deterministic farthest-point sampling of n object vertices (topology is fixed across frames)."""
    N = obj_verts.shape[1]
    if N <= n:
        return obj_verts, torch.arange(N, device=obj_verts.device)

    pts = obj_verts[0]
    idx = torch.empty(n, dtype=torch.long, device=pts.device)
    min_d2 = torch.full((N,), float("inf"), dtype=pts.dtype, device=pts.device)
    farthest = int(seed % N)
    for i in range(n):
        idx[i] = farthest
        d2 = ((pts - pts[farthest]) ** 2).sum(dim=1)
        min_d2 = torch.minimum(min_d2, d2)
        farthest = int(torch.argmax(min_d2).item())
    return obj_verts[:, idx, :], idx


def precompute_contacts(data: Dict[str, torch.Tensor], mano_model, cfg: RefineConfig, side: str) -> dict:
    """Compute, from the ORIGINAL geometry, the per-frame contact gate and the fixed hand-to-object correspondences."""
    with torch.no_grad():
        hand_verts0 = mano_forward(mano_model, data["hand_rot0"], data["hand_pose0"],
                                   data["hand_beta"], data["hand_mTc_world"])
        obj_verts_sub, obj_sample_idx = subsample_object(data["obj_verts"], cfg.obj_sample_n)
        obj_normals_sub = data["obj_normals"][:, obj_sample_idx, :]

        dists0 = torch.cdist(hand_verts0, obj_verts_sub)
        min_dist = dists0.amin(dim=(1, 2))
        gate = smoothstep(min_dist, cfg.dist_near_m, cfg.dist_far_m)
        hand_nearest_dist, hand_nearest_idx = dists0.min(dim=2)
        hand_contact_target = smoothstep(hand_nearest_dist, cfg.contact_near_m, cfg.contact_far_m)

    log.info("[%s] gate: mean=%.3f frac_off=%.3f", side, gate.mean().item(), (gate < 0.05).float().mean().item())
    return {
        "obj_verts_sub": obj_verts_sub, "obj_normals_sub": obj_normals_sub,
        "hand_nearest_idx": hand_nearest_idx, "hand_contact_target": hand_contact_target,
        "gate": gate, "min_dist": min_dist,
    }


def temporal_refine(seq: SequenceData, data: Dict[str, torch.Tensor], precomp: dict, mano_model,
                    cfg: RefineConfig, side: str, device: str) -> dict:
    """Optimise one gated correction per frame and return the best iterate's rot, pose and world translation."""
    T = data["hand_rot0"].shape[0]
    obj_verts_sub = precomp["obj_verts_sub"]
    obj_normals_sub = precomp["obj_normals_sub"]
    nearest_idx = precomp["hand_nearest_idx"].unsqueeze(-1).expand(-1, -1, 3)
    hand_contact_target = precomp["hand_contact_target"]
    gate = precomp["gate"]
    gate_bcast = gate.unsqueeze(1)  # forces exactly zero correction on frames the gate judged "not near the object"

    delta = torch.zeros((T, 3 + 45 + 3), device=device)  # [rot(3), pose(45), trans(3)]
    delta.requires_grad = True
    optimizer = torch.optim.Adam([delta], lr=cfg.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(cfg.n_iter * 0.4), int(cfg.n_iter * 0.72)], gamma=0.2)

    loss_history = []
    best_loss = float("inf")
    best_delta = delta.detach().clone()
    best_iter = 0

    for it in range(cfg.n_iter):
        optimizer.zero_grad()

        rot = data["hand_rot0"] + gate_bcast * delta[:, 0:3]
        pose = data["hand_pose0"] + gate_bcast * delta[:, 3:48]
        mTc = data["hand_mTc_world"].clone()
        mTc[:, :3, 3] = mTc[:, :3, 3] + gate_bcast * delta[:, 48:51]

        hand_verts = mano_forward(mano_model, rot, pose, data["hand_beta"], mTc)

        obj_at_target = torch.gather(obj_verts_sub, 1, nearest_idx)
        attract_per_vert = ((hand_verts - obj_at_target) ** 2).sum(dim=-1)
        loss_attract = (gate * (hand_contact_target * attract_per_vert).sum(dim=1)).sum()

        obj_normals_at_target = torch.gather(obj_normals_sub, 1, nearest_idx)
        signed_dist = ((hand_verts - obj_at_target) * obj_normals_at_target).sum(dim=-1)
        excess_penetration = torch.clamp(-signed_dist - cfg.soft_tissue_allowance_m, min=0.0)
        loss_pen = (hand_contact_target * excess_penetration ** 2).sum()

        loss_anchor = (delta ** 2).sum()

        # Second-order smoothing keeps constant-velocity motion; wrist translation is in the world frame.
        trans_world = mTc[:, :3, 3]
        loss_temporal = (second_diff(trans_world) ** 2).sum() + (second_diff(pose) ** 2).sum()

        delta_effective = gate_bcast * delta
        loss_accel = ((second_diff(delta_effective[:, 48:51]) ** 2).sum()
                      + (second_diff(delta_effective[:, 3:48]) ** 2).sum())

        loss = (loss_attract + cfg.lam_pen * loss_pen + cfg.lam_anchor * loss_anchor
                + cfg.lam_temporal * loss_temporal + cfg.lam_accel * loss_accel)

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_delta = delta.detach().clone()
            best_iter = it

        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss_value)
        if len(loss_history) >= cfg.early_stop_patience:
            recent = loss_history[-cfg.early_stop_patience:]
            if (max(recent) - min(recent)) / (abs(max(recent)) + 1e-8) < cfg.early_stop_tol:
                log.info("[%s] early stop at iter %d", side, it)
                break

    with torch.no_grad():
        delta.copy_(best_delta)  # return the best iterate, not merely the last one
        rot = data["hand_rot0"] + gate_bcast * delta[:, 0:3]
        pose = data["hand_pose0"] + gate_bcast * delta[:, 3:48]
        mTc = data["hand_mTc_world"].clone()
        mTc[:, :3, 3] = mTc[:, :3, 3] + gate_bcast * delta[:, 48:51]
        hand_verts_final = mano_forward(mano_model, rot, pose, data["hand_beta"], mTc)

    return {
        "hand_verts": hand_verts_final.detach().cpu().numpy(),
        "rot": rot.detach().cpu().numpy(),
        "pose": pose.detach().cpu().numpy(),
        "trans_world_final": mTc[:, :3, 3].detach().cpu().numpy(),
        "gate": gate.detach().cpu().numpy(),
        "best_loss": best_loss, "best_iter": best_iter, "n_iter_actual": len(loss_history),
        "frame_ids": seq.frame_ids, "side": side,
    }
