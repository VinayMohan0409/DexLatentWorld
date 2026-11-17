"""Hand-object fit metrics: penetration depth and contact-region distance of a hand mesh to the object it holds."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import trimesh

from lawm.preprocessing.contact_refinement.optimizer import make_mano_layer

ACTIVE_FRAME_THRESHOLD_M = 0.03  # a frame is "active" when the raw hand's contact region is within 3 cm of the object
CHUNK_FRAMES = 16                # frames per distance-matrix chunk, to bound memory
PENETRATION_ALLOWANCE_M = 0.002  # overlap the optimiser tolerates on purpose (RefineConfig.soft_tissue_allowance_m)


def contact_region_mask(side: str, mano_root: Path) -> np.ndarray:
    """Bool mask (778,) of the MANO rest-pose vertices on the contact side (palm and finger pads), taken as those not facing the negative mean normal."""
    model = make_mano_layer(mano_root, side, "cpu")
    with torch.no_grad():
        rest_verts, _ = model(torch.zeros(1, 48), torch.zeros(1, 10))
    mesh = trimesh.Trimesh(vertices=rest_verts[0].numpy() / 1000.0, faces=model.th_faces.numpy(), process=False)
    normals = mesh.vertex_normals
    palmar_dir = -normals.mean(axis=0)
    palmar_dir /= np.linalg.norm(palmar_dir)
    return ~((normals @ palmar_dir) > 0.2)


def object_vertex_normals(obj_verts: np.ndarray, obj_faces: np.ndarray) -> np.ndarray:
    """Vertex normals for every frame of a rigidly moving object, from frame 0's normals and a batched Kabsch fit."""
    mesh0 = trimesh.Trimesh(vertices=obj_verts[0], faces=obj_faces, process=False)
    normals0 = np.asarray(mesh0.vertex_normals, dtype=np.float64)

    V0c = obj_verts[0].astype(np.float64) - obj_verts[0].astype(np.float64).mean(axis=0)
    Vt = obj_verts.astype(np.float64)
    Vtc = Vt - Vt.mean(axis=1, keepdims=True)

    H = np.einsum('ni,tnj->tij', V0c, Vtc)  # per-frame cross-covariance
    U, _, Vh = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('tij,tjk->tik', Vh.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.zeros((len(d), 3, 3))
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum('tij,tjk,tkl->til', Vh.transpose(0, 2, 1), D, U.transpose(0, 2, 1))

    return np.einsum('tij,nj->tni', R, normals0).astype(np.float32)


def penetration_stats(hand_verts: np.ndarray, obj_verts: np.ndarray, obj_normals: np.ndarray,
                      frame_ids: np.ndarray, device: str) -> dict:
    """Penetration depth of each hand vertex: how far it lies behind the tangent plane of its nearest object vertex."""
    hand = torch.as_tensor(hand_verts, dtype=torch.float32, device=device)
    obj = torch.as_tensor(obj_verts, dtype=torch.float32, device=device)
    normals = torch.as_tensor(obj_normals, dtype=torch.float32, device=device)

    per_frame_max, per_frame_mean, per_frame_pct, all_pen = [], [], [], []
    for start in range(0, hand.shape[0], CHUNK_FRAMES):
        stop = min(start + CHUNK_FRAMES, hand.shape[0])
        hand_chunk, obj_chunk, normal_chunk = hand[start:stop], obj[start:stop], normals[start:stop]
        dists = torch.cdist(hand_chunk, obj_chunk)
        nearest_idx = dists.argmin(dim=2)
        nearest_obj = torch.gather(obj_chunk, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, 3))
        nearest_normal = torch.gather(normal_chunk, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, 3))
        signed_dist = ((hand_chunk - nearest_obj) * nearest_normal).sum(dim=-1)
        pen = torch.clamp(-signed_dist, min=0.0)
        per_frame_max.append(pen.max(dim=1).values)
        per_frame_mean.append(pen.mean(dim=1))
        per_frame_pct.append((pen > 0).float().mean(dim=1))
        all_pen.append(pen.reshape(-1))

    frame_max = torch.cat(per_frame_max).cpu().numpy()
    frame_mean = torch.cat(per_frame_mean).cpu().numpy()
    frame_pct = torch.cat(per_frame_pct).cpu().numpy()
    all_pen_np = torch.cat(all_pen).cpu().numpy()
    worst_idx = int(np.argmax(frame_max))
    return {
        "mean_vertex_penetration_m": float(all_pen_np.mean()),
        "p95_vertex_penetration_m": float(np.quantile(all_pen_np, 0.95)),
        "mean_frame_max_penetration_m": float(frame_max.mean()),
        "p95_frame_max_penetration_m": float(np.quantile(frame_max, 0.95)),
        "max_penetration_m": float(frame_max[worst_idx]),
        "pct_penetrating_vertices": float((all_pen_np > 0).mean()),
        "pct_vertices_penetrating_over_allowance": float((all_pen_np > PENETRATION_ALLOWANCE_M).mean()),
        "mean_pct_penetrating_vertices_per_frame": float(frame_pct.mean()),
        "worst_frame_idx": worst_idx,
        "worst_frame_h2o_id": int(frame_ids[worst_idx]),
        "per_frame_mean_m": frame_mean.tolist(),
    }


def contact_chamfer_per_frame(hand_verts: np.ndarray, obj_verts: np.ndarray, contact_mask: np.ndarray,
                              device: str, nearby_radius_m: float = 0.03) -> dict:
    """Per-frame distance from the contact-region hand vertices to the object, plus a nearby-object symmetric term."""
    hand_contact = torch.as_tensor(hand_verts[:, contact_mask, :], dtype=torch.float32, device=device)
    obj = torch.as_tensor(obj_verts, dtype=torch.float32, device=device)

    h2o_means, sym_means, near_counts = [], [], []
    for start in range(0, hand_contact.shape[0], CHUNK_FRAMES):
        stop = min(start + CHUNK_FRAMES, hand_contact.shape[0])
        d = torch.cdist(hand_contact[start:stop], obj[start:stop])
        h2o_min = d.min(dim=2).values  # contact vertex -> nearest object point
        o2h_min = d.min(dim=1).values  # object point -> nearest contact vertex
        near_mask = (o2h_min < nearby_radius_m).float()
        near_count = near_mask.sum(dim=1)
        near_sum = (o2h_min * near_mask).sum(dim=1)
        h2o_mean = h2o_min.mean(dim=1)
        sym_mean = 0.5 * (h2o_mean + near_sum / near_count.clamp(min=1))
        sym_mean = torch.where(near_count > 0, sym_mean, torch.full_like(sym_mean, float("nan")))
        h2o_means.append(h2o_mean)
        sym_means.append(sym_mean)
        near_counts.append(near_count)

    return {
        "h2o_mean": torch.cat(h2o_means).cpu().numpy(),
        "sym_mean": torch.cat(sym_means).cpu().numpy(),
        "near_count": torch.cat(near_counts).cpu().numpy(),
    }


def aggregate_chamfer(per_frame: dict, mask: Optional[np.ndarray] = None) -> dict:
    """Summarise per-frame chamfer arrays over all frames, or over the frames selected by `mask`."""
    h2o_arr, sym_arr, near_counts_arr = per_frame["h2o_mean"], per_frame["sym_mean"], per_frame["near_count"]
    per_frame_mean_m = h2o_arr  # unfiltered, so callers can derive an active-frame mask from it

    if mask is not None:
        h2o_arr, sym_arr, near_counts_arr = h2o_arr[mask], sym_arr[mask], near_counts_arr[mask]
    if len(h2o_arr) == 0:
        return {
            "mean_contact_to_surface_m": None, "p95_contact_to_surface_m": None,
            "max_contact_to_surface_m": None, "mean_symmetric_nearby_chamfer_m": None,
            "n_frames_with_nearby_object_points": 0, "n_frames_total": 0,
            "per_frame_mean_m": per_frame_mean_m.tolist(),
        }
    has_sym = ~np.isnan(sym_arr)
    return {
        "mean_contact_to_surface_m": float(h2o_arr.mean()),
        "p95_contact_to_surface_m": float(np.quantile(h2o_arr, 0.95)),
        "max_contact_to_surface_m": float(h2o_arr.max()),
        "mean_symmetric_nearby_chamfer_m": float(sym_arr[has_sym].mean()) if has_sym.any() else None,
        "n_frames_with_nearby_object_points": int(near_counts_arr.astype(bool).sum()),
        "n_frames_total": int(len(h2o_arr)),
        "per_frame_mean_m": per_frame_mean_m.tolist(),
    }
