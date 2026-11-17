"""Static scene pieces of a sequence: canonical object samples, the fitted table, and distractor placement."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh

from lawm.preprocessing.h2o_readers import (
    H2O_OBJECT_NAMES,
    first_valid_object_id,
    load_camera_to_world,
    load_depth_metres,
    load_intrinsics,
    load_object_pose,
    resolve_object_mesh,
)
from lawm.preprocessing.sampling import apply_barycentric, presample_barycentric, stable_seed

log = logging.getLogger(__name__)

# Embodiment order of every stored cloud: the human hand first, then the 8 robot hands.
EMBODIMENT_NAMES = ("human", "Ability", "Allegro", "Inspire", "Leap", "Oymotion", "Shadow", "Wuji", "Xhand")


def sequence_tag(subject: str, session: str, take: str) -> str:
    """The tag seeding all random sampling of a sequence, e.g. 'subject1_ego_h1_0_cam4'; the exact spelling reproduces the stored clouds."""
    return f"{subject}_ego_{session}_{take}_cam4"


def unproject_to_world(depth_m, fx, fy, cx, cy, T_wc, stride=4, zmax=1.15):
    """Back-project a depth map (every `stride`-th pixel, depth below `zmax`) into world-frame points."""
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[vs, us]
    m = (z > 1e-3) & (z < zmax)
    z, us, vs = z[m], us[m].astype(np.float64), vs[m].astype(np.float64)
    pts_c = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)
    return pts_c @ T_wc[:3, :3].T + T_wc[:3, 3]


def ransac_plane(points, thresh=0.01, iters=200, seed=0):
    """Fit a plane by RANSAC, then refit it by SVD to the inliers; return (normal, d, inlier mask, PCA basis, centroid)."""
    rng = np.random.default_rng(seed)
    N = len(points)
    best_n, best_d, best_mask, best_c = None, None, None, -1
    for _ in range(iters):
        p0, p1, p2 = points[rng.choice(N, 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ p0)
        mask = np.abs(points @ n + d) < thresh
        c = int(mask.sum())
        if c > best_c:
            best_c, best_n, best_d, best_mask = c, n, d, mask
    inl = points[best_mask]
    centroid = inl.mean(0)
    _, _, vh = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vh[-1] / np.linalg.norm(vh[-1])
    d = -float(n @ centroid)
    return n, d, best_mask, vh, centroid


def rot_about_axis(axis, theta):
    """Rotation matrix for an angle `theta` about `axis` (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_between(a, b):
    """Rotation matrix taking direction `a` onto direction `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return rot_about_axis(np.cross(a, perp), np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def fit_table(cam_dir, table_frames, fx, fy, cx, cy, anchor_world, n_pts, pad, seed):
    """Fit the table plane to depth from `table_frames` and sample `n_pts` points on a padded rectangle of it."""
    clouds = [unproject_to_world(load_depth_metres(cam_dir, f), fx, fy, cx, cy,
                                 load_camera_to_world(cam_dir, f)) for f in table_frames]
    pooled = np.concatenate(clouds, 0)
    if len(pooled) > 30000:
        pooled = pooled[np.random.default_rng(0).choice(len(pooled), 30000, replace=False)]

    n, d, mask, vh, c0 = ransac_plane(pooled)
    if (float(n @ anchor_world) + d) < 0:  # make +n point toward the scene
        n, d = -n, -d

    inl = pooled[mask]
    u = vh[0]  # max-variance in-plane direction
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    a, b = (inl - c0) @ u, (inl - c0) @ v
    a_lo, a_hi = np.percentile(a, [2, 98])
    b_lo, b_hi = np.percentile(b, [2, 98])
    a_lo -= pad
    a_hi += pad
    b_lo -= pad
    b_hi += pad
    width, height = a_hi - a_lo, b_hi - b_lo
    center = c0 + 0.5 * (a_lo + a_hi) * u + 0.5 * (b_lo + b_hi) * v
    center = center - (float(n @ center) + d) * n
    rng = np.random.default_rng(seed)
    su = rng.uniform(-width / 2, width / 2, n_pts)
    sv = rng.uniform(-height / 2, height / 2, n_pts)
    pts = center[None] + su[:, None] * u[None] + sv[:, None] * v[None]
    return pts.astype(np.float32), n, d, u, v, center, width, height, float(mask.mean())


def uv_bounds(points_w, center, u, v):
    """Min and max of world points projected into the table-plane (u, v) axes; (None, None) if there are none."""
    points_w = np.asarray(points_w, dtype=np.float64)
    if points_w.size == 0:
        return None, None
    P = points_w.reshape(-1, 3) - center
    uv = np.stack([P @ u, P @ v], axis=1)
    return uv.min(axis=0), uv.max(axis=0)


def roll_distractors(object_root, exclude_ids, n_distractors, n, d, center, u, v,
                     width, height, mfp_lo, mfp_hi, margin, spacing, n_pts, seed):
    """Place random objects flat on the table outside the swept hand/object box (19 Aug rule, unlike the earlier 7 Jul stored clouds); returns static world clouds."""
    rng = np.random.default_rng(seed)
    names = [v_ for k, v_ in H2O_OBJECT_NAMES.items()
             if k not in exclude_ids and (object_root / v_ / f"{v_}.obj").exists()]
    prepared = {}  # per object: vertices, faces and its resting-face normal; loading and hulling are deterministic

    def object_data(name):
        """Vertices, faces and resting-face normal of the named object, loaded once and cached."""
        if name not in prepared:
            mesh = trimesh.load(object_root / name / f"{name}.obj", process=False, force="mesh")
            # Rest the object on its largest convex-hull face: all hull vertices lie on one side of a face plane, so
            # aligning that face's normal with -n puts a whole face flush on the table, never on a single vertex.
            hull = mesh.convex_hull
            prepared[name] = (np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64),
                              hull.face_normals[int(np.argmax(hull.area_faces))])
        return prepared[name]

    placed_uv, clouds = [], []
    attempts = 0
    while len(clouds) < n_distractors and attempts < 2000:
        attempts += 1
        name = names[rng.integers(len(names))]
        verts, faces, face_normal = object_data(name)
        V = verts - verts.mean(0)
        V = (rotation_between(face_normal, -n) @ V.T).T
        V = (rot_about_axis(n, rng.uniform(0, 2 * np.pi)) @ V.T).T
        r = float(np.max(np.hypot(V @ u, V @ v)))
        lim_u, lim_v = max(width / 2 - r, 0.0), max(height / 2 - r, 0.0)
        su, sv = rng.uniform(-lim_u, lim_u), rng.uniform(-lim_v, lim_v)
        if (mfp_lo[0] - margin <= su <= mfp_hi[0] + margin and
                mfp_lo[1] - margin <= sv <= mfp_hi[1] + margin):
            continue
        if any(np.hypot(su - pu, sv - pv) < spacing + r for pu, pv in placed_uv):
            continue
        Vw = V + (center + su * u + sv * v)
        Vw = Vw - (Vw @ n + d).min() * n  # drop onto the plane
        fi, ba = presample_barycentric(Vw, faces, n_pts, stable_seed("distractor", name, seed, len(clouds)))
        clouds.append(apply_barycentric(Vw, faces, fi, ba).astype(np.float32))
        placed_uv.append((su, sv))
    return clouds


def pick_frames(n_total, k):
    """Return all frame indices if k >= n_total, else k evenly spaced ones."""
    return list(range(n_total)) if k >= n_total else list(np.linspace(0, n_total - 1, k).round().astype(int))


def build_sequence_scene(cam_dir: Path, object_root: Path, hands: np.ndarray, hand_valid: np.ndarray, tag: str,
                         points: int = 2048, table_k: int = 8, table_pad: float = 0.05, distractor_max: int = 5,
                         distractor_margin: float = 0.12, distractor_spacing: float = 0.15, traj_pad: float = 0.05,
                         seed: int = 0, names: Sequence[str] = EMBODIMENT_NAMES) -> Dict[str, object]:
    """Build a sequence's shared scene from raw data, hands (F, E, points, 3) and hand_valid (F, E); returns the sequence.npz contents plus object_world_pose (F, 4, 4); distractor seeds are keyed by `names`."""
    cam_dir, object_root = Path(cam_dir), Path(object_root)
    n_frames, E = hands.shape[0], hands.shape[1]
    names = tuple(names)[:E] if tuple(names) == EMBODIMENT_NAMES else tuple(names)
    if len(names) != E or len(set(names)) != E:
        raise ValueError(f"need {E} distinct embodiment names, got {names}")
    fx, fy, cx, cy = load_intrinsics(cam_dir)
    all_frames = list(range(n_frames))
    T_wc = np.stack([load_camera_to_world(cam_dir, f) for f in all_frames])
    obj_id = first_valid_object_id(cam_dir, all_frames)
    log.info("object id=%d -> %s", obj_id, H2O_OBJECT_NAMES.get(obj_id))

    obj_world_pose = np.zeros((n_frames, 4, 4), np.float64)
    obj_centers_w = np.zeros((n_frames, 3))
    for f in all_frames:
        T_co, _ = load_object_pose(cam_dir, f)
        Tw = T_wc[f] @ T_co
        obj_world_pose[f] = Tw
        obj_centers_w[f] = Tw[:3, 3]

    omesh = trimesh.load(resolve_object_mesh(object_root, obj_id), process=False, force="mesh")
    ov, of = np.asarray(omesh.vertices, np.float64), np.asarray(omesh.faces, np.int64)
    ofi, oba = presample_barycentric(ov, of, points, stable_seed("object", tag, seed))
    object_canonical = apply_barycentric(ov, of, ofi, oba).astype(np.float32)

    table_pts, n, d, u, v, center, t_w, t_h, ratio = fit_table(
        cam_dir, pick_frames(n_frames, table_k), fx, fy, cx, cy,
        obj_centers_w.mean(0), points, table_pad, stable_seed("table", tag, seed))
    log.info("table inlier_ratio=%.2f extent long=%.2f short=%.2f", ratio, t_w, t_h)

    # Per-embodiment keep-out box on the table: the object's and that embodiment's hand's swept (u, v) extent.
    lo = np.full((E, 2), np.inf, dtype=np.float64)
    hi = np.full((E, 2), -np.inf, dtype=np.float64)
    for f in all_frames:
        obj_pts_f = object_canonical @ obj_world_pose[f][:3, :3].T + obj_world_pose[f][:3, 3]
        obj_lo, obj_hi = uv_bounds(obj_pts_f, center, u, v)
        if obj_lo is not None:
            lo = np.minimum(lo, obj_lo[None, :])
            hi = np.maximum(hi, obj_hi[None, :])
        for ei in range(E):
            if hand_valid[f, ei]:
                hand_lo, hand_hi = uv_bounds(hands[f, ei], center, u, v)
                if hand_lo is not None:
                    lo[ei] = np.minimum(lo[ei], hand_lo)
                    hi[ei] = np.maximum(hi[ei], hand_hi)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise RuntimeError("Non-finite per-embodiment footprint; object/hand export likely failed.")
    lo, hi = lo - traj_pad, hi + traj_pad

    distractor_clouds: List[np.ndarray] = []
    distractor_counts: List[int] = []
    for ei, emb in enumerate(names):
        k = int(np.random.default_rng(stable_seed("ndist", tag, emb, seed)).integers(1, distractor_max + 1))
        cfg_clouds = roll_distractors(
            object_root, {obj_id}, k, n, d, center, u, v, t_w, t_h, lo[ei], hi[ei],
            distractor_margin, distractor_spacing, points, stable_seed("distractors", tag, emb, seed))
        distractor_clouds.append(np.stack(cfg_clouds) if cfg_clouds else np.zeros((0, points, 3), np.float32))
        distractor_counts.append(len(cfg_clouds))
    log.info("per-embodiment distractor counts: %s", distractor_counts)

    return {
        "table": table_pts, "object_canonical": object_canonical, "object_id": obj_id,
        "plane_n": n.astype(np.float32), "plane_d": np.float32(d),
        "table_uvc": np.stack([u, v, center]).astype(np.float32),
        "distractor_points": distractor_clouds, "distractor_counts": np.array(distractor_counts, np.int32),
        "cam_poses": T_wc.astype(np.float32), "embodiment_names": np.array(names),
        "n_frames": np.int32(n_frames), "object_world_pose": obj_world_pose,
    }
