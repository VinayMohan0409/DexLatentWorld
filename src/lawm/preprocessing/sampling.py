"""Deterministic surface sampling shared by the hand, object, table and distractor samplers."""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np


def stable_seed(*items: object) -> int:
    """Return a reproducible 32-bit seed derived from the given items."""
    h = hashlib.blake2b("|".join(map(str, items)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0xFFFFFFFF


def split_points(total: int, n_parts: int) -> List[int]:
    """Split total into n_parts near-equal counts, giving the remainder to the first parts."""
    if n_parts <= 0:
        return []
    base, rem = divmod(total, n_parts)
    return [base + int(i < rem) for i in range(n_parts)]


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N,3) points."""
    return points @ T[:3, :3].T + T[:3, 3]


def presample_barycentric(
    vertices: np.ndarray, faces: np.ndarray, n_points: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw area-weighted face ids and barycentric weights for a fixed set of surface samples."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tris = vertices[faces]
    double_area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    valid = double_area > 1e-12
    if not np.any(valid):
        raise ValueError("Cannot sample mesh: all triangle areas are zero.")

    valid_faces = np.nonzero(valid)[0]
    probs = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(seed)
    face_idx = valid_faces[rng.choice(len(valid_faces), size=n_points, replace=True, p=probs)]

    # Uniform triangle sampling.
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2],
        axis=1,
    )
    return face_idx.astype(np.int64), bary.astype(np.float64)


def apply_barycentric(
    vertices: np.ndarray, faces: np.ndarray, face_idx: np.ndarray, bary: np.ndarray
) -> np.ndarray:
    """Evaluate fixed barycentric samples on a (possibly deformed) mesh."""
    tri = vertices[faces[face_idx]]
    return np.einsum("nij,ni->nj", tri, bary)
