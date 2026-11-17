"""Meshes for predicted clouds: robot link meshes with their point rows, a chumpy-free MANO layer, and rigid fits."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SIDES = ("left", "right")


def export_robot_links(unidex_root: Path, hand: str, points_total: int, out: Path) -> None:
    """(UniDex env) Per side and link record, in the stored cloud's row order: link-frame sample points and visual mesh."""
    from lawm.preprocessing.hand_samplers import RobotFastSampler
    from lawm.preprocessing.sampling import transform_points
    from lawm.preprocessing.unidex_adapter import open_unidex

    arrays, row = {}, 0
    with open_unidex(unidex_root) as (uv, _):
        sampler = RobotFastSampler(uv, "H2o", hand, points_total)
        for side in SIDES:
            meshes = {rec[0]: rec for rec in sampler._mesh_records(sampler.vis[side])}
            for r, (link_idx, pts) in enumerate(sampler.link_samples[side]):
                _, T_local, verts, faces, _ = meshes[link_idx]
                arrays[f"{side}_{r}_rows"] = np.array([row, row + len(pts)])
                arrays[f"{side}_{r}_points"] = pts
                arrays[f"{side}_{r}_verts"] = transform_points(verts, T_local)
                arrays[f"{side}_{r}_faces"] = faces
                row += len(pts)
    np.savez_compressed(out, hand=hand, n_rows=row, **arrays)


def load_robot_links(path: Path) -> List[dict]:
    """Records of export_robot_links: rows (start, end), points, verts, faces."""
    z = np.load(path)
    keys = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_rows") and k != "n_rows"}, key=lambda k: (k.split("_")[0], int(k.split("_")[1])))
    return [dict(rows=z[f"{k}_rows"], points=z[f"{k}_points"], verts=z[f"{k}_verts"], faces=z[f"{k}_faces"]) for k in keys]


def kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R src + t - dst| (rows are corresponding points)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def robot_meshes(links: List[dict], hand_points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each link mesh placed by the rigid fit of its sample points to their (predicted) positions in hand_points."""
    out = []
    for rec in links:
        a, b = rec["rows"]
        R, t = kabsch(rec["points"], hand_points[a:b])
        out.append((rec["verts"] @ R.T + t, rec["faces"]))
    return out


class _Ch(np.ndarray):
    """Stand-in for chumpy arrays inside the MANO pickles: keeps the pickled state."""

    def __new__(cls, *a, **k):
        """An empty array of this class."""
        return np.zeros(0).view(cls)

    def __setstate__(self, state):
        """Keep the pickled state (a dict, or the last item of a tuple)."""
        self.state = state if isinstance(state, dict) else state[-1]


class _Unpickler(pickle.Unpickler):
    """Unpickler that loads chumpy objects as _Ch."""
    def find_class(self, module, name):
        """Map chumpy classes to the stand-in."""
        if module.startswith("chumpy"):
            return _Ch
        return super().find_class(module, name)


def load_mano(path: Path) -> Dict[str, np.ndarray]:
    """MANO model arrays from the official pickle without chumpy."""
    d = _Unpickler(open(path, "rb"), encoding="latin1").load()

    def arr(v):
        """Plain array of a pickled value: chumpy Ch keeps its data in 'x'; a chumpy selection is a[idxs] reshaped."""
        if isinstance(v, _Ch):
            st = v.state
            if "x" in st:
                return arr(st["x"])
            base = arr(st["a"]).ravel()
            return base[np.asarray(st["idxs"])].reshape(st["preferred_shape"])
        return np.asarray(v.toarray() if hasattr(v, "toarray") else v, np.float64)
    return dict(v_template=arr(d["v_template"]), shapedirs=arr(d["shapedirs"]), posedirs=arr(d["posedirs"]),
                J_regressor=arr(d["J_regressor"]), weights=arr(d["weights"]), parents=np.asarray(d["kintree_table"][0], int),
                faces=np.asarray(d["f"], np.int64))


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Axis-angle (n, 3) -> rotation matrices (n, 3, 3)."""
    th = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
    k = r / th
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(0, 2, 1)
    th = th[:, :, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mano_vertices(m: Dict[str, np.ndarray], pose48: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """MANO vertices in metres for full axis-angle pose (flat hand mean, global rotation first), as manopth computes them."""
    v = m["v_template"] + m["shapedirs"] @ betas
    J = m["J_regressor"] @ v
    R = _rodrigues(pose48.reshape(16, 3))
    v = v + m["posedirs"] @ (R[1:] - np.eye(3)).reshape(-1)
    G = np.zeros((16, 4, 4))
    for i in range(16):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[i], J[i] - (J[m["parents"][i]] if i else 0)
        G[i] = T if i == 0 else G[m["parents"][i]] @ T
    G[:, :3, 3] -= np.einsum("nij,nj->ni", G[:, :3, :3], J)
    Tv = np.einsum("vk,kij->vij", m["weights"], G)
    return np.einsum("vij,vj->vi", Tv[:, :3, :3], v) + Tv[:, :3, 3]
