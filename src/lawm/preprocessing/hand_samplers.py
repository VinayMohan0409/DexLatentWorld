"""Fixed-surface samplers (point i is the same surface point in every frame) for the human MANO hand and the eight robot hands, plus the per-sequence loop."""
from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

from lawm.preprocessing.contact_refinement.mano_refined import read_refined_frame, validate_complete
from lawm.preprocessing.h2o_readers import load_camera_to_world, parse_mano_file
from lawm.preprocessing.sampling import (
    apply_barycentric, presample_barycentric, split_points, stable_seed, transform_points)

log = logging.getLogger(__name__)

SIDES = ("left", "right")
HAND_TYPES = ("Ability", "Allegro", "Inspire", "Leap", "Oymotion", "Shadow", "Wuji", "Xhand")
X_FLIP_180 = np.diag([1.0, -1.0, -1.0]).astype(np.float64)  # H2O/UniDex camera convention
F_FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
Valid = Dict[str, bool]


def link_name(robot: int, link_idx: int) -> str:
    """URDF name of a PyBullet link; index -1 is the base (root) link."""
    import pybullet as p

    if link_idx < 0:
        return p.getBodyInfo(robot)[0].decode("utf-8")
    return p.getJointInfo(robot, link_idx)[12].decode("utf-8")


def read_mano_params(path: Path):
    """Read per-side (trans, pose48, beta) from a raw H2O MANO text file or a refined .npz file."""
    return read_refined_frame(path) if path.suffix == ".npz" else parse_mano_file(path)


class ManoFastSampler:
    """Fixed-surface sampler for H2O MANO hands."""

    def __init__(self, mano_root: Path, points_total: int, seed_prefix: str = "human") -> None:
        """Loads the MANO layer and fixes the surface sample for each hand."""
        if not hasattr(inspect, "getargspec"):  # chumpy (loads the MANO pickles) still calls it; removed in Python 3.11
            inspect.getargspec = inspect.getfullargspec
        from manopth.manolayer import ManoLayer

        self.layers, self.faces, self.samples = {}, {}, {}
        self.points_per_side = dict(zip(SIDES, split_points(points_total, len(SIDES))))
        for side in SIDES:
            layer = ManoLayer(mano_root=str(mano_root), side=side, use_pca=False, ncomps=45, flat_hand_mean=True).double()
            layer.eval()
            faces = np.asarray(layer.th_faces.detach().cpu().numpy(), dtype=np.int64)
            # The neutral template only defines topology and area for the deterministic sampling.
            with torch.no_grad():
                v0, _ = layer(torch.zeros((1, 48), dtype=torch.float64), torch.zeros((1, 10), dtype=torch.float64))
            template_vertices = v0[0].detach().cpu().numpy().astype(np.float64) / 1000.0
            n_side = self.points_per_side[side]
            self.layers[side], self.faces[side] = layer, faces
            self.samples[side] = presample_barycentric(template_vertices, faces, n_side, stable_seed(seed_prefix, side, n_side))

    def forward_mesh(self, side: str, trans: np.ndarray, pose: np.ndarray, betas: np.ndarray):
        """H2OPlayer convention: MANO output is millimetres, the translation is metres; returns (vertices, joints)."""
        with torch.no_grad():
            verts_t, joints_t = self.layers[side](torch.as_tensor(pose[None], dtype=torch.float64),
                                                  torch.as_tensor(betas[None], dtype=torch.float64))
        vertices = verts_t[0].detach().cpu().numpy().astype(np.float64) / 1000.0 + trans[None, :]
        joints = joints_t[0].detach().cpu().numpy().astype(np.float64) / 1000.0 + trans[None, :]
        return (X_FLIP_180 @ vertices.T).T, (X_FLIP_180 @ joints.T).T  # UniDex H2O convention

    def sample(self, mano_path: Path, frame_data: dict) -> Tuple[np.ndarray, Valid]:
        """Sample both hands (those tracked in this frame) and return (points, which sides were valid)."""
        params = read_mano_params(mano_path)
        all_points: List[np.ndarray] = []
        valid = {"left": False, "right": False}
        for side in SIDES:
            if params[side] is None:
                continue
            trans, pose, betas = params[side]
            vertices, joints = self.forward_mesh(side, trans, pose, betas)
            reference = frame_data.get(f"{side}_pose")  # diagnostic only: compare MANO wrist with the UniDex wrist
            if reference is not None:
                wrist_err = float(np.linalg.norm(joints[0] - reference[0]))
                if wrist_err > 0.05:
                    log.warning("%s MANO wrist error %.4f m", side, wrist_err)
            face_idx, bary = self.samples[side]
            all_points.append(apply_barycentric(vertices, self.faces[side], face_idx, bary))
            valid[side] = True
        if not all_points:
            return np.zeros((0, 3), dtype=np.float32), valid
        return np.concatenate(all_points, axis=0).astype(np.float32), valid


class RobotFastSampler:
    """Fixed local surface points per robot link; per frame uses IK + FK transforms."""

    def __init__(self, uv, dataset_name: str, hand_type: str, points_total: int,
                 exclude_links: Sequence[str] = (), seed_name: Optional[str] = None) -> None:
        """Per-side samplers; exclude_links get no points, seed_name (default hand_type, as stored) prefixes the sampling seeds."""
        self.uv, self.hand_type = uv, hand_type
        self.exclude_links = frozenset(exclude_links)
        self.seed_name = seed_name or hand_type
        self.vis, self.link_samples = {}, {}
        self.points_per_side = dict(zip(SIDES, split_points(points_total, len(SIDES))))
        for side in SIDES:
            urdf = uv.get_urdf_path(dataset_name, hand_type, side)
            vis = uv.WebHandVisualizer(urdf, is_left=(side == "left"))
            self.vis[side] = vis
            self.link_samples[side] = self._presample_links(vis, side, self.points_per_side[side])

    def _mesh_records(self, vis) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, float]]:
        """Visual mesh records of the robot, one (link index, local transform, vertices, faces, surface area) per link mesh that loads."""
        import pybullet as p

        records, found = [], set()
        for vs in p.getVisualShapeData(vis.robot):
            _, link_idx, geom_type, scale, filename, *_ = vs
            if geom_type != p.GEOM_MESH or filename == b"":
                continue
            if self.exclude_links:
                name = link_name(vis.robot, link_idx)
                if name in self.exclude_links:
                    found.add(name)
                    continue
            mesh_path = filename.decode("utf-8")
            if not os.path.isabs(mesh_path):
                mesh_path = str(Path(vis.urdf_path).parent / mesh_path)
            tm = trimesh.load(mesh_path, force="mesh", process=False)
            if tm is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
                continue
            vertices = np.asarray(tm.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
            faces = np.asarray(tm.faces, dtype=np.int64)
            tris = vertices[faces]
            area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum()
            T_local = vis.mesh_bank.get(link_idx, (None, np.eye(4)))[1]
            records.append((link_idx, T_local, vertices, faces, float(area)))
        missing = self.exclude_links - found
        if missing:  # a misspelt name would silently keep the link
            raise ValueError(f"{self.hand_type}: exclude_links {sorted(missing)} match no visual mesh of {vis.urdf_path}")
        return records

    def _presample_links(self, vis, side: str, n_points: int):
        """Draw n_points surface points over the link meshes in proportion to area; returns (link index, points in the visual mesh frame) per link."""
        records = self._mesh_records(vis)
        if not records:
            return []
        areas = np.array([r[4] for r in records], dtype=np.float64)
        quotas = n_points * areas / areas.sum()
        counts = np.floor(quotas).astype(int)
        for idx in np.argsort(-(quotas - counts))[: n_points - int(counts.sum())]:
            counts[idx] += 1

        samples = []
        for rec_idx, ((link_idx, T_local, vertices, faces, _), count) in enumerate(zip(records, counts)):
            if count <= 0:
                continue
            face_idx, bary = presample_barycentric(vertices, faces, int(count),
                                                   stable_seed(self.seed_name, side, link_idx, rec_idx, count))
            points = apply_barycentric(vertices, faces, face_idx, bary)
            samples.append((link_idx, transform_points(points, T_local).astype(np.float64)))  # in the visual mesh frame
        return samples

    def _pose_side(self, side: str, frame_data: dict) -> bool:
        """Pose one side's robot hand by IK from the fingertip goals in frame_data; returns False if that side's pose is missing."""
        import pybullet as p

        pose, R_hand = frame_data.get(f"{side}_pose"), frame_data.get(f"{side}_R_hand_world")
        if pose is None or R_hand is None:
            return False
        vis = self.vis[side]
        vis.load_hand_pose_and_rot(pose, R_hand, is_left=(side == "left"))
        joints = vis.pose.copy()
        if vis.mano_scale != 1.0:
            joints = (joints - joints[0]) * vis.mano_scale + joints[0]

        R_o2b = self.uv.B2O[:3, :3].T
        t_o2b = -R_o2b @ self.uv.B2O[:3, 3]
        p.resetBasePositionAndOrientation(
            vis.robot, (R_o2b @ joints[0] + t_o2b).tolist(), R.from_matrix(R_o2b @ vis.R_hand_world).as_quat().tolist())
        goals = [(R_o2b @ joints[i] + t_o2b).tolist() for i in self.uv.MANO_TIP_INDEX]
        q = vis.solve_ik(goals)
        for joint_idx, angle in zip(vis.actuated, q):
            p.resetJointState(vis.robot, joint_idx, float(angle))
        return True

    def sample(self, frame_data: dict) -> Tuple[np.ndarray, Valid]:
        """Pose each hand by IK from the H2O fingertips and return (points, which sides were valid)."""
        import pybullet as p

        all_points: List[np.ndarray] = []
        valid = {"left": False, "right": False}
        for side in SIDES:
            if not self._pose_side(side, frame_data):
                continue
            vis = self.vis[side]
            side_points = []
            for link_idx, points_local in self.link_samples[side]:
                if link_idx >= 0:
                    pos, orn = p.getLinkState(vis.robot, link_idx, computeForwardKinematics=True)[4:6]
                else:
                    pos, orn = p.getBasePositionAndOrientation(vis.robot)
                T_link = np.eye(4, dtype=np.float64)
                T_link[:3, :3] = R.from_quat(orn).as_matrix()
                T_link[:3, 3] = pos
                side_points.append(transform_points(points_local, self.uv.B2O @ T_link))
            if side_points:
                all_points.append(np.concatenate(side_points, axis=0))
                valid[side] = True
        if not all_points:
            return np.zeros((0, 3), dtype=np.float32), valid
        return np.concatenate(all_points, axis=0).astype(np.float32), valid


def sample_sequence_hands(uv, seq, cam_dir: Path, mano_root: Path, points: int = 2048, dataset_name: str = "H2o",
                          hands: Sequence[str] = HAND_TYPES, max_frames: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame hand clouds of one sequence, (hands (F, E, points, 3) float32, valid (F, E) bool); human from refined MANO, robots by IK in frame order."""
    cam_dir = Path(cam_dir)
    n_frames = min(len(seq), max_frames) if max_frames else len(seq)
    validate_complete(cam_dir)
    T_wc = np.stack([load_camera_to_world(cam_dir, f) for f in range(n_frames)])
    human = ManoFastSampler(mano_root, points)
    robots = {h: RobotFastSampler(uv, dataset_name, h, points) for h in hands}
    E = 1 + len(robots)
    out, valid = np.zeros((n_frames, E, points, 3), np.float32), np.zeros((n_frames, E), bool)
    for f in range(n_frames):
        frame_data = seq[f]
        T = T_wc[f] @ F_FLIP4
        clouds = [human.sample(cam_dir / "mano_refined" / f"{f:06d}.npz", frame_data)[0]]
        clouds += [sampler.sample(frame_data)[0] for sampler in robots.values()]
        for ei, coord in enumerate(clouds):
            if coord is not None and len(coord) == points:  # a missing hand gives fewer points and the embodiment is invalid
                out[f, ei] = (np.asarray(coord, np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
                valid[f, ei] = True
    return out, valid


def sample_sequence_robots(seq, cam_dir: Path, samplers: Dict[str, "RobotFastSampler"], points: int = 2048,
                           max_frames: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame robot-hand clouds (F, K, points, 3) and validity (F, K), as sample_sequence_hands but without MANO."""
    cam_dir = Path(cam_dir)
    n_frames = min(len(seq), max_frames) if max_frames else len(seq)
    out = np.zeros((n_frames, len(samplers), points, 3), np.float32)
    valid = np.zeros((n_frames, len(samplers)), bool)
    for f in range(n_frames):  # in order: the IK warm-starts from the previous frame
        frame_data = seq[f]
        T = load_camera_to_world(cam_dir, f) @ F_FLIP4
        for ei, sampler in enumerate(samplers.values()):
            coord = sampler.sample(frame_data)[0]
            if len(coord) == points:  # a missing hand gives fewer points and the embodiment is invalid
                out[f, ei] = (np.asarray(coord, np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
                valid[f, ei] = True
    return out, valid
