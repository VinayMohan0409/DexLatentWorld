"""Load one H2O cam4 sequence into world-frame arrays for geometric refinement."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

from lawm.preprocessing.h2o_readers import (
    load_camera_to_world,
    load_object_pose,
    parse_mano_file,
    resolve_object_mesh,
)

SIDES = ("right", "left")


@dataclass
class SequenceData:
    """World-frame hand and posed-object arrays for one hand, over the frames where that hand is tracked."""

    hand_pose: np.ndarray       # [T, 45] finger axis-angle pose, MANO/camera convention
    hand_rot: np.ndarray        # [T, 3] global wrist axis-angle, camera convention
    hand_mTc_world: np.ndarray  # [T, 4, 4] camera-frame MANO output -> world
    hand_beta: np.ndarray       # [10] MANO shape (constant over a sequence in H2O)
    obj_verts: np.ndarray       # [T, N_obj, 3] world frame, per-frame posed
    obj_faces: np.ndarray       # [F_obj, 3] static topology
    frame_ids: np.ndarray       # [T] original H2O frame indices
    side: str


def translation_matrix(t: np.ndarray) -> np.ndarray:
    """Return the 4x4 pure-translation matrix for t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = t
    return T


def apply_homogeneous(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform through homogeneous coordinates (the arithmetic the refinement run used)."""
    points_h = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (T @ points_h.T).T[:, :3]


def load_sequence(seq_path: Path, object_root: Path) -> Dict[str, SequenceData]:
    """Load a cam4 sequence; the result has a key only for hands tracked in at least one frame."""
    seq_path = Path(seq_path)
    mano_files = sorted((seq_path / "hand_pose_mano").glob("*.txt"))

    per_side = {side: defaultdict(list) for side in SIDES}
    obj_verts_model = obj_faces = None

    for mano_file in mano_files:
        frame = int(mano_file.stem)
        params = parse_mano_file(mano_file)
        if params["right"] is None and params["left"] is None:
            continue

        T_world_cam = load_camera_to_world(seq_path, frame)
        T_cam_obj, obj_id = load_object_pose(seq_path, frame)
        if obj_verts_model is None:
            mesh = trimesh.load(resolve_object_mesh(Path(object_root), obj_id), process=False, force="mesh")
            obj_verts_model = np.asarray(mesh.vertices, dtype=np.float64)
            obj_faces = np.asarray(mesh.faces, dtype=np.int64)
        obj_verts_world = apply_homogeneous(obj_verts_model, T_world_cam @ T_cam_obj)

        for side in SIDES:
            if params[side] is None:
                continue
            trans, pose48, beta10 = params[side]
            d = per_side[side]
            d["frame_ids"].append(frame)
            d["hand_rot"].append(pose48[:3])
            d["hand_pose"].append(pose48[3:])
            d["hand_beta"].append(beta10)
            d["hand_mTc_world"].append(T_world_cam @ translation_matrix(trans))
            d["obj_verts"].append(obj_verts_world)

    result: Dict[str, SequenceData] = {}
    for side in SIDES:
        d = per_side[side]
        if not d["frame_ids"]:
            continue
        result[side] = SequenceData(
            hand_pose=np.stack(d["hand_pose"]).astype(np.float32),
            hand_rot=np.stack(d["hand_rot"]).astype(np.float32),
            hand_mTc_world=np.stack(d["hand_mTc_world"]).astype(np.float32),
            hand_beta=np.mean(np.stack(d["hand_beta"]), axis=0).astype(np.float32),
            obj_verts=np.stack(d["obj_verts"]).astype(np.float32),
            obj_faces=obj_faces,
            frame_ids=np.array(d["frame_ids"], dtype=np.int64),
            side=side,
        )

    if not result:
        raise ValueError(f"No valid hand frames (left or right) found in {seq_path}")
    return result
