"""Readers for raw H2O cam4 files: MANO hand parameters, camera poses, object poses and meshes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MANO_BLOCK = 62  # per hand: [valid flag, trans(3), pose(48), beta(10)]
MANO_SIDES = (("left", 0), ("right", MANO_BLOCK))  # the file stores the left block first
MANO_TRANS_SLICE = slice(1, 4)
MANO_POSE_SLICE = slice(4, 52)
MANO_BETA_SLICE = slice(52, 62)

ManoParams = Tuple[np.ndarray, np.ndarray, np.ndarray]  # trans (3,), pose48 (48,), beta (10,)

# H2O object id -> folder name. The id ordering is assumed, not taken from dataset documentation.
H2O_OBJECT_NAMES = {
    1: "book", 2: "espresso", 3: "lotion", 4: "spray",
    5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
}


def parse_mano_file(path: Path) -> Dict[str, Optional[ManoParams]]:
    """Read an H2O hand_pose_mano file into per-side (trans, pose48, beta); None if that hand is untracked."""
    vals = np.fromfile(path, sep=" ").astype(np.float64)
    if vals.size != 2 * MANO_BLOCK:
        raise ValueError(f"Expected 124 MANO values in {path}, got {vals.size}.")
    out: Dict[str, Optional[ManoParams]] = {"left": None, "right": None}
    for side, off in MANO_SIDES:
        block = vals[off:off + MANO_BLOCK]
        if int(round(block[0])) != 1:
            continue
        out[side] = (block[MANO_TRANS_SLICE], block[MANO_POSE_SLICE], block[MANO_BETA_SLICE])
    return out


def load_camera_to_world(cam_dir: Path, frame: int) -> np.ndarray:
    """Read cam_pose/<frame>.txt as a 4x4 camera-to-world matrix (16 floats, row-major)."""
    m = np.fromfile(cam_dir / "cam_pose" / f"{frame:06d}.txt", sep=" ").astype(np.float64)
    return m.reshape(4, 4)


def load_object_pose(cam_dir: Path, frame: int) -> Tuple[np.ndarray, Optional[int]]:
    """Read obj_pose_rt/<frame>.txt as (4x4 model-to-camera matrix, object id or None if the file has 16 values)."""
    vals = np.fromfile(cam_dir / "obj_pose_rt" / f"{frame:06d}.txt", sep=" ").astype(np.float64)
    if vals.size == 17:
        return vals[1:].reshape(4, 4), int(round(vals[0]))
    if vals.size == 16:
        return vals.reshape(4, 4), None
    raise ValueError(f"Unexpected obj_pose_rt size {vals.size} (expected 16 or 17).")


def resolve_object_mesh(object_root: Path, obj_id: Optional[int]) -> Path:
    """Return the .obj for an H2O object id, falling back to the object folder's only .obj."""
    if obj_id is None:
        raise ValueError(f"obj_pose_rt has no object id, so a mesh cannot be resolved under {object_root}.")
    name = H2O_OBJECT_NAMES.get(obj_id)
    if name is None:
        raise ValueError(f"Unknown H2O object id {obj_id}; extend H2O_OBJECT_NAMES.")
    mesh = object_root / name / f"{name}.obj"
    if mesh.exists():
        return mesh
    folder = object_root / name
    objs = sorted(folder.glob("*.obj")) if folder.is_dir() else []
    if len(objs) == 1:  # e.g. spray/lotion_spray.obj
        return objs[0]
    avail = sorted(d.name for d in object_root.iterdir() if d.is_dir())
    raise FileNotFoundError(
        f"id {obj_id}->{name} but {mesh} missing and folder .objs={[o.name for o in objs]}. Available: {avail}"
    )


def load_intrinsics(cam_dir: Path) -> Tuple[float, float, float, float]:
    """Read cam_intrinsics.txt ('fx fy cx cy width height') and return (fx, fy, cx, cy)."""
    v = np.fromfile(cam_dir / "cam_intrinsics.txt", sep=" ").astype(np.float64)
    return v[0], v[1], v[2], v[3]


def load_depth_metres(cam_dir: Path, frame: int) -> np.ndarray:
    """Read depth/<frame>.png (16-bit millimetres) as a float32 HxW depth map in metres."""
    from PIL import Image

    d = np.asarray(Image.open(cam_dir / "depth" / f"{frame:06d}.png")).astype(np.float64)
    return (d / 1000.0).astype(np.float32)


def first_valid_object_id(cam_dir: Path, frames: Sequence[int]) -> int:
    """Return the object id found in the first 20 of `frames`; one object is held per sequence."""
    for f in frames[:20]:
        _, oid = load_object_pose(cam_dir, f)
        if oid:
            return oid
    raise RuntimeError("No valid object id in first frames of obj_pose_rt.")


def list_sequences(h2o_root: Path) -> List[Tuple[int, Path]]:
    """List cam4 sequences that have MANO files as (1-based position in sorted path order, path)."""
    seqs = sorted((p for p in Path(h2o_root).glob("subject*/*/*/cam4") if p.is_dir()),
                  key=lambda p: p.relative_to(h2o_root).as_posix())
    return [(i, p) for i, p in enumerate(seqs, 1) if any((p / "hand_pose_mano").glob("*.txt"))]
