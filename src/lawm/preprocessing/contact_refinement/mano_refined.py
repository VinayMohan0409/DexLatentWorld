"""Atomic writers and strict readers for per-frame refined MANO files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from lawm.preprocessing.h2o_readers import ManoParams

SIDES = ("right", "left")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    """Write JSON via a temp file and rename, so readers never see a partial file."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **payload) -> None:
    """Write a compressed npz via a temp file and rename."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def write_refined_frame(path: Path, frame: int, params: Dict[str, Optional[ManoParams]]) -> None:
    """Write one frame's camera-frame MANO parameters; untracked sides are stored as zeros with valid=False."""
    payload = {"frame_id": np.int64(frame)}
    for side in SIDES:
        side_params = params[side]
        payload[f"{side}_valid"] = np.bool_(side_params is not None)
        trans, pose48, beta = side_params if side_params is not None else (np.zeros(3), np.zeros(48), np.zeros(10))
        payload[f"{side}_trans_cam"] = np.asarray(trans, np.float32)
        payload[f"{side}_pose48"] = np.asarray(pose48, np.float32)
        payload[f"{side}_beta"] = np.asarray(beta, np.float32)
    atomic_npz(path, **payload)


def read_refined_frame(path: Path) -> Dict[str, Optional[ManoParams]]:
    """Read one refined frame as per-side (trans, pose48, beta) in float64, with strict shape and finiteness checks."""
    if not path.exists():
        raise FileNotFoundError(f"Required refined MANO frame is absent: {path}")
    out: Dict[str, Optional[ManoParams]] = {"left": None, "right": None}
    with np.load(path, allow_pickle=False) as z:
        frame_id = int(z["frame_id"])
        if path.stem.isdigit() and frame_id != int(path.stem):
            raise ValueError(f"{path}: frame_id={frame_id} disagrees with filename")
        for side in out:
            if not bool(z[f"{side}_valid"]):
                continue
            trans = np.asarray(z[f"{side}_trans_cam"], np.float64)
            pose = np.asarray(z[f"{side}_pose48"], np.float64)
            beta = np.asarray(z[f"{side}_beta"], np.float64)
            if trans.shape != (3,) or pose.shape != (48,) or beta.shape != (10,):
                raise ValueError(f"{path}: invalid {side} shapes {trans.shape}/{pose.shape}/{beta.shape}")
            if not all(np.isfinite(x).all() for x in (trans, pose, beta)):
                raise ValueError(f"{path}: {side} contains NaN/Inf")
            out[side] = trans, pose, beta
    return out


def validate_complete(seq_path: Path, dir_name: str = "mano_refined") -> dict:
    """Check a refined directory is complete and self-consistent; return its metadata."""
    root = seq_path / dir_name
    complete = json.loads((root / "complete.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    if complete.get("config_hash") != metadata.get("config_hash"):
        raise ValueError(f"{root}: completion/config hash mismatch")
    for stem in metadata["frame_stems"]:
        read_refined_frame(root / f"{stem}.npz")
    return metadata
