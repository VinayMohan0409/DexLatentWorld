"""Read and write a sequence's point-cloud files: per-frame hand clouds, the shared sequence.npz and export_metadata.json."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

FRAME_FILE = re.compile(r"^\d{6}\.npz$")
SCHEMA_VERSION = 1


def read_hand_clouds(clouds_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read every per-frame file of a sequence: hands (frames, embodiments, points, 3) and hand_valid (frames, embodiments)."""
    files = sorted(p for p in Path(clouds_dir).iterdir() if FRAME_FILE.match(p.name))
    if not files:
        raise FileNotFoundError(f"No per-frame hand files in {clouds_dir}")
    with np.load(files[0]) as z:  # fill preallocated arrays so a long sequence is held in memory only once
        hands = np.empty((len(files),) + z["hands"].shape, z["hands"].dtype)
        valid = np.empty((len(files),) + z["hand_valid"].shape, z["hand_valid"].dtype)
    for i, p in enumerate(files):
        with np.load(p) as z:
            hands[i], valid[i] = z["hands"], z["hand_valid"]
    return hands, valid


def write_sequence(out_dir: Path, scene: Dict[str, object], hands: np.ndarray, hand_valid: np.ndarray,
                   metadata: dict) -> None:
    """Write a sequence's per-frame files and sequence.npz into a temporary folder, then rename it to out_dir so a partial run never looks finished."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_frames = hands.shape[0]
    for f in range(n_frames):
        np.savez_compressed(tmp / f"{f:06d}.npz", hands=hands[f], hand_valid=hand_valid[f],
                            object_pose=scene["object_world_pose"][f].astype(np.float32))

    clouds = scene["distractor_points"]
    distractors = np.empty(len(clouds), dtype=object)  # one (k, points, 3) array per embodiment; k differs between them
    for i, c in enumerate(clouds):
        distractors[i] = c
    np.savez(tmp / "sequence.npz",
             table=scene["table"], object_canonical=scene["object_canonical"], object_id=scene["object_id"],
             plane_n=scene["plane_n"], plane_d=scene["plane_d"], table_uvc=scene["table_uvc"],
             distractor_points=distractors, distractor_counts=scene["distractor_counts"],
             cam_poses=scene["cam_poses"], embodiment_names=scene["embodiment_names"], n_frames=scene["n_frames"])
    meta = {"schema_version": SCHEMA_VERSION, "frames": int(n_frames), "points_per_component": int(hands.shape[2]),
            "embodiments": [str(e) for e in scene["embodiment_names"]], **metadata}
    (tmp / "export_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")  # written last

    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)


def is_complete(out_dir: Path) -> bool:
    """True if a sequence folder was fully written (its metadata file exists)."""
    return (Path(out_dir) / "export_metadata.json").exists()
