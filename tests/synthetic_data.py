"""Tiny synthetic datasets in the compact point-cloud format, written with the real exporter."""
import numpy as np

from lawm.preprocessing.scene_export import write_sequence

E, PTS = 9, 8
COUNTS = [1, 2, 3, 1, 2, 3, 1, 2, 3]


def make_sequence(out_dir, n_frames, seed, invalid=None, counts=COUNTS):
    """Write one sequence and return the raw arrays; invalid maps embodiment -> frames where it is not valid."""
    rng = np.random.default_rng(seed)
    hands = rng.random((n_frames, E, PTS, 3), dtype=np.float32)
    valid = np.ones((n_frames, E), bool)
    for e, frames in (invalid or {}).items():
        valid[list(frames), e] = False
    poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    poses[:, :3, 3] = rng.random((n_frames, 3), dtype=np.float32)
    scene = {
        "table": rng.random((PTS, 3), dtype=np.float32), "object_canonical": rng.random((PTS, 3), dtype=np.float32),
        "object_id": 1, "plane_n": np.array([0, 0, 1], np.float32), "plane_d": np.float32(0.0),
        "table_uvc": np.eye(3, dtype=np.float32), "cam_poses": np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1)),
        "distractor_points": [rng.random((k, PTS, 3), dtype=np.float32) for k in counts],
        "distractor_counts": np.array(counts, np.int32),
        "embodiment_names": np.array(["human"] + [f"Hand{i}" for i in range(E - 1)]), "n_frames": np.int32(n_frames),
        "object_world_pose": poses,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    write_sequence(out_dir, scene, hands, valid, {"note": "synthetic"})
    return dict(hands=hands, valid=valid, poses=poses, scene=scene)


def make_dataset(root, spec):
    """spec: {"subject1/h1/0": (n_frames, seed, invalid)}; returns {name: raw arrays}."""
    return {name: make_sequence(root / name, n, seed, inv) for name, (n, seed, inv) in spec.items()}
