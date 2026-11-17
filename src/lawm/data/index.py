"""Build and cache the table of usable training windows: sequence, start frame, valid embodiments, point count."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence as Seq

import numpy as np

from lawm.data.sequences import Sequence


def sequence_windows(valid: np.ndarray, gap: int, n_steps: int):
    """Start frames and per-embodiment validity of the windows whose frames f..f+gap and f+k*gap (k <= n_steps) are all valid."""
    n = len(valid)
    if n <= gap:
        return np.zeros(0, np.int32), np.zeros((0, valid.shape[1]), bool)
    starts = np.arange(n - gap)
    ok = np.lib.stride_tricks.sliding_window_view(valid, gap + 1, axis=0).all(-1)
    for k in range(2, n_steps + 1):
        later = starts + k * gap
        ok = ok & (later < n)[:, None] & valid[np.minimum(later, n - 1)]
    keep = ok.any(axis=1)
    return starts[keep].astype(np.int32), ok[keep]


def fingerprint(ds_dir, sequences: Seq[str]) -> str:
    """Hash of the dataset files an index depends on: each sequence's shared scene, metadata and frame-file count."""
    h = hashlib.sha256()
    for s in sorted(sequences):
        d = Path(ds_dir) / s
        h.update(s.encode())
        for name in ("sequence.npz", "export_metadata.json"):
            h.update(hashlib.sha256((d / name).read_bytes()).digest())
        h.update(str(len(list(d.glob("[0-9]" * 6 + ".npz")))).encode())
    return h.hexdigest()


def build_index(ds_dir, sequences: Seq[str], gap: int, n_steps: int, points_per_entity: int = 0) -> Dict[str, np.ndarray]:
    """Windows of the given sequences (sorted by name then start frame); length is the human cloud's point count."""
    seq, start, valid_window, length = [], [], [], []
    for s in sorted(sequences):
        q = Sequence(Path(ds_dir) / s, points_per_entity)
        f, vw = sequence_windows(q.frame_valid(), gap, n_steps)
        seq += [s] * len(f)
        start.append(f)
        valid_window.append(vw)
        length += [q.n_points(0)] * len(f)
    n_emb = valid_window[0].shape[1] if valid_window else 0
    return dict(seq=np.array(seq), f=np.concatenate(start) if start else np.zeros(0, np.int32),
                valid_window=np.concatenate(valid_window) if valid_window else np.zeros((0, n_emb), bool),
                length=np.array(length, np.int32))


def load_or_build_index(cache_dir, name: str, ds_dir, sequences: Seq[str], gap: int, n_steps: int,
                        points_per_entity: int = 0) -> Dict[str, np.ndarray]:
    """Load the cached index for this exact dataset, gap, n_steps and points per entity, or build and cache it; a mismatch raises."""
    fp = fingerprint(ds_dir, sequences)
    points = f"_p{points_per_entity}" if points_per_entity else ""  # the cloud length column depends on it
    path = Path(cache_dir) / f"index_{name}_g{gap}_n{n_steps}{points}_{fp[:12]}.npz"
    if path.exists():
        with np.load(path) as z:
            got = {k: z[k] for k in z.files}
        stored_points = int(got.pop("points_per_entity")) if "points_per_entity" in got else 0
        if str(got.pop("fingerprint")) != fp or (int(got.pop("gap")), int(got.pop("n_steps")), stored_points) != (gap, n_steps, points_per_entity):
            raise ValueError(f"{path} was built for a different dataset, gap, n_steps or points per entity; delete it to rebuild")
        return got
    index = build_index(ds_dir, sequences, gap, n_steps, points_per_entity)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(path, fingerprint=fp, gap=gap, n_steps=n_steps, points_per_entity=points_per_entity, **index)
    return index
