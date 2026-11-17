"""Read one sequence from the compact point-cloud files and assemble the clouds of chosen embodiments."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence as Seq, Tuple

import numpy as np

SEG_HAND, SEG_OBJECT, SEG_TABLE, SEG_DISTRACTOR = 0, 1, 2, 3
SEG_STATIC = (SEG_TABLE, SEG_DISTRACTOR)  # never move, so their displacement target is zero


class Sequence:
    """One sequence folder: the shared scene in sequence.npz and, per frame, hand clouds, validity and the object pose."""

    def __init__(self, seq_dir, points_per_entity: int = 0) -> None:
        """points_per_entity > 0 keeps every k-th stored point of the object, table and each distractor (k = stored / points); hands keep all, 0 keeps everything."""
        self.dir = Path(seq_dir)
        with np.load(self.dir / "sequence.npz", allow_pickle=True) as s:
            self.n_frames = int(s["n_frames"])
            self.names = [str(n).lower() for n in s["embodiment_names"]]
            object_canonical = s["object_canonical"].astype(np.float32)
            table = s["table"].astype(np.float32)
            distractors = [s["distractor_points"][e].astype(np.float32).reshape(-1, 3) if s["distractor_counts"][e] > 0
                           else np.zeros((0, 3), np.float32) for e in range(len(self.names))]
        stored = len(object_canonical)
        step = 1 if points_per_entity == 0 else stored // points_per_entity
        if points_per_entity and (stored % points_per_entity or len(table) != stored or any(len(d) % stored for d in distractors)):
            raise ValueError(f"{self.dir}: {points_per_entity} points per entity does not divide the {stored} stored points of each entity")
        self.object_canonical = object_canonical[::step]  # every entity block holds `stored` points, so a stride keeps whole blocks
        table, distractors = table[::step], [d[::step] for d in distractors]
        self.n_table = len(table)
        self.n_distractor = [len(d) for d in distractors]
        self.static = [np.concatenate([table, d]) for d in distractors]  # table then the distractors of each embodiment
        with np.load(self._frame_path(0)) as z:
            self.n_hand = z["hands"].shape[1]
        self._frames = {}  # optional in-memory frame cache (preload); empty = read every frame from disk, as before

    def _frame(self, f: int):
        """(object_pose, hands of every embodiment) of frame f, from the cache when preloaded."""
        if f in self._frames:
            return self._frames[f]
        with np.load(self._frame_path(f)) as z:
            return z["object_pose"], z["hands"]

    def preload(self, frames, workers: int = 16) -> None:
        """Read the given frames once into memory with parallel reads (file decompression releases the GIL); later clouds() calls never touch the disk for them."""
        from concurrent.futures import ThreadPoolExecutor

        def read(f):
            """(f, (object_pose, hands)) of frame f read from disk."""
            with np.load(self._frame_path(f)) as z:
                return f, (z["object_pose"], z["hands"])
        with ThreadPoolExecutor(workers) as pool:
            self._frames.update(pool.map(read, sorted(set(frames) - set(self._frames))))

    def _frame_path(self, f: int) -> Path:
        """Path of the .npz file of frame f."""
        return self.dir / f"{f:06d}.npz"

    def n_points(self, e: int) -> int:
        """Points in embodiment e's full cloud: hand, object, table and its distractors."""
        return self.n_hand + len(self.object_canonical) + len(self.static[e])

    def frame_valid(self) -> np.ndarray:
        """hand_valid of every frame, (n_frames, embodiments); fails if the frame files do not match n_frames."""
        found = len(list(self.dir.glob("[0-9]" * 6 + ".npz")))
        if found != self.n_frames:
            raise ValueError(f"{self.dir}: {found} frame files but sequence.npz says {self.n_frames}")
        valid = []
        for f in range(self.n_frames):
            with np.load(self._frame_path(f)) as z:
                valid.append(z["hand_valid"])
        return np.stack(valid)

    def clouds(self, frames: Seq[int], embodiments: Seq[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """For each embodiment: its cloud at each frame (len(frames), N, 3) float32 and the point segments (N,) int64."""
        hands, objects = [], []
        for f in frames:
            pose, all_hands = self._frame(f)
            pose = pose.astype(np.float32)
            hands.append(all_hands[list(embodiments)].astype(np.float32))
            objects.append(self.object_canonical @ pose[:3, :3].T + pose[:3, 3])
        out = []
        for j, e in enumerate(embodiments):
            coords = np.stack([np.concatenate([h[j], o, self.static[e]]) for h, o in zip(hands, objects)])
            segment = np.concatenate([np.full(self.n_hand, SEG_HAND), np.full(len(self.object_canonical), SEG_OBJECT),
                                      np.full(self.n_table, SEG_TABLE), np.full(self.n_distractor[e], SEG_DISTRACTOR)])
            out.append((coords, segment.astype(np.int64)))
        return out
