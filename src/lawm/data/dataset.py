"""Training and validation windows: n_steps+1 frames of one or more embodiments, each centred on its own frame-0 mean."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence as Seq

import numpy as np
import torch
from torch.utils.data import Dataset

from lawm.data.sequences import Sequence
from lawm.data.voxel import voxel_select


def load_window(sequence: Sequence, f: int, embodiments: Seq[int], gap: int, n_steps: int, voxel_size: float = 0.0, select_seed=None):
    """Windows at frame f, one dict per embodiment: coords (n_steps+1, N, 3) centred on the frame-0 mean, and segment (N,); voxel_size > 0 keeps one frame-0 point per voxel, the same rows in every frame."""
    frames = [f + k * gap for k in range(n_steps + 1)]
    samples = []
    for e, (coords, segment) in zip(embodiments, sequence.clouds(frames, embodiments)):
        if voxel_size:
            kept = voxel_select(coords[0], voxel_size, None if select_seed is None else [select_seed, int(e)])
            coords, segment = coords[:, kept], segment[kept]
        coords = coords - coords[0].mean(axis=0, keepdims=True)  # the embodiment's own frame-0 mean over the cloud the model receives, no labels
        samples.append(dict(emb=int(e), coords=torch.from_numpy(coords), segment=torch.from_numpy(segment)))
    return samples


class WindowDataset(Dataset):
    """One item per index row: m distinct valid embodiments of one window; in train mode an index i means epoch i // len(self) and row i % len(self)."""

    def __init__(self, ds_dir, index: Dict[str, np.ndarray], gap: int, n_steps: int, m: int = 1,
                 embodiments: Optional[Seq[int]] = None, seed: int = 0, train: bool = True, points_per_entity: int = 0,
                 voxel_size: float = 0.0) -> None:
        """Stores the index rows, drops windows with fewer than m usable embodiments, and records the point-selection settings."""
        allowed = np.ones(index["valid_window"].shape[1], bool)
        if embodiments is not None:
            allowed[:] = False
            allowed[list(embodiments)] = True
        valid = index["valid_window"] & allowed
        keep = valid.sum(axis=1) >= m  # rows with fewer than m usable embodiments are dropped
        self.seq, self.f, self.valid, self.length = (index["seq"][keep], index["f"][keep], valid[keep],
                                                     index["length"][keep])
        self.ds_dir, self.gap, self.n_steps, self.m, self.seed, self.train = Path(ds_dir), gap, n_steps, m, seed, train
        self.points_per_entity = points_per_entity  # 0 = every stored point; see Sequence
        self.voxel_size = voxel_size  # 0 = no voxel reduction; see load_window (random point per voxel in training, the first in evaluation)
        self._sequences: Dict[str, Sequence] = {}

    def __len__(self) -> int:
        """Number of index rows that have at least m usable embodiments."""
        return len(self.f)

    def draw(self, row: int, epoch: int) -> np.ndarray:
        """The m embodiments of a row for an epoch: a pure function of (seed, sequence, frame, row, epoch)."""
        key = f"{self.seed}|{self.seq[row]}|{self.f[row]}|{row}|{epoch}"
        rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "little"))
        return rng.choice(np.flatnonzero(self.valid[row]), size=self.m, replace=False)

    def sequence(self, name: str) -> Sequence:
        """The Sequence called name, opened once and cached."""
        if name not in self._sequences:
            self._sequences[name] = Sequence(self.ds_dir / name, self.points_per_entity)
        return self._sequences[name]

    def __getitem__(self, i: int):
        """Loads the window for index i (epoch i // len(self) in training) and returns its m embodiment samples."""
        epoch, row = divmod(int(i), len(self)) if self.train else (0, int(i))
        name, f = str(self.seq[row]), int(self.f[row])
        select_seed = None
        if self.train and self.voxel_size:
            key = f"{self.seed}|select|{name}|{f}|{row}|{epoch}"
            select_seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "little")
        samples = load_window(self.sequence(name), f, self.draw(row, epoch), self.gap, self.n_steps, self.voxel_size, select_seed)
        for s in samples:
            s.update(seq=name, frame=f)
        return dict(transition_idx=row, samples=samples)


def collate(batch):
    """Flatten groups into one ragged batch: coords (n_steps+1, total points, 3), per-sample offsets and segments."""
    samples = [s for g in batch for s in g["samples"]]
    n = torch.tensor([s["coords"].shape[1] for s in samples])
    return dict(seq=[s["seq"] for s in samples], frame=[s["frame"] for s in samples], emb=[s["emb"] for s in samples],
                coords=torch.cat([s["coords"] for s in samples], dim=1), offset=torch.cumsum(n, 0).int(),
                segment=torch.cat([s["segment"] for s in samples]).long(),
                transition_idx=torch.tensor([g["transition_idx"] for g in batch for _ in g["samples"]]))
