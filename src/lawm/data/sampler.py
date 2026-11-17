"""Batch sampler that groups similar cloud sizes, keeps batches diverse, and balances ranks."""
from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import Sampler

log = logging.getLogger(__name__)


class LengthGroupedBatchSampler(Sampler):
    """Yields batches of similar-size clouds as epoch * len(lengths) + row, equal in number on every rank; set epoch before iterating, skip drops leading batches."""

    def __init__(self, lengths, batch_size, chunk_mult=50, seed=0, drop_last=True, rank=0, world_size=1,
                 group_keys=None, min_start_frame_gap=0):
        """Stores the sizes, seeds and grouping options; epoch is set by the caller before each pass."""
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.chunk_size = batch_size * chunk_mult
        self.drop_last = drop_last
        self.epoch = 0
        self.skip = 0
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.group_keys = group_keys
        self.min_start_frame_gap = min_start_frame_gap

    def _grouped_chunk_batches(self, chunk, seqs, start_frames):
        """Fill batches left to right from a length-sorted chunk, skipping rows that conflict; returns (batches, fallbacks)."""
        remaining = [int(x) for x in chunk]
        batches, n_fallback = [], 0
        while remaining:
            batch, used_seqs, used_frames, leftover = [], set(), [], []
            for idx in remaining:
                if len(batch) == self.batch_size:
                    leftover.append(idx)
                    continue
                s, sf = seqs[idx], start_frames[idx]
                if s in used_seqs or any(abs(sf - uf) < self.min_start_frame_gap for uf in used_frames):
                    leftover.append(idx)
                    continue
                batch.append(idx)
                used_seqs.add(s)
                used_frames.append(sf)
            if len(batch) < self.batch_size and leftover:
                n_needed = self.batch_size - len(batch)
                batch.extend(leftover[:n_needed])
                leftover = leftover[n_needed:]
                n_fallback += 1
            remaining = leftover
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches, n_fallback

    def __iter__(self):
        """Batches of epoch * len(lengths) + row for this rank, minus the first skip batches."""
        this_epoch = self.epoch
        g = np.random.default_rng(self.seed + this_epoch)
        idx = g.permutation(len(self.lengths))
        batches, n_fallback = [], 0
        for start in range(0, len(idx), self.chunk_size):
            chunk = idx[start:start + self.chunk_size]
            chunk = chunk[np.argsort(self.lengths[chunk])]  # sort within the chunk only
            if self.group_keys is not None:
                chunk_batches, fell_back = self._grouped_chunk_batches(chunk, *self.group_keys)
                batches.extend(chunk_batches)
                n_fallback += fell_back
            else:
                for b in range(0, len(chunk), self.batch_size):
                    batch = chunk[b:b + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch.tolist())
        if n_fallback:
            log.warning("%d batch(es) this epoch could not satisfy the sequence/start-frame constraint", n_fallback)
        if self.world_size > 1:
            batches = batches[:(len(batches) // self.world_size) * self.world_size]  # equal batch counts avoid a DDP hang
            sizes = np.array([self.lengths[b].mean() for b in batches])
            clusters = np.argsort(sizes).reshape(-1, self.world_size)  # each row: batches of similar size, one per rank
            g.shuffle(clusters, axis=0)
            for row in clusters:
                g.shuffle(row)
            batches = [batches[clusters[i, self.rank]] for i in range(len(clusters))]
        else:
            g.shuffle(batches)
        n = len(self.lengths)
        return iter([[this_epoch * n + row for row in b] for b in batches[self.skip:]])

    def __len__(self):
        """Number of batches this rank yields per epoch."""
        n = len(self.lengths)
        total = n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        return total // self.world_size if self.world_size > 1 else total
