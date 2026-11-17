"""Unit tests for lawm.data.sampler.LengthGroupedBatchSampler."""
import logging

import numpy as np
import pytest

from lawm.data.sampler import LengthGroupedBatchSampler

N_SEQ, N_FRAMES, BATCH, GAP = 30, 40, 4, 5
SEQS = [f"seq{s}" for s in range(N_SEQ) for _ in range(N_FRAMES)]
FRAMES = [f for _ in range(N_SEQ) for f in range(N_FRAMES)]
LENGTHS = np.random.default_rng(0).integers(1000, 5000, len(SEQS))
N = len(SEQS)


def make(rank=0, world=1, seed=0, grouped=True):
    """Build a LengthGroupedBatchSampler over the shared lengths for the given rank, world size and seed, with or without sequence grouping"""
    return LengthGroupedBatchSampler(LENGTHS, BATCH, seed=seed, rank=rank, world_size=world,
                                     group_keys=(SEQS, FRAMES) if grouped else None, min_start_frame_gap=GAP)


def test_batches_have_the_right_size_and_encode_the_epoch():
    """Batches have the configured size, repeat identically on re-iteration, use each row at most once and encode the epoch in the row ids"""
    s = make()
    for epoch in range(3):
        s.epoch = epoch
        batches = list(s)
        assert list(s) == batches  # iterating does not advance the epoch
        assert len(batches) == len(s) and all(len(b) == BATCH for b in batches)
        flat = [i for b in batches for i in b]
        assert {i // N for i in flat} == {epoch} and len(set(flat)) == len(flat)  # each row at most once per epoch


def test_batches_repeat_no_sequence_or_close_start_frames_except_for_the_counted_fallbacks(caplog):
    """Batches avoid repeated sequences and close start frames except for a few conflicts, all of which are reported as a warning"""
    with caplog.at_level(logging.WARNING):
        batches = list(make())
    bad = 0
    for b in batches:
        rows = [i % N for i in b]
        fr = sorted(FRAMES[r] for r in rows)
        bad += len({SEQS[r] for r in rows}) < BATCH or any(y - x < GAP for x, y in zip(fr, fr[1:]))
    reported = caplog.records[0].args[0] if caplog.records else 0
    assert bad <= reported < len(batches) / 10  # a conflict is only allowed at the end of a chunk, and is always reported


def test_ranks_get_equal_disjoint_shares_and_the_same_seed_repeats():
    """Ranks get equally sized, disjoint batches, and the same seed repeats while a different seed changes them"""
    parts = [list(make(rank=r, world=3)) for r in range(3)]
    assert len({len(p) for p in parts}) == 1 and len(parts[0]) == len(make(world=3))
    flat = [i for p in parts for b in p for i in b]
    assert len(set(flat)) == len(flat)
    assert list(make(rank=1, world=3)) == parts[1] and list(make(rank=1, world=3, seed=1)) != parts[1]


def test_an_epoch_can_be_reproduced_after_a_resume():
    """A fresh sampler set to a given epoch yields the same batches as one that iterated up to it"""
    s = make()
    for epoch in range(3):
        s.epoch = epoch
        last = list(s)
    resumed = make()
    resumed.epoch = 2  # the third epoch
    assert list(resumed) == last
    assert list(make()) != last


def test_without_group_keys_batches_are_size_grouped():
    """Without group keys, batches hold similar lengths, with a much smaller spread than the whole set"""
    batches = list(make(grouped=False))
    spread = [np.ptp(LENGTHS[[i % N for i in b]]) for b in batches]
    assert np.mean(spread) < np.ptp(LENGTHS) / 4


def test_a_chunk_that_cannot_satisfy_the_constraint_warns_instead_of_dropping(caplog):
    """A single sequence cannot satisfy the spacing constraint, so the sampler warns but still emits every batch"""
    one_seq = LengthGroupedBatchSampler(np.arange(40), BATCH, group_keys=(["a"] * 40, list(range(40))), min_start_frame_gap=GAP)
    with caplog.at_level(logging.WARNING):
        batches = list(one_seq)
    assert len(batches) == 10 and caplog.records


def test_skip_drops_only_the_first_batches_of_the_epoch():
    """Setting skip removes only that many leading batches and leaves the rest of the epoch unchanged"""
    s = make(rank=1, world=2)
    s.epoch = 3
    full = list(s)
    s.skip = 5
    assert list(s) == full[5:]
