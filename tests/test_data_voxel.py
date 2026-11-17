"""Tests for the voxel reduction: one point per voxel, and the same rows (the same physical points) in every frame."""
import numpy as np
import pytest
from synthetic_data import make_dataset

from lawm.data.dataset import WindowDataset, load_window
from lawm.data.index import build_index
from lawm.data.sequences import Sequence
from lawm.data.stats import compute_norm_stats, load_norm_stats, save_norm_stats
from lawm.data.voxel import voxel_select

GAP, STEPS, SIZE = 2, 2, 0.5  # a coarse voxel on unit-cube points, so that voxels hold several points
SPEC = {"subject1/h1/0": (30, 1, None), "subject1/h1/1": (25, 2, None)}


def voxels(c, size=SIZE):
    """Return each point's voxel index tuple on a grid anchored at the cloud's minimum"""
    return [tuple(v) for v in np.floor((c - c.min(0)) / size).astype(int)]


def test_exactly_one_point_per_voxel_and_the_first_one_by_default():
    """voxel_select keeps exactly one point per voxel, in ascending order and the first point of each voxel by default"""
    c = np.random.default_rng(0).random((200, 3))
    kept = voxel_select(c, SIZE)
    v = voxels(c)
    assert len(kept) == len(set(v)) and len({v[i] for i in kept}) == len(kept) and kept.tolist() == sorted(kept.tolist())
    first = {}
    for i, k in enumerate(v):
        first.setdefault(k, i)
    assert kept.tolist() == sorted(first.values())


def test_a_random_choice_stays_inside_the_voxel_and_depends_only_on_the_seed():
    """A seeded random choice keeps one point per voxel, repeats for the same seed and differs for another seed or the default"""
    c = np.random.default_rng(1).random((300, 3))
    v = voxels(c)
    a, b, other = voxel_select(c, SIZE, 5), voxel_select(c, SIZE, 5), voxel_select(c, SIZE, 6)
    assert np.array_equal(a, b) and not np.array_equal(a, other) and not np.array_equal(a, voxel_select(c, SIZE))
    assert len({v[i] for i in a}) == len(a) == len(set(v))


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    """Module-scoped fixture that writes the synthetic dataset and returns its root with the first sequence"""
    root = tmp_path_factory.mktemp("vox")
    make_dataset(root, SPEC)
    return root, Sequence(root / "subject1/h1/0")


def test_the_same_rows_are_kept_in_every_frame_so_displacements_and_segments_are_unchanged(sequence):
    """A voxel-reduced window keeps the same rows in every frame, so displacements and segments match the full window's selected rows and the cloud stays centred"""
    _, q = sequence
    (full,) = load_window(q, 4, [3], GAP, STEPS)
    (red,) = load_window(q, 4, [3], GAP, STEPS, SIZE)
    raw0 = q.clouds([4 + k * GAP for k in range(STEPS + 1)], [3])[0][0][0]
    kept = voxel_select(raw0, SIZE)
    n = len(kept)
    assert 0 < n < full["coords"].shape[1] and red["coords"].shape == (STEPS + 1, n, 3)
    for k in range(1, STEPS + 1):  # the displacement of row i of the reduced window is that of row kept[i] of the full one
        assert np.allclose((red["coords"][k] - red["coords"][0]).numpy(), (full["coords"][k] - full["coords"][0]).numpy()[kept], atol=1e-6)
    assert np.array_equal(red["segment"].numpy(), full["segment"].numpy()[kept])
    assert np.allclose(red["coords"][0].numpy().mean(axis=0), 0, atol=1e-6)  # centred on the cloud the model receives
    full_voxels = voxels(raw0)  # the voxels of the whole cloud (the origin is the minimum of the full cloud)
    assert len({full_voxels[i] for i in kept}) == n and len(set(full_voxels)) == n  # one kept point per voxel, every voxel represented


def test_training_picks_a_random_point_per_voxel_from_the_seed_and_epoch_and_evaluation_the_first(sequence):
    """Training picks voxel points as a function of seed, epoch and row, and evaluation is deterministic"""
    root, _ = sequence
    index = build_index(root, list(SPEC), GAP, STEPS)
    ds = WindowDataset(root, index, GAP, STEPS, m=1, seed=3, train=True, voxel_size=SIZE)
    ev = WindowDataset(root, index, GAP, STEPS, m=1, train=False, voxel_size=SIZE)
    n = len(ds)
    same_a, same_b = ds[5]["samples"][0]["coords"], WindowDataset(root, index, GAP, STEPS, m=1, seed=3, train=True, voxel_size=SIZE)[5]["samples"][0]["coords"]
    assert np.array_equal(same_a.numpy(), same_b.numpy())  # a function of seed, epoch and row only
    assert not np.array_equal(ds[5]["samples"][0]["coords"].numpy(), ds[n + 5]["samples"][0]["coords"].numpy())  # another epoch picks other points
    assert np.array_equal(ev[5]["samples"][0]["coords"].numpy(), ev[5]["samples"][0]["coords"].numpy())


def test_statistics_record_the_voxel_size_and_refuse_another(sequence, tmp_path):
    """Saved normalisation statistics record the voxel size and loading with a different voxel size raises"""
    root, _ = sequence
    ds = WindowDataset(root, build_index(root, list(SPEC), GAP, STEPS), GAP, STEPS, m=1, train=False, voxel_size=SIZE)
    stats = compute_norm_stats(ds, windows=10)
    assert float(stats["voxel_size"]) == SIZE
    path = tmp_path / "s.npz"
    save_norm_stats(path, stats, "fp", "train")
    ok = dict(fingerprint="fp", split="train", n_steps=STEPS, gap=GAP, windows=10, seed=0, voxel_size=SIZE)
    load_norm_stats(path, **ok)
    with pytest.raises(ValueError, match="do not match"):
        load_norm_stats(path, **{**ok, "voxel_size": 0.0})
