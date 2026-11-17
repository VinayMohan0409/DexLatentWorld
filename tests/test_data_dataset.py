"""Unit tests for lawm.data.dataset: centring, deterministic draws, batching."""
import numpy as np
import pytest
import torch
from synthetic_data import make_dataset
from torch.utils.data import DataLoader

from lawm.data.dataset import WindowDataset, collate
from lawm.data.index import build_index

from test_data_index import expected_cloud

GAP, STEPS = 3, 3
SPEC = {"subject1/h1/0": (40, 1, {2: range(10, 14)}), "subject1/h1/1": (35, 2, None)}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Module-scoped fixture that writes the synthetic dataset and returns its root, raw arrays and index"""
    root = tmp_path_factory.mktemp("synth")
    raw = make_dataset(root, SPEC)
    return root, raw, build_index(root, list(SPEC), GAP, STEPS)


def make_ds(synth, **kw):
    """Build a WindowDataset on the synthetic data with m=2 and seed=5 unless overridden"""
    root, _, index = synth
    return WindowDataset(root, index, GAP, STEPS, **{"m": 2, "seed": 5, **kw})


def test_each_embodiment_is_centred_on_its_own_frame_zero_mean(synth):
    """Each sample's coordinates equal the raw cloud minus its own frame-0 mean, in float32 with int64 segments"""
    ds = make_ds(synth)
    row = 7
    item = ds[row]
    name, f = str(ds.seq[row]), int(ds.f[row])
    assert item["transition_idx"] == row and len(item["samples"]) == 2
    for s in item["samples"]:
        raw_cloud = np.stack([expected_cloud(synth[1][name], f + k * GAP, s["emb"]) for k in range(STEPS + 1)])
        centre = raw_cloud[0].mean(axis=0, keepdims=True)
        assert np.array_equal(s["coords"].numpy(), raw_cloud - centre)
        assert np.allclose(s["coords"][0].numpy().mean(axis=0), 0, atol=1e-6)
        assert s["coords"].dtype == torch.float32 and s["segment"].dtype == torch.int64


def test_draws_are_distinct_valid_and_a_pure_function_of_seed_row_epoch(synth):
    """Embodiment draws are distinct and valid, repeat for the same seed, row and epoch, and change with another epoch or seed"""
    ds, same, other_seed = make_ds(synth), make_ds(synth), make_ds(synth, seed=6)
    changed_epoch = changed_seed = 0
    for row in range(len(ds)):
        for epoch in range(3):
            d = ds.draw(row, epoch)
            assert len(set(d.tolist())) == 2 and ds.valid[row][d].all()
            assert np.array_equal(d, same.draw(row, epoch))
            changed_epoch += not np.array_equal(d, ds.draw(row, epoch + 1))
            changed_seed += not np.array_equal(d, other_seed.draw(row, epoch))
    assert changed_epoch > 0 and changed_seed > 0


def test_train_index_encodes_the_epoch_and_eval_ignores_it(synth):
    """In training mode an index encodes the epoch (epoch * n + row), while evaluation mode always uses epoch 0"""
    ds, ev = make_ds(synth), make_ds(synth, train=False)
    n, row = len(ds), 5
    for epoch in range(3):
        got = [s["emb"] for s in ds[epoch * n + row]["samples"]]
        assert got == ds.draw(row, epoch).tolist()
    assert [s["emb"] for s in ev[row]["samples"]] == ev.draw(row, 0).tolist()
    assert ev[row]["transition_idx"] == row == ds[2 * n + row]["transition_idx"]


def test_rows_without_enough_valid_embodiments_are_dropped(synth):
    """Windows that lack enough valid embodiments for m are dropped from the dataset"""
    full = make_ds(synth, m=2)
    assert len(make_ds(synth, m=3, embodiments=[0, 1])) == 0
    only_two = make_ds(synth, m=2, embodiments=[0, 2])  # embodiment 2 is invalid in some windows of the first sequence
    assert 0 < len(only_two) < len(full)
    assert only_two.valid[:, [1] + list(range(3, 9))].sum() == 0


def test_batches_do_not_depend_on_the_number_of_workers(synth):
    """DataLoader batches are identical with zero and two workers"""
    ds = make_ds(synth)
    n = len(ds)
    batches = [[e * n + r for r in range(b, b + 4)] for e in range(2) for b in (0, 10, 20)]
    runs = [list(DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=w)) for w in (0, 2)]
    for a, b in zip(*runs):
        assert a["emb"] == b["emb"] and a["seq"] == b["seq"] and torch.equal(a["coords"], b["coords"])


def test_collate_flattens_groups_into_one_ragged_batch(synth):
    """collate concatenates all samples into one ragged batch with matching coordinate shapes, offsets, segments and per-sample metadata"""
    ds = make_ds(synth)
    out = collate([ds[0], ds[1]])
    sizes = [s["coords"].shape[1] for g in (ds[0], ds[1]) for s in g["samples"]]
    assert out["coords"].shape == (STEPS + 1, sum(sizes), 3) and out["offset"].tolist() == np.cumsum(sizes).tolist()
    assert out["segment"].shape == (sum(sizes),) and out["transition_idx"].tolist() == [0, 0, 1, 1]
    assert len(out["emb"]) == len(out["seq"]) == len(out["frame"]) == 4


def test_dataset_points_per_entity_gives_smaller_clouds_with_the_same_windows(synth):
    """points_per_entity yields smaller clouds with the same windows and draws, keeps all hand points and stays centred"""
    root, _, _ = synth
    half = build_index(root, list(SPEC), GAP, STEPS, points_per_entity=4)
    full_ds, half_ds = make_ds(synth), WindowDataset(root, half, GAP, STEPS, m=2, seed=5, points_per_entity=4)
    assert np.array_equal(half["f"], synth[2]["f"]) and np.array_equal(half_ds.draw(3, 1), full_ds.draw(3, 1))
    a, b = full_ds[3]["samples"][0], half_ds[3]["samples"][0]
    assert b["coords"].shape[1] == (a["coords"].shape[1] - 8) // 2 + 8 and np.allclose(b["coords"][0].numpy().mean(axis=0), 0, atol=1e-6)
    assert (b["segment"] == 0).sum() == 8  # every hand point is kept
