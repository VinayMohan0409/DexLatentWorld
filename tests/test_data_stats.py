"""Unit tests for lawm.data.stats: correct moments, reproducible sums, failures raise."""
import numpy as np
import pytest
from synthetic_data import make_dataset

from lawm.data.dataset import WindowDataset
from lawm.data.index import build_index
from lawm.data.stats import VAR_FLOOR, compute_norm_stats, load_norm_stats, save_norm_stats

from test_data_index import expected_cloud

GAP, STEPS = 3, 2
SPEC = {"subject1/h1/0": (30, 1, {2: range(10, 14)}), "subject1/h1/1": (25, 2, None)}


@pytest.fixture()
def synth(tmp_path):
    """Fixture that writes the synthetic dataset and returns its root, raw arrays and an evaluation WindowDataset"""
    raw = make_dataset(tmp_path, SPEC)
    index = build_index(tmp_path, list(SPEC), GAP, STEPS)
    return tmp_path, raw, WindowDataset(tmp_path, index, GAP, STEPS, m=1, train=False)


def test_stats_equal_a_direct_computation_over_the_sampled_windows(synth):
    """compute_norm_stats matches per-step mean and variance computed directly from the raw windows"""
    root, raw, ds = synth
    stats = compute_norm_stats(ds, windows=10_000, seed=3)  # more windows than exist, so all are used
    means, sqs = [], []
    for j in range(len(ds)):
        e = int(np.random.default_rng(3 * 1_000_003 + j).choice(np.flatnonzero(ds.valid[j])))
        cloud = np.stack([expected_cloud(raw[str(ds.seq[j])], int(ds.f[j]) + k * GAP, e) for k in range(STEPS + 1)])
        cloud = cloud - cloud[0].mean(axis=0, keepdims=True)
        delta = (cloud[1:] - cloud[0:1]).astype(np.float64)
        means.append(delta.mean(axis=1))
        sqs.append((delta ** 2).mean(axis=1))
    mean = np.mean(means, axis=0)
    var = np.maximum(np.mean(sqs, axis=0) - mean ** 2, VAR_FLOOR)
    assert stats["windows"] == len(ds) and stats["per_step_mean"].shape == (STEPS, 3)
    assert np.allclose(stats["per_step_mean"], mean, rtol=1e-5, atol=1e-8)
    assert np.allclose(stats["per_step_var"], var, rtol=1e-5, atol=1e-10)


def test_the_sample_and_the_result_do_not_depend_on_the_number_of_workers(synth):
    """Statistics are identical with one or two workers and change with the seed"""
    _, _, ds = synth
    one = compute_norm_stats(ds, windows=40, seed=1, workers=1)
    two = compute_norm_stats(ds, windows=40, seed=1, workers=2)
    assert one["windows"] == 40
    assert np.array_equal(one["per_step_mean"], two["per_step_mean"]) and np.array_equal(one["per_step_var"], two["per_step_var"])
    assert not np.array_equal(one["per_step_mean"], compute_norm_stats(ds, windows=40, seed=2)["per_step_mean"])


def test_a_missing_frame_file_raises_instead_of_being_skipped(synth):
    """compute_norm_stats raises FileNotFoundError when a needed frame file is missing"""
    root, _, ds = synth
    (root / "subject1/h1/1" / "000010.npz").unlink()
    with pytest.raises(FileNotFoundError):
        compute_norm_stats(ds, windows=10_000)


def test_a_scene_without_motion_is_floored(tmp_path):
    """A scene with no motion gets its variance floored and the floored entries counted"""
    make_dataset(tmp_path, {"subject1/h1/0": (20, 1, None)})
    with np.load(tmp_path / "subject1/h1/0" / "000000.npz") as z:
        first = {k: z[k] for k in z.files}
    for f in range(20):
        np.savez(tmp_path / "subject1/h1/0" / f"{f:06d}.npz", **first)
    ds = WindowDataset(tmp_path, build_index(tmp_path, ["subject1/h1/0"], GAP, STEPS), GAP, STEPS, m=1, train=False)
    stats = compute_norm_stats(ds)
    assert stats["n_floored"] == STEPS * 3 and np.allclose(stats["per_step_var"], VAR_FLOOR)


def test_saved_statistics_are_refused_for_another_dataset_or_setting(synth, tmp_path):
    """Saved statistics reload exactly with matching settings and raise when the fingerprint, split, windows, seed, steps or gap differ"""
    _, _, ds = synth
    stats = compute_norm_stats(ds, windows=20, seed=1)
    path = tmp_path / "stats.npz"
    save_norm_stats(path, stats, "fingerprint-a", "train")
    ok = dict(fingerprint="fingerprint-a", split="train", n_steps=STEPS, gap=GAP, windows=20, seed=1)
    got = load_norm_stats(path, **ok)
    assert np.array_equal(got["per_step_mean"], stats["per_step_mean"]) and np.array_equal(got["per_step_var"], stats["per_step_var"])
    for wrong in (dict(fingerprint="fingerprint-b"), dict(split="val"), dict(windows=30), dict(seed=2), dict(n_steps=STEPS + 1), dict(gap=GAP + 1)):
        with pytest.raises(ValueError, match="do not match"):
            load_norm_stats(path, **{**ok, **wrong})


def test_stats_record_and_check_points_per_entity(synth, tmp_path):
    """Statistics record points_per_entity, have fewer total points than the full version and raise when loaded with a different value"""
    root, _, _ = synth
    ds = WindowDataset(root, build_index(root, list(SPEC), GAP, STEPS, points_per_entity=4), GAP, STEPS, m=1, train=False, points_per_entity=4)
    stats = compute_norm_stats(ds, windows=20, seed=1)
    assert int(stats["points_per_entity"]) == 4 and stats["total_points"] < compute_norm_stats(synth[2], windows=20, seed=1)["total_points"]
    path = tmp_path / "stats.npz"
    save_norm_stats(path, stats, "fp", "train")
    ok = dict(fingerprint="fp", split="train", n_steps=STEPS, gap=GAP, windows=20, seed=1, points_per_entity=4)
    load_norm_stats(path, **ok)
    with pytest.raises(ValueError, match="do not match"):
        load_norm_stats(path, **{**ok, "points_per_entity": 0})
