"""Unit tests for lawm.data.config, sequences and index on tiny synthetic sequences."""
import numpy as np
import pytest
from synthetic_data import COUNTS, E, PTS, make_dataset, make_sequence

from lawm.data.config import data_root
from lawm.data.index import build_index, fingerprint, load_or_build_index, sequence_windows
from lawm.data.sequences import SEG_DISTRACTOR, SEG_HAND, SEG_OBJECT, SEG_TABLE, Sequence


def test_data_root_needs_the_environment_variable(monkeypatch, tmp_path):
    """data_root raises without LAWM_DATA_ROOT and returns that path once it is set"""
    monkeypatch.delenv("LAWM_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="LAWM_DATA_ROOT"):
        data_root()
    monkeypatch.setenv("LAWM_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path


def expected_cloud(raw, f, e):
    """Build the cloud the dataset should return for frame f and embodiment e: hand, posed object, table, then distractors"""
    pose = raw["poses"][f]
    obj = raw["scene"]["object_canonical"] @ pose[:3, :3].T + pose[:3, 3]
    return np.concatenate([raw["hands"][f, e], obj, raw["scene"]["table"], raw["scene"]["distractor_points"][e].reshape(-1, 3)])


def test_clouds_are_hand_object_table_distractors_in_that_order(tmp_path):
    """Sequence.clouds returns float32 coordinates and int64 segment labels laid out as hand, object, table, distractors"""
    raw = make_sequence(tmp_path / "s", 10, 0)
    q = Sequence(tmp_path / "s")
    pairs = q.clouds([2, 5], [3, 0])
    for (coords, segment), e in zip(pairs, [3, 0]):
        assert coords.dtype == np.float32 and segment.dtype == np.int64
        assert np.array_equal(coords, np.stack([expected_cloud(raw, f, e) for f in (2, 5)]))
        want = [SEG_HAND] * PTS + [SEG_OBJECT] * PTS + [SEG_TABLE] * PTS + [SEG_DISTRACTOR] * PTS * COUNTS[e]
        assert segment.tolist() == want and q.n_points(e) == len(want)
    assert q.names[0] == "human" and q.names[1] == "hand0"


def test_frame_validity_and_missing_frame_files(tmp_path):
    """frame_valid reports the stored validity and raises when a frame file has been removed"""
    raw = make_sequence(tmp_path / "s", 10, 0, invalid={4: [3, 4]})
    q = Sequence(tmp_path / "s")
    assert np.array_equal(q.frame_valid(), raw["valid"])
    (tmp_path / "s" / "000007.npz").unlink()
    with pytest.raises(ValueError, match="frame files"):
        q.frame_valid()


@pytest.mark.parametrize("gap,n_steps", [(3, 4), (2, 2)])
def test_sequence_windows_match_a_brute_force_check(gap, n_steps):
    """sequence_windows agrees with a brute-force scan of which start frames and embodiments are valid across a whole window"""
    rng = np.random.default_rng(0)
    valid = rng.random((60, E)) > 0.04
    f, vw = sequence_windows(valid, gap, n_steps)
    want_f, want_vw = [], []
    for s in range(len(valid)):
        frames = list(range(s, s + gap + 1)) + [s + k * gap for k in range(2, n_steps + 1)]
        if max(frames) >= len(valid):
            continue
        ok = valid[frames].all(axis=0)
        if ok.any():
            want_f.append(s)
            want_vw.append(ok)
    assert f.tolist() == want_f and np.array_equal(vw, np.array(want_vw))
    assert sequence_windows(valid[:gap], gap, n_steps)[0].size == 0


def test_build_index_orders_rows_and_records_the_human_point_count(tmp_path):
    """build_index sorts rows by sequence, records each cloud's point count and marks windows invalid where an embodiment is missing"""
    make_dataset(tmp_path, {"subject1/h1/1": (30, 1, None), "subject1/h1/0": (25, 2, {2: range(10, 13)})})
    idx = build_index(tmp_path, ["subject1/h1/1", "subject1/h1/0"], 3, 2)
    seq = idx["seq"].tolist()
    assert seq == sorted(seq) and seq[0] == "subject1/h1/0"
    assert (idx["length"] == PTS * (3 + COUNTS[0])).all()
    first = idx["seq"] == "subject1/h1/0"
    assert (~idx["valid_window"][first][:, 2]).any() and idx["valid_window"][~first].all()


def test_cached_index_is_reused_and_a_changed_dataset_gets_a_new_one(tmp_path):
    """load_or_build_index reuses its cache file until the dataset is rewritten, which produces a second cache file"""
    data, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(data, {"subject1/h1/0": (25, 1, None)})
    first = load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2)
    again = load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2)
    assert all(np.array_equal(first[k], again[k]) for k in first) and len(list(cache.glob("*.npz"))) == 1
    make_sequence(data / "subject1/h1/0", 25, 99)  # the dataset is rewritten
    load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2)
    assert len(list(cache.glob("*.npz"))) == 2
    assert fingerprint(data, ["subject1/h1/0"]) != fingerprint(data, ["subject1/h1/0", "subject1/h1/0"])


def test_an_index_built_for_other_settings_is_refused(tmp_path):
    """Loading a cached index stored under different gap settings raises a different-dataset error"""
    data, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(data, {"subject1/h1/0": (25, 1, None)})
    load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2)
    (path,) = cache.glob("*.npz")
    path.rename(cache / path.name.replace("g3_n2", "g4_n2"))
    with pytest.raises(ValueError, match="different dataset"):
        load_or_build_index(cache, "train", data, ["subject1/h1/0"], 4, 2)


def test_points_per_entity_keeps_every_other_point_of_each_entity_and_all_hand_points(tmp_path):
    """points_per_entity subsamples every non-hand entity by striding while keeping all hand points, in both the coordinates and segment labels"""
    raw = make_sequence(tmp_path / "s", 10, 0)
    full, half = Sequence(tmp_path / "s"), Sequence(tmp_path / "s", points_per_entity=PTS // 2)
    for e in (0, 4):
        (c_full, s_full), (c_half, s_half) = full.clouds([2, 5], [e])[0], half.clouds([2, 5], [e])[0]
        keep = np.concatenate([np.arange(PTS)] + [PTS * b + np.arange(0, PTS, 2) for b in range(1, 3 + COUNTS[e])])  # hand, then entity blocks
        assert np.array_equal(c_half, c_full[:, keep]) and np.array_equal(s_half, s_full[keep])
        assert (s_half == SEG_HAND).sum() == PTS and (s_half == SEG_DISTRACTOR).sum() == PTS // 2 * COUNTS[e]
        assert half.n_points(e) == len(keep)
    assert np.array_equal(full.clouds([3], [1])[0][0], np.stack([expected_cloud(raw, 3, 1)]))


def test_points_per_entity_must_divide_the_stored_points(tmp_path):
    """Sequence rejects a points_per_entity that does not divide the stored point count"""
    make_sequence(tmp_path / "s", 10, 0)
    with pytest.raises(ValueError, match="does not divide"):
        Sequence(tmp_path / "s", points_per_entity=3)
    with pytest.raises(ValueError, match="does not divide"):
        Sequence(tmp_path / "s", points_per_entity=PTS * 2)


def test_index_length_and_cache_depend_on_points_per_entity(tmp_path):
    """The index's cloud lengths and cache file differ with points_per_entity and are reused on a repeat call"""
    data, cache = tmp_path / "data", tmp_path / "cache"
    make_dataset(data, {"subject1/h1/0": (25, 1, None)})
    full = load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2)
    half = load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2, points_per_entity=PTS // 2)
    assert len(list(cache.glob("*.npz"))) == 2
    assert (full["length"] == PTS * (3 + COUNTS[0])).all() and (half["length"] == PTS + PTS // 2 * (2 + COUNTS[0])).all()
    again = load_or_build_index(cache, "train", data, ["subject1/h1/0"], 3, 2, points_per_entity=PTS // 2)
    assert np.array_equal(again["length"], half["length"]) and len(list(cache.glob("*.npz"))) == 2
