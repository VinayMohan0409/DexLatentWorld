"""Tests of the DexWM benchmark episode list: official indices, start/target frames per task, manifest fingerprints."""
import json

import numpy as np
import pytest

from lawm.sim.dexwm.env import benchmark as D

WAYPOINTS = np.array([b"s0-w1"] * 4 + [b"s1-w1"] * 3 + [b"s1-w2"] * 2 + [b"s2-w1"] * 5 + [b"s2-w2"] * 3 + [b"s3-w1"] * 4 + [b"s3-w2"] * 2)  # 23 frames: s0 0-3, s1-w1 4-6, s1-w2 7-8, s2-w1 9-13, s2-w2 14-16, s3-w1 17-20, s3-w2 21-22


def test_the_official_indices_are_fifty_distinct_sorted_values():
    """DexWM lists 50 distinct test-split indices in increasing order."""
    assert len(D.OFFICIAL_INDICES) == 50 and len(set(D.OFFICIAL_INDICES)) == 50 and list(D.OFFICIAL_INDICES) == sorted(D.OFFICIAL_INDICES)


def test_task_frames_follow_the_waypoint_stages():
    """Reach ends before s1-w1, grasp runs s1-w1 to before s2-w1, and place starts 20 frames after s2-w1 (clipped) and ends before s3-w1."""
    assert D.get_indices(WAYPOINTS, "reach") == (0, 3)
    assert D.get_indices(WAYPOINTS, "grasp") == (4, 8)
    assert D.get_indices(WAYPOINTS, "place") == (22, 16)  # s2-w1 starts at 9 and 9 + 20 is clipped to the last frame (22), as in DexWM


def test_place_target_is_the_last_frame_without_a_third_stage():
    """With no s3-w1 the place target is the last frame."""
    assert D.get_indices(WAYPOINTS[:17], "place")[1] == 16


def test_unknown_task_is_an_error():
    """An unknown task name raises."""
    with pytest.raises(ValueError):
        D.get_indices(WAYPOINTS, "push")


def test_official_entries_select_fifty_demos_in_split_order(tmp_path):
    """Entries come out in the split file's order, only for the official indices."""
    test = [[f"combine_demos_{i % 3}.hdf5", f"demo_{i}"] for i in range(200)]
    (tmp_path / "utils").mkdir()
    (tmp_path / D.SPLIT_FILE).write_text(json.dumps({"test": test}))
    entries = D.official_entries(tmp_path)
    assert [e[1] for e in entries] == [f"demo_{i}" for i in D.OFFICIAL_INDICES]


def test_fingerprint_of_byte_strings_does_not_depend_on_memory_addresses():
    """Two separately built object arrays of equal byte strings hash equal, and a changed string changes the hash."""
    a = np.array([b"s0-w1", b"s1-w1"], dtype=object)
    b = np.array([bytes(x) for x in (bytearray(b"s0-w1"), bytearray(b"s1-w1"))], dtype=object)
    assert D.sha256_of(a) == D.sha256_of(b) and D.sha256_of(a) != D.sha256_of(np.array([b"s0-w1", b"s1-w2"], dtype=object))


def write_episode(root, demo, shard, states):
    """Write a minimal extracted episode file."""
    import h5py
    path = D.episode_path(root, demo)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f["states"], f["actions"], f["waypoint_generation_stage_string"] = states, np.zeros((len(WAYPOINTS), 4)), WAYPOINTS
        f.attrs.update(model_file="<mujoco/>", ep_meta="{}", env_args="{}", source_shard=f"datasets/facebook/dexwm/pick-and-place-2.0/{shard}", source_demo=demo)


def test_manifest_detects_a_changed_state(tmp_path):
    """A manifest verifies against the files it was built from and flags a demo whose states changed by one bit."""
    pytest.importorskip("h5py")
    states = np.random.default_rng(0).normal(size=(len(WAYPOINTS), 7))
    write_episode(tmp_path, "demo_1", "combine_demos_1.hdf5", states)
    manifest = dict(episodes=[D.describe(D.load_episode(tmp_path, "demo_1"), "combine_demos_1.hdf5")])
    assert D.verify_manifest(manifest, tmp_path) == []
    changed = tmp_path / "other"
    write_episode(changed, "demo_1", "combine_demos_1.hdf5", np.nextafter(states, np.inf))
    assert D.verify_manifest(manifest, changed) == ["demo_1"]


def test_describe_rejects_an_episode_from_another_shard(tmp_path):
    """An episode recorded as coming from a different shard than DexWM lists is refused."""
    pytest.importorskip("h5py")
    write_episode(tmp_path, "demo_1", "combine_demos_2.hdf5", np.zeros((len(WAYPOINTS), 3)))
    with pytest.raises(AssertionError):
        D.describe(D.load_episode(tmp_path, "demo_1"), "combine_demos_1.hdf5")
