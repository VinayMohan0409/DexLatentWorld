"""Tests of the slim episode extractor: the tuning-demo rule, a shard to slim file round trip, and the evaluation copy checks."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from lawm.sim.dexwm.env import benchmark as D  # noqa: E402
from lawm.sim.dexwm.env import extract_episodes as X  # noqa: E402

SHARD = "combine_demos_7.hdf5"
WAYPOINTS = np.array([b"s0-w1"] * 4 + [b"s1-w1"] * 3 + [b"s2-w1"] * 5 + [b"s3-w1"] * 4, dtype=object)


def make_repo(root, train, test):
    """A DexWM repo folder with a split file."""
    (root / "utils").mkdir(parents=True)
    (root / D.SPLIT_FILE).write_text(json.dumps({"train": train, "test": test}))
    return root


def make_shard(path, demos):
    """A small shard with the layout of the real ones (data/<demo>/{states,actions,waypoint_generation_stage_string} + attrs)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = '{"env_name": "x"}'
        for demo in demos:
            g = f.create_group("data/" + demo)
            g["states"], g["actions"] = rng.normal(size=(len(WAYPOINTS), 6)), rng.normal(size=(len(WAYPOINTS), 3))
            g.create_dataset("waypoint_generation_stage_string", data=WAYPOINTS, dtype=h5py.string_dtype())
            g.attrs.update(model_file=f"<mujoco {demo}/>", ep_meta="{}")


def test_tuning_demos_are_the_first_train_demos_by_number(tmp_path):
    """Numbers sort numerically (demo_9 before demo_10), other shards and test demos are left out."""
    train = [[SHARD, "demo_10"], [SHARD, "demo_9"], ["combine_demos_8.hdf5", "demo_1"], [SHARD, "demo_11"]]
    repo = make_repo(tmp_path, train, [[SHARD, "demo_50"]])
    assert X.tuning_demos(repo, SHARD, 2) == ["demo_9", "demo_10"]


def test_tuning_demos_refuse_a_test_split_demo(tmp_path):
    """A demo that is also in the test split stops the extraction."""
    repo = make_repo(tmp_path, [[SHARD, "demo_1"], [SHARD, "demo_2"]], [[SHARD, "demo_1"]])
    with pytest.raises(AssertionError):
        X.tuning_demos(repo, SHARD, 2)


def test_tuning_demos_refuse_more_than_the_shard_has(tmp_path):
    """Asking for more demos than the shard's train split has is an error."""
    repo = make_repo(tmp_path, [[SHARD, "demo_1"]], [])
    with pytest.raises(AssertionError):
        X.tuning_demos(repo, SHARD, 2)


def test_shard_to_slim_file_round_trip_and_manifest(tmp_path):
    """The written files read back equal to the shard; the manifest has one entry per demo with the tasks' frames; the shard itself is untouched."""
    repo = make_repo(tmp_path / "repo", [[SHARD, f"demo_{i}"] for i in range(1, 5)], [])
    shard = tmp_path / SHARD
    make_shard(shard, [f"demo_{i}" for i in range(1, 5)])
    before = shard.read_bytes()
    demos = X.extract_tuning(repo, shard, tmp_path / "out", 3)
    assert demos == ["demo_1", "demo_2", "demo_3"] and shard.read_bytes() == before
    with h5py.File(shard, "r") as f:
        assert X.same_episode(D.load_episode(tmp_path / "out", "demo_2"), X.shard_episode(f, "demo_2"))
    manifest = X.tuning_manifest(repo, SHARD, tmp_path / "out", demos)
    assert [e["demo"] for e in manifest["episodes"]] == demos and manifest["episodes"][0]["tasks"]["reach"] == dict(start=0, target=3)
    assert D.verify_manifest(manifest, tmp_path / "out") == []


def test_writing_never_overwrites(tmp_path):
    """A second write of the same demo is refused."""
    shard = tmp_path / SHARD
    make_shard(shard, ["demo_1"])
    with h5py.File(shard, "r") as f:
        episode = X.shard_episode(f, "demo_1")
    X.write_episode(episode, tmp_path / "out")
    with pytest.raises(AssertionError):
        X.write_episode(episode, tmp_path / "out")


def test_same_episode_sees_every_kind_of_difference(tmp_path):
    """A changed array, waypoint string or text field makes two episodes unequal."""
    shard = tmp_path / SHARD
    make_shard(shard, ["demo_1"])
    with h5py.File(shard, "r") as f:
        a = X.shard_episode(f, "demo_1")
    assert X.same_episode(a, dict(a))
    assert not X.same_episode(a, dict(a, states=a["states"] + 1e-9))
    assert not X.same_episode(a, dict(a, actions=a["actions"] * 2))
    assert not X.same_episode(a, dict(a, waypoints=np.array([b"x"] * len(a["waypoints"]), dtype=object)))
    for key in ("model_file", "ep_meta", "env_args", "source_shard", "source_demo"):
        assert not X.same_episode(a, dict(a, **{key: a[key] + "!"}))


def test_eval_copy_matches_the_manifest(tmp_path):
    """Slim copies of stored episodes pass the manifest check; a manifest that no longer matches the source stops the copy."""
    source = tmp_path / "source"
    shard = tmp_path / SHARD
    make_shard(shard, ["demo_1", "demo_2"])
    with h5py.File(shard, "r") as f:
        for demo in ("demo_1", "demo_2"):
            episode = X.shard_episode(f, demo)
            X.write_episode(dict(episode, source_shard=f"datasets/facebook/dexwm/pick-and-place-2.0/{SHARD}"), source)
    manifest = dict(episodes_dir=str(source), episodes=[D.describe(D.load_episode(source, d), SHARD) for d in ("demo_1", "demo_2")])
    assert X.extract_eval(manifest, tmp_path / "copy") == ["demo_1", "demo_2"]
    manifest["episodes"][0]["sha256"]["states"] = "0" * 64
    with pytest.raises(AssertionError):
        X.extract_eval(manifest, tmp_path / "copy2")
