"""Slim per-demo episode files for the DexWM benchmark: only the fields the simulator reads, taken from a shard or from an older per-demo file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from lawm.sim.dexwm.env import benchmark

SPLIT_FILE = benchmark.SPLIT_FILE


def tuning_demos(dexwm_repo, shard, count):
    """The first count train-split demos of a shard by demo number (a fixed rule: no choosing by outcome); none can be a test-split demo."""
    split = json.loads((Path(dexwm_repo) / SPLIT_FILE).read_text())
    train = sorted((demo for name, demo in map(tuple, split["train"]) if name == shard), key=lambda d: int(d.rsplit("_", 1)[1]))
    test = {demo for _, demo in map(tuple, split["test"])}
    assert len(train) >= count, f"{shard} has only {len(train)} train-split demos"
    assert not test & set(train[:count]), "a tuning demo is in the test split"
    return train[:count]


def shard_episode(shard_file, demo):
    """The episode dict (as benchmark.load_episode returns it) of a demo in an open shard, with the shard's file name as its source."""
    group = shard_file["data/" + demo]
    return dict(demo=demo, states=group["states"][()], actions=group["actions"][()], waypoints=group["waypoint_generation_stage_string"][()],
                model_file=benchmark.as_text(group.attrs["model_file"]), ep_meta=benchmark.as_text(group.attrs["ep_meta"]),
                env_args=benchmark.as_text(shard_file["data"].attrs["env_args"]), source_shard=str(shard_file.filename), source_demo=demo)


def write_episode(episode, out_dir):
    """Write an episode dict to out_dir/<demo>/trajectory_<demo>.hdf5 (refuses to overwrite) and return the path."""
    path = benchmark.episode_path(out_dir, episode["demo"])
    assert not path.exists(), f"{path} exists: not overwriting"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("states", data=episode["states"])
        f.create_dataset("actions", data=episode["actions"])
        f.create_dataset("waypoint_generation_stage_string", data=episode["waypoints"], dtype=h5py.string_dtype())
        for key in ("model_file", "ep_meta", "env_args", "source_shard", "source_demo"):
            f.attrs[key] = episode[key]
    return path


def same_episode(a, b):
    """True when two episode dicts hold identical arrays and text."""
    return (a["demo"] == b["demo"] and all(np.array_equal(a[k], b[k]) for k in ("states", "actions")) and a["waypoints"].tolist() == b["waypoints"].tolist()
            and all(a[k] == b[k] for k in ("model_file", "ep_meta", "env_args", "source_shard", "source_demo")))


def extract_demos(shard_path, demos, out_dir):
    """Write the named demos of a local shard; each file is read back and must equal the shard's data."""
    with h5py.File(shard_path, "r") as shard:
        for demo in demos:
            episode = shard_episode(shard, demo)
            write_episode(episode, out_dir)
            assert same_episode(benchmark.load_episode(out_dir, demo), episode), f"{demo} differs after writing"


def extract_tuning(dexwm_repo, shard_path, out_dir, count):
    """Write the first count train-split demos of a local shard and return their names."""
    demos = tuning_demos(dexwm_repo, Path(shard_path).name, count)
    extract_demos(shard_path, demos, out_dir)
    return demos


def extract_eval(manifest, out_dir):
    """Write slim copies of the manifest's episodes from their stored files; each copy must equal its source and match the manifest fingerprints. Returns the demo names."""
    demos = [e["demo"] for e in manifest["episodes"]]
    for demo in demos:
        source = benchmark.load_episode(manifest["episodes_dir"], demo)
        write_episode(source, out_dir)
        assert same_episode(benchmark.load_episode(out_dir, demo), source), f"{demo} differs after writing"
    bad = benchmark.verify_manifest(manifest, out_dir)
    assert not bad, f"fingerprints differ for {bad}"
    return demos


def tuning_manifest(dexwm_repo, shard_name, episodes_dir, demos):
    """Manifest of the tuning episodes in the format of the evaluation manifest."""
    entries = [benchmark.describe(benchmark.load_episode(episodes_dir, d), shard_name) for d in demos]
    return dict(source=f"first {len(demos)} train-split demos of {shard_name} by demo number ({SPLIT_FILE})", dexwm_repo=str(dexwm_repo),
                episodes_dir=str(episodes_dir), n_episodes=len(entries), episodes=entries)


def main():
    """python -m lawm.sim.dexwm.env.extract_episodes tuning|eval|demos ...: write the slim tuning episodes (and their manifest), slim copies of the evaluation episodes, or named demos of a shard."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="what", required=True)
    tune = sub.add_parser("tuning")
    tune.add_argument("--dexwm-repo", required=True)
    tune.add_argument("--shard", required=True)
    tune.add_argument("--out", required=True)
    tune.add_argument("--manifest", required=True)
    tune.add_argument("--count", type=int, default=5)
    ev = sub.add_parser("eval")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--out", required=True)
    named = sub.add_parser("demos")
    named.add_argument("--shard", required=True)
    named.add_argument("--out", required=True)
    named.add_argument("--demos", nargs="+", required=True)
    args = parser.parse_args()
    if args.what == "demos":
        extract_demos(args.shard, args.demos, args.out)
        print(f"wrote {len(args.demos)} episodes to {args.out}")
    elif args.what == "tuning":
        demos = extract_tuning(args.dexwm_repo, args.shard, args.out, args.count)
        manifest = tuning_manifest(args.dexwm_repo, Path(args.shard).name, args.out, demos)
        Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {len(demos)} tuning episodes to {args.out}: {demos}; manifest {args.manifest}")
    else:
        demos = extract_eval(json.loads(Path(args.manifest).read_text()), args.out)
        print(f"wrote {len(demos)} evaluation episodes to {args.out}, all equal to their sources and matching the manifest")


if __name__ == "__main__":
    main()
