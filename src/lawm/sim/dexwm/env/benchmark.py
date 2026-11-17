"""The 50 episodes of DexWM's robot-manipulation benchmark: which demos, the start and target frame of each task, and a fingerprint of each stored state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# sim_test.py of the DexWM repository: the test-split entries "with clear goal observations and well-defined tasks"
OFFICIAL_INDICES = (3, 9, 12, 15, 17, 19, 20, 25, 26, 28, 35, 36, 37, 39, 41, 44, 55, 56, 57, 59, 64, 68, 69, 70, 83, 86, 87, 88, 104, 109, 110, 111,
                    112, 115, 120, 129, 131, 134, 141, 145, 146, 149, 151, 152, 155, 157, 158, 160, 163, 168)
TASKS = ("reach", "grasp", "place")
SPLIT_FILE = "utils/split_indices_robocasa_pp.json"


def first_index(waypoints, tag):
    """Index of the first waypoint string equal to tag (bytes), or None."""
    return next((i for i, w in enumerate(waypoints) if w == tag), None)


def get_indices(waypoints, task):
    """(start frame, target frame) of a task, as sim_test.get_indices computes them from the waypoint-stage strings."""
    if task == "reach":
        return 0, first_index(waypoints, b"s1-w1") - 1
    if task == "grasp":
        return first_index(waypoints, b"s1-w1"), first_index(waypoints, b"s2-w1") - 1
    if task == "place":
        target = first_index(waypoints, b"s3-w1")
        return min(first_index(waypoints, b"s2-w1") + 20, len(waypoints) - 1), (len(waypoints) - 1 if target is None else target - 1)
    raise ValueError(f"unknown task {task!r}; choose from {TASKS}")


def official_entries(dexwm_repo):
    """(shard file, demo name) of the 50 benchmark demos, in the order DexWM's sim_test.py runs them."""
    test = json.loads((Path(dexwm_repo) / SPLIT_FILE).read_text())["test"]
    wanted = set(OFFICIAL_INDICES)
    entries = [tuple(e) for i, e in enumerate(test) if i in wanted]
    assert len(entries) == len(OFFICIAL_INDICES) == len({e[1] for e in entries}), "the official indices do not select 50 distinct demos"
    return entries


def episode_path(episodes_dir, demo):
    """Path of the extracted per-demo file."""
    return Path(episodes_dir) / demo / f"trajectory_{demo}.hdf5"


def as_text(value):
    """An HDF5 attribute as str."""
    return value.decode() if isinstance(value, bytes) else str(value)


def load_episode(episodes_dir, demo):
    """The stored episode: states, actions, waypoint-stage strings, scene XML (model_file), ep_meta, env_args and the recorded source shard."""
    import h5py
    with h5py.File(episode_path(episodes_dir, demo), "r") as f:
        return dict(demo=demo, states=f["states"][()], actions=f["actions"][()], waypoints=f["waypoint_generation_stage_string"][()],
                    model_file=as_text(f.attrs["model_file"]), ep_meta=as_text(f.attrs["ep_meta"]), env_args=as_text(f.attrs["env_args"]),
                    source_shard=as_text(f.attrs["source_shard"]), source_demo=as_text(f.attrs["source_demo"]))


def sha256_of(value):
    """Hex SHA-256 of a str, of an array of byte strings (their text joined by newlines), or of a numeric array's bytes."""
    if isinstance(value, str):
        data = value.encode()
    elif value.dtype.kind in "OSU":  # object arrays hold pointers, so their raw bytes change from run to run
        data = b"\n".join(x if isinstance(x, bytes) else str(x).encode() for x in value.tolist())
    else:
        data = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(data).hexdigest()


def describe(episode, shard):
    """Manifest entry of one episode: provenance, frame counts, each task's start and target frame, and fingerprints of states, scene and metadata."""
    assert episode["source_demo"] == episode["demo"] and episode["source_shard"].endswith("/" + shard), (
        f"{episode['demo']} says it came from {episode['source_shard']} / {episode['source_demo']}, DexWM lists {shard}")
    return dict(demo=episode["demo"], shard=shard, n_frames=int(len(episode["states"])), state_length=int(episode["states"].shape[1]),
                tasks={t: dict(zip(("start", "target"), (int(i) for i in get_indices(episode["waypoints"], t)))) for t in TASKS},
                sha256=dict(states=sha256_of(episode["states"]), model_file=sha256_of(episode["model_file"]), ep_meta=sha256_of(episode["ep_meta"]),
                            waypoints=sha256_of(episode["waypoints"])))


def build_manifest(dexwm_repo, episodes_dir):
    """Manifest of the 50 benchmark episodes read from episodes_dir, in DexWM's run order."""
    entries = [describe(load_episode(episodes_dir, demo), shard) for shard, demo in official_entries(dexwm_repo)]
    return dict(source="sim_test.py indices into " + SPLIT_FILE + " of the DexWM repository", dexwm_repo=str(dexwm_repo), episodes_dir=str(episodes_dir),
                n_episodes=len(entries), episodes=entries)


def verify_manifest(manifest, episodes_dir):
    """Names of the demos whose stored files no longer match the manifest's fingerprints and frame choices (empty if all match)."""
    bad = []
    for entry in manifest["episodes"]:
        now = describe(load_episode(episodes_dir, entry["demo"]), entry["shard"])
        if now != entry:
            bad.append(entry["demo"])
    return bad


def main():
    """Write the manifest (python -m lawm.sim.dexwm.env.benchmark --dexwm-repo R --episodes-dir E --out configs/stage3/dexwm_episodes.json), or verify an existing one with --verify."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexwm-repo", required=True)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true", help="check the existing --out file instead of writing it")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(Path(args.out).read_text())
        bad = verify_manifest(manifest, args.episodes_dir)
        print(f"{len(manifest['episodes']) - len(bad)} of {len(manifest['episodes'])} episodes match" + (f"; differ: {bad}" if bad else ""))
        raise SystemExit(1 if bad else 0)
    manifest = build_manifest(args.dexwm_repo, args.episodes_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.out}: {manifest['n_episodes']} episodes")


if __name__ == "__main__":
    main()
