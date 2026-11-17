"""Append robot embodiments (new hands or variants) to stored sequences, written to a new dataset root; stored data stay byte-identical."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]


def resolve_cam_dir(h2o_root: Path, name: str) -> Path:
    """Raw cam4 folder of 'subjectN/<session>/<take>', accepting UniDex's 'subjectN_ego' spelling of the subject folder."""
    subject, session, take = name.split("/")
    for s in (subject, f"{subject}_ego"):
        cam = Path(h2o_root) / s / session / take / "cam4"
        if cam.is_dir():
            return cam
    raise FileNotFoundError(f"no cam4 folder for {name} under {h2o_root}")


def git_rev(path: Path) -> str:
    """Commit (plus '-dirty' if modified) of the git checkout at path, or 'unknown'."""
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


SCENE_KEYS = ("table", "object_canonical", "plane_n", "plane_d", "table_uvc", "cam_poses")
SCENE_TOL_M = 1e-6  # the stored data were made on aarch64; x86 differs at float32 rounding level (~1e-7 m)


def scene_difference(base: dict, scene: dict, n_base: int) -> float:
    """Largest difference (m) between rebuilt and stored scene; raises on any discrete difference."""
    for key in ("object_id", "n_frames"):
        if not np.array_equal(np.asarray(base[key]), np.asarray(scene[key])):
            raise RuntimeError(f"rebuilt scene differs from the stored one in '{key}'")
    if not np.array_equal(base["distractor_counts"], scene["distractor_counts"][:n_base]):
        raise RuntimeError("rebuilt distractor counts differ from the stored ones")
    pairs = [(np.asarray(base[k], np.float64), np.asarray(scene[k], np.float64)) for k in SCENE_KEYS]
    pairs += [(np.asarray(base["distractor_points"][e], np.float64), np.asarray(scene["distractor_points"][e], np.float64))
              for e in range(n_base)]
    worst = 0.0
    for a, b in pairs:
        if a.shape != b.shape:
            raise RuntimeError(f"rebuilt scene array shape {b.shape} differs from stored {a.shape}")
        if a.size:
            worst = max(worst, float(np.abs(a - b).max()))
    return worst


def keep_stored_scene(base: dict, scene: dict, n_base: int) -> dict:
    """Rebuilt scene with the stored arrays put back; only the added embodiments' distractors come from the rebuild."""
    out = dict(scene)
    for k in SCENE_KEYS:
        out[k] = base[k]
    out["distractor_points"] = [base["distractor_points"][e] for e in range(n_base)] + list(scene["distractor_points"][n_base:])
    return out


def export_sequence(name: str, base_root: Path, out_root: Path, h2o_root: Path, object_root: Path, unidex_root: Path,
                    add: Sequence[str], registry_path: Optional[Path] = None, max_frames: int = 0, force: bool = False) -> str:
    """Write base_root/name plus the added embodiments to out_root/name; returns 'done' or 'skipped'."""
    from lawm.embodiments.registry import load_registry
    from lawm.preprocessing.hand_samplers import RobotFastSampler, sample_sequence_robots
    from lawm.preprocessing.scene_export import is_complete, read_hand_clouds, write_sequence
    from lawm.preprocessing.scene_geometry import build_sequence_scene, sequence_tag
    from lawm.preprocessing.unidex_adapter import open_unidex

    out_dir = Path(out_root) / name
    if is_complete(out_dir) and not force:
        return "skipped"
    registry = load_registry(registry_path)
    base_dir = Path(base_root) / name
    hands, valid = read_hand_clouds(base_dir)
    with np.load(base_dir / "sequence.npz", allow_pickle=True) as z:
        base = {k: z[k] for k in z.files}
    base_names = [str(n) for n in base["embodiment_names"]]  # stored spelling: the distractor seeds use it
    clash = {a for a in add if a in {n.lower() for n in base_names}}
    if clash:
        raise ValueError(f"{name}: {sorted(clash)} already stored")
    n_frames, points = hands.shape[0], hands.shape[2]
    if max_frames:
        n_frames = min(n_frames, max_frames)
        hands, valid = hands[:n_frames], valid[:n_frames]

    cam_dir = resolve_cam_dir(h2o_root, name)
    with open_unidex(unidex_root) as (uv, dataset):
        index_of = {str(dataset[i].index).replace("_ego", ""): i for i in range(len(dataset))}
        seq = dataset[index_of[f"{name}/cam4"]]
        if len(seq) < n_frames:
            raise RuntimeError(f"{name}: UniDex has {len(seq)} frames, the stored data {n_frames}")
        samplers = {}
        for a in add:
            e = registry[a]
            if e.kind != "robot":
                raise ValueError(f"{a}: only robot embodiments can be added")
            samplers[a] = RobotFastSampler(uv, "H2o", e.unidex_hand, points, exclude_links=e.exclude_links, seed_name=e.seed_name)
        new_hands, new_valid = sample_sequence_robots(seq, cam_dir, samplers, points, max_frames=n_frames)

    all_hands = np.concatenate([hands, new_hands], axis=1)
    all_valid = np.concatenate([valid, new_valid], axis=1)
    subject, session, take = name.split("/")
    scene = build_sequence_scene(cam_dir, object_root, all_hands, all_valid, sequence_tag(subject, session, take),
                                 points=points, names=base_names + list(add))
    diff = None
    if not max_frames:  # a truncated test run changes the table frames, so the comparison only holds for full sequences
        diff = scene_difference(base, scene, len(base_names))
        if diff > SCENE_TOL_M:
            raise RuntimeError(f"{name}: rebuilt scene differs from the stored one by {diff:.3g} m (> {SCENE_TOL_M} m); not written")
        scene = keep_stored_scene(base, scene, len(base_names))

    meta = {"base_root": str(base_root), "added_embodiments": {a: vars(registry[a]) for a in add},
            "added_valid_fraction": {a: float(new_valid[:, i].mean()) for i, a in enumerate(add)},
            "unidex_commit": git_rev(Path(unidex_root)), "lawm_commit": git_rev(REPO),
            "stored_scene_max_abs_diff_m": diff, "max_frames": max_frames}
    stored_meta = base_dir / "export_metadata.json"
    if stored_meta.exists():
        meta["base_export_metadata"] = json.loads(stored_meta.read_text())
    write_sequence(out_dir, scene, all_hands, all_valid, meta)
    return "done"


def _run_one(kwargs):
    """Process-pool entry point; returns (name, status, seconds)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    return kwargs["name"], export_sequence(**kwargs), round(time.time() - t0, 1)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-root", type=Path, required=True, help="stored dataset root (read only)")
    ap.add_argument("--out-root", type=Path, required=True, help="new dataset root; must differ from --base-root")
    ap.add_argument("--h2o-root", type=Path, required=True, help="raw H2O: subjectN[_ego]/<session>/<take>/cam4")
    ap.add_argument("--object-root", type=Path, required=True, help="H2O object meshes: <name>/<name>.obj")
    ap.add_argument("--unidex-root", type=Path, required=True)
    ap.add_argument("--add", nargs="+", required=True, help="registry names to add, in output order")
    ap.add_argument("--registry", type=Path, default=None, help="default: configs/embodiments.json")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sequences", nargs="+", help="e.g. subject3/k2/0")
    sel.add_argument("--split", type=Path, help="split json; with --split-key")
    ap.add_argument("--split-key", default="val")
    ap.add_argument("--max-frames", type=int, default=0, help="test runs only: skips the stored-scene check")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.out_root.resolve() == args.base_root.resolve():
        ap.error("--out-root must differ from --base-root (the stored dataset is never overwritten)")
    names = args.sequences or json.loads(args.split.read_text())[args.split_key]

    jobs = [dict(name=n, base_root=args.base_root, out_root=args.out_root, h2o_root=args.h2o_root,
                 object_root=args.object_root, unidex_root=args.unidex_root, add=args.add, registry_path=args.registry,
                 max_frames=args.max_frames, force=args.force) for n in names]
    args.out_root.mkdir(parents=True, exist_ok=True)
    failed = []
    if args.workers <= 1:
        for j in jobs:
            log.info("%s %s (%.1fs)", *_run_one(j))
    else:  # spawn: each worker opens its own UniDex/PyBullet state
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_run_one, j): j["name"] for j in jobs}
            for fut in as_completed(futures):
                try:
                    log.info("%s %s (%.1fs)", *fut.result())
                except Exception as exc:  # report every failure, then exit non-zero
                    failed.append(futures[fut])
                    log.error("%s FAILED: %s", futures[fut], exc)
    if failed:
        raise SystemExit(f"{len(failed)} sequence(s) failed: {failed}")


if __name__ == "__main__":
    main()
