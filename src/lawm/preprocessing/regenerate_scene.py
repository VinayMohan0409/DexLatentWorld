"""Rebuild every sequence's shared scene (table, object sample, distractors) from raw H2O data and the stored hand clouds: `python -m lawm.preprocessing.regenerate_scene --h2o-root <H2O> --object-root <objects> --out-root <dir> --workers 4`."""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from lawm.preprocessing.h2o_readers import list_sequences
from lawm.preprocessing.scene_export import is_complete, read_hand_clouds, write_sequence
from lawm.preprocessing.scene_geometry import build_sequence_scene, sequence_tag

log = logging.getLogger(__name__)
DISTRACTOR_RULE = "largest convex-hull face (exporter of 19 Aug 2026)"


def regenerate_sequence(cam_dir: Path, object_root: Path, out_dir: Path, h2o_root: Path, force: bool = False) -> str:
    """Rebuild one sequence; returns 'done' or 'skipped' (already complete)."""
    cam_dir, out_dir = Path(cam_dir), Path(out_dir)
    if is_complete(out_dir) and not force:
        return "skipped"
    stored = cam_dir / "pointclouds_world"
    subject, session, take, _ = cam_dir.relative_to(h2o_root).parts
    hands, valid = read_hand_clouds(stored)
    scene = build_sequence_scene(cam_dir, object_root, hands, valid, sequence_tag(subject, session, take))

    meta = {}
    stored_meta = stored / "export_metadata.json"
    if stored_meta.exists():  # keep how the hands were made
        old = json.loads(stored_meta.read_text())
        meta = {k: old[k] for k in ("mano_source", "refined_metadata_sha256", "robot_retarget_source") if k in old}
    meta.update({"distractor_rest_rule": DISTRACTOR_RULE, "hands_copied_from": str(stored)})
    write_sequence(out_dir, scene, hands, valid, meta)
    return "done"


def _run_one(args):
    """Process-pool entry point; returns (index, status, seconds)."""
    index, cam_dir, object_root, out_dir, h2o_root, force = args
    t0 = time.time()
    return index, regenerate_sequence(cam_dir, object_root, out_dir, h2o_root, force), round(time.time() - t0, 1)


def main(argv: Optional[list] = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2o-root", type=Path, required=True, help="H2O tree containing <seq>/cam4/pointclouds_world")
    ap.add_argument("--object-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seq-indices", type=int, nargs="+", help="1-based sequence positions; default all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs = []
    for index, cam_dir in list_sequences(args.h2o_root):
        if args.seq_indices and index not in args.seq_indices:
            continue
        rel = cam_dir.relative_to(args.h2o_root).parts[:3]
        jobs.append((index, cam_dir, args.object_root, args.out_root.joinpath(*rel), args.h2o_root, args.force))

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        results = (_run_one(j) for j in jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = (f.result() for f in as_completed([pool.submit(_run_one, j) for j in jobs]))
    for index, status, seconds in results:
        log.info("[%d] %s (%.1fs)", index, status, seconds)


if __name__ == "__main__":
    main()
