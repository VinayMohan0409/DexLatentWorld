"""Refine every tracked hand of one H2O sequence and write per-frame camera-frame MANO files: `python -m lawm.preprocessing.contact_refinement.run --seq-path <sequence cam dir> --object-root <meshes> --mano-root <mano models>`."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from lawm.preprocessing.h2o_readers import load_camera_to_world, parse_mano_file
from lawm.preprocessing.contact_refinement.optimizer import (
    RefineConfig, build_batch, make_mano_layer, mano_forward, precompute_contacts, temporal_refine)
from lawm.preprocessing.contact_refinement.sequence import load_sequence
from lawm.preprocessing.contact_refinement.mano_refined import (
    atomic_json, file_sha256, read_refined_frame, write_refined_frame)

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1
ROUNDTRIP_TOL_M = 1e-5


def config_hash(cfg: RefineConfig, source_hashes: dict):
    """Hash the hyperparameters and raw input files, so a rerun is skipped only when nothing relevant changed."""
    payload = {"schema": SCHEMA_VERSION, **asdict(cfg), "source_hashes": source_hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def world_roundtrip_error(seq_path: Path, result: dict, side: str, indexes, mano_root: Path) -> dict:
    """Max vertex error after converting the world translation back to camera frame and forward again."""
    model = make_mano_layer(mano_root, side, "cpu")
    errs = {}
    for i in indexes:
        frame = int(result["frame_ids"][i])
        T_wc = load_camera_to_world(seq_path, frame)
        t_cam = T_wc[:3, :3].T @ (result["trans_world_final"][i] - T_wc[:3, 3])
        mTc = torch.tensor((T_wc @ np.block([[np.eye(3), t_cam[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))[None],
                           dtype=torch.float32)
        beta = torch.tensor(result["beta"][i:i + 1], dtype=torch.float32)
        with torch.no_grad():
            verts = mano_forward(model, torch.tensor(result["rot"][i:i + 1]),
                                 torch.tensor(result["pose"][i:i + 1]), beta, mTc)
        errs[str(frame)] = float(np.max(np.abs(verts.numpy()[0] - result["hand_verts"][i])))
    return errs


def refine_sequence(seq_path: Path, out: Path, object_root: Path, mano_root: Path, device: str,
                    cfg: RefineConfig, force: bool = False) -> None:
    """Refine both hands of one sequence and atomically publish <out>/NNNNNN.npz plus metadata."""
    raw_files = sorted((seq_path / "hand_pose_mano").glob("*.txt"))
    cfg_hash, cfg_payload = config_hash(cfg, {p.name: file_sha256(p) for p in raw_files})
    if (out / "complete.json").exists() and not force:
        if json.loads((out / "complete.json").read_text()).get("config_hash") == cfg_hash:
            log.info("[SKIP] compatible complete output %s", out)
            return

    staging = out.with_name(out.name + f".staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.time()

    results, timings = {}, {}
    for side, seq in load_sequence(seq_path, object_root).items():
        t0 = time.time()
        mano_model = make_mano_layer(mano_root, side, device)
        data = build_batch(seq, device)
        precomp = precompute_contacts(data, mano_model, cfg, side)
        result = temporal_refine(seq, data, precomp, mano_model, cfg, side, device)
        result["beta"] = np.tile(seq.hand_beta, (len(seq.frame_ids), 1))
        for key in ("hand_verts", "rot", "pose", "trans_world_final", "beta"):
            if not np.isfinite(result[key]).all():
                raise FloatingPointError(f"{side}/{key} contains NaN/Inf")
        results[side] = result
        timings[side] = {"seconds": time.time() - t0, "frames": len(seq.frame_ids), "best_loss": result["best_loss"],
                         "best_iter": result["best_iter"], "n_iter_actual": result["n_iter_actual"]}

    by_side = {s: {int(f): i for i, f in enumerate(r["frame_ids"])} for s, r in results.items()}
    for raw in raw_files:
        frame = int(raw.stem)
        raw_params = parse_mano_file(raw)
        refined = {}
        for side in ("right", "left"):
            if raw_params[side] is None:
                refined[side] = None
                continue
            if frame not in by_side.get(side, {}):
                raise RuntimeError(f"{side} frame {frame} was valid raw but not optimized")
            i, r = by_side[side][frame], results[side]
            T_wc = load_camera_to_world(seq_path, frame)
            trans_cam = T_wc[:3, :3].T @ (r["trans_world_final"][i] - T_wc[:3, 3])
            refined[side] = (trans_cam, np.r_[r["rot"][i], r["pose"][i]], r["beta"][i])
        write_refined_frame(staging / f"{raw.stem}.npz", frame, refined)
        read_refined_frame(staging / f"{raw.stem}.npz")

    roundtrip = {}
    for side, r in results.items():
        picks = sorted({0, len(r["frame_ids"]) // 2, len(r["frame_ids"]) - 1})
        roundtrip[side] = world_roundtrip_error(seq_path, r, side, picks, mano_root)
        if max(roundtrip[side].values()) >= ROUNDTRIP_TOL_M:
            raise RuntimeError(f"{side} MANO world round trip failed: {roundtrip[side]}")

    metadata = {**cfg_payload, "config_hash": cfg_hash, "seq_path": str(seq_path),
                "frame_stems": [p.stem for p in raw_files], "available_sides": list(results),
                "coordinate_convention": "trans_cam=R_world_cam.T@(t_hand_world-t_world_cam)",
                "roundtrip_max_abs_vertex_error_m": roundtrip,
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "completed_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(staging / "metadata.json", metadata)
    atomic_json(staging / "timing_metrics.json", {"wall_seconds": time.time() - started, "sides": timings})
    atomic_json(staging / "complete.json", {"status": "complete", "config_hash": cfg_hash, "frames": len(raw_files)})
    if out.exists():  # keep any previous output rather than deleting it
        os.replace(out, out.with_name(out.name + f".incompatible.{int(time.time())}"))
    os.replace(staging, out)
    log.info("[DONE] %s", out)


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-path", type=Path, required=True, help="H2O cam4 sequence directory")
    p.add_argument("--object-root", type=Path, required=True, help="directory with one folder per H2O object")
    p.add_argument("--mano-root", type=Path, required=True, help="directory with MANO_LEFT.pkl / MANO_RIGHT.pkl")
    p.add_argument("--output-dir", type=Path, help="default: <seq-path>/mano_refined")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seq_path = args.seq_path.resolve()
    refine_sequence(seq_path, args.output_dir or seq_path / "mano_refined", args.object_root, args.mano_root,
                    args.device, RefineConfig(), args.force)


if __name__ == "__main__":
    main()
