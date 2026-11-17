"""Compare raw and refined MANO hands against the manipulated object, per sequence and hand: `python -m lawm.preprocessing.contact_refinement.evaluation.evaluate --h2o-root <H2O> --object-root <meshes> --mano-root <mano models> --out-dir <dir>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from lawm.preprocessing.contact_refinement.evaluation.metrics import (
    ACTIVE_FRAME_THRESHOLD_M, aggregate_chamfer, contact_chamfer_per_frame, contact_region_mask,
    object_vertex_normals, penetration_stats)
from lawm.preprocessing.contact_refinement.mano_refined import read_refined_frame
from lawm.preprocessing.contact_refinement.optimizer import make_mano_layer, mano_forward
from lawm.preprocessing.contact_refinement.sequence import SequenceData, load_sequence, translation_matrix
from lawm.preprocessing.h2o_readers import (
    H2O_OBJECT_NAMES, list_sequences, load_camera_to_world, load_object_pose)


def load_refined_for_side(seq_path: Path, side: str, frame_ids: np.ndarray):
    """Load refined MANO for `frame_ids`; returns (valid mask, rot, pose, beta, camera-to-world placement), frame-aligned."""
    T = len(frame_ids)
    valid = np.zeros(T, dtype=bool)
    rot = np.zeros((T, 3), np.float32)
    pose = np.zeros((T, 45), np.float32)
    beta = np.zeros((T, 10), np.float32)
    mTc_world = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for i, fid in enumerate(frame_ids):
        npz_path = seq_path / "mano_refined" / f"{int(fid):06d}.npz"
        if not npz_path.exists():
            continue
        parsed = read_refined_frame(npz_path)
        if parsed[side] is None:
            continue
        trans_cam, pose48, beta10 = parsed[side]
        rot[i], pose[i], beta[i] = pose48[:3], pose48[3:], beta10
        mTc_world[i] = (load_camera_to_world(seq_path, int(fid)) @ translation_matrix(trans_cam)).astype(np.float32)
        valid[i] = True
    return valid, rot, pose, beta, mTc_world


def forward_hand_verts(side: str, rot, pose, beta, mTc_world, mano_root: Path, device: str) -> np.ndarray:
    """World-frame MANO vertices (T, 778, 3) for the given parameters."""
    model = make_mano_layer(mano_root, side, device)
    with torch.no_grad():
        verts = mano_forward(
            model,
            torch.tensor(rot, dtype=torch.float32, device=device),
            torch.tensor(pose, dtype=torch.float32, device=device),
            torch.tensor(beta, dtype=torch.float32, device=device),
            torch.tensor(mTc_world, dtype=torch.float32, device=device),
        )
    return verts.detach().cpu().numpy()


def evaluate_sequence_side(seq_path: Path, side: str, seq: SequenceData, mano_root: Path, device: str,
                           contact_mask: np.ndarray):
    """Evaluate one hand of one sequence; None if the hand has no refined frames."""
    valid, r_rot, r_pose, r_beta, r_mTc = load_refined_for_side(seq_path, side, seq.frame_ids)
    if valid.sum() == 0:
        return None

    raw_verts = forward_hand_verts(side, seq.hand_rot[valid], seq.hand_pose[valid],
                                   np.tile(seq.hand_beta, (int(valid.sum()), 1)), seq.hand_mTc_world[valid],
                                   mano_root, device)
    refined_verts = forward_hand_verts(side, r_rot[valid], r_pose[valid], r_beta[valid], r_mTc[valid],
                                       mano_root, device)
    obj_verts, frame_ids = seq.obj_verts[valid], seq.frame_ids[valid]
    obj_normals = object_vertex_normals(obj_verts, seq.obj_faces)

    raw_chamfer = contact_chamfer_per_frame(raw_verts, obj_verts, contact_mask, device)
    refined_chamfer = contact_chamfer_per_frame(refined_verts, obj_verts, contact_mask, device)

    # "Active" frames come from the RAW hand only, so raw and refined are compared over the same frames: the
    # optimiser only corrects near-object frames, and averaging over the idle majority would hide the improvement.
    active_mask = raw_chamfer["h2o_mean"] < ACTIVE_FRAME_THRESHOLD_M

    return {
        "n_frames": int(valid.sum()),
        "n_active_frames": int(active_mask.sum()),
        "raw": {
            "penetration": penetration_stats(raw_verts, obj_verts, obj_normals, frame_ids, device),
            "chamfer": aggregate_chamfer(raw_chamfer),
            "chamfer_active_frames_only": aggregate_chamfer(raw_chamfer, mask=active_mask),
        },
        "refined": {
            "penetration": penetration_stats(refined_verts, obj_verts, obj_normals, frame_ids, device),
            "chamfer": aggregate_chamfer(refined_chamfer),
            "chamfer_active_frames_only": aggregate_chamfer(refined_chamfer, mask=active_mask),
        },
    }


def evaluate_sequence(seq_path: Path, manifest_index: int, h2o_root: Path, object_root: Path, mano_root: Path,
                      device: str, contact_masks: dict) -> dict:
    """Evaluate both hands of one sequence and return the per-sequence result dict."""
    seq_by_side = load_sequence(seq_path, object_root)
    first = next(iter(seq_by_side.values()))
    _, obj_id = load_object_pose(seq_path, int(first.frame_ids[0]))
    object_name = H2O_OBJECT_NAMES.get(obj_id, f"unknown_{obj_id}")
    subject, session_activity, take, _ = seq_path.relative_to(h2o_root).parts

    out = {"manifest_index": manifest_index, "seq_path": str(seq_path), "subject": subject,
           "session_activity": session_activity, "take": take, "object_name": object_name}
    for side, seq in seq_by_side.items():
        result = evaluate_sequence_side(seq_path, side, seq, mano_root, device, contact_masks[side])
        if result is not None:
            result["object_name"] = object_name
            out[side] = result
    return out


def main() -> None:
    """Command-line entry point: evaluate all (or selected) sequences and write one JSON per sequence."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2o-root", type=Path, required=True)
    ap.add_argument("--object-root", type=Path, required=True)
    ap.add_argument("--mano-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-indices", type=int, nargs="+", help="1-based sequence positions; default all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks = {side: contact_region_mask(side, args.mano_root) for side in ("right", "left")}
    for index, seq_path in list_sequences(args.h2o_root):
        if args.seq_indices and index not in args.seq_indices:
            continue
        result = evaluate_sequence(seq_path, index, args.h2o_root, args.object_root, args.mano_root, args.device, masks)
        name = f"{result['subject']}_{result['session_activity']}_{result['take']}_cam4.json"
        (args.out_dir / name).write_text(json.dumps(result, indent=2))
        print(f"[{index}] {name}", flush=True)


if __name__ == "__main__":
    main()
