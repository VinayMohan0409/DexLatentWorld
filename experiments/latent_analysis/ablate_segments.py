"""Latent ablation of adaln_s0_k4 with errors split by entity, recorded per window."""
# Same protocol as the original final ablation (trainer.py, end of main): ablation_rows(val_ds, 4000, 8), eval stream seed 10000,
# donor = final_rows[len // 2], batches of 8 consecutive rows (shuffled_across = roll by 1 inside a batch). Conditions:
# correct, zero, shuffled_across, fixed_donor, plus no_motion (predicted displacement = 0; needs no decoder).
#
# Saved (npz): stats[window, condition, step, segment, threshold, (sum_mm, count)] float64, per window, so any aggregation
# (pooling, per-step mean, bootstrap, per sequence) can be redone without the GPU. Segments: 0 hand, 1 object, 2 table, 3 distractor.
# Thresholds (on the ground-truth displacement norm): 0 = >=5 mm, 1 = >=2 mm, 2 = all points.
#
# Run:  PYTHONPATH=<repo>/src LAWM_DATA_ROOT=<data> python ablate_segments.py --out <dir> [--max-windows N] [--no-amp]
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lawm.data.config import read_json, resolve_data_path
from lawm.data.dataset import WindowDataset, collate
from lawm.data.index import fingerprint, load_or_build_index
from lawm.sim import model_adapter
from lawm.train.evaluation import apply_condition, compute_context, donor_latents, to_device, ablation_rows
from lawm.train.reproducibility import configure_determinism, describe_environment, evaluation_stream, file_sha256

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("correct", "zero", "shuffled_across", "fixed_donor", "no_motion")
THRESHOLDS = (0.005, 0.002, 0.0)  # metres, ground-truth displacement norm >= t
N_SEG = 4


def window_stats(err_mm, moved, segment, bounds):
    """(B, N_SEG, len(THRESHOLDS), 2) summed error and count per window, from per-point error (mm), gt norm (m) and segments."""
    out = np.zeros((len(bounds) - 1, N_SEG, len(THRESHOLDS), 2))
    for w in range(len(bounds) - 1):
        a, b = bounds[w], bounds[w + 1]
        e, m, s = err_mm[a:b], moved[a:b], segment[a:b]
        for si in range(N_SEG):
            for ti, t in enumerate(THRESHOLDS):
                sel = (s == si) & (m >= t)
                out[w, si, ti] = (e[sel].sum(), sel.sum())
    return out


def main():
    """Run the latent ablation over the selected validation windows and save the per-window error statistics (stats.npz)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=4000)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test: use only the first N of the selected rows")
    ap.add_argument("--batch-size", type=int, default=8, help="must stay 8 to reproduce the shuffled_across assignment")
    ap.add_argument("--workers", type=int, default=8, help="data-loading processes (windows are assembled on the CPU)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="--no-amp on Turing (no bf16 attention kernels)")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = REPO / "results/adaln_s0_k4/run_manifest.json"
    cfg = read_json(REPO / "configs/h2o_50pct.json")
    window = cfg["window"]
    gap, n_steps, points = window["gap"], window["n_steps"], window["points_per_entity"]
    split = read_json(REPO / cfg["split_file"])
    ds_dir = resolve_data_path(cfg["data_dir"])
    val_idx = load_or_build_index(out / "_cache", "val", ds_dir, split["val"], gap, n_steps, points)
    val_ds = WindowDataset(ds_dir, val_idx, gap, n_steps, m=1, train=False, points_per_entity=points, voxel_size=0.0)
    rows = ablation_rows(val_ds, args.windows, args.batch_size)
    donor_row = rows[len(rows) // 2]  # chosen from the full list, as in the original
    if args.max_windows:
        rows = rows[: args.max_windows]
    device = torch.device("cuda")
    numerics = configure_determinism("strict")
    model = model_adapter.load_model(REPO / "checkpoints/adaln_s0_k4/best.pt", manifest, device)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_ds, rows), batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate, num_workers=args.workers, pin_memory=True, prefetch_factor=4 if args.workers else None)
    stats = np.zeros((len(rows), len(CONDITIONS), n_steps, N_SEG, len(THRESHOLDS), 2))
    meta = dict(row=[], seq=[], frame=[], emb=[])
    w0 = 0
    with torch.no_grad(), evaluation_stream(10_000, device):
        fixed = donor_latents(model, val_ds, donor_row, device, args.amp)
        for bi, batch in enumerate(loader):
            seqs = batch["seq"]
            meta["row"] += batch["transition_idx"].tolist(); meta["seq"] += seqs; meta["frame"] += batch["frame"]; meta["emb"] += batch["emb"]
            batch = to_device(batch, device)
            coords, offset, segment = batch["coords"], batch["offset"], batch["segment"]
            bounds = [0] + offset.tolist()
            seg_np = segment.cpu().numpy()
            with compute_context(device, args.amp):
                chain_0, zs = model.encode_window(coords, offset)
                for ci, c in enumerate(CONDITIONS):
                    if c == "no_motion":
                        preds = [torch.zeros_like(coords[0]).unsqueeze(1)] * n_steps
                    else:
                        changed = apply_condition(c, zs, seqs, fixed)
                        if changed is None:
                            raise RuntimeError(f"{c} cannot be formed for a batch of {len(seqs)}")  # a last batch of 1
                        preds = [o["delta"].float() for o in model.decode_all_steps(chain_0, changed)]
                    for k in range(1, n_steps + 1):
                        gt = (coords[k] - coords[0]).unsqueeze(1)
                        err = ((preds[k - 1] - gt).norm(dim=-1).reshape(-1) * 1000).double().cpu().numpy()
                        moved = gt.norm(dim=-1).reshape(-1).cpu().numpy()
                        stats[w0:w0 + len(seqs), ci, k - 1] = window_stats(err, moved, seg_np, bounds)
            w0 += len(seqs)
            if bi % 25 == 0:
                print(f"batch {bi + 1}/{len(loader)}  {time.time() - t0:.0f} s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    assert w0 == len(rows)
    np.savez_compressed(out / "stats.npz", stats=stats, conditions=np.array(CONDITIONS), segments=np.array(["hand", "object", "table", "distractor"]),
                        thresholds_m=np.array(THRESHOLDS), row=np.array(meta["row"]), seq=np.array(meta["seq"]), frame=np.array(meta["frame"]),
                        emb=np.array(meta["emb"]), donor_row=donor_row)
    (out / "run_info.json").write_text(json.dumps(dict(
        windows=len(rows), donor_row=int(donor_row), amp=args.amp, batch_size=args.batch_size, eval_stream_seed=10_000, numerics=numerics,
        checkpoint_sha256=file_sha256(REPO / "checkpoints/adaln_s0_k4/best.pt"), data_dir=str(ds_dir), val_fingerprint=fingerprint(ds_dir, split["val"]),
        environment=describe_environment(device, REPO), seconds=round(time.time() - t0, 1),
        note="stats[window, condition, step, segment, threshold, (sum_mm, count)]; segments hand,object,table,distractor; thresholds >=5mm,>=2mm,all"),
        indent=1, default=str) + "\n")
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
