"""Cross-decode heatmap of a Stage 2 checkpoint: python experiments/latent_analysis/cross_decode.py --config configs/eval/cross_decode_val.json --out <dir> [--embodiments a b ...] [--max-windows N]; needs LAWM_DATA_ROOT and the GPU."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from lawm.data.config import read_json, resolve_data_path
from lawm.data.index import fingerprint, load_or_build_index
from lawm.data.sequences import Sequence
from lawm.embodiments.registry import load_registry
from lawm.eval import cross_decode as cd
from lawm.model.metrics import NAMES
from lawm.sim import model_adapter
from lawm.train.reproducibility import configure_determinism, describe_environment, evaluation_stream, file_sha256, write_manifest

REPO = Path(__file__).resolve().parents[2]


def repo_path(p) -> Path:
    """A config path: absolute as written, else relative to the repository."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p

# File names that cannot be mistaken for each other: step 1 alone vs the mean over the window's steps.
REPORT_NAMES = {"step1": "STEP1_ONLY", "mean_over_steps": "MEAN_OVER_ALL_STEPS"}


def write_reports(out: Path, totals, names, registry, ev: dict, gap: int, n_windows: int) -> dict:
    """Matrices (CSV), heatmaps and summary.json from the summed totals; labels and trained flags come from the registry."""
    labels = [registry[n].display for n in names]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two embodiments in the grid share a label: {labels}")
    trained = [registry[n].trained for n in names]
    mats = cd.matrices(totals, ev["metric"])
    summary = {}
    for key in [k for k in mats if k != "n_step1"]:
        M = mats[key]
        summary[key] = cd.regime_summary(M, trained)
        with open(out / f"{ev['metric']}_{REPORT_NAMES[key] if key in REPORT_NAMES else key + '_cumulative'}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["latents_from \\ decode_into"] + labels)
            w.writerows([[labels[i]] + [f"{v:.4f}" for v in M[i]] for i in range(len(labels))])
    label = f"{ev['metric']} error (mm)"
    title = f"{Path(ev['checkpoint']).parent.name}: {n_windows} val windows"
    n_steps = len([k for k in mats if k.startswith("step")])
    cd.plot_heatmap(mats["step1"], labels, trained, out / f"heatmap_{REPORT_NAMES['step1']}.png",
                    title + f"\nSTEP 1 ONLY: one latent, {gap} frames ahead", label)
    cd.plot_heatmap(mats["mean_over_steps"], labels, trained, out / f"heatmap_{REPORT_NAMES['mean_over_steps']}.png",
                    title + f"\nMEAN OVER ALL {n_steps} STEPS ({gap} to {gap * n_steps} frames ahead)", label)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    """Select the windows, run every (source, target) pair, write matrices, heatmaps and a manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embodiments", nargs="+", help="override the config's list (plot order)")
    ap.add_argument("--max-windows", type=int, default=0, help="timing/smoke runs only: keep the first N selected windows")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="override the config's bf16 autocast; --no-amp on GPUs without bf16 attention kernels (e.g. Turing: RTX 20xx)")
    ap.add_argument("--replot", action="store_true", help="only redraw the reports from <out>/totals.npz (no GPU)")
    ap.add_argument("--reuse", type=Path, default=None,
                    help="a previous run on the same windows and settings: its cells are copied, only pairs involving new "
                         "embodiments are computed (plus the reused diagonal, as a consistency check)")
    args = ap.parse_args(argv)
    t0 = time.time()
    ev = read_json(args.config)
    if args.replot:
        z = np.load(Path(args.out) / "totals.npz")
        cfg = read_json(repo_path(ev["training_config"]))
        n_windows = sum(1 for _ in open(Path(args.out) / "windows.csv")) - 1
        print(json.dumps(write_reports(Path(args.out), z["totals"], [str(n) for n in z["names"]], load_registry(), ev,
                                       cfg["window"]["gap"], n_windows)["mean_over_steps"], indent=1))
        return
    if args.amp is not None:
        ev["amp"] = args.amp  # recorded in the manifest's eval_config
    names = args.embodiments or ev["embodiments"]
    registry = load_registry()
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"not in configs/embodiments.json: {unknown}")
    trained = [registry[n].trained for n in names]

    cfg = read_json(repo_path(ev["training_config"]))
    window = cfg["window"]
    gap, n_steps, points = window["gap"], window["n_steps"], window.get("points_per_entity", 0)
    split = read_json(repo_path(cfg["split_file"]))[ev["split_key"]]
    ds_dir = resolve_data_path(ev["data_dir"])
    stored = {tuple(Sequence(ds_dir / s, points).names) for s in split}
    if len(stored) != 1:
        raise ValueError(f"sequences store different embodiment lists: {stored}")
    stored = list(stored.pop())
    missing = [n for n in names if n not in stored]
    if missing:
        raise ValueError(f"{ds_dir} does not contain {missing}; it has {stored}")
    emb_idx = [stored.index(n) for n in names]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = load_or_build_index(out / "_cache", ev["split_key"], ds_dir, split, gap, n_steps, points)
    windows = cd.select_windows(index, emb_idx, ev["windows_per_sequence"], ev["seed"])
    if args.max_windows:
        windows = windows[: args.max_windows]
    with open(out / "windows.csv", "w", newline="") as f:
        csv.writer(f).writerows([("sequence", "start_frame")] + windows)

    device = torch.device("cuda")
    numerics = configure_determinism(ev["determinism"])
    model = model_adapter.load_model(repo_path(ev["checkpoint"]), repo_path(ev["run_manifest"]), device)
    cells, reused, reuse_info = None, None, None
    if args.reuse:
        old = np.load(args.reuse / "totals.npz")
        old_names = [str(n) for n in old["names"]]
        old_windows = list(csv.reader(open(args.reuse / "windows.csv")))[1:]
        old_ev = json.loads((args.reuse / "run_manifest.json").read_text())["eval_config"]
        same = {k: (old_ev.get(k), ev.get(k)) for k in ("checkpoint", "training_config", "split_key", "windows_per_sequence",
                                                         "seed", "batch_size", "amp", "metric") if old_ev.get(k) != ev.get(k)}
        if same:
            raise ValueError(f"--reuse: settings differ from {args.reuse}: {same}")
        if [(s, int(f)) for s, f in old_windows] != windows:
            raise ValueError(f"--reuse: {args.reuse} used different windows (validity of the new embodiments differs)")
        pos = {n: i for i, n in enumerate(old_names)}
        reused = np.array([[a in pos and b in pos for b in names] for a in names])
        cells = ~reused | np.eye(len(names), dtype=bool)  # recompute reused diagonal cells only to compare
        reuse_info = dict(source=str(args.reuse), reused_cells=int(reused.sum()), computed_cells=int(cells.sum()))
    with evaluation_stream(ev["eval_stream_seed"], device):
        totals = cd.cross_decode(model, ds_dir, windows, emb_idx, gap, points, ev["batch_size"], device, ev["amp"],
                                 cells=cells, names=names, seed=ev["eval_stream_seed"])
    if reused is not None:
        m = NAMES.index(ev["metric"])
        check = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if reused[i, j]:
                    prev = old["totals"][pos[a], pos[b]]
                    if i == j:
                        new_mm = totals[i, i, :, m, 0].sum() / totals[i, i, :, m, 1].sum()
                        old_mm = prev[:, m, 0].sum() / prev[:, m, 1].sum()
                        check[a] = dict(stored_mm=round(float(old_mm), 3), recomputed_mm=round(float(new_mm), 3))
                    totals[i, j] = prev
        reuse_info["diagonal_check"] = check
        print("reused-diagonal check (pooled over steps):", json.dumps(check))
    np.savez(out / "totals.npz", totals=totals, names=np.array(names), trained=np.array(trained), metric_names=np.array(NAMES))

    summary = write_reports(out, totals, names, registry, ev, gap, len(windows))
    write_manifest(out, command=["python", "experiments/latent_analysis/cross_decode.py", *(argv or __import__("sys").argv[1:])], eval_config=ev,
                   embodiments=dict(zip(names, trained)), windows=len(windows), checkpoint_sha256=file_sha256(repo_path(ev["checkpoint"])),
                   dataset=str(ds_dir), dataset_fingerprint=fingerprint(ds_dir, split), numerics=numerics, reuse=reuse_info,
                   environment=describe_environment(device, REPO), seconds=round(time.time() - t0, 1))
    print(json.dumps(summary["mean_over_steps"], indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
