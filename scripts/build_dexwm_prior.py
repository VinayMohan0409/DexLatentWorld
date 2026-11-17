"""Build the latent prior of a Stage 2 checkpoint for the DexWM planner: the mean and spread of the latents its encoder gives evenly spaced training windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lawm.data.config import data_root, dataset_dir, read_json
from lawm.data.dataset import WindowDataset
from lawm.data.index import load_or_build_index
from lawm.sim import model_adapter

REPO = Path(__file__).resolve().parents[1]


def main():
    """python scripts/build_dexwm_prior.py --checkpoint best.pt --manifest run_manifest.json --out prior.npz [--config <training config>] [--windows 400]; needs the GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="run_manifest.json of the training run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(REPO / "configs/h2o_50pct.json"))
    parser.add_argument("--runs-root", default=None, help="where the window index of the training run is cached (default: <data root>/../runs)")
    parser.add_argument("--windows", type=int, default=400)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = read_json(cfg_path)
    split, window = read_json(cfg_path.parent.parent / cfg["split_file"]), cfg["window"]
    gap, n_steps, points, voxel = window["gap"], window["n_steps"], window.get("points_per_entity", 0), window.get("voxel_size", 0.0)
    cache_dir = (Path(args.runs_root) if args.runs_root else data_root().parent / "runs") / "_cache"
    ds_dir = dataset_dir(cfg)
    index = load_or_build_index(cache_dir, "train", ds_dir, split["train"], gap, n_steps, points)
    dataset = WindowDataset(ds_dir, index, gap, n_steps, m=1, train=False, points_per_entity=points, voxel_size=voxel)
    rows = np.linspace(0, len(dataset) - 1, min(args.windows, len(dataset))).astype(int)
    model = model_adapter.load_model(args.checkpoint, args.manifest, torch.device("cuda"))
    prior = model_adapter.build_prior(model, (dataset[int(i)]["samples"][0]["coords"].numpy() for i in rows))
    model_adapter.save_prior(args.out, prior, args.checkpoint, len(rows))
    print(f"prior from {len(rows)} of {len(dataset)} training windows -> {args.out}; mean |z| {np.abs(prior[0]).mean():.3f}, spread mean {prior[1].mean():.3f} range [{prior[1].min():.3f}, {prior[1].max():.3f}]")


if __name__ == "__main__":
    main()
