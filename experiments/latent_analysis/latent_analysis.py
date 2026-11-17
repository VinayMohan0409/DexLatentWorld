"""Embodiment information in the latent actions: `extract` (GPU) saves latents, `analyze` runs the embodiment probe and a UMAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAPER_NAMES = {"ability": "Ability", "allegro": "Allegro", "inspire": "Inspire", "leap": "LEAP", "oymotion": "OYMotion",
               "shadow_noforearm": "Shadow", "wuji": "Wuji", "xhand": "XHand", "svh": "SVH", "dg5f": "DG-5F", "dex3": "Dex3"}
TRAINED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # validated 8-hue order
HELDOUT_STYLE = [("#2b2b2b", "^"), ("#5c5c5c", "s"), ("#8a8a8a", "D")]  # >8 hues are not separable in a scatter: markers instead


def extract(args) -> None:
    """Latents and pooled frame-0 backbone features of every grid embodiment, one row per (window, embodiment)."""
    import torch
    from lawm.data.config import read_json, resolve_data_path
    from lawm.data.index import load_or_build_index
    from lawm.data.sequences import Sequence
    from lawm.eval import cross_decode as cd
    from lawm.sim import model_adapter
    from lawm.train.evaluation import compute_context, to_device
    from lawm.train.reproducibility import configure_determinism, evaluation_stream

    ev = read_json(args.config)
    cfg = read_json(REPO / ev["training_config"])
    gap, n_steps, points = cfg["window"]["gap"], cfg["window"]["n_steps"], cfg["window"]["points_per_entity"]
    ds_dir = resolve_data_path(ev["data_dir"])
    split = read_json(REPO / cfg["split_file"])[ev["split_key"]]
    names = ev["embodiments"]
    stored = Sequence(ds_dir / split[0], points).names
    index = load_or_build_index(args.out / "_cache", ev["split_key"], ds_dir, split, gap, n_steps, points)
    ok = index["valid_window"][:, [stored.index(n) for n in names]].all(axis=1)
    windows = [(str(s), int(f)) for s, f, v in zip(index["seq"], index["f"], ok) if v and int(f) % args.stride == 0]
    if args.exclude:  # top up an earlier extraction with `add` evenly spaced new windows
        seen = set().union(*({(str(s), int(f)) for s, f in zip(np.load(e)["sequence"], np.load(e)["frame"])} for e in args.exclude))
        windows = [w for w in windows if w not in seen]
        windows = [windows[i] for i in np.linspace(0, len(windows) - 1, args.add).round().astype(int)]
    seqs = {s: Sequence(ds_dir / s, points) for s in sorted({s for s, _ in windows})}
    import time
    t0 = time.time()
    for s, seq in seqs.items():  # every needed frame read once, in parallel (neighbouring windows share most of their frames)
        seq.preload([f + k * gap for s2, f in windows if s2 == s for k in range(n_steps + 1)])
    print(f"preloaded {sum(len(q._frames) for q in seqs.values())} frames in {time.time() - t0:.0f} s", flush=True)
    batch_size = args.batch_size or ev["batch_size"]
    if args.one_hand:  # every window encoded for ONE embodiment only (balanced random assignment), so no motion appears once per hand
        hands = np.resize(np.arange(len(names)), len(windows))
        np.random.default_rng(args.seed).shuffle(hands)
        jobs = [(name, [w for w, h in zip(windows, hands) if h == i]) for i, name in enumerate(names)]
    else:
        jobs = [(None, windows)]
    device = torch.device("cuda")
    configure_determinism(ev["determinism"])
    amp = ev["amp"] if args.amp is None else args.amp
    model = model_adapter.load_model(REPO / ev["checkpoint"], REPO / ev["run_manifest"], device)
    Z, F, rows = [], [], []
    groups = [(name, group) for only, job_windows in jobs for group in cd.batches(job_windows, batch_size)
              for name in (names if only is None else [only])]
    with torch.no_grad():
        for i, (name, group) in enumerate(groups):
            if i % 50 == 0:
                print(f"[{time.time() - t0:.0f} s] batch {i}/{len(groups)}", flush=True)
            batch = to_device(cd.window_batch(seqs, group, stored.index(name), gap, n_steps), device)
            with evaluation_stream(cd.encode_seed(ev["eval_stream_seed"], group, name), device), compute_context(device, amp):
                chain_0, z = model.encode_window(batch["coords"], batch["offset"])
            bounds = [0] + chain_0[0].offset.tolist()
            F.append(torch.stack([chain_0[0].feat[a:b].float().mean(0) for a, b in zip(bounds[:-1], bounds[1:])]).cpu().numpy())
            Z.append(z.float().permute(1, 0, 2, 3).cpu().numpy())
            rows += [(name, s, f) for s, f in group]
    emb, seq, frame = zip(*rows)
    np.savez(args.out / args.latents[0], z=np.concatenate(Z), frame_feat=np.concatenate(F), embodiment=np.array(emb),
             sequence=np.array(seq), frame=np.array(frame), names=np.array(names))
    print(f"wrote {args.out / args.latents[0]} ({len(windows)} windows)")


def probe(X, y, groups, test_sequences: int = 4, seed: int = 0) -> dict:
    """Embodiment classification accuracy (linear and random forest), tested on sequences unseen in training."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    test = np.isin(groups, np.random.default_rng(seed).choice(sorted(set(groups.tolist())), size=test_sequences, replace=False))
    out = {"chance": 1.0 / len(np.unique(y)), "train": int((~test).sum()), "test": int(test.sum())}
    for name, clf in (("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                      ("random_forest", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))):
        out[name] = float((clf.fit(X[~test], y[~test]).predict(X[test]) == y[test]).mean())
    return out


def analyze(args) -> None:
    """Probe the latents (frame features as a positive control) and plot a UMAP with one point per window and embodiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from matplotlib import font_manager
    from sklearn.preprocessing import StandardScaler
    from lawm.embodiments.registry import load_registry

    parts = [np.load(args.out / f) for f in args.latents]
    cat = lambda k: np.concatenate([d[k] for d in parts])  # noqa: E731
    Z, emb, groups = cat("z"), cat("embodiment"), cat("sequence")
    Z = Z.reshape(len(Z), -1)  # all steps of a window: one point per (window, embodiment)
    names = [str(n) for n in parts[0]["names"]]
    if not args.no_probe:
        result = {"points": len(Z), "latent_actions": probe(Z, emb, groups), "control_frame_features": probe(cat("frame_feat"), emb, groups)}
        (args.out / "embodiment_probe.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))

    P = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(Z))
    family = "DejaVu Sans"
    if args.font_file:
        font_manager.fontManager.addfont(str(args.font_file))
        family = font_manager.FontProperties(fname=str(args.font_file)).get_name()
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42, "svg.fonttype": "path", "axes.linewidth": 0.5})
    trained = [n for n in names if load_registry()[n].trained]
    style = {**dict(zip(trained, [(c, "o") for c in TRAINED_COLOURS])), **dict(zip([n for n in names if n not in trained], HELDOUT_STYLE))}
    fig, ax = plt.subplots(figsize=(args.width_in, args.width_in * 0.95))
    for chunk in np.array_split(np.random.default_rng(1).permutation(len(P)), 40):  # random draw order: no embodiment hides the others
        for n in names:
            m = chunk[emb[chunk] == n]
            ax.scatter(P[m, 0], P[m, 1], s=3.0, c=style[n][0], marker=style[n][1], linewidths=0, alpha=0.75, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker=style[n][1], color=style[n][0], markersize=3.5, label=PAPER_NAMES.get(n, n)) for n in names],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=6, frameon=False, handletextpad=0.1, columnspacing=0.9)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"latents_umap.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}/latents_umap.* ({len(P)} points)")
    if args.three_d:
        umap_3d(args, Z, emb, names, style, plt)


def umap_3d(args, Z, emb, names, style, plt) -> None:
    """The same latents embedded in 3 UMAP dimensions, one static view (same colours / markers / draw order as the 2D figure)."""
    import umap
    from sklearn.preprocessing import StandardScaler

    P = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(Z))
    np.savez(args.out / "latents_umap3d.npz", P=P, embodiment=emb)
    fig = plt.figure(figsize=(args.width_in, args.width_in))
    ax = fig.add_subplot(projection="3d")
    for chunk in np.array_split(np.random.default_rng(1).permutation(len(P)), 40):
        for n in names:
            m = chunk[emb[chunk] == n]
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=2.0, c=style[n][0], marker=style[n][1], linewidths=0, alpha=0.7, depthshade=False, rasterized=True)
    ax.view_init(elev=20, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel("UMAP 1", fontsize=7, labelpad=-12); ax.set_ylabel("UMAP 2", fontsize=7, labelpad=-12); ax.set_zlabel("UMAP 3", fontsize=7, labelpad=-12)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker=style[n][1], color=style[n][0], markersize=3.5, label=PAPER_NAMES.get(n, n)) for n in names],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=6, frameon=False, handletextpad=0.1, columnspacing=0.9)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"latents_umap3d.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {args.out}/latents_umap3d.* ({len(P)} points)")


def main(argv=None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("extract", "analyze"))
    ap.add_argument("--out", type=Path, required=True, help="output folder (latents, probe JSON, figure)")
    ap.add_argument("--config", type=Path, default=REPO / "configs/eval/cross_decode_val.json")
    ap.add_argument("--latents", nargs="+", default=["latents.npz"], help="extract: file to write; analyze: files to combine")
    ap.add_argument("--stride", type=int, default=12, help="extract: every stride-th start frame of every validation sequence")
    ap.add_argument("--exclude", nargs="*", default=[], type=Path, help="extract: earlier latents files whose windows to skip")
    ap.add_argument("--add", type=int, default=0, help="extract with --exclude: number of new windows")
    ap.add_argument("--one-hand", action="store_true", help="extract: each window encoded for one embodiment only (balanced random assignment, --seed)")
    ap.add_argument("--seed", type=int, default=0, help="extract --one-hand: seed of the window-to-embodiment assignment")
    ap.add_argument("--three-d", action="store_true", help="analyze: also a 3D UMAP figure (latents_umap3d.*)")
    ap.add_argument("--no-probe", action="store_true", help="analyze: UMAP figures only, no embodiment probe")
    ap.add_argument("--batch-size", type=int, default=None, help="extract: windows per encode (default: the config's batch_size)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--n-neighbors", type=int, default=100, help="UMAP neighbourhood (each window has one near-identical latent per embodiment)")
    ap.add_argument("--font-file", type=Path, default=None)
    ap.add_argument("--width-in", type=float, default=3.25)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    extract(args) if args.command == "extract" else analyze(args)


if __name__ == "__main__":
    main()
