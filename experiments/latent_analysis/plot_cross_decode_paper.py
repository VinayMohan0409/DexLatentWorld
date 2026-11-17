"""Paper heatmap of a cross-decode run (no GPU): PDF with embedded font, SVG with outlined text, 600-dpi PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from lawm.embodiments.registry import load_registry  # noqa: E402
from lawm.eval import cross_decode as cd  # noqa: E402

PAPER_NAMES = {"human": "Human", "ability": "Ability", "allegro": "Allegro", "inspire": "Inspire", "leap": "LEAP",
               "oymotion": "OYMotion", "shadow": "Shadow", "shadow_noforearm": "Shadow", "wuji": "Wuji", "xhand": "XHand",
               "svh": "SVH", "dg5f": "DG-5F", "dex3": "Dex3"}
WHICH = {"step1": "step1", "mean": "mean_over_steps"}
# Single-hue sequential ramp (light = low error, darker = high), steps 100..500 of a validated blue ramp: muted, print- and
# colour-blind-safe; the trained/held-out divider stays red.
BLUE_RAMP = ["#f2f7fd", "#e1edfc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec"]  # pale: near-white .. soft mid-blue


def main(argv=None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="cross-decode output folder (with totals.npz)")
    ap.add_argument("--which", choices=sorted(WHICH), default="step1")
    ap.add_argument("--font-file", type=Path, default=None, help=".ttf/.otf to use for all text (default: DejaVu Sans)")
    ap.add_argument("--width-in", type=float, default=3.25, help="figure width; 3.25 in = one column of most two-column venues")
    ap.add_argument("--cmap", default="blue", help="'blue' (the ramp above) or any matplotlib colormap name")
    ap.add_argument("--no-labels", action="store_true",
                    help="omit tick labels, axis titles and the colourbar title (keep cell values): to add them in another tool")
    ap.add_argument("--metric", default="mover_5mm")
    ap.add_argument("--no-line", action="store_true", help="omit the line between trained and held-out embodiments")
    args = ap.parse_args(argv)

    z = np.load(args.run / "totals.npz")
    names = [str(n) for n in z["names"]]
    registry = load_registry()
    trained = [registry[n].trained for n in names]
    order = [i for i in range(len(names)) if trained[i]] + [i for i in range(len(names)) if not trained[i]]
    M = cd.matrices(z["totals"], args.metric)[WHICH[args.which]][np.ix_(order, order)]
    labels = [PAPER_NAMES.get(names[i], names[i].title()) for i in order]
    n_seen = sum(trained)

    if args.font_file:
        font_manager.fontManager.addfont(str(args.font_file))
        family = font_manager.FontProperties(fname=str(args.font_file)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42, "svg.fonttype": "path", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
    E = len(order)
    fig, ax = plt.subplots(figsize=(args.width_in, args.width_in * 0.93))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP) if args.cmap == "blue" else plt.get_cmap(args.cmap)
    im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(E))
    ax.set_yticks(range(E))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for t, i in zip(ticks, order):
            if not trained[i]:
                t.set_color("#b30000")
    if not args.no_line and 0 < n_seen < E:
        for f in (ax.axhline, ax.axvline):
            f(n_seen - 0.5, color="#b30000", linewidth=0.8)
    ax.set_xlabel("Decode Target Embodiment", fontsize=7)
    ax.set_ylabel("Latent Source Embodiment", fontsize=7)
    if args.no_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    lo, hi = np.nanmin(M), np.nanmax(M)
    for r in range(E):
        for c in range(E):
            v = M[r, c]
            rgb = cmap((v - lo) / (hi - lo))[:3]
            dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5  # text colour by cell luminance
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=4.6, color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if not args.no_labels:
        cb.set_label("EPE (mm)", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    fig.tight_layout(pad=0.2)
    out = args.run / "paper"
    suffix = "_NO_LABELS_for_canva" if args.no_labels else ""
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"cross_decode_{args.which}{suffix}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"font: {family}; wrote {out}/cross_decode_{args.which}{suffix}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
