"""Summarise per-sequence evaluation JSONs into a CSV and a raw-vs-refined bar chart: `python -m lawm.preprocessing.contact_refinement.evaluation.report --in-dir <dir>`."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

FIELDS = [
    ("penetration", "mean_frame_max_penetration_m"),
    ("penetration", "p95_frame_max_penetration_m"),
    ("penetration", "max_penetration_m"),
    ("penetration", "pct_penetrating_vertices"),
    ("penetration", "pct_vertices_penetrating_over_allowance"),  # absent in older evaluation JSONs
    ("chamfer", "mean_contact_to_surface_m"),
    ("chamfer", "p95_contact_to_surface_m"),
    ("chamfer_active_frames_only", "mean_contact_to_surface_m"),
    ("chamfer_active_frames_only", "p95_contact_to_surface_m"),
]

# (title, raw column, refined column, scale, unit)
HEADLINE_METRICS = [
    ("P95 Penetration Depth", "raw_penetration_p95_frame_max_penetration_m",
     "refined_penetration_p95_frame_max_penetration_m", 1000, "mm"),
    ("Penetrating Vertices", "raw_penetration_pct_penetrating_vertices",
     "refined_penetration_pct_penetrating_vertices", 100, "%"),
    ("Active-Frame Chamfer", "raw_chamfer_active_frames_only_mean_contact_to_surface_m",
     "refined_chamfer_active_frames_only_mean_contact_to_surface_m", 1000, "mm"),
]
COLOR_RAW, COLOR_REFINED = "#A7BBD3", "#D9AE8E"  # pale dusty blue / terracotta


def build_rows(in_dir: Path) -> list:
    """One row per (sequence, hand) from the per-sequence JSONs, with raw_* and refined_* metric columns."""
    rows = []
    for path in sorted(in_dir.glob("*.json")):
        d = json.loads(path.read_text())
        for side in ("right", "left"):
            if side not in d:
                continue
            r = d[side]
            row = {
                "seq_id": path.stem, "manifest_index": d["manifest_index"], "subject": d["subject"],
                "session_activity": d["session_activity"], "take": d["take"], "side": side,
                "object_name": r.get("object_name", d.get("object_name", "")), "n_frames": r["n_frames"],
                "n_active_frames": r.get("n_active_frames", ""),
            }
            for group in ("raw", "refined"):
                for section, key in FIELDS:
                    row[f"{group}_{section}_{key}"] = r[group][section].get(key)
            rows.append(row)
    return rows


def write_summary_csv(rows: list, out_csv: Path) -> None:
    """Write the rows to a CSV file."""
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_raw_vs_refined(rows: list, out_png: Path) -> None:
    """Bar chart of the median raw and refined value of each headline metric over all hand-tracks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 12, "font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, len(HEADLINE_METRICS), figsize=(11.5, 4.6))
    fig.suptitle(f"Raw vs. Refined MANO: Hand-Object Fit\nMedian Over {len(rows)} Hand-Tracks", fontsize=12.5, y=1.04)
    x, w = np.arange(1), 0.32
    for ax, (label, raw_col, ref_col, scale, unit) in zip(axes, HEADLINE_METRICS):
        raw_vals = [float(r[raw_col]) * scale for r in rows if r[raw_col] not in ("", "None", None)]
        ref_vals = [float(r[ref_col]) * scale for r in rows if r[ref_col] not in ("", "None", None)]
        raw_med, ref_med = st.median(raw_vals), st.median(ref_vals)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        b1 = ax.bar(x - w / 2, [raw_med], width=w, color=COLOR_RAW, label="Raw", zorder=3)
        b2 = ax.bar(x + w / 2, [ref_med], width=w, color=COLOR_REFINED, label="Refined", zorder=3)
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=11, fontweight="bold", zorder=4)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_ylim(0, max(raw_med, ref_med) * 1.28)
        ax.set_title(f"{label} ({unit})", fontsize=11.5)
        ax.tick_params(axis="y", labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--out-png", type=Path)
    args = ap.parse_args()
    rows = build_rows(args.in_dir)
    if not rows:
        raise SystemExit(f"No per-sequence JSONs found in {args.in_dir}")
    write_summary_csv(rows, args.out_csv or args.in_dir / "summary.csv")
    plot_raw_vs_refined(rows, args.out_png or args.in_dir / "raw_vs_refined_bar_chart.png")
    for group in ("raw", "refined"):
        for section, key in FIELDS:
            vals = [r[f"{group}_{section}_{key}"] for r in rows if r[f"{group}_{section}_{key}"] is not None]
            if vals:
                print(f"{group}_{section}_{key}: mean {st.mean(vals):.6f}  median {st.median(vals):.6f}")


if __name__ == "__main__":
    main()
