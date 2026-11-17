"""Tables from stats.npz (no GPU): per condition and step, EPE for mover/hand/object (>=5 mm), hand/object (all points), scene (all points), moving-point counts, and a check of the recomputed Mover EPE against the original ablation.csv."""
# python report.py <run_dir>      writes <run_dir>/tables.csv, <run_dir>/moving_points.csv, <run_dir>/validation.txt
# Aggregation as the original: pooled over windows within a step, then the plain mean of the 5 per-step means.
import csv
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
z = np.load(run / "stats.npz")
S, conds = z["stats"], [str(c) for c in z["conditions"]]  # (W, C, step, seg, thr, 2)
HAND, OBJ, TABLE, DIS = 0, 1, 2, 3
T5, T2, ALL = 0, 1, 2
tot = S.sum(0)  # (C, step, seg, thr, 2)

# metric name -> (segments, threshold)
METRICS = {
    "mover_5mm (hand+object)": ((HAND, OBJ), T5),
    "hand_5mm": ((HAND,), T5),
    "object_5mm": ((OBJ,), T5),
    "hand_all": ((HAND,), ALL),
    "object_all": ((OBJ,), ALL),
    "scene_all (table+distractors)": ((TABLE, DIS), ALL),
    "scene_5mm (table+distractors)": ((TABLE, DIS), T5),
}


def cell(c, k, segs, thr):
    """(error sum in mm, point count) of condition c at step k, summed over segments segs at threshold thr."""
    v = tot[c, k, list(segs), thr].sum(0)
    return v[0], v[1]


rows = [["condition", "metric", "step1", "step2", "step3", "step4", "step5", "mean_of_steps", "points_total"]]
for ci, c in enumerate(conds):
    for name, (segs, thr) in METRICS.items():
        per = [cell(ci, k, segs, thr) for k in range(S.shape[2])]
        eps = [s / n if n else float("nan") for s, n in per]
        rows.append([c, name] + [f"{e:.4f}" for e in eps] + [f"{np.nanmean(eps):.4f}", int(sum(n for _, n in per))])
with open(run / "tables.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

# moving points (>=5 mm), identical for every condition: depends on ground truth only
mp = [["step", "hand_moving", "object_moving", "hand_total", "object_total", "hand_moving_pct", "object_moving_pct", "scene_moving", "scene_total"]]
for k in range(S.shape[2]):
    hm, om, ht, ot = tot[0, k, HAND, T5, 1], tot[0, k, OBJ, T5, 1], tot[0, k, HAND, ALL, 1], tot[0, k, OBJ, ALL, 1]
    mp.append([k + 1, int(hm), int(om), int(ht), int(ot), f"{100 * hm / ht:.2f}", f"{100 * om / ot:.2f}",
               int(tot[0, k, [TABLE, DIS], T5, 1].sum()), int(tot[0, k, [TABLE, DIS], ALL, 1].sum())])
with open(run / "moving_points.csv", "w", newline="") as f:
    csv.writer(f).writerows(mp)

# validation against the original published numbers
orig = {}
for r in csv.reader(open(Path(__file__).resolve().parents[2] / "results/adaln_s0_k4/ablation.csv")):
    if r[:2] == ["best", "val_subset_final"] and r[4] == "mover_5mm":
        orig[(r[2], r[3])] = (float(r[5]), int(r[6]))
out, worst = [f"windows: {S.shape[0]}  (original: 4000)"], 0.0
for c in conds:
    if c == "no_motion":
        continue
    ci = conds.index(c)
    for k in range(S.shape[2]):
        s, n = cell(ci, k, (HAND, OBJ), T5)
        o_mm, o_n = orig[(c, str(k + 1))]
        d = s / n - o_mm
        worst = max(worst, abs(d))
        out.append(f"{c:16s} step{k + 1}: new {s / n:8.4f}  orig {o_mm:8.4f}  diff {d:+.4f}  count new {int(n)} orig {o_n} {'OK' if int(n) == o_n else 'COUNT MISMATCH'}")
out.append(f"max |diff| mm over 4 conditions x 5 steps: {worst:.4f}")
# hand+object must equal the mover set; no moving static points expected
assert np.allclose(tot[..., HAND, T5, :] + tot[..., OBJ, T5, :], tot[..., [HAND, OBJ], T5, :].sum(-2))
out.append(f"moving scene points (>=5mm, should be 0): {int(tot[0, :, [TABLE, DIS], T5, 1].sum())}")
(run / "validation.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
