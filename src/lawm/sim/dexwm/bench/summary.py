"""Success rate of a strict benchmark run, in DexWM's manifest order: running total after every 10 finished episodes ("6/10", "13/20", ...), plus per-demo rows."""
# Episodes skipped because their goal hand was flagged (<demo>_<task>.goal_flagged.json, planspace/evaluate.py) are listed and counted as FAILURES (never dropped).
#
# Usage: python -m lawm.sim.dexwm.bench.summary --out EVAL_DIR [--task place] [--goal-root R] [--goal-fits fits_torch]   (writes EVAL_DIR/summary.json)
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    """Print the per-demo rows and running success totals of a strict run and write summary.json."""
    from lawm.sim.dexwm.planspace.common import eval_demos
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--task", default="place")
    p.add_argument("--goal-root", type=Path, default=None)
    p.add_argument("--goal-fits", default="fits_torch")
    a = p.parse_args()
    check = {}
    cj = a.goal_root / a.task / ("check.json" if a.goal_fits == "fits" else f"check_{a.goal_fits}.json") if a.goal_root else None
    if cj and cj.exists():
        check = json.loads(cj.read_text())["rows"]
    rows = []
    for demo in eval_demos():
        f = a.out / f"{demo}_{a.task}.json"
        if f.exists():
            r = json.loads(f.read_text())
            g = check.get(demo, {}) if isinstance(check.get(demo), dict) else {}
            rows.append(dict(demo=demo, success=bool(r["success"]), replans=r["n_replans"], stop=r["stop_reason"], final_error_mm=round(r["final_error_mm"], 1),
                             goal_hand_err_mm=g.get("err_mm"), video=f"{demo}_{a.task}.mp4" if (a.out / f"{demo}_{a.task}.mp4").exists() else None))
        elif (a.out / f"{demo}_{a.task}.goal_flagged.json").exists():
            r = json.loads((a.out / f"{demo}_{a.task}.goal_flagged.json").read_text())
            rows.append(dict(demo=demo, success=False, replans=0, stop="goal_flagged", final_error_mm=None, goal_hand_err_mm=None, video=None, reason=r["reason"]))
    for r in rows:
        if r["stop"] == "goal_flagged":
            print(f"{r['demo']:>10}  GOAL FLAGGED, not run (counted as a failure): {r['reason']}")
            continue
        print(f"{r['demo']:>10}  {'SUCCESS' if r['success'] else 'fail   '}  {r['stop']:>9} after {r['replans']} replans  final {r['final_error_mm']} mm  goal hand err {r['goal_hand_err_mm']} mm")
    blocks = []
    for n in range(10, len(rows) + 1, 10):
        s = sum(r["success"] for r in rows[:n])
        blocks.append(f"{s}/{n}")
    total = sum(r["success"] for r in rows)
    print("running success every 10:", "  ".join(blocks) or "-")
    n_flag = sum(r["stop"] == "goal_flagged" for r in rows)
    print(f"total: {total}/{len(rows)} = {100 * total / max(1, len(rows)):.0f}%" + (f"  ({n_flag} goal-flagged episodes counted as failures)" if n_flag else ""))
    (a.out / "summary.json").write_text(json.dumps(dict(task=a.task, done=len(rows), successes=total, every_10=blocks, rows=rows), indent=1))


if __name__ == "__main__":
    main()
