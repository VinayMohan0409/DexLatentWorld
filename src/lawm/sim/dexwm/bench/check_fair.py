"""Fairness gate before the strict DexWM benchmark run: exits 1 (and the run must not start) unless every check passes. Prints one line per check."""
#   1 camera      every goal/start frame is DexWM's camera alone (robot0_robotview_2, 224 x 392)
#   2 ghosting    camera points vs FK robot <= 12 mm median at every demo's start (the stored alignment) and in every play clip; play poses at the mono centre
#   3 play data   single camera, no evaluation kitchen among the play scenes
#   4 goal        a goal-hand fit for every demo in --goal-fits (default fits_torch) with truth-free checks; goal-flagged demos are listed (the runner skips them, the
#                 summary counts them as failures), not fatal
#   5 loop        the usual loop: up to max_replans replans, success = a sustain_steps streak of DexWM's success check; strict mode drops only the 'converged' stop
#   6 model       the prior was built by this checkpoint (SHA-256), motion check passes, held-out hand error reported
#   7 cem         the CEM budget evaluate.py uses for the task is CEM_BUDGET (population x iterations), the budget of the planspace ablation runs
# Usage: python -m lawm.sim.dexwm.bench.check_fair --goal-root R --task place|grasp|reach --play PLAY --checkpoint-dir CK [--n 50]
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ALIGN_MAX_MM = 12.0
SHAPE = (224, 392)
CEM_BUDGET = (64, 4)


def main():
    """Run the fairness checks, one line each; exit 1 if any fails."""
    from lawm.sim import model_adapter as ma
    from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG
    from lawm.sim.dexwm.runner import load_config
    from lawm.sim.dexwm.bench.goal_hands import check as goal_check
    from lawm.sim.dexwm.planspace.common import eval_demos
    p = argparse.ArgumentParser()
    p.add_argument("--goal-root", required=True, type=Path)
    p.add_argument("--task", default="place", choices=["reach", "grasp", "place"])
    p.add_argument("--play", required=True, type=Path)
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--goal-fits", default="fits_torch")
    a = p.parse_args()
    demos = eval_demos()[: a.n]
    fails = []

    def report(name, ok, msg):
        """Print one PASS / FAIL line for a check and record a failure."""
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
        if not ok:
            fails.append(name)

    d = a.goal_root / a.task
    missing = [x for x in demos if not (d / "frames" / f"{x}.npz").exists()]
    report("frames", not missing, f"{len(demos) - len(missing)}/{len(demos)} demos have start + goal frames" + (f"; missing {missing[:5]}" if missing else ""))
    cams, shapes, aligns = set(), set(), []
    for x in demos:
        if (d / "frames" / f"{x}.npz").exists():
            z = np.load(d / "frames" / f"{x}.npz")
            cams.add(str(z["camera"]))
            shapes |= {z["start_depth"].shape, z["goal_depth"].shape}
            aligns.append(float(np.max(z["alignment_mm"])))
    report("1 camera", cams == {"robot0_robotview_2"} and shapes == {SHAPE}, f"cameras {sorted(cams)}, image shapes {sorted(shapes)}")
    report("2 ghosting (eval)", bool(aligns) and max(aligns) <= ALIGN_MAX_MM, f"start-frame camera vs FK robot: median {np.median(aligns):.1f} mm, worst {max(aligns, default=np.nan):.1f} mm (limit {ALIGN_MAX_MM})")

    man = json.loads((a.play / "play_manifest.json").read_text())
    clash = set(man["scenes"]) & set(eval_demos())
    report("3 play data", man.get("cameras") == "dexwm" and not clash, f"cameras {man.get('cameras')}, {len(man['scenes'])} kitchens, overlap with eval set: {sorted(clash) or 'none'}")
    rgbd = sorted(a.play.glob("demo_*.rgbd.npz"))
    bad_pose = [f.name for f in rgbd if not ("pose_centred" in (z := np.load(f)) and bool(z["pose_centred"]))]
    play_align = [float(np.max(np.load(f)["alignment_mm"])) for f in rgbd if "alignment_mm" in np.load(f)]
    report("2 ghosting (play)", bool(rgbd) and not bad_pose and man.get("pose_centred") is True and max(play_align, default=99) <= ALIGN_MAX_MM,
           f"{len(rgbd)} clips, {len(bad_pose)} without centre pose, worst alignment {max(play_align, default=np.nan):.1f} mm")

    print("--- goal hands ---")
    goal_check(a.goal_root, a.task, demos, a.goal_fits)
    rows = {x: json.loads(str(np.load(d / a.goal_fits / f"{x}.npz")["row"])) for x in demos if (d / a.goal_fits / f"{x}.npz").exists()}
    fits = [x for x, r in rows.items() if r["fitted"]]
    no_checks = [x for x in fits if "checks" not in rows[x]]
    goal_flagged = [x for x in fits if rows[x].get("goal_flags")]
    report("4 goal", len(fits) == len(demos) and not no_checks,
           f"{len(fits)}/{len(demos)} fitted goal hands in {a.goal_fits}" + (f"; {len(no_checks)} without truth-free checks (run goal_hands flags)" if no_checks else "")
           + (f"; GOAL-FLAGGED (skipped by the runner, counted as failures): {goal_flagged}" if goal_flagged else "; no goal flags"))

    loop = load_config(DEXWM_CONFIG).loop[a.task]
    report("5 loop", loop.max_replans >= 80 and loop.sustain_steps == 10, f"up to {loop.max_replans} replans, success = {loop.sustain_steps}-step streak at any point, stall stop after {loop.stall_patience} replans without improvement")

    ck = a.checkpoint_dir
    try:
        ma.load_prior(ck / "prior.npz", ck / "stage1.pt")
        sha_ok, sha_msg = True, "prior built by this checkpoint"
    except Exception as e:
        sha_ok, sha_msg = False, str(e)
    mc = json.loads((ck / "motion_check.json").read_text())
    rep = json.loads((ck / "stage1_report.json").read_text())
    motion_ok = 0.7 <= mc["size_ratio_step5"] <= 1.3 and mc["direction_cosine_step5_median"] >= 0.9
    report("6 model", sha_ok and motion_ok, f"{sha_msg}; motion size ratio {mc['size_ratio_step5']}, direction cosine {mc['direction_cosine_step5_median']}; "
           f"held-out hand error {rep['before']['hand_err_mm']} -> {rep['after']['hand_err_mm']} mm")
    from lawm.sim.dexwm.planspace.evaluate import cem_for
    cem = cem_for(load_config(DEXWM_CONFIG), a.task)
    report("7 cem", (cem.pop, cem.iters) == CEM_BUDGET, f"{a.task}: {cem.pop} candidates x {cem.iters} iterations (required {CEM_BUDGET[0]} x {CEM_BUDGET[1]}), elite {cem.elite_frac}, min std {cem.min_std}")
    print(f"ALL CHECKS PASS ({a.task})" if not fails else f"FAILED ({a.task}): {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
