"""Goal hands of the benchmark demos from their goal frames (bench/goal_frames.py output), method: bench/goal_hand.goal_hand (SAM 3 robot mask -> right arm + hand fit -> barycentric hand rows)."""
# Inputs: the goal RGB-D frame, the camera, SAM 3 with generic prompts, the robot model, the start proprioception. Never the goal state.
#
#   python -m lawm.sim.dexwm.bench.goal_hands model --root R              compiled robot model (one kitchen build) -> R/robot_model/{robot.mjb,meta.pkl}
#   python -m lawm.sim.dexwm.bench.goal_hands masks --root R --task T     SAM 3 (GPU), one model load, every frame, top N_INSTANCES per prompt -> R/T/masks/
#   python -m lawm.sim.dexwm.bench.goal_hands fit   --root R --task T     the fit (CPU), split over --workers processes (run with OMP_NUM_THREADS=1) -> R/T/fits/
#   python -m lawm.sim.dexwm.bench.goal_hands fit   --root R --task T --method torch [--workers N]   batched silhouette + depth fit (goal_hand_torch.py, GPU),
#                                                                         N demos in parallel (one process + CUDA context each; OMP_NUM_THREADS per process) -> R/T/fits_torch/
#   python -m lawm.sim.dexwm.bench.goal_hands flags --root R --task T [--fits fits_torch]   (re)compute the truth-free checks of existing fits (no refit) into each fit row
#   python -m lawm.sim.dexwm.bench.goal_hands check --root R --task T [--fits fits_torch]   table + flags; exit 1 if any demo is flagged
#   python -m lawm.sim.dexwm.bench.goal_hands report --root R [--fits fits_torch] [--tasks ...]  all tasks: every demo over 30 mm / goal-flagged, + overlays -> R/goal_report.{txt,json}, R/overlays/
#
# Truth-free flags (goal_hand.goal_checks, limits in goal_hand.CHECK_LIMITS) are stored in each fit row as row["checks"] / row["goal_flags"]; the strict planner
# (planspace/evaluate.py) refuses a goal whose row has goal flags or no checks.
#
# R/T/frames and R/T/score come from goal_frames.py. check reads R/T/score (goal FK) for the error column ONLY; the planner reads R/T/fits.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from lawm.sim.dexwm.bench import goal_hand as gh
from lawm.sim.dexwm.bench.goal_frames import load_view

ERR_MM, RESID_MM, MIN_HAND_POINTS = 50.0, 5.0, 50  # check flags: error vs goal FK (reporting only), fit residual, observed points on the fitted hand


def demos_of(d, demos):
    """Demos with a frames file under d, restricted to `demos` if given."""
    have = sorted(p.stem for p in (d / "frames").glob("demo_*.npz"))
    return [x for x in have if not demos or x in demos]


def model(root):
    """Compile the robot model once from a kitchen build and save robot.mjb + meta.pkl under root/robot_model."""
    import pickle
    import mujoco
    from lawm.sim.dexwm.env import benchmark, kitchen
    from lawm.sim.dexwm.planspace import production as P
    from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, eval_dir
    from lawm.sim.dexwm.runner import load_config
    out = root / "robot_model"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.pkl").exists():
        print("robot model exists", out)
        return
    cfg = load_config(DEXWM_CONFIG)
    sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), "demo_133"), cfg.env)
    try:
        sim.reset("place", cfg.scene_seed)
        m = sim.model
        mujoco.mj_saveModel(m, str(out / "robot.mjb"), None)
        meta = dict(robot_geoms=P.robot_geoms(m), kin=sim.kinematics, nq=m.nq, nv=m.nv)
        pickle.dump(meta, open(out / "meta.pkl", "wb"))
    finally:
        sim.env.close()
    print("saved", out)


def masks(root, task, demos):
    """SAM 3 instance masks (top N_INSTANCES per prompt) of every goal frame not done yet, saved under root/task/masks."""
    from lawm.sim.dexwm.planspace.segment import Segmenter
    d = root / task
    (d / "masks").mkdir(exist_ok=True)
    todo = [x for x in demos_of(d, demos) if not (d / "masks" / f"{x}.npz").exists()]
    if not todo:
        print("masks: all done")
        return
    seg = Segmenter(prompts=gh.PROMPTS, threshold=0.0)
    t0 = time.time()
    try:
        for i in range(0, len(todo), 16):  # chunks of 16 frames per request (one SAM 3 load for all)
            chunk = todo[i:i + 16]
            res = seg.segment([np.load(d / "frames" / f"{x}.npz")["goal_rgb"] for x in chunk])
            for x, r in zip(chunk, res):
                save = {}
                for j, p in enumerate(gh.PROMPTS):
                    o = np.argsort(-np.asarray(r[p]["scores"]))[: gh.N_INSTANCES]
                    save[f"p{j}_masks"] = np.asarray(r[p]["masks"])[o].astype(bool)
                    save[f"p{j}_scores"] = np.asarray(r[p]["scores"], np.float32)[o]
                np.savez_compressed(d / "masks" / f"{x}.npz", **save)
            print(f"masks: {min(i + 16, len(todo))}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        seg.close()


def instances(f):
    """SAM instances of a masks file as a list of (mask, score, prompt)."""
    mk = np.load(f)
    return [(m, float(s), p) for j, p in enumerate(gh.PROMPTS) for m, s in zip(mk[f"p{j}_masks"], mk[f"p{j}_scores"])]


_RF = None  # the robot model, loaded once in the parent and shared with the forked workers (copy-on-write); torch: one per spawned worker


def _torch_fitter(root):
    """TorchFit with its defaults; GOAL_FIT_OPTS='{"n_starts": 16, ...}' overrides constructor arguments (quick code-path checks only, never for results)."""
    from lawm.sim.dexwm.bench.goal_hand_torch import TorchFit
    opts = json.loads(os.environ.get("GOAL_FIT_OPTS", "{}"))
    if "steps" in opts:
        opts["steps"] = tuple(opts["steps"])
    return TorchFit(root / "robot_model" / "robot.mjb", root / "robot_model" / "meta.pkl", **opts)


def _init_torch_worker(root):
    """Pool initializer: set the torch thread count and build this worker's TorchFit."""
    global _RF
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    _RF = _torch_fitter(Path(root))


def _fit_one(args):
    """Fit the goal hand of one demo and save its fit file atomically; args = (task dir, demo, fits dir name); returns (demo, row)."""
    d, demo, out = args
    d = Path(d)
    t0 = time.time()
    z = np.load(d / "frames" / f"{demo}.npz")
    r = gh.goal_hand(_RF, load_view(z, "goal"), instances(d / "masks" / f"{demo}.npz"), z["start_q"], z["start_root"])
    row = dict(fitted=r is not None)
    save = {}
    if r is not None:
        row.update(resid_mm=round(float(r["resid_mm"]), 2), n_points=int(r["n_points"]), hand_points=int(r["hand_points"]), hand_seen=bool(r["hand_seen"]),
                   n_pieces_kept=int(r["n_pieces_kept"]), n_pieces_dropped=int(r["n_pieces_dropped"]))
        if "seed_kind" in r:
            row.update(seed_kind=r["seed_kind"], score_mm=round(float(r["score_mm"]), 2))
        row.update(checks=r["checks"], goal_flags=r["flags"])
        save = dict(goal_hand=r["hand"].astype(np.float32), hand_q=r["hand_q"], hand_root=r["hand_root"], robot_q=r["robot_q"], model_to_world=r["model_to_world"], mask=r["mask"])
        if r.get("candidates"):
            save.update(cand_score=np.array([c[0] for c in r["candidates"]]), cand_q=np.stack([c[1] for c in r["candidates"]]))
    row["seconds"] = round(time.time() - t0, 1)
    tmp = d / out / f".{demo}.tmp.npz"
    np.savez_compressed(tmp, row=json.dumps(row), **save)
    tmp.replace(d / out / f"{demo}.npz")
    return demo, row


def fit(root, task, demos, workers, method="gn"):
    """Goal-hand fits (GN RobotFit or batched TorchFit) of every demo not fitted yet, over `workers` processes."""
    global _RF
    d = root / task
    out = "fits" if method == "gn" else "fits_torch"
    (d / out).mkdir(exist_ok=True)
    todo = [x for x in demos_of(d, demos) if not (d / out / f"{x}.npz").exists()]
    missing = [x for x in todo if not (d / "masks" / f"{x}.npz").exists()]
    assert not missing, f"run masks first: {missing[:5]}"
    if not todo:
        print("fit: all done")
        return
    t0 = time.time()
    if method == "torch":
        n = max(1, min(workers, len(todo)))
        if n == 1:   # in this process
            _RF = _torch_fitter(root)
            print("torch fit on", _RF.dev, flush=True)
            results = (_fit_one((str(d), x, out)) for x in todo)
            pool = None
        else:        # n spawned workers (CUDA cannot be forked), each with its own fitter; demos are handed out one at a time
            print(f"torch fit: {n} worker processes, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)
            pool = get_context("spawn").Pool(n, initializer=_init_torch_worker, initargs=(str(root),))
            results = pool.imap_unordered(_fit_one, [(str(d), x, out) for x in todo])
        try:
            for i, (demo, row) in enumerate(results):
                short = {k: row[k] for k in ("fitted", "resid_mm", "score_mm", "hand_points", "goal_flags", "seconds") if k in row}
                print(f"fit {i + 1}/{len(todo)} {demo} {short} ({time.time() - t0:.0f} s)", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        return
    _RF = gh.RobotFit(root / "robot_model" / "robot.mjb", root / "robot_model" / "meta.pkl")
    _RF.dense_sampler = gh.clouds.SurfaceSampler(_RF.m, _RF.sampler_geoms, 30000, np.random.default_rng(2))  # built once here, shared by the workers (as RobotFit.fit builds it)
    with get_context("fork").Pool(max(1, min(workers, len(todo)))) as pool:
        for i, (demo, row) in enumerate(pool.imap_unordered(_fit_one, [(str(d), x, out) for x in todo])):
            print(f"fit {i + 1}/{len(todo)} {demo} {row} ({time.time() - t0:.0f} s)", flush=True)


def flags(root, task, demos, fits="fits"):
    """Recompute the truth-free checks of existing fits (frames + the fit's saved mask, robot q and candidates; no refit) and store them in each fit row."""
    from scipy.spatial import cKDTree
    d = root / task
    rf = gh.RobotFit(root / "robot_model" / "robot.mjb", root / "robot_model" / "meta.pkl")
    rf.dense_sampler = gh.clouds.SurfaceSampler(rf.m, rf.sampler_geoms, 30000, np.random.default_rng(2))
    for demo in demos_of(d, demos):
        f = d / fits / f"{demo}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f))
        row = json.loads(str(z["row"]))
        if not row["fitted"]:
            continue
        fz = np.load(d / "frames" / f"{demo}.npz")
        view = load_view(fz, "goal")
        pts = gh.mask_points(view, z["mask"])
        r = dict(robot_q=z["robot_q"], model_to_world=z["model_to_world"], hand=z["goal_hand"].astype(float))
        if "score_mm" in row:
            r["score_mm"] = row["score_mm"]
        else:  # GN fits: the same score as the fit's selection (median data->model on the dense model + model_to_view)
            T = z["model_to_world"]
            Ti = np.linalg.inv(T)
            Y = pts @ Ti[:3, :3].T + Ti[:3, 3]
            rf._fk(fz["start_q"])
            Y = Y[cKDTree(rf.fixed_sampler(rf.d)).query(Y)[0] > gh.FIXED_MARGIN]
            rf._fk(z["robot_q"])
            r["score_mm"] = 1000 * (float(np.median(cKDTree(rf.dense_sampler(rf.d).astype(float)).query(Y)[0])) + rf.model_to_view(z["robot_q"], T, view, z["mask"]))
        cands = list(zip(z["cand_score"], z["cand_q"])) if "cand_q" in z else None
        row["checks"], row["goal_flags"] = gh.goal_checks(rf, r, view, z["mask"], pts, cands)
        z["row"] = json.dumps(row)
        tmp = d / fits / f".{demo}.tmp.npz"
        np.savez_compressed(tmp, **z)
        tmp.replace(f)
        print(demo, row["checks"], "FLAGGED " + str(row["goal_flags"]) if row["goal_flags"] else "ok", flush=True)


def report(root, tasks, fits="fits_torch", limit_mm=30.0, per_image=10):
    """All tasks: every demo whose goal hand is more than limit_mm from the goal FK (scoring only) or carries goal flags (truth-free; the planner would skip it), with overlays of those demos (predicted robot + true hand outline)."""
    # Writes root/goal_report.txt, root/goal_report.json and root/overlays/<task>_<k>.png.
    from lawm.sim.dexwm.bench import goal_overlay
    lines, out = [], dict(fits=fits, limit_mm=limit_mm, tasks={})
    (root / "overlays").mkdir(exist_ok=True)
    for task in tasks:
        d = root / task
        rows = {}
        for demo in demos_of(d, None):
            f = d / fits / f"{demo}.npz"
            if not f.exists():
                rows[demo] = dict(status="not fitted (no fit file)")
                continue
            z = np.load(f)
            row = json.loads(str(z["row"]))
            if not row["fitted"]:
                rows[demo] = dict(status="fit failed (too few robot points)")
                continue
            s = np.load(d / "score" / f"{demo}.npz")
            err = float(np.linalg.norm(z["goal_hand"] - s["goal_hand_fk"], axis=1).mean() * 1000)
            rows[demo] = dict(err_mm=round(err, 1), goal_flags=row.get("goal_flags"), checks=row.get("checks"), seconds=row.get("seconds"))
        e = np.array([r["err_mm"] for r in rows.values() if "err_mm" in r])
        bad = {k: r for k, r in rows.items() if "err_mm" not in r or r["err_mm"] > limit_mm or r.get("goal_flags") or r.get("checks") is None}
        summ = dict(demos=len(rows), fitted=int(len(e)), median_mm=round(float(np.median(e)), 1) if len(e) else None,
                    p90_mm=round(float(np.percentile(e, 90)), 1) if len(e) else None, under_30mm=int((e <= 30).sum()), within_30_50mm=int(((e > 30) & (e <= 50)).sum()),
                    over_50mm=int((e > 50).sum()), goal_flagged=sorted(k for k, r in rows.items() if r.get("goal_flags")),
                    hand_not_detected=sorted(k for k, r in rows.items() if "err_mm" not in r or (r.get("checks") or {}).get("hand_points", 0) < gh.CHECK_LIMITS["hand_points"]))
        from collections import Counter
        summ["flag_counts"] = dict(Counter(f for r in rows.values() for f in (r.get("goal_flags") or [])))
        out["tasks"][task] = dict(summary=summ, listed=bad, rows=rows)
        lines.append(f"== {task}: {summ['fitted']}/{summ['demos']} fitted, hand detected in {summ['demos'] - len(summ['hand_not_detected'])}/{summ['demos']}, "
                     f"median {summ['median_mm']} mm, p90 {summ['p90_mm']} mm | <=30 mm: {summ['under_30mm']}, 30-50 mm: {summ['within_30_50mm']}, >50 mm: {summ['over_50mm']} | "
                     f"goal-flagged (planner skips): {len(summ['goal_flagged'])} {summ['flag_counts'] or ''}")
        for k, r in sorted(bad.items(), key=lambda kv: -kv[1].get("err_mm", 1e9)):
            if "err_mm" not in r:
                lines.append(f"   {k}: {r['status']}")
            else:
                lines.append(f"   {k}: {r['err_mm']} mm" + (f"  FLAGGED {r['goal_flags']}" if r.get("goal_flags") else "") + f"  checks {r['checks']}")
        shown = [k for k, r in sorted(bad.items(), key=lambda kv: -kv[1].get("err_mm", 0)) if "err_mm" in r]
        for i in range(0, len(shown), per_image):
            png = root / "overlays" / f"{task}_{i // per_image}.png"
            goal_overlay.fits(root, task, fits, shown[i:i + per_image], png)
            lines.append(f"   overlay: {png}")
    text = "\n".join(lines)
    print(text)
    (root / "goal_report.txt").write_text(text + "\n")
    (root / "goal_report.json").write_text(json.dumps(out, indent=1))


def check(root, task, demos, fits="fits"):
    """Table of the fits against the goal FK (scoring only) with flags; writes the check json and returns 1 if any demo is flagged, else 0."""
    d = root / task
    rows, flagged = [], []
    for demo in demos_of(d, demos):
        f = d / fits / f"{demo}.npz"
        if not f.exists():
            rows.append((demo, "not fitted")); flagged.append(demo); continue
        z = np.load(f)
        row = json.loads(str(z["row"]))
        s = np.load(d / "score" / f"{demo}.npz")  # scoring only
        flags = []
        if not row["fitted"]:
            flags.append("no fit")
        else:
            err = float(np.linalg.norm(z["goal_hand"] - s["goal_hand_fk"], axis=1).mean() * 1000)
            row["err_mm"] = round(err, 1)
            flags += [n for n, bad in (("err>50mm", err > ERR_MM), ("resid>5mm", row["resid_mm"] > RESID_MM), ("hand<50pts", row["hand_points"] < MIN_HAND_POINTS)) if bad]
        row["flags"] = flags
        rows.append((demo, row))
        if flags:
            flagged.append(demo)
    for demo, row in rows:
        print(demo, row if isinstance(row, str) else {k: row[k] for k in ("err_mm", "resid_mm", "hand_points", "seed_kind", "flags", "goal_flags", "seconds") if k in row})
    e = np.array([r["err_mm"] for _, r in rows if isinstance(r, dict) and "err_mm" in r])
    summary = dict(task=task, fits=fits, demos=len(rows), fitted=int(len(e)), median_err_mm=round(float(np.median(e)), 1) if len(e) else None,
                   under_30mm=int((e < 30).sum()), under_50mm=int((e < 50).sum()), flagged=flagged,
                   goal_flagged=[k for k, v in rows if isinstance(v, dict) and (v.get("goal_flags") or "checks" not in v)])   # truth-free: what the planner would refuse
    print(json.dumps(summary))
    (d / ("check.json" if fits == "fits" else f"check_{fits}.json")).write_text(json.dumps(dict(summary=summary, rows={k: v for k, v in rows}), indent=1))
    return 1 if flagged else 0


def main():
    """Command-line entry point: model, masks, fit, flags, check or report."""
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["model", "masks", "fit", "flags", "check", "report"])
    p.add_argument("--tasks", nargs="*", default=["place", "grasp", "reach"], help="report: the tasks")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--task", default="place", choices=["reach", "grasp", "place"])
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--method", default="torch", choices=["gn", "torch"], help="torch (default): TorchFit batched silhouette + depth (-> fits_torch/); gn: the old RobotFit (-> fits/)")
    p.add_argument("--fits", default="fits_torch", help="flags / check / report: the fits directory (fits_torch, or fits for the old GN fit)")
    a = p.parse_args()
    if a.cmd == "model":
        model(a.root)
    elif a.cmd == "masks":
        masks(a.root, a.task, a.demos)
    elif a.cmd == "fit":
        fit(a.root, a.task, a.demos, a.workers, a.method)
    elif a.cmd == "flags":
        flags(a.root, a.task, a.demos, a.fits)
    elif a.cmd == "report":
        report(a.root, a.tasks, a.fits)
    else:
        sys.exit(check(a.root, a.task, a.demos, a.fits))


if __name__ == "__main__":
    main()
