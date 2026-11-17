"""The two images of a benchmark episode, rendered once per demo: the start frame and the goal frame from DexWM's camera (robot0_robotview_2, 224 x 392 RGB-D), exactly the start and goal images DexWM's planner is given."""
# The strict runner (planspace/evaluate.py --strict) reads the goal frame from here and never touches the
# target state itself.
#
#   <out>/<task>/frames/<demo>.npz   planner inputs only: start_/goal_ rgb, depth, K, pose (camera at the mono centre), start_q (robot qpos 0..48, proprioception),
#                                    start_root (4x4 hand root, FK of start_q), alignment_mm (camera points vs FK robot at the start; > 12 mm raises = ghosting guard)
#   <out>/<task>/score/<demo>.npz    SCORING ONLY, never read by the planner: goal_hand_fk (goal FK hand rows), goal_q
#
# Usage: python -m lawm.sim.dexwm.bench.goal_frames --out DIR --task place [--demos ...] [--n N] [--workers W]   (resumable per demo)
from __future__ import annotations

import argparse
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

N_ROBOT_Q = 49  # robot qpos 0..48 (base, right arm + hand, left arm + hand); bench/goal_hand.py ROBOT_Q


def render(args):
    """Render and save the frames and the scoring file of one demo; args = (demo, task, out); returns (demo, message)."""
    demo, task, out = args
    from lawm.sim.dexwm.env import benchmark, kitchen
    from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, eval_dir
    from lawm.sim.dexwm.planspace.production import ProductionScene
    from lawm.sim.dexwm.runner import load_config
    out = Path(out)
    f = out / "frames" / f"{demo}.npz"
    if f.exists():
        return demo, "done"
    cfg = load_config(DEXWM_CONFIG)
    sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), demo), cfg.env)
    try:
        obs = sim.reset(task, cfg.scene_seed)
        prod = ProductionScene(sim, seed=int(demo.split("_")[1]), cameras="dexwm")
        prod.set_start(obs.root_pose)  # raises if the camera points are > 12 mm off the FK robot (wrong camera pose / ghosting)
        k = sim.kinematics
        root = k.bodies[k.root]

        def grab():
            """(view dict, 4x4 hand root pose, robot qpos) at the current sim state."""
            (rgb, depth, K, pose), = prod.views(sim.data)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = sim.data.xmat[root].reshape(3, 3), sim.data.xpos[root]
            return dict(rgb=rgb, depth=depth.astype(np.float32), K=K, pose=pose), T, sim.data.qpos[:N_ROBOT_Q].copy()

        start, target = benchmark.get_indices(sim.episode["waypoints"], task)
        sim.switch(start)
        s, s_root, s_q = grab()
        sim.switch(target)
        g, g_root, g_q = grab()
        goal_hand_fk = k.points(g_root[:3, :3], g_root[:3, 3], sim.data.qpos[k.qpos_adr])
    finally:
        sim.env.close()
    frames = {**{f"start_{key}": v for key, v in s.items()}, **{f"goal_{key}": v for key, v in g.items()}}
    frames.update(start_q=s_q, start_root=s_root, alignment_mm=np.asarray(prod.alignment_mm, np.float32), task=task, demo=demo, camera="robot0_robotview_2")
    for sub, data in (("score", dict(goal_hand_fk=goal_hand_fk.astype(np.float32), goal_q=g_q)), ("frames", frames)):  # frames last: the demo counts as done once it exists
        (out / sub).mkdir(parents=True, exist_ok=True)
        tmp = out / sub / f".{demo}.tmp.npz"
        np.savez_compressed(tmp, **data)
        tmp.replace(out / sub / f"{demo}.npz")
    return demo, f"alignment {np.round(prod.alignment_mm, 1).tolist()} mm"


def load_view(z, which):
    """(rgb, depth, K, pose) of the 'start' or 'goal' frame of a frames file."""
    return tuple(z[f"{which}_{key}"] for key in ("rgb", "depth", "K", "pose"))


def main():
    """Render the frames of the chosen demos in parallel worker processes."""
    from lawm.sim.dexwm.planspace.common import eval_demos
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True, choices=["reach", "grasp", "place"])
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--n", type=int, default=None, help="first N of the 50 DexWM demos (manifest order)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    demos = a.demos or eval_demos()[: a.n]
    out = Path(a.out) / a.task
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for demo, msg in pool.imap_unordered(render, [(d, a.task, str(out)) for d in demos]):
            print(f"{demo}: {msg}", flush=True)


if __name__ == "__main__":
    main()
