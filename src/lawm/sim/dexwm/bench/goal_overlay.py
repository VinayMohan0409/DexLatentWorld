"""Overlays of fitted goal hands on the goal frame, for eyeballing (reads score/ for the true goal pose: inspection only, never used by the planner)."""
#   python -m lawm.sim.dexwm.bench.goal_overlay fits  --root R --task T [--fits fits_torch] [--demos ...] --out F.png   predicted robot + true hand per demo
#   python -m lawm.sim.dexwm.bench.goal_overlay scale --root R --task T --demo D --out F.png                           the true hand vs copies shifted by 3 cm
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lawm.sim import clouds
from lawm.sim.dexwm.bench import goal_hand as gh
from lawm.sim.dexwm.bench.goal_frames import load_view

UP = 3  # display upscale of the 224 x 392 frame


def robot(root):
    """RobotFit with a dense surface sampler and a hand flag for each of its points (bodies of the hand subtree)."""
    rf = gh.RobotFit(root / "robot_model" / "robot.mjb", root / "robot_model" / "meta.pkl")
    rf.dense_sampler = clouds.SurfaceSampler(rf.m, rf.sampler_geoms, 30000, np.random.default_rng(2))
    hand = np.zeros(rf.m.nbody, bool)                       # bodies of the hand subtree (root = the Allegro hand body)
    for b in range(rf.m.nbody):
        a = b
        while a > 0:
            if a == rf.root_body:
                hand[b] = True
                break
            a = rf.m.body_parentid[a]
    rf.is_hand = hand[rf.m.geom_bodyid[rf.dense_sampler.owner]]
    return rf


def world_points(rf, q, T):
    """Dense model points at joint angles q in the world frame (model-to-world pose T)."""
    rf._fk(q)
    return rf.dense_sampler(rf.d).astype(float) @ T[:3, :3].T + T[:3, 3]


def project(view, P):
    """Pixel coords (u, v) and a visibility flag (in front of the measured depth, in the image) of world points P."""
    _, depth, K, pose = view
    pc = (P - pose[:3, 3]) @ pose[:3, :3]
    z = np.maximum(pc[:, 2], 1e-6)
    u, v = K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]
    H, W = depth.shape
    ins = (pc[:, 2] > 0.05) & (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
    ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
    vis = ins & (depth[vi, ui] > pc[:, 2] - 0.02)            # not hidden behind measured geometry (e.g. the held object)
    return u, v, vis


def silhouette(view, P):
    """Binary image of the projected points (dilated a little so the sampled surface reads as a solid)."""
    from scipy.ndimage import binary_closing, binary_dilation
    _, depth, _, _ = view
    u, v, vis = project(view, P)
    m = np.zeros(depth.shape, bool)
    m[np.round(v[vis]).astype(int), np.round(u[vis]).astype(int)] = True
    return binary_closing(binary_dilation(m, iterations=1), iterations=2)


def draw(ax, rgb, layers, title):
    """layers: list of (mask, rgba fill, edge colour or None)."""
    from scipy.ndimage import binary_erosion
    img = rgb.astype(float) / 255
    for m, fill, edge in layers:
        if fill is not None:
            img[m] = img[m] * (1 - fill[3]) + np.asarray(fill[:3]) * fill[3]
        if edge is not None:
            img[m & ~binary_erosion(m)] = edge
    ax.imshow(np.kron(img, np.ones((UP, UP, 1))), interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def fits(root, task, fits_dir, demos, out):
    """Figure with, per demo, the goal frame, the predicted robot and the predicted vs true hand outline; saved to out."""
    rf = robot(root)
    d = root / task
    demos = demos or sorted(p.stem for p in (d / fits_dir).glob("demo_*.npz"))
    fig, axes = plt.subplots(len(demos), 3, figsize=(15, 2.9 * len(demos)), squeeze=False)
    for row, demo in zip(axes, demos):
        fr, z, s = np.load(d / "frames" / f"{demo}.npz"), np.load(d / fits_dir / f"{demo}.npz"), np.load(d / "score" / f"{demo}.npz")
        view, T = load_view(fr, "goal"), z["model_to_world"]
        q_true = np.r_[fr["start_q"][:3], s["goal_q"][3:26], fr["start_q"][26:]]
        Pf, Pt = world_points(rf, z["robot_q"], T), world_points(rf, q_true, T)
        fh, fa, th = silhouette(view, Pf[rf.is_hand]), silhouette(view, Pf[~rf.is_hand]), silhouette(view, Pt[rf.is_hand])
        err = float(np.linalg.norm(z["goal_hand"] - s["goal_hand_fk"], axis=1).mean() * 1000)
        rgb = fr["goal_rgb"]
        draw(row[0], rgb, [], f"{task} {demo}: goal frame")
        draw(row[1], rgb, [(fa, (0.2, 0.5, 1.0, 0.35), None), (fh, (0.1, 1.0, 0.2, 0.55), (0.0, 0.6, 0.0))], "predicted (green hand, blue arm)")
        draw(row[2], rgb, [(fh, (0.1, 1.0, 0.2, 0.45), None), (th, None, (1.0, 0.0, 0.0))], f"predicted hand vs TRUE hand outline (red): {err:.1f} mm")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print("saved", out)


def scale(root, task, demo, out, shift=0.03):
    """The true hand and copies of it shifted by `shift` along the camera's right, down and depth axes."""
    rf = robot(root)
    d = root / task
    fr, s = np.load(d / "frames" / f"{demo}.npz"), np.load(d / "score" / f"{demo}.npz")
    z = next((np.load(d / f / f"{demo}.npz") for f in ("fits_torch", "fits") if (d / f / f"{demo}.npz").exists()))
    view, T = load_view(fr, "goal"), z["model_to_world"]
    q_true = np.r_[fr["start_q"][:3], s["goal_q"][3:26], fr["start_q"][26:]]
    Pt = world_points(rf, q_true, T)[rf.is_hand]
    R = view[3][:3, :3]
    th = silhouette(view, Pt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, (name, axis) in zip(axes, (("right", 0), ("down", 1), ("away from camera", 2))):
        sh = silhouette(view, Pt + shift * R[:, axis])
        draw(ax, fr["goal_rgb"], [(th, None, (1.0, 0.0, 0.0)), (sh, (1.0, 0.8, 0.0, 0.5), (0.9, 0.6, 0.0))],
             f"{task} {demo}: true hand (red) vs shifted {shift * 100:.0f} cm {name} (yellow)")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print("saved", out)


def main():
    """Command-line entry point: fits or scale."""
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fits", "scale"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--task", default="grasp")
    p.add_argument("--fits", default="fits_torch")
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--demo", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "fits":
        fits(a.root, a.task, a.fits, a.demos, a.out)
    else:
        scale(a.root, a.task, a.demo, a.out)


if __name__ == "__main__":
    main()
