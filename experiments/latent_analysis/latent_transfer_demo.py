"""Human latent actions driving held-out hands: `links` (UniDex env) exports link meshes, `run` (GPU) rolls out, `plot` draws the GIF and mesh figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HAND, OBJECT, TABLE = 0, 1, 2
TITLES = {"human": "Human", "svh": "SVH", "dg5f": "DG-5F", "dex3": "Dex3"}
HAND_COLOURS = {"human": "#7fa7d9", "svh": "#7cc3a0", "dg5f": "#e8a67c", "dex3": "#d9c06a"}
OBJECT_COLOUR, TABLE_COLOUR = "#a29e95", "#c9c9c4"


def links(args) -> None:
    """Export each target hand's link meshes and point layout (run in the UniDex environment)."""
    from lawm.embodiments.registry import load_registry
    from lawm.viz.hand_geometry import export_robot_links
    args.out.mkdir(parents=True, exist_ok=True)
    for t in args.targets:
        export_robot_links(args.unidex_root, load_registry()[t].unidex_hand, 2048, args.out / f"links_{t}.npz")


def run(args) -> None:
    """Encode the human's latents; roll each target out from its own first frame with them; save clouds and errors."""
    import torch
    from lawm.data.config import read_json, resolve_data_path
    from lawm.data.dataset import load_window
    from lawm.data.sequences import Sequence
    from lawm.sim import model_adapter
    from lawm.train.evaluation import compute_context
    from lawm.train.reproducibility import configure_determinism, evaluation_stream

    ev = read_json(args.config)
    window = read_json(REPO / ev["training_config"])["window"]
    gap, n_steps, span = window["gap"], window["n_steps"], window["gap"] * window["n_steps"]
    seq = Sequence(resolve_data_path(ev["data_dir"]) / args.sequence, window["points_per_entity"])
    device = torch.device("cuda")
    configure_determinism(ev["determinism"])
    model = model_adapter.load_model(REPO / ev["checkpoint"], REPO / ev["run_manifest"], device)
    amp = ev["amp"] if args.amp is None else args.amp
    idx = {n: seq.names.index(n) for n in ["human"] + args.targets}
    seg = {n: s for n, (_, s) in zip(idx, seq.clouds([args.start], list(idx.values())))}

    def human_latents(start):
        """Encoder latents of the human window starting at frame `start` (steps first), as numpy."""
        coords = load_window(seq, start, [idx["human"]], gap, n_steps)[0]["coords"].to(device)
        with compute_context(device, amp):
            return model.encode_window(coords, torch.tensor([coords.shape[1]], device=device).int())[1].float().permute(1, 0, 2, 3).cpu().numpy()

    def decode(cloud, z):
        """Decode latents z on the encoded cloud; returns the predicted cloud of every step."""
        with compute_context(device, amp):
            q = model_adapter.repeated_queries(model, model_adapter.encode_cloud(model, cloud), copies=1)
            delta = model_adapter.decode_chunked(model, q, z, len(cloud), copies=1)[0]
        return [(cloud + d).astype(np.float32) for d in delta]

    # Rollout over the whole demonstration: each window's last prediction starts the next.
    frames = [args.start + k * gap for k in range(args.windows * n_steps + 1)]
    real = {n: c for n, (c, _) in zip(idx, seq.clouds(frames, list(idx.values())))}
    with torch.no_grad(), evaluation_stream(ev["eval_stream_seed"], device):
        zs = [human_latents(args.start + w * span) for w in range(args.windows)]
        pred = {}
        for t in args.targets:
            out = [real[t][0]]
            for z in zs:
                out += decode(out[-1].astype(np.float64), z)
            pred[t] = np.stack(out)
    # One window for the figure: where the human's object moves most; each target from its real frame.
    obj = real["human"][:, seg["human"] == OBJECT]
    w0 = int(np.argmax([np.linalg.norm(obj[(w + 1) * n_steps] - obj[w * n_steps], axis=1).mean() for w in range(args.windows)]))
    wframes = frames[w0 * n_steps:(w0 + 1) * n_steps + 1]
    with torch.no_grad(), evaluation_stream(ev["eval_stream_seed"], device):
        wpred = {t: np.stack([real[t][w0 * n_steps]] + decode(real[t][w0 * n_steps].astype(np.float64), zs[w0])) for t in args.targets}
    with np.load(seq.dir / "sequence.npz", allow_pickle=True) as sq:
        scene = dict(plane_n=sq["plane_n"], cam_pose=sq["cam_poses"][wframes[0]], object_id=int(sq["object_id"]))
    args.out.mkdir(parents=True, exist_ok=True)
    segs = {f"seg_{n}": seg[n] for n in idx}
    np.savez(args.out / "rollout.npz", frames=frames, human=real["human"], **{f"pred_{t}": pred[t] for t in args.targets}, **segs)
    np.savez(args.out / "window.npz", frames=wframes, human=real["human"][w0 * n_steps:(w0 + 1) * n_steps + 1], targets=args.targets,
             **{f"pred_{t}": wpred[t] for t in args.targets}, **segs, object_canonical=seq.object_canonical, sequence=args.sequence, **scene)
    err = {t: {s: float(np.linalg.norm(pred[t][-1] - real[t][-1], axis=1)[seg[t] == v].mean() * 1000) for s, v in (("hand_mm", HAND), ("object_mm", OBJECT))}
           for t in args.targets}
    (args.out / "rollout_error.json").write_text(json.dumps({"frames": [frames[0], frames[-1]], "last_frame_error": err}, indent=1) + "\n")
    print(json.dumps(err, indent=1))


def table_frame(plane_n, cam_pose, centre):
    """Rotation into a frame with z = table normal and x = horizontal direction to the camera."""
    z = np.asarray(plane_n, float) / np.linalg.norm(plane_n)
    x = np.asarray(cam_pose)[:3, 3] - centre
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def mesh_collection(parts, alpha):
    """One depth-sorted, Lambert-shaded Poly3DCollection of (verts, faces, colour) parts."""
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri = np.concatenate([v[f] for v, f, _ in parts])
    base = np.concatenate([np.repeat(np.asarray(to_rgb(c))[None], len(f), axis=0) for _, f, c in parts])
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.68 + 0.32 * np.abs(n @ (np.array([0.6, -0.3, 0.75]) / np.linalg.norm([0.6, -0.3, 0.75])))
    col = np.c_[np.clip(base * shade[:, None], 0, 1), np.full(len(tri), alpha)]
    return Poly3DCollection(tri, facecolors=col, edgecolors="none", rasterized=True, zsort="average")


def decimate(links_, max_faces):
    """Simplify link meshes to about max_faces triangles per hand (drawing speed only)."""
    import trimesh
    total = sum(len(r["faces"]) for r in links_)
    out = []
    for r in links_:
        m = trimesh.Trimesh(r["verts"], r["faces"], process=False)
        m.merge_vertices()  # STL repeats each triangle's corners; join them so simplification keeps the surface
        target = max(40, int(len(r["faces"]) * max_faces / total))
        if total > max_faces and len(m.faces) > target:
            m = m.simplify_quadric_decimation(face_count=target)
        out.append(dict(r, verts=np.asarray(m.vertices), faces=np.asarray(m.faces)))
    return out


def plot(args) -> None:
    """GIF of the rollout (points) and the front/top mesh figure of one window (first frame translucent, step 5 solid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from matplotlib import font_manager
    from matplotlib.animation import FuncAnimation, PillowWriter
    from lawm.preprocessing.h2o_readers import load_camera_to_world, parse_mano_file, resolve_object_mesh
    from lawm.viz.hand_geometry import kabsch, load_mano, load_robot_links, mano_vertices, robot_meshes

    family = "DejaVu Sans"
    if args.font_file:
        font_manager.fontManager.addfont(str(args.font_file))
        family = font_manager.FontProperties(fname=str(args.font_file)).get_name()
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42, "svg.fonttype": "path"})

    w = np.load(args.out / "window.npz")
    targets = [str(t) for t in w["targets"]]
    names = ["human"] + targets
    clouds = {"human": w["human"], **{t: w[f"pred_{t}"] for t in targets}}
    segs = {n: w[f"seg_{n}"] for n in names}
    act = np.concatenate([clouds[n][:, np.isin(segs[n], (HAND, OBJECT))].reshape(-1, 3) for n in names])
    centre = (act.min(0) + act.max(0)) / 2
    R = table_frame(w["plane_n"], w["cam_pose"], centre)
    loc = lambda x: (np.asarray(x) - centre) @ R.T  # noqa: E731
    half = float(np.abs(loc(act)).max()) + 0.005
    near = lambda c: np.all(np.abs(loc(c)) < half, axis=1)  # noqa: E731

    def axes(fig, pos, elev):
        """A 3D axis at subplot position pos with cubic limits, viewed from elevation elev, axes hidden."""
        ax = fig.add_subplot(*pos, projection="3d", computed_zorder=False)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
        ax.view_init(elev=elev, azim=0.0)
        ax.set_box_aspect((1, 1, 1), zoom=args.zoom)
        ax.set_axis_off()
        return ax

    # GIF: the whole rollout as points, front view.
    r = np.load(args.out / "rollout.npz")
    rc = {"human": r["human"], **{t: r[f"pred_{t}"] for t in targets}}
    fig, axs = plt.figure(figsize=(2.2 * len(names), 2.4)), []

    def frame(k):
        """Draw frame k of the rollout: one point-cloud subplot per source (the human and each target)."""
        for ax in axs:
            ax.remove()
        axs.clear()
        for i, n in enumerate(names):
            ax = axes(fig, (1, len(names), i + 1), 8.0)
            c, s = rc[n][k], r[f"seg_{n}"]
            for v, col in ((TABLE, TABLE_COLOUR), (OBJECT, OBJECT_COLOUR), (HAND, HAND_COLOURS[n])):
                m = (s == v) & (near(c) if v == TABLE else True)
                ax.scatter(*loc(c[m]).T, s=1.0, c=col, linewidths=0, depthshade=False)
            ax.set_title(TITLES[n] + ("" if n == "human" else " (Human Latents)"), fontsize=8)
            axs.append(ax)
    FuncAnimation(fig, frame, frames=len(r["human"]), interval=200).save(args.out / "pour_transfer.gif", writer=PillowWriter(fps=5), dpi=110)
    plt.close(fig)

    # Meshes: human from the raw H2O MANO annotation; robot links and the object by rigid fits to the predicted points.
    subj, sess, take = str(w["sequence"]).split("/")
    cam = next(p for p in (args.h2o_root / subj / sess / take / "cam4", args.h2o_root / f"{subj}_ego" / sess / take / "cam4") if p.is_dir())
    mano = {sd: load_mano(args.mano_root / f"MANO_{sd.upper()}.pkl") for sd in ("left", "right")}
    hand_links = {t: decimate(load_robot_links(args.out / f"links_{t}.npz"), args.max_faces) for t in targets}
    om = trimesh.load(resolve_object_mesh(args.object_root, int(w["object_id"])), force="mesh", process=False)
    meshes = {}
    for k in (0, -1):
        f = int(w["frames"][k])
        T = load_camera_to_world(cam, f) @ np.diag([1.0, -1.0, -1.0, 1.0])
        meshes["human", k] = [((np.diag([1.0, -1.0, -1.0]) @ (mano_vertices(mano[sd], p[1], p[2]) + p[0]).T).T @ T[:3, :3].T + T[:3, 3], mano[sd]["faces"])
                              for sd, p in parse_mano_file(cam / "hand_pose_mano" / f"{f:06d}.txt").items() if p is not None]
        for t in targets:
            meshes[t, k] = robot_meshes(hand_links[t], clouds[t][k][segs[t] == HAND])
        for n in names:
            Ro, to = kabsch(w["object_canonical"].astype(np.float64), clouds[n][k][segs[n] == OBJECT].astype(np.float64))
            meshes[n, k] = meshes[n, k] + [(np.asarray(om.vertices) @ Ro.T + to, np.asarray(om.faces), OBJECT_COLOUR)]

    fig = plt.figure(figsize=(args.width_in, args.width_in * 2 / len(names) + 0.25))
    for row, (view, elev) in enumerate((("Front View", 8.0), ("Top View", 90.0))):
        for i, n in enumerate(names):
            ax = axes(fig, (2, len(names), row * len(names) + i + 1), elev)
            c0 = clouds[n][0]
            ax.scatter(*loc(c0[(segs[n] == TABLE) & near(c0)]).T, s=0.4, c=TABLE_COLOUR, alpha=0.35, linewidths=0, depthshade=False, rasterized=True)
            for k, alpha in ((0, 0.18), (-1, 1.0)):
                ax.add_collection3d(mesh_collection([(loc(m[0]), m[1], m[2] if len(m) > 2 else HAND_COLOURS[n]) for m in meshes[n, k]], alpha))
            path = loc(clouds[n][:, segs[n] == OBJECT].mean(axis=1))
            ax.plot(*path.T, ls=(0, (1.2, 1.2)), c="black", lw=1.1, zorder=20)
            ax.scatter(*path[1:].T, s=2.5, c="black", zorder=21, depthshade=False)
            if row == 0:
                ax.set_title(TITLES[n] + ("" if n == "human" else "\n(Human Latents)"), fontsize=args.font_size, pad=0,
                             color="black" if n == "human" else "#b30000")
            if i == 0:
                ax.text2D(-0.04, 0.5, view, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=args.font_size)
    fig.subplots_adjust(left=0.03, right=1, bottom=0, top=0.9, wspace=0, hspace=-0.06)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"pour_window_mesh.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}/pour_transfer.gif and pour_window_mesh.* (frames {int(w['frames'][0])}-{int(w['frames'][-1])})")


def main(argv=None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("links", "run", "plot"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs/eval/cross_decode_val.json")
    ap.add_argument("--sequence", default="subject3/k2/1", help="default: H2O validation sequence, 'pour milk' at frames 358-474")
    ap.add_argument("--start", type=int, default=358)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--targets", nargs="+", default=["svh", "dg5f", "dex3"])
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--unidex-root", type=Path, help="links: UniDex checkout")
    ap.add_argument("--h2o-root", type=Path, help="plot: raw H2O (subjectN[_ego]/<session>/<take>/cam4) for the MANO mesh")
    ap.add_argument("--mano-root", type=Path, help="plot: folder with MANO_LEFT.pkl / MANO_RIGHT.pkl")
    ap.add_argument("--object-root", type=Path, help="plot: H2O object meshes")
    ap.add_argument("--font-file", type=Path, default=None)
    ap.add_argument("--width-in", type=float, default=7.0)
    ap.add_argument("--zoom", type=float, default=1.5)
    ap.add_argument("--font-size", type=float, default=14)
    ap.add_argument("--max-faces", type=int, default=20000)
    args = ap.parse_args(argv)
    {"links": links, "run": run, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
