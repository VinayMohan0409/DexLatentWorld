"""Starting wrist offset (UniDex base_joint origin) for a new hand: minimise fingertip + knuckle IK error on training frames."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import itertools
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lawm.embodiments import unidex_hand as uh
from lawm.preprocessing.unidex_adapter import open_unidex

REPO = Path(__file__).resolve().parents[3]
POSE_KEYS = ("left_pose", "right_pose", "left_R_hand_world", "right_R_hand_world")


def frames_for_calibration(dataset, sequences: List[str], per_sequence: int) -> List[dict]:
    """Evenly spaced frames with both hands present from each named sequence."""
    index_of = {str(dataset[i].index).replace("_ego", ""): i for i in range(len(dataset))}
    out = []
    for name in sequences:
        seq = dataset[index_of[f"{name}/cam4"]]
        for f in np.linspace(0, len(seq) - 1, per_sequence).round().astype(int):
            d = seq[int(f)]
            if all(d.get(k) is not None for k in POSE_KEYS):
                out.append({k: d[k] for k in POSE_KEYS})
    return out


def set_origin(urdf_text: str, xyz, rpy) -> str:
    """URDF text with the base_joint origin replaced."""
    root = ET.fromstring(urdf_text)
    for j in root.findall("joint"):
        if j.get("name") == "base_joint":
            j.find("origin").set("xyz", " ".join(f"{v:.6f}" for v in xyz))
            j.find("origin").set("rpy", " ".join(f"{v:.6f}" for v in rpy))
    return ET.tostring(root, encoding="unicode")


def fit_errors_mm(uv, urdf: Path, side: str, frames: List[dict], knuckles: Dict[int, str]):
    """Mean (fingertip, knuckle) distance in mm after UniDex IK; knuckles maps an H2O MCP joint index to a robot link."""
    import pybullet as p
    with contextlib.redirect_stdout(io.StringIO()):  # UniDex prints every IK solve
        if uv.WebHandVisualizer.physics_client is not None:
            p.resetSimulation()  # one robot at a time: removeBody leaves PyBullet's IK in a bad state
        vis = uv.WebHandVisualizer(str(urdf), is_left=(side == "left"))
        if len(vis.tip_links) != len(vis.config["tips"]):  # UniDex silently drops tip names it cannot find
            raise ValueError(f"{urdf}: tips {vis.config['tips']} not all found")
        link = {p.getJointInfo(vis.robot, j)[12].decode(): j for j in range(p.getNumJoints(vis.robot))}
        R_o2b = uv.B2O[:3, :3].T
        t_o2b = -R_o2b @ uv.B2O[:3, 3]
        pos = lambda i: np.array(p.getLinkState(vis.robot, i, computeForwardKinematics=True)[4])  # noqa: E731
        tip_err, knuckle_err = [], []
        for d in frames:  # posed exactly as the dataset export poses it
            vis.load_hand_pose_and_rot(d[f"{side}_pose"], d[f"{side}_R_hand_world"], is_left=(side == "left"))
            joints = vis.pose @ R_o2b.T + t_o2b
            p.resetBasePositionAndOrientation(vis.robot, joints[0].tolist(), R.from_matrix(R_o2b @ vis.R_hand_world).as_quat().tolist())
            goals = [joints[i] for i in uv.MANO_TIP_INDEX][: len(vis.tip_links)]
            for joint, angle in zip(vis.actuated, vis.solve_ik([g.tolist() for g in goals])):
                p.resetJointState(vis.robot, joint, float(angle))
            tip_err.append(np.mean([np.linalg.norm(pos(t) - g) for t, g in zip(vis.tip_links, goals)]))
            if knuckles:
                knuckle_err.append(np.mean([np.linalg.norm(pos(link[n]) - joints[h]) for h, n in knuckles.items()]))
    return 1000.0 * float(np.mean(tip_err)), 1000.0 * float(np.mean(knuckle_err)) if knuckle_err else 0.0


def axis_rotations() -> List[np.ndarray]:
    """The 24 axis-aligned proper rotations, as rpy."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            M[range(3), perm] = signs
            if np.isclose(np.linalg.det(M), 1):
                out.append(R.from_matrix(M).as_euler("xyz"))
    return out


def calibrate_side(uv, side_dir: Path, side: str, frames: List[dict], knuckles: Dict[int, str], refine_top: int = 2, maxfev: int = 80) -> Dict:
    """Best base_joint origin of one side: 24 axis-aligned starts, the best refined by Nelder-Mead over xyz + rpy."""
    work = Path(tempfile.mkdtemp(prefix="lawm_calib_"))  # private copy: another side finishing may rebuild the hand folder
    try:
        shutil.copytree(side_dir.parent, work / side_dir.parent.name)
        text = (work / side_dir.parent.name / side_dir.name / "main.urdf").read_text()
        trial = work / side_dir.parent.name / side_dir.name / "_calib.urdf"  # next to main.urdf, so relative meshes resolve

        def err(x):
            """Summed fit error (mm) with the base_joint origin set to x (xyz + rpy)."""
            trial.write_text(set_origin(text, x[:3], x[3:]))
            return sum(fit_errors_mm(uv, trial, side, frames, knuckles))
        coarse = sorted((err(np.r_[0, 0, 0, rpy]), i, rpy) for i, rpy in enumerate(axis_rotations()))
        steps = np.r_[0.02, 0.02, 0.02, 0.3, 0.3, 0.3]  # 2 cm and 0.3 rad (scipy's default simplex barely moves a 0 coordinate)
        best = min((minimize(err, np.r_[0, 0, 0, rpy], method="Nelder-Mead",
                             options=dict(maxfev=maxfev, xatol=1e-3, fatol=0.05,
                                          initial_simplex=np.vstack([np.r_[0, 0, 0, rpy]] + [np.r_[0, 0, 0, rpy] + np.eye(6)[i] * steps[i] for i in range(6)])))
                    for _, _, rpy in coarse[:refine_top]), key=lambda r: r.fun)
        rpy = (best.x[3:] + np.pi) % (2 * np.pi) - np.pi  # into [-pi, pi): the GUI sliders' range
        trial.write_text(set_origin(text, best.x[:3], rpy))
        tip, knuckle = fit_errors_mm(uv, trial, side, frames, knuckles)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"xyz": [round(float(v), 5) for v in best.x[:3]], "rpy": [round(float(v), 5) for v in rpy],
            "tip_error_mm": round(tip, 2), "knuckle_error_mm": round(knuckle, 2)}


def main(argv=None) -> None:
    """Calibrate the given sides of each spec, store the result in the spec and rebuild the hand."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unidex-root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, nargs="+", required=True)
    ap.add_argument("--sides", nargs="+", default=list(uh.SIDES), choices=uh.SIDES, help="run sides as separate processes to parallelise")
    ap.add_argument("--split", type=Path, default=REPO / "splits" / "h2o_50pct.json")
    ap.add_argument("--sequences", type=int, default=4, help="training sequences, evenly spread")
    ap.add_argument("--frames", type=int, default=4, help="frames per sequence")
    args = ap.parse_args(argv)
    train = json.loads(args.split.read_text())["train"]
    seqs = [train[i] for i in np.linspace(0, len(train) - 1, args.sequences).round().astype(int)]
    root, specs, results = args.unidex_root.resolve(), [s.resolve() for s in args.spec], {}
    with open_unidex(root) as (uv, dataset):
        frames = frames_for_calibration(dataset, seqs, args.frames)
        for spec_path in specs:
            spec = uh.read_spec(spec_path)
            results[spec_path] = {s: calibrate_side(uv, root / "HandAdapter/urdf/H2o" / spec["unidex_hand"] / s, s, frames,
                                                    {int(k): v for k, v in spec["sides"][s]["knuckles"].items()}) for s in args.sides}
            print(f"{spec['unidex_hand']}: {results[spec_path]}", flush=True)
    for spec_path, result in results.items():  # locked: the other side's process may finish at the same time
        with open(spec_path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            spec = uh.read_spec(spec_path)
            spec["base_joint_origin"] = {**(spec.get("base_joint_origin") or {}), **{s: {"xyz": r["xyz"], "rpy": r["rpy"]} for s, r in result.items()}}
            cal = spec.get("calibration") or {"sides": {}}
            cal["sides"].update({s: {k: r[k] for k in ("tip_error_mm", "knuckle_error_mm")} for s, r in result.items()})
            spec["calibration"] = {**cal, "frames": len(frames), "sequences": seqs}
            spec_path.write_text(json.dumps(spec, indent=1) + "\n")
            uh.build(spec_path, root)
        spec_path.with_suffix(".lock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
