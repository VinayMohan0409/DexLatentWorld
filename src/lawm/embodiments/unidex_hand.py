"""Build a UniDex hand folder from a spec (configs/hands/<name>.json): `build` writes it, `pull-gui` saves GUI-tuned wrist offsets back."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SIDES = ("left", "right")
CACHE = Path.home() / ".cache" / "lawm_hand_sources"


def fetch_source(source: dict, cache: Path = CACHE) -> Path:
    """Checkout of source['repo'] at source['commit'] (a cached shallow clone); refuses a checkout at another commit."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", source["repo"].rstrip("/").removesuffix(".git").split("github.com/")[-1])
    dest = Path(cache) / f"{name}_{source['commit'][:12]}"
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", source["repo"], source["commit"]], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != source["commit"]:
        raise RuntimeError(f"{dest} is at {head}, the spec pins {source['commit']}")
    return dest


def _origin(el: Optional[ET.Element]) -> np.ndarray:
    """4x4 transform of a URDF <origin> element (identity if absent)."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = R.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()  # URDF rpy = fixed-axis XYZ
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def farthest_mesh_point(link: ET.Element, mesh_dir: Path, from_point: np.ndarray) -> np.ndarray:
    """In the link frame: the vertex of the link's visual meshes farthest from from_point (the fingertip end of a distal link)."""
    import trimesh
    best, best_d = None, -1.0
    for vis in link.findall("visual"):
        m = vis.find("geometry/mesh")
        if m is None:
            continue
        mesh = trimesh.load(mesh_dir / m.get("filename"), force="mesh", process=False)
        v = np.asarray(mesh.vertices, np.float64) * np.array([float(s) for s in (m.get("scale") or "1 1 1").split()])
        v = v @ _origin(vis.find("origin"))[:3, :3].T + _origin(vis.find("origin"))[:3, 3]
        d = np.linalg.norm(v - from_point, axis=1)
        if d.max() > best_d:
            best, best_d = v[int(d.argmax())], float(d.max())
    if best is None:
        raise ValueError(f"link {link.get('name')} has no visual mesh to place a tip on")
    return best


def build_side_urdf(src_urdf: Path, out_dir: Path, spec: dict, side: str, origin: dict) -> Dict[str, List]:
    """Write out_dir/main.urdf (+ meshes/) for one side; returns that side's mimic relations {slave: [master, mult, offset]}."""
    side_spec = spec["sides"][side]
    root_el = ET.parse(src_urdf).getroot()
    repo_root = Path(side_spec["_repo_root"])
    mesh_src_root = repo_root / side_spec.get("mesh_root", str(Path(side_spec["urdf"]).parent))
    meshes_out = out_dir / "meshes"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    meshes_out.mkdir(parents=True)

    if spec.get("visuals_from_collision"):  # e.g. visuals only as .dae: use the (STL/OBJ) collision meshes as visuals
        for link in root_el.findall("link"):
            for el in link.findall("visual"):
                link.remove(el)
            for col in link.findall("collision"):
                vis = copy.deepcopy(col)
                vis.tag = "visual"
                link.insert(0, vis)
    drop = set(spec.get("drop_visuals", []))
    for link in root_el.findall("link"):
        if link.get("name") in drop:  # e.g. a mounting flange that is not part of the hand
            for el in link.findall("visual") + link.findall("collision"):
                link.remove(el)
    copied = {}
    for m in root_el.iter("mesh"):  # rewrite every mesh to meshes/<path relative to the mesh root>, copying the file
        fn = m.get("filename")
        for prefix, repl in spec.get("package_paths", {}).items():
            if fn.startswith(prefix):
                fn = repl + fn[len(prefix):]
        src = (mesh_src_root / fn) if not fn.startswith("/") else Path(fn)
        if not src.exists():
            src = repo_root / fn
        if not src.exists():
            raise FileNotFoundError(f"{side}: mesh {m.get('filename')} not found (tried {mesh_src_root / fn})")
        rel = copied.setdefault(src, Path("meshes") / src.relative_to(repo_root).as_posix().replace("/", "__"))
        if not (out_dir / rel).exists():
            shutil.copy2(src, out_dir / rel)
        m.set("filename", rel.as_posix())

    links = {l.get("name"): l for l in root_el.findall("link")}
    children = {j.find("child").get("link") for j in root_el.findall("joint")}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"{src_urdf}: expected one root link, found {roots}")
    for tip_name, tip in side_spec.get("add_tips", {}).items():  # fixed tip links at the far end of the distal link's mesh
        parent = links[tip["parent"]]
        xyz = farthest_mesh_point(parent, out_dir, np.zeros(3)) if tip.get("at", "mesh_end") == "mesh_end" else np.asarray(tip["xyz"], float)
        ET.SubElement(root_el, "link", name=tip_name)
        j = ET.SubElement(root_el, "joint", name=f"{tip_name}_joint", type="fixed")
        ET.SubElement(j, "parent", link=tip["parent"])
        ET.SubElement(j, "child", link=tip_name)
        ET.SubElement(j, "origin", xyz=" ".join(f"{v:.6f}" for v in xyz), rpy="0 0 0")

    base = ET.Element("link", name="base")
    bj = ET.Element("joint", name="base_joint", type="fixed")
    ET.SubElement(bj, "parent", link="base")
    ET.SubElement(bj, "child", link=roots[0])
    ET.SubElement(bj, "origin", xyz=" ".join(f"{v:g}" for v in origin["xyz"]), rpy=" ".join(f"{v:g}" for v in origin["rpy"]))
    root_el.insert(0, bj)
    root_el.insert(0, base)
    ET.indent(root_el)
    (out_dir / "main.urdf").write_text(ET.tostring(root_el, encoding="unicode") + "\n")

    mimic = {}
    for j in root_el.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            mimic[j.get("name")] = [m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0))]
    return mimic


def hand_config(spec: dict, mimic: Dict[str, List]) -> dict:
    """UniDex config.json: tips thumb first, poses = base then tips, mimic relations of both sides."""
    tips = spec["tips"]
    return {"base_joint": "base_joint", "tips": tips, "poses": ["base"] + tips, "mimic_relations": mimic}


def read_spec(path: Path) -> dict:
    """The hand spec JSON."""
    return json.loads(Path(path).read_text())


def build(spec_path: Path, unidex_root: Path, cache: Path = CACHE) -> Path:
    """Build urdf/base/<Hand> (origin zero) and urdf/H2o/<Hand> (spec origin) inside unidex_root; returns the H2o folder."""
    spec = read_spec(spec_path)
    repo = fetch_source(spec["source"], cache)
    hand = spec["unidex_hand"]
    urdf_root = Path(unidex_root) / "HandAdapter" / "urdf"
    zero = {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    out = None
    for variant in ("base", "H2o"):
        mimic = {}
        for side in SIDES:
            s = copy.deepcopy(spec)
            s["sides"][side]["_repo_root"] = str(repo)
            origin = zero if variant == "base" else (spec.get("base_joint_origin") or {}).get(side, zero)
            mimic.update(build_side_urdf(repo / s["sides"][side]["urdf"], urdf_root / variant / hand / side, s, side, origin))
        (urdf_root / variant / hand / "config.json").write_text(json.dumps(hand_config(spec, mimic), indent=4) + "\n")
        out = urdf_root / variant / hand
    register_in_gui(unidex_root, hand)
    return out


def register_in_gui(unidex_root: Path, hand: str) -> None:
    """Add the hand to UniDex's HAND_TYPES list (the GUI's hand menu); idempotent."""
    vis = Path(unidex_root) / "HandAdapter" / "visualizer.py"
    text = vis.read_text()
    m = re.search(r"^HAND_TYPES = \[(.*)\]$", text, re.M)
    names = [n.strip().strip("\"'") for n in m.group(1).split(",") if n.strip()]
    if hand not in names:
        names.append(hand)
        vis.write_text(text[:m.start()] + "HAND_TYPES = [" + ", ".join(f'"{n}"' for n in names) + "]" + text[m.end():])


def read_base_joint_origin(urdf: Path) -> dict:
    """xyz/rpy of the base_joint origin in a UniDex hand URDF."""
    for j in ET.parse(urdf).getroot().findall("joint"):
        if j.get("name") == "base_joint":
            o = j.find("origin")
            return {"xyz": [float(v) for v in o.get("xyz", "0 0 0").split()], "rpy": [float(v) for v in o.get("rpy", "0 0 0").split()]}
    raise ValueError(f"{urdf} has no base_joint")


def pull_gui(spec_path: Path, unidex_root: Path) -> dict:
    """Store the wrist offsets currently in urdf/H2o/<Hand>/<side>/main.urdf (the GUI writes there) into the spec."""
    spec = read_spec(spec_path)
    folder = Path(unidex_root) / "HandAdapter" / "urdf" / "H2o" / spec["unidex_hand"]
    spec["base_joint_origin"] = {side: read_base_joint_origin(folder / side / "main.urdf") for side in SIDES}
    Path(spec_path).write_text(json.dumps(spec, indent=1) + "\n")
    return spec["base_joint_origin"]


def main(argv=None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("build", "pull-gui"))
    ap.add_argument("--spec", type=Path, nargs="+", required=True)
    ap.add_argument("--unidex-root", type=Path, required=True)
    args = ap.parse_args(argv)
    for spec in args.spec:
        if args.command == "build":
            print(f"{spec}: built {build(spec, args.unidex_root)}")
        else:
            print(f"{spec}: stored {pull_gui(spec, args.unidex_root)}")


if __name__ == "__main__":
    main()
