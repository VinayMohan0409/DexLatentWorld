"""UniDex hand folder building from a spec: URDF rewriting, added tips, wrist-offset round trip, GUI registration."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from lawm.embodiments import unidex_hand as uh  # noqa: E402
from lawm.embodiments.calibrate import axis_rotations, set_origin  # noqa: E402

URDF = """<robot name="toy">
  <link name="palm"><visual><geometry><mesh filename="package://toy_description/meshes/palm.stl"/></geometry></visual></link>
  <link name="mount"><visual><geometry><mesh filename="package://toy_description/meshes/palm.stl"/></geometry></visual></link>
  <link name="finger"><visual><origin xyz="0 0 0.01" rpy="0 0 0"/><geometry><mesh filename="package://toy_description/meshes/rod.stl"/></geometry></visual></link>
  <joint name="j_mount" type="fixed"><parent link="palm"/><child link="mount"/></joint>
  <joint name="j1" type="revolute"><parent link="palm"/><child link="finger"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="palm"/><child link="finger2"/><mimic joint="j1" multiplier="0.5" offset="0.1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
  <link name="finger2"/>
</robot>"""


@pytest.fixture
def spec(tmp_path):
    """A toy source tree and a spec that drops the mount, adds a tip at the end of the finger and maps package:// paths."""
    repo = tmp_path / "repo"
    (repo / "toy_description" / "urdf").mkdir(parents=True)
    (repo / "toy_description" / "meshes").mkdir()
    (repo / "toy_description" / "urdf" / "toy.urdf").write_text(URDF)
    trimesh.creation.box((0.1, 0.1, 0.02)).export(repo / "toy_description" / "meshes" / "palm.stl")
    rod = trimesh.creation.box((0.01, 0.01, 0.05))
    rod.apply_translation([0, 0, 0.025])  # spans z 0..0.05 in the mesh frame, 0.01..0.06 in the link frame
    rod.export(repo / "toy_description" / "meshes" / "rod.stl")
    side = {"urdf": "toy_description/urdf/toy.urdf", "mesh_root": ".", "add_tips": {"tip": {"parent": "finger"}}, "_repo_root": str(repo)}
    return repo, {"unidex_hand": "Toy", "package_paths": {"package://toy_description/": "toy_description/"}, "drop_visuals": ["mount"],
                  "sides": {"left": dict(side), "right": dict(side)}, "tips": ["tip"]}


def test_side_urdf_has_unidex_base_relative_meshes_tip_and_mimic(spec, tmp_path):
    """base/base_joint wrap the root, meshes are copied and relative, the mount loses its visual, the tip sits at the mesh end"""
    repo, s = spec
    out = tmp_path / "out" / "right"
    mimic = uh.build_side_urdf(repo / s["sides"]["right"]["urdf"], out, s, "right", {"xyz": [0.1, 0, 0], "rpy": [0, 0, 1]})
    root = ET.parse(out / "main.urdf").getroot()
    assert root.findall("link")[0].get("name") == "base"
    assert uh.read_base_joint_origin(out / "main.urdf") == {"xyz": [0.1, 0, 0], "rpy": [0, 0, 1]}
    files = [m.get("filename") for m in root.iter("mesh")]
    assert files and all(f.startswith("meshes/") and (out / f).exists() for f in files)
    assert not [l for l in root.findall("link") if l.get("name") == "mount"][0].findall("visual")
    tip_joint = [j for j in root.findall("joint") if j.get("name") == "tip_joint"][0]
    xyz = np.array([float(v) for v in tip_joint.find("origin").get("xyz").split()])
    assert np.isclose(np.linalg.norm(xyz), np.linalg.norm([0.005, 0.005, 0.06]), atol=1e-6)  # a far corner of the rod
    assert mimic == {"j2": ["j1", 0.5, 0.1]}
    assert uh.hand_config(s, mimic)["poses"] == ["base", "tip"]


def test_gui_offsets_round_trip_into_the_spec(spec, tmp_path):
    """pull-gui stores what the GUI wrote into the H2o URDFs back into the spec file"""
    repo, s = spec
    unidex = tmp_path / "unidex"
    for side in uh.SIDES:
        d = unidex / "HandAdapter" / "urdf" / "H2o" / "Toy" / side
        uh.build_side_urdf(repo / s["sides"][side]["urdf"], d, s, side, {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
        (d / "main.urdf").write_text(set_origin((d / "main.urdf").read_text(), [0.01, 0.02, 0.03], [0.1, 0.2, 0.3]))
    path = tmp_path / "toy.json"
    path.write_text(json.dumps({k: v for k, v in s.items()}))
    got = uh.pull_gui(path, unidex)
    assert got["left"] == {"xyz": [0.01, 0.02, 0.03], "rpy": [0.1, 0.2, 0.3]}
    assert json.loads(path.read_text())["base_joint_origin"] == got


def test_gui_registration_is_idempotent(tmp_path):
    """the hand is appended to HAND_TYPES once"""
    vis = tmp_path / "HandAdapter" / "visualizer.py"
    vis.parent.mkdir(parents=True)
    vis.write_text('x = 1\nHAND_TYPES = ["Inspire", "Leap"]\ny = 2\n')
    uh.register_in_gui(tmp_path, "Toy")
    uh.register_in_gui(tmp_path, "Toy")
    assert vis.read_text() == 'x = 1\nHAND_TYPES = ["Inspire", "Leap", "Toy"]\ny = 2\n'


def test_there_are_24_axis_rotations():
    """the coarse search covers every axis-aligned proper rotation once"""
    from scipy.spatial.transform import Rotation as R
    mats = {tuple(np.round(R.from_euler("xyz", r).as_matrix()).astype(int).ravel()) for r in axis_rotations()}
    assert len(mats) == 24
