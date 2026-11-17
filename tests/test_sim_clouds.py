"""Tests of the MuJoCo body clouds: placement against independently built transforms, area weighting, geom selection."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from lawm.sim import clouds  # noqa: E402

XML = """
<mujoco>
  <asset>
    <mesh name="cube" vertex="-1 -1 -1  -1 -1 1  -1 1 -1  -1 1 1  1 -1 -1  1 -1 1  1 1 -1  1 1 1" scale="0.03 0.02 0.01"/>
  </asset>
  <worldbody>
    <body name="boxed" pos="0.3 -0.2 0.5" euler="0 0 90">
      <joint type="free"/>
      <geom name="slab" type="box" size="0.05 0.02 0.01" pos="0.1 0 0" contype="1" conaffinity="1"/>
      <geom name="marker" type="box" size="0.2 0.2 0.2" pos="0 0 0.5" contype="0" conaffinity="0"/>
    </body>
    <body name="can" pos="-0.1 0.4 0.2" euler="0 90 0">
      <geom name="can_geom" type="cylinder" size="0.03 0.05"/>
    </body>
    <body name="meshed" pos="0 0 1" euler="30 0 0">
      <geom name="m" type="mesh" mesh="cube" pos="0.05 0 0"/>
    </body>
    <body name="ball" pos="0 0 2">
      <geom name="ball_geom" type="sphere" size="0.04"/>
    </body>
    <body name="table" pos="0 0 0.4">
      <geom name="table_top" type="box" size="0.4 0.3 0.02" pos="0 0 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def setup():
    """Compile the model and run forward kinematics."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def box_distance(points, centre, rot, half):
    """Unsigned distance of world points to the surface of a box with the given world centre, rotation matrix and half-extents."""
    q = np.abs((points - centre) @ rot) - half
    return np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0))


def rot_z(deg):
    """Rotation matrix about z."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_box_body_matches_hand_built_transform():
    """Points of an offset box on a body turned 90 degrees about z lie on the box placed by an independent transform."""
    model, data = setup()
    pts, _ = clouds.body_cloud(model, data, "boxed", 4000, np.random.default_rng(0), kind="collision")
    rot = rot_z(90)
    centre = np.array([0.3, -0.2, 0.5]) + rot @ np.array([0.1, 0, 0])
    assert box_distance(pts.astype(np.float64), centre, rot, np.array([0.05, 0.02, 0.01])).max() < 1e-6


def test_cylinder_body_matches_hand_built_transform():
    """Points of a cylinder whose axis was turned from z to x lie on its side wall or caps."""
    model, data = setup()
    pts, _ = clouds.body_cloud(model, data, "can", 4000, np.random.default_rng(0), kind="collision")
    local = pts - np.array([-0.1, 0.4, 0.2])  # euler y=90 sends the cylinder axis z to world x
    radial, axial = np.linalg.norm(local[:, [1, 2]], axis=1), np.abs(local[:, 0])
    on_wall, on_cap = np.isclose(radial, 0.03, atol=1e-3) & (axial <= 0.05 + 1e-6), np.isclose(axial, 0.05, atol=1e-6) & (radial <= 0.03 + 1e-6)
    assert (on_wall | on_cap).all()


def test_mesh_geom_matches_box_hull():
    """A scaled cube mesh on a body rotated 30 degrees about x is sampled on the surface of the same box placed by hand."""
    model, data = setup()
    pts, _ = clouds.body_cloud(model, data, "meshed", 4000, np.random.default_rng(0), kind="collision")
    a = np.radians(30)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    centre = np.array([0, 0, 1.0]) + rot @ np.array([0.05, 0, 0])
    assert box_distance(pts.astype(np.float64), centre, rot, np.array([0.03, 0.02, 0.01])).max() < 1e-6


def test_kind_selects_collision_or_visual_geoms():
    """The collision cloud uses only the slab and the visual cloud only the marker, as reported by the geom ids."""
    model, data = setup()
    slab, marker = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("slab", "marker"))
    assert set(clouds.body_cloud(model, data, "boxed", 500, np.random.default_rng(0), kind="collision")[1]) == {slab}
    assert set(clouds.body_cloud(model, data, "boxed", 500, np.random.default_rng(0), kind="visual")[1]) == {marker}


def test_points_are_split_by_surface_area():
    """The share of points on each face of a box matches the face's share of the surface area."""
    model, data = setup()
    pts, _ = clouds.body_cloud(model, data, "boxed", 60000, np.random.default_rng(0), kind="collision")
    local = (pts.astype(np.float64) - (np.array([0.3, -0.2, 0.5]) + rot_z(90) @ np.array([0.1, 0, 0]))) @ rot_z(90)
    half = np.array([0.05, 0.02, 0.01])
    areas = np.array([4 * half[1] * half[2], 4 * half[0] * half[2], 4 * half[0] * half[1]])  # both faces of each axis
    share = np.array([np.isclose(np.abs(local[:, a]), half[a], atol=1e-6).mean() for a in range(3)])
    assert np.allclose(share, areas / areas.sum(), atol=0.01)


def test_sphere_points_lie_on_the_sphere():
    """Sphere points are within the tessellation error of the sphere radius and none are inside it."""
    model, data = setup()
    pts, _ = clouds.body_cloud(model, data, "ball", 2000, np.random.default_rng(0), kind="collision")
    radius = np.linalg.norm(pts - np.array([0, 0, 2.0]), axis=1)
    assert radius.max() <= 0.04 + 1e-6 and radius.min() > 0.04 * np.cos(np.pi / 8) * np.cos(np.pi / 12) - 1e-6


def test_box_triangles_face_outward():
    """Every triangle normal of the box, cylinder and sphere tessellations points away from the centre."""
    for verts, faces in (clouds.box_triangles([0.05, 0.02, 0.01]), clouds.cylinder_triangles(0.03, 0.05), clouds.sphere_triangles(0.04)):
        tri = verts[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert ((normal * tri.mean(axis=1)).sum(axis=1) > 0).all()


def test_cloud_follows_the_body():
    """After the free joint moves the body, the cloud moves with it by the same translation."""
    model, data = setup()
    before, _ = clouds.body_cloud(model, data, "boxed", 500, np.random.default_rng(0), kind="collision")
    data.qpos[:3] += [0.1, 0.2, -0.05]
    mujoco.mj_forward(model, data)
    after, _ = clouds.body_cloud(model, data, "boxed", 500, np.random.default_rng(0), kind="collision")
    assert np.allclose(after - before, [0.1, 0.2, -0.05], atol=1e-5)


def test_same_seed_gives_same_cloud():
    """The same seed gives the same points and a different seed gives different ones."""
    model, data = setup()
    a, _ = clouds.body_cloud(model, data, "meshed", 300, np.random.default_rng(1), kind="collision")
    b, _ = clouds.body_cloud(model, data, "meshed", 300, np.random.default_rng(1), kind="collision")
    c, _ = clouds.body_cloud(model, data, "meshed", 300, np.random.default_rng(2), kind="collision")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_table_top_points_sit_on_the_top_face():
    """Table top points have z equal to the box centre plus its half height and stay inside its x and y extent."""
    model, data = setup()
    pts = clouds.box_top_cloud(model, data, "table_top", 1000, np.random.default_rng(0))
    assert np.allclose(pts[:, 2], 0.4 + 0.01 + 0.02) and (np.abs(pts[:, 0]) <= 0.4).all() and (np.abs(pts[:, 1]) <= 0.3).all()


def test_missing_body_or_unsupported_geom_is_an_error():
    """An unknown body name and an ellipsoid geom both fail loudly."""
    model, data = setup()
    with pytest.raises(AssertionError):
        clouds.body_cloud(model, data, "nope", 10, np.random.default_rng(0), kind="collision")
    ellipsoid = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><geom type="ellipsoid" size="0.1 0.1 0.2"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):
        clouds.body_cloud(ellipsoid, mujoco.MjData(ellipsoid), "b", 10, np.random.default_rng(0), kind="collision")


def test_geoms_cloud_covers_exactly_the_given_geoms():
    """A cloud over two chosen geoms of different bodies uses only those geoms."""
    model, data = setup()
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("slab", "can_geom")]
    _, owner = clouds.geoms_cloud(model, data, ids, 2000, np.random.default_rng(0))
    assert set(owner) == set(ids)


def test_surface_sampler_keeps_the_same_physical_points_when_the_body_moves():
    """Calling a fitted sampler after a rigid motion gives the earlier points moved by that motion, row for row."""
    model, data = setup()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "slab")
    sampler = clouds.SurfaceSampler(model, [gid], 300, np.random.default_rng(0))
    before = sampler(data)
    data.qpos[:3] += [0.3, -0.1, 0.2]
    mujoco.mj_forward(model, data)
    assert np.allclose(sampler(data) - before, [0.3, -0.1, 0.2], atol=1e-5)


def test_surface_sampler_matches_the_one_shot_cloud():
    """A sampler and geoms_cloud with the same seed give the same points."""
    model, data = setup()
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("slab", "can_geom")]
    a, _ = clouds.geoms_cloud(model, data, ids, 500, np.random.default_rng(4))
    assert np.array_equal(a, clouds.SurfaceSampler(model, ids, 500, np.random.default_rng(4))(data))
