"""Unit tests for lawm.preprocessing.scene_geometry on small synthetic scenes."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from lawm.preprocessing.scene_geometry import (  # noqa: E402
    EMBODIMENT_NAMES,
    fit_table,
    pick_frames,
    ransac_plane,
    roll_distractors,
    rot_about_axis,
    rotation_between,
    sequence_tag,
    unproject_to_world,
    uv_bounds,
)


def test_sequence_tag_keeps_the_unidex_ego_spelling():
    """sequence_tag uses the UniDex ego spelling and the embodiment names start with human and number nine"""
    assert sequence_tag("subject1", "h1", "0") == "subject1_ego_h1_0_cam4"
    assert EMBODIMENT_NAMES[0] == "human" and len(EMBODIMENT_NAMES) == 9


def test_pick_frames():
    """pick_frames returns all frames when few and evenly spaced frames otherwise"""
    assert pick_frames(5, 8) == [0, 1, 2, 3, 4]
    assert pick_frames(100, 3) == [0, 50, 99] or pick_frames(100, 3) == [0, 49, 99]  # evenly spaced, rounded


def test_rotations():
    """rot_about_axis and rotation_between give proper rotations that map the source direction onto the target, including opposite directions"""
    R = rot_about_axis(np.array([0.0, 0.0, 2.0]), np.pi / 2)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    a, b = np.array([1.0, 2.0, 3.0]), np.array([-3.0, 0.5, 1.0])
    R = rotation_between(a, b)
    assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(rotation_between(a, -a) @ (a / np.linalg.norm(a)), -a / np.linalg.norm(a))  # opposite directions


def test_unproject_places_points_on_a_flat_floor():
    """unproject_to_world places a flat depth image at its depth and drops pixels beyond the cutoff"""
    H, W, fx = 48, 64, 50.0
    depth = np.full((H, W), 1.0, np.float32)           # a fronto-parallel wall one metre away
    pts = unproject_to_world(depth, fx, fx, W / 2, H / 2, np.eye(4), stride=4)
    assert np.allclose(pts[:, 2], 1.0) and len(pts) == (H // 4) * (W // 4)
    far = np.full((H, W), 2.0, np.float32)  # beyond the 1.15 m cutoff, so every pixel is dropped
    assert len(unproject_to_world(far, fx, fx, W / 2, H / 2, np.eye(4))) == 0


def test_ransac_plane_recovers_a_noisy_plane_and_ignores_outliers():
    """ransac_plane finds the noisy z=0 plane and marks most of its points as inliers despite outliers"""
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(-1, 1, (500, 2)), rng.normal(0, 0.001, 500)]   # z ~ 0
    outliers = rng.uniform(-1, 1, (60, 3)) + [0, 0, 0.5]
    n, d, mask, _, _ = ransac_plane(np.r_[plane, outliers], seed=1)
    assert abs(abs(n[2]) - 1.0) < 1e-2 and abs(d) < 5e-3 and mask[:500].mean() > 0.95


def test_ransac_plane_is_deterministic():
    """ransac_plane gives the same plane for the same seed"""
    pts = np.random.default_rng(3).uniform(-1, 1, (300, 3)) * [1, 1, 0.01]
    a, b = ransac_plane(pts, seed=7), ransac_plane(pts, seed=7)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_uv_bounds():
    """uv_bounds returns the min and max in-plane coordinates relative to the centre, and None for no points"""
    u, v, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 0])
    lo, hi = uv_bounds(np.array([[2, 3, 9], [0, 0, -4]], float), c, u, v)
    assert np.allclose(lo, [-1, -1]) and np.allclose(hi, [1, 2])
    assert uv_bounds(np.zeros((0, 3)), c, u, v) == (None, None)


def test_fit_table_on_a_synthetic_depth_scene(tmp_path):
    """fit_table recovers a horizontal table from synthetic depth images, with sampled points on the plane, an upward normal and a plausible extent"""
    Image = pytest.importorskip("PIL.Image")
    cam = tmp_path
    (cam / "depth").mkdir()
    (cam / "cam_pose").mkdir()
    H, W, fx = 120, 160, 100.0
    # camera 0.8 m above a horizontal table, looking straight down: world z up, camera z = -world z
    T_wc = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.8], [0, 0, 0, 1.0]])
    for f in range(2):
        Image.fromarray(np.full((H, W), 800, np.uint16)).save(cam / "depth" / f"{f:06d}.png")
        np.savetxt(cam / "cam_pose" / f"{f:06d}.txt", T_wc.reshape(1, 16))
    pts, n, d, u, v, center, width, height, ratio = fit_table(
        cam, [0, 1], fx, fx, W / 2, H / 2, np.array([0.0, 0.0, 0.1]), 256, 0.05, seed=0)
    assert pts.shape == (256, 3) and np.allclose(pts[:, 2], 0.0, atol=1e-3)     # points lie on the table plane
    assert n[2] > 0.99 and ratio > 0.99                                           # normal points toward the scene
    assert width > 0.5 and height > 0.3


@pytest.fixture
def objects(tmp_path):
    """Fixture that writes three small tetrahedron meshes as object files"""
    for name, scale in (("book", 0.08), ("milk", 0.10), ("chips", 0.06)):
        (tmp_path / name).mkdir()
        trimesh.Trimesh(vertices=scale * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
                        faces=[[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]).export(tmp_path / name / f"{name}.obj")
    return tmp_path  # regular tetrahedra: a vertex or edge is the easy wrong way to rest one


def test_roll_distractors_rest_flush_on_a_face_outside_the_keep_out_box(objects):
    """roll_distractors places objects flat on the table outside the keep-out box, deterministically for a given seed"""
    n, d = np.array([0.0, 0.0, 1.0]), 0.0                       # table: the z = 0 plane
    u, v, center = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3)
    lo, hi = np.array([-0.1, -0.1]), np.array([0.1, 0.1])
    clouds = roll_distractors(objects, {1}, 3, n, d, center, u, v, 1.0, 1.0, lo, hi, 0.05, 0.1, 400, seed=4)
    assert 1 <= len(clouds) <= 3
    for c in clouds:
        assert c.shape == (400, 3) and c.dtype == np.float32 and abs(c[:, 2].min()) < 1e-5
        assert (c[:, 2] < 2e-3).mean() > 0.12            # a whole face (about a quarter of the surface) lies on the table
        centroid = c.mean(0)
        assert not (lo[0] - 0.05 <= centroid[0] <= hi[0] + 0.05 and lo[1] - 0.05 <= centroid[1] <= hi[1] + 0.05)
    again = roll_distractors(objects, {1}, 3, n, d, center, u, v, 1.0, 1.0, lo, hi, 0.05, 0.1, 400, seed=4)
    assert all(np.array_equal(a, b) for a, b in zip(clouds, again))   # deterministic for a given seed
