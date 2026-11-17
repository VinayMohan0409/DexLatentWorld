"""Unit tests for lawm.preprocessing.sampling."""
import numpy as np
import pytest

from lawm.preprocessing.sampling import (
    apply_barycentric,
    presample_barycentric,
    split_points,
    stable_seed,
    transform_points,
)

# Two triangles in the z=0 plane: areas 0.5 and 2.0, plus one degenerate (zero-area) face.
VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0], [4, 2, 0], [2, 4, 0], [5, 5, 5]], float)
FACES = np.array([[0, 1, 2], [3, 4, 5], [6, 6, 6]])


def test_stable_seed_is_pinned_and_deterministic():
    # blake2b is process-independent, unlike hash(); the value is the original exporter's output.
    """stable_seed returns a pinned value, differs across arguments and stays within 32 bits"""
    assert stable_seed("human", "left", 1024) == 3695549632
    assert stable_seed("human", "left", 1024) != stable_seed("human", "right", 1024)
    assert 0 <= stable_seed("x") < 2**32


@pytest.mark.parametrize("total,parts,expected", [(2048, 2, [1024, 1024]), (7, 3, [3, 2, 2]), (5, 0, [])])
def test_split_points(total, parts, expected):
    """split_points divides the total as evenly as possible across parts and returns an empty list for zero parts"""
    assert split_points(total, parts) == expected
    assert sum(expected) == (total if parts > 0 else 0)


def test_transform_points_identity_and_translation():
    """transform_points leaves points unchanged for the identity and shifts them for a translation"""
    pts = np.random.default_rng(0).random((5, 3))
    T = np.eye(4)
    assert np.array_equal(transform_points(pts, T), pts)
    T[:3, 3] = [1, 2, 3]
    assert np.allclose(transform_points(pts, T), pts + [1, 2, 3])


def test_presample_is_repeatable_and_valid():
    """presample_barycentric repeats for a seed, gives valid barycentric weights and never selects a degenerate face"""
    f1, b1 = presample_barycentric(VERTS, FACES, 500, seed=7)
    f2, b2 = presample_barycentric(VERTS, FACES, 500, seed=7)
    assert np.array_equal(f1, f2) and np.array_equal(b1, b2)
    assert np.allclose(b1.sum(axis=1), 1.0) and (b1 >= 0).all() and (b1 <= 1).all()
    assert not (f1 == 2).any()  # degenerate face is never drawn


def test_presample_is_area_weighted():
    """Faces are drawn in proportion to their area"""
    f, _ = presample_barycentric(VERTS, FACES, 20000, seed=1)
    assert abs((f == 1).mean() - 0.8) < 0.02  # areas 2.0 : 0.5 -> 80% on the larger face


def test_apply_barycentric_lands_on_the_face_plane():
    """apply_barycentric returns points of the right shape lying in the plane of the faces"""
    f, b = presample_barycentric(VERTS, FACES, 200, seed=3)
    pts = apply_barycentric(VERTS, FACES, f, b)
    assert pts.shape == (200, 3) and np.allclose(pts[:, 2], 0.0)


def test_all_zero_area_mesh_raises():
    """presample_barycentric raises ValueError when every face has zero area"""
    with pytest.raises(ValueError):
        presample_barycentric(VERTS, np.array([[6, 6, 6]]), 10, seed=0)
