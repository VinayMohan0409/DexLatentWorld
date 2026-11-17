"""Unit tests for lawm.preprocessing.contact_refinement.sequence on a tiny synthetic cam4 directory."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from lawm.preprocessing.contact_refinement.sequence import apply_homogeneous, load_sequence  # noqa: E402

N_FRAMES = 4


def mano_row(left, right):
    """Build one MANO text row from left and right tracked flags with a fixed translation per hand"""
    block = lambda flag, t: np.r_[flag, t, np.zeros(48), np.zeros(10)]
    return np.r_[block(left, [0.0, 0.0, 0.5]), block(right, [0.1, 0.0, 0.5])]


@pytest.fixture
def cam4(tmp_path):
    """Fixture that writes a four-frame cam4 directory with MANO, camera and object poses plus a box mesh, with the right hand untracked in frame 1"""
    seq, objects = tmp_path / "cam4", tmp_path / "objects"
    for sub in ("hand_pose_mano", "cam_pose", "obj_pose_rt"):
        (seq / sub).mkdir(parents=True)
    (objects / "book").mkdir(parents=True)
    trimesh.creation.box().export(objects / "book" / "book.obj")
    right_tracked = [1, 0, 1, 1]  # the right hand is untracked in frame 1: a tracking gap
    for f in range(N_FRAMES):
        np.savetxt(seq / "hand_pose_mano" / f"{f:06d}.txt", mano_row(1, right_tracked[f])[None], fmt="%.6f")
        np.savetxt(seq / "cam_pose" / f"{f:06d}.txt", np.eye(4).reshape(1, 16))
        np.savetxt(seq / "obj_pose_rt" / f"{f:06d}.txt", np.r_[1, np.eye(4).ravel()][None])
    return seq, objects


def test_load_sequence_shapes_and_sides(cam4):
    """load_sequence returns both sides with frame ids and hand and object arrays of the expected shapes"""
    seq, objects = cam4
    out = load_sequence(seq, objects)
    assert set(out) == {"right", "left"}
    assert out["left"].frame_ids.tolist() == [0, 1, 2, 3]
    assert out["left"].hand_pose.shape == (4, 45) and out["left"].hand_rot.shape == (4, 3)
    assert out["left"].hand_mTc_world.shape == (4, 4, 4) and out["left"].hand_beta.shape == (10,)
    assert out["left"].obj_verts.shape == (4, 8, 3) and out["left"].obj_faces.shape == (12, 3)


def test_tracking_gap_leaves_frame_ids_non_consecutive(cam4):
    # Documents the original behaviour: the right hand's frames are stacked as [0, 2, 3], so the
    # optimiser's temporal smoothing treats frames 0 and 2 as neighbours.
    """A tracking gap leaves the right hand's frame ids as [0, 2, 3], documenting the original behaviour"""
    seq, objects = cam4
    assert load_sequence(seq, objects)["right"].frame_ids.tolist() == [0, 2, 3]


def test_hand_placement_uses_camera_pose_and_translation(cam4):
    """The hand's world placement takes the MANO translation through the camera pose"""
    seq, objects = cam4
    right = load_sequence(seq, objects)["right"]
    assert np.allclose(right.hand_mTc_world[0, :3, 3], [0.1, 0.0, 0.5])


def test_no_tracked_hand_is_an_error(tmp_path):
    """load_sequence raises ValueError when no hand is tracked"""
    seq = tmp_path / "cam4"
    (seq / "hand_pose_mano").mkdir(parents=True)
    np.savetxt(seq / "hand_pose_mano" / "000000.txt", mano_row(0, 0)[None], fmt="%.6f")
    with pytest.raises(ValueError):
        load_sequence(seq, tmp_path)


def test_apply_homogeneous_matches_rotation_translation_form():
    """apply_homogeneous equals applying the rotation and then the translation explicitly"""
    rng = np.random.default_rng(0)
    pts = rng.random((6, 3))
    T = np.eye(4)
    T[:3, :3] = np.linalg.qr(rng.random((3, 3)))[0]
    T[:3, 3] = [1, 2, 3]
    assert np.allclose(apply_homogeneous(pts, T), pts @ T[:3, :3].T + T[:3, 3])
