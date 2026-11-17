"""Unit tests for lawm.preprocessing.h2o_readers, using tiny synthetic files."""
import numpy as np
import pytest

from lawm.preprocessing.h2o_readers import (
    first_valid_object_id,
    list_sequences,
    load_camera_to_world,
    load_depth_metres,
    load_intrinsics,
    load_object_pose,
    parse_mano_file,
    resolve_object_mesh,
)


def write_mano(path, left_tracked, right_tracked):
    """Write a one-row MANO text file with left and right blocks whose tracked flags are given"""
    block = lambda flag, base: np.r_[flag, np.full(3, base), np.arange(48) + base, np.full(10, base + 0.5)]
    np.savetxt(path, np.r_[block(left_tracked, 1.0), block(right_tracked, 2.0)][None], fmt="%.6f")


def test_parse_mano_file_both_hands(tmp_path):
    """parse_mano_file returns translation, pose and beta of the right shapes with the values of the correct hand"""
    write_mano(tmp_path / "f.txt", 1, 1)
    out = parse_mano_file(tmp_path / "f.txt")
    trans, pose, beta = out["right"]
    assert trans.shape == (3,) and pose.shape == (48,) and beta.shape == (10,)
    assert np.allclose(trans, 2.0) and np.allclose(pose[0], 2.0) and np.allclose(beta, 2.5)
    assert np.allclose(out["left"][0], 1.0)


def test_parse_mano_file_untracked_side_is_none(tmp_path):
    """An untracked hand is returned as None while the tracked one is kept"""
    write_mano(tmp_path / "f.txt", 1, 0)
    out = parse_mano_file(tmp_path / "f.txt")
    assert out["left"] is not None and out["right"] is None


def test_parse_mano_file_rejects_wrong_size(tmp_path):
    """parse_mano_file raises ValueError for a file with the wrong number of values"""
    (tmp_path / "f.txt").write_text("1 2 3")
    with pytest.raises(ValueError):
        parse_mano_file(tmp_path / "f.txt")


def test_load_camera_to_world_is_row_major(tmp_path):
    """load_camera_to_world reads the 16 values as a row-major 4x4 matrix"""
    (tmp_path / "cam_pose").mkdir()
    (tmp_path / "cam_pose" / "000003.txt").write_text(" ".join(map(str, range(16))))
    assert np.array_equal(load_camera_to_world(tmp_path, 3), np.arange(16, dtype=float).reshape(4, 4))


@pytest.mark.parametrize("values,expected_id", [(np.r_[7, np.arange(16)], 7), (np.arange(16), None)])
def test_load_object_pose_with_and_without_id(tmp_path, values, expected_id):
    """load_object_pose reads a 4x4 pose with or without a leading object id, returning None for the id when absent"""
    (tmp_path / "obj_pose_rt").mkdir()
    (tmp_path / "obj_pose_rt" / "000000.txt").write_text(" ".join(map(str, values)))
    T, obj_id = load_object_pose(tmp_path, 0)
    assert T.shape == (4, 4) and obj_id == expected_id and T[0, 0] == 0 and T[3, 3] == 15


def test_load_object_pose_rejects_other_sizes(tmp_path):
    """load_object_pose raises ValueError for a file with an unexpected number of values"""
    (tmp_path / "obj_pose_rt").mkdir()
    (tmp_path / "obj_pose_rt" / "000000.txt").write_text("1 2 3")
    with pytest.raises(ValueError):
        load_object_pose(tmp_path, 0)


def test_resolve_object_mesh(tmp_path):
    """resolve_object_mesh finds each object's mesh file, including names that differ from the folder, and raises for unknown, missing or None ids"""
    (tmp_path / "book").mkdir(); (tmp_path / "book" / "book.obj").write_text("")
    (tmp_path / "spray").mkdir(); (tmp_path / "spray" / "lotion_spray.obj").write_text("")  # name differs from folder
    assert resolve_object_mesh(tmp_path, 1) == tmp_path / "book" / "book.obj"
    assert resolve_object_mesh(tmp_path, 4) == tmp_path / "spray" / "lotion_spray.obj"
    with pytest.raises(ValueError):
        resolve_object_mesh(tmp_path, 99)
    with pytest.raises(ValueError):
        resolve_object_mesh(tmp_path, None)
    with pytest.raises(FileNotFoundError):
        resolve_object_mesh(tmp_path, 5)  # milk folder absent


def test_load_intrinsics_returns_fx_fy_cx_cy(tmp_path):
    """load_intrinsics returns only fx, fy, cx and cy from the intrinsics file"""
    (tmp_path / "cam_intrinsics.txt").write_text("600 601 320 240 640 480")
    assert load_intrinsics(tmp_path) == (600.0, 601.0, 320.0, 240.0)


def test_load_depth_metres_converts_millimetres(tmp_path):
    """load_depth_metres converts a 16-bit millimetre depth image to float32 metres"""
    Image = pytest.importorskip("PIL.Image")
    (tmp_path / "depth").mkdir()
    Image.fromarray(np.array([[0, 1500], [250, 65535]], dtype=np.uint16)).save(tmp_path / "depth" / "000000.png")
    d = load_depth_metres(tmp_path, 0)
    assert d.dtype == np.float32 and np.allclose(d, [[0.0, 1.5], [0.25, 65.535]])


def test_first_valid_object_id_skips_files_without_an_id(tmp_path):
    """first_valid_object_id skips pose files that have no id and raises when none has one"""
    (tmp_path / "obj_pose_rt").mkdir()
    (tmp_path / "obj_pose_rt" / "000000.txt").write_text(" ".join(map(str, np.arange(16))))       # no id
    (tmp_path / "obj_pose_rt" / "000001.txt").write_text(" ".join(map(str, np.r_[5, np.arange(16)])))
    assert first_valid_object_id(tmp_path, [0, 1]) == 5
    with pytest.raises(RuntimeError):
        first_valid_object_id(tmp_path, [0])


def test_list_sequences_orders_by_path_and_skips_dirs_without_mano(tmp_path):
    """list_sequences returns sequences in path order with their running index and skips directories lacking MANO files"""
    for rel, has_mano in [("subject1/h1/0", True), ("subject1/h1/1", False), ("subject2/k1/0", True)]:
        cam4 = tmp_path / rel / "cam4"
        (cam4 / "hand_pose_mano").mkdir(parents=True)
        if has_mano:
            (cam4 / "hand_pose_mano" / "000000.txt").write_text("0")
    found = list_sequences(tmp_path)
    assert [(i, p.relative_to(tmp_path).as_posix()) for i, p in found] == [
        (1, "subject1/h1/0/cam4"), (3, "subject2/k1/0/cam4")]  # the skipped directory still uses up index 2
