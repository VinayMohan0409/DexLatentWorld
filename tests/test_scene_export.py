"""Unit tests for lawm.preprocessing.scene_export and regenerate_scene on a tiny synthetic sequence."""
import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
Image = pytest.importorskip("PIL.Image")

from lawm.preprocessing.regenerate_scene import regenerate_sequence  # noqa: E402
from lawm.preprocessing.scene_export import is_complete, read_hand_clouds, write_sequence  # noqa: E402

N_FRAMES, E, PTS = 4, 9, 64
TETRA = 0.06 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def make_scene(rng):
    """Build a random scene dict with table, object, distractor and embodiment arrays for N_FRAMES frames"""
    return {
        "table": rng.random((PTS, 3), dtype=np.float32), "object_canonical": rng.random((PTS, 3), dtype=np.float32),
        "object_id": 1, "plane_n": np.array([0, 0, 1], np.float32), "plane_d": np.float32(0.0),
        "table_uvc": np.eye(3, dtype=np.float32), "cam_poses": np.tile(np.eye(4, dtype=np.float32), (N_FRAMES, 1, 1)),
        "distractor_points": [rng.random((k, PTS, 3), dtype=np.float32) for k in (1, 2, 3, 1, 2, 3, 1, 2, 3)],
        "distractor_counts": np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], np.int32),
        "embodiment_names": np.array(["human"] + [f"h{i}" for i in range(8)]), "n_frames": np.int32(N_FRAMES),
        "object_world_pose": np.tile(np.eye(4), (N_FRAMES, 1, 1)),
    }


def test_write_then_read_round_trip_and_atomic_publish(tmp_path):
    """write_sequence output is complete, leaves no temp folder and round-trips hands, validity, distractors, metadata and dtypes"""
    rng = np.random.default_rng(0)
    hands = rng.random((N_FRAMES, E, PTS, 3), dtype=np.float32)
    valid = rng.random((N_FRAMES, E)) > 0.2
    out = tmp_path / "subject1" / "h1" / "0"
    out.parent.mkdir(parents=True)
    write_sequence(out, make_scene(rng), hands, valid, {"note": "x"})
    assert is_complete(out) and not list(out.parent.glob("*.tmp*"))          # temp folder was renamed away
    got_hands, got_valid = read_hand_clouds(out)
    assert np.array_equal(got_hands, hands) and np.array_equal(got_valid, valid)
    seq = np.load(out / "sequence.npz", allow_pickle=True)
    assert [len(c) for c in seq["distractor_points"]] == [1, 2, 3, 1, 2, 3, 1, 2, 3] and seq["distractor_points"].shape == (E,)
    meta = json.loads((out / "export_metadata.json").read_text())
    assert meta["frames"] == N_FRAMES and meta["note"] == "x" and meta["embodiments"][0] == "human"
    with np.load(out / "000002.npz") as z:
        assert z["object_pose"].dtype == np.float32 and z["hands"].shape == (E, PTS, 3)


def test_read_hand_clouds_ignores_non_frame_files_and_empty_dirs(tmp_path):
    """read_hand_clouds raises on an empty directory and reads only the numbered frame files"""
    with pytest.raises(FileNotFoundError):
        read_hand_clouds(tmp_path)
    np.savez(tmp_path / "000000.npz", hands=np.zeros((E, PTS, 3), np.float32), hand_valid=np.ones(E, bool))
    np.savez(tmp_path / "sequence.npz", table=np.zeros(3))
    (tmp_path / "sample_human.ply").write_text("x")
    assert read_hand_clouds(tmp_path)[0].shape == (1, E, PTS, 3)


@pytest.fixture
def h2o(tmp_path):
    """A one-sequence H2O tree: raw cam4 files, stored hand clouds and one object mesh."""
    root = tmp_path / "H2O"
    cam = root / "subject1" / "h1" / "0" / "cam4"
    for sub in ("depth", "cam_pose", "obj_pose_rt", "hand_pose_mano", "pointclouds_world"):
        (cam / sub).mkdir(parents=True)
    (cam / "cam_intrinsics.txt").write_text("100 100 80 60 160 120")
    T_wc = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.8], [0, 0, 0, 1.0]])  # camera 0.8 m above a flat table
    rng = np.random.default_rng(1)
    hands = (rng.random((N_FRAMES, E, 2048, 3)) * 0.05 + [0.0, 0.0, 0.1]).astype(np.float32)
    for f in range(N_FRAMES):
        Image.fromarray(np.full((120, 160), 800, np.uint16)).save(cam / "depth" / f"{f:06d}.png")
        np.savetxt(cam / "cam_pose" / f"{f:06d}.txt", T_wc.reshape(1, 16))
        T_co = np.eye(4)
        T_co[:3, 3] = [0.0, 0.0, 0.7]
        np.savetxt(cam / "obj_pose_rt" / f"{f:06d}.txt", np.r_[1, T_co.ravel()][None])
        np.savetxt(cam / "hand_pose_mano" / f"{f:06d}.txt", np.zeros((1, 124)))
        np.savez(cam / "pointclouds_world" / f"{f:06d}.npz", hands=hands[f], hand_valid=np.ones(E, bool),
                 object_pose=np.eye(4, dtype=np.float32))
    objects = tmp_path / "objects"
    for name in ("book", "milk", "chips", "cocoa"):
        (objects / name).mkdir(parents=True)
        trimesh.Trimesh(TETRA, [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]).export(objects / name / f"{name}.obj")
    return root, cam, objects


def test_regenerate_sequence_copies_hands_and_is_deterministic(h2o, tmp_path):
    """regenerate_sequence copies the stored hands unchanged, skips a complete output, is deterministic and records the scene fit and distractor rule"""
    root, cam, objects = h2o
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    assert regenerate_sequence(cam, objects, out1, root) == "done"
    assert regenerate_sequence(cam, objects, out1, root) == "skipped"                     # already complete
    regenerate_sequence(cam, objects, out2, root)
    hands_in, valid_in = read_hand_clouds(cam / "pointclouds_world")
    hands_out, valid_out = read_hand_clouds(out1)
    assert np.array_equal(hands_in, hands_out) and np.array_equal(valid_in, valid_out)    # hands are copied unchanged
    a, b = np.load(out1 / "sequence.npz", allow_pickle=True), np.load(out2 / "sequence.npz", allow_pickle=True)
    for key in ("table", "object_canonical", "plane_n", "distractor_counts"):
        assert np.array_equal(a[key], b[key])
    assert all(np.array_equal(x, y) for x, y in zip(a["distractor_points"], b["distractor_points"]))
    assert int(a["object_id"]) == 1 and abs(float(a["plane_n"][2])) > 0.99 and abs(float(a["plane_d"])) < 1e-3
    meta = json.loads((out1 / "export_metadata.json").read_text())
    assert "convex-hull face" in meta["distractor_rest_rule"]
