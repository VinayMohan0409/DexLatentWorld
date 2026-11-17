"""Unit tests for the refined-MANO file format in lawm.preprocessing.contact_refinement.mano_refined."""
import json

import numpy as np
import pytest

from lawm.preprocessing.contact_refinement.mano_refined import (
    atomic_json,
    read_refined_frame,
    validate_complete,
    write_refined_frame,
)

RIGHT = (np.array([0.1, 0.2, 0.3]), np.linspace(-1, 1, 48), np.linspace(0, 1, 10))


def test_roundtrip_keeps_values_and_marks_untracked_side(tmp_path):
    """A written refined frame reads back as float64 values equal to the originals within float32 precision, with the untracked side None"""
    write_refined_frame(tmp_path / "000005.npz", 5, {"right": RIGHT, "left": None})
    out = read_refined_frame(tmp_path / "000005.npz")
    assert out["left"] is None
    for stored, original in zip(out["right"], RIGHT):
        assert stored.dtype == np.float64 and np.allclose(stored, original, atol=1e-6)  # stored as float32


def test_frame_id_must_match_filename(tmp_path):
    """Reading a frame whose stored id differs from its filename raises ValueError"""
    write_refined_frame(tmp_path / "000006.npz", 5, {"right": RIGHT, "left": None})
    with pytest.raises(ValueError):
        read_refined_frame(tmp_path / "000006.npz")


def test_nan_is_rejected(tmp_path):
    """Reading a frame containing NaN raises ValueError"""
    bad = (RIGHT[0] * np.nan, RIGHT[1], RIGHT[2])
    write_refined_frame(tmp_path / "000000.npz", 0, {"right": bad, "left": None})
    with pytest.raises(ValueError):
        read_refined_frame(tmp_path / "000000.npz")


def test_missing_frame_is_an_error(tmp_path):
    """Reading a non-existent frame raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        read_refined_frame(tmp_path / "000000.npz")


def make_refined_dir(root, complete_hash, meta_hash):
    """Create a mano_refined directory with one frame, a completion marker and metadata carrying the given config hashes"""
    d = root / "mano_refined"
    d.mkdir()
    write_refined_frame(d / "000000.npz", 0, {"right": RIGHT, "left": None})
    atomic_json(d / "complete.json", {"status": "complete", "config_hash": complete_hash})
    atomic_json(d / "metadata.json", {"config_hash": meta_hash, "frame_stems": ["000000"]})


def test_validate_complete_accepts_consistent_directory(tmp_path):
    """validate_complete returns the metadata for a directory whose hashes agree"""
    make_refined_dir(tmp_path, "abc", "abc")
    assert validate_complete(tmp_path)["frame_stems"] == ["000000"]


def test_validate_complete_rejects_hash_mismatch(tmp_path):
    """validate_complete raises ValueError when the completion and metadata config hashes differ"""
    make_refined_dir(tmp_path, "abc", "xyz")
    with pytest.raises(ValueError):
        validate_complete(tmp_path)


def test_atomic_json_leaves_no_temp_files(tmp_path):
    """atomic_json writes the JSON content and leaves no temporary file behind"""
    atomic_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
