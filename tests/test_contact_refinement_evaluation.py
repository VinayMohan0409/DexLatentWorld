"""Unit tests for lawm.preprocessing.contact_refinement.evaluation on small synthetic geometry."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from lawm.preprocessing.contact_refinement.evaluation.metrics import (  # noqa: E402
    aggregate_chamfer,
    contact_chamfer_per_frame,
    object_vertex_normals,
    penetration_stats,
)
from lawm.preprocessing.contact_refinement.evaluation.report import FIELDS, build_rows, write_summary_csv  # noqa: E402

SPHERE = trimesh.creation.icosphere(subdivisions=2, radius=0.05)


def test_penetration_counts_only_vertices_behind_the_surface():
    """penetration_stats counts only hand vertices behind the object surface and reports the maximum depth, fractions and worst frame"""
    v0, n0 = SPHERE.vertices[0], SPHERE.vertex_normals[0]
    hand = np.stack([v0 - 0.005 * n0, v0 + 0.010 * n0, v0 + 0.5 * n0]).astype(np.float32)[None]  # inside, outside, far
    obj = SPHERE.vertices.astype(np.float32)[None]
    normals = SPHERE.vertex_normals.astype(np.float32)[None]
    out = penetration_stats(hand, obj, normals, np.array([7]), "cpu")
    assert out["max_penetration_m"] == pytest.approx(0.005, abs=1e-4)
    assert out["pct_penetrating_vertices"] == pytest.approx(1 / 3)
    assert out["pct_vertices_penetrating_over_allowance"] == pytest.approx(1 / 3)  # 5 mm > the 2 mm allowance
    assert out["worst_frame_h2o_id"] == 7 and len(out["per_frame_mean_m"]) == 1


def test_over_allowance_ignores_overlap_the_optimiser_tolerates():
    """The over-allowance fraction excludes shallow overlap that the optimiser tolerates"""
    v0, n0 = SPHERE.vertices[0], SPHERE.vertex_normals[0]
    hand = np.stack([v0 - 0.001 * n0, v0 - 0.005 * n0]).astype(np.float32)[None]  # 1 mm and 5 mm inside
    out = penetration_stats(hand, SPHERE.vertices.astype(np.float32)[None],
                            SPHERE.vertex_normals.astype(np.float32)[None], np.array([0]), "cpu")
    assert out["pct_penetrating_vertices"] == pytest.approx(1.0)
    assert out["pct_vertices_penetrating_over_allowance"] == pytest.approx(0.5)


def test_contact_chamfer_uses_only_masked_vertices():
    """contact_chamfer_per_frame measures distance to the surface only for vertices selected by the mask"""
    grid = np.stack(np.meshgrid(np.linspace(0, 0.2, 21), np.linspace(0, 0.2, 21)), -1).reshape(-1, 2)
    plane = np.c_[grid, np.zeros(len(grid))].astype(np.float32)[None]                       # object: z = 0 plane
    hand = np.array([[[0.05, 0.05, 0.02], [0.10, 0.10, 0.02], [0.10, 0.10, 0.90]]], np.float32)  # last is masked out
    out = contact_chamfer_per_frame(hand, plane, np.array([True, True, False]), "cpu")
    assert out["h2o_mean"][0] == pytest.approx(0.02, abs=1e-6)


def test_aggregate_chamfer_mask_and_empty_selection():
    """aggregate_chamfer ignores NaN frames, respects a frame mask and returns None for an empty selection"""
    per_frame = {"h2o_mean": np.array([0.01, 0.05, 0.09]), "sym_mean": np.array([0.02, np.nan, 0.08]),
                 "near_count": np.array([3.0, 0.0, 5.0])}
    all_frames = aggregate_chamfer(per_frame)
    assert all_frames["mean_contact_to_surface_m"] == pytest.approx(0.05) and all_frames["n_frames_total"] == 3
    assert all_frames["mean_symmetric_nearby_chamfer_m"] == pytest.approx(0.05)  # NaN frame ignored
    active = aggregate_chamfer(per_frame, mask=np.array([True, False, False]))
    assert active["mean_contact_to_surface_m"] == pytest.approx(0.01) and len(active["per_frame_mean_m"]) == 3
    empty = aggregate_chamfer(per_frame, mask=np.zeros(3, bool))
    assert empty["mean_contact_to_surface_m"] is None and empty["n_frames_total"] == 0


def test_object_vertex_normals_follow_the_rigid_motion():
    """object_vertex_normals matches normals recomputed from the mesh at each rigidly moved frame"""
    angles = [0.0, 0.4, 1.1]
    frames = []
    for a in angles:
        R = trimesh.transformations.rotation_matrix(a, [0.3, 0.5, 0.8])[:3, :3]
        frames.append(SPHERE.vertices @ R.T + [0.1, -0.2, 0.3])
    verts = np.stack(frames).astype(np.float32)
    got = object_vertex_normals(verts, SPHERE.faces)
    for t in range(len(angles)):
        expected = trimesh.Trimesh(vertices=verts[t], faces=SPHERE.faces, process=False).vertex_normals
        assert np.allclose(got[t], expected, atol=1e-4)


def metric_block(p95, pct, cham):
    """Build a metrics dict with penetration and chamfer entries filled from the given values"""
    return {"penetration": {"mean_frame_max_penetration_m": p95, "p95_frame_max_penetration_m": p95,
                            "max_penetration_m": p95, "pct_penetrating_vertices": pct},
            "chamfer": {"mean_contact_to_surface_m": cham, "p95_contact_to_surface_m": cham},
            "chamfer_active_frames_only": {"mean_contact_to_surface_m": cham, "p95_contact_to_surface_m": cham}}


def test_report_rows_and_csv(tmp_path):
    """build_rows turns a result JSON into one row per side and write_summary_csv writes the expected header columns"""
    result = {"manifest_index": 1, "subject": "subject1", "session_activity": "h1", "take": "0", "object_name": "book",
              "right": {"n_frames": 10, "n_active_frames": 4, "object_name": "book",
                        "raw": metric_block(0.02, 0.01, 0.03), "refined": metric_block(0.01, 0.03, 0.02)}}
    (tmp_path / "subject1_h1_0_cam4.json").write_text(json.dumps(result))
    rows = build_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["side"] == "right"
    assert rows[0]["raw_penetration_p95_frame_max_penetration_m"] == 0.02
    assert rows[0]["refined_penetration_pct_penetrating_vertices"] == 0.03
    write_summary_csv(rows, tmp_path / "summary.csv")
    header = (tmp_path / "summary.csv").read_text().splitlines()[0].split(",")
    assert "raw_chamfer_active_frames_only_mean_contact_to_surface_m" in header
    assert "raw_penetration_pct_vertices_penetrating_over_allowance" in header and len(header) == 9 + 2 * len(FIELDS)
