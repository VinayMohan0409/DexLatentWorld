"""Unit tests for lawm.preprocessing.contact_refinement.optimizer with a stub MANO layer (no model files needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from lawm.preprocessing.contact_refinement.optimizer import (  # noqa: E402
    RefineConfig,
    build_batch,
    mano_forward,
    precompute_contacts,
    second_diff,
    smoothstep,
    subsample_object,
    temporal_refine,
)
from lawm.preprocessing.contact_refinement.sequence import SequenceData  # noqa: E402

T, V = 5, 30


class StubMano(torch.nn.Module):
    """Stand-in for manopth's layer: a fixed point set (mm) that the first 3 pose values translate by 1 m/unit."""

    def __init__(self):
        """Create a fixed random point set in millimetres from a seeded generator"""
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.base = torch.rand(V, 3, generator=g) * 20.0  # millimetres, a ~2 cm blob

    def forward(self, pose_full, betas):
        """Translate the fixed points by the first three pose values (metres converted to millimetres) and return them with zero joints"""
        verts = self.base[None] + pose_full[:, None, :3] * 1000.0
        return verts, torch.zeros(pose_full.shape[0], 16, 3)


def make_sequence(obj_offset):
    """Build a short synthetic hand sequence with slow wrist motion and an icosphere object shifted by obj_offset"""
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    mTc = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    mTc[:, :3, 3] = np.linspace(0, 0.01, T)[:, None] * np.ones(3)  # slow wrist motion
    obj = np.tile(sphere.vertices.astype(np.float32) + np.float32(obj_offset), (T, 1, 1))
    return SequenceData(
        hand_pose=np.zeros((T, 45), np.float32), hand_rot=np.zeros((T, 3), np.float32), hand_mTc_world=mTc,
        hand_beta=np.zeros(10, np.float32), obj_verts=obj, obj_faces=np.asarray(sphere.faces, np.int64),
        frame_ids=np.arange(T), side="right")


def refine(obj_offset, **cfg_kwargs):
    """Run contact precomputation and temporal refinement on a synthetic sequence with the stub MANO layer, returning the sequence, precomputed contacts and result"""
    seq, model = make_sequence(obj_offset), StubMano()
    cfg = RefineConfig(n_iter=15, obj_sample_n=16, **cfg_kwargs)
    data = build_batch(seq, "cpu")
    precomp = precompute_contacts(data, model, cfg, "right")
    return seq, precomp, temporal_refine(seq, data, precomp, model, cfg, "right", "cpu")


def test_smoothstep_endpoints_and_midpoint():
    """smoothstep is 1 below the lower edge, 0 above the upper edge and 0.5 at the midpoint"""
    x = torch.tensor([0.0, 0.008, 0.014, 0.02, 1.0])
    assert torch.allclose(smoothstep(x, 0.008, 0.02), torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))


def test_second_diff_is_zero_for_linear_motion_and_constant_for_quadratic():
    """second_diff gives zero for linear motion and a constant 2 for a quadratic"""
    t = torch.arange(6.0)[:, None]
    assert torch.allclose(second_diff(2 * t + 1), torch.zeros(4, 1))
    assert torch.allclose(second_diff(t ** 2), torch.full((4, 1), 2.0))


def test_subsample_object_is_deterministic_farthest_point_sampling():
    """subsample_object picks far-apart points deterministically, skips near-duplicates and keeps all points when fewer than requested"""
    pts = torch.tensor([[[0.0, 0, 0], [0.01, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0.5, 0.5, 0]]])
    sub, idx = subsample_object(pts, 3)
    assert idx[0].item() == 0 and set(idx.tolist()) == {0, 2, 3}  # picks the far-apart points, skips the near-duplicate
    assert torch.equal(subsample_object(pts, 3)[1], idx)
    assert subsample_object(pts, 10)[1].tolist() == [0, 1, 2, 3, 4]  # fewer points than requested: keep all


def test_mano_forward_converts_mm_to_metres_and_places_in_world():
    """mano_forward converts the layer's millimetre vertices to metres and applies the world translation"""
    mTc = torch.eye(4)[None].clone()
    mTc[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    zeros = torch.zeros(1, 3)
    out = mano_forward(StubMano(), zeros, torch.zeros(1, 45), torch.zeros(1, 10), mTc)
    assert torch.allclose(out[0], StubMano().base / 1000.0 + torch.tensor([1.0, 2.0, 3.0]))


def test_far_object_gives_zero_gate_and_leaves_the_hand_untouched():
    """With a distant object the contact gate is zero and refinement returns the original hand pose, rotation and translation"""
    seq, precomp, result = refine(obj_offset=5.0)
    assert float(precomp["gate"].max()) == 0.0
    assert np.array_equal(result["rot"], seq.hand_rot) and np.array_equal(result["pose"], seq.hand_pose)
    assert np.array_equal(result["trans_world_final"], seq.hand_mTc_world[:, :3, 3])


def test_nearby_object_moves_the_hand_and_stays_finite():
    """With an overlapping object the gate opens, the hand moves, the loss improves and all outputs stay finite"""
    seq, precomp, result = refine(obj_offset=0.0, lr=1e-3)  # 1 mm steps: the stub moves 1 m per pose unit
    assert float(precomp["gate"].max()) > 0.5
    moved = np.abs(result["trans_world_final"] - seq.hand_mTc_world[:, :3, 3]).max() + np.abs(result["rot"]).max()
    assert moved > 0.0 and result["best_iter"] > 0  # some step improved on the starting loss
    assert all(np.isfinite(result[k]).all() for k in ("hand_verts", "rot", "pose", "trans_world_final"))
    assert result["frame_ids"].tolist() == list(range(T)) and result["n_iter_actual"] <= 15
