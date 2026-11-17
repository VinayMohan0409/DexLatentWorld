"""Tests of the model adapter on CPU with the stand-in backbone: frame-0 encoding equals the model's own, chunked decoding of the repeated queries equals decoding each candidate alone and ignores padding, prior files, loading."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("lawm.model.chunked")
fake_backbone = pytest.importorskip("fake_backbone")

from lawm.model.chunked import ChunkedLAWM  # noqa: E402
from lawm.sim import model_adapter as M  # noqa: E402

N_STEPS = 3
CFG = dict(enc_channels=(48, 96, 144, 192), enc_num_head=(2, 4, 6, 8))
STATS = dict(per_step_mean=np.zeros((N_STEPS, 3)), per_step_var=np.full((N_STEPS, 3), 1e-4))


def make_model(dedup=False, seed=0):
    """A seeded model with the stand-in backbone."""
    torch.manual_seed(seed)
    return ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=fake_backbone.FakeBackbone(), voxel_dedup=dedup).eval()


def cloud(n=24, seed=0, shared=False):
    """A random cloud (float64, off-centre); shared = rows 4 and 5 are the same point (one voxel)."""
    c = np.random.default_rng(seed).normal(size=(n, 3)) * 0.1 + [1.0, -2.0, 0.5]
    if shared:
        c[5] = c[4]
    return c


def latents(pop, seed=1, model=None):
    """Random candidate latents (pop, n_steps, n_queries, z_dim)."""
    model = model or make_model()
    return np.random.default_rng(seed).normal(size=(pop, N_STEPS, model.decoder.n_queries, 8)).astype(np.float32)


@pytest.mark.parametrize("dedup", [False, True])
def test_encoding_equals_the_models_own_frame_zero(dedup):
    """The chain equals what encode_window builds for frame 0 of a window whose every frame is the centred cloud."""
    model, c = make_model(dedup), cloud(shared=dedup)
    chain = M.encode_cloud(model, c)
    centred = torch.from_numpy((c - c.mean(0)).astype(np.float32))
    with torch.no_grad():
        reference, _ = model.encode_window(centred[None].repeat(N_STEPS + 1, 1, 1), torch.tensor([len(c)]).int())
    assert len(chain) == len(reference) == 4
    for a, b in zip(chain, reference):
        assert torch.equal(a.feat, b.feat) and torch.equal(a.coord, b.coord)
    if dedup:
        assert torch.equal(chain[-1]["dedup_rows"], reference[-1]["dedup_rows"]) and torch.equal(chain[-1]["dedup_offset"], reference[-1]["dedup_offset"])


def test_repeated_queries_are_the_one_cloud_queries_in_every_slot():
    """Every slot holds the queries of the one encoded cloud, and the offsets end each copy."""
    model, c = make_model(), cloud()
    chain = M.encode_cloud(model, c)
    with torch.no_grad():
        one, _ = model.decoder.point_queries(chain)
    queries, offset = M.repeated_queries(model, chain, copies=4)
    assert torch.equal(queries, one.repeat(4, 1)) and offset.tolist() == [24, 48, 72, 96]


def alone(model, chain, z, n):
    """Each candidate decoded on its own through the model's decode_all_steps (batch of one): (pop, n_steps, n, 3)."""
    with torch.no_grad():
        return np.stack([np.stack([d["delta"].squeeze(1).numpy().reshape(n, 3) for d in model.decode_all_steps(chain, torch.from_numpy(zi[:, None].copy()))]) for zi in z])


@pytest.mark.parametrize("dedup", [False, True])
def test_chunked_decode_equals_each_candidate_decoded_alone(dedup):
    """One chunk of candidates against the repeated queries gives what the model gives each candidate alone, per step."""
    model, c = make_model(dedup), cloud(shared=dedup)
    chain, z = M.encode_cloud(model, c), latents(4, model=model)
    got = M.decode_chunked(model, M.repeated_queries(model, chain, copies=4), z, len(c), copies=4)
    assert got.shape == (4, N_STEPS, len(c), 3) and got.dtype == np.float32 and np.allclose(got, alone(model, chain, z, len(c)), atol=1e-6)


def test_padding_is_dropped_and_a_candidate_does_not_depend_on_its_slot():
    """A population that is not a multiple of the chunk gives the same rows as a full population containing it, and the single-candidate call gives the first row."""
    model, c = make_model(), cloud()
    chain, z = M.encode_cloud(model, c), latents(8, model=model)
    queries = M.repeated_queries(model, chain, copies=4)
    full, five, one = (M.decode_chunked(model, queries, z[:n], len(c), copies=4) for n in (8, 5, 1))
    assert five.shape[0] == 5 and one.shape[0] == 1
    assert np.allclose(full[:5], five, atol=1e-6) and np.allclose(full[:1], one, atol=1e-6)
    assert np.allclose(full[1], M.decode_chunked(model, queries, z[1:2], len(c), copies=4)[0], atol=1e-6)


def test_the_adapter_decoder_is_built_per_cloud_and_ignores_position_and_labels():
    """The decode function answers for the cloud it was built from, gives the same displacements after a shift of the whole cloud, and does not use the segment labels."""
    model = make_model()
    adapter = M.ModelAdapter(model, (np.zeros((N_STEPS, 4, 8), np.float32), np.ones((N_STEPS, 4, 8), np.float32)), copies=4)
    c, z = cloud(), latents(6, model=model)
    a = adapter.decoder(c, np.zeros(24, int))(z)
    b = adapter.decoder(c + [3.0, 1.0, -2.0], np.arange(24) % 4)(z)
    assert a.shape == (6, N_STEPS, 24, 3) and np.allclose(a, b, atol=1e-4)
    assert not np.allclose(a, adapter.decoder(cloud(seed=5), None)(z))
    assert adapter.prior[0].shape == (N_STEPS, 4, 8)


def test_build_prior_is_the_mean_and_spread_of_the_encoded_windows():
    """The prior equals numpy's mean and population standard deviation of the latents of each window."""
    model = make_model()
    windows = [np.random.default_rng(i).normal(size=(N_STEPS + 1, 20, 3)).astype(np.float32) * 0.1 for i in range(5)]
    mean, std = M.build_prior(model, windows)
    with torch.no_grad():
        zs = np.stack([model.encode_window(torch.from_numpy(w), torch.tensor([20]).int())[1].squeeze(1).numpy() for w in windows])
    assert mean.shape == std.shape == (N_STEPS, 4, 8) and np.allclose(mean, zs.mean(0), atol=1e-6) and np.allclose(std, zs.std(0), atol=1e-6) and (std > 0).all()


def test_prior_file_round_trip_and_checkpoint_guard(tmp_path):
    """A stored prior loads back unchanged for its checkpoint and is refused for another file."""
    ck, other = tmp_path / "a.pt", tmp_path / "b.pt"
    ck.write_bytes(b"one"), other.write_bytes(b"two")
    prior = (np.arange(6, dtype=np.float32).reshape(1, 2, 3), np.ones((1, 2, 3), np.float32))
    M.save_prior(tmp_path / "p.npz", prior, ck, 400)
    mean, std = M.load_prior(tmp_path / "p.npz", ck)
    assert np.array_equal(mean, prior[0]) and np.array_equal(std, prior[1]) and int(np.load(tmp_path / "p.npz")["n_windows"]) == 400
    with pytest.raises(AssertionError):
        M.load_prior(tmp_path / "p.npz", other)


def test_load_model_restores_the_checkpoint_including_its_statistics(tmp_path):
    """The model built from a manifest takes the checkpoint's weights and its own normalisation buffers, so it predicts what the saved model predicts."""
    saved = ChunkedLAWM(dict(per_step_mean=np.full((N_STEPS, 3), 0.01), per_step_var=np.full((N_STEPS, 3), 4e-4)), n_steps=N_STEPS, cfg=CFG,
                        backbone=fake_backbone.FakeBackbone(), seed=3, voxel_dedup=True).eval()
    torch.save({"model": saved.state_dict()}, tmp_path / "best.pt")
    manifest = dict(args=dict(z_dim=8, n_queries=4, ada_weight_init_std=0.02, decoder_self_attention="adaln", voxel_dedup=True),
                    run=dict(ptv3_config=dict(CFG, enc_channels=list(CFG["enc_channels"]))), config=dict(window=dict(n_steps=N_STEPS)))
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    loaded = M.load_model(tmp_path / "best.pt", tmp_path / "run_manifest.json", "cpu", backbone=fake_backbone.FakeBackbone())
    assert not loaded.training and loaded.voxel_dedup and torch.equal(loaded.out_std_windowed, saved.out_std_windowed)
    c, z = cloud(), latents(2, model=saved)
    a = M.ModelAdapter(saved, None, 2).decoder(c, None)(z)
    b = M.ModelAdapter(loaded, None, 2).decoder(c, None)(z)
    assert np.array_equal(a, b)
