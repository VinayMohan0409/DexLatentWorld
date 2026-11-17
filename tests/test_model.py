"""Unit tests for lawm.model on CPU, with a stand-in backbone: blocks, causal decoding, the model's shapes, the losses and the metrics."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fake_backbone import FakeBackbone, Point

from lawm.model.blocks import AdaLNBlock, CrossAttnBlock, SelfAttnBlock, pad_stack, split_by_offset, unpad_concat
from lawm.model.chunked import ChunkedLAWM
from lawm.model.losses import embodiment_info_nce_loss, movement_weighted_huber_loss
from lawm.model.metrics import NAMES, step_sums, summarise

N_STEPS = 3
CFG = dict(enc_channels=(48, 96, 144, 192), enc_num_head=(2, 4, 6, 8))
STATS = dict(per_step_mean=np.zeros((N_STEPS, 3)), per_step_var=np.full((N_STEPS, 3), 1e-4))


def make_model(seed=0):
    """Build a seeded ChunkedLAWM with the fake backbone"""
    torch.manual_seed(seed)
    return ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=FakeBackbone())


def make_window(sizes=(24, 16), seed=0):
    """Build random window coordinates and the cumulative cloud offsets"""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(N_STEPS + 1, sum(sizes), 3, generator=g) * 0.1
    return coords, torch.cumsum(torch.tensor(sizes), 0).int()


def test_pad_helpers_round_trip():
    """split_by_offset, pad_stack and unpad_concat round-trip a ragged batch with a correct padding mask"""
    x = torch.randn(10, 4)
    offset = torch.tensor([6, 10])
    parts = split_by_offset(x, offset)
    padded, mask = pad_stack(parts)
    assert padded.shape == (2, 6, 4) and mask.sum().item() == 2 and torch.equal(unpad_concat(padded, [6, 4]), x)


def test_padding_does_not_change_a_cloud_in_any_block():
    """Self-attention and AdaLN blocks give the same output for a cloud alone or padded alongside a longer one"""
    torch.manual_seed(0)
    a, b = torch.randn(5, 192), torch.randn(9, 192)
    for blk, call in ((SelfAttnBlock(192, 8), lambda m, x, mask: m(x, mask)),
                      (AdaLNBlock(192, 8, cond_dim=16, ada_weight_init_std=0.5), lambda m, x, mask: m(x, torch.ones(x.shape[0], 16), mask))):
        alone = call(blk, *[t[:1] if t.dim() > 1 else t for t in pad_stack([a])])
        together = call(blk, *pad_stack([a, b]))
        assert torch.allclose(alone[0], together[0, :5], atol=1e-5)


def test_cross_attention_ignores_masked_keys():
    """Cross-attention output is unchanged when masked key and value entries are altered"""
    torch.manual_seed(0)
    blk, q = CrossAttnBlock(192, 8), torch.randn(1, 7, 192)
    kv = torch.randn(1, 6, 192)
    mask = torch.tensor([[False, False, False, True, True, True]])
    kv2 = kv.clone()
    kv2[:, 3:] = torch.randn(1, 3, 192) * 10
    assert torch.allclose(blk(q, kv, mask), blk(q, kv2, mask), atol=1e-6)


def test_adaln_block_is_the_identity_at_zero_gates():
    """An AdaLN block with zero gates and zero modulation weights returns its input unchanged"""
    blk = AdaLNBlock(192, 8, cond_dim=16, gate_init=0.0, ada_weight_init_std=0.0)
    x, mask = torch.randn(2, 5, 192), torch.zeros(2, 5, dtype=torch.bool)
    assert torch.equal(blk(x, torch.randn(2, 16), mask), x)


def test_model_shapes_buffers_and_unnormalising():
    """The model returns outputs and latents of the expected shapes, has the expected output-std buffer, and unnormalize_step rescales by it"""
    model = make_model()
    coords, offset = make_window()
    outs, zs = model(coords, offset, return_latents=True)
    assert len(outs) == N_STEPS and zs.shape == (N_STEPS, 2, 4, 8)
    assert outs[0]["delta"].shape == (coords.shape[1], 1, 3) and torch.isfinite(outs[0]["delta"]).all()
    assert torch.allclose(model.out_std_windowed, torch.full((N_STEPS, 3), 0.01)) and model._order_checked
    x = torch.randn(5, 1, 3)
    assert torch.allclose(model.unnormalize_step(x, 2), x * 0.01)


def test_the_model_is_deterministic_for_a_seed():
    """Models built with the same seed give identical outputs and a different seed differs"""
    coords, offset = make_window()
    a, b = make_model(1)(coords, offset), make_model(1)(coords, offset)
    assert all(torch.equal(x["delta"], y["delta"]) for x, y in zip(a, b))
    assert not torch.equal(make_model(2)(coords, offset)[0]["delta"], a[0]["delta"])


def test_a_horizon_depends_only_on_the_latents_up_to_it():
    """Changing the latent of step k alters only the horizons from k onwards"""
    model = make_model().eval()
    coords, offset = make_window()
    with torch.no_grad():
        chain_0, zs = model.encode_window(coords, offset)
        base = [o["delta_norm"] for o in model.decode_all_steps(chain_0, zs)]
        for k in range(N_STEPS):  # change the latent of step k + 1
            zs2 = zs.clone()
            zs2[k] += torch.randn_like(zs2[k])
            changed = [o["delta_norm"] for o in model.decode_all_steps(chain_0, zs2)]
            for h in range(N_STEPS):
                same = torch.equal(base[h], changed[h])
                assert same == (h < k), f"changing z_{k + 1} {'must not' if h < k else 'must'} change horizon {h + 1}"


def test_the_backbone_row_order_check_fires():
    """The model raises an AssertionError when the backbone reorders its points"""
    class Shuffling(FakeBackbone):
        """Fake backbone that reverses the finest level's coordinates so the model's row-order check should fail."""
        def forward(self, data):
            """Run the fake backbone, then reverse the coordinates of the finest level so the row order check should fail"""
            out = super().forward(data)
            out["pooling_parent"]["pooling_parent"]["pooling_parent"]["coord"] = data["coord"].flip(0)
            return out
    model = ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=Shuffling())
    with pytest.raises(AssertionError, match="reordered"):
        model(*make_window())


def reference_loss(pred, gt, offset, mean, std, delta=5.0, tau=0.005, kappa=1000.0):
    """Straightforward per-cloud reference of the movement-weighted Huber loss"""
    total, start = 0.0, 0
    for end in offset.tolist():
        w = torch.sigmoid(kappa * (gt[start:end].norm(dim=-1).reshape(-1) - tau))
        w = w / w.sum().clamp_min(1e-6)
        diff = (pred[start:end] - (gt[start:end] - mean) / std).reshape(end - start, 3)
        huber = torch.where(diff.abs() <= delta, 0.5 * diff ** 2, delta * (diff.abs() - 0.5 * delta)).mean(-1)
        total += (w * huber).sum()
        start = end
    return total / len(offset)


def test_movement_weighted_loss_matches_a_plain_reference():
    """movement_weighted_huber_loss equals the plain reference implementation"""
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(30, 1, 3, generator=g) * 0.01
    pred, offset = torch.randn(30, 1, 3, generator=g), torch.tensor([12, 30])
    mean, std = torch.zeros(1, 3), torch.full((1, 3), 0.01)
    assert torch.allclose(movement_weighted_huber_loss(pred, gt, offset, mean, std), reference_loss(pred, gt, offset, mean, std), rtol=1e-5)


def test_the_loss_weights_moving_points_more():
    """The loss penalises errors on a moving point far more than the same error on a still point"""
    gt = torch.zeros(10, 1, 3)
    gt[0, 0, 0] = 0.05  # one moving point among ten
    offset, mean, std = torch.tensor([10]), torch.zeros(1, 3), torch.full((1, 3), 0.01)
    right = torch.zeros(10, 1, 3)
    right[0, 0, 0] = 5.0  # the mover's normalised target is 0.05 / 0.01
    wrong_on_mover = torch.zeros(10, 1, 3)  # predicts no motion for the mover
    wrong_on_still = right.clone()
    wrong_on_still[5, 0, 0] = 5.0  # predicts motion for a still point
    assert movement_weighted_huber_loss(right, gt, offset, mean, std) == 0
    assert movement_weighted_huber_loss(wrong_on_mover, gt, offset, mean, std) > 100 * movement_weighted_huber_loss(wrong_on_still, gt, offset, mean, std)


def test_contrastive_loss_pulls_positives_together():
    """The contrastive loss is lower for aligned positives, equals log(5) at collapse and is zero without positives"""
    torch.manual_seed(0)
    group = torch.tensor([0, 0, 1, 1, 2, 2])
    random_z = torch.randn(6, 4, 8)
    aligned = random_z[[0, 0, 2, 2, 4, 4]]  # each group's two samples share a latent
    assert embodiment_info_nce_loss(aligned, group) < 0.5 * embodiment_info_nce_loss(random_z, group)
    assert math.isclose(embodiment_info_nce_loss(torch.ones(6, 4, 8), group).item(), math.log(5), rel_tol=1e-5)  # collapse is the worst case
    assert embodiment_info_nce_loss(random_z, torch.arange(6)).item() == 0.0  # no positives


def test_metrics_on_a_known_example():
    """step_sums and summarise give the expected per-set errors and counts, and NaN with zero count for an empty set"""
    gt = torch.zeros(6, 1, 3)
    gt[0, 0, 0], gt[1, 0, 0], gt[2, 0, 0] = 0.010, 0.003, 0.001
    pred = gt.clone()
    pred[0, 0, 0] += 0.004  # 4 mm error on the 10 mm point
    pred[5, 0, 1] += 0.002  # 2 mm error on a still table point
    segment = torch.tensor([0, 1, 0, 1, 2, 3])
    out = summarise(step_sums(pred, gt, segment)[None].numpy())
    got = dict(zip(NAMES, zip(out["per_step"][0], out["count"][0])))
    assert got["mover_5mm"] == (pytest.approx(4.0), 1) and got["mover_2mm"] == (pytest.approx(2.0), 2)
    assert got["dynamic"] == (pytest.approx(1.0), 4) and got["static"] == (pytest.approx(1.0), 2) and got["overall"] == (pytest.approx(1.0), 6)
    empty = summarise(step_sums(gt, gt, torch.full((6,), 3))[None].numpy())
    assert math.isnan(empty["per_step"][0, NAMES.index("dynamic")]) and empty["count"][0, NAMES.index("dynamic")] == 0


def test_summed_metrics_do_not_depend_on_how_points_are_batched():
    """step_sums of the whole cloud equals the sum over arbitrary chunks of it"""
    g = torch.Generator().manual_seed(0)
    gt, pred = torch.randn(40, 1, 3, generator=g) * 0.01, torch.randn(40, 1, 3, generator=g) * 0.01
    segment = torch.randint(0, 4, (40,), generator=g)
    whole = step_sums(pred, gt, segment)
    parts = sum(step_sums(pred[a:b], gt[a:b], segment[a:b]) for a, b in ((0, 7), (7, 8), (8, 40)))
    assert torch.allclose(whole, parts)


def test_mean_over_steps_skips_undefined_sets():
    """summarise averages over steps while ignoring steps where a set is undefined"""
    a = step_sums(torch.zeros(4, 1, 3), torch.zeros(4, 1, 3), torch.zeros(4, dtype=torch.long))  # no mover and no static points
    b = a.clone()
    b[0] = torch.tensor([6.0, 2.0])  # step 2 has two movers with a summed error of 6 mm
    out = summarise(torch.stack([a, b]).numpy())
    assert np.isnan(out["per_step"][0, 0]) and out["per_step"][1, 0] == pytest.approx(3.0) and out["mean"][0] == pytest.approx(3.0)
    assert np.isnan(out["mean"][NAMES.index("static")])


def test_seeded_initialisation_leaves_shared_modules_identical_when_only_z_dim_changes():
    """With a seed, changing z_dim leaves every weight of unchanged shape identical, other than the modules that depend on z_dim"""
    def build(z):
        """Build the model state dict with a given latent size under a fixed torch seed"""
        torch.manual_seed(123)  # the stand-in backbone is built outside the seeded streams
        return ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, z_dim=z, backbone=FakeBackbone(), seed=0).state_dict()
    a, b, again = build(8), build(4), build(8)
    assert all(torch.equal(a[k], again[k]) for k in a)  # repeatable
    for k in a:
        if a[k].shape == b[k].shape and "adaln.ada" not in k and "code_proj" not in k and a[k].numel() > 1 and a[k].std() > 0:
            assert torch.equal(a[k], b[k]), k  # every weight whose shape agrees is identical
    assert not torch.equal(a["decoder.code_proj.weight"][:, :4], b["decoder.code_proj.weight"][:, :4])


def test_decoder_with_plain_self_attention_has_no_conditioning_and_stays_causal():
    """The plain-self-attention decoder has no AdaLN module, fewer parameters and still depends only on latents up to each horizon"""
    torch.manual_seed(0)
    model = ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=FakeBackbone(), decoder_self_attention="plain").eval()
    assert model.decoder.self_adaln is None and not any("self_adaln" in k for k in model.state_dict())
    assert any(k.startswith("decoder.self_attn.qkv_proj") for k in model.state_dict())
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in make_model().parameters())
    coords, offset = make_window()
    with torch.no_grad():
        chain_0, zs = model.encode_window(coords, offset)
        base = [o["delta_norm"] for o in model.decode_all_steps(chain_0, zs)]
        zs2 = zs.clone()
        zs2[1] += torch.randn_like(zs2[1])
        changed = [o["delta_norm"] for o in model.decode_all_steps(chain_0, zs2)]
    assert torch.equal(base[0], changed[0]) and not torch.equal(base[1], changed[1]) and not torch.equal(base[2], changed[2])


def test_voxel_representatives_keep_the_first_point_of_every_voxel_and_map_every_row_to_one():
    """voxel_representatives keeps the first point per voxel and cloud, maps every row to its representative and is repeatable"""
    from lawm.model.backbone import voxel_representatives
    g = torch.Generator().manual_seed(0)
    coord = torch.rand(300, 3, generator=g) * 0.03  # 6 voxels per axis: many points share a voxel
    offset = torch.tensor([120, 300])
    kept, rows, new_offset, grid = voxel_representatives(coord, offset, grid_size=0.005)
    full = torch.div(coord - coord.min(0)[0], 0.005, rounding_mode="trunc").int()
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor([120, 180]))
    keys = [(int(b), *map(int, v)) for b, v in zip(batch, full)]
    first = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    assert kept.tolist() == sorted(first.values()) and torch.equal(grid, full[kept])  # first point of each voxel, original order
    assert all(kept[rows[i]].item() == first[keys[i]] for i in range(300))  # every row maps to the representative of its own voxel
    assert new_offset.tolist() == [int((kept < 120).sum()), len(kept)]
    again = voxel_representatives(coord, offset, grid_size=0.005)
    assert all(torch.equal(a, b) for a, b in zip((kept, rows, new_offset, grid), again))


def test_a_cloud_without_shared_voxels_is_unchanged_by_the_wrapper():
    """voxel_representatives keeps all points and offsets when no two points share a voxel"""
    from lawm.model.backbone import voxel_representatives
    coord = torch.arange(30.0).reshape(10, 3) * 0.1  # far apart
    kept, rows, offset, _ = voxel_representatives(coord, torch.tensor([4, 10]))
    assert kept.tolist() == list(range(10)) and rows.tolist() == list(range(10)) and offset.tolist() == [4, 10]


def test_with_voxel_dedup_every_row_keeps_its_prediction_and_points_in_one_voxel_share_it():
    """With voxel_dedup every row still gets a prediction and points in the same voxel share it"""
    torch.manual_seed(0)
    model = ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=FakeBackbone(), voxel_dedup=True).eval()
    coords, offset = make_window()
    coords[:, 5] = coords[:, 4]  # rows 4 and 5 of the first cloud are the same point in every frame
    with torch.no_grad():
        outs = model(coords, offset)
    assert all(o["delta"].shape == (coords.shape[1], 1, 3) for o in outs)  # no row is dropped
    assert all(torch.equal(o["delta_norm"][4], o["delta_norm"][5]) for o in outs)
    assert not torch.equal(outs[0]["delta_norm"][4], outs[0]["delta_norm"][6])


def test_the_training_loss_and_every_gradient_ignore_the_segment_labels():
    """The training loss and gradients are identical for different segment labels, which only affect the reported sums"""
    from lawm.train.trainer import train_step_loss
    coords, offset = make_window()
    args = SimpleNamespace(contrastive_lambda=0.15, contrastive_tau=0.1)
    runs = []
    for segment in (torch.zeros(40, dtype=torch.long), torch.randint(0, 4, (40,), generator=torch.Generator().manual_seed(1))):
        model = make_model()
        batch = dict(coords=coords, offset=offset, segment=segment, transition_idx=torch.tensor([0, 0]))
        total, _, _, sums = train_step_loss(model, model, batch, args)
        total.backward()
        runs.append((total.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None], sums))
    assert torch.equal(runs[0][0], runs[1][0]) and all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
    assert not torch.equal(runs[0][2], runs[1][2])  # the labels do change the reported sets, which is all they are for


@pytest.mark.parametrize("kind", ["plain", "adaln"])
def test_every_parameter_receives_a_gradient_from_the_loss_in_both_decoder_variants(kind):
    """Every parameter gets a non-zero gradient from the training loss for both decoder self-attention variants"""
    from lawm.train.trainer import train_step_loss
    torch.manual_seed(0)
    model = ChunkedLAWM(STATS, n_steps=N_STEPS, cfg=CFG, backbone=FakeBackbone(), decoder_self_attention=kind)
    coords, offset = make_window()
    batch = dict(coords=coords, offset=offset, segment=torch.zeros(40, dtype=torch.long), transition_idx=torch.tensor([0, 0]))
    train_step_loss(model, model, batch, SimpleNamespace(contrastive_lambda=0.15, contrastive_tau=0.1))[0].backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.norm()) == 0.0]
    assert not dead, dead
