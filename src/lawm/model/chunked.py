"""Chunked latent-action world model: one latent per consecutive frame pair, and the cumulative motion from frame 0 at every step decoded in one call."""
from __future__ import annotations

import torch
import torch.nn as nn

from lawm.model.backbone import PTV3_CONFIG, build_backbone, collect_pooling_chain, make_data_dict, voxel_representatives
from lawm.model.blocks import AdaLNBlock, CrossAttnBlock, SelfAttnBlock, build_mlp, pad_stack, split_by_offset, unpad_concat
from lawm.model.latent import LatentEncoder
from lawm.model.seeding import init_seed, own_stream

STD_FLOOR = 0.001  # metres: no step or axis is divided by less than this


def unpool_concat_to_finest(chain):
    """chain = [level3, level2, level1, level0] (coarsest first): every level's features copied down to level 0's points and concatenated."""
    level3, level2, level1, level0 = chain
    f1 = level1.feat[level1.pooling_inverse]
    f2 = level2.feat[level2.pooling_inverse][level1.pooling_inverse]
    f3 = level3.feat[level3.pooling_inverse][level2.pooling_inverse][level1.pooling_inverse]
    return torch.cat([level0.feat, f1, f2, f3], dim=-1)


class ChunkedCrossAdaLNDecoder(nn.Module):
    """Predicts every point's displacement at every step; horizon k sees only latents z_1..z_k, and the horizons run as one batch of n_steps * B clouds."""

    def __init__(self, dim, num_heads, widths, n_steps, mlp_ratio=4, z_dim=8, n_queries=4,
                 head_hidden=(64,), ada_weight_init_std=0.02, self_attention="adaln"):
        """self_attention is the block after the cross-attention: "adaln" (self-attention modulated by the visible latents) or "plain" (unconditioned self-attention)."""
        super().__init__()
        assert self_attention in ("adaln", "plain"), self_attention
        self.n_steps, self.n_queries, self.self_attention = n_steps, n_queries, self_attention
        with own_stream("decoder.concat_proj"):
            self.concat_proj = nn.Linear(sum(widths), dim)
        with own_stream("decoder.code_proj"):
            self.code_proj = nn.Linear(z_dim, dim)
        with own_stream("decoder.step_embed"):
            self.step_embed = nn.Parameter(torch.randn(n_steps, dim) * 0.02)
        with own_stream("decoder.cross"):
            self.cross = CrossAttnBlock(dim, num_heads, mlp_ratio)
        self.self_adaln = self.self_attn = None
        if self_attention == "adaln":
            with own_stream("decoder.self_adaln"):
                self.self_adaln = AdaLNBlock(dim, num_heads, mlp_ratio, cond_dim=n_steps * n_queries * z_dim,
                                             ada_weight_init_std=ada_weight_init_std)
        else:
            with own_stream("decoder.self_attn"):
                self.self_attn = SelfAttnBlock(dim, num_heads, mlp_ratio)
        with own_stream("decoder.head"):
            self.head = build_mlp(dim, head_hidden, 3 * n_steps, norm=False)

    def point_queries(self, chain):
        """Per-point queries (rows of all clouds) and their offsets from frame 0's feature levels."""
        level0 = chain[-1]
        queries = self.concat_proj(unpool_concat_to_finest(chain))
        if "dedup_rows" in level0.keys():  # the backbone saw one point per voxel: give every point its voxel's features
            return queries[level0["dedup_rows"]], level0["dedup_offset"]
        return queries, level0.offset

    def forward(self, chain, zs, queries=None):
        """chain: frame 0's feature levels; zs: (n_steps, B, n_queries, z_dim); queries: point_queries(chain) if already computed. Returns (points of frame 0, n_steps, 3)."""
        q_list = split_by_offset(*(queries if queries is not None else self.point_queries(chain)))
        lengths = [t.shape[0] for t in q_list]
        q_pad, q_mask = pad_stack(q_list)
        B, L = q_pad.shape[0], q_pad.shape[1]
        n_steps, n_queries, device = self.n_steps, self.n_queries, q_pad.device

        kv = self.code_proj(zs) + self.step_embed.view(n_steps, 1, 1, -1)
        kv = kv.permute(1, 0, 2, 3).reshape(B, n_steps * n_queries, -1)
        token_step = torch.arange(n_steps, device=device).repeat_interleave(n_queries)
        horizons = torch.arange(1, n_steps + 1, device=device)
        causal_kv_mask = token_step.unsqueeze(0) >= horizons.unsqueeze(1)  # True where horizon k must not look

        q_rep = q_pad.unsqueeze(0).expand(n_steps, B, L, -1).reshape(n_steps * B, L, -1)
        q_mask_rep = q_mask.unsqueeze(0).expand(n_steps, B, L).reshape(n_steps * B, L)
        kv_rep = kv.unsqueeze(0).expand(n_steps, B, n_steps * n_queries, -1).reshape(n_steps * B, n_steps * n_queries, -1)
        kv_mask_rep = causal_kv_mask.unsqueeze(1).expand(n_steps, B, n_steps * n_queries).reshape(n_steps * B, n_steps * n_queries)
        q_rep = self.cross(q_rep, kv_rep, key_padding_mask=kv_mask_rep)

        if self.self_adaln is not None:
            step_visible = (torch.arange(n_steps, device=device).unsqueeze(0) < horizons.unsqueeze(1)).float()
            zs_causal = zs.unsqueeze(0) * step_visible.view(n_steps, n_steps, 1, 1, 1)
            cond_rep = zs_causal.permute(0, 2, 1, 3, 4).reshape(n_steps * B, -1)
            q_rep = self.self_adaln(q_rep, cond_rep, key_padding_mask=q_mask_rep)
        else:
            q_rep = self.self_attn(q_rep, q_mask_rep)

        out_rep = unpad_concat(q_rep, lengths * n_steps)
        pred_rep = self.head(out_rep).reshape(n_steps, out_rep.shape[0] // n_steps, n_steps * 3)
        return torch.stack([pred_rep[h, :, 3 * h:3 * (h + 1)] for h in range(n_steps)], dim=1)  # horizon h uses its own 3 channels


class ChunkedLAWM(nn.Module):
    """Latent-action world model; norm_stats holds per_step_mean and per_step_var, (n_steps, 3), from the training split, and backbone defaults to PTv3."""

    def __init__(self, norm_stats, n_steps=5, cfg=PTV3_CONFIG, z_dim=8, n_queries=4, ada_weight_init_std=0.02, backbone=None, seed=None,
                 decoder_self_attention="adaln", voxel_dedup=False):
        """seed, if given, makes each module draw its starting weights from its own stream keyed by (seed, module name); None uses the global random stream."""
        super().__init__()
        self.n_steps, self.voxel_dedup = n_steps, voxel_dedup
        dim, heads = cfg["enc_channels"][-1], cfg["enc_num_head"][-1]
        with init_seed(seed):
            with own_stream("ptv3"):
                self.ptv3 = backbone if backbone is not None else build_backbone(cfg)
            self.latent_encoder = LatentEncoder(dim, heads, z_dim=z_dim, n_queries=n_queries)
            self.decoder = ChunkedCrossAdaLNDecoder(dim, heads, cfg["enc_channels"], n_steps, z_dim=z_dim, n_queries=n_queries,
                                                    ada_weight_init_std=ada_weight_init_std, self_attention=decoder_self_attention)
        mean = torch.as_tensor(norm_stats["per_step_mean"], dtype=torch.float32)
        std = torch.sqrt(torch.as_tensor(norm_stats["per_step_var"], dtype=torch.float32)).clamp_min(STD_FLOOR)
        assert mean.shape == (n_steps, 3) and std.shape == (n_steps, 3), f"statistics have shape {tuple(mean.shape)}, need ({n_steps}, 3)"
        self.register_buffer("out_mean_windowed", mean)
        self.register_buffer("out_std_windowed", std)
        self._order_checked = False

    def unnormalize_step(self, delta_norm, k):
        """Step k (1-based): normalised prediction -> metres of cumulative displacement from frame 0."""
        return delta_norm * self.out_std_windowed[k - 1:k] + self.out_mean_windowed[k - 1:k]

    def encode_window(self, coords, offset):
        """coords: (n_steps + 1, total points, 3), offset: (B,). Returns frame 0's feature levels and the latents (n_steps, B, n_queries, z_dim)."""
        with torch.autocast(device_type="cuda", enabled=False):  # the sparse convolutions run in fp32
            if self.voxel_dedup:  # every frame: the backbone sees one point per voxel, no tracked point is dropped from the window
                reps = [voxel_representatives(coords[k], offset) for k in range(self.n_steps + 1)]
                real_points = [self.ptv3(make_data_dict(coords[k][kept], kept_offset, grid_coord=grid)) for k, (kept, _, kept_offset, grid) in enumerate(reps)]
            else:
                real_points = [self.ptv3(make_data_dict(coords[k], offset)) for k in range(self.n_steps + 1)]
        chain_0 = collect_pooling_chain(real_points[0])
        expected = coords[0][reps[0][0]] if self.voxel_dedup else coords[0]
        if not self._order_checked:  # predictions are matched to targets by row, so the backbone must keep the input row order
            assert torch.equal(chain_0[-1].coord, expected), "the backbone reordered the points of a cloud"
            self._order_checked = True
        if self.voxel_dedup:
            chain_0[-1]["dedup_rows"], chain_0[-1]["dedup_offset"] = reps[0][1], offset
        zs = [self.latent_encoder(real_points[k - 1], real_points[k]) for k in range(1, self.n_steps + 1)]
        return chain_0, torch.stack(zs, dim=0)

    def decode_all_steps(self, chain_0, zs, queries=None):
        """One decoder call for all steps: a list of n_steps dicts with "delta_norm" and "delta" (points, 1, 3), the latter in metres; queries as in the decoder."""
        delta_grid = self.decoder(chain_0, zs, queries)
        outs = []
        for k in range(1, self.n_steps + 1):
            delta_norm = delta_grid[:, k - 1:k, :]
            delta = self.unnormalize_step(delta_norm.float(), k)
            assert torch.isfinite(delta_norm).all() and torch.isfinite(delta).all()
            outs.append({"delta_norm": delta_norm, "delta": delta})
        return outs

    def forward(self, coords, offset, return_latents=False):
        """Call this module itself (also under DistributedDataParallel) so the gradient hooks see the whole forward pass."""
        chain_0, zs = self.encode_window(coords, offset)
        outs = self.decode_all_steps(chain_0, zs)
        return (outs, zs) if return_latents else outs
