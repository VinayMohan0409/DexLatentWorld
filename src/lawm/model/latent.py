"""Latent action encoder: joint attention over the coarse tokens of two frames, read out at learned query tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from lawm.model.seeding import own_stream
from lawm.model.blocks import SelfAttnBlock, build_mlp, pad_stack, sinusoidal_time_embedding, split_by_offset


class LatentEncoder(nn.Module):
    """Encodes frame t and t+k tokens, a separator and n_queries learned queries with self-attention; the query outputs form z, (B, n_queries, z_dim)."""

    def __init__(self, dim, num_heads, mlp_ratio=4, n_layers=2, z_dim=8, n_queries=4, time_dim=32, preq_hidden=(128,)):
        """Creates the time embedding, the learned queries, the transformer layers and the projection to z."""
        super().__init__()
        self.n_queries, self.z_dim, self.time_dim = n_queries, z_dim, time_dim
        with own_stream("latent_encoder.time_mlp"):
            self.time_mlp = build_mlp(time_dim, (dim,), dim)
        with own_stream("latent_encoder.pos_mlp"):
            self.pos_mlp = build_mlp(3, (dim,), dim)
        with own_stream("latent_encoder.sep_token"):
            self.sep_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        with own_stream("latent_encoder.query_tokens"):
            self.query_tokens = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        blocks = []
        for i in range(n_layers):
            with own_stream(f"latent_encoder.blocks.{i}"):
                blocks.append(SelfAttnBlock(dim, num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)
        with own_stream("latent_encoder.pre_q"):
            self.pre_q = build_mlp(dim, preq_hidden, z_dim, norm=False)

    def _embed(self, point, t_norm):
        """Backbone features plus the frame-time embedding (t_norm) and the coordinate embedding."""
        t = point.feat.new_full((point.feat.shape[0],), t_norm)
        return point.feat + self.time_mlp(sinusoidal_time_embedding(t, self.time_dim)) + self.pos_mlp(point.coord)

    def forward(self, point_t, point_future):
        """Latent z (B, n_queries, z_dim) from the coarse points of frame t and a future frame."""
        list_t = split_by_offset(self._embed(point_t, 0.0), point_t.offset)
        list_f = split_by_offset(self._embed(point_future, 1.0), point_future.offset)
        seqs, q_starts = [], []
        for a, b in zip(list_t, list_f):
            seqs.append(torch.cat([a, self.sep_token, b, self.query_tokens], dim=0))
            q_starts.append(a.shape[0] + 1 + b.shape[0])
        padded, mask = pad_stack(seqs)
        for blk in self.blocks:
            padded = blk(padded, key_padding_mask=mask)
        q_out = torch.stack([padded[i, s:s + self.n_queries] for i, s in enumerate(q_starts)])
        return self.pre_q(q_out)
