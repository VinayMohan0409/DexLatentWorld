"""Attention blocks and helpers for ragged point clouds padded into (batch, longest cloud, dim) tensors."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_offset(x, offset):
    """x: (N, D) rows of all clouds; offset: (B,) cumulative cloud ends. Returns a list of B tensors (n_i, D)."""
    starts = torch.cat([offset.new_zeros(1), offset[:-1]])
    return [x[lo:hi] for lo, hi in zip(starts.tolist(), offset.tolist())]


def pad_stack(tensors):
    """List of (n_i, D) -> zero-padded (B, longest, D) and a (B, longest) mask that is True on padding."""
    assert all(t.shape[0] > 0 for t in tensors), "a cloud has zero rows: an all-masked sequence gives NaN attention"
    longest = max(t.shape[0] for t in tensors)
    out = tensors[0].new_zeros(len(tensors), longest, tensors[0].shape[-1])
    pad_mask = torch.ones(len(tensors), longest, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
        pad_mask[i, :t.shape[0]] = False
    return out, pad_mask


def unpad_concat(x_padded, lengths):
    """(B, longest, D) and the true lengths -> (sum of lengths, D) in the original row order."""
    return torch.cat([x_padded[i, :n] for i, n in enumerate(lengths)], dim=0)


def build_mlp(in_dim, hidden, out_dim, norm=True):
    """Linear layers with GELU between them and, if norm, a final LayerNorm."""
    dims = [in_dim] + list(hidden) + [out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    if norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def sinusoidal_time_embedding(t_norm, dim, min_period=0.004, max_period=4.0):
    """Fixed sine and cosine embedding of a value in [0, 1] with log-spaced periods: (*t_norm.shape, dim)."""
    half = dim // 2
    i = torch.arange(half, device=t_norm.device, dtype=torch.float32)
    freqs = 2 * math.pi / (min_period * (max_period / min_period) ** (i / max(half - 1, 1)))
    args = t_norm.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CrossAttnBlock(nn.Module):
    """Pre-LayerNorm cross-attention from queries q to key/value tokens kv, then a feed-forward layer; both with residuals."""

    def __init__(self, d_model, num_heads, mlp_ratio=4):
        """Creates the query, key/value and output projections, the LayerNorms and the feed-forward layer."""
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Linear(d_model * mlp_ratio, d_model))

    def forward(self, q, kv, key_padding_mask):
        """q: (B, Sq, d); kv: (B, Sk, d); key_padding_mask: (B, Sk), True = ignore this key."""
        B, Sq, d = q.shape
        Sk, H, hd = kv.shape[1], self.num_heads, self.head_dim
        q_n, kv_n = self.norm_q(q), self.norm_kv(kv)
        Q = self.q_proj(q_n).view(B, Sq, H, hd).transpose(1, 2)
        K = self.k_proj(kv_n).view(B, Sk, H, hd).transpose(1, 2)
        V = self.v_proj(kv_n).view(B, Sk, H, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=(~key_padding_mask)[:, None, None, :])
        out = self.out_proj(out.transpose(1, 2).reshape(B, Sq, d))
        q = q + out
        return q + self.ffn(self.norm_ffn(q))


class SelfAttnBlock(nn.Module):
    """Pre-LayerNorm self-attention within each padded sequence, then a feed-forward layer; both with residuals."""

    def __init__(self, d_model, num_heads, mlp_ratio=4):
        """Creates the fused qkv and output projections, the LayerNorms and the feed-forward layer."""
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Linear(d_model * mlp_ratio, d_model))

    def forward(self, x, key_padding_mask):
        """x: (B, S, d); key_padding_mask: (B, S), True = padding (never attended to)."""
        B, S, d = x.shape
        H, hd = self.num_heads, self.head_dim
        qkv = self.qkv_proj(self.norm_attn(x)).view(B, S, 3, H, hd).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=(~key_padding_mask)[:, None, None, :])
        x = x + self.out_proj(out.transpose(1, 2).reshape(B, S, d))
        return x + self.ffn(self.norm_ffn(x))


class AdaLNBlock(nn.Module):
    """Self-attention and feed-forward block whose LayerNorm scale, shift and residual gate come from a condition vector, starting close to the identity."""

    def __init__(self, d_model, num_heads, mlp_ratio=4, cond_dim=None, gate_init=0.1, ada_weight_init_std=0.0):
        """Creates the attention and feed-forward layers and the condition-to-modulation projection."""
        super().__init__()
        assert d_model % num_heads == 0
        cond_dim = cond_dim or d_model
        self.d_model, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Linear(d_model * mlp_ratio, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * d_model))
        if ada_weight_init_std > 0:
            nn.init.normal_(self.ada[-1].weight, mean=0.0, std=ada_weight_init_std)
        else:
            nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        if gate_init != 0.0:
            with torch.no_grad():
                self.ada[-1].bias[2 * d_model:3 * d_model] = gate_init  # gate of the attention branch
                self.ada[-1].bias[5 * d_model:6 * d_model] = gate_init  # gate of the feed-forward branch

    def forward(self, x, cond, key_padding_mask):
        """x: (B, S, d); cond: (B, cond_dim), one vector per sample; key_padding_mask: (B, S), True = padding."""
        B, S, d = x.shape
        H, hd = self.num_heads, self.head_dim
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        qkv = self.qkv_proj(h).view(B, S, 3, H, hd).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=(~key_padding_mask)[:, None, None, :])
        x = x + gate1.unsqueeze(1) * self.out_proj(attn.transpose(1, 2).reshape(B, S, d))
        h2 = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + gate2.unsqueeze(1) * self.ffn(h2)
