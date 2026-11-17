"""Validation and latent ablations: one pass encodes each window once and decodes it under every requested latent condition."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from lawm.data.dataset import collate
from lawm.model.metrics import NAMES, step_sums

CONDITIONS = ("correct", "zero", "shuffled_across", "fixed_donor")


def compute_context(device, amp: bool):
    """bf16 autocast on the GPU, and attention restricted to the flash and memory-efficient kernels (cuDNN attention has crashed under autocast)."""
    if device.type != "cuda" or not amp:
        return nullcontext()
    from contextlib import ExitStack
    from torch.nn.attention import SDPBackend, sdpa_kernel
    stack = ExitStack()
    stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
    return stack


def to_device(batch: dict, device) -> dict:
    """Move the tensors of a batch dict to device, leaving other values unchanged."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def sharded_loader(dataset, rows, rank: int, world_size: int, batch_size: int, workers: int, device) -> DataLoader:
    """This rank's contiguous share of the rows: every row is used exactly once across ranks, none repeated."""
    mine = np.array_split(np.asarray(rows), world_size)[rank].tolist()
    return DataLoader(Subset(dataset, mine), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=workers,
                      pin_memory=device.type == "cuda")


def ablation_rows(dataset, n: int, batch_size: int) -> list:
    """About n windows spread evenly over the sequences and interleaved one sequence at a time, so any batch_size consecutive windows come from different sequences."""
    seqs = sorted(set(dataset.seq.tolist()))
    if batch_size > len(seqs):
        raise ValueError(f"batch size {batch_size} exceeds the {len(seqs)} sequences available for the shuffled-across condition")
    quota = -(-n // len(seqs))
    picks = []
    for s in seqs:
        rows = np.flatnonzero(dataset.seq == s)
        picks.append(rows[np.linspace(0, len(rows) - 1, min(quota, len(rows))).round().astype(int)])
    out = [int(p[i]) for i in range(max(len(p) for p in picks)) for p in picks if i < len(p)]
    return out[:n]


def apply_condition(name: str, zs: torch.Tensor, seqs, fixed_zs=None):
    """zs: (n_steps, B, n_queries, z_dim). Returns the changed latents, or None if the condition cannot be formed for this batch."""
    if name == "correct":
        return zs
    if name == "zero":
        return torch.zeros_like(zs)
    if name == "shuffled_across":
        if zs.shape[1] < 2:
            return None
        if any(seqs[i] == seqs[i - 1] for i in range(len(seqs))):
            raise ValueError("shuffled_across needs consecutive windows from different sequences (use ablation_rows)")
        return torch.roll(zs, 1, dims=1)  # window i receives the latents of window i - 1
    if name == "fixed_donor":
        return fixed_zs.expand(-1, zs.shape[1], -1, -1)
    raise ValueError(name)


@torch.no_grad()
def donor_latents(model, dataset, row: int, device, amp: bool):
    """The latents of one fixed window, (n_steps, 1, n_queries, z_dim), used as the fixed donor."""
    batch = to_device(collate([dataset[row]]), device)
    with compute_context(device, amp):
        return model.encode_window(batch["coords"], batch["offset"])[1].float()


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, conditions=("correct",), fixed_zs=None, distributed: bool = False) -> dict:
    """{condition: (n_steps, len(NAMES), 2) summed error and count over all ranks}; model must be in eval mode."""
    n_steps = model.n_steps
    totals = {c: torch.zeros(n_steps, len(NAMES), 2, dtype=torch.float64, device=device) for c in conditions}
    for batch in loader:
        seqs = batch["seq"]
        batch = to_device(batch, device)
        coords, offset, segment = batch["coords"], batch["offset"], batch["segment"]
        with compute_context(device, amp):
            chain_0, zs = model.encode_window(coords, offset)
            for c in conditions:
                changed = apply_condition(c, zs, seqs, fixed_zs)
                if changed is None:
                    continue
                outs = model.decode_all_steps(chain_0, changed)
                for k in range(1, n_steps + 1):
                    totals[c][k - 1] += step_sums(outs[k - 1]["delta"].float(), (coords[k] - coords[0]).unsqueeze(1), segment)
    if distributed:
        for t in totals.values():
            dist.all_reduce(t)
    return {c: t.cpu().numpy() for c, t in totals.items()}
