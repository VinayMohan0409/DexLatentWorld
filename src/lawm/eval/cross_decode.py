"""Cross-decode: latents encoded from embodiment i's window, decoded on embodiment j's frame-0 cloud, scored on j's motion."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np
import torch

from lawm.data.dataset import collate, load_window
from lawm.data.sequences import Sequence
from lawm.model.metrics import NAMES, step_sums
from lawm.train.evaluation import compute_context, to_device

# Protocol: 16 seeded windows per validation sequence, one set for every cell, batches never mixing embodiments, mover_5mm;
# each window centred on its own frame 0, valid for every embodiment and used once; each embodiment encoded once per batch
# and reused for every pair; eval mode, a seeded serialisation order and a row-order check in encode_window.

Window = Tuple[str, int]


def select_windows(index: Dict[str, np.ndarray], embodiments: Seq[int], per_sequence: int = 16, seed: int = 0) -> List[Window]:
    """Seeded per-sequence sample of windows valid for every listed embodiment, as [(sequence, start frame)]."""
    ok = index["valid_window"][:, list(embodiments)].all(axis=1)
    rng = np.random.default_rng(seed)
    rows: List[Window] = []
    for s in sorted(set(index["seq"].tolist())):
        pool = np.flatnonzero((index["seq"] == s) & ok)
        if len(pool) == 0:
            continue
        pick = np.sort(rng.choice(len(pool), size=min(per_sequence, len(pool)), replace=False))
        rows += [(str(s), int(index["f"][pool[i]])) for i in pick]
    return rows


def batches(windows: List[Window], batch_size: int) -> List[List[Window]]:
    """Consecutive groups of windows, never spanning two sequences (one-sequence batches)."""
    out: List[List[Window]] = []
    for w in windows:
        if out and len(out[-1]) < batch_size and out[-1][-1][0] == w[0]:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def window_batch(sequences: Dict[str, Sequence], group: List[Window], e: int, gap: int, n_steps: int) -> dict:
    """The collated batch of one embodiment's windows, built as WindowDataset items are (own frame-0 centring)."""
    items = []
    for r, (s, f) in enumerate(group):
        samples = load_window(sequences[s], f, [e], gap, n_steps)
        for x in samples:
            x.update(seq=s, frame=f)
        items.append(dict(transition_idx=r, samples=samples))
    return collate(items)


def encode_seed(base: int, group: List[Window], name: str) -> int:
    """Seed of one embodiment's encode of one batch, so its result does not depend on the other embodiments in the grid."""
    key = f"{base}|{group[0][0]}|{group[0][1]}|{len(group)}|{name}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "little")


@torch.no_grad()
def cross_decode_batch(model, sequences: Dict[str, Sequence], group: List[Window], embodiments: Seq[int], gap: int,
                       device, amp: bool, cells: Optional[np.ndarray] = None, names: Optional[Seq[str]] = None,
                       seed: Optional[int] = None) -> np.ndarray:
    """Summed errors (E, E, n_steps, len(NAMES), 2) of one batch, [i, j] = i's latents on j's cloud; only `cells` are decoded."""
    from lawm.train.reproducibility import evaluation_stream
    n_steps, E = model.n_steps, len(embodiments)
    cells = np.ones((E, E), bool) if cells is None else np.asarray(cells, bool)
    zs, queries, targets = [], [], []
    for idx, e in enumerate(embodiments):  # one forward pass per embodiment over the same windows in the same order
        batch = to_device(window_batch(sequences, group, e, gap, n_steps), device)
        coords = batch["coords"]
        stream = evaluation_stream(encode_seed(seed, group, names[idx]), device) if seed is not None else nullcontext()
        with stream, compute_context(device, amp):
            chain_0, z = model.encode_window(coords, batch["offset"])
            q = model.decoder.point_queries(chain_0)
        del chain_0
        zs.append(z)
        queries.append(q)
        targets.append([(coords[k] - coords[0]).unsqueeze(1) for k in range(1, n_steps + 1)] + [batch["segment"]])
    totals = np.zeros((E, E, n_steps, len(NAMES), 2))
    for j in range(E):
        *gt, segment = targets[j]
        for i in range(E):
            if not cells[i, j]:
                continue
            with compute_context(device, amp):
                outs = model.decode_all_steps(None, zs[i], queries[j])
            for k in range(n_steps):
                totals[i, j, k] = step_sums(outs[k]["delta"].float(), gt[k], segment).cpu().numpy()
    return totals


def cross_decode(model, ds_dir, windows: List[Window], embodiments: Seq[int], gap: int, points_per_entity: int,
                 batch_size: int, device, amp: bool, log=print, cells=None, names=None, seed=None) -> np.ndarray:
    """Summed errors (E, E, n_steps, len(NAMES), 2) over all windows; with seed, every encode has its own random stream."""
    assert not model.training, "call model.eval() first: drop_path is active in training mode"
    from pathlib import Path
    sequences = {s: Sequence(Path(ds_dir) / s, points_per_entity) for s in sorted({s for s, _ in windows})}
    for s, q in sequences.items():
        missing = [e for e in embodiments if e >= len(q.names)]
        if missing:
            raise ValueError(f"{s} has {len(q.names)} embodiments; asked for index {missing}")
    groups = batches(windows, batch_size)
    totals = None
    for b, group in enumerate(groups):
        t = cross_decode_batch(model, sequences, group, embodiments, gap, device, amp, cells, names, seed)
        totals = t if totals is None else totals + t
        log(f"[cross_decode] batch {b + 1}/{len(groups)} ({group[0][0]}, {len(group)} windows)")
    return totals


def matrices(totals: np.ndarray, metric: str = "mover_5mm") -> Dict[str, np.ndarray]:
    """Mean error (mm) per cell for each step, their mean over steps, and the step-1 point count."""
    m = NAMES.index(metric)
    s, n = totals[..., m, 0], totals[..., m, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_step = np.where(n > 0, s / n, np.nan)  # (E, E, n_steps)
    out = {f"step{k + 1}": per_step[..., k] for k in range(per_step.shape[-1])}
    out["mean_over_steps"] = np.nanmean(per_step, axis=-1)
    out["n_step1"] = n[..., 0]
    return out


def regime_summary(M: np.ndarray, trained: Seq[bool]) -> Dict[str, float]:
    """Diagonal and off-diagonal means split by whether the source / target embodiment was seen in training."""
    trained = np.asarray(trained, bool)
    E = len(trained)
    off = ~np.eye(E, dtype=bool)

    def mean(mask):
        """Mean of M over a mask, NaN if empty."""
        return float(np.nanmean(M[mask])) if mask.any() else float("nan")
    src, dst = trained[:, None], trained[None, :]
    return {"diag_seen": mean(np.eye(E, dtype=bool) & src & dst), "diag_unseen": mean(np.eye(E, dtype=bool) & ~src & ~dst),
            "seen_to_seen": mean(off & src & dst), "seen_to_unseen": mean(off & src & ~dst),
            "unseen_to_seen": mean(off & ~src & dst), "unseen_to_unseen": mean(off & ~src & ~dst),
            "gap_mean_offdiag": float(np.nanmean((M - np.diag(M)[None, :])[off]))}


def plot_heatmap(M: np.ndarray, names: Seq[str], trained: Seq[bool], path, title: str, label: str) -> None:
    """Annotated heatmap, trained embodiments first; embodiments the checkpoint never saw get red labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [e for e in range(len(names)) if trained[e]] + [e for e in range(len(names)) if not trained[e]]
    sub = M[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(order), 0.4 + 0.68 * len(order)))
    im = ax.imshow(sub, cmap="viridis")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels([names[e] for e in order], rotation=45, ha="right")
    ax.set_yticklabels([names[e] for e in order])
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for t, e in zip(ticks, order):
            if not trained[e]:
                t.set_color("#b30000")
                t.set_fontweight("bold")
    ax.set_xlabel("decode into (frame-0 cloud of)")
    ax.set_ylabel("latents from")
    ax.set_title(title)
    fig.colorbar(im, label=label)
    hi = np.nanmax(sub)
    for r in range(len(order)):
        for c in range(len(order)):
            if np.isfinite(sub[r, c]):
                ax.text(c, r, f"{sub[r, c]:.1f}", ha="center", va="center", fontsize=7, color="white" if sub[r, c] > 0.5 * hi else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
